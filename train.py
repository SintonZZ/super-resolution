import argparse
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from archs import build_span
from dataset import PairedSRDataset
from util import (
    AverageMeter,
    calc_psnr,
    count_parameters,
    forward_tiled,
    get_lr,
    load_pretrained_span_backbone,
    load_torch,
    resolve_auto_device,
    save_json,
    set_random_seed,
    setup_logger,
    worker_init_fn,
)


DEFAULT_CONFIG_PATH = "config/train.json"


def load_config(config_path):
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def expand_save_dir(config):
    training = config["training"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    desc = training.get("desc", "span_x2_finetune")
    training["save_dir"] = training["save_dir"].format(timestamp=timestamp, desc=desc)
    training["device"] = resolve_auto_device(training.get("device", "auto"))


def build_loss(loss_type):
    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss type: {loss_type}")


def build_optimizer(model, training_config):
    base_lr = float(training_config["max_lr"])
    head_multiplier = float(training_config.get("head_lr_multiplier", 1.0))
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("upsampler."):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    parameter_groups = [
        {"params": backbone_parameters, "lr": base_lr, "lr_scale": 1.0},
        {
            "params": head_parameters,
            "lr": base_lr * head_multiplier,
            "lr_scale": head_multiplier,
        },
    ]
    return optim.Adam(
        parameter_groups,
        lr=base_lr,
        betas=tuple(training_config.get("betas", (0.9, 0.99))),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )


def set_epoch_lr(optimizer, epoch, training_config):
    base_lr = get_lr(
        epoch,
        cycle_length=int(training_config.get("cycle_length", training_config["total_epochs"])),
        max_lr=float(training_config["max_lr"]),
        min_lr=float(training_config["min_lr"]),
    )
    for group in optimizer.param_groups:
        group["lr"] = base_lr * float(group.get("lr_scale", 1.0))
    return base_lr


def get_amp_tools(enabled, device):
    amp_module = getattr(torch.cuda, "amp", None)
    use_amp = bool(
        enabled
        and device.type == "cuda"
        and amp_module is not None
        and hasattr(amp_module, "autocast")
        and hasattr(amp_module, "GradScaler")
    )
    scaler = amp_module.GradScaler(enabled=True) if use_amp else None
    return use_amp, scaler


def amp_context(use_amp):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def build_checkpoint(model, optimizer, scaler, epoch, best_metric, metrics, config):
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "metrics": metrics,
        "config": config,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    return checkpoint


def resume_training(model, optimizer, scaler, resume_path, device, logger):
    if not resume_path:
        return 0, -float("inf")

    checkpoint = load_torch(resume_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Resume checkpoint must be produced by this train.py.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", -float("inf")))
    logger.info("Resumed %s at epoch %d", resume_path, start_epoch)
    return start_epoch, best_metric


def run_one_epoch(model, loader, criterion, optimizer, scaler, use_amp, device, config, epoch, logger):
    model.train()
    loss_meter = AverageMeter("Loss")
    psnr_meter = AverageMeter("PSNR")
    base_lr = set_epoch_lr(optimizer, epoch, config["training"])
    clip_grad_norm = float(config["training"].get("clip_grad_norm", 0.0))

    progress = tqdm(loader, desc=f"Epoch[{epoch + 1:04d}] LR:{base_lr:.2e}", leave=False)
    for batch in progress:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad()

        with amp_context(use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        batch_size = inputs.shape[0]
        loss_meter.update(loss.item(), batch_size)
        psnr_meter.update(calc_psnr(outputs, targets), batch_size)
        progress.set_postfix(loss=f"{loss_meter.avg:.5f}", psnr=f"{psnr_meter.avg:.2f}")

    metrics = {"loss": loss_meter.avg, "psnr": psnr_meter.avg, "lr": base_lr}
    logger.info(
        "Train epoch %d | loss %.6f | RGB PSNR %.3f | backbone lr %.3e",
        epoch + 1,
        metrics["loss"],
        metrics["psnr"],
        base_lr,
    )
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, config, epoch, logger):
    model.eval()
    loss_meter = AverageMeter("Loss")
    rgb_psnr_meter = AverageMeter("RGB_PSNR")
    y_psnr_meter = AverageMeter("Y_PSNR")
    validation = config.get("validation", {})
    scale = int(config["model"]["upscale"])
    crop_border = int(validation.get("crop_border", scale))
    tile_size = int(validation.get("tile_size", 0))
    tile_pad = int(validation.get("tile_pad", 24))

    for batch in tqdm(loader, desc="Validation", leave=False):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        outputs = forward_tiled(model, inputs, scale, tile_size, tile_pad)
        loss = criterion(outputs, targets)

        batch_size = inputs.shape[0]
        loss_meter.update(loss.item(), batch_size)
        rgb_psnr_meter.update(calc_psnr(outputs, targets, crop_border, False), batch_size)
        y_psnr_meter.update(calc_psnr(outputs, targets, crop_border, True), batch_size)

    metrics = {
        "loss": loss_meter.avg,
        "rgb_psnr": rgb_psnr_meter.avg,
        "y_psnr": y_psnr_meter.avg,
    }
    logger.info(
        "Val epoch %d | loss %.6f | RGB PSNR %.3f | Y PSNR %.3f",
        epoch + 1,
        metrics["loss"],
        metrics["rgb_psnr"],
        metrics["y_psnr"],
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune official SPAN x4 weights for x2 SR.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-hr-dir", default=None)
    parser.add_argument("--train-lr-dir", default=None)
    parser.add_argument("--val-hr-dir", default=None)
    parser.add_argument("--val-lr-dir", default=None)
    parser.add_argument("--pretrained-4x", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr-patch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config, args):
    dataset_updates = {
        "train_hr_dir": args.train_hr_dir,
        "train_lr_dir": args.train_lr_dir,
        "val_hr_dir": args.val_hr_dir,
        "val_lr_dir": args.val_lr_dir,
        "lr_patch_size": args.lr_patch_size,
        "max_images": args.max_images,
    }
    for key, value in dataset_updates.items():
        if value is not None:
            config["dataset"][key] = value

    training_updates = {
        "pretrained_4x_path": args.pretrained_4x,
        "resume": args.resume,
        "save_dir": args.save_dir,
        "total_epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
    }
    for key, value in training_updates.items():
        if value is not None:
            config["training"][key] = value


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    expand_save_dir(config)

    if int(config["model"].get("upscale", 2)) != 2:
        raise ValueError("This fine-tuning project expects model.upscale=2.")
    if int(config["dataset"].get("scale", 2)) != 2:
        raise ValueError("dataset.scale must be 2.")

    training = config["training"]
    save_dir = training["save_dir"]
    logger = setup_logger(save_dir)
    save_json(config, Path(save_dir) / "config.json")
    logger.info("=" * 60)
    logger.info("Starting SPAN x2 fine-tuning")
    logger.info("=" * 60)
    logger.info("Configuration:\n%s", json.dumps(config, indent=4, ensure_ascii=False))

    seed = int(training.get("seed", 1234))
    set_random_seed(seed)
    device = torch.device(training["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))

    logger.info("Loading datasets...")
    train_dataset = PairedSRDataset(config["dataset"], split="train")
    val_dataset = PairedSRDataset(config["dataset"], split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=bool(training.get("drop_last", True)),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    logger.info("Datasets loaded: train=%d, val=%d", len(train_dataset), len(val_dataset))

    model = build_span(config["model"]).to(device)
    logger.info("Model parameters: %.3f M", count_parameters(model) / 1e6)

    resume_path = training.get("resume")
    pretrained_path = training.get("pretrained_4x_path")
    if not resume_path and pretrained_path:
        report = load_pretrained_span_backbone(
            model,
            pretrained_path,
            map_location=device,
            strict_backbone=bool(training.get("strict_pretrained_backbone", True)),
        )
        logger.info(
            "Loaded pretrained checkpoint state '%s': scale=%s, loaded=%d, "
            "shape_mismatches=%s",
            report["state_key"],
            report["checkpoint_scale"],
            report["loaded_count"],
            report["shape_mismatches"],
        )
        if report["checkpoint_scale"] not in (None, 4):
            logger.warning(
                "The checkpoint head appears to be x%s rather than x4.",
                report["checkpoint_scale"],
            )
    elif not resume_path:
        logger.warning("No pretrained_4x_path configured; training SPAN x2 from scratch.")

    criterion = build_loss(config["loss"].get("type", "l1")).to(device)
    optimizer = build_optimizer(model, training)
    use_amp, scaler = get_amp_tools(training.get("amp", True), device)
    logger.info("AMP enabled: %s", use_amp)

    start_epoch, best_metric = resume_training(
        model,
        optimizer,
        scaler,
        resume_path,
        device,
        logger,
    )

    total_epochs = int(training["total_epochs"])
    for epoch in range(start_epoch, total_epochs):
        train_metrics = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            use_amp,
            device,
            config,
            epoch,
            logger,
        )

        val_metrics = None
        val_interval = int(training.get("val_interval", 1))
        if val_interval > 0 and (epoch + 1) % val_interval == 0:
            val_metrics = validate(model, val_loader, criterion, device, config, epoch, logger)

        current_metric = (
            val_metrics["y_psnr"] if val_metrics is not None else train_metrics["psnr"]
        )
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric

        metrics = {"train": train_metrics, "val": val_metrics}
        checkpoint = build_checkpoint(
            model,
            optimizer,
            scaler,
            epoch,
            best_metric,
            metrics,
            config,
        )
        torch.save(checkpoint, Path(save_dir) / "latest_model.pth")
        if (epoch + 1) % int(training.get("save_interval", 10)) == 0:
            torch.save(checkpoint, Path(save_dir) / f"model_epoch_{epoch + 1:04d}.pth")
        if is_best:
            torch.save(checkpoint, Path(save_dir) / "best_model.pth")
            logger.info("Saved best_model.pth with Y PSNR %.3f", best_metric)

    logger.info("Training finished: %s", Path(save_dir) / "latest_model.pth")


if __name__ == "__main__":
    main()
