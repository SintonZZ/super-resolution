import argparse
import copy
import json
import math
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from archs import build_model
from dataset import PairedSRDataset
from util import (
    AverageMeter,
    calc_psnr,
    count_parameters,
    forward_tiled,
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
    desc = training.get("desc", "spanf_x2_train")
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
    return optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=base_lr,
        betas=tuple(training_config.get("betas", (0.9, 0.99))),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )


def get_step_lr(step, training_config):
    total_steps = int(training_config["total_steps"])
    warmup_steps = int(training_config.get("warmup_steps", 0))
    max_lr = float(training_config["max_lr"])
    min_lr = float(training_config["min_lr"])
    if total_steps <= 0:
        raise ValueError("training.total_steps must be positive.")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("training.warmup_steps must be in [0, total_steps).")
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}.")

    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)

    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return min_lr
    progress = float(step - warmup_steps) / float(decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_step_lr(optimizer, step, training_config):
    base_lr = get_step_lr(step, training_config)
    for group in optimizer.param_groups:
        group["lr"] = base_lr
    return base_lr


class ModelEMA:
    """Exponential moving average of all model state tensors."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("training.ema_decay must be in [0, 1).")
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self._invalidate_reparameterized_convs()

    def _invalidate_reparameterized_convs(self):
        for module in self.module.modules():
            if hasattr(module, "eval_params_ready") and not getattr(module, "deploy", False):
                module.eval_params_ready = False

    def copy_from(self, model):
        self.module.load_state_dict(model.state_dict(), strict=True)
        self._invalidate_reparameterized_convs()

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)
        self._invalidate_reparameterized_convs()


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


def build_checkpoint(
    model,
    ema,
    optimizer,
    scaler,
    step,
    data_epoch,
    best_metric,
    metrics,
    config,
):
    checkpoint = {
        "step": int(step),
        "epoch": int(data_epoch),
        "data_epoch": int(data_epoch),
        "model_state_dict": model.state_dict(),
        "params_ema": ema.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "metrics": metrics,
        "config": config,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    return checkpoint


def resume_training(
    model,
    ema,
    optimizer,
    scaler,
    resume_path,
    device,
    logger,
    steps_per_epoch,
):
    if not resume_path:
        return 0, 0, -float("inf")

    checkpoint = load_torch(resume_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Resume checkpoint must be produced by this train.py.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if checkpoint.get("params_ema"):
        ema.module.load_state_dict(checkpoint["params_ema"], strict=True)
        ema._invalidate_reparameterized_convs()
    else:
        ema.copy_from(model)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    data_epoch = int(checkpoint.get("data_epoch", checkpoint.get("epoch", 0)))
    if "step" in checkpoint:
        start_step = int(checkpoint["step"])
    else:
        start_step = int(checkpoint.get("epoch", 0)) * int(steps_per_epoch)
        logger.warning(
            "Legacy epoch checkpoint has no global step; inferred step=%d from %d step(s)/epoch.",
            start_step,
            steps_per_epoch,
        )
    best_metric = float(checkpoint.get("best_metric", -float("inf")))
    logger.info(
        "Resumed %s at step %d (data epoch %d)",
        resume_path,
        start_step,
        data_epoch,
    )
    return start_step, data_epoch, best_metric


def train_one_step(model, batch, criterion, optimizer, scaler, ema, use_amp, device, config, step):
    model.train()
    base_lr = set_step_lr(optimizer, step, config["training"])
    clip_grad_norm = float(config["training"].get("clip_grad_norm", 0.0))

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

    ema.update(model)
    return {
        "loss": loss.item(),
        "psnr": calc_psnr(outputs, targets),
        "lr": base_lr,
        "batch_size": inputs.shape[0],
    }


def update_best_metric(val_metrics, best_metric, metric_name):
    if val_metrics is None:
        return best_metric, False
    if metric_name not in val_metrics:
        raise ValueError(
            f"validation.best_metric={metric_name!r} is not available; "
            f"choose one of {sorted(val_metrics)}."
        )
    current_metric = float(val_metrics[metric_name])
    if current_metric > best_metric:
        return current_metric, True
    return best_metric, False


@torch.no_grad()
def validate(model, loader, criterion, device, config, step, logger):
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
        "Val step %d | loss %.6f | RGB PSNR %.3f | Y PSNR %.3f",
        step,
        metrics["loss"],
        metrics["rgb_psnr"],
        metrics["y_psnr"],
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPAN-F x2 for single-image SR.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-hr-dir", default=None)
    parser.add_argument("--train-lr-dir", default=None)
    parser.add_argument("--val-hr-dir", default=None)
    parser.add_argument("--val-lr-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
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
        "resume": args.resume,
        "save_dir": args.save_dir,
        "total_steps": args.steps,
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
        raise ValueError("This training project expects model.upscale=2.")
    if int(config["dataset"].get("scale", 2)) != 2:
        raise ValueError("dataset.scale must be 2.")

    training = config["training"]
    save_dir = training["save_dir"]
    logger = setup_logger(save_dir)
    save_json(config, Path(save_dir) / "config.json")
    logger.info("=" * 60)
    logger.info("Starting SPAN-F x2 training")
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

    model = build_model(config["model"]).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info(
        "Model parameters: %.3f M registered, %.3f M trainable",
        count_parameters(model) / 1e6,
        trainable_parameters / 1e6,
    )

    resume_path = training.get("resume")

    criterion = build_loss(config["loss"].get("type", "l1")).to(device)
    optimizer = build_optimizer(model, training)
    ema = ModelEMA(model, decay=float(training.get("ema_decay", 0.999)))
    use_amp, scaler = get_amp_tools(training.get("amp", True), device)
    logger.info("AMP enabled: %s", use_amp)
    logger.info("EMA decay: %.6f", ema.decay)

    start_step, data_epoch, best_metric = resume_training(
        model,
        ema,
        optimizer,
        scaler,
        resume_path,
        device,
        logger,
        steps_per_epoch=len(train_loader),
    )

    total_steps = int(training["total_steps"])
    if start_step >= total_steps:
        logger.info("Checkpoint step %d already reached total_steps=%d.", start_step, total_steps)
        return

    log_interval = int(training.get("log_interval", 100))
    val_interval = int(training.get("val_interval", 5000))
    save_interval = int(training.get("save_interval", 5000))
    best_metric_name = str(config.get("validation", {}).get("best_metric", "y_psnr"))
    if log_interval <= 0:
        raise ValueError("training.log_interval must be positive.")
    if val_interval < 0 or save_interval < 0:
        raise ValueError("training.val_interval and save_interval must be non-negative.")

    train_iterator = iter(train_loader)
    loss_meter = AverageMeter("Loss")
    psnr_meter = AverageMeter("PSNR")
    last_train_metrics = None
    step = start_step
    progress = tqdm(total=total_steps, initial=start_step, desc="Training", unit="step")
    while step < total_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            data_epoch += 1
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        step_metrics = train_one_step(
            model,
            batch,
            criterion,
            optimizer,
            scaler,
            ema,
            use_amp,
            device,
            config,
            step,
        )
        step += 1
        batch_size = step_metrics.pop("batch_size")
        loss_meter.update(step_metrics["loss"], batch_size)
        psnr_meter.update(step_metrics["psnr"], batch_size)
        progress.update(1)
        progress.set_postfix(
            loss=f"{loss_meter.avg:.5f}",
            psnr=f"{psnr_meter.avg:.2f}",
            lr=f"{step_metrics['lr']:.2e}",
        )

        should_log = step % log_interval == 0 or step == total_steps
        if should_log:
            last_train_metrics = {
                "loss": loss_meter.avg,
                "psnr": psnr_meter.avg,
                "lr": step_metrics["lr"],
            }
            logger.info(
                "Train step %d/%d | data epoch %d | loss %.6f | RGB PSNR %.3f | lr %.3e",
                step,
                total_steps,
                data_epoch,
                last_train_metrics["loss"],
                last_train_metrics["psnr"],
                last_train_metrics["lr"],
            )
            loss_meter.reset()
            psnr_meter.reset()

        should_validate = val_interval > 0 and (
            step % val_interval == 0 or step == total_steps
        )
        val_metrics = None
        is_best = False
        if should_validate:
            val_metrics = validate(
                ema.module,
                val_loader,
                criterion,
                device,
                config,
                step,
                logger,
            )
            best_metric, is_best = update_best_metric(
                val_metrics,
                best_metric,
                best_metric_name,
            )

        should_save = (
            (save_interval > 0 and step % save_interval == 0)
            or should_validate
            or step == total_steps
        )
        if should_save:
            metrics = {"train": last_train_metrics, "val": val_metrics}
            checkpoint = build_checkpoint(
                model,
                ema,
                optimizer,
                scaler,
                step,
                data_epoch,
                best_metric,
                metrics,
                config,
            )
            torch.save(checkpoint, Path(save_dir) / "latest_model.pth")
            if save_interval > 0 and step % save_interval == 0:
                torch.save(checkpoint, Path(save_dir) / f"model_step_{step:08d}.pth")
            if is_best:
                torch.save(checkpoint, Path(save_dir) / "best_model.pth")
                logger.info(
                    "Saved best_model.pth at step %d with %s %.3f",
                    step,
                    best_metric_name,
                    best_metric,
                )

    progress.close()

    logger.info("Training finished: %s", Path(save_dir) / "latest_model.pth")


if __name__ == "__main__":
    main()
