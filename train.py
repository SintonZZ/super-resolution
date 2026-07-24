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

from archs import build_discriminator, build_model
from dataset import PairedSRDataset
from losses import GANLoss, PerceptualLoss, USMSharp, build_pixel_loss
from util import (
    AverageMeter,
    calc_psnr,
    calc_ssim,
    clean_state_dict,
    count_parameters,
    extract_state_dict,
    forward_tiled,
    load_torch,
    resolve_auto_device,
    save_json,
    set_random_seed,
    setup_logger,
    worker_init_fn,
)


DEFAULT_CONFIG_PATH = "config/train.json"
STAGE_NAMES = ("stage1", "stage2")


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
    training["save_dir"] = training["save_dir"].format(
        timestamp=timestamp,
        desc=desc,
    )
    training["device"] = resolve_auto_device(training.get("device", "auto"))


def resolve_stage_config(training_config, stage_name):
    """Merge root training defaults with one explicit stage configuration."""
    if stage_name not in STAGE_NAMES:
        raise ValueError(f"Unsupported training stage: {stage_name}")
    common = {
        key: value
        for key, value in training_config.items()
        if key not in STAGE_NAMES
    }
    common.update(training_config[stage_name])
    return common


def validate_training_config(config):
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("config.training must be an object.")
    missing = [name for name in STAGE_NAMES if name not in training]
    if missing:
        raise ValueError(
            "Two-stage training schema is required; missing "
            + ", ".join(f"training.{name}" for name in missing)
        )

    enabled = {
        name: bool(training[name].get("enabled", True)) for name in STAGE_NAMES
    }
    if not any(enabled.values()):
        raise ValueError("At least one training stage must be enabled.")

    resume_path = training.get("resume")
    stage2_pretrained = training["stage2"].get("pretrained_generator")
    if enabled["stage1"] and stage2_pretrained and not resume_path:
        raise ValueError(
            "training.stage2.pretrained_generator must be null when stage1 is enabled."
        )
    if enabled["stage2"] and not enabled["stage1"] and not stage2_pretrained and not resume_path:
        raise ValueError(
            "training.stage2.pretrained_generator is required when stage1 is disabled."
        )

    for name in STAGE_NAMES:
        if not enabled[name]:
            continue
        stage = resolve_stage_config(training, name)
        total_steps = int(stage.get("total_steps", 0))
        if total_steps <= 0:
            raise ValueError(f"training.{name}.total_steps must be positive.")
        if int(stage.get("batch_size", 0)) <= 0:
            raise ValueError(f"training.{name}.batch_size must be positive.")
        loss_config = stage.get("loss")
        if not isinstance(loss_config, dict) or "pixel" not in loss_config:
            raise ValueError(f"training.{name}.loss.pixel is required.")

    if enabled["stage2"] and "discriminator" not in config:
        raise ValueError("config.discriminator is required when stage2 is enabled.")


def build_loss(loss_type):
    """Backward-compatible public helper for basic losses."""
    return build_pixel_loss(loss_type)


def _optimizer_options(training_config, key="optimizer_g"):
    options = dict(training_config.get(key, {}))
    options.setdefault("type", "adam")
    options.setdefault(
        "betas",
        training_config.get("betas", (0.9, 0.99)),
    )
    options.setdefault(
        "weight_decay",
        training_config.get("weight_decay", 0.0),
    )
    return options


def build_optimizer(model, training_config, key="optimizer_g"):
    options = _optimizer_options(training_config, key)
    optimizer_type = str(options["type"]).lower()
    if optimizer_type != "adam":
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    base_lr = float(options.get("lr", training_config["max_lr"]))
    return optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=base_lr,
        betas=tuple(options["betas"]),
        weight_decay=float(options["weight_decay"]),
    )


def get_step_lr(step, training_config):
    total_steps = int(training_config["total_steps"])
    warmup_steps = max(int(training_config.get("warmup_steps", 0)), 0)
    max_lr = float(training_config["max_lr"])
    if total_steps <= 0:
        raise ValueError("training stage total_steps must be positive.")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps).")
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}.")
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)

    scheduler = training_config.get("scheduler")
    if scheduler is not None:
        scheduler_type = str(scheduler.get("type", "multistep")).lower()
        if scheduler_type != "multistep":
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        gamma = float(scheduler.get("gamma", 0.5))
        milestones = [int(value) for value in scheduler.get("milestones", ())]
        decay_count = sum(step >= milestone for milestone in milestones)
        return max_lr * (gamma ** decay_count)

    # Keep the previous cosine helper available for focused unit tests.
    min_lr = float(training_config.get("min_lr", max_lr))
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return min_lr
    progress = float(step - warmup_steps) / float(decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def set_step_lr(optimizer, step, training_config):
    base_lr = get_step_lr(step, training_config)
    for group in optimizer.param_groups:
        group["lr"] = base_lr
    return base_lr


class ModelEMA:
    """Exponential moving average of all generator state tensors."""

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
                ema_value.mul_(self.decay).add_(
                    model_value,
                    alpha=1.0 - self.decay,
                )
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
    active_stage="stage1",
    completed_stages=None,
    discriminator=None,
    optimizer_d=None,
    best_metrics=None,
):
    checkpoint = {
        "active_stage": active_stage,
        "completed_stages": list(completed_stages or ()),
        "step": int(step),
        "epoch": int(data_epoch),
        "data_epoch": int(data_epoch),
        "model_state_dict": model.state_dict(),
        "params_ema": ema.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_g_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "best_metrics": dict(best_metrics or {"y_psnr": best_metric}),
        "metrics": metrics,
        "config": config,
    }
    if discriminator is not None:
        checkpoint["discriminator_state_dict"] = discriminator.state_dict()
    if optimizer_d is not None:
        checkpoint["optimizer_d_state_dict"] = optimizer_d.state_dict()
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
    discriminator=None,
    optimizer_d=None,
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

    optimizer_state = checkpoint.get(
        "optimizer_g_state_dict",
        checkpoint.get("optimizer_state_dict"),
    )
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    if discriminator is not None:
        state = checkpoint.get("discriminator_state_dict")
        if not state:
            raise ValueError("Stage 2 resume checkpoint has no discriminator state.")
        discriminator.load_state_dict(state, strict=True)
    if optimizer_d is not None:
        state = checkpoint.get("optimizer_d_state_dict")
        if not state:
            raise ValueError("Stage 2 resume checkpoint has no discriminator optimizer.")
        optimizer_d.load_state_dict(state)
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    data_epoch = int(checkpoint.get("data_epoch", checkpoint.get("epoch", 0)))
    if "step" in checkpoint:
        start_step = int(checkpoint["step"])
    else:
        start_step = int(checkpoint.get("epoch", 0)) * int(steps_per_epoch)
        logger.warning(
            "Legacy epoch checkpoint has no global step; inferred step=%d.",
            start_step,
        )
    best_metric = float(checkpoint.get("best_metric", -float("inf")))
    logger.info(
        "Resumed %s at %s step %d (data epoch %d)",
        resume_path,
        checkpoint.get("active_stage", "stage1"),
        start_step,
        data_epoch,
    )
    return start_step, data_epoch, best_metric


def validate_initialization_config(training_config):
    """Retained for callers of the old public helper."""
    resume_path = training_config.get("resume")
    pretrained_path = training_config.get("pretrained")
    if resume_path and pretrained_path:
        raise ValueError(
            "training.resume and training.pretrained are mutually exclusive."
        )


def load_pretrained_model(model, pretrained_path, device, logger):
    """Load preferred inference/EMA weights without restoring training state."""
    if not pretrained_path:
        return None
    checkpoint = load_torch(pretrained_path, map_location=device)
    state_dict, state_key = extract_state_dict(checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    logger.info(
        "Loaded generator weights from %s (state key: %s).",
        pretrained_path,
        state_key,
    )
    return state_key


def _pixel_loss_from_config(loss_config):
    options = loss_config["pixel"]
    return build_pixel_loss(options.get("type", "l1")), float(
        options.get("weight", 1.0)
    )


def build_perceptual_loss(loss_config, device):
    options = loss_config.get("perceptual")
    if not options or not bool(options.get("enabled", True)):
        return None
    if str(options.get("type", "vgg19")).lower() != "vgg19":
        raise ValueError("Only VGG19 perceptual loss is supported.")
    return PerceptualLoss(
        layer_weights=options["layer_weights"],
        criterion=options.get("criterion", "l1"),
        use_input_norm=options.get("use_input_norm", True),
        range_norm=options.get("range_norm", False),
        perceptual_weight=options.get("weight", 1.0),
        style_weight=options.get("style_weight", 0.0),
    ).to(device)


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def _backward_and_step(loss, optimizer, scaler, parameters, clip_grad_norm):
    if scaler is not None:
        scaler.scale(loss).backward()
        if clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
        scaler.step(optimizer)
    else:
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
        optimizer.step()


def train_stage1_step(
    model,
    batch,
    pixel_criterion,
    pixel_weight,
    optimizer,
    scaler,
    ema,
    use_amp,
    device,
    stage_config,
    step,
    usm_sharpener=None,
):
    model.train()
    learning_rate = set_step_lr(optimizer, step, stage_config)
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    with torch.no_grad():
        pixel_targets = usm_sharpener(targets) if usm_sharpener else targets

    optimizer.zero_grad()
    with amp_context(use_amp):
        outputs = model(inputs)
        pixel_loss = pixel_criterion(outputs, pixel_targets) * pixel_weight
    _backward_and_step(
        pixel_loss,
        optimizer,
        scaler,
        model.parameters(),
        float(stage_config.get("clip_grad_norm", 0.0)),
    )
    if scaler is not None:
        scaler.update()
    ema.update(model)
    return {
        "loss": pixel_loss.item(),
        "pixel_loss": pixel_loss.item(),
        "psnr": calc_psnr(outputs, targets),
        "lr_g": learning_rate,
        "batch_size": inputs.shape[0],
    }


def train_stage2_step(
    model,
    discriminator,
    batch,
    pixel_criterion,
    pixel_weight,
    perceptual_criterion,
    gan_criterion,
    gan_weight,
    optimizer_g,
    optimizer_d,
    scaler,
    ema,
    use_amp,
    device,
    stage_config,
    step,
    usm_sharpener,
):
    model.train()
    discriminator.train()
    lr_g = set_step_lr(optimizer_g, step, stage_config)
    lr_d = set_step_lr(optimizer_d, step, stage_config)
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    usm_config = stage_config.get("usm", {})
    with torch.no_grad():
        targets_usm = usm_sharpener(targets) if usm_sharpener else targets
        pixel_targets = targets_usm if usm_config.get("pixel_target", True) else targets
        perceptual_targets = (
            targets_usm if usm_config.get("perceptual_target", True) else targets
        )
        gan_targets = targets_usm if usm_config.get("gan_target", False) else targets

    optimizer_g.zero_grad()
    set_requires_grad(discriminator, False)
    with amp_context(use_amp):
        outputs = model(inputs)
        pixel_loss = pixel_criterion(outputs, pixel_targets) * pixel_weight
        perceptual_loss, style_loss = perceptual_criterion(
            outputs,
            perceptual_targets,
        )
        generator_gan_loss = (
            gan_criterion(discriminator(outputs), True) * gan_weight
        )
        generator_loss = (
            pixel_loss + perceptual_loss + style_loss + generator_gan_loss
        )
    _backward_and_step(
        generator_loss,
        optimizer_g,
        scaler,
        model.parameters(),
        float(stage_config.get("clip_grad_norm", 0.0)),
    )

    set_requires_grad(discriminator, True)
    optimizer_d.zero_grad()
    with amp_context(use_amp):
        real_logits = discriminator(gan_targets)
        fake_logits = discriminator(outputs.detach())
        discriminator_real_loss = gan_criterion(real_logits, True)
        discriminator_fake_loss = gan_criterion(fake_logits, False)
        discriminator_loss = (
            discriminator_real_loss + discriminator_fake_loss
        ) * 0.5
    _backward_and_step(
        discriminator_loss,
        optimizer_d,
        scaler,
        discriminator.parameters(),
        float(stage_config.get("clip_grad_norm_d", 0.0)),
    )
    if scaler is not None:
        scaler.update()
    ema.update(model)
    return {
        "loss": generator_loss.item(),
        "pixel_loss": pixel_loss.item(),
        "perceptual_loss": perceptual_loss.item(),
        "style_loss": style_loss.item(),
        "gan_loss": generator_gan_loss.item(),
        "d_loss": discriminator_loss.item(),
        "d_real": real_logits.detach().sigmoid().mean().item(),
        "d_fake": fake_logits.detach().sigmoid().mean().item(),
        "psnr": calc_psnr(outputs, targets),
        "lr_g": lr_g,
        "lr_d": lr_d,
        "batch_size": inputs.shape[0],
    }


def update_best_metric(val_metrics, best_metric, metric_name, mode="max"):
    if val_metrics is None:
        return best_metric, False
    if metric_name not in val_metrics:
        raise ValueError(
            f"Validation metric {metric_name!r} is unavailable; "
            f"choose one of {sorted(val_metrics)}."
        )
    current_metric = float(val_metrics[metric_name])
    improved = current_metric > best_metric if mode == "max" else current_metric < best_metric
    return (current_metric, True) if improved else (best_metric, False)


def build_lpips_metric(validation_config, device):
    options = validation_config.get("lpips", {})
    if not bool(options.get("enabled", False)):
        return None
    try:
        import lpips
    except ImportError as error:
        raise ImportError(
            "The lpips package is required when validation.lpips.enabled=true."
        ) from error
    metric = lpips.LPIPS(net=options.get("net", "alex")).to(device).eval()
    metric.requires_grad_(False)
    return metric


@torch.no_grad()
def validate(model, loader, criterion, device, config, step, logger, lpips_metric=None):
    model.eval()
    loss_meter = AverageMeter("PixelLoss")
    rgb_psnr_meter = AverageMeter("RGB_PSNR")
    y_psnr_meter = AverageMeter("Y_PSNR")
    ssim_meter = AverageMeter("SSIM")
    lpips_meter = AverageMeter("LPIPS")
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
        rgb_psnr_meter.update(
            calc_psnr(outputs, targets, crop_border, False),
            batch_size,
        )
        y_psnr_meter.update(
            calc_psnr(outputs, targets, crop_border, True),
            batch_size,
        )
        ssim_meter.update(calc_ssim(outputs, targets, crop_border), batch_size)
        if lpips_metric is not None:
            lpips_value = lpips_metric(
                outputs.clamp(0.0, 1.0) * 2.0 - 1.0,
                targets.clamp(0.0, 1.0) * 2.0 - 1.0,
            ).mean()
            lpips_meter.update(lpips_value.item(), batch_size)

    metrics = {
        "pixel_loss": loss_meter.avg,
        "loss": loss_meter.avg,
        "rgb_psnr": rgb_psnr_meter.avg,
        "y_psnr": y_psnr_meter.avg,
        "ssim": ssim_meter.avg,
    }
    if lpips_metric is not None:
        metrics["lpips"] = lpips_meter.avg
    logger.info(
        "Val step %d | pixel %.6f | RGB PSNR %.3f | Y PSNR %.3f | "
        "SSIM %.5f%s",
        step,
        metrics["pixel_loss"],
        metrics["rgb_psnr"],
        metrics["y_psnr"],
        metrics["ssim"],
        f" | LPIPS {metrics['lpips']:.5f}" if "lpips" in metrics else "",
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPAN-F x2 in two stages.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-hr-dir", default=None)
    parser.add_argument("--train-lr-dir", default=None)
    parser.add_argument("--val-hr-dir", default=None)
    parser.add_argument("--val-lr-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--stage2-pretrained", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--stage1-batch-size", type=int, default=None)
    parser.add_argument("--stage2-batch-size", type=int, default=None)
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

    training = config["training"]
    for key, value in {
        "resume": args.resume,
        "save_dir": args.save_dir,
        "num_workers": args.num_workers,
        "device": args.device,
    }.items():
        if value is not None:
            training[key] = value
    for stage, key, value in (
        ("stage1", "total_steps", args.stage1_steps),
        ("stage2", "total_steps", args.stage2_steps),
        ("stage1", "batch_size", args.stage1_batch_size),
        ("stage2", "batch_size", args.stage2_batch_size),
        ("stage2", "pretrained_generator", args.stage2_pretrained),
    ):
        if value is not None:
            training[stage][key] = value


def make_loaders(config, stage_config, device):
    train_dataset = PairedSRDataset(config["dataset"], split="train")
    val_dataset = PairedSRDataset(config["dataset"], split="val")
    workers = int(stage_config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(stage_config["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=bool(stage_config.get("drop_last", True)),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader


def _save_stage_checkpoint(checkpoint, run_dir, stage_dir, step, save_interval):
    torch.save(checkpoint, run_dir / "latest_model.pth")
    torch.save(checkpoint, stage_dir / "latest_model.pth")
    if save_interval > 0 and step % save_interval == 0:
        torch.save(checkpoint, stage_dir / f"model_step_{step:08d}.pth")


def run_stage(
    stage_name,
    model,
    config,
    device,
    run_dir,
    completed_stages,
    resume_path=None,
):
    training = config["training"]
    stage_config = resolve_stage_config(training, stage_name)
    stage_dir = run_dir / stage_name
    logger = setup_logger(stage_dir, name=f"SPANFx2-{stage_name}")
    logger.info("Starting %s with configuration:\n%s", stage_name, json.dumps(
        stage_config, indent=2, ensure_ascii=False
    ))
    train_loader, val_loader = make_loaders(config, stage_config, device)
    pixel_criterion, pixel_weight = _pixel_loss_from_config(stage_config["loss"])
    pixel_criterion = pixel_criterion.to(device)
    optimizer_g = build_optimizer(model, stage_config, "optimizer_g")
    discriminator = None
    optimizer_d = None
    perceptual_criterion = None
    gan_criterion = None
    gan_weight = 0.0
    if stage_name == "stage2":
        discriminator = build_discriminator(config["discriminator"]).to(device)
        optimizer_d = build_optimizer(discriminator, stage_config, "optimizer_d")
        perceptual_criterion = build_perceptual_loss(
            stage_config["loss"],
            device,
        )
        if perceptual_criterion is None:
            raise ValueError("Stage 2 perceptual loss must be enabled.")
        gan_options = stage_config["loss"].get("gan", {})
        if str(gan_options.get("type", "vanilla")).lower() != "vanilla":
            raise ValueError("Only vanilla GAN loss is supported.")
        gan_criterion = GANLoss(
            real_label=gan_options.get("real_label", 1.0),
            fake_label=gan_options.get("fake_label", 0.0),
        ).to(device)
        gan_weight = float(gan_options.get("weight", 0.1))

    usm_options = stage_config.get("usm", {})
    usm_enabled = bool(usm_options.get("enabled", False))
    usm_sharpener = None
    if usm_enabled:
        usm_sharpener = USMSharp(
            radius=usm_options.get("radius", 51),
            sigma=usm_options.get("sigma", 0.0),
            weight=usm_options.get("weight", 0.5),
            threshold=usm_options.get("threshold", 10.0),
        ).to(device)

    ema = ModelEMA(model, decay=float(stage_config.get("ema_decay", 0.999)))
    use_amp, scaler = get_amp_tools(stage_config.get("amp", True), device)
    start_step = 0
    data_epoch = 0
    best_metrics = {"y_psnr": -float("inf"), "lpips": float("inf")}
    if resume_path:
        checkpoint = load_torch(resume_path, map_location=device)
        if checkpoint.get("active_stage", "stage1") != stage_name:
            raise ValueError(
                f"Resume checkpoint is for {checkpoint.get('active_stage')}, "
                f"not {stage_name}."
            )
        start_step, data_epoch, best_metrics["y_psnr"] = resume_training(
            model,
            ema,
            optimizer_g,
            scaler,
            resume_path,
            device,
            logger,
            steps_per_epoch=len(train_loader),
            discriminator=discriminator,
            optimizer_d=optimizer_d,
        )
        best_metrics.update(checkpoint.get("best_metrics", {}))

    total_steps = int(stage_config["total_steps"])
    log_interval = int(stage_config.get("log_interval", 100))
    val_interval = int(stage_config.get("val_interval", 5000))
    save_interval = int(stage_config.get("save_interval", 5000))
    lpips_metric = build_lpips_metric(config.get("validation", {}), device)
    meters = {}
    last_train_metrics = None
    train_iterator = iter(train_loader)
    step = start_step
    progress = tqdm(
        total=total_steps,
        initial=min(start_step, total_steps),
        desc=stage_name,
        unit="step",
    )
    while step < total_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            data_epoch += 1
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        if stage_name == "stage1":
            step_metrics = train_stage1_step(
                model,
                batch,
                pixel_criterion,
                pixel_weight,
                optimizer_g,
                scaler,
                ema,
                use_amp,
                device,
                stage_config,
                step,
                usm_sharpener,
            )
        else:
            step_metrics = train_stage2_step(
                model,
                discriminator,
                batch,
                pixel_criterion,
                pixel_weight,
                perceptual_criterion,
                gan_criterion,
                gan_weight,
                optimizer_g,
                optimizer_d,
                scaler,
                ema,
                use_amp,
                device,
                stage_config,
                step,
                usm_sharpener,
            )
        step += 1
        batch_size = step_metrics.pop("batch_size")
        for name, value in step_metrics.items():
            meters.setdefault(name, AverageMeter(name)).update(value, batch_size)
        progress.update(1)
        progress.set_postfix(
            loss=f"{meters['loss'].avg:.5f}",
            psnr=f"{meters['psnr'].avg:.2f}",
            lr=f"{step_metrics['lr_g']:.2e}",
        )

        should_log = step % log_interval == 0 or step == total_steps
        if should_log:
            last_train_metrics = {name: meter.avg for name, meter in meters.items()}
            logger.info(
                "%s step %d/%d | %s",
                stage_name,
                step,
                total_steps,
                " | ".join(
                    f"{name} {value:.6g}"
                    for name, value in last_train_metrics.items()
                ),
            )
            for meter in meters.values():
                meter.reset()

        should_validate = val_interval > 0 and (
            step % val_interval == 0 or step == total_steps
        )
        val_metrics = None
        best_psnr = False
        best_lpips = False
        if should_validate:
            val_metrics = validate(
                ema.module,
                val_loader,
                pixel_criterion,
                device,
                config,
                step,
                logger,
                lpips_metric=lpips_metric,
            )
            best_metrics["y_psnr"], best_psnr = update_best_metric(
                val_metrics,
                best_metrics["y_psnr"],
                "y_psnr",
                mode="max",
            )
            if "lpips" in val_metrics:
                best_metrics["lpips"], best_lpips = update_best_metric(
                    val_metrics,
                    best_metrics["lpips"],
                    "lpips",
                    mode="min",
                )

        should_save = (
            (save_interval > 0 and step % save_interval == 0)
            or should_validate
            or step == total_steps
        )
        if should_save:
            finished_stages = list(completed_stages)
            if step == total_steps and stage_name not in finished_stages:
                finished_stages.append(stage_name)
            checkpoint = build_checkpoint(
                model,
                ema,
                optimizer_g,
                scaler,
                step,
                data_epoch,
                best_metrics["y_psnr"],
                {"train": last_train_metrics, "val": val_metrics},
                config,
                active_stage=stage_name,
                completed_stages=finished_stages,
                discriminator=discriminator,
                optimizer_d=optimizer_d,
                best_metrics=best_metrics,
            )
            _save_stage_checkpoint(
                checkpoint,
                run_dir,
                stage_dir,
                step,
                save_interval,
            )
            if best_psnr:
                torch.save(checkpoint, stage_dir / "best_psnr_model.pth")
                torch.save(checkpoint, stage_dir / "best_model.pth")
            if best_lpips:
                torch.save(checkpoint, stage_dir / "best_lpips_model.pth")

    progress.close()
    logger.info("%s finished at step %d.", stage_name, step)
    return ema, list(dict.fromkeys([*completed_stages, stage_name]))


def _resume_run_dir(resume_path):
    parent = Path(resume_path).expanduser().resolve().parent
    if parent.name in STAGE_NAMES:
        return parent.parent
    return parent


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    validate_training_config(config)
    if int(config["model"].get("upscale", 2)) != 2:
        raise ValueError("This training project expects model.upscale=2.")
    if int(config["dataset"].get("scale", 2)) != 2:
        raise ValueError("dataset.scale must be 2.")

    training = config["training"]
    resume_path = training.get("resume")
    if resume_path:
        training["save_dir"] = str(_resume_run_dir(resume_path))
        training["device"] = resolve_auto_device(training.get("device", "auto"))
    else:
        expand_save_dir(config)
    run_dir = Path(training["save_dir"])
    logger = setup_logger(run_dir)
    save_json(config, run_dir / "config.json")
    logger.info("Starting two-stage SPAN-F x2 training.")

    seed = int(training.get("seed", 1234))
    set_random_seed(seed)
    device = torch.device(training["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
    model = build_model(config["model"]).to(device)
    logger.info(
        "Generator parameters: %.3f M registered.",
        count_parameters(model) / 1e6,
    )

    active_stage = None
    completed_stages = []
    if resume_path:
        resume_checkpoint = load_torch(resume_path, map_location="cpu")
        active_stage = resume_checkpoint.get("active_stage")
        if active_stage not in STAGE_NAMES:
            raise ValueError(
                "Two-stage resume checkpoint must contain active_stage."
            )
        completed_stages = list(resume_checkpoint.get("completed_stages", ()))

    stage1_enabled = bool(training["stage1"].get("enabled", True))
    stage2_enabled = bool(training["stage2"].get("enabled", True))
    if active_stage in (None, "stage1") and stage1_enabled:
        stage1_ema, completed_stages = run_stage(
            "stage1",
            model,
            config,
            device,
            run_dir,
            completed_stages,
            resume_path=resume_path if active_stage == "stage1" else None,
        )
        model.load_state_dict(stage1_ema.module.state_dict(), strict=True)
        resume_path = None
        active_stage = None

    if stage2_enabled:
        if active_stage == "stage2":
            pass
        elif not stage1_enabled:
            load_pretrained_model(
                model,
                training["stage2"]["pretrained_generator"],
                device,
                logger,
            )
        elif "stage1" not in completed_stages:
            raise RuntimeError("Stage 2 cannot start before Stage 1 is complete.")
        _, completed_stages = run_stage(
            "stage2",
            model,
            config,
            device,
            run_dir,
            completed_stages,
            resume_path=resume_path if active_stage == "stage2" else None,
        )

    logger.info("Training finished: %s", run_dir / "latest_model.pth")


if __name__ == "__main__":
    main()
