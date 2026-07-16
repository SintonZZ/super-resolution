import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from archs import build_model
from dataset import PairedSRDataset
from util import (
    AverageMeter,
    calc_psnr,
    clean_state_dict,
    extract_state_dict,
    forward_tiled,
    load_torch,
    resolve_auto_device,
    save_image,
    save_json,
    save_sr_comparison,
)


DEFAULT_CONFIG_PATH = "config/test.json"


def load_config(config_path):
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model(weights_path, fallback_model_config, device, strict=True):
    checkpoint = load_torch(weights_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_config = checkpoint_config.get("model", fallback_model_config)
    if not model_config:
        raise ValueError("Model config is missing from both checkpoint and test config.")

    model = build_model(model_config).to(device)
    state_dict, _ = extract_state_dict(checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=strict)
    model.switch_to_deploy()
    return model, checkpoint_config, model_config


@torch.no_grad()
def run_test(model, loader, device, config, model_config):
    output_dir = Path(config["test"]["output_dir"])
    image_dir = output_dir / "images"
    comparison_dir = output_dir / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = int(model_config["upscale"])
    crop_border = int(config["test"].get("crop_border", scale))
    tile_size = int(config["test"].get("tile_size", 0))
    tile_pad = int(config["test"].get("tile_pad", 24))
    save_images = bool(config["test"].get("save_images", True))
    max_save_comparisons = int(config["test"].get("max_save_comparisons", 20))

    rgb_meter = AverageMeter("RGB_PSNR")
    y_meter = AverageMeter("Y_PSNR")
    rows = []
    for index, batch in enumerate(tqdm(loader, desc="Testing")):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        outputs = forward_tiled(model, inputs, scale, tile_size, tile_pad).clamp(0.0, 1.0)

        rgb_psnr = calc_psnr(outputs, targets, crop_border, False)
        y_psnr = calc_psnr(outputs, targets, crop_border, True)
        rgb_meter.update(rgb_psnr, inputs.shape[0])
        y_meter.update(y_psnr, inputs.shape[0])

        name = f"{index:05d}_{batch['name'][0]}"
        rows.append({
            "name": batch["name"][0],
            "lr_path": batch["lr_path"][0],
            "hr_path": batch["hr_path"][0],
            "rgb_psnr": rgb_psnr,
            "y_psnr": y_psnr,
        })
        if save_images:
            save_image(outputs, image_dir / f"{name}_x{scale}.png")
        if index < max_save_comparisons:
            save_sr_comparison(
                inputs,
                outputs,
                targets,
                comparison_dir / f"{name}_input_output_target.png",
            )

    metrics = {
        "num_images": len(rows),
        "rgb_psnr": rgb_meter.avg,
        "y_psnr": y_meter.avg,
        "items": rows,
    }
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SPAN-F x2 checkpoint.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--test-hr-dir", default=None)
    parser.add_argument("--test-lr-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--filename-template", default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-pad", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config, args):
    if args.weights:
        config["model"]["weights_path"] = args.weights
    for key in ("test_hr_dir", "test_lr_dir", "filename_template", "max_images"):
        value = getattr(args, key)
        if value is not None:
            config["dataset"][key] = value
    for key in ("output_dir", "tile_size", "tile_pad", "device"):
        value = getattr(args, key)
        if value is not None:
            config["test"][key] = value


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    config["test"]["device"] = resolve_auto_device(config["test"].get("device", "auto"))
    weights_path = config["model"].get("weights_path")
    if not weights_path:
        raise ValueError("Set model.weights_path in config or pass --weights.")

    device = torch.device(config["test"]["device"])
    model, train_config, model_config = load_model(
        weights_path,
        config.get("model"),
        device,
        strict=bool(config["model"].get("strict", True)),
    )
    if int(model_config.get("upscale", 2)) != 2:
        raise ValueError("The loaded checkpoint is not configured for x2 upscaling.")

    dataset_config = copy.deepcopy(train_config.get("dataset", {}))
    dataset_config.update(config["dataset"])
    dataset_config["scale"] = 2
    dataset = PairedSRDataset(dataset_config, split="test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["test"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    metrics = run_test(model, loader, device, config, model_config)
    print(json.dumps({key: value for key, value in metrics.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
