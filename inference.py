import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from archs import build_span
from dataset import load_rgb_tensor
from util import (
    clean_state_dict,
    extract_state_dict,
    forward_tiled,
    list_image_files,
    load_torch,
    resolve_auto_device,
    save_image,
)


def load_model(weights_path, device):
    checkpoint = load_torch(weights_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_config = config.get("model")
    if not model_config:
        raise ValueError("Checkpoint does not contain config.model; use a checkpoint from train.py.")
    if int(model_config.get("upscale", 2)) != 2:
        raise ValueError("Checkpoint model.upscale is not 2.")

    model = build_span(model_config).to(device)
    state_dict, _ = extract_state_dict(checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.switch_to_deploy()
    return model, int(model_config["upscale"])


def parse_args():
    parser = argparse.ArgumentParser(description="Upscale one image or a directory with SPAN x2.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True, help="Input image or directory.")
    parser.add_argument("--output-dir", default="results/inference")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--tile-size", type=int, default=0, help="LR tile size; 0 disables tiling.")
    parser.add_argument("--tile-pad", type=int, default=24)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(resolve_auto_device(args.device))
    model, scale = load_model(args.weights, device)
    paths = list_image_files(args.input, recursive=args.recursive)
    if not paths:
        raise ValueError(f"No images found: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(tqdm(paths, desc="Inference")):
        input_tensor = load_rgb_tensor(path).unsqueeze(0).to(device)
        output = forward_tiled(
            model,
            input_tensor,
            scale,
            tile_size=args.tile_size,
            tile_pad=args.tile_pad,
        ).clamp(0.0, 1.0)
        save_image(output, output_dir / f"{index:05d}_{path.stem}_x{scale}.png")

    print(f"Saved {len(paths)} image(s) to {output_dir}")


if __name__ == "__main__":
    main()
