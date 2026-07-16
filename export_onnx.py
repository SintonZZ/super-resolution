import argparse
import json
from pathlib import Path

import torch

from archs import build_model
from util import clean_state_dict, extract_state_dict, load_torch, resolve_auto_device


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "train.json"


def default_output_path(weights_path=None):
    if not weights_path:
        return Path("spanf_x2_random_deploy.onnx")
    path = Path(weights_path)
    return path.with_name(f"{path.stem}_deploy.onnx")


def load_model_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    model_config = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"Config does not contain a non-empty model section: {config_path}")
    return model_config


def load_model(weights_path, config_path, device):
    if weights_path:
        checkpoint = load_torch(weights_path, map_location=device)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model_config = config.get("model")
        if not model_config:
            raise ValueError("Checkpoint does not contain config.model.")
    else:
        model_config = load_model_config(config_path)

    if int(model_config.get("upscale", 2)) != 2:
        raise ValueError("Model upscale is not 2.")
    model = build_model(model_config).to(device)
    if weights_path:
        state_dict, _ = extract_state_dict(checkpoint)
        model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.switch_to_deploy()
    return model, model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Export a re-parameterized SPAN-F x2 ONNX model.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional training checkpoint. Omit it to export a randomly initialized model.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Model config used when --weights is omitted (default: config/train.json).",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--static-shape", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.input_height, args.input_width, args.batch_size) <= 0:
        raise ValueError("Input height, width, and batch size must be positive.")

    device = torch.device(resolve_auto_device(args.device))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, model_config = load_model(args.weights, args.config, device)
    output_path = Path(args.output) if args.output else default_output_path(args.weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(
        args.batch_size,
        int(model_config.get("in_channels", 3)),
        args.input_height,
        args.input_width,
        dtype=torch.float32,
        device=device,
    )
    dynamic_axes = None
    if not args.static_shape:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height_x2", 3: "width_x2"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
    shape_text = "dynamic NCHW" if dynamic_axes else str(tuple(dummy_input.shape))
    print(f"Exported ONNX: {output_path}")
    if args.weights:
        print(f"Parameters: checkpoint {args.weights}")
    else:
        print(
            "Parameters: random initialization "
            f"(seed={args.seed}, performance/compatibility testing only)"
        )
    print(f"Input: float32 {shape_text}, RGB [0, 1]")
    print("Output: float32 RGB x2")


if __name__ == "__main__":
    main()
