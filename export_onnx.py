import argparse
from pathlib import Path

import torch

from archs import build_span
from util import clean_state_dict, extract_state_dict, load_torch, resolve_auto_device


def default_output_path(weights_path):
    path = Path(weights_path)
    return path.with_name(f"{path.stem}_deploy.onnx")


def load_model(weights_path, device):
    checkpoint = load_torch(weights_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_config = config.get("model")
    if not model_config:
        raise ValueError("Checkpoint does not contain config.model.")
    model = build_span(model_config).to(device)
    state_dict, _ = extract_state_dict(checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.switch_to_deploy()
    return model, model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Export a re-parameterized SPAN x2 ONNX model.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--static-shape", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.input_height, args.input_width, args.batch_size) <= 0:
        raise ValueError("Input height, width, and batch size must be positive.")

    device = torch.device(resolve_auto_device(args.device))
    model, model_config = load_model(args.weights, device)
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
    print(f"Input: float32 {shape_text}, RGB [0, 1]")
    print("Output: float32 RGB x2")


if __name__ == "__main__":
    main()
