#!/usr/bin/env python3
"""Static PTQ for a deployed SPAN-F ONNX model with ONNX Runtime.

Default settings:
  - calibration: min-max
  - activation quantization: per-tensor asymmetric uint8
  - weight quantization: per-tensor asymmetric uint8
  - quantized op types: Conv
  - output format: QDQ

Calibration data can be RGB images or NCHW/CHW ``.npy`` arrays. Image inputs
are converted to RGB float32 in [0, 1], resized only when too small, center
cropped, transposed to NCHW, and wrapped in a batch of one.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
ONNX_TO_NUMPY_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT8: np.int8,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT16: np.int16,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT32: np.int32,
    TensorProto.UINT32: np.uint32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT64: np.uint64,
    TensorProto.BOOL: np.bool_,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize a deployed SPAN-F ONNX model using ONNX Runtime static "
            "PTQ with image or NumPy calibration samples."
        )
    )
    parser.add_argument("--model-input", type=Path, required=True, help="Input FP32 ONNX model.")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Output QDQ ONNX model. Default: <model-input-stem>_ptq.onnx",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        required=True,
        help="Directory containing LR RGB images or .npy calibration samples.",
    )
    parser.add_argument(
        "--sample-format",
        choices=("auto", "images", "npy"),
        default="auto",
        help="Calibration sample format. Auto prefers images when both exist.",
    )
    parser.add_argument(
        "--input-name",
        default=None,
        help="Model input name. If omitted, the model must have one real input.",
    )
    parser.add_argument(
        "--calib-height",
        type=int,
        default=None,
        help="LR calibration crop height. Uses a fixed model height or 256 by default.",
    )
    parser.add_argument(
        "--calib-width",
        type=int,
        default=None,
        help="LR calibration crop width. Uses a fixed model width or 256 by default.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum calibration samples; 0 uses all samples. Default: 128.",
    )
    parser.add_argument(
        "--quant-format",
        choices=("qdq", "qoperator"),
        default="qdq",
        help="Quantized model format. Default: qdq.",
    )
    parser.add_argument(
        "--op-types",
        nargs="+",
        default=["Conv"],
        help="Operator types to quantize. Default: Conv.",
    )
    parser.add_argument(
        "--calib-max-intermediate-outputs",
        type=int,
        default=0,
        help="Limit cached calibration outputs; 0 leaves the ORT default.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output model.")
    return parser.parse_args()


def default_output_path(model_input: Path) -> Path:
    return model_input.with_name(f"{model_input.stem}_ptq.onnx")


def load_quantization_api():
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except AttributeError as exc:
        if "INT4" in str(exc):
            raise RuntimeError(
                "onnxruntime.quantization requires a newer onnx package. "
                "Install this project's requirements (onnx>=1.16)."
            ) from exc
        raise
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime quantization is unavailable. Install requirements.txt."
        ) from exc
    return CalibrationMethod, QuantFormat, QuantType, quantize_static


def get_real_model_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return [
        graph_input
        for graph_input in model.graph.input
        if graph_input.name not in initializer_names
    ]


def input_shape(value_info: onnx.ValueInfoProto) -> Tuple[object, ...]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return tuple(dims)


def resolve_input(
    model: onnx.ModelProto,
    requested_name: Optional[str],
) -> onnx.ValueInfoProto:
    inputs = get_real_model_inputs(model)
    if requested_name is not None:
        for value_info in inputs:
            if value_info.name == requested_name:
                return value_info
        names = [value_info.name for value_info in inputs]
        raise ValueError(f"Input {requested_name!r} not found; model inputs: {names}")
    if len(inputs) != 1:
        names = [value_info.name for value_info in inputs]
        raise ValueError(f"Expected one real model input, found: {names}")
    return inputs[0]


def resolve_calibration_size(
    value_info: onnx.ValueInfoProto,
    requested_height: Optional[int],
    requested_width: Optional[int],
) -> Tuple[int, int]:
    shape = input_shape(value_info)
    if len(shape) != 4:
        raise ValueError(f"SPAN-F input must be NCHW, got {shape}")

    batch, channels, model_height, model_width = shape
    if isinstance(batch, int) and batch != 1:
        raise ValueError(f"Calibration currently requires model batch size 1, got {batch}")
    if isinstance(channels, int) and channels != 3:
        raise ValueError(f"RGB calibration requires 3 input channels, got {channels}")

    height = _resolve_dimension("height", model_height, requested_height)
    width = _resolve_dimension("width", model_width, requested_width)
    return height, width


def _resolve_dimension(name: str, model_dim: object, requested: Optional[int]) -> int:
    if requested is not None and requested <= 0:
        raise ValueError(f"Calibration {name} must be positive, got {requested}")
    if isinstance(model_dim, int) and model_dim > 0:
        if requested is not None and requested != model_dim:
            raise ValueError(
                f"Calibration {name}={requested} does not match static model {name}={model_dim}"
            )
        return model_dim
    return requested if requested is not None else 256


def conv_group(node: onnx.NodeProto) -> int:
    for attribute in node.attribute:
        if attribute.name == "group":
            return int(attribute.i)
    return 1


def validate_deploy_graph(model: onnx.ModelProto) -> None:
    grouped = [
        (node.name or "<unnamed>", conv_group(node))
        for node in model.graph.node
        if node.op_type == "Conv" and conv_group(node) != 1
    ]
    if grouped:
        raise ValueError(
            "Input ONNX still contains grouped convolutions. Export it with the current "
            f"export_onnx.py first. Found: {grouped[:4]}"
        )

    unsupported_activations = [
        node.op_type for node in model.graph.node if node.op_type.lower() in ("silu", "swish")
    ]
    if unsupported_activations:
        raise ValueError(
            "Input ONNX contains SiLU/Swish nodes; expected the Sigmoid + Mul decomposition."
        )


def discover_samples(
    calib_dir: Path,
    sample_format: str,
    max_samples: int,
) -> Tuple[str, List[Path]]:
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration directory does not exist: {calib_dir}")
    if max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}")

    image_paths = sorted(
        path
        for path in calib_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    npy_paths = sorted(path for path in calib_dir.rglob("*.npy") if path.is_file())

    if sample_format == "images":
        selected_format, samples = "images", image_paths
    elif sample_format == "npy":
        selected_format, samples = "npy", npy_paths
    elif image_paths:
        selected_format, samples = "images", image_paths
    else:
        selected_format, samples = "npy", npy_paths

    if not samples:
        expected = "RGB images" if selected_format == "images" else ".npy files"
        raise FileNotFoundError(f"No {expected} found under {calib_dir}")
    if max_samples > 0:
        samples = samples[:max_samples]
    return selected_format, samples


def image_to_nchw(path: Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        current_width, current_height = image.size
        if current_height < height or current_width < width:
            scale = max(height / current_height, width / current_width)
            resized = (
                max(int(round(current_width * scale)), width),
                max(int(round(current_height * scale)), height),
            )
            image = image.resize(resized, resample=Image.Resampling.BICUBIC)

        left = (image.width - width) // 2
        top = (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        array = np.asarray(image, dtype=np.float32) / 255.0

    array = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])
    return array


def npy_to_nchw(path: Path, height: int, width: int) -> np.ndarray:
    array = np.load(path)
    if array.ndim == 3:
        array = array[None, ...]
    if array.shape != (1, 3, height, width):
        raise ValueError(
            f"{path} has shape {array.shape}; expected CHW or NCHW matching "
            f"(1, 3, {height}, {width})"
        )
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    if array.min() < 0.0 or array.max() > 1.0:
        raise ValueError(f"{path} values must be in [0, 1]")
    return np.ascontiguousarray(array)


class SRCalibrationDataReader:
    """Streams deterministic RGB image or NumPy calibration samples."""

    def __init__(
        self,
        input_info: onnx.ValueInfoProto,
        samples: Sequence[Path],
        sample_format: str,
        height: int,
        width: int,
    ) -> None:
        self.input_info = input_info
        self.input_name = input_info.name
        self.samples = list(samples)
        self.sample_format = sample_format
        self.height = int(height)
        self.width = int(width)
        self.index = 0

    def __len__(self) -> int:
        return len(self.samples)

    def rewind(self) -> None:
        self.index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.index >= len(self.samples):
            return None
        path = self.samples[self.index]
        self.index += 1
        if self.sample_format == "images":
            array = image_to_nchw(path, self.height, self.width)
        else:
            array = npy_to_nchw(path, self.height, self.width)

        dtype = ONNX_TO_NUMPY_DTYPE.get(self.input_info.type.tensor_type.elem_type)
        if dtype is not None and array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        return {self.input_name: array}


def quant_format_from_arg(name: str, quant_format_type):
    if name == "qdq":
        return quant_format_type.QDQ
    if name == "qoperator":
        return quant_format_type.QOperator
    raise ValueError(f"Unsupported quantization format: {name}")


def summarize_quantized_model(model_path: Path) -> Counter:
    model = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(model)
    return Counter(node.op_type for node in model.graph.node)


def main() -> int:
    args = parse_args()
    output_path = args.model_output or default_output_path(args.model_input)

    if not args.model_input.exists():
        raise FileNotFoundError(f"Input model does not exist: {args.model_input}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output exists; use --force to overwrite: {output_path}")

    model = onnx.load(str(args.model_input), load_external_data=False)
    onnx.checker.check_model(model)
    validate_deploy_graph(model)
    input_info = resolve_input(model, args.input_name)
    height, width = resolve_calibration_size(
        input_info,
        args.calib_height,
        args.calib_width,
    )
    sample_format, samples = discover_samples(
        args.calib_dir,
        args.sample_format,
        args.max_samples,
    )
    reader = SRCalibrationDataReader(
        input_info,
        samples,
        sample_format,
        height,
        width,
    )

    CalibrationMethod, QuantFormat, QuantType, quantize_static = load_quantization_api()
    extra_options = {
        "ActivationSymmetric": False,
        "WeightSymmetric": False,
        "CalibTensorRangeSymmetric": False,
    }
    if args.calib_max_intermediate_outputs > 0:
        extra_options["CalibMaxIntermediateOutputs"] = args.calib_max_intermediate_outputs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input model: {args.model_input}")
    print(f"Output model: {output_path}")
    print(f"Model input: {input_info.name} {input_shape(input_info)}")
    print(f"Calibration: {len(reader)} {sample_format} samples, NCHW=(1, 3, {height}, {width})")
    print(f"Quantized op types: {args.op_types}")
    print("Quantization: activation=QUInt8 asymmetric, weight=QUInt8 asymmetric, per-tensor")
    print("Calibration method: MinMax")

    quantize_static(
        model_input=str(args.model_input),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=quant_format_from_arg(args.quant_format, QuantFormat),
        op_types_to_quantize=args.op_types,
        per_channel=False,
        reduce_range=False,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options=extra_options,
    )

    counts = summarize_quantized_model(output_path)
    print(
        "Quantized graph: "
        f"Conv={counts['Conv']}, QuantizeLinear={counts['QuantizeLinear']}, "
        f"DequantizeLinear={counts['DequantizeLinear']}"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
