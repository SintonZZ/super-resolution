#!/usr/bin/env python3
"""Build x2 RGB super-resolution pairs from clean OV50Q RAW frames.

Each clean RAW produces two degraded variants. One uses the parameterized
synthetic noise model from raw2dnr, and the other adds shot noise to a real
black frame captured at a matching analog-gain setting. Both variants pass
through Raw2DNR and the simple ``raw2rgb_stable`` ISP before being downsampled
to the LR full-frame image. The clean, brightness-adjusted RAW bypasses DNR and
uses the same ISP gains as its degraded counterpart to form the HR reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW2DNR_REPO = REPO_ROOT.parent / "raw2dnr"
DEFAULT_INPUT_ROOT = Path("/mnt/d/ov50q_real_raw")
DEFAULT_BLACK_FRAME_ROOT = Path("/mnt/d/ov50q/dark/processed")
DEFAULT_OUTPUT_ROOT = Path("/mnt/d/ov50q_sr_dataset")
DEFAULT_CHECKPOINT = (
    DEFAULT_RAW2DNR_REPO
    / "checkpoints"
    / "run_20260710_172943_raw2dnr_OV50Q_24_3"
    / "latest_model.pth"
)
DEFAULT_SENSOR_INFO = DEFAULT_RAW2DNR_REPO / "sensor_infos" / "ov50q.json"
DEFAULT_TRAIN_GROUPS = ("20260617", "20260622")
DEFAULT_VAL_GROUPS = ("20260624",)
DEFAULT_TEST_GROUPS = ("20260701",)
DEFAULT_BLACK_LEVEL_CH = (63.812873, 63.795504, 63.960863, 64.089181)
NOISE_BRANCHES = ("synthetic", "blackframe")
RESIZE_MODES = ("area", "bicubic", "lanczos")
RESIZE_WEIGHTS = (0.25, 0.5, 0.25)
RESIZE_INTERPOLATION = {
    "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def path_arg(value: str | Path) -> Path:
    return Path(value).expanduser()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\0".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f"OpenCV failed to write image: {temporary}")
    os.replace(temporary, path)


def pack_bayer(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2 or raw.shape[0] % 2 or raw.shape[1] % 2:
        raise ValueError(f"Expected an even-sized 2D Bayer image, got {raw.shape}.")
    return np.stack(
        (
            raw[0::2, 0::2],
            raw[0::2, 1::2],
            raw[1::2, 0::2],
            raw[1::2, 1::2],
        ),
        axis=0,
    )


def unpack_bayer(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected packed Bayer with shape (4,H,W), got {packed.shape}.")
    _, height, width = packed.shape
    raw = np.empty((height * 2, width * 2), dtype=packed.dtype)
    raw[0::2, 0::2] = packed[0]
    raw[0::2, 1::2] = packed[1]
    raw[1::2, 0::2] = packed[2]
    raw[1::2, 1::2] = packed[3]
    return raw


def normalize_active_signal(
    raw_dn: np.ndarray,
    black_level_ch: Sequence[float],
    white_level: float,
) -> np.ndarray:
    packed = pack_bayer(raw_dn.astype(np.float32, copy=False))
    black = np.asarray(black_level_ch, dtype=np.float32).reshape(4, 1, 1)
    if black.shape != (4, 1, 1) or np.any(black < 0) or np.any(black >= white_level):
        raise ValueError(f"Invalid Bayer black levels: {black.ravel().tolist()}")
    return np.clip((packed - black) / (white_level - black), 0.0, 1.0).astype(
        np.float32
    )


def apply_bayer_blc(
    raw_norm: np.ndarray,
    black_level_ch: Sequence[float],
    white_level: float,
) -> np.ndarray:
    raw_dn = raw_norm.astype(np.float32, copy=False) * float(white_level)
    active = normalize_active_signal(raw_dn, black_level_ch, white_level)
    return unpack_bayer(active)


def adjust_brightness(
    active: np.ndarray,
    rng: np.random.Generator,
    min_brightness: float = 0.005,
    brightness_power: float = 3.0,
    highlight_threshold: float = 0.98,
) -> Tuple[np.ndarray, float]:
    """Apply raw2dnr's low-light sampling rule to packed active Bayer data."""
    if active.ndim != 3 or active.shape[0] != 4:
        raise ValueError(f"Expected packed Bayer active signal, got {active.shape}.")
    if not 0.0 < min_brightness <= 1.0:
        raise ValueError("min_brightness must be in (0, 1].")
    if brightness_power <= 0:
        raise ValueError("brightness_power must be positive.")

    green = 0.5 * (active[1:2] + active[2:3])
    valid = green < highlight_threshold
    image_mean = float(green[valid].mean()) if np.any(valid) else float(green.mean())
    if image_mean <= min_brightness:
        return np.clip(active, 0.0, 1.0).astype(np.float32), 1.0

    max_ratio = max(image_mean / min_brightness, 1.0)
    unit = float(rng.random())
    ratio = 1.0 + (max_ratio - 1.0) * unit ** (1.0 / brightness_power)
    return np.clip(active / ratio, 0.0, 1.0).astype(np.float32), float(ratio)


def discover_clean_raws(
    input_root: Path,
    split_groups: Mapping[str, Sequence[str]],
    expected_bytes: int,
) -> Tuple[Dict[str, List[Path]], List[Dict[str, Any]]]:
    assigned: Dict[str, str] = {}
    for split, groups in split_groups.items():
        for group in groups:
            if group in assigned:
                raise ValueError(f"Capture group {group!r} appears in two splits.")
            assigned[group] = split

    valid = {split: [] for split in split_groups}
    rejected: List[Dict[str, Any]] = []
    for group, split in assigned.items():
        group_root = input_root / group
        if not group_root.is_dir():
            raise FileNotFoundError(f"Capture group does not exist: {group_root}")
        for path in sorted(group_root.rglob("*.raw")):
            size = path.stat().st_size
            if size != expected_bytes:
                rejected.append(
                    {
                        "path": os.fspath(path),
                        "reason": "unexpected_size",
                        "actual_bytes": size,
                        "expected_bytes": expected_bytes,
                    }
                )
            else:
                valid[split].append(path)
    return valid, rejected


def discover_black_frames(
    root: Path,
    gains: Sequence[int],
    expected_bytes: int,
) -> Dict[int, List[Path]]:
    index: Dict[int, List[Path]] = {}
    for gain in gains:
        directory = root / f"ag{gain}"
        paths = sorted(
            path
            for path in directory.glob("*.raw")
            if path.is_file() and path.stat().st_size == expected_bytes
        )
        if not paths:
            raise FileNotFoundError(f"No valid black frames found for gain {gain}: {directory}")
        index[int(gain)] = paths
    return index


def read_raw_uint16(path: Path, height: int, width: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    expected = height * width
    if raw.size != expected:
        raise ValueError(f"{path} contains {raw.size} pixels; expected {expected}.")
    return raw.reshape(height, width)


def choose_path(paths: Sequence[Path], rng: np.random.Generator) -> Path:
    return paths[int(rng.integers(0, len(paths)))]


def black_frame_channel_means(black_dn: np.ndarray) -> np.ndarray:
    return pack_bayer(black_dn.astype(np.float32, copy=False)).mean(
        axis=(1, 2), keepdims=True
    )


def k_stratum(k_values: Sequence[float], index: int) -> Tuple[float, float]:
    values = np.asarray(k_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(values <= 0):
        raise ValueError(f"Invalid K values: {k_values}")
    if index < 0 or index >= len(values):
        raise IndexError(index)
    boundaries = np.sqrt(values[:-1] * values[1:])
    low = values[0] if index == 0 else boundaries[index - 1]
    high = values[-1] if index == len(values) - 1 else boundaries[index]
    return float(low), float(high)


def add_synthetic_noise(
    active: np.ndarray,
    sensor: Mapping[str, Any],
    stratum_index: int,
    black_frames: Mapping[int, Sequence[Path]],
    height: int,
    width: int,
    white_level: float,
    rng: np.random.Generator,
    k_bias_power: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    gains = [int(value) for value in sensor["again"]]
    k_values = [float(value) for value in sensor["sample_params"]["K_list"]]
    low, high = k_stratum(k_values, stratum_index)
    k_value = low + (high - low) * float(rng.random()) ** (1.0 / k_bias_power)

    log_k = math.log(k_value)
    row_config = sensor["sample_params"]["row"]
    tl_config = sensor["sample_params"]["tl"]
    log_sig_row = rng.normal(
        float(row_config["a"]) * log_k + float(row_config["b"]),
        float(row_config["std"]),
    )
    log_sig_tl = rng.normal(
        float(tl_config["a"]) * log_k + float(tl_config["b"]),
        float(tl_config["std"]),
    )
    sig_row = float(np.exp(log_sig_row))
    sig_tl = float(np.exp(log_sig_tl))
    lam_values = [float(value) for value in sensor["sample_params"]["lam_list"]]
    lam = float(rng.uniform(min(lam_values), max(lam_values)))

    nearest_index = int(np.argmin(np.abs(np.log(k_values) - log_k)))
    nearest_gain = gains[nearest_index]
    black_path = choose_path(black_frames[nearest_gain], rng)
    black_dn = read_raw_uint16(black_path, height, width)
    black = black_frame_channel_means(black_dn)

    signal_dn = active * (white_level - black)
    shot = rng.poisson(np.maximum(signal_dn / k_value, 0.0)).astype(np.float32)
    shot *= k_value
    if math.isclose(sig_row, 1.0, rel_tol=0.0, abs_tol=1e-7):
        row_noise: np.ndarray | float = 0.0
    else:
        row_noise = rng.standard_normal((4, active.shape[1], 1)).astype(np.float32)
        row_noise *= sig_row
    read_noise = stats.tukeylambda.rvs(
        lam,
        loc=0.0,
        scale=sig_tl,
        size=active.shape,
        random_state=rng,
    ).astype(np.float32)
    quant_noise = rng.uniform(-0.5, 0.5, size=active.shape).astype(np.float32)
    noisy = (shot + row_noise + read_noise + quant_noise + black) / white_level
    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
    return unpack_bayer(noisy), {
        "method": "synthetic",
        "gain_stratum": gains[stratum_index],
        "K": k_value,
        "K_range": [low, high],
        "lambda": lam,
        "sig_row": sig_row,
        "sig_tl": sig_tl,
        "black_level_ch": [float(value) for value in black.ravel()],
        "black_frame_for_bl": os.fspath(black_path),
    }


def add_black_frame_noise(
    active: np.ndarray,
    sensor: Mapping[str, Any],
    stratum_index: int,
    black_frames: Mapping[int, Sequence[Path]],
    height: int,
    width: int,
    white_level: float,
    rng: np.random.Generator,
    k_jitter: float = 0.3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    gains = [int(value) for value in sensor["again"]]
    k_values = [float(value) for value in sensor["sample_params"]["K_list"]]
    gain = gains[stratum_index]
    base_k = k_values[stratum_index]
    k_value = base_k * float(rng.uniform(1.0, 1.0 + k_jitter))
    black_path = choose_path(black_frames[gain], rng)
    black_dn = read_raw_uint16(black_path, height, width)
    black_packed = pack_bayer(black_dn.astype(np.float32, copy=False))
    black = black_packed.mean(axis=(1, 2), keepdims=True)

    signal_dn = active * (white_level - black)
    shot = rng.poisson(np.maximum(signal_dn / k_value, 0.0)).astype(np.float32)
    shot *= k_value
    noisy = np.clip((shot + black_packed) / white_level, 0.0, 1.0).astype(
        np.float32
    )
    return unpack_bayer(noisy), {
        "method": "blackframe",
        "gain": gain,
        "K_base": base_k,
        "K": k_value,
        "k_jitter": k_jitter,
        "black_level_ch": [float(value) for value in black.ravel()],
        "black_frame": os.fspath(black_path),
    }


def load_external_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_torch(path: Path, map_location):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def ensure_torch_pixel_ops() -> None:
    """Provide pixel shuffle helpers for the older local PyTorch test runtime."""
    import torch.nn.functional as functional

    if not hasattr(functional, "pixel_unshuffle"):
        def pixel_unshuffle(inputs, downscale_factor):
            batch, channels, height, width = inputs.shape
            factor = int(downscale_factor)
            if height % factor or width % factor:
                raise RuntimeError("pixel_unshuffle input dimensions must be divisible by factor")
            return (
                inputs.view(
                    batch,
                    channels,
                    height // factor,
                    factor,
                    width // factor,
                    factor,
                )
                .permute(0, 1, 3, 5, 2, 4)
                .contiguous()
                .view(
                    batch,
                    channels * factor * factor,
                    height // factor,
                    width // factor,
                )
            )

        functional.pixel_unshuffle = pixel_unshuffle

    if not hasattr(functional, "pixel_shuffle"):
        def pixel_shuffle(inputs, upscale_factor):
            batch, channels, height, width = inputs.shape
            factor = int(upscale_factor)
            if channels % (factor * factor):
                raise RuntimeError("pixel_shuffle channels must be divisible by factor squared")
            output_channels = channels // (factor * factor)
            return (
                inputs.view(
                    batch,
                    output_channels,
                    factor,
                    factor,
                    height,
                    width,
                )
                .permute(0, 1, 4, 2, 5, 3)
                .contiguous()
                .view(
                    batch,
                    output_channels,
                    height * factor,
                    width * factor,
                )
            )

        functional.pixel_shuffle = pixel_shuffle


def clean_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "params", "params_ema"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def load_raw2dnr_model(repo: Path, checkpoint_path: Path, device: torch.device):
    ensure_torch_pixel_ops()
    module = load_external_module(
        "sr_dataset_external_raw2dnr_arch",
        repo / "archs" / "raw2dnr.py",
    )
    state = clean_state_dict(load_torch(checkpoint_path, map_location=device))
    first_weight = state.get("inc.double_conv.0.weight")
    if first_weight is None:
        raise ValueError("Cannot infer Raw2DNR base channels from checkpoint.")
    level_indices = [
        int(key.split(".")[1])
        for key in state
        if key.startswith("downs.")
        and key.endswith(".0.weight")
        and key.split(".")[1].isdigit()
    ]
    if not level_indices:
        raise ValueError("Cannot infer Raw2DNR levels from checkpoint.")
    model = module.Raw2DNR(
        base_ch=int(first_weight.shape[0]),
        num_levels=max(level_indices) + 1,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_simple_isp(repo: Path) -> Callable[..., Tuple[np.ndarray, float, Tuple[float, ...]]]:
    module = load_external_module("sr_dataset_external_raw2dnr_util", repo / "util.py")
    return module.raw2rgb_stable


@torch.no_grad()
def run_raw2dnr(
    model,
    raw_norm: np.ndarray,
    device: torch.device,
    pad_base: int = 32,
) -> np.ndarray:
    height, width = raw_norm.shape
    pad_h = (pad_base - height % pad_base) % pad_base
    pad_w = (pad_base - width % pad_base) % pad_base
    padded = np.pad(raw_norm, ((0, pad_h), (0, pad_w)), mode="reflect")
    input_tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
    output = model(input_tensor).clamp(0.0, 1.0)
    return output.squeeze().cpu().numpy()[:height, :width].astype(np.float32)


def choose_resize_mode(rng: np.random.Generator) -> str:
    index = int(rng.choice(len(RESIZE_MODES), p=np.asarray(RESIZE_WEIGHTS)))
    return RESIZE_MODES[index]


def downsample_x2(image: np.ndarray, mode: str) -> np.ndarray:
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError(f"Image dimensions must be even for x2 downsampling: {image.shape}")
    if mode not in RESIZE_INTERPOLATION:
        raise ValueError(f"Unsupported resize mode: {mode}")
    return cv2.resize(
        image,
        (width // 2, height // 2),
        interpolation=RESIZE_INTERPOLATION[mode],
    )


def sample_id_for(source: Path, input_root: Path, branch: str) -> str:
    relative = source.relative_to(input_root)
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{relative.parent.name}_{source.stem}_{branch}_{digest}"


def sidecar_path(output_root: Path, split: str, sample_id: str) -> Path:
    return output_root / "records" / split / f"{sample_id}.json"


def completed_record(output_root: Path, split: str, sample_id: str) -> Dict[str, Any] | None:
    path = sidecar_path(output_root, split, sample_id)
    if not path.is_file():
        return None
    record = load_json(path)
    for key in ("lr_path", "hr_path"):
        target = output_root / record[key]
        if not target.is_file() or target.stat().st_size == 0:
            return None
    return record


def collect_records(output_root: Path) -> List[Dict[str, Any]]:
    records = [load_json(path) for path in sorted((output_root / "records").glob("*/*.json"))]
    records.sort(key=lambda item: (item["split"], item["name"]))
    return records


def build_split_groups(args) -> Dict[str, Sequence[str]]:
    return {
        "train": tuple(args.train_groups),
        "val": tuple(args.val_groups),
        "test": tuple(args.test_groups),
    }


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build full-frame x2 SR pairs from clean OV50Q RAW images."
    )
    parser.add_argument("--input-root", type=path_arg, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--black-frame-root", type=path_arg, default=DEFAULT_BLACK_FRAME_ROOT)
    parser.add_argument("--output-root", type=path_arg, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw2dnr-repo", type=path_arg, default=DEFAULT_RAW2DNR_REPO)
    parser.add_argument("--checkpoint", type=path_arg, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sensor-info", type=path_arg, default=DEFAULT_SENSOR_INFO)
    parser.add_argument("--train-groups", nargs="+", default=list(DEFAULT_TRAIN_GROUPS))
    parser.add_argument("--val-groups", nargs="+", default=list(DEFAULT_VAL_GROUPS))
    parser.add_argument("--test-groups", nargs="+", default=list(DEFAULT_TEST_GROUPS))
    parser.add_argument("--black-level-ch", nargs=4, type=float, default=DEFAULT_BLACK_LEVEL_CH)
    parser.add_argument("--min-brightness", type=float, default=0.005)
    parser.add_argument("--brightness-power", type=float, default=3.0)
    parser.add_argument("--highlight-threshold", type=float, default=0.98)
    parser.add_argument("--synthetic-k-bias-power", type=float, default=3.0)
    parser.add_argument("--k-jitter", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_configuration(args, sensor: Mapping[str, Any]) -> Tuple[int, int, float, List[int]]:
    height = int(sensor["height"])
    width = int(sensor["width"])
    white_level = float(2 ** int(sensor["bit_depth"]) - 1)
    gains = [int(value) for value in sensor["again"]]
    if height % 2 or width % 2:
        raise ValueError("Sensor dimensions must be even.")
    if len(gains) != len(sensor["sample_params"]["K_list"]):
        raise ValueError("sensor.again and sample_params.K_list must have equal lengths.")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive.")
    if args.k_jitter < 0:
        raise ValueError("--k-jitter must be non-negative.")
    return height, width, white_level, gains


def summarize_scan(
    valid: Mapping[str, Sequence[Path]],
    rejected: Sequence[Mapping[str, Any]],
    black_frames: Mapping[int, Sequence[Path]],
) -> Dict[str, Any]:
    split_counts = {split: len(paths) for split, paths in valid.items()}
    valid_count = sum(split_counts.values())
    return {
        "valid_clean_raws": valid_count,
        "rejected_clean_raws": len(rejected),
        "split_clean_raws": split_counts,
        "expected_pairs": valid_count * len(NOISE_BRANCHES),
        "expected_noise_methods": {
            branch: valid_count for branch in NOISE_BRANCHES
        },
        "black_frames": {f"ag{gain}": len(paths) for gain, paths in black_frames.items()},
    }


def limit_sources(valid: Dict[str, List[Path]], max_images: int | None) -> Dict[str, List[Path]]:
    if max_images is None:
        return valid
    remaining = max_images
    limited: Dict[str, List[Path]] = {}
    for split in ("train", "val", "test"):
        paths = valid.get(split, [])
        limited[split] = paths[:remaining]
        remaining -= len(limited[split])
        if remaining <= 0:
            remaining = 0
    return limited


def process_variant(
    *,
    args,
    source: Path,
    split: str,
    split_index: int,
    branch: str,
    active: np.ndarray,
    sensor: Mapping[str, Any],
    gains: Sequence[int],
    black_frames: Mapping[int, Sequence[Path]],
    model,
    simple_isp,
    device: torch.device,
    height: int,
    width: int,
    white_level: float,
) -> Dict[str, Any]:
    sample_id = sample_id_for(source, args.input_root, branch)
    existing = completed_record(args.output_root, split, sample_id)
    if existing is not None and args.resume and not args.force:
        return existing

    branch_offset = 0 if branch == "synthetic" else len(gains) // 2
    stratum_index = (split_index + branch_offset) % len(gains)
    seed = stable_seed(args.seed, source.relative_to(args.input_root), branch)
    rng = np.random.default_rng(seed)
    low_light, brightness_ratio = adjust_brightness(
        active,
        rng,
        min_brightness=args.min_brightness,
        brightness_power=args.brightness_power,
        highlight_threshold=args.highlight_threshold,
    )
    if branch == "synthetic":
        noisy_raw, noise_metadata = add_synthetic_noise(
            low_light,
            sensor,
            stratum_index,
            black_frames,
            height,
            width,
            white_level,
            rng,
            k_bias_power=args.synthetic_k_bias_power,
        )
    else:
        noisy_raw, noise_metadata = add_black_frame_noise(
            low_light,
            sensor,
            stratum_index,
            black_frames,
            height,
            width,
            white_level,
            rng,
            k_jitter=args.k_jitter,
        )

    denoised_raw = run_raw2dnr(model, noisy_raw, device)
    denoised_blc = apply_bayer_blc(denoised_raw, args.black_level_ch, white_level)
    clean_blc = unpack_bayer(low_light)
    degraded_bgr, amplifier, wb_gains = simple_isp(denoised_blc)
    reference_bgr = simple_isp(clean_blc, amplifier, wb_gains)[0]
    resize_mode = choose_resize_mode(rng)
    lr_bgr = downsample_x2(degraded_bgr, resize_mode)

    lr_relative = Path(split) / "LR" / f"{sample_id}.png"
    hr_relative = Path(split) / "HR" / f"{sample_id}.png"
    atomic_write_png(args.output_root / lr_relative, lr_bgr)
    atomic_write_png(args.output_root / hr_relative, reference_bgr)
    record = {
        "name": sample_id,
        "split": split,
        "source_raw": os.fspath(source),
        "noise_method": branch,
        "seed": seed,
        "brightness_ratio": brightness_ratio,
        "noise": noise_metadata,
        "isp": {
            "type": "raw2dnr.util.raw2rgb_stable",
            "amplifier": float(amplifier),
            "wb_gains": [float(value) for value in wb_gains],
        },
        "resize_mode": resize_mode,
        "lr_path": lr_relative.as_posix(),
        "hr_path": hr_relative.as_posix(),
        "lr_size": [width // 2, height // 2],
        "hr_size": [width, height],
    }
    atomic_write_json(sidecar_path(args.output_root, split, sample_id), record)
    return record


def main() -> None:
    args = parse_args()
    sensor = load_json(args.sensor_info)
    height, width, white_level, gains = validate_configuration(args, sensor)
    expected_bytes = height * width * np.dtype(np.uint16).itemsize
    split_groups = build_split_groups(args)
    valid, rejected = discover_clean_raws(args.input_root, split_groups, expected_bytes)
    valid = limit_sources(valid, args.max_images)
    black_frames = discover_black_frames(args.black_frame_root, gains, expected_bytes)
    scan_summary = summarize_scan(valid, rejected, black_frames)
    print(json.dumps(scan_summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    existing_sidecars = list((args.output_root / "records").glob("*/*.json"))
    if existing_sidecars and not (args.resume or args.force):
        raise FileExistsError(
            f"Found {len(existing_sidecars)} existing records under {args.output_root}. "
            "Use --resume to continue or --force to overwrite deterministic outputs."
        )
    device = resolve_device(args.device)
    print(f"Loading Raw2DNR on {device}: {args.checkpoint}")
    model = load_raw2dnr_model(args.raw2dnr_repo, args.checkpoint, device)
    simple_isp = load_simple_isp(args.raw2dnr_repo)

    processed = Counter()
    for split in ("train", "val", "test"):
        sources = valid.get(split, [])
        progress = tqdm(sources, desc=f"Building {split}")
        for split_index, source in enumerate(progress):
            raw_dn = read_raw_uint16(source, height, width)
            active = normalize_active_signal(raw_dn, args.black_level_ch, white_level)
            for branch in NOISE_BRANCHES:
                record = process_variant(
                    args=args,
                    source=source,
                    split=split,
                    split_index=split_index,
                    branch=branch,
                    active=active,
                    sensor=sensor,
                    gains=gains,
                    black_frames=black_frames,
                    model=model,
                    simple_isp=simple_isp,
                    device=device,
                    height=height,
                    width=width,
                    white_level=white_level,
                )
                processed[(record["split"], record["noise_method"])] += 1

    records = collect_records(args.output_root)
    atomic_write_jsonl(args.output_root / "manifest.jsonl", records)
    metadata = {
        "input_root": os.fspath(args.input_root),
        "black_frame_root": os.fspath(args.black_frame_root),
        "checkpoint": os.fspath(args.checkpoint),
        "sensor_info": os.fspath(args.sensor_info),
        "output_root": os.fspath(args.output_root),
        "seed": args.seed,
        "height": height,
        "width": width,
        "white_level": white_level,
        "black_level_ch": list(args.black_level_ch),
        "split_groups": split_groups,
        "scan": scan_summary,
        "completed_records": len(records),
        "completed_by_split_and_method": {
            f"{split}:{method}": count
            for (split, method), count in sorted(processed.items())
        },
        "manifest": "manifest.jsonl",
    }
    atomic_write_json(args.output_root / "metadata.json", metadata)
    atomic_write_json(args.output_root / "rejected.json", {"items": rejected})
    print(f"Wrote {len(records)} records to {args.output_root / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
