import argparse
import hashlib
import io
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset

from util import IMAGE_EXTENSIONS, list_image_files


def load_rgb_tensor(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(array)


def tensor_to_pil(tensor):
    array = tensor.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8))


def bicubic_downsample(hr, scale):
    height, width = hr.shape[-2:]
    if height % scale != 0 or width % scale != 0:
        raise ValueError(
            f"HR shape {(height, width)} must be divisible by scale={scale} "
            "before synthetic downsampling."
        )
    image = tensor_to_pil(hr)
    image = image.resize((width // scale, height // scale), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _config_key(path, key):
    return f"{path}.{key}"


def _numeric_range(config, key, default, minimum=None, maximum=None, path="degradation"):
    config_key = _config_key(path, key)
    values = config.get(key, default)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{config_key} must contain exactly two numbers.")
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise ValueError(f"{config_key} must be ordered low-to-high, got {values}.")
    if minimum is not None and low < minimum:
        raise ValueError(f"{config_key} must be >= {minimum}, got {values}.")
    if maximum is not None and high > maximum:
        raise ValueError(f"{config_key} must be <= {maximum}, got {values}.")
    return low, high


def _probability(config, key, default, path="degradation"):
    value = float(config.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{_config_key(path, key)} must be in [0, 1], got {value}.")
    return value


def _weights(config, key, default, expected_size, path="degradation"):
    config_key = _config_key(path, key)
    values = config.get(key, default)
    if not isinstance(values, (list, tuple)) or len(values) != expected_size:
        raise ValueError(f"{config_key} must contain exactly {expected_size} numbers.")
    values = [float(value) for value in values]
    if any(value < 0 for value in values):
        raise ValueError(f"{config_key} cannot contain negative values.")
    if sum(values) <= 0:
        raise ValueError(f"{config_key} must have a positive sum.")
    return values


def _anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation):
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    cos_theta = math.cos(rotation)
    sin_theta = math.sin(rotation)
    rotated_x = cos_theta * xx + sin_theta * yy
    rotated_y = -sin_theta * xx + cos_theta * yy
    kernel = np.exp(
        -0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2)
    )
    return kernel / kernel.sum()


def _filter_rgb_tensor(image, kernel):
    kernel_tensor = image.new_tensor(kernel).view(1, 1, *kernel.shape)
    kernel_tensor = kernel_tensor.repeat(image.shape[0], 1, 1, 1)
    padding = kernel.shape[0] // 2
    if image.shape[-2] <= padding or image.shape[-1] <= padding:
        raise ValueError(
            f"Image shape {tuple(image.shape[-2:])} is too small for "
            f"degradation kernel size {kernel.shape[0]}."
        )
    padded = F.pad(image.unsqueeze(0), (padding,) * 4, mode="reflect")
    return F.conv2d(padded, kernel_tensor, groups=image.shape[0]).squeeze(0)


def _resize_tensor(image, size, mode):
    resampling_modes = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if mode not in resampling_modes:
        raise ValueError(
            f"Unsupported degradation resize mode {mode!r}; "
            f"expected one of {sorted(resampling_modes)}."
        )
    pil_image = tensor_to_pil(image)
    pil_image = pil_image.resize(size, resample=resampling_modes[mode])
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _add_gaussian_noise(image, sigma_range, gray_probability, py_rng, np_rng):
    sigma = py_rng.uniform(*sigma_range) / 255.0
    channels = 1 if py_rng.random() < gray_probability else image.shape[0]
    noise = np_rng.normal(
        0.0,
        sigma,
        size=(channels, image.shape[1], image.shape[2]),
    ).astype(np.float32)
    if channels == 1:
        noise = np.repeat(noise, image.shape[0], axis=0)
    return (image + image.new_tensor(noise)).clamp(0.0, 1.0)


def _add_poisson_noise(image, peak_range, gray_probability, py_rng, np_rng):
    peak = py_rng.uniform(*peak_range)
    array = image.numpy()
    if py_rng.random() < gray_probability:
        luminance = (
            array[0] * 0.299 + array[1] * 0.587 + array[2] * 0.114
        )[None, ...]
        noisy_luminance = np_rng.poisson(luminance * peak).astype(np.float32) / peak
        noise = noisy_luminance - luminance
        noisy = array + noise
    else:
        noisy = np_rng.poisson(array * peak).astype(np.float32) / peak
    return torch.from_numpy(np.ascontiguousarray(noisy)).clamp(0.0, 1.0)


def _jpeg_compress(image, quality, subsampling):
    buffer = io.BytesIO()
    tensor_to_pil(image).save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=subsampling,
    )
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        array = np.asarray(compressed.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _windowed_sinc_kernel(kernel_size, cutoff):
    """Build a normalized separable low-pass sinc kernel with a Hamming window."""
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = cutoff * np.sinc(cutoff * coordinates)
    kernel_1d *= np.hamming(kernel_size)
    kernel = np.outer(kernel_1d, kernel_1d)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


class _DegradationStage:
    """One blur/resize/noise/JPEG stage in the high-order pipeline."""

    _RESIZE_DIRECTIONS = ("down", "up", "keep")

    def __init__(self, config, defaults, path):
        self.blur_probability = _probability(
            config,
            "blur_probability",
            defaults["blur_probability"],
            path,
        )
        self.kernel_size = int(config.get("kernel_size", defaults["kernel_size"]))
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError(f"{path}.kernel_size must be an odd integer >= 3.")
        self.isotropic_probability = _probability(
            config,
            "isotropic_probability",
            defaults["isotropic_probability"],
            path,
        )
        self.sigma_range = _numeric_range(
            config,
            "sigma_range",
            defaults["sigma_range"],
            minimum=0.01,
            path=path,
        )
        self.rotation_range = _numeric_range(
            config,
            "rotation_range",
            defaults["rotation_range"],
            path=path,
        )

        self.resize_scale_range = _numeric_range(
            config,
            "resize_scale_range",
            defaults["resize_scale_range"],
            minimum=0.01,
            path=path,
        )
        self.resize_direction_probabilities = _weights(
            config,
            "resize_direction_probabilities",
            defaults["resize_direction_probabilities"],
            len(self._RESIZE_DIRECTIONS),
            path,
        )
        low_scale, high_scale = self.resize_scale_range
        if self.resize_direction_probabilities[0] > 0 and low_scale >= 1.0:
            raise ValueError(
                f"{path}.resize_scale_range must include values below 1 when downsampling "
                "has non-zero probability."
            )
        if self.resize_direction_probabilities[1] > 0 and high_scale <= 1.0:
            raise ValueError(
                f"{path}.resize_scale_range must include values above 1 when upsampling "
                "has non-zero probability."
            )

        self.resize_modes = list(config.get("resize_modes", defaults["resize_modes"]))
        if not self.resize_modes:
            raise ValueError(f"{path}.resize_modes must not be empty.")
        for mode in self.resize_modes:
            _resize_tensor(torch.zeros(3, 2, 2), (1, 1), mode)
        default_mode_probabilities = (
            defaults["resize_mode_probabilities"]
            if "resize_modes" not in config
            else [1.0] * len(self.resize_modes)
        )
        self.resize_mode_probabilities = _weights(
            config,
            "resize_mode_probabilities",
            default_mode_probabilities,
            len(self.resize_modes),
            path,
        )

        self.noise_probability = _probability(
            config,
            "noise_probability",
            defaults["noise_probability"],
            path,
        )
        self.gaussian_noise_probability = _probability(
            config,
            "gaussian_noise_probability",
            defaults["gaussian_noise_probability"],
            path,
        )
        self.gray_noise_probability = _probability(
            config,
            "gray_noise_probability",
            defaults["gray_noise_probability"],
            path,
        )
        self.gaussian_sigma_range = _numeric_range(
            config,
            "gaussian_sigma_range",
            defaults["gaussian_sigma_range"],
            minimum=0.0,
            path=path,
        )
        self.poisson_peak_range = _numeric_range(
            config,
            "poisson_peak_range",
            defaults["poisson_peak_range"],
            minimum=1.0,
            path=path,
        )

        self.jpeg_probability = _probability(
            config,
            "jpeg_probability",
            defaults["jpeg_probability"],
            path,
        )
        self.jpeg_quality_range = _numeric_range(
            config,
            "jpeg_quality_range",
            defaults["jpeg_quality_range"],
            minimum=1,
            maximum=100,
            path=path,
        )
        self.jpeg_subsampling = int(
            config.get("jpeg_subsampling", defaults["jpeg_subsampling"])
        )
        if self.jpeg_subsampling not in (0, 1, 2):
            raise ValueError(f"{path}.jpeg_subsampling must be 0, 1, or 2.")

    def apply_blur(self, image, py_rng):
        if py_rng.random() >= self.blur_probability:
            return image
        sigma_x = py_rng.uniform(*self.sigma_range)
        if py_rng.random() < self.isotropic_probability:
            sigma_y = sigma_x
            rotation = 0.0
        else:
            sigma_y = py_rng.uniform(*self.sigma_range)
            rotation = py_rng.uniform(*self.rotation_range)
        kernel = _anisotropic_gaussian_kernel(
            self.kernel_size,
            sigma_x,
            sigma_y,
            rotation,
        )
        return _filter_rgb_tensor(image, kernel)

    def choose_resize_mode(self, py_rng):
        return py_rng.choices(
            self.resize_modes,
            weights=self.resize_mode_probabilities,
            k=1,
        )[0]

    def apply_resize(self, image, base_size, py_rng):
        direction = py_rng.choices(
            self._RESIZE_DIRECTIONS,
            weights=self.resize_direction_probabilities,
            k=1,
        )[0]
        if direction == "down":
            factor = py_rng.uniform(self.resize_scale_range[0], 1.0)
        elif direction == "up":
            factor = py_rng.uniform(1.0, self.resize_scale_range[1])
        else:
            factor = 1.0
        target_size = (
            max(1, round(base_size[0] * factor)),
            max(1, round(base_size[1] * factor)),
        )
        return _resize_tensor(image, target_size, self.choose_resize_mode(py_rng))

    def apply_noise(self, image, py_rng, np_rng):
        if py_rng.random() >= self.noise_probability:
            return image
        if py_rng.random() < self.gaussian_noise_probability:
            return _add_gaussian_noise(
                image,
                self.gaussian_sigma_range,
                self.gray_noise_probability,
                py_rng,
                np_rng,
            )
        return _add_poisson_noise(
            image,
            self.poisson_peak_range,
            self.gray_noise_probability,
            py_rng,
            np_rng,
        )

    def apply_jpeg(self, image, py_rng):
        if py_rng.random() >= self.jpeg_probability:
            return image
        quality = round(py_rng.uniform(*self.jpeg_quality_range))
        return _jpeg_compress(image, quality, self.jpeg_subsampling)


class HighOrderDegradation:
    """Two-stage high-order blur/resize/noise/JPEG degradation for synthetic LR."""

    _FIRST_DEFAULTS = {
        "blur_probability": 0.8,
        "kernel_size": 15,
        "isotropic_probability": 0.5,
        "sigma_range": (0.2, 2.0),
        "rotation_range": (-math.pi, math.pi),
        "resize_scale_range": (0.5, 1.5),
        "resize_direction_probabilities": (0.7, 0.2, 0.1),
        "resize_modes": ("bicubic", "bilinear", "lanczos"),
        "resize_mode_probabilities": (0.5, 0.25, 0.25),
        "noise_probability": 0.8,
        "gaussian_noise_probability": 0.6,
        "gray_noise_probability": 0.2,
        "gaussian_sigma_range": (1.0, 10.0),
        "poisson_peak_range": (100.0, 1000.0),
        "jpeg_probability": 0.8,
        "jpeg_quality_range": (60, 95),
        "jpeg_subsampling": 2,
    }
    _SECOND_DEFAULTS = {
        "blur_probability": 0.4,
        "kernel_size": 15,
        "isotropic_probability": 0.7,
        "sigma_range": (0.2, 1.2),
        "rotation_range": (-math.pi, math.pi),
        "resize_scale_range": (0.7, 1.2),
        "resize_direction_probabilities": (0.4, 0.3, 0.3),
        "resize_modes": ("bicubic", "bilinear", "lanczos"),
        "resize_mode_probabilities": (0.5, 0.25, 0.25),
        "noise_probability": 0.8,
        "gaussian_noise_probability": 0.6,
        "gray_noise_probability": 0.2,
        "gaussian_sigma_range": (1.0, 8.0),
        "poisson_peak_range": (100.0, 1000.0),
        "jpeg_probability": 1.0,
        "jpeg_quality_range": (60, 95),
        "jpeg_subsampling": 2,
    }

    def __init__(self, config):
        if "first_order" in config:
            first_config = config["first_order"]
        else:
            first_config = {
                key: config[key]
                for key in self._FIRST_DEFAULTS
                if key in config
            }
            if "resize_probabilities" in config:
                first_config["resize_mode_probabilities"] = config[
                    "resize_probabilities"
                ]
        second_config = config.get("second_order", {})
        final_config = config.get("final", {})
        for name, value in (
            ("first_order", first_config),
            ("second_order", second_config),
            ("final", final_config),
        ):
            if not isinstance(value, dict):
                raise ValueError(f"degradation.{name} must be an object.")

        self.first = _DegradationStage(
            first_config,
            self._FIRST_DEFAULTS,
            "degradation.first_order",
        )
        self.second = _DegradationStage(
            second_config,
            self._SECOND_DEFAULTS,
            "degradation.second_order",
        )

        final_path = "degradation.final"
        self.sinc_probability = _probability(
            final_config,
            "sinc_probability",
            0.8,
            final_path,
        )
        self.sinc_kernel_size = int(final_config.get("sinc_kernel_size", 15))
        if self.sinc_kernel_size < 3 or self.sinc_kernel_size % 2 == 0:
            raise ValueError(f"{final_path}.sinc_kernel_size must be an odd integer >= 3.")
        self.sinc_cutoff_range = _numeric_range(
            final_config,
            "sinc_cutoff_range",
            (1.0 / 3.0, 1.0),
            minimum=0.01,
            maximum=1.0,
            path=final_path,
        )
        self.jpeg_before_resize_probability = _probability(
            final_config,
            "jpeg_before_resize_probability",
            0.5,
            final_path,
        )
        self.final_resize_modes = list(
            final_config.get("resize_modes", self._SECOND_DEFAULTS["resize_modes"])
        )
        if not self.final_resize_modes:
            raise ValueError(f"{final_path}.resize_modes must not be empty.")
        for mode in self.final_resize_modes:
            _resize_tensor(torch.zeros(3, 2, 2), (1, 1), mode)
        default_final_mode_probabilities = (
            self._SECOND_DEFAULTS["resize_mode_probabilities"]
            if "resize_modes" not in final_config
            else [1.0] * len(self.final_resize_modes)
        )
        self.final_resize_mode_probabilities = _weights(
            final_config,
            "resize_mode_probabilities",
            default_final_mode_probabilities,
            len(self.final_resize_modes),
            final_path,
        )

    def _resize_to_target(self, image, target_size, py_rng):
        mode = py_rng.choices(
            self.final_resize_modes,
            weights=self.final_resize_mode_probabilities,
            k=1,
        )[0]
        return _resize_tensor(image, target_size, mode)

    def _apply_final_sinc(self, image, py_rng):
        if py_rng.random() >= self.sinc_probability:
            return image
        cutoff = py_rng.uniform(*self.sinc_cutoff_range)
        kernel = _windowed_sinc_kernel(self.sinc_kernel_size, cutoff)
        return _filter_rgb_tensor(image, kernel)

    def __call__(self, hr, scale, py_rng=None, np_rng=None):
        py_rng = random if py_rng is None else py_rng
        np_rng = np.random if np_rng is None else np_rng
        height, width = hr.shape[-2:]
        if height % scale != 0 or width % scale != 0:
            raise ValueError(
                f"HR shape {(height, width)} must be divisible by scale={scale} "
                "before high-order degradation."
            )

        target_size = (width // scale, height // scale)
        degraded = self.first.apply_blur(hr, py_rng)
        degraded = self.first.apply_resize(degraded, (width, height), py_rng)
        degraded = self.first.apply_noise(degraded, py_rng, np_rng)
        degraded = self.first.apply_jpeg(degraded, py_rng)

        degraded = self.second.apply_blur(degraded, py_rng)
        degraded = self.second.apply_resize(degraded, target_size, py_rng)
        degraded = self.second.apply_noise(degraded, py_rng, np_rng)

        if py_rng.random() < self.jpeg_before_resize_probability:
            degraded = self.second.apply_jpeg(degraded, py_rng)
            degraded = self._resize_to_target(degraded, target_size, py_rng)
            degraded = self._apply_final_sinc(degraded, py_rng)
        else:
            degraded = self._resize_to_target(degraded, target_size, py_rng)
            degraded = self._apply_final_sinc(degraded, py_rng)
            degraded = self.second.apply_jpeg(degraded, py_rng)
        return degraded.clamp(0.0, 1.0).contiguous()


# Backward-compatible import name; "realistic" now uses the two-stage pipeline.
RealisticDegradation = HighOrderDegradation


def split_train_val_paths(paths, split, val_ratio=0.05, seed=1234):
    rng = random.Random(seed)
    indices = list(range(len(paths)))
    rng.shuffle(indices)
    num_val = int(round(len(paths) * val_ratio))
    if val_ratio > 0 and len(paths) >= 2:
        num_val = max(num_val, 1)
    num_val = min(max(num_val, 0), len(paths))
    val_indices = set(indices[:num_val])
    if split == "val":
        selected = [path for index, path in enumerate(paths) if index in val_indices]
    else:
        selected = [path for index, path in enumerate(paths) if index not in val_indices]
    if not selected:
        raise ValueError(
            f"No images left for split={split}; images={len(paths)}, val_ratio={val_ratio}."
        )
    return selected


def _roots_for_split(args, split):
    if split == "train":
        hr_root = args.get("train_hr_dir")
        lr_root = args.get("train_lr_dir")
    elif split == "val":
        hr_root = args.get("val_hr_dir") or args.get("train_hr_dir")
        lr_root = args.get("val_lr_dir")
        if not args.get("val_hr_dir"):
            lr_root = lr_root or args.get("train_lr_dir")
    elif split == "test":
        hr_root = args.get("test_hr_dir")
        lr_root = args.get("test_lr_dir")
    else:
        raise ValueError(f"Unsupported split: {split}")

    if not hr_root:
        raise ValueError(f"dataset.{split}_hr_dir must be configured.")
    return Path(hr_root).expanduser(), Path(lr_root).expanduser() if lr_root else None


def _find_lr_path(hr_path, hr_root, lr_root, filename_template):
    relative = hr_path.relative_to(hr_root)
    expected_stem = filename_template.format(hr_path.stem)
    expected = lr_root / relative.parent / f"{expected_stem}{hr_path.suffix}"
    if expected.exists():
        return expected

    parent = lr_root / relative.parent
    if parent.exists():
        matches = [
            path for path in parent.iterdir()
            if path.is_file()
            and path.stem == expected_stem
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(
        f"No LR image found for HR image {hr_path}. Expected a file like {expected}. "
        "Adjust dataset.filename_template, e.g. '{}x2'."
    )


def _resolve_manifest_image_path(manifest_path, value, key, line_number):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{manifest_path}:{line_number} field {key!r} must be a non-empty path."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(
            f"{manifest_path}:{line_number} field {key!r} does not exist: {path}"
        )
    return path


def load_manifest_samples(manifest_path, split):
    """Load explicit paired LR/HR samples for one split from JSON Lines."""
    manifest_path = Path(manifest_path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")

    samples = []
    names = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in dataset manifest {manifest_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{manifest_path}:{line_number} must contain a JSON object."
                )
            record_split = record.get("split")
            if record_split not in ("train", "val", "test"):
                raise ValueError(
                    f"{manifest_path}:{line_number} has invalid split {record_split!r}."
                )
            if record_split != split:
                continue

            lr_path = _resolve_manifest_image_path(
                manifest_path,
                record.get("lr_path"),
                "lr_path",
                line_number,
            )
            hr_path = _resolve_manifest_image_path(
                manifest_path,
                record.get("hr_path"),
                "hr_path",
                line_number,
            )
            name = record.get("name", hr_path.stem)
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"{manifest_path}:{line_number} field 'name' must be a non-empty string."
                )
            if name in names:
                raise ValueError(
                    f"Duplicate sample name {name!r} for split={split} in {manifest_path}."
                )
            names.add(name)
            samples.append((lr_path, hr_path, name))

    if not samples:
        raise ValueError(f"No samples for split={split} in manifest {manifest_path}.")
    return samples


class PairedSRDataset(Dataset):
    """Paired LR/HR dataset with optional on-the-fly synthetic LR generation."""

    def __init__(self, args, split="train"):
        super().__init__()
        self.args = dict(args)
        self.split = split
        self.scale = int(args.get("scale", 2))
        self.lr_patch_size = int(args.get("lr_patch_size", 64))
        self.augment = bool(args.get("augment", split == "train")) and split == "train"
        manifest_path = args.get("manifest_path")
        self.manifest_path = (
            Path(manifest_path).expanduser() if manifest_path else None
        )
        if self.manifest_path is None:
            self.hr_root, self.lr_root = _roots_for_split(args, split)
        else:
            self.hr_root, self.lr_root = None, None

        degradation_config = args.get("degradation", {})
        if degradation_config is None:
            degradation_config = {}
        if not isinstance(degradation_config, dict):
            raise ValueError("dataset.degradation must be an object.")
        self.degradation_type = str(degradation_config.get("type", "bicubic")).lower()
        if self.degradation_type not in ("bicubic", "high_order", "realistic"):
            raise ValueError(
                "dataset.degradation.type must be 'bicubic' or 'high_order' "
                "('realistic' is kept as a compatibility alias)."
            )
        self.high_order_degradation = (
            HighOrderDegradation(degradation_config)
            if self.degradation_type in ("high_order", "realistic")
            and split in ("train", "val")
            else None
        )
        self.validation_degradation_seed = int(
            degradation_config.get("validation_seed", args.get("seed", 1234))
        )

        if self.manifest_path is not None:
            self.samples = load_manifest_samples(self.manifest_path, split)
            max_images = args.get("max_images")
            if max_images is not None:
                self.samples = self.samples[:int(max_images)]
            if not self.samples:
                raise ValueError(
                    f"No samples left for split={split} after applying max_images."
                )
        else:
            hr_paths = list_image_files(
                self.hr_root,
                recursive=bool(args.get("recursive", True)),
            )
            uses_train_val_split = split in ("train", "val") and not args.get("val_hr_dir")
            if uses_train_val_split:
                hr_paths = split_train_val_paths(
                    hr_paths,
                    split=split,
                    val_ratio=float(args.get("val_ratio", 0.05)),
                    seed=int(args.get("seed", 1234)),
                )

            max_images = args.get("max_images")
            if max_images is not None:
                hr_paths = hr_paths[:int(max_images)]
            if not hr_paths:
                raise ValueError(f"No HR images found under {self.hr_root}")

            filename_template = args.get("filename_template", "{}")
            self.samples = []
            for hr_path in hr_paths:
                lr_path = None
                if self.lr_root is not None:
                    lr_path = _find_lr_path(
                        hr_path,
                        self.hr_root,
                        self.lr_root,
                        filename_template,
                    )
                self.samples.append((lr_path, hr_path, hr_path.stem))

    def __len__(self):
        return len(self.samples)

    def _random_crop_paired(self, lr, hr):
        lr_height, lr_width = lr.shape[-2:]
        patch = self.lr_patch_size
        if lr_height < patch or lr_width < patch:
            raise ValueError(
                f"LR image {(lr_height, lr_width)} is smaller than lr_patch_size={patch}."
            )
        top = random.randint(0, lr_height - patch)
        left = random.randint(0, lr_width - patch)
        lr = lr[..., top:top + patch, left:left + patch]
        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_patch = patch * self.scale
        hr = hr[..., hr_top:hr_top + hr_patch, hr_left:hr_left + hr_patch]
        return lr, hr

    def _validation_seed_for_path(self, hr_path):
        relative_path = hr_path.relative_to(self.hr_root).as_posix().encode("utf-8")
        path_seed = int.from_bytes(hashlib.sha256(relative_path).digest()[:4], "little")
        return (self.validation_degradation_seed + path_seed) % (2 ** 32)

    def _make_synthetic_pair(self, hr, hr_path):
        height, width = hr.shape[-2:]
        if self.split == "train":
            hr_patch = self.lr_patch_size * self.scale
            if height < hr_patch or width < hr_patch:
                raise ValueError(
                    f"HR image {(height, width)} is smaller than required patch "
                    f"{(hr_patch, hr_patch)}."
                )
            top = random.randint(0, height - hr_patch)
            left = random.randint(0, width - hr_patch)
            hr = hr[..., top:top + hr_patch, left:left + hr_patch]
        else:
            valid_height = height - height % self.scale
            valid_width = width - width % self.scale
            if valid_height == 0 or valid_width == 0:
                raise ValueError(f"HR image is too small for scale={self.scale}: {(height, width)}")
            hr = hr[..., :valid_height, :valid_width]
        if self.high_order_degradation is not None:
            if self.split == "train":
                lr = self.high_order_degradation(hr, self.scale)
            else:
                validation_seed = self._validation_seed_for_path(hr_path)
                lr = self.high_order_degradation(
                    hr,
                    self.scale,
                    py_rng=random.Random(validation_seed),
                    np_rng=np.random.RandomState(validation_seed),
                )
        else:
            lr = bicubic_downsample(hr, self.scale)
        return lr, hr

    def _augment_pair(self, lr, hr):
        if not self.augment:
            return lr, hr
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        if random.random() < 0.5:
            lr = lr.transpose(1, 2)
            hr = hr.transpose(1, 2)
        return lr.contiguous(), hr.contiguous()

    def __getitem__(self, index):
        lr_path, hr_path, name = self.samples[index]
        hr = load_rgb_tensor(hr_path)

        if lr_path is None:
            lr, hr = self._make_synthetic_pair(hr, hr_path)
        else:
            lr = load_rgb_tensor(lr_path)
            expected_hr_shape = (lr.shape[-2] * self.scale, lr.shape[-1] * self.scale)
            if tuple(hr.shape[-2:]) != expected_hr_shape:
                raise ValueError(
                    f"Scale mismatch for {lr_path} and {hr_path}: LR={tuple(lr.shape[-2:])}, "
                    f"HR={tuple(hr.shape[-2:])}, expected HR={expected_hr_shape}."
                )
            if self.split == "train":
                lr, hr = self._random_crop_paired(lr, hr)

        lr, hr = self._augment_pair(lr, hr)
        return {
            "input": lr,
            "target": hr,
            "name": name,
            "lr_path": (
                os.fspath(lr_path)
                if lr_path is not None
                else (
                    "<synthetic:high-order>"
                    if self.high_order_degradation is not None
                    else "<bicubic>"
                )
            ),
            "hr_path": os.fspath(hr_path),
        }


def make_lr_hr_comparison(lr, hr, resize_mode="bicubic"):
    """Create a labeled side-by-side preview with LR enlarged to HR size."""
    resampling_modes = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if resize_mode not in resampling_modes:
        raise ValueError(
            f"Unsupported preview resize mode {resize_mode!r}; "
            f"expected one of {sorted(resampling_modes)}."
        )

    lr_image = tensor_to_pil(lr)
    hr_image = tensor_to_pil(hr)
    lr_preview = lr_image.resize(hr_image.size, resample=resampling_modes[resize_mode])

    header_height = 32
    gap = 8
    canvas = Image.new(
        "RGB",
        (hr_image.width * 2 + gap, hr_image.height + header_height),
        color=(24, 24, 24),
    )
    canvas.paste(lr_preview, (0, header_height))
    canvas.paste(hr_image, (hr_image.width + gap, header_height))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 9),
        f"LR {lr_image.width}x{lr_image.height} ({resize_mode} preview)",
        fill=(255, 255, 255),
    )
    draw.text(
        (hr_image.width + gap + 8, 9),
        f"HR {hr_image.width}x{hr_image.height}",
        fill=(255, 255, 255),
    )
    return canvas


def _load_preview_config(config_path):
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    dataset_config = config.get("dataset", config)
    if not isinstance(dataset_config, dict):
        raise ValueError("Config must contain a dataset object.")
    return dataset_config


def preview_main():
    parser = argparse.ArgumentParser(
        description="Visualize a synthetic/paired LR sample next to its HR target."
    )
    parser.add_argument("--config", default="config/train.json")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resize-mode",
        choices=("nearest", "bilinear", "bicubic", "lanczos"),
        default="bicubic",
        help="How to enlarge LR for the side-by-side preview.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    dataset_config = _load_preview_config(args.config)
    seed = int(dataset_config.get("seed", 1234) if args.seed is None else args.seed)
    if args.seed is not None and args.split == "val":
        dataset_config = dict(dataset_config)
        degradation_config = dict(dataset_config.get("degradation") or {})
        degradation_config["validation_seed"] = seed
        dataset_config["degradation"] = degradation_config
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = PairedSRDataset(dataset_config, split=args.split)
    index = args.index if args.index >= 0 else len(dataset) + args.index
    if not 0 <= index < len(dataset):
        raise IndexError(f"Sample index {args.index} is outside dataset size {len(dataset)}.")
    item = dataset[index]
    comparison = make_lr_hr_comparison(
        item["input"],
        item["target"],
        resize_mode=args.resize_mode,
    )

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        output_path = (
            Path(__file__).resolve().parent
            / "results"
            / "dataset_preview"
            / f"{args.split}_{index:04d}_{item['name']}.png"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)

    print(f"Saved comparison: {output_path}")
    print(f"LR: {tuple(item['input'].shape)} from {item['lr_path']}")
    print(f"HR: {tuple(item['target'].shape)} from {item['hr_path']}")
    print(f"Seed: {seed}")
    if args.show:
        comparison.show()


if __name__ == "__main__":
    preview_main()
