import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
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


class PairedSRDataset(Dataset):
    """Paired LR/HR dataset with optional on-the-fly bicubic LR generation."""

    def __init__(self, args, split="train"):
        super().__init__()
        self.args = dict(args)
        self.split = split
        self.scale = int(args.get("scale", 2))
        self.lr_patch_size = int(args.get("lr_patch_size", 64))
        self.augment = bool(args.get("augment", split == "train")) and split == "train"
        self.hr_root, self.lr_root = _roots_for_split(args, split)

        hr_paths = list_image_files(self.hr_root, recursive=bool(args.get("recursive", True)))
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
            self.samples.append((lr_path, hr_path))

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

    def _make_synthetic_pair(self, hr):
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
        return bicubic_downsample(hr, self.scale), hr

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
        lr_path, hr_path = self.samples[index]
        hr = load_rgb_tensor(hr_path)

        if lr_path is None:
            lr, hr = self._make_synthetic_pair(hr)
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
            "name": hr_path.stem,
            "lr_path": os.fspath(lr_path) if lr_path is not None else "<bicubic>",
            "hr_path": os.fspath(hr_path),
        }
