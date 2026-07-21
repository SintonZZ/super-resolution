import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import (
    HighOrderDegradation,
    PairedSRDataset,
    make_lr_hr_comparison,
)


def save_random_image(path, height, width, seed):
    rng = np.random.RandomState(seed)
    array = rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def high_order_config(first_overrides=None, second_overrides=None, final_overrides=None):
    def stage_config(overrides):
        config = {
            "blur_probability": 0.0,
            "kernel_size": 7,
            "resize_scale_range": [1.0, 1.0],
            "resize_direction_probabilities": [0.0, 0.0, 1.0],
            "resize_modes": ["bicubic"],
            "resize_mode_probabilities": [1.0],
            "noise_probability": 0.0,
            "jpeg_probability": 0.0,
        }
        config.update(overrides or {})
        return config

    final = {
        "sinc_probability": 0.0,
        "sinc_kernel_size": 7,
        "jpeg_before_resize_probability": 0.0,
        "resize_modes": ["bicubic"],
        "resize_mode_probabilities": [1.0],
    }
    final.update(final_overrides or {})
    return {
        "type": "high_order",
        "first_order": stage_config(first_overrides),
        "second_order": stage_config(second_overrides),
        "final": final,
    }


class DatasetTest(unittest.TestCase):
    def test_manifest_loads_relative_pairs_and_filters_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_random_image(root / "train" / "HR" / "train.png", 32, 40, 20)
            save_random_image(root / "train" / "LR" / "train.png", 16, 20, 21)
            save_random_image(root / "val" / "HR" / "val.png", 24, 28, 22)
            save_random_image(root / "val" / "LR" / "val.png", 12, 14, 23)
            manifest = root / "manifest.jsonl"
            rows = [
                {
                    "name": "train_sample",
                    "split": "train",
                    "lr_path": "train/LR/train.png",
                    "hr_path": "train/HR/train.png",
                },
                {
                    "name": "val_sample",
                    "split": "val",
                    "lr_path": "val/LR/val.png",
                    "hr_path": "val/HR/val.png",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            train_dataset = PairedSRDataset(
                {
                    "manifest_path": str(manifest),
                    "scale": 2,
                    "lr_patch_size": 8,
                    "augment": False,
                },
                split="train",
            )
            val_dataset = PairedSRDataset(
                {"manifest_path": str(manifest), "scale": 2},
                split="val",
            )
            train_item = train_dataset[0]
            val_item = val_dataset[0]

        self.assertEqual(len(train_dataset), 1)
        self.assertEqual(train_item["name"], "train_sample")
        self.assertEqual(tuple(train_item["input"].shape), (3, 8, 8))
        self.assertEqual(tuple(train_item["target"].shape), (3, 16, 16))
        self.assertEqual(val_item["name"], "val_sample")
        self.assertEqual(tuple(val_item["input"].shape), (3, 12, 14))
        self.assertEqual(tuple(val_item["target"].shape), (3, 24, 28))

    def test_hr_only_generates_aligned_x2_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                save_random_image(root / f"{index:04d}.png", 36, 40, index)
            dataset = PairedSRDataset({
                "train_hr_dir": str(root),
                "train_lr_dir": None,
                "val_hr_dir": str(root),
                "scale": 2,
                "lr_patch_size": 8,
                "augment": False,
            }, split="train")
            item = dataset[0]
        self.assertEqual(tuple(item["input"].shape), (3, 8, 8))
        self.assertEqual(tuple(item["target"].shape), (3, 16, 16))
        self.assertEqual(item["lr_path"], "<bicubic>")

    def test_paired_filename_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr_root = root / "HR"
            lr_root = root / "LR"
            save_random_image(hr_root / "scene.png", 32, 40, 1)
            save_random_image(lr_root / "scenex2.png", 16, 20, 2)
            dataset = PairedSRDataset({
                "val_hr_dir": str(hr_root),
                "val_lr_dir": str(lr_root),
                "scale": 2,
                "filename_template": "{}x2",
            }, split="val")
            item = dataset[0]
        self.assertEqual(tuple(item["input"].shape), (3, 16, 20))
        self.assertEqual(tuple(item["target"].shape), (3, 32, 40))

    def test_high_order_degradation_generates_valid_aligned_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_random_image(root / "scene.png", 40, 48, 3)
            dataset = PairedSRDataset({
                "train_hr_dir": str(root),
                "train_lr_dir": None,
                "val_hr_dir": str(root),
                "scale": 2,
                "lr_patch_size": 16,
                "augment": False,
                "degradation": high_order_config(
                    first_overrides={
                        "blur_probability": 1.0,
                        "sigma_range": [1.2, 1.2],
                        "resize_scale_range": [0.8, 1.2],
                        "resize_direction_probabilities": [1.0, 0.0, 0.0],
                        "resize_modes": ["bilinear"],
                        "resize_mode_probabilities": [1.0],
                        "noise_probability": 1.0,
                        "gaussian_noise_probability": 1.0,
                        "gaussian_sigma_range": [5.0, 5.0],
                        "jpeg_probability": 1.0,
                        "jpeg_quality_range": [70, 70],
                    },
                    second_overrides={
                        "blur_probability": 1.0,
                        "sigma_range": [0.8, 0.8],
                        "resize_scale_range": [0.8, 1.2],
                        "resize_direction_probabilities": [0.0, 1.0, 0.0],
                        "noise_probability": 1.0,
                        "gaussian_noise_probability": 0.0,
                        "poisson_peak_range": [500.0, 500.0],
                        "jpeg_probability": 1.0,
                        "jpeg_quality_range": [80, 80],
                    },
                    final_overrides={
                        "sinc_probability": 1.0,
                        "sinc_cutoff_range": [0.7, 0.7],
                    },
                ),
            }, split="train")
            random.seed(9)
            np.random.seed(9)
            item = dataset[0]

        self.assertEqual(tuple(item["input"].shape), (3, 16, 16))
        self.assertEqual(tuple(item["target"].shape), (3, 32, 32))
        self.assertGreaterEqual(float(item["input"].min()), 0.0)
        self.assertLessEqual(float(item["input"].max()), 1.0)
        self.assertEqual(item["lr_path"], "<synthetic:high-order>")

    def test_second_order_training_degradation_changes_between_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_random_image(root / "scene.png", 32, 32, 4)
            dataset = PairedSRDataset({
                "train_hr_dir": str(root),
                "train_lr_dir": None,
                "val_hr_dir": str(root),
                "scale": 2,
                "lr_patch_size": 16,
                "augment": False,
                "degradation": high_order_config(second_overrides={
                    "noise_probability": 1.0,
                    "gaussian_noise_probability": 1.0,
                    "gaussian_sigma_range": [10.0, 10.0],
                }),
            }, split="train")
            random.seed(10)
            np.random.seed(10)
            first = dataset[0]["input"]
            second = dataset[0]["input"]

        self.assertFalse(torch.equal(first, second))

    def test_validation_uses_fixed_high_order_degradation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_random_image(root / "scene.png", 32, 40, 5)
            config = {
                "train_hr_dir": str(root),
                "train_lr_dir": None,
                "val_hr_dir": str(root),
                "val_lr_dir": None,
                "scale": 2,
                "augment": False,
                "degradation": high_order_config(second_overrides={
                    "noise_probability": 1.0,
                    "gaussian_noise_probability": 1.0,
                    "gaussian_sigma_range": [25.0, 25.0],
                }),
            }
            config["degradation"]["validation_seed"] = 99
            dataset = PairedSRDataset(config, split="val")
            first = dataset[0]

            random.seed(123456)
            np.random.seed(123456)
            second = dataset[0]
            recreated = PairedSRDataset(config, split="val")[0]

        self.assertTrue(torch.equal(first["input"], second["input"]))
        self.assertTrue(torch.equal(first["input"], recreated["input"]))
        self.assertEqual(first["lr_path"], "<synthetic:high-order>")

    def test_validation_seed_changes_fixed_high_order_degradation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_random_image(root / "scene.png", 32, 40, 6)
            config = {
                "val_hr_dir": str(root),
                "val_lr_dir": None,
                "scale": 2,
                "degradation": high_order_config(second_overrides={
                    "noise_probability": 1.0,
                    "gaussian_noise_probability": 1.0,
                    "gaussian_sigma_range": [25.0, 25.0],
                }),
            }
            config["degradation"]["validation_seed"] = 10
            first = PairedSRDataset(config, split="val")[0]["input"]
            config["degradation"]["validation_seed"] = 11
            second = PairedSRDataset(config, split="val")[0]["input"]

        self.assertFalse(torch.equal(first, second))

    def test_realistic_type_is_a_high_order_compatibility_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_random_image(root / "scene.png", 32, 40, 7)
            degradation_config = high_order_config()
            degradation_config["type"] = "realistic"
            dataset = PairedSRDataset({
                "val_hr_dir": str(root),
                "val_lr_dir": None,
                "scale": 2,
                "degradation": degradation_config,
            }, split="val")
            item = dataset[0]

        self.assertEqual(tuple(item["input"].shape), (3, 16, 20))
        self.assertEqual(item["lr_path"], "<synthetic:high-order>")

    def test_flat_realistic_options_migrate_to_first_order(self):
        degradation = HighOrderDegradation({
            "type": "realistic",
            "blur_probability": 0.25,
            "kernel_size": 7,
            "resize_modes": ["bilinear"],
            "resize_probabilities": [1.0],
            "noise_probability": 0.5,
            "jpeg_probability": 0.75,
        })

        self.assertEqual(degradation.first.kernel_size, 7)
        self.assertEqual(degradation.first.resize_modes, ["bilinear"])
        self.assertEqual(degradation.first.resize_mode_probabilities, [1.0])
        self.assertAlmostEqual(degradation.first.blur_probability, 0.25)
        self.assertAlmostEqual(degradation.first.noise_probability, 0.5)
        self.assertAlmostEqual(degradation.first.jpeg_probability, 0.75)

    def test_lr_hr_comparison_has_labeled_side_by_side_layout(self):
        lr = torch.zeros(3, 8, 10)
        hr = torch.ones(3, 16, 20)
        comparison = make_lr_hr_comparison(lr, hr, resize_mode="nearest")

        self.assertEqual(comparison.mode, "RGB")
        self.assertEqual(comparison.size, (48, 48))
        array = np.asarray(comparison)
        self.assertTrue(np.all(array[32:, :20] == 0))
        self.assertTrue(np.all(array[32:, 28:] == 255))


if __name__ == "__main__":
    unittest.main()
