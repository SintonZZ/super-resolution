import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from prepare_raw2dnr_dataset import (
    add_black_frame_noise,
    add_synthetic_noise,
    adjust_brightness,
    discover_black_frames,
    discover_clean_raws,
    downsample_x2,
    pack_bayer,
    run_raw2dnr,
    stable_seed,
    unpack_bayer,
)


def write_raw(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(np.uint16).tofile(path)


def tiny_sensor():
    return {
        "again": [1, 4],
        "sample_params": {
            "K_list": [0.5, 2.0],
            "row": {"a": 0.0, "b": 0.0, "std": 0.0},
            "tl": {"a": 0.0, "b": -2.0, "std": 0.0},
            "lam_list": [-0.05, 0.05],
        },
    }


class PrepareRaw2DNRDatasetTest(unittest.TestCase):
    def test_pack_unpack_round_trip(self):
        raw = np.arange(48, dtype=np.float32).reshape(6, 8)
        packed = pack_bayer(raw)
        restored = unpack_bayer(packed)
        self.assertEqual(packed.shape, (4, 3, 4))
        self.assertTrue(np.array_equal(restored, raw))

    def test_brightness_sampling_is_deterministic_and_only_darkens(self):
        active = np.full((4, 4, 5), 0.2, dtype=np.float32)
        first, first_ratio = adjust_brightness(
            active,
            np.random.default_rng(9),
            min_brightness=0.01,
            brightness_power=3.0,
        )
        second, second_ratio = adjust_brightness(
            active,
            np.random.default_rng(9),
            min_brightness=0.01,
            brightness_power=3.0,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_ratio, second_ratio)
        self.assertGreaterEqual(first_ratio, 1.0)
        self.assertLessEqual(float(first.max()), float(active.max()))

    def test_noise_branches_are_deterministic_and_bounded(self):
        height, width = 8, 10
        active = np.full((4, height // 2, width // 2), 0.08, dtype=np.float32)
        black = np.full((height, width), 64, dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for gain in (1, 4):
                write_raw(root / f"ag{gain}" / "black.raw", black + gain)
            index = discover_black_frames(root, [1, 4], height * width * 2)

            synthetic_a, synthetic_meta = add_synthetic_noise(
                active,
                tiny_sensor(),
                0,
                index,
                height,
                width,
                1023.0,
                np.random.default_rng(11),
            )
            synthetic_b, _ = add_synthetic_noise(
                active,
                tiny_sensor(),
                0,
                index,
                height,
                width,
                1023.0,
                np.random.default_rng(11),
            )
            blackframe, blackframe_meta = add_black_frame_noise(
                active,
                tiny_sensor(),
                1,
                index,
                height,
                width,
                1023.0,
                np.random.default_rng(12),
            )

        self.assertTrue(np.array_equal(synthetic_a, synthetic_b))
        for output in (synthetic_a, blackframe):
            self.assertEqual(output.shape, (height, width))
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)
        self.assertEqual(synthetic_meta["method"], "synthetic")
        self.assertEqual(blackframe_meta["method"], "blackframe")
        self.assertEqual(blackframe_meta["gain"], 4)

    def test_scan_rejects_wrong_sized_raw_and_separates_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_raw(root / "train_day" / "good.raw", np.zeros((4, 6), np.uint16))
            write_raw(root / "val_day" / "good.raw", np.zeros((4, 6), np.uint16))
            (root / "val_day" / "bad.raw").write_bytes(b"bad")
            valid, rejected = discover_clean_raws(
                root,
                {"train": ["train_day"], "val": ["val_day"], "test": []},
                4 * 6 * 2,
            )

        self.assertEqual(len(valid["train"]), 1)
        self.assertEqual(len(valid["val"]), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "unexpected_size")

    def test_dnr_uses_model_output_directly(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.full_like(inputs, 0.25)

        raw = np.full((8, 10), 0.8, dtype=np.float32)
        output = run_raw2dnr(ConstantModel(), raw, torch.device("cpu"), pad_base=4)
        self.assertTrue(np.allclose(output, 0.25))

    def test_downsample_and_seed_contract(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        self.assertEqual(downsample_x2(image, "bicubic").shape, (4, 5, 3))
        self.assertEqual(stable_seed(1, "scene", "synthetic"), stable_seed(1, "scene", "synthetic"))
        self.assertNotEqual(stable_seed(1, "scene", "synthetic"), stable_seed(1, "scene", "blackframe"))


if __name__ == "__main__":
    unittest.main()
