import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import PairedSRDataset


def save_random_image(path, height, width, seed):
    rng = np.random.RandomState(seed)
    array = rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class DatasetTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
