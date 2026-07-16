import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from quantization.split_calibration_image import (
    compute_tile_boxes,
    split_calibration_image,
)


class SplitCalibrationImageTest(unittest.TestCase):
    def test_boxes_are_fixed_size_and_non_overlapping(self):
        boxes = compute_tile_boxes(7, 5, tile_width=3, tile_height=2)
        self.assertEqual(
            boxes,
            [
                (0, 0, 3, 2),
                (3, 0, 6, 2),
                (0, 2, 3, 4),
                (3, 2, 6, 4),
            ],
        )
        occupied = set()
        for left, top, right, bottom in boxes:
            pixels = {
                (x, y)
                for y in range(top, bottom)
                for x in range(left, right)
            }
            self.assertTrue(occupied.isdisjoint(pixels))
            occupied.update(pixels)

    def test_split_writes_exact_rgb_tiles_and_drops_border(self):
        array = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "large.png"
            output_dir = root / "tiles"
            Image.fromarray(array, mode="RGB").save(input_path)

            paths, image_size, discarded = split_calibration_image(
                input_path,
                output_dir,
                tile_width=3,
                tile_height=2,
            )
            second_row_second_column = np.asarray(Image.open(paths[3]))

        self.assertEqual(len(paths), 4)
        self.assertEqual(image_size, (7, 5))
        self.assertEqual(discarded, (1, 1))
        np.testing.assert_array_equal(second_row_second_column, array[2:4, 3:6])

    def test_max_tiles_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "large.png"
            output_dir = root / "tiles"
            Image.new("RGB", (8, 8), color=(1, 2, 3)).save(input_path)

            paths, _, _ = split_calibration_image(
                input_path,
                output_dir,
                tile_width=4,
                tile_height=4,
                max_tiles=2,
            )
            self.assertEqual(len(paths), 2)
            with self.assertRaises(FileExistsError):
                split_calibration_image(
                    input_path,
                    output_dir,
                    tile_width=4,
                    tile_height=4,
                    max_tiles=2,
                )

    def test_rejects_image_smaller_than_tile(self):
        with self.assertRaisesRegex(ValueError, "smaller than one"):
            compute_tile_boxes(127, 256, tile_width=128, tile_height=256)


if __name__ == "__main__":
    unittest.main()
