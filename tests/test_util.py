import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from util import save_sr_comparison, tensor_to_uint8_image


class SRComparisonTest(unittest.TestCase):
    def test_saves_bilinear_input_and_model_output_side_by_side(self):
        inputs = torch.tensor(
            [[
                [[0.0, 1.0], [0.25, 0.75]],
                [[1.0, 0.0], [0.75, 0.25]],
                [[0.5, 0.25], [1.0, 0.0]],
            ]]
        )
        outputs = torch.full((1, 3, 4, 4), 0.6)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            save_sr_comparison(inputs, outputs, path)
            canvas = np.asarray(Image.open(path).convert("RGB"))

        expected_bilinear = tensor_to_uint8_image(
            F.interpolate(
                inputs,
                size=(4, 4),
                mode="bilinear",
                align_corners=False,
            )
        )
        expected_output = tensor_to_uint8_image(outputs)

        self.assertEqual(canvas.shape, (36, 16, 3))
        np.testing.assert_array_equal(canvas[32:, :4], expected_bilinear)
        np.testing.assert_array_equal(canvas[32:, 12:], expected_output)


if __name__ == "__main__":
    unittest.main()
