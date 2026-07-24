import unittest

import torch

from archs import UNetDiscriminatorSN, build_discriminator


class UNetDiscriminatorTest(unittest.TestCase):
    def test_preserves_spatial_shape_and_outputs_logits(self):
        model = UNetDiscriminatorSN(num_in_ch=3, num_feat=4)
        inputs = torch.rand(2, 3, 32, 32)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 1, 32, 32))

    def test_factory_validates_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported discriminator"):
            build_discriminator({"type": "unknown"})


if __name__ == "__main__":
    unittest.main()
