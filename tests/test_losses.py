import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from losses import GANLoss, PerceptualLoss, USMSharp


class FakeFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, image):
        return OrderedDict(
            conv1_2=image * self.scale,
            conv2_2=image.mean(dim=1, keepdim=True),
        )


class PerceptualLossTest(unittest.TestCase):
    def test_identical_inputs_have_zero_content_loss(self):
        criterion = PerceptualLoss(
            {"conv1_2": 0.1, "conv2_2": 1.0},
            feature_extractor=FakeFeatureExtractor(),
        )
        image = torch.rand(1, 3, 8, 8)
        content, style = criterion(image, image)
        self.assertEqual(content.item(), 0.0)
        self.assertEqual(style.item(), 0.0)

    def test_gradient_only_flows_to_prediction(self):
        extractor = FakeFeatureExtractor()
        criterion = PerceptualLoss(
            {"conv1_2": 1.0, "conv2_2": 1.0},
            feature_extractor=extractor,
        )
        prediction = torch.rand(1, 3, 8, 8, requires_grad=True)
        target = torch.rand(1, 3, 8, 8, requires_grad=True)
        content, _ = criterion(prediction, target)
        content.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(extractor.scale.grad)

    def test_gan_loss_prefers_matching_labels(self):
        criterion = GANLoss()
        real_logits = torch.full((1, 1, 4, 4), 5.0)
        fake_logits = torch.full((1, 1, 4, 4), -5.0)
        self.assertLess(
            criterion(real_logits, True).item(),
            criterion(fake_logits, True).item(),
        )

    def test_usm_preserves_shape_and_range(self):
        sharpener = USMSharp(radius=5)
        image = torch.rand(2, 3, 16, 16)
        output = sharpener(image)
        self.assertEqual(output.shape, image.shape)
        self.assertGreaterEqual(output.min().item(), 0.0)
        self.assertLessEqual(output.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
