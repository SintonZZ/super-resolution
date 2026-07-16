import tempfile
import unittest
from pathlib import Path

import torch

from archs.span import Conv3XC, SPAN
from util import forward_tiled, load_pretrained_span_backbone


class SPANTest(unittest.TestCase):
    def test_x2_output_shape(self):
        model = SPAN(feature_channels=4, upscale=2)
        model.train()
        output = model(torch.rand(1, 3, 8, 9))
        self.assertEqual(tuple(output.shape), (1, 3, 16, 18))

    def test_conv3xc_reparameterization(self):
        torch.manual_seed(7)
        module = Conv3XC(3, 5, gain1=2)
        inputs = torch.rand(2, 3, 9, 11)
        module.train()
        train_output = module(inputs)
        module.eval()
        eval_output = module(inputs)
        self.assertTrue(torch.allclose(train_output, eval_output, atol=2e-6, rtol=1e-5))

    def test_load_x4_backbone_skips_only_head(self):
        torch.manual_seed(11)
        model_x4 = SPAN(feature_channels=4, upscale=4)
        model_x2 = SPAN(feature_channels=4, upscale=2)
        original_head = model_x2.upsampler[0].weight.detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "span_x4.pth"
            torch.save({"params_ema": model_x4.state_dict()}, checkpoint_path)
            report = load_pretrained_span_backbone(
                model_x2,
                checkpoint_path,
                strict_backbone=True,
            )

        self.assertEqual(report["checkpoint_scale"], 4)
        mismatch_keys = {item["key"] for item in report["shape_mismatches"]}
        self.assertEqual(mismatch_keys, {"upsampler.0.weight", "upsampler.0.bias"})
        self.assertTrue(torch.equal(model_x2.conv_1.sk.weight, model_x4.conv_1.sk.weight))
        self.assertTrue(torch.equal(model_x2.upsampler[0].weight, original_head))

    def test_tiled_inference_matches_full_image(self):
        torch.manual_seed(13)
        model = SPAN(feature_channels=4, upscale=2).eval()
        inputs = torch.rand(1, 3, 32, 35)
        with torch.no_grad():
            full = model(inputs)
            tiled = forward_tiled(model, inputs, scale=2, tile_size=16, tile_pad=24)
        self.assertTrue(torch.allclose(full, tiled, atol=2e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
