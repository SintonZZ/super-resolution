import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import onnx
except ImportError:  # pragma: no cover - ONNX is part of the deployment environment.
    onnx = None

from archs.spanf import Conv3XC, SPANF
from util import forward_tiled


class SPANFTest(unittest.TestCase):
    def test_x2_output_shape(self):
        model = SPANF(feature_channels=4, upscale=2)
        model.train()
        output = model(torch.rand(1, 3, 8, 9))
        self.assertEqual(tuple(output.shape), (1, 3, 16, 18))

    def test_x2_topology_matches_spanf_design(self):
        model = SPANF(feature_channels=32, upscale=2)
        self.assertEqual(model.conv_near.in_channels, 3)
        self.assertEqual(model.conv_near.out_channels, 12)
        self.assertEqual(model.conv_near.groups, 3)
        self.assertEqual(model.block_1.c1_r.eval_conv.in_channels, 3)
        self.assertEqual(model.block_1.c1_r.eval_conv.out_channels, 32)
        self.assertEqual(model.conv_cat.in_channels, 76)
        self.assertEqual(model.conv_2.eval_conv.out_channels, 12)
        self.assertFalse(hasattr(model, "block_6"))

    def test_nearest_shortcut_initialization(self):
        model = SPANF(feature_channels=4, upscale=2, nearest_init=True)
        inputs = torch.rand(1, 3, 7, 9)
        encoded = model.conv_near(inputs)
        decoded = model.depth_to_space(encoded)
        expected = F.interpolate(inputs, scale_factor=2, mode="nearest")
        self.assertTrue(torch.equal(decoded, expected))

    def test_conv3xc_reparameterization(self):
        torch.manual_seed(7)
        module = Conv3XC(3, 5, gain1=2)
        inputs = torch.rand(2, 3, 9, 11)
        module.train()
        train_output = module(inputs)
        module.eval()
        eval_output = module(inputs)
        self.assertTrue(torch.allclose(train_output, eval_output, atol=2e-6, rtol=1e-5))

    def test_full_model_deploy_matches_training_branches(self):
        torch.manual_seed(11)
        model = SPANF(feature_channels=4, upscale=2)
        inputs = torch.rand(1, 3, 13, 15)
        model.train()
        with torch.no_grad():
            training_output = model(inputs)
        model.switch_to_deploy()
        with torch.no_grad():
            deploy_output = model(inputs)
        self.assertTrue(torch.allclose(training_output, deploy_output, atol=2e-5, rtol=1e-5))

    def test_deploy_converts_grouped_shortcut_to_dense(self):
        torch.manual_seed(12)
        model = SPANF(feature_channels=4, upscale=2, nearest_init=False).eval()
        inputs = torch.rand(2, 3, 11, 13)
        with torch.no_grad():
            grouped_output = model.conv_near(inputs)

        model.switch_to_deploy()
        with torch.no_grad():
            dense_output = model.conv_near(inputs)

        self.assertEqual(model.conv_near.groups, 1)
        self.assertEqual(tuple(model.conv_near.weight.shape), (12, 3, 3, 3))
        self.assertTrue(torch.equal(grouped_output, dense_output))

    @unittest.skipIf(onnx is None, "onnx is not installed")
    def test_deploy_onnx_contains_no_grouped_convolution(self):
        model = SPANF(feature_channels=4, upscale=2).switch_to_deploy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spanf_x2.onnx"
            torch.onnx.export(
                model,
                torch.rand(1, 3, 8, 9),
                str(path),
                opset_version=11,
                input_names=["input"],
                output_names=["output"],
            )
            graph = onnx.load(str(path)).graph

        grouped_convolutions = []
        for node in graph.node:
            if node.op_type != "Conv":
                continue
            group = next(
                (attribute.i for attribute in node.attribute if attribute.name == "group"),
                1,
            )
            if group != 1:
                grouped_convolutions.append((node.name, group))
        self.assertEqual(grouped_convolutions, [])

    def test_tiled_inference_matches_full_image(self):
        torch.manual_seed(13)
        model = SPANF(feature_channels=4, upscale=2).eval()
        inputs = torch.rand(1, 3, 32, 35)
        with torch.no_grad():
            full = model(inputs)
            tiled = forward_tiled(model, inputs, scale=2, tile_size=16, tile_pad=24)
        self.assertTrue(torch.allclose(full, tiled, atol=2e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
