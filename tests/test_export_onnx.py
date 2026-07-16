import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from export_onnx import default_output_path, load_model


class ExportOnnxTest(unittest.TestCase):
    def test_default_output_paths(self):
        self.assertEqual(
            default_output_path("checkpoints/best_model.pth"),
            Path("checkpoints/best_model_deploy.onnx"),
        )
        self.assertEqual(default_output_path(), Path("spanf_x2_random_deploy.onnx"))

    def test_load_random_model_from_config_and_prepare_deploy_graph(self):
        config = {
            "model": {
                "type": "spanf",
                "in_channels": 3,
                "out_channels": 3,
                "feature_channels": 8,
                "upscale": 2,
                "bias": True,
                "nearest_init": True,
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train.json"
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file)

            model, model_config = load_model(None, config_path, torch.device("cpu"))

        self.assertEqual(model_config["feature_channels"], 8)
        self.assertFalse(model.training)
        self.assertTrue(all(module.groups == 1 for module in model.modules() if isinstance(module, nn.Conv2d)))
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 8, 9))
        self.assertEqual(tuple(output.shape), (1, 3, 16, 18))


if __name__ == "__main__":
    unittest.main()
