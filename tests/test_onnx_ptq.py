import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image

from quantization.onnx_ptq import (
    SRCalibrationDataReader,
    default_output_path,
    discover_samples,
    resolve_calibration_size,
    validate_deploy_graph,
)


def make_model_with_node(node):
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 9])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 9])
    graph = helper.make_graph([node], "test", [graph_input], [graph_output])
    return helper.make_model(graph)


class ONNXPTQTest(unittest.TestCase):
    def test_image_reader_returns_rgb_nchw_float(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.random.RandomState(3).randint(0, 256, (12, 14, 3), dtype=np.uint8)
            image_path = root / "sample.png"
            Image.fromarray(array).save(image_path)
            input_info = helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                [1, 3, 8, 9],
            )
            reader = SRCalibrationDataReader(
                input_info,
                [image_path],
                "images",
                height=8,
                width=9,
            )
            sample = reader.get_next()["input"]
            self.assertIsNone(reader.get_next())
            reader.rewind()
            self.assertIsNotNone(reader.get_next())

        self.assertEqual(sample.shape, (1, 3, 8, 9))
        self.assertEqual(sample.dtype, np.float32)
        self.assertGreaterEqual(float(sample.min()), 0.0)
        self.assertLessEqual(float(sample.max()), 1.0)

    def test_discover_samples_prefers_images_and_honors_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                Image.new("RGB", (8, 8), color=(index, index, index)).save(
                    root / f"{index}.png"
                )
            np.save(root / "sample.npy", np.zeros((1, 3, 8, 8), dtype=np.float32))
            sample_format, samples = discover_samples(root, "auto", max_samples=2)

        self.assertEqual(sample_format, "images")
        self.assertEqual(len(samples), 2)

    def test_dynamic_input_uses_requested_or_default_size(self):
        input_info = helper.make_tensor_value_info(
            "input",
            TensorProto.FLOAT,
            ["batch", 3, "height", "width"],
        )
        self.assertEqual(resolve_calibration_size(input_info, None, None), (256, 256))
        self.assertEqual(resolve_calibration_size(input_info, 128, 192), (128, 192))

    def test_rejects_grouped_convolution_and_silu(self):
        grouped_conv = helper.make_node(
            "Conv",
            ["input", "weight"],
            ["output"],
            group=3,
        )
        with self.assertRaisesRegex(ValueError, "grouped convolutions"):
            validate_deploy_graph(make_model_with_node(grouped_conv))

        silu = helper.make_node("SiLU", ["input"], ["output"])
        with self.assertRaisesRegex(ValueError, "SiLU/Swish"):
            validate_deploy_graph(make_model_with_node(silu))

    def test_default_output_path(self):
        self.assertEqual(
            default_output_path(Path("models/spanf.onnx")),
            Path("models/spanf_ptq.onnx"),
        )


if __name__ == "__main__":
    unittest.main()
