from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from share.kernel.infer.faster_rcnn_onnx import LocalFasterRCNNOnnxInferencer


class _Input:
    name = "images"


class _FakeSession:
    def __init__(self, *_args, **_kwargs) -> None:
        self.last_feed = None

    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, feed):
        self.last_feed = feed
        return [
            np.array(
                [
                    [1.0, 2.0, 30.0, 40.0],
                    [0.0, 0.0, 20.0, 20.0],
                    [5.0, 5.0, 10.0, 10.0],
                ],
                dtype=np.float32,
            ),
            np.array([1, 2, 0], dtype=np.int64),
            np.array([0.9, 0.2, 0.99], dtype=np.float32),
        ]


class FasterRCNNOnnxInferTests(unittest.TestCase):
    def test_infer_frame_maps_torchvision_labels_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.onnx"
            model_path.write_text("fake", encoding="utf-8")
            frame = np.zeros((64, 64, 3), dtype=np.uint8)

            with patch(
                "onnxruntime.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ), patch("onnxruntime.InferenceSession", _FakeSession):
                inferencer = LocalFasterRCNNOnnxInferencer(
                    model_path=model_path,
                    class_names=["person", "car"],
                    confidence=0.5,
                    device="cuda:0",
                )
                detections, result, latency_ms = inferencer.infer_frame(frame)

            self.assertGreaterEqual(latency_ms, 0.0)
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0].class_id, 0)
            self.assertEqual(detections[0].class_name, "person")
            self.assertEqual(detections[0].bbox_xyxy, [1.0, 2.0, 30.0, 40.0])
            self.assertEqual(result.plot().shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
