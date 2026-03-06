from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from share.types.detection import Detection


class _FakeResult:
    def plot(self):
        return "annotated"


class _FakeInferencer:
    def infer_frame(self, frame_bgr):
        det = Detection(
            schema_version=1,
            class_id=0,
            class_name="person",
            score=0.9,
            bbox_xyxy=[1.0, 2.0, 30.0, 40.0],
        )
        return [det], _FakeResult(), 12.5


class RemoteProtocolTests(unittest.TestCase):
    def test_remote_response_contains_metadata_and_detections(self) -> None:
        from share.kernel.deploy.remote_server import create_remote_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "run-001"
            run_dir.mkdir(parents=True)
            model_dir = root / "models" / "exp001"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "model.onnx"
            model_path.write_text("fake", encoding="utf-8")
            (model_dir / "model_manifest.json").write_text(
                '{"schema_version":1,"model_id":"exp001-yolo","backend":"yolo"}',
                encoding="utf-8",
            )

            cfg = {
                "workspace": {"root": str(root)},
                "train": {"img_size": 640, "device": "cpu"},
                "deploy": {
                    "edge": {"jpeg_quality": 80},
                    "remote": {
                        "stats_endpoint": "",
                        "api_key": "",
                        "stats_timeout_sec": 2.0,
                        "ingest_api_key": "",
                        "max_payload_mb": 8,
                        "save_annotated": False,
                        "model": str(model_path),
                    },
                },
            }

            class _Logger:
                def info(self, *args, **kwargs):
                    return None

                def warn(self, *args, **kwargs):
                    return None

            with patch("share.kernel.deploy.remote_server.decode_jpeg_base64", return_value="frame"), patch(
                "share.kernel.deploy.remote_server.encode_jpeg_base64", return_value="encoded"
            ):
                app = create_remote_app(
                    cfg=cfg,
                    run_id="remote-run-001",
                    run_dir=run_dir,
                    logger=_Logger(),
                    inferencer=_FakeInferencer(),
                )
                endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/v1/frame")
                payload = endpoint(
                    {
                        "schema_version": 1,
                        "request_id": "req-123",
                        "run_id": "edge-run-001",
                        "source_id": "edge-001",
                        "frame_index": 3,
                        "frame_name": "frame-3.jpg",
                        "image_jpeg_base64": "ZmFrZQ==",
                        "return_annotated": True,
                    },
                    x_api_key=None,
                )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["request_id"], "req-123")
            self.assertEqual(payload["model_id"], "exp001-yolo")
            self.assertEqual(payload["backend"], "yolo")
            self.assertEqual(payload["metadata"]["edge_run_id"], "edge-run-001")
            self.assertEqual(payload["metadata"]["remote_run_id"], "remote-run-001")
            self.assertEqual(len(payload["detections"]), 1)
            self.assertEqual(payload["annotated_jpeg_base64"], "encoded")
