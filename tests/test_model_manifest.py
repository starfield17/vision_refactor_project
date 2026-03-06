from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from share.kernel.model_manifest import (
    build_train_model_manifest,
    resolve_model_identity,
    write_model_manifest,
)


class ModelManifestTests(unittest.TestCase):
    def test_build_and_resolve_model_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "exp001"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "model-int8.onnx"
            model_path.write_text("fake", encoding="utf-8")

            cfg = {
                "workspace": {"run_name": "exp001"},
                "train": {"backend": "yolo", "img_size": 640},
                "export": {"onnx": True},
                "class_map": {
                    "names": ["person", "car"],
                    "id_map": {"person": 0, "car": 1},
                },
            }
            run_ctx = {"run_id": "exp001-20260306-120000"}
            artifacts = {
                "backend": "yolo",
                "weights": {
                    "model_pt": str(model_dir / "model.pt"),
                    "labels_snapshot": str(model_dir / "labels.json"),
                },
                "export": {
                    "onnx_path": str(model_dir / "model.onnx"),
                    "quantized_onnx_path": str(model_path),
                    "fp16_onnx_path": "",
                    "final_infer_model_path": str(model_path),
                    "export_ok": True,
                    "quantize_ok": True,
                    "quantize_strategy": "onnxruntime-dynamic",
                    "messages": ["ok"],
                },
            }

            manifest = build_train_model_manifest(cfg, run_ctx, artifacts)
            path = write_model_manifest(model_dir, manifest)
            loaded = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(loaded["model_id"], "exp001-yolo")
            self.assertTrue(loaded["deployment_ready"])
            self.assertIn("deploy-remote", loaded["supported_modes"])

            identity = resolve_model_identity(
                model_path,
                default_backend="yolo",
                default_model_id="fallback-model",
            )
            self.assertEqual(identity["backend"], "yolo")
            self.assertEqual(identity["model_id"], "exp001-yolo")
            self.assertTrue(identity["manifest_path"].endswith("model_manifest.json"))
