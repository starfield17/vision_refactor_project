from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from share.application.autolabel_service import build_autolabel_overrides_from_payload
from share.application.common import normalize_run_result
from share.application.train_service import build_train_overrides_from_payload
from share.kernel.kernel import RunContext, RunResult


class ApplicationServiceTests(unittest.TestCase):
    def test_build_train_overrides_from_payload(self) -> None:
        overrides = build_train_overrides_from_payload(
            {
                "run_name": "exp42",
                "dataset_dir": "/tmp/dataset",
                "backend": "faster_rcnn",
                "device": "cpu",
                "seed": 7,
                "epochs": 5,
                "batch_size": 2,
                "img_size": 512,
                "dry_run": False,
                "yolo_weights": "/tmp/model.pt",
                "frcnn_variant": "mobilenet_v3",
                "frcnn_lr": 0.01,
                "frcnn_momentum": 0.8,
                "frcnn_weight_decay": 0.001,
                "frcnn_num_workers": 3,
                "frcnn_max_samples": 25,
                "export_onnx": True,
                "export_opset": 17,
                "export_quantize": False,
                "export_quantize_mode": "dynamic",
                "export_calib_samples": 9,
            }
        )

        self.assertIn("workspace.run_name=exp42", overrides)
        self.assertIn("data.yolo_dataset_dir=/tmp/dataset", overrides)
        self.assertIn("train.backend=faster_rcnn", overrides)
        self.assertIn("train.faster_rcnn.lr=0.01", overrides)
        self.assertIn("export.quantize=false", overrides)

    def test_build_autolabel_overrides_from_payload(self) -> None:
        overrides = build_autolabel_overrides_from_payload(
            {
                "run_name": "auto-exp",
                "device": "cuda:0",
                "labeled_dir": "/tmp/labeled",
                "unlabeled_dir": "/tmp/unlabeled",
                "mode": "llm",
                "confidence": 0.7,
                "batch_size": 4,
                "visualize": True,
                "on_conflict": "overwrite",
                "model_backend": "yolo",
                "model_onnx": "/tmp/model.onnx",
                "llm_base_url": "https://example.test/v1",
                "llm_model": "demo",
                "llm_api_key": "secret",
                "llm_api_key_env_name": "VISION_API_KEY",
                "llm_prompt": "return json",
                "llm_timeout_sec": 30.0,
                "llm_max_retries": 1,
                "llm_retry_backoff_sec": 2.0,
                "llm_qps_limit": 0.5,
                "llm_max_images": 10,
            }
        )

        self.assertIn("autolabel.mode=llm", overrides)
        self.assertIn("train.device=cuda:0", overrides)
        self.assertIn("autolabel.visualize=true", overrides)
        self.assertIn("autolabel.llm.base_url=https://example.test/v1", overrides)
        self.assertIn("autolabel.llm.max_images=10", overrides)

    def test_normalize_run_result_reads_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            run_dir = workdir / "runs" / "exp001-20260327-010101"
            run_dir.mkdir(parents=True)
            (run_dir / "artifacts.json").write_text(
                '{"run_id":"exp001-20260327-010101","status":"ok","artifacts":{"model":"a.onnx"}}',
                encoding="utf-8",
            )
            (run_dir / "config.resolved.toml").write_text("x = 1\n", encoding="utf-8")

            result = RunResult(
                run_context=RunContext(
                    run_id="exp001-20260327-010101",
                    mode="train",
                    workdir=workdir,
                    run_dir=run_dir,
                ),
                status="ok",
                backend="yolo",
                elapsed_ms=1234.0,
                artifacts={"model": "a.onnx"},
                error=None,
            )
            normalized = normalize_run_result(
                {
                    "workspace": {
                        "root": str(workdir),
                        "log_file": "log.txt",
                        "log_level": "INFO",
                    }
                },
                result,
            )

            self.assertEqual(normalized["status"], "ok")
            self.assertEqual(normalized["run_dir"], str(run_dir))
            self.assertTrue(str(normalized["resolved_config"]).endswith("config.resolved.toml"))
            self.assertEqual(normalized["artifacts"]["artifacts"]["model"], "a.onnx")
            self.assertTrue(str(normalized["log_path"]).endswith("log.txt"))
