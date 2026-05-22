from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from share.config.schema import DEFAULT_CONFIG, deep_merge_dict
from share.types.stats import StatsEvent


def _base_cfg(root: Path) -> dict:
    return deep_merge_dict(
        DEFAULT_CONFIG,
        {
            "workspace": {
                "root": str(root),
                "run_name": "svc-test",
                "log_file": "log.txt",
                "log_level": "INFO",
            },
            "class_map": {"names": ["person"], "id_map": {"person": 0}},
            "data": {
                "yolo_dataset_dir": str(root / "datasets" / "yolo"),
                "labeled_dir": str(root / "datasets" / "labeled"),
                "unlabeled_dir": str(root / "datasets" / "unlabeled"),
            },
            "train": {"backend": "yolo", "yolo": {"weights": ""}},
            "autolabel": {"mode": "llm", "llm": {"base_url": "http://x", "model": "m", "api_key": "k", "prompt": "p"}},
            "deploy": {
                "edge": {"local_model": str(root / "model.onnx"), "images_dir": str(root / "images")},
                "remote": {"model": str(root / "model.onnx")},
                "statistics": {"db_path": str(root / "stats" / "stats.db")},
            },
            "services": {
                "train_autolabel": {"job_db_path": str(root / "state" / "ta.db")},
                "deploy_statistics": {"job_db_path": str(root / "state" / "ds.db")},
            },
        },
    )


class ServiceApiTests(unittest.TestCase):
    def test_train_api_submits_job(self) -> None:
        from fastapi.testclient import TestClient
        from services.train_autolabel.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _base_cfg(root)
            config_path = root / "config.toml"
            config_path.write_text("", encoding="utf-8")
            with patch("services.train_autolabel.api.SubprocessJobRunner.start") as start_mock:
                def _start(job):
                    job["status"] = "running"
                    return job

                start_mock.side_effect = _start
                client = TestClient(create_app(cfg=cfg, config_path=config_path))
                response = client.post("/api/v1/train/jobs", json={"overrides": ["train.dry_run=true"]})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["job"]["kind"], "train")
            self.assertEqual(payload["job"]["status"], "running")

    def test_deploy_statistics_dashboard_reads_pushed_event(self) -> None:
        from fastapi.testclient import TestClient
        from services.deploy_statistics.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _base_cfg(root)
            config_path = root / "config.toml"
            config_path.write_text("", encoding="utf-8")
            client = TestClient(create_app(cfg=cfg, config_path=config_path))
            event = StatsEvent.now(
                source_id="edge-1",
                total_detections=2,
                counts_by_class={"person": 2},
                latency_ms=12.5,
            )
            push = client.post("/api/v1/push", json=event.to_dict())
            dashboard = client.get("/api/v1/statistics/dashboard")

            self.assertEqual(push.status_code, 200)
            self.assertEqual(dashboard.status_code, 200)
            payload = dashboard.json()["dashboard"]
            self.assertEqual(payload["overview"]["events_total"], 1)
            self.assertEqual(payload["filtered_summary"]["detections_total"], 2.0)
            self.assertEqual(payload["sources"], ["edge-1"])


if __name__ == "__main__":
    unittest.main()
