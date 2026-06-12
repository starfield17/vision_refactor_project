from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from autolabel_worker.config.schema import DEFAULT_CONFIG as AUTOLABEL_DEFAULT
from common.config.schema import deep_merge_dict
from common.types.stats import StatsEvent
from edge_agent.config.schema import DEFAULT_CONFIG as EDGE_DEFAULT
from stats_service.config.schema import DEFAULT_CONFIG as STATS_DEFAULT
from train_worker.config.schema import DEFAULT_CONFIG as TRAIN_DEFAULT


def _role_cfg(default: dict, root: Path, db_name: str) -> dict:
    return deep_merge_dict(
        default,
        {
            "workspace": {
                "root": str(root),
                "run_name": "svc-test",
                "log_file": "log.txt",
                "log_level": "INFO",
            },
            "server": {"host": "127.0.0.1", "api_token": ""},
            "job_store": {"db_path": str(root / "state" / db_name)},
            "control_plane": {"url": ""},
        },
    )


class DistributedServiceApiTests(unittest.TestCase):
    def test_control_plane_dispatches_job_to_registered_worker(self) -> None:
        from control_plane.api import create_app
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {
                "workspace": {
                    "root": str(root),
                    "run_name": "control-test",
                    "log_file": "log.txt",
                    "log_level": "INFO",
                },
                "server": {
                    "host": "127.0.0.1",
                    "port": 7800,
                    "api_token": "",
                    "api_token_env_name": "",
                },
                "storage": {
                    "db_path": str(root / "state" / "control.db"),
                    "artifact_root": str(root / "artifacts"),
                    "model_registry": str(root / "models" / "registry"),
                },
                "nodes": {"offline_ttl_sec": 45},
            }
            client = TestClient(create_app(cfg))
            heartbeat = client.post(
                "/api/v1/nodes/heartbeat",
                json={
                    "node_id": "train-worker-001",
                    "role": "train_worker",
                    "status": "online",
                    "endpoint": "http://worker",
                },
            )
            self.assertEqual(heartbeat.status_code, 200)

            with patch(
                "control_plane.api.post_json",
                return_value={
                    "ok": True,
                    "job": {"job_id": "worker-job-1", "kind": "train", "status": "running"},
                },
            ) as post_mock:
                response = client.post(
                    "/api/v1/jobs", json={"kind": "train", "payload": {"dry_run": True}}
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["job"]["upstream_job_id"], "worker-job-1")
            self.assertEqual(payload["job"]["target_node_id"], "train-worker-001")
            post_mock.assert_called_once()

    def test_control_plane_marks_stale_nodes_offline(self) -> None:
        from control_plane.api import create_app
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "control.db"
            cfg = {
                "workspace": {
                    "root": str(root),
                    "run_name": "control-test",
                    "log_file": "log.txt",
                    "log_level": "INFO",
                },
                "server": {
                    "host": "127.0.0.1",
                    "port": 7800,
                    "api_token": "",
                    "api_token_env_name": "",
                },
                "storage": {
                    "db_path": str(db_path),
                    "artifact_root": str(root / "artifacts"),
                    "model_registry": str(root / "models" / "registry"),
                },
                "nodes": {"offline_ttl_sec": 1},
            }
            client = TestClient(create_app(cfg))
            response = client.post(
                "/api/v1/nodes/heartbeat",
                json={
                    "node_id": "train-worker-001",
                    "role": "train_worker",
                    "status": "online",
                    "endpoint": "http://worker",
                },
            )
            self.assertEqual(response.status_code, 200)
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "UPDATE nodes SET last_seen_utc = ? WHERE node_id = ?",
                    ("2000-01-01T00:00:00+00:00", "train-worker-001"),
                )

            payload = client.get("/api/v1/nodes").json()
            self.assertEqual(payload["nodes"][0]["status"], "offline")

    def test_train_worker_submits_job(self) -> None:
        from fastapi.testclient import TestClient
        from train_worker.service import create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _role_cfg(TRAIN_DEFAULT, root, "train.db")
            cfg["data"]["yolo_dataset_dir"] = str(root / "datasets" / "yolo")
            config_path = root / "train.toml"
            config_path.write_text("", encoding="utf-8")
            with patch("train_worker.service.SubprocessJobRunner.start") as start_mock:
                start_mock.side_effect = lambda job: {**job, "status": "running"}
                client = TestClient(create_app(role_cfg=cfg, config_path=config_path))
                response = client.post("/api/v1/jobs", json={"dry_run": True})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["job"]["kind"], "train")
            self.assertEqual(payload["job"]["status"], "running")

    def test_autolabel_worker_submits_job(self) -> None:
        from autolabel_worker.service import create_app
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _role_cfg(AUTOLABEL_DEFAULT, root, "autolabel.db")
            cfg["data"]["labeled_dir"] = str(root / "datasets" / "labeled")
            cfg["data"]["unlabeled_dir"] = str(root / "datasets" / "unlabeled")
            cfg["runtime"]["model"]["onnx_model"] = str(root / "models" / "model.onnx")
            config_path = root / "autolabel.toml"
            config_path.write_text("", encoding="utf-8")
            with patch("autolabel_worker.service.SubprocessJobRunner.start") as start_mock:
                start_mock.side_effect = lambda job: {**job, "status": "running"}
                client = TestClient(create_app(role_cfg=cfg, config_path=config_path))
                response = client.post("/api/v1/jobs", json={"mode": "model"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["job"]["kind"], "autolabel")

    def test_edge_agent_submits_job(self) -> None:
        from edge_agent.service import create_app
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _role_cfg(EDGE_DEFAULT, root, "edge.db")
            cfg["runtime"]["images_dir"] = str(root / "images")
            cfg["runtime"]["local_model"] = str(root / "model.onnx")
            config_path = root / "edge.toml"
            config_path.write_text("", encoding="utf-8")
            with patch("edge_agent.service.SubprocessJobRunner.start") as start_mock:
                start_mock.side_effect = lambda job: {**job, "status": "running"}
                client = TestClient(create_app(role_cfg=cfg, config_path=config_path))
                response = client.post("/api/v1/jobs", json={"mode": "local", "max_frames": 1})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["job"]["kind"], "edge_run")

    def test_statistics_dashboard_reads_pushed_event(self) -> None:
        from fastapi.testclient import TestClient
        from stats_service.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _role_cfg(STATS_DEFAULT, root, "stats_jobs.db")
            cfg["runtime"]["db_path"] = str(root / "stats" / "stats.db")
            client = TestClient(create_app(role_cfg=cfg))
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
