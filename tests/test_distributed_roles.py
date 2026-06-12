from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autolabel_worker.config.schema import load_config as load_autolabel
from common.config.role_schema import role_to_kernel_config
from control_plane.config.schema import load_config as load_control_plane
from control_plane.store import ControlPlaneStore
from edge_agent.config.schema import load_config as load_edge
from remote_worker.config.schema import load_config as load_remote
from stats_service.config.schema import load_config as load_stats
from train_worker.config.schema import load_config as load_train


class DistributedRoleConfigTests(unittest.TestCase):
    def test_role_example_configs_load(self) -> None:
        cases = [
            ("train", load_train, Path("train_worker/config/config.example.toml")),
            ("autolabel", load_autolabel, Path("autolabel_worker/config/config.example.toml")),
            ("edge", load_edge, Path("edge_agent/config/config.example.toml")),
            ("remote", load_remote, Path("remote_worker/config/config.example.toml")),
            ("stats", load_stats, Path("stats_service/config/config.example.toml")),
            ("control", load_control_plane, Path("control_plane/config/config.example.toml")),
        ]
        for name, loader, path in cases:
            with self.subTest(name=name):
                cfg = loader(path.resolve())
                self.assertIn("workspace", cfg)
                self.assertTrue(str(cfg["workspace"]["root"]).endswith("work-dir"))

    def test_role_config_adapts_to_kernel_shape(self) -> None:
        cfg = load_train(Path("train_worker/config/config.example.toml").resolve())
        kernel_cfg = role_to_kernel_config(cfg, "train", "train_worker")
        self.assertIn("train", kernel_cfg)
        self.assertIn("class_map", kernel_cfg)
        self.assertNotIn("services", kernel_cfg)

    def test_control_plane_store_upserts_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ControlPlaneStore(Path(tmp) / "control.db")
            node = store.upsert_node(
                {
                    "node_id": "edge-001",
                    "role": "edge",
                    "status": "online",
                    "endpoint": "http://127.0.0.1:7810",
                }
            )
            self.assertEqual(node["node_id"], "edge-001")
            self.assertEqual(store.list_nodes()[0]["role"], "edge")

    def test_control_plane_store_tracks_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ControlPlaneStore(Path(tmp) / "control.db")
            job = store.create_job(kind="train", target_role="train_worker", payload={"epochs": 1})
            self.assertEqual(job["status"], "queued")
            attached = store.attach_upstream(
                job["job_id"],
                target_node_id="train-worker-001",
                upstream_endpoint="http://127.0.0.1:7811",
                upstream_job_id="upstream-1",
                status="running",
            )
            self.assertIsNotNone(attached)
            self.assertEqual(attached["upstream_job_id"], "upstream-1")
            self.assertEqual(store.list_jobs()[0]["target_role"], "train_worker")

    def test_remote_config_exposes_capabilities(self) -> None:
        cfg = load_remote(Path("remote_worker/config/config.example.toml").resolve())
        self.assertEqual(cfg["capabilities"]["protocol"], "frame_http")
        self.assertEqual(cfg["capabilities"]["model_path"], cfg["runtime"]["model"])


if __name__ == "__main__":
    unittest.main()
