from __future__ import annotations

import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from common.application.api_common import get_json, post_json
from common.config.schema import deep_merge_dict
from edge_agent.config.schema import DEFAULT_CONFIG as EDGE_DEFAULT


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _UvicornThread:
    def __init__(self, app: Any, port: int) -> None:
        self.app = app
        self.port = port
        self.thread: threading.Thread | None = None
        self.server: Any = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "_UvicornThread":
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                get_json(self.base_url, "/health", timeout_sec=0.5)
                return self
            except Exception:
                time.sleep(0.05)
        raise RuntimeError(f"server did not start on port {self.port}")

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)


class _ImmediateRunner:
    def __init__(self, job_store: Any, log_dir: Path | str, worker_module: str) -> None:
        self.job_store = job_store
        self.log_dir = Path(log_dir)
        self.worker_module = worker_module

    def start(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = str(job["job_id"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{job_id}.log"
        log_path.write_text(f"smoke job {job['kind']} via {self.worker_module}\n", encoding="utf-8")
        self.job_store.mark_running(job_id=job_id, worker_pid=0, log_path=log_path)
        self.job_store.finish_job(
            job_id=job_id,
            status="succeeded",
            result={"status": "ok", "smoke": True},
            log_path=log_path,
        )
        updated = self.job_store.get_job(job_id)
        if updated is None:
            raise RuntimeError("smoke runner failed to update job")
        return updated

    def cancel(self, job_id: str) -> bool:
        return self.job_store.request_cancel(job_id)


def _role_cfg(
    default: dict[str, Any], root: Path, db_name: str, port: int, cp_url: str
) -> dict[str, Any]:
    return deep_merge_dict(
        default,
        {
            "workspace": {
                "root": str(root),
                "run_name": "e2e-smoke",
                "log_file": "log.txt",
                "log_level": "INFO",
            },
            "server": {
                "host": "127.0.0.1",
                "port": port,
                "api_token": "",
                "api_token_env_name": "",
                "advertise_url": f"http://127.0.0.1:{port}",
            },
            "job_store": {"db_path": str(root / "state" / db_name)},
            "control_plane": {
                "url": cp_url,
                "api_token": "",
                "api_token_env_name": "",
                "heartbeat_interval_sec": 0,
            },
        },
    )


class ControlPlaneHttpE2ETests(unittest.TestCase):
    def test_control_plane_dispatch_refreshes_and_proxies_edge_logs_over_http(self) -> None:
        from control_plane.api import create_app as create_control_app
        from edge_agent.service import create_app as create_edge_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp_port = _free_port()
            edge_port = _free_port()
            cp_url = f"http://127.0.0.1:{cp_port}"
            cp_cfg = {
                "workspace": {
                    "root": str(root),
                    "run_name": "e2e-smoke",
                    "log_file": "log.txt",
                    "log_level": "INFO",
                },
                "server": {
                    "host": "127.0.0.1",
                    "port": cp_port,
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
            edge_cfg = _role_cfg(EDGE_DEFAULT, root, "edge.db", edge_port, cp_url)
            edge_path = root / "edge.toml"
            edge_path.write_text("", encoding="utf-8")

            with patch("edge_agent.service.SubprocessJobRunner", _ImmediateRunner):
                control_app = create_control_app(cp_cfg)
                edge_app = create_edge_app(edge_cfg, edge_path)

            with (
                _UvicornThread(control_app, cp_port),
                _UvicornThread(edge_app, edge_port),
            ):
                response = post_json(
                    f"http://127.0.0.1:{edge_port}", "/api/v1/nodes/register", payload={}
                )
                self.assertTrue(response["ok"])

                nodes = get_json(cp_url, "/api/v1/nodes")["nodes"]
                self.assertEqual({node["role"] for node in nodes}, {"edge"})

                created = post_json(
                    cp_url, "/api/v1/jobs", payload={"kind": "edge_run", "payload": {}}
                )
                self.assertTrue(created["ok"])
                job_id = created["job"]["job_id"]
                refreshed = get_json(cp_url, f"/api/v1/jobs/{job_id}")["job"]
                self.assertEqual(refreshed["status"], "succeeded")
                logs = get_json(cp_url, f"/api/v1/jobs/{job_id}/logs")["text"]
                self.assertIn("smoke job edge_run", logs)


if __name__ == "__main__":
    unittest.main()
