from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import autolabel.cli
import train.cli


class CliJsonSummaryTests(unittest.TestCase):
    def test_train_cli_config_only_emits_json_summary(self) -> None:
        stdout = io.StringIO()
        with patch(
            "train.cli.load_service_connection",
            return_value=("http://service", ""),
        ), patch("train.cli.patch_config", return_value={"ok": True}) as patch_mock, redirect_stdout(stdout):
            rc = train.cli.main(
                [
                    "--config",
                    "./work-dir/config.toml",
                    "--save-config",
                    "--config-only",
                    "--json-summary",
                    "--run-name",
                    "gui-smoke",
                ]
            )

        self.assertEqual(rc, 0)
        patch_mock.assert_called_once()
        payload = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["config_only"])

    def test_train_cli_run_emits_json_summary(self) -> None:
        stdout = io.StringIO()
        job = {
            "job_id": "job-001",
            "status": "succeeded",
            "error": "",
            "result": {
                "status": "ok",
                "error": None,
                "run_id": "exp-001",
                "elapsed_ms": 12.0,
                "run_dir": "/tmp/run",
                "resolved_config": "/tmp/run/config.resolved.toml",
                "artifacts_path": "/tmp/run/artifacts.json",
                "artifacts": {"ok": True},
                "log_path": "/tmp/log.txt",
            },
        }
        with patch(
            "train.cli.load_service_connection",
            return_value=("http://service", ""),
        ), patch("train.cli.submit_job", return_value={"job_id": "job-001"}) as submit_mock, patch(
            "train.cli.wait_for_job", return_value=job
        ), redirect_stdout(stdout):
            rc = train.cli.main(
                [
                    "--config",
                    "./work-dir/config.toml",
                    "--json-summary",
                    "--run-name",
                    "gui-smoke",
                ]
            )

        self.assertEqual(rc, 0)
        submit_mock.assert_called_once()
        payload = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual(payload["run_id"], "exp-001")
        self.assertEqual(payload["status"], "ok")

    def test_autolabel_cli_failed_run_emits_json_summary_and_error(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        job = {
            "job_id": "job-002",
            "status": "failed",
            "error": "boom",
            "result": {
                "status": "failed",
                "error": "boom",
                "run_id": "auto-001",
                "elapsed_ms": 44.0,
                "run_dir": "/tmp/run",
                "resolved_config": "/tmp/run/config.resolved.toml",
                "artifacts_path": "/tmp/run/artifacts.json",
                "artifacts": {},
                "log_path": "/tmp/log.txt",
            },
        }
        with patch(
            "autolabel.cli.load_service_connection",
            return_value=("http://service", ""),
        ), patch("autolabel.cli.submit_job", return_value={"job_id": "job-002"}), patch(
            "autolabel.cli.wait_for_job", return_value=job
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            rc = autolabel.cli.main(
                [
                    "--config",
                    "./work-dir/config.toml",
                    "--json-summary",
                    "--mode",
                    "model",
                ]
            )

        self.assertEqual(rc, 1)
        payload = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "boom")
        self.assertIn("error=boom", stderr.getvalue())
