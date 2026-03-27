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
        with patch("train.cli.save_train_config") as save_mock, redirect_stdout(stdout):
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
        save_mock.assert_called_once()
        payload = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["config_only"])

    def test_train_cli_run_emits_json_summary(self) -> None:
        stdout = io.StringIO()
        with patch(
            "train.cli.run_train",
            return_value={
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
        ) as run_mock, redirect_stdout(stdout):
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
        run_mock.assert_called_once()
        payload = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual(payload["run_id"], "exp-001")
        self.assertEqual(payload["status"], "ok")

    def test_autolabel_cli_failed_run_emits_json_summary_and_error(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "autolabel.cli.run_autolabel",
            return_value={
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
