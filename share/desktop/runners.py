"""Qt API runner for backend service jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from share.application.api_common import get_json, post_json
from share.application.service_client import load_service_connection, wait_for_job


class _JobThread(QThread):
    output = Signal(str)
    done = Signal(object)
    failed = Signal(object)

    def __init__(
        self,
        kind: str,
        config_path: str,
        workdir_override: str | None,
        overrides: list[str],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self.config_path = config_path
        self.workdir_override = workdir_override
        self.overrides = overrides
        self.cancel_requested = False
        self.api_url = ""
        self.token = ""
        self.job_id = ""

    def run(self) -> None:
        try:
            self.api_url, self.token = load_service_connection(
                config_path=Path(self.config_path).resolve(),
                service_name="train_autolabel",
                workdir_override=self.workdir_override,
            )
            path = "/api/v1/train/jobs" if self.kind == "train" else "/api/v1/autolabel/jobs"
            response = post_json(
                base_url=self.api_url,
                path=path,
                payload={"overrides": self.overrides},
                token=self.token,
            )
            job = response["job"]
            self.job_id = str(job["job_id"])
            self.output.emit(f"submitted job_id={self.job_id}\n")
            final_job = wait_for_job(
                api_url=self.api_url,
                token=self.token,
                job_id=self.job_id,
                poll_sec=1.0,
            )
            summary = self._job_to_summary(final_job)
            if summary.get("status") == "ok" and final_job.get("status") == "succeeded":
                self.done.emit(summary)
            else:
                self.failed.emit(summary)
        except Exception as exc:
            self.failed.emit(
                {
                    "status": "failed",
                    "error": str(exc),
                    "run_id": "",
                    "elapsed_ms": 0.0,
                    "run_dir": "",
                    "resolved_config": None,
                    "artifacts_path": None,
                    "artifacts": {},
                    "log_path": None,
                }
            )

    def cancel(self) -> None:
        self.cancel_requested = True
        if self.api_url and self.job_id:
            try:
                post_json(
                    base_url=self.api_url,
                    path=f"/api/v1/jobs/{self.job_id}/cancel",
                    payload={},
                    token=self.token,
                )
            except Exception:
                return

    @staticmethod
    def _job_to_summary(job: dict[str, Any]) -> dict[str, Any]:
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        summary = dict(result)
        status = "ok" if job.get("status") == "succeeded" else str(job.get("status"))
        summary.setdefault("status", status)
        summary.setdefault("error", job.get("error") or None)
        summary.setdefault("run_id", job.get("run_id", ""))
        summary.setdefault("elapsed_ms", 0.0)
        summary.setdefault("run_dir", job.get("run_dir", ""))
        summary.setdefault("resolved_config", None)
        summary.setdefault("artifacts_path", None)
        summary.setdefault("artifacts", {})
        summary.setdefault("log_path", job.get("log_path") or None)
        summary["job_id"] = job.get("job_id", "")
        return summary


class CliProcessRunner(QObject):
    started = Signal()
    output = Signal(str)
    error_output = Signal(str)
    finished_ok = Signal(object)
    finished_failed = Signal(object)
    state_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: _JobThread | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start_train(
        self,
        config_path: str,
        workdir_override: str | None = None,
        overrides: list[str] | None = None,
    ) -> None:
        self._start_job("train", config_path, workdir_override, overrides or [])

    def start_autolabel(
        self,
        config_path: str,
        workdir_override: str | None = None,
        overrides: list[str] | None = None,
    ) -> None:
        self._start_job("autolabel", config_path, workdir_override, overrides or [])

    def stop(self) -> None:
        if not self._thread:
            return
        self.state_changed.emit("stopping")
        self._thread.cancel()

    def _start_job(
        self,
        kind: str,
        config_path: str,
        workdir_override: str | None,
        overrides: list[str],
    ) -> None:
        if self.is_running():
            raise RuntimeError("A job is already running.")
        thread = _JobThread(kind, config_path, workdir_override, overrides, self)
        thread.output.connect(self.output.emit)
        thread.done.connect(self._handle_done)
        thread.failed.connect(self._handle_failed)
        thread.finished.connect(self._handle_thread_finished)
        self._thread = thread
        self.state_changed.emit("starting")
        self.started.emit()
        self.state_changed.emit("running")
        thread.start()

    def _handle_done(self, summary: object) -> None:
        self.finished_ok.emit(summary)

    def _handle_failed(self, summary: object) -> None:
        self.error_output.emit(json.dumps(summary, ensure_ascii=True) + "\n")
        self.finished_failed.emit(summary)

    def _handle_thread_finished(self) -> None:
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        self.state_changed.emit("idle")
