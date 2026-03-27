"""QProcess wrappers for running CLI modules from the desktop GUI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal


class CliProcessRunner(QObject):
    started = Signal()
    output = Signal(str)
    error_output = Signal(str)
    finished_ok = Signal(object)
    finished_failed = Signal(object)
    state_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._stdout_chunks: list[str] = []
        self._stderr_chunks: list[str] = []
        self._cancel_requested = False

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def start_train(
        self,
        config_path: str,
        workdir_override: str | None = None,
        overrides: list[str] | None = None,
    ) -> None:
        self._start_process(
            module="train.cli",
            config_path=config_path,
            workdir_override=workdir_override,
            overrides=overrides or [],
        )

    def start_autolabel(
        self,
        config_path: str,
        workdir_override: str | None = None,
        overrides: list[str] | None = None,
    ) -> None:
        self._start_process(
            module="autolabel.cli",
            config_path=config_path,
            workdir_override=workdir_override,
            overrides=overrides or [],
        )

    def stop(self) -> None:
        if not self._process or self._process.state() == QProcess.NotRunning:
            return
        self._cancel_requested = True
        self.state_changed.emit("stopping")
        self._process.terminate()
        QTimer.singleShot(3000, self._force_kill_if_running)

    def _force_kill_if_running(self) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()

    def _start_process(
        self,
        module: str,
        config_path: str,
        workdir_override: str | None,
        overrides: list[str],
    ) -> None:
        if self.is_running():
            raise RuntimeError("A process is already running.")

        self._stdout_chunks = []
        self._stderr_chunks = []
        self._cancel_requested = False

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(self._build_args(module, config_path, workdir_override, overrides))
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        process.readyReadStandardOutput.connect(self._handle_stdout)
        process.readyReadStandardError.connect(self._handle_stderr)
        process.finished.connect(self._handle_finished)
        process.errorOccurred.connect(self._handle_process_error)
        process.stateChanged.connect(self._handle_state_changed)
        self._process = process
        process.start()

    @staticmethod
    def _build_args(
        module: str,
        config_path: str,
        workdir_override: str | None,
        overrides: list[str],
    ) -> list[str]:
        args = [
            "-m",
            module,
            "--config",
            str(Path(config_path).resolve()),
            "--json-summary",
        ]
        if workdir_override:
            args.extend(["--workdir", str(Path(workdir_override).resolve())])
        for item in overrides:
            args.extend(["--set", item])
        return args

    def _handle_stdout(self) -> None:
        if not self._process:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._stdout_chunks.append(text)
            self.output.emit(text)

    def _handle_stderr(self) -> None:
        if not self._process:
            return
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self._stderr_chunks.append(text)
            self.error_output.emit(text)

    def _handle_process_error(self, _error: QProcess.ProcessError) -> None:
        if not self._process:
            return
        self.error_output.emit(self._process.errorString() + "\n")

    def _handle_state_changed(self, state: QProcess.ProcessState) -> None:
        if state == QProcess.Starting:
            self.state_changed.emit("starting")
            return
        if state == QProcess.Running:
            self.started.emit()
            self.state_changed.emit("running")
            return
        self.state_changed.emit("idle")

    def _handle_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        stdout_text = "".join(self._stdout_chunks)
        stderr_text = "".join(self._stderr_chunks)
        summary = self._extract_summary(stdout_text)
        if summary is None:
            summary = self._fallback_summary(stdout_text, stderr_text, exit_code)

        if self._cancel_requested:
            summary["status"] = "cancelled"
            summary["error"] = summary.get("error") or "Cancelled by user"

        summary.setdefault("stdout", stdout_text)
        summary.setdefault("stderr", stderr_text)
        summary.setdefault("exit_code", exit_code)

        self._process.deleteLater()
        self._process = None

        if summary.get("status") == "ok" and exit_code == 0 and not self._cancel_requested:
            self.finished_ok.emit(summary)
        else:
            self.finished_failed.emit(summary)

    @staticmethod
    def _extract_summary(stdout_text: str) -> dict[str, Any] | None:
        for line in reversed(stdout_text.splitlines()):
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "status" in payload:
                return payload
        return None

    @staticmethod
    def _fallback_summary(stdout_text: str, stderr_text: str, exit_code: int) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "status": "ok" if exit_code == 0 else "failed",
            "error": None,
            "run_id": "",
            "elapsed_ms": 0.0,
            "run_dir": "",
            "resolved_config": None,
            "artifacts_path": None,
            "artifacts": {},
            "log_path": None,
        }
        for line in stdout_text.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "run_id":
                summary["run_id"] = value
            elif key == "status":
                summary["status"] = value
            elif key == "resolved_config":
                summary["resolved_config"] = value
            elif key == "artifacts":
                summary["artifacts_path"] = value
        if exit_code != 0:
            summary["error"] = stderr_text.strip() or f"CLI exited with code {exit_code}"
        return summary

