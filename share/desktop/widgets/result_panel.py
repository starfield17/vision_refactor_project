"""Result summary widget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from share.application.common import format_elapsed, safe_open_path


class ResultPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.run_id_label = QLabel("-", self)
        self.status_label = QLabel("IDLE", self)
        self.elapsed_label = QLabel("-", self)
        self.run_dir_label = QLabel("-", self)
        self.resolved_config_label = QLabel("-", self)
        self.artifacts_path_label = QLabel("-", self)
        self.run_dir_label.setWordWrap(True)
        self.resolved_config_label.setWordWrap(True)
        self.artifacts_path_label.setWordWrap(True)

        form.addRow("Run ID", self.run_id_label)
        form.addRow("Status", self.status_label)
        form.addRow("Elapsed", self.elapsed_label)
        form.addRow("Run Dir", self.run_dir_label)
        form.addRow("Resolved Config", self.resolved_config_label)
        form.addRow("Artifacts", self.artifacts_path_label)

        buttons = QHBoxLayout()
        self.open_output_button = QPushButton("Open Output Dir", self)
        self.open_run_dir_button = QPushButton("Open Run Dir", self)
        self.open_config_button = QPushButton("Open Config", self)
        buttons.addWidget(self.open_output_button)
        buttons.addWidget(self.open_run_dir_button)
        buttons.addWidget(self.open_config_button)

        layout.addLayout(form)
        layout.addLayout(buttons)

        self.open_output_button.clicked.connect(self._open_output_dir)
        self.open_run_dir_button.clicked.connect(self._open_run_dir)
        self.open_config_button.clicked.connect(self._open_config)
        self.clear()

    def clear(self) -> None:
        self._result = {}
        self.run_id_label.setText("-")
        self.status_label.setText("IDLE")
        self.elapsed_label.setText("-")
        self.run_dir_label.setText("-")
        self.resolved_config_label.setText("-")
        self.artifacts_path_label.setText("-")
        self._sync_button_state()

    def set_result(self, result: dict[str, Any]) -> None:
        self._result = result
        self.run_id_label.setText(str(result.get("run_id") or "-"))
        self.status_label.setText(str(result.get("status", "idle")).upper())
        elapsed_ms = result.get("elapsed_ms")
        self.elapsed_label.setText(
            format_elapsed(float(elapsed_ms)) if isinstance(elapsed_ms, (int, float)) else "-"
        )
        self.run_dir_label.setText(str(result.get("run_dir") or "-"))
        self.resolved_config_label.setText(str(result.get("resolved_config") or "-"))
        self.artifacts_path_label.setText(str(result.get("artifacts_path") or "-"))
        self._sync_button_state()

    def _sync_button_state(self) -> None:
        run_dir = bool(self._result.get("run_dir"))
        resolved_config = bool(self._result.get("resolved_config"))
        self.open_output_button.setEnabled(run_dir)
        self.open_run_dir_button.setEnabled(run_dir)
        self.open_config_button.setEnabled(resolved_config)

    def _open_output_dir(self) -> None:
        run_dir = self._result.get("run_dir")
        if not run_dir:
            return
        safe_open_path(Path(run_dir).resolve().parents[1] / "outputs")

    def _open_run_dir(self) -> None:
        run_dir = self._result.get("run_dir")
        if run_dir:
            safe_open_path(run_dir)

    def _open_config(self) -> None:
        resolved_config = self._result.get("resolved_config")
        if resolved_config:
            safe_open_path(resolved_config)

