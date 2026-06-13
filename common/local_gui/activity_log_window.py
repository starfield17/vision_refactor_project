"""Activity log window for local GUI tools."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.local_gui.window_geometry import clamped_window_size


class ActivityLogWindow(QMainWindow):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.entries: list[tuple[str, str]] = []
        self._pending_entries: list[tuple[str, str]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        self.resize(clamped_window_size(980, 640, minimum_width=560, minimum_height=360))
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All", "all")
        self.filter_combo.addItem("Process", "process")
        self.filter_combo.addItem("Error", "error")
        self.export_button = QPushButton("Export")
        self.clear_button = QPushButton("Clear")

        controls.addWidget(QLabel("Filter"))
        controls.addWidget(self.filter_combo)
        controls.addStretch(1)
        controls.addWidget(self.export_button)
        controls.addWidget(self.clear_button)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(50000)

        layout.addLayout(controls)
        layout.addWidget(self.log_output, 1)

        self.flush_timer = QTimer(self)
        self.flush_timer.setInterval(75)
        self.flush_timer.timeout.connect(self._flush_pending_logs)
        self.filter_combo.currentIndexChanged.connect(self._refresh_log_view)
        self.clear_button.clicked.connect(self.clear_messages)
        self.export_button.clicked.connect(self._export_logs)

    def append_message(self, message: str) -> None:
        category = self._classify_message(message)
        self.entries.append((category, message))
        self._pending_entries.append((category, message))
        if not self.flush_timer.isActive():
            self.flush_timer.start()

    def clear_messages(self) -> None:
        self.entries.clear()
        self._pending_entries.clear()
        self.flush_timer.stop()
        self.log_output.clear()

    def _classify_message(self, message: str) -> str:
        lowered = message.lower()
        if "error" in lowered or "failed" in lowered or "traceback" in lowered:
            return "error"
        return "process"

    def _flush_pending_logs(self) -> None:
        if not self._pending_entries:
            self.flush_timer.stop()
            return
        selected = self.filter_combo.currentData() or "all"
        pending = self._pending_entries
        self._pending_entries = []
        lines = [text for category, text in pending if selected == "all" or selected == category]
        if lines:
            self.log_output.appendPlainText("\n".join(lines))
            scrollbar = self.log_output.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        if not self._pending_entries:
            self.flush_timer.stop()

    def _refresh_log_view(self) -> None:
        self._pending_entries = []
        self.flush_timer.stop()
        selected = self.filter_combo.currentData() or "all"
        lines = [
            text for category, text in self.entries if selected == "all" or selected == category
        ]
        self.log_output.setPlainText("\n".join(lines))
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _export_logs(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log",
            str(Path.home() / "vision-local-tool.log"),
            "Log Files (*.log *.txt);;All Files (*)",
        )
        if not path:
            return
        selected = self.filter_combo.currentData() or "all"
        lines = [
            text for category, text in self.entries if selected == "all" or selected == category
        ]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
