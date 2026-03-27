"""Path picker widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)


class PathPicker(QWidget):
    text_changed = Signal(str)

    def __init__(
        self,
        label: str,
        *,
        pick_mode: str = "file",
        placeholder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pick_mode = pick_mode

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.label = QLabel(label, self)
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.button = QPushButton("Browse", self)

        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

        self.edit.textChanged.connect(self.text_changed.emit)
        self.button.clicked.connect(self._browse)

    def text(self) -> str:
        return self.edit.text().strip()

    def set_text(self, value: str) -> None:
        self.edit.setText(value)

    def _browse(self) -> None:
        start_dir = str(Path(self.text()).expanduser().parent) if self.text() else ""
        if self._pick_mode == "dir":
            selected = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
            if selected:
                self.edit.setText(selected)
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            start_dir,
            "All Files (*)",
        )
        if selected:
            self.edit.setText(selected)

