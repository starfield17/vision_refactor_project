"""Log display widget."""

from __future__ import annotations

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class LogPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.auto_scroll = QCheckBox("Auto Scroll", self)
        self.auto_scroll.setChecked(True)
        self.copy_button = QPushButton("Copy", self)
        self.clear_button = QPushButton("Clear", self)
        controls.addWidget(self.auto_scroll)
        controls.addStretch(1)
        controls.addWidget(self.copy_button)
        controls.addWidget(self.clear_button)

        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setReadOnly(True)

        layout.addLayout(controls)
        layout.addWidget(self.text_edit, 1)

        self.clear_button.clicked.connect(self.text_edit.clear)
        self.copy_button.clicked.connect(self._copy)

    def append_text(self, text: str) -> None:
        if not text:
            return
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        if self.auto_scroll.isChecked():
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()

    def set_text(self, text: str) -> None:
        self.text_edit.setPlainText(text)

    def clear(self) -> None:
        self.text_edit.clear()

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.text_edit.toPlainText())
