"""Small form helpers for desktop pages."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QWidget


def create_group(title: str, parent: QWidget | None = None) -> tuple[QGroupBox, QVBoxLayout]:
    group = QGroupBox(title, parent)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)
    return group, layout

