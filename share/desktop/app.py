"""Desktop application launcher."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from share.desktop.main_window import MainWindow


def launch_gui(default_mode: str = "train", argv: list[str] | None = None) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(argv or sys.argv)

    window = MainWindow(default_mode=default_mode)
    window.show()

    if owns_app:
        return app.exec()
    return 0
