"""Main desktop window."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from share.desktop.pages.autolabel_page import AutoLabelPage
from share.desktop.pages.train_page import TrainPage


class MainWindow(QMainWindow):
    def __init__(self, default_mode: str = "train", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vision Desktop")
        self.resize(1600, 980)

        self.tabs = QTabWidget(self)
        self.train_page = TrainPage(self.tabs)
        self.autolabel_page = AutoLabelPage(self.tabs)
        self.tabs.addTab(self.train_page, "Train")
        self.tabs.addTab(self.autolabel_page, "AutoLabel")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Ready")

        self.train_page.status_changed.connect(
            lambda status: self._update_status_bar("Train", status)
        )
        self.autolabel_page.status_changed.connect(
            lambda status: self._update_status_bar("AutoLabel", status)
        )
        self.tabs.currentChanged.connect(self._sync_current_page_status)

        if default_mode == "autolabel":
            self.tabs.setCurrentIndex(1)
        else:
            self.tabs.setCurrentIndex(0)
        self._sync_current_page_status(self.tabs.currentIndex())

    def _update_status_bar(self, page_name: str, status: str) -> None:
        current_name = self.tabs.tabText(self.tabs.currentIndex())
        if current_name == page_name:
            self.statusBar().showMessage(f"{page_name}: {status}")

    def _sync_current_page_status(self, index: int) -> None:
        page_name = self.tabs.tabText(index)
        page = self.train_page if index == 0 else self.autolabel_page
        self.statusBar().showMessage(f"{page_name}: {page.current_status}")

