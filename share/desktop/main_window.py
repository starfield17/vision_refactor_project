"""Main desktop window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from share.desktop.pages.autolabel_page import AutoLabelPage
from share.desktop.pages.train_page import TrainPage


class ToolWindow(QMainWindow):
    def __init__(self, title: str, page: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setCentralWidget(page)
        self.resize(1500, 920)

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide()


class MainWindow(QMainWindow):
    def __init__(self, default_mode: str = "train", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vision Desktop")
        self.resize(760, 360)
        self.setMinimumSize(620, 300)

        self.train_page = TrainPage()
        self.autolabel_page = AutoLabelPage()
        self.train_window = ToolWindow("Vision Desktop - Train", self.train_page, self)
        self.autolabel_window = ToolWindow(
            "Vision Desktop - AutoLabel",
            self.autolabel_page,
            self,
        )

        self._build_ui()

        self.train_page.status_changed.connect(
            lambda status: self._update_status_bar("Train", status)
        )
        self.autolabel_page.status_changed.connect(
            lambda status: self._update_status_bar("AutoLabel", status)
        )

        if default_mode == "autolabel":
            self.open_autolabel_window()
        else:
            self.open_train_window()

    def _build_ui(self) -> None:
        toolbar = QToolBar(self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        style = self.style()
        self.train_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon),
            "Train",
            self,
        )
        self.autolabel_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView),
            "AutoLabel",
            self,
        )
        toolbar.addAction(self.train_action)
        toolbar.addAction(self.autolabel_action)
        self.train_action.triggered.connect(self.open_train_window)
        self.autolabel_action.triggered.connect(self.open_autolabel_window)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("Vision Desktop", central)
        title.setObjectName("launcherTitle")
        subtitle = QLabel("Open a focused workflow window for training or automatic labeling.", central)
        subtitle.setObjectName("launcherSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        self.train_card, self.train_button = self._create_launcher_card(
            "Train",
            "Configure YOLO or Faster R-CNN training and export settings.",
            central,
        )
        self.autolabel_card, self.autolabel_button = self._create_launcher_card(
            "AutoLabel",
            "Run model, LLM, or LocateAnything labeling workflows.",
            central,
        )
        actions.addWidget(self.train_card)
        actions.addWidget(self.autolabel_card)
        root.addLayout(actions, 1)
        root.addStretch(1)

        self.train_button.clicked.connect(self.open_train_window)
        self.autolabel_button.clicked.connect(self.open_autolabel_window)
        self.statusBar().showMessage("Ready")

    def _create_launcher_card(
        self,
        title: str,
        description: str,
        parent: QWidget,
    ) -> tuple[QGroupBox, QPushButton]:
        card = QGroupBox(title, parent)
        card.setMinimumWidth(260)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        desc_label = QLabel(description, card)
        desc_label.setObjectName("summaryTitle")
        desc_label.setWordWrap(True)
        button = QPushButton(f"Open {title}", card)
        button.setObjectName("primaryAction")

        layout.addWidget(desc_label)
        layout.addStretch(1)
        layout.addWidget(button)
        return card, button

    def _show_child_window(self, window: ToolWindow) -> None:
        window.show()
        window.raise_()
        window.activateWindow()

    def open_train_window(self) -> None:
        self._show_child_window(self.train_window)
        self.statusBar().showMessage(f"Train: {self.train_page.current_status}")

    def open_autolabel_window(self) -> None:
        self._show_child_window(self.autolabel_window)
        self.statusBar().showMessage(f"AutoLabel: {self.autolabel_page.current_status}")

    def _update_status_bar(self, page_name: str, status: str) -> None:
        self.statusBar().showMessage(f"{page_name}: {status}")
