"""Shared desktop UI theme."""

from __future__ import annotations

from typing import Protocol


class Stylable(Protocol):
    def setStyleSheet(self, style_sheet: str) -> None:
        """Apply a Qt stylesheet."""


def apply_theme(target: Stylable) -> None:
    target.setStyleSheet(
        """
        QMainWindow, QDialog, QWidget {
            background: #F6F8FB;
            color: #0F172A;
        }
        QScrollArea {
            background: transparent;
            border: none;
        }
        QGroupBox {
            background: #FFFFFF;
            border: 1px solid #D8E0EA;
            border-radius: 8px;
            margin-top: 14px;
            padding: 14px 12px 12px 12px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #1E3A5F;
        }
        QToolBar {
            background: #FFFFFF;
            border-bottom: 1px solid #D8E0EA;
            spacing: 4px;
            padding: 6px;
        }
        QToolButton {
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 6px 9px;
            color: #0F172A;
        }
        QToolButton:hover {
            background: #EAF1FB;
            border-color: #C8D7EA;
        }
        QToolButton:pressed {
            background: #DDEAF8;
        }
        QLabel#launcherTitle {
            color: #0F172A;
            font-size: 22px;
            font-weight: 700;
        }
        QLabel#launcherSubtitle {
            color: #64748B;
            font-size: 12px;
        }
        QLabel#summaryTitle {
            color: #64748B;
            font-size: 12px;
            font-weight: 500;
        }
        QLabel#summaryValue {
            color: #0F172A;
            font-size: 13px;
            font-weight: 700;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 5px;
            min-height: 28px;
            padding: 3px 7px;
            selection-background-color: #2563EB;
            selection-color: #FFFFFF;
        }
        QPlainTextEdit {
            min-height: 96px;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
        QPlainTextEdit:focus {
            border: 1px solid #2563EB;
        }
        QPushButton {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 5px;
            min-height: 28px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background: #EAF1FB;
            border-color: #9DB7D8;
        }
        QPushButton:pressed {
            background: #DDEAF8;
        }
        QPushButton:disabled {
            color: #94A3B8;
            background: #F1F5F9;
        }
        QPushButton#primaryAction {
            background: #2563EB;
            border-color: #1D4ED8;
            color: #FFFFFF;
            font-weight: 600;
        }
        QPushButton#primaryAction:hover {
            background: #1D4ED8;
        }
        QTabWidget::pane {
            border: 1px solid #D8E0EA;
            border-radius: 8px;
            background: #FFFFFF;
            top: -1px;
        }
        QTabBar::tab {
            background: #EEF3F8;
            border: 1px solid #D8E0EA;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 7px 14px;
            margin-right: 2px;
            color: #334155;
        }
        QTabBar::tab:selected {
            background: #FFFFFF;
            color: #0F172A;
            font-weight: 600;
        }
        QTabBar::tab:disabled {
            color: #94A3B8;
            background: #F1F5F9;
        }
        QSplitter::handle {
            background: #E2E8F0;
        }
        QStatusBar {
            background: #FFFFFF;
            border-top: 1px solid #D8E0EA;
            color: #334155;
        }
        """
    )
