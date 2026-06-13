"""Train local desktop window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from common.application.train_service import build_train_overrides_from_payload, run_train
from common.config.config_loader import apply_overrides, save_resolved_config
from common.config.role_schema import role_to_kernel_config
from common.local_gui.activity_log_window import ActivityLogWindow
from common.local_gui.queue import (
    LocalQueueItem,
    LocalQueueManager,
    LocalQueueTableModel,
    QueueWindow,
    create_queue_item,
    create_queue_view,
)
from common.local_gui.theme import apply_theme
from common.local_gui.window_geometry import clamped_window_size
from train.config.schema import load_config, validate_config


class TrainMainWindow(QMainWindow):
    def __init__(self, config_path: Path, workdir_override: str | None = None) -> None:
        super().__init__()
        self.config_path = config_path
        self.workdir_override = workdir_override
        self.queue_model = LocalQueueTableModel(self)
        self.queue_manager = LocalQueueManager(self.queue_model, self._run_item, self)
        self.queue_window = QueueWindow("Train Queue", self.queue_model, self)
        self.activity_log_window = ActivityLogWindow("Train Activity Log", self)
        self.queue_busy = False
        self._build_ui()
        self._connect_signals()
        self._refresh_action_state()
        self._update_metrics(self.queue_model.metrics())

    def closeEvent(self, event) -> None:
        if not self.queue_busy:
            event.accept()
            return
        result = QMessageBox.question(
            self,
            "Training is running",
            "A queued training task is running. Stop after the current task and close?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if result == QMessageBox.Yes:
            self.queue_manager.stop()
        event.ignore()

    def _build_ui(self) -> None:
        apply_theme(self)
        self.resize(clamped_window_size(1280, 820, minimum_width=760, minimum_height=520))
        self.setWindowTitle("Vision Train")

        toolbar = QToolBar(self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        style = self.style()
        self.add_action = QAction(style.standardIcon(QStyle.SP_FileDialogNewFolder), "Add", self)
        self.start_action = QAction(style.standardIcon(QStyle.SP_MediaPlay), "Start", self)
        self.stop_action = QAction(style.standardIcon(QStyle.SP_MediaStop), "Stop", self)
        self.queue_action = QAction(style.standardIcon(QStyle.SP_FileDialogListView), "Queue", self)
        self.log_action = QAction(style.standardIcon(QStyle.SP_FileDialogInfoView), "Log", self)
        for action in [
            self.add_action,
            self.start_action,
            self.stop_action,
            self.queue_action,
            self.log_action,
        ]:
            toolbar.addAction(action)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        form_box = QGroupBox("Training Experiment")
        form = QGridLayout(form_box)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)
        self.run_name_edit = QLineEdit("exp001")
        self.dataset_edit = QLineEdit()
        self.dataset_button = QPushButton("Browse")
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["yolo", "faster_rcnn"])
        self.device_edit = QLineEdit("cpu")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(1)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 100000)
        self.batch_spin.setValue(4)
        self.dry_run_check = QCheckBox("Dry run")
        self.dry_run_check.setChecked(True)
        form.addWidget(QLabel("Run name"), 0, 0)
        form.addWidget(self.run_name_edit, 0, 1)
        form.addWidget(QLabel("Dataset"), 1, 0)
        form.addWidget(self.dataset_edit, 1, 1)
        form.addWidget(self.dataset_button, 1, 2)
        form.addWidget(QLabel("Backend"), 2, 0)
        form.addWidget(self.backend_combo, 2, 1)
        form.addWidget(QLabel("Device"), 2, 2)
        form.addWidget(self.device_edit, 2, 3)
        form.addWidget(QLabel("Epochs"), 3, 0)
        form.addWidget(self.epochs_spin, 3, 1)
        form.addWidget(QLabel("Batch size"), 3, 2)
        form.addWidget(self.batch_spin, 3, 3)
        form.addWidget(self.dry_run_check, 4, 1)
        root.addWidget(form_box)

        summary = QHBoxLayout()
        self.summary_label = QLabel()
        self.progress_label = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        summary.addWidget(self.summary_label)
        summary.addStretch(1)
        summary.addWidget(self.progress_label)
        summary.addWidget(self.progress_bar)
        root.addLayout(summary)

        self.table_view = create_queue_view(self)
        self.table_view.setModel(self.queue_model)
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        root.addWidget(self.table_view, 1)
        self.statusBar().showMessage("Ready")

    def _connect_signals(self) -> None:
        self.dataset_button.clicked.connect(self._choose_dataset)
        self.add_action.triggered.connect(self._add_current_to_queue)
        self.start_action.triggered.connect(self._start_queue)
        self.stop_action.triggered.connect(self.queue_manager.stop)
        self.queue_action.triggered.connect(self._show_queue_window)
        self.log_action.triggered.connect(self._show_log_window)
        self.queue_model.metricsChanged.connect(self._update_metrics)
        self.queue_model.metricsChanged.connect(self.queue_window.update_metrics)
        self.queue_manager.busyChanged.connect(self._on_busy_changed)
        self.queue_manager.log.connect(self._append_log)
        self.queue_manager.queueFinished.connect(
            lambda: self.statusBar().showMessage("Queue finished")
        )
        self.table_view.customContextMenuRequested.connect(
            lambda pos: self._show_queue_context_menu(self.table_view, pos)
        )
        self.queue_window.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.queue_window.table_view.customContextMenuRequested.connect(
            lambda pos: self._show_queue_context_menu(self.queue_window.table_view, pos)
        )

    def _choose_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select YOLO Dataset", self.dataset_edit.text()
        )
        if path:
            self.dataset_edit.setText(path)

    def _build_payload(self) -> dict[str, object]:
        return {
            "run_name": self.run_name_edit.text().strip() or "exp001",
            "dataset_dir": self.dataset_edit.text().strip() or None,
            "backend": self.backend_combo.currentText(),
            "device": self.device_edit.text().strip() or "cpu",
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "dry_run": self.dry_run_check.isChecked(),
        }

    def _add_current_to_queue(self) -> None:
        payload = self._build_payload()
        overrides = build_train_overrides_from_payload(payload)
        item = create_queue_item(
            name=str(payload["run_name"]),
            source=str(payload.get("dataset_dir") or ""),
            output="work-dir/models",
            mode=str(payload["backend"]),
            config_path=self.config_path,
            workdir_override=self.workdir_override,
            overrides=overrides,
        )
        self.queue_manager.add_item(item)
        self._append_log(f"[QUEUE] added train run {item.name}")
        self._refresh_action_state()

    def _run_item(self, item: LocalQueueItem) -> dict[str, object]:
        role_cfg = load_config(item.config_path, workdir_override=item.workdir_override)
        role_cfg = validate_config(apply_overrides(role_cfg, item.overrides))
        kernel_cfg = role_to_kernel_config(role_cfg, "train", "train")
        temp_path = Path(kernel_cfg["workspace"]["root"]) / "tmp" / f"{item.item_id}.train.toml"
        save_resolved_config(kernel_cfg, temp_path)
        return run_train(config_path=temp_path)

    def _start_queue(self) -> None:
        if not self.queue_manager.start():
            QMessageBox.information(self, "Queue", "No queued training tasks.")

    def _on_busy_changed(self, busy: bool) -> None:
        self.queue_busy = busy
        self.statusBar().showMessage("Running" if busy else "Ready")
        self._refresh_action_state()

    def _update_metrics(self, metrics) -> None:
        self.summary_label.setText(
            f"Total {metrics.total} | Queued {metrics.queued} | Running {metrics.running} | "
            f"Done {metrics.succeeded} | Failed {metrics.failed}"
        )
        self.progress_label.setText(f"{metrics.percent:.1f}%")
        self.progress_bar.setValue(int(round(metrics.percent * 10)))
        self._refresh_action_state()

    def _refresh_action_state(self) -> None:
        has_queued = bool(self.queue_model.execution_records())
        self.add_action.setEnabled(not self.queue_busy)
        self.start_action.setEnabled(not self.queue_busy and has_queued)
        self.stop_action.setEnabled(self.queue_busy)
        mode = QAbstractItemView.NoDragDrop if self.queue_busy else QAbstractItemView.InternalMove
        for view in [self.table_view, self.queue_window.table_view]:
            view.setDragDropMode(mode)

    def _append_log(self, message: str) -> None:
        self.activity_log_window.append_message(message)
        self.statusBar().showMessage(message, 5000)

    def _show_queue_window(self) -> None:
        self.queue_window.show()
        self.queue_window.raise_()
        self.queue_window.activateWindow()

    def _show_log_window(self) -> None:
        self.activity_log_window.show()
        self.activity_log_window.raise_()
        self.activity_log_window.activateWindow()

    def _selected_rows(self, view) -> list[int]:
        return sorted({index.row() for index in view.selectionModel().selectedRows()})

    def _show_queue_context_menu(self, view, pos: QPoint) -> None:
        rows = self._selected_rows(view)
        menu = QMenu(self)
        retry_action = menu.addAction("Retry")
        remove_action = menu.addAction("Remove")
        clear_action = menu.addAction("Clear Completed")
        retry_action.setEnabled(bool(rows) and not self.queue_busy)
        remove_action.setEnabled(bool(rows) and not self.queue_busy)
        clear_action.setEnabled(not self.queue_busy)
        chosen = menu.exec(view.viewport().mapToGlobal(pos))
        if chosen == retry_action:
            self.queue_manager.retry_rows(rows)
        elif chosen == remove_action:
            self.queue_manager.remove_rows(rows)
        elif chosen == clear_action:
            self.queue_manager.clear_completed()
        self._refresh_action_state()
