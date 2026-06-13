"""Queue model and worker helpers for local GUI tools."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMainWindow,
    QProgressBar,
    QTableView,
    QVBoxLayout,
    QLabel,
    QWidget,
)

from common.local_gui.window_geometry import clamped_window_size


class QueueItemStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class LocalQueueItem:
    item_id: str
    name: str
    source: str
    output: str
    mode: str
    config_path: Path
    workdir_override: str | None
    overrides: list[str]
    status: QueueItemStatus = QueueItemStatus.QUEUED
    progress: float = 0.0
    error: str = ""
    run_dir: str = ""
    artifacts_path: str = ""
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueueMetrics:
    total: int = 0
    queued: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    percent: float = 0.0


class QueueColumn(IntEnum):
    NAME = 0
    SOURCE = 1
    OUTPUT = 2
    MODE = 3
    STATUS = 4
    PROGRESS = 5
    ARTIFACTS = 6
    ERROR = 7


COLUMN_HEADERS = {
    QueueColumn.NAME: "Name",
    QueueColumn.SOURCE: "Source",
    QueueColumn.OUTPUT: "Output",
    QueueColumn.MODE: "Mode",
    QueueColumn.STATUS: "Status",
    QueueColumn.PROGRESS: "Progress",
    QueueColumn.ARTIFACTS: "Artifacts",
    QueueColumn.ERROR: "Error",
}


def _short_path(raw: str) -> str:
    if not raw:
        return "-"
    path = Path(raw)
    return path.name if path.name else str(path)


def compute_metrics(records: list[LocalQueueItem]) -> QueueMetrics:
    metrics = QueueMetrics(total=len(records))
    for record in records:
        if record.status == QueueItemStatus.QUEUED:
            metrics.queued += 1
        elif record.status == QueueItemStatus.RUNNING:
            metrics.running += 1
        elif record.status == QueueItemStatus.SUCCEEDED:
            metrics.succeeded += 1
        elif record.status == QueueItemStatus.FAILED:
            metrics.failed += 1
        elif record.status == QueueItemStatus.CANCELLED:
            metrics.cancelled += 1
    if metrics.total:
        metrics.percent = (
            100.0 * (metrics.succeeded + metrics.failed + metrics.cancelled) / metrics.total
        )
    return metrics


class LocalQueueTableModel(QAbstractTableModel):
    metricsChanged = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: list[LocalQueueItem] = []
        self._metrics = QueueMetrics()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(QueueColumn)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Vertical:
            return section + 1
        return COLUMN_HEADERS.get(QueueColumn(section), "")

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        record = self._records[index.row()]
        column = QueueColumn(index.column())
        if role == Qt.DisplayRole:
            if column == QueueColumn.NAME:
                return record.name
            if column == QueueColumn.SOURCE:
                return _short_path(record.source)
            if column == QueueColumn.OUTPUT:
                return _short_path(record.output)
            if column == QueueColumn.MODE:
                return record.mode
            if column == QueueColumn.STATUS:
                return record.status.value
            if column == QueueColumn.PROGRESS:
                return f"{record.progress:.1f}%"
            if column == QueueColumn.ARTIFACTS:
                return _short_path(record.artifacts_path or record.run_dir)
            if column == QueueColumn.ERROR:
                return record.error
        if role == Qt.ToolTipRole:
            return "\n".join(
                [
                    f"Config: {record.config_path}",
                    f"Source: {record.source or '-'}",
                    f"Output: {record.output or '-'}",
                    f"Run dir: {record.run_dir or '-'}",
                    f"Artifacts: {record.artifacts_path or '-'}",
                    f"Error: {record.error or '-'}",
                ]
            )
        if role == Qt.ForegroundRole and column in {QueueColumn.STATUS, QueueColumn.PROGRESS}:
            return {
                QueueItemStatus.RUNNING: QColor("#0B5394"),
                QueueItemStatus.SUCCEEDED: QColor("#38761D"),
                QueueItemStatus.FAILED: QColor("#A61C00"),
                QueueItemStatus.CANCELLED: QColor("#7F6000"),
            }.get(record.status)
        if role == Qt.TextAlignmentRole and column in {QueueColumn.STATUS, QueueColumn.PROGRESS}:
            return int(Qt.AlignCenter)
        return None

    def records(self) -> list[LocalQueueItem]:
        return list(self._records)

    def metrics(self) -> QueueMetrics:
        return self._metrics

    def execution_records(self) -> list[LocalQueueItem]:
        return [record for record in self._records if record.status == QueueItemStatus.QUEUED]

    def add_record(self, record: LocalQueueItem) -> None:
        row = len(self._records)
        self.beginInsertRows(QModelIndex(), row, row)
        self._records.append(record)
        self.endInsertRows()
        self._refresh_metrics()

    def remove_rows(self, rows: list[int]) -> int:
        removed = 0
        for row in sorted(set(rows), reverse=True):
            if not (0 <= row < len(self._records)):
                continue
            if self._records[row].status == QueueItemStatus.RUNNING:
                continue
            self.beginRemoveRows(QModelIndex(), row, row)
            self._records.pop(row)
            self.endRemoveRows()
            removed += 1
        if removed:
            self._refresh_metrics()
        return removed

    def retry_rows(self, rows: list[int]) -> int:
        changed = 0
        for row in sorted(set(rows)):
            if not (0 <= row < len(self._records)):
                continue
            record = self._records[row]
            if record.status not in {QueueItemStatus.FAILED, QueueItemStatus.CANCELLED}:
                continue
            record.status = QueueItemStatus.QUEUED
            record.progress = 0.0
            record.error = ""
            changed += 1
            self.dataChanged.emit(self.index(row, 0), self.index(row, len(QueueColumn) - 1))
        if changed:
            self._refresh_metrics()
        return changed

    def clear_completed(self) -> int:
        rows = [
            index
            for index, record in enumerate(self._records)
            if record.status
            in {QueueItemStatus.SUCCEEDED, QueueItemStatus.FAILED, QueueItemStatus.CANCELLED}
        ]
        return self.remove_rows(rows)

    def mark_running(self, item_id: str) -> None:
        self._update_record(item_id, status=QueueItemStatus.RUNNING, progress=0.0, error="")

    def mark_succeeded(self, item_id: str, result: dict[str, Any]) -> None:
        self._update_record(
            item_id,
            status=QueueItemStatus.SUCCEEDED,
            progress=100.0,
            result=result,
            run_dir=str(result.get("run_dir") or ""),
            artifacts_path=str(result.get("artifacts_path") or ""),
            error="",
        )

    def mark_failed(self, item_id: str, error: str, result: dict[str, Any] | None = None) -> None:
        self._update_record(
            item_id,
            status=QueueItemStatus.FAILED,
            progress=100.0,
            result=result or {},
            error=error,
        )

    def mark_cancelled(self, item_id: str) -> None:
        self._update_record(
            item_id,
            status=QueueItemStatus.CANCELLED,
            progress=100.0,
            error="cancelled",
        )

    def _update_record(self, item_id: str, **updates: Any) -> None:
        row_record = self._find_record(item_id)
        if row_record is None:
            return
        row, record = row_record
        for key, value in updates.items():
            setattr(record, key, value)
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(QueueColumn) - 1))
        self._refresh_metrics()

    def _find_record(self, item_id: str) -> tuple[int, LocalQueueItem] | None:
        for row, record in enumerate(self._records):
            if record.item_id == item_id:
                return row, record
        return None

    def _refresh_metrics(self) -> None:
        self._metrics = compute_metrics(self._records)
        self.metricsChanged.emit(self._metrics)


def create_queue_item(
    *,
    name: str,
    source: str,
    output: str,
    mode: str,
    config_path: Path,
    workdir_override: str | None,
    overrides: list[str],
) -> LocalQueueItem:
    return LocalQueueItem(
        item_id=uuid.uuid4().hex,
        name=name,
        source=source,
        output=output,
        mode=mode,
        config_path=config_path,
        workdir_override=workdir_override,
        overrides=overrides,
    )


def create_queue_view(parent=None) -> QTableView:
    view = QTableView(parent)
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QAbstractItemView.SelectRows)
    view.setSelectionMode(QAbstractItemView.ExtendedSelection)
    view.setSortingEnabled(False)
    view.setWordWrap(False)
    view.verticalHeader().setVisible(False)
    header = view.horizontalHeader()
    header.setStretchLastSection(True)
    header.setSectionResizeMode(QHeaderView.Interactive)
    for column, width in {
        QueueColumn.NAME: 180,
        QueueColumn.SOURCE: 180,
        QueueColumn.OUTPUT: 180,
        QueueColumn.MODE: 120,
        QueueColumn.STATUS: 110,
        QueueColumn.PROGRESS: 100,
        QueueColumn.ARTIFACTS: 180,
    }.items():
        view.setColumnWidth(int(column), width)
    return view


class QueueWindow(QMainWindow):
    def __init__(self, title: str, model: LocalQueueTableModel, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.model = model
        self._build_ui()

    def _build_ui(self) -> None:
        self.resize(clamped_window_size(1280, 760, minimum_width=640, minimum_height=420))
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.summary_label = QLabel()
        self.progress_label = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.table_view = create_queue_view(self)
        self.table_view.setModel(self.model)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.table_view, 1)
        self.update_metrics(self.model.metrics())

    def update_metrics(self, metrics: QueueMetrics) -> None:
        self.summary_label.setText(
            "Total {total} | Queued {queued} | Running {running} | "
            "Succeeded {succeeded} | Failed {failed} | Cancelled {cancelled}".format(
                total=metrics.total,
                queued=metrics.queued,
                running=metrics.running,
                succeeded=metrics.succeeded,
                failed=metrics.failed,
                cancelled=metrics.cancelled,
            )
        )
        self.progress_label.setText(f"Queue progress: {metrics.percent:.1f}%")
        self.progress_bar.setValue(int(round(metrics.percent * 10)))


class LocalQueueWorker(QThread):
    itemStarted = Signal(str)
    itemSucceeded = Signal(str, object)
    itemFailed = Signal(str, str, object)
    itemCancelled = Signal(str)
    log = Signal(str)
    finishedQueue = Signal()

    def __init__(
        self,
        records: list[LocalQueueItem],
        run_item: Callable[[LocalQueueItem], dict[str, Any]],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.records = records
        self.run_item = run_item
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            for record in self.records:
                if self._cancel_requested:
                    self.itemCancelled.emit(record.item_id)
                    continue
                self.itemStarted.emit(record.item_id)
                self.log.emit(f"[START] {record.name}")
                try:
                    result = self.run_item(record)
                except Exception as exc:
                    self.log.emit(traceback.format_exc())
                    self.itemFailed.emit(record.item_id, str(exc), {})
                    continue
                status = str(result.get("status") or "")
                if status == "ok":
                    self.itemSucceeded.emit(record.item_id, result)
                    self.log.emit(f"[DONE] {record.name}")
                else:
                    error = str(result.get("error") or "run failed")
                    self.itemFailed.emit(record.item_id, error, result)
                    self.log.emit(f"[FAILED] {record.name}: {error}")
        finally:
            self.finishedQueue.emit()


class LocalQueueManager(QObject):
    busyChanged = Signal(bool)
    log = Signal(str)
    error = Signal(str)
    queueFinished = Signal()

    def __init__(
        self,
        model: LocalQueueTableModel,
        run_item: Callable[[LocalQueueItem], dict[str, Any]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.run_item = run_item
        self._worker: LocalQueueWorker | None = None

    def is_busy(self) -> bool:
        return self._worker is not None

    def add_item(self, item: LocalQueueItem) -> None:
        self.model.add_record(item)

    def start(self) -> bool:
        if self._worker is not None:
            return False
        records = self.model.execution_records()
        if not records:
            return False
        worker = LocalQueueWorker(records, self.run_item, self)
        worker.itemStarted.connect(self.model.mark_running)
        worker.itemSucceeded.connect(self.model.mark_succeeded)
        worker.itemFailed.connect(self.model.mark_failed)
        worker.itemCancelled.connect(self.model.mark_cancelled)
        worker.log.connect(self.log.emit)
        worker.finishedQueue.connect(self._on_worker_finished)
        self._worker = worker
        self.busyChanged.emit(True)
        worker.start()
        return True

    def stop(self) -> bool:
        if self._worker is None:
            return False
        self._worker.cancel()
        return True

    def remove_rows(self, rows: list[int]) -> int:
        return self.model.remove_rows(rows)

    def retry_rows(self, rows: list[int]) -> int:
        return self.model.retry_rows(rows)

    def clear_completed(self) -> int:
        return self.model.clear_completed()

    def _on_worker_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self.busyChanged.emit(False)
        self.queueFinished.emit()
