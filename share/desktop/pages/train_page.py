"""Train desktop page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from share.application.train_service import build_train_overrides_from_payload, save_train_config
from share.config.editing import load_merged_user_config
from share.config.schema import FASTER_RCNN_VARIANTS, QUANTIZE_MODES, TRAIN_BACKENDS
from share.desktop.runners import CliProcessRunner
from share.desktop.state import PageState
from share.desktop.widgets.common_form import create_group
from share.desktop.widgets.log_panel import LogPanel
from share.desktop.widgets.path_picker import PathPicker
from share.desktop.widgets.result_panel import ResultPanel
from share.types.errors import ConfigError


def _cfg_get(cfg: dict[str, Any], path: tuple[str, ...], fallback: Any) -> Any:
    cur: Any = cfg
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return fallback
        cur = cur[part]
    return cur


def _parse_override_text(raw: str) -> list[str]:
    items: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        items.append(text)
    return items


class TrainPage(QWidget):
    status_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.page_state = PageState()
        self.runner = CliProcessRunner(self)
        self._build_ui()
        self._connect_runner()
        self._load_initial_config()

    @property
    def current_status(self) -> str:
        return self.page_state.status

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal, self)
        root.addWidget(splitter, 1)

        left_panel = self._wrap_panel(self._build_left_panel())
        center_panel = self._wrap_panel(self._build_center_panel())
        right_panel = self._build_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 4)

    def _wrap_panel(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        return scroll

    def _build_left_panel(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        config_group, config_layout = create_group("Paths & Config", container)
        self.config_path = PathPicker(
            "Config",
            pick_mode="file",
            placeholder="./work-dir/config.toml",
            parent=config_group,
        )
        self.workdir_path = PathPicker(
            "Workdir",
            pick_mode="dir",
            placeholder="Optional runtime override",
            parent=config_group,
        )
        self.dataset_dir = PathPicker(
            "Dataset",
            pick_mode="dir",
            placeholder="./work-dir/datasets/yolo",
            parent=config_group,
        )
        self.run_name = QLineEdit(config_group)
        self.reload_button = QPushButton("Reload Config", config_group)
        self.save_before_run = QCheckBox("Save Before Run", config_group)

        config_layout.addWidget(self.config_path)
        config_layout.addWidget(self.workdir_path)
        config_layout.addWidget(self.dataset_dir)
        config_layout.addWidget(QLabel("Run Name", config_group))
        config_layout.addWidget(self.run_name)
        config_layout.addWidget(self.save_before_run)
        config_layout.addWidget(self.reload_button)

        extra_group, extra_layout = create_group("Extra Overrides", container)
        self.extra_overrides = QPlainTextEdit(extra_group)
        self.extra_overrides.setPlaceholderText("One KEY=VALUE per line")
        extra_layout.addWidget(self.extra_overrides)

        layout.addWidget(config_group)
        layout.addWidget(extra_group)
        layout.addStretch(1)

        self.reload_button.clicked.connect(self._load_config)
        return container

    def _build_center_panel(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        runtime_group, runtime_layout = create_group("Runtime", container)
        runtime_form = QFormLayout()
        self.backend = QComboBox(runtime_group)
        self.backend.addItems(sorted(TRAIN_BACKENDS))
        self.device = QComboBox(runtime_group)
        self.device.addItems(["cpu", "cuda:0"])
        self.seed = QSpinBox(runtime_group)
        self.seed.setRange(0, 999999)
        self.epochs = QSpinBox(runtime_group)
        self.epochs.setRange(1, 100000)
        self.batch_size = QSpinBox(runtime_group)
        self.batch_size.setRange(1, 4096)
        self.img_size = QSpinBox(runtime_group)
        self.img_size.setRange(32, 8192)
        self.img_size.setSingleStep(32)
        self.dry_run = QCheckBox("Enable Dry Run", runtime_group)
        runtime_form.addRow("Backend", self.backend)
        runtime_form.addRow("Device", self.device)
        runtime_form.addRow("Seed", self.seed)
        runtime_form.addRow("Epochs", self.epochs)
        runtime_form.addRow("Batch Size", self.batch_size)
        runtime_form.addRow("Img Size", self.img_size)
        runtime_layout.addLayout(runtime_form)
        runtime_layout.addWidget(self.dry_run)

        yolo_group, yolo_layout = create_group("YOLO", container)
        self.yolo_weights = PathPicker(
            "Weights",
            pick_mode="file",
            placeholder="./weights/model.pt",
            parent=yolo_group,
        )
        yolo_layout.addWidget(self.yolo_weights)

        frcnn_group, frcnn_layout = create_group("Faster R-CNN", container)
        frcnn_form = QFormLayout()
        self.frcnn_variant = QComboBox(frcnn_group)
        self.frcnn_variant.addItems(sorted(FASTER_RCNN_VARIANTS))
        self.frcnn_lr = QDoubleSpinBox(frcnn_group)
        self.frcnn_lr.setDecimals(6)
        self.frcnn_lr.setRange(0.000001, 1000.0)
        self.frcnn_lr.setSingleStep(0.001)
        self.frcnn_momentum = QDoubleSpinBox(frcnn_group)
        self.frcnn_momentum.setDecimals(4)
        self.frcnn_momentum.setRange(0.0, 1.0)
        self.frcnn_momentum.setSingleStep(0.01)
        self.frcnn_weight_decay = QDoubleSpinBox(frcnn_group)
        self.frcnn_weight_decay.setDecimals(6)
        self.frcnn_weight_decay.setRange(0.0, 100.0)
        self.frcnn_weight_decay.setSingleStep(0.0001)
        self.frcnn_num_workers = QSpinBox(frcnn_group)
        self.frcnn_num_workers.setRange(0, 256)
        self.frcnn_max_samples = QSpinBox(frcnn_group)
        self.frcnn_max_samples.setRange(0, 100000000)
        frcnn_form.addRow("Variant", self.frcnn_variant)
        frcnn_form.addRow("Learning Rate", self.frcnn_lr)
        frcnn_form.addRow("Momentum", self.frcnn_momentum)
        frcnn_form.addRow("Weight Decay", self.frcnn_weight_decay)
        frcnn_form.addRow("Num Workers", self.frcnn_num_workers)
        frcnn_form.addRow("Max Samples", self.frcnn_max_samples)
        frcnn_layout.addLayout(frcnn_form)

        export_group, export_layout = create_group("Export", container)
        export_form = QFormLayout()
        self.export_onnx = QCheckBox("Export ONNX", export_group)
        self.export_quantize = QCheckBox("Quantize", export_group)
        self.export_opset = QSpinBox(export_group)
        self.export_opset.setRange(1, 100)
        self.export_quantize_mode = QComboBox(export_group)
        self.export_quantize_mode.addItems(sorted(QUANTIZE_MODES))
        self.export_calib_samples = QSpinBox(export_group)
        self.export_calib_samples.setRange(0, 1000000)
        export_layout.addWidget(self.export_onnx)
        export_layout.addWidget(self.export_quantize)
        export_form.addRow("ONNX Opset", self.export_opset)
        export_form.addRow("Quantize Mode", self.export_quantize_mode)
        export_form.addRow("Calib Samples", self.export_calib_samples)
        export_layout.addLayout(export_form)

        layout.addWidget(runtime_group)
        layout.addWidget(yolo_group)
        layout.addWidget(frcnn_group)
        layout.addWidget(export_group)
        layout.addStretch(1)
        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        actions_group, actions_layout = create_group("Run", container)
        self.status_label = QLabel("Status: IDLE", actions_group)
        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save Config", actions_group)
        self.run_button = QPushButton("Run", actions_group)
        self.stop_button = QPushButton("Stop", actions_group)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        actions_layout.addWidget(self.status_label)
        actions_layout.addLayout(buttons)

        result_group, result_layout = create_group("Latest Run Summary", container)
        self.result_panel = ResultPanel(result_group)
        result_layout.addWidget(self.result_panel)

        log_group, log_layout = create_group("Log Tail", container)
        self.log_panel = LogPanel(log_group)
        log_layout.addWidget(self.log_panel, 1)

        layout.addWidget(actions_group)
        layout.addWidget(result_group)
        layout.addWidget(log_group, 1)

        self.save_button.clicked.connect(self._save_config)
        self.run_button.clicked.connect(self._run)
        self.stop_button.clicked.connect(self.runner.stop)
        return container

    def _connect_runner(self) -> None:
        self.runner.started.connect(lambda: self._set_status("running"))
        self.runner.state_changed.connect(self._on_runner_state_changed)
        self.runner.output.connect(self.log_panel.append_text)
        self.runner.error_output.connect(self.log_panel.append_text)
        self.runner.finished_ok.connect(self._on_run_finished)
        self.runner.finished_failed.connect(self._on_run_finished)

    def _default_config_path(self) -> str:
        default = Path.cwd() / "work-dir" / "config.toml"
        return str(default.resolve()) if default.exists() else ""

    def _load_initial_config(self) -> None:
        self.config_path.set_text(self._default_config_path())
        if self.config_path.text():
            self._load_config()

    def _load_config(self) -> None:
        config_path = self.config_path.text()
        if not config_path:
            self._show_error("Config path is required.")
            return
        try:
            cfg = load_merged_user_config(Path(config_path).resolve())
        except ConfigError as exc:
            self._show_error(str(exc))
            return

        self.run_name.setText(str(_cfg_get(cfg, ("workspace", "run_name"), "exp001")))
        self.dataset_dir.set_text(str(_cfg_get(cfg, ("data", "yolo_dataset_dir"), "")))
        self.backend.setCurrentText(str(_cfg_get(cfg, ("train", "backend"), "yolo")))
        self.device.setCurrentText(str(_cfg_get(cfg, ("train", "device"), "cpu")))
        self.seed.setValue(int(_cfg_get(cfg, ("train", "seed"), 42)))
        self.epochs.setValue(int(_cfg_get(cfg, ("train", "epochs"), 1)))
        self.batch_size.setValue(int(_cfg_get(cfg, ("train", "batch_size"), 4)))
        self.img_size.setValue(int(_cfg_get(cfg, ("train", "img_size"), 640)))
        self.dry_run.setChecked(bool(_cfg_get(cfg, ("train", "dry_run"), False)))
        self.yolo_weights.set_text(str(_cfg_get(cfg, ("train", "yolo", "weights"), "")))
        self.frcnn_variant.setCurrentText(
            str(_cfg_get(cfg, ("train", "faster_rcnn", "variant"), "mobilenet_v3"))
        )
        self.frcnn_lr.setValue(float(_cfg_get(cfg, ("train", "faster_rcnn", "lr"), 0.005)))
        self.frcnn_momentum.setValue(
            float(_cfg_get(cfg, ("train", "faster_rcnn", "momentum"), 0.9))
        )
        self.frcnn_weight_decay.setValue(
            float(_cfg_get(cfg, ("train", "faster_rcnn", "weight_decay"), 0.0005))
        )
        self.frcnn_num_workers.setValue(
            int(_cfg_get(cfg, ("train", "faster_rcnn", "num_workers"), 0))
        )
        self.frcnn_max_samples.setValue(
            int(_cfg_get(cfg, ("train", "faster_rcnn", "max_samples"), 0))
        )
        self.export_onnx.setChecked(bool(_cfg_get(cfg, ("export", "onnx"), True)))
        self.export_opset.setValue(int(_cfg_get(cfg, ("export", "opset"), 17)))
        self.export_quantize.setChecked(bool(_cfg_get(cfg, ("export", "quantize"), True)))
        self.export_quantize_mode.setCurrentText(
            str(_cfg_get(cfg, ("export", "quantize_mode"), "dynamic"))
        )
        self.export_calib_samples.setValue(int(_cfg_get(cfg, ("export", "calib_samples"), 32)))

    def _collect_payload(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name.text().strip(),
            "dataset_dir": self.dataset_dir.text(),
            "backend": self.backend.currentText(),
            "device": self.device.currentText(),
            "seed": self.seed.value(),
            "epochs": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "img_size": self.img_size.value(),
            "dry_run": self.dry_run.isChecked(),
            "yolo_weights": self.yolo_weights.text(),
            "frcnn_variant": self.frcnn_variant.currentText(),
            "frcnn_lr": self.frcnn_lr.value(),
            "frcnn_momentum": self.frcnn_momentum.value(),
            "frcnn_weight_decay": self.frcnn_weight_decay.value(),
            "frcnn_num_workers": self.frcnn_num_workers.value(),
            "frcnn_max_samples": self.frcnn_max_samples.value(),
            "export_onnx": self.export_onnx.isChecked(),
            "export_opset": self.export_opset.value(),
            "export_quantize": self.export_quantize.isChecked(),
            "export_quantize_mode": self.export_quantize_mode.currentText(),
            "export_calib_samples": self.export_calib_samples.value(),
        }

    def _collect_overrides(self) -> list[str]:
        overrides = build_train_overrides_from_payload(self._collect_payload())
        overrides.extend(_parse_override_text(self.extra_overrides.toPlainText()))
        return overrides

    def _save_config(self) -> None:
        config_path = self._require_config_path()
        if not config_path:
            return
        try:
            save_train_config(config_path=config_path, overrides=self._collect_overrides())
        except ConfigError as exc:
            self._show_error(str(exc))
            return
        self._show_info("Config saved.")

    def _run(self) -> None:
        config_path = self._require_config_path()
        if not config_path:
            return
        if self.save_before_run.isChecked():
            try:
                save_train_config(config_path=config_path, overrides=self._collect_overrides())
            except ConfigError as exc:
                self._show_error(str(exc))
                return

        self.log_panel.clear()
        self.result_panel.clear()
        self.runner.start_train(
            config_path=config_path,
            workdir_override=self.workdir_path.text() or None,
            overrides=self._collect_overrides(),
        )
        self._set_status("starting")

    def _require_config_path(self) -> str | None:
        config_path = self.config_path.text()
        if not config_path:
            self._show_error("Config path is required.")
            return None
        if not Path(config_path).expanduser().exists():
            self._show_error(f"Config file not found: {config_path}")
            return None
        return str(Path(config_path).resolve())

    def _on_runner_state_changed(self, status: str) -> None:
        if status in {"starting", "running", "stopping"}:
            self._set_status(status)
        elif status == "idle" and self.page_state.status == "running":
            self._set_status("idle")

    def _on_run_finished(self, result: dict[str, Any]) -> None:
        self.page_state.last_result = result
        final_status = str(result.get("status") or "failed")
        self.result_panel.set_result(result)
        self._set_status(final_status)

    def _set_status(self, status: str) -> None:
        self.page_state.status = status
        self.status_label.setText(f"Status: {status.upper()}")
        running = status in {"starting", "running", "stopping"}
        self.run_button.setEnabled(not running)
        self.save_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.status_changed.emit(status)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Train", message)

    def _show_info(self, message: str) -> None:
        QMessageBox.information(self, "Train", message)

