"""Autolabel desktop page."""

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

from share.application.autolabel_service import (
    build_autolabel_overrides_from_payload,
    save_autolabel_config,
)
from share.config.editing import load_merged_user_config
from share.config.schema import AUTOLABEL_CONFLICTS, AUTOLABEL_MODEL_BACKENDS, AUTOLABEL_MODES
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


class AutoLabelPage(QWidget):
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
        self.labeled_dir = PathPicker(
            "Labeled",
            pick_mode="dir",
            placeholder="./work-dir/datasets/labeled",
            parent=config_group,
        )
        self.unlabeled_dir = PathPicker(
            "Unlabeled",
            pick_mode="dir",
            placeholder="./work-dir/datasets/unlabeled",
            parent=config_group,
        )
        self.model_onnx = PathPicker(
            "ONNX Model",
            pick_mode="file",
            placeholder="./work-dir/models/exp/model-int8.onnx",
            parent=config_group,
        )
        self.run_name = QLineEdit(config_group)
        self.reload_button = QPushButton("Reload Config", config_group)
        self.save_before_run = QCheckBox("Save Before Run", config_group)
        config_layout.addWidget(self.config_path)
        config_layout.addWidget(self.workdir_path)
        config_layout.addWidget(self.labeled_dir)
        config_layout.addWidget(self.unlabeled_dir)
        config_layout.addWidget(self.model_onnx)
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

        general_group, general_layout = create_group("General", container)
        general_form = QFormLayout()
        self.mode = QComboBox(general_group)
        self.mode.addItems(sorted(AUTOLABEL_MODES))
        self.device = QComboBox(general_group)
        self.device.addItems(["cpu", "cuda:0"])
        self.confidence = QDoubleSpinBox(general_group)
        self.confidence.setDecimals(3)
        self.confidence.setRange(0.0, 1.0)
        self.confidence.setSingleStep(0.01)
        self.batch_size = QSpinBox(general_group)
        self.batch_size.setRange(1, 4096)
        self.on_conflict = QComboBox(general_group)
        self.on_conflict.addItems(sorted(AUTOLABEL_CONFLICTS))
        self.visualize = QCheckBox("Visualize Results", general_group)
        general_form.addRow("Mode", self.mode)
        general_form.addRow("Device", self.device)
        general_form.addRow("Confidence", self.confidence)
        general_form.addRow("Batch Size", self.batch_size)
        general_form.addRow("On Conflict", self.on_conflict)
        general_layout.addLayout(general_form)
        general_layout.addWidget(self.visualize)

        model_group, model_layout = create_group("Model", container)
        model_form = QFormLayout()
        self.model_backend = QComboBox(model_group)
        self.model_backend.addItems(sorted(AUTOLABEL_MODEL_BACKENDS))
        model_form.addRow("Backend", self.model_backend)
        model_layout.addLayout(model_form)

        llm_group, llm_layout = create_group("LLM", container)
        llm_form = QFormLayout()
        self.llm_base_url = QLineEdit(llm_group)
        self.llm_model = QLineEdit(llm_group)
        self.llm_api_key = QLineEdit(llm_group)
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.llm_api_key_env_name = QLineEdit(llm_group)
        self.llm_prompt = QPlainTextEdit(llm_group)
        self.llm_timeout_sec = QDoubleSpinBox(llm_group)
        self.llm_timeout_sec.setDecimals(2)
        self.llm_timeout_sec.setRange(0.1, 100000.0)
        self.llm_timeout_sec.setSingleStep(0.5)
        self.llm_max_retries = QSpinBox(llm_group)
        self.llm_max_retries.setRange(0, 1000)
        self.llm_retry_backoff_sec = QDoubleSpinBox(llm_group)
        self.llm_retry_backoff_sec.setDecimals(2)
        self.llm_retry_backoff_sec.setRange(0.1, 100000.0)
        self.llm_retry_backoff_sec.setSingleStep(0.1)
        self.llm_qps_limit = QDoubleSpinBox(llm_group)
        self.llm_qps_limit.setDecimals(2)
        self.llm_qps_limit.setRange(0.1, 100000.0)
        self.llm_qps_limit.setSingleStep(0.1)
        self.llm_max_images = QSpinBox(llm_group)
        self.llm_max_images.setRange(0, 1000000)

        llm_form.addRow("Base URL", self.llm_base_url)
        llm_form.addRow("Model", self.llm_model)
        llm_form.addRow("API Key", self.llm_api_key)
        llm_form.addRow("API Key Env", self.llm_api_key_env_name)
        llm_layout.addLayout(llm_form)
        llm_layout.addWidget(QLabel("Prompt", llm_group))
        llm_layout.addWidget(self.llm_prompt)

        limits_form = QFormLayout()
        limits_form.addRow("Timeout (sec)", self.llm_timeout_sec)
        limits_form.addRow("Max Retries", self.llm_max_retries)
        limits_form.addRow("Retry Backoff", self.llm_retry_backoff_sec)
        limits_form.addRow("QPS Limit", self.llm_qps_limit)
        limits_form.addRow("Max Images", self.llm_max_images)
        llm_layout.addLayout(limits_form)

        layout.addWidget(general_group)
        layout.addWidget(model_group)
        layout.addWidget(llm_group)
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
        self.labeled_dir.set_text(str(_cfg_get(cfg, ("data", "labeled_dir"), "")))
        self.unlabeled_dir.set_text(str(_cfg_get(cfg, ("data", "unlabeled_dir"), "")))
        self.model_onnx.set_text(str(_cfg_get(cfg, ("autolabel", "model", "onnx_model"), "")))
        self.mode.setCurrentText(str(_cfg_get(cfg, ("autolabel", "mode"), "model")))
        self.device.setCurrentText(str(_cfg_get(cfg, ("train", "device"), "cpu")))
        self.confidence.setValue(float(_cfg_get(cfg, ("autolabel", "confidence"), 0.5)))
        self.batch_size.setValue(int(_cfg_get(cfg, ("autolabel", "batch_size"), 2)))
        self.on_conflict.setCurrentText(
            str(_cfg_get(cfg, ("autolabel", "on_conflict"), "skip"))
        )
        self.visualize.setChecked(bool(_cfg_get(cfg, ("autolabel", "visualize"), False)))
        self.model_backend.setCurrentText(
            str(_cfg_get(cfg, ("autolabel", "model", "backend"), "yolo"))
        )
        self.llm_base_url.setText(str(_cfg_get(cfg, ("autolabel", "llm", "base_url"), "")))
        self.llm_model.setText(str(_cfg_get(cfg, ("autolabel", "llm", "model"), "")))
        self.llm_api_key.setText(str(_cfg_get(cfg, ("autolabel", "llm", "api_key"), "")))
        self.llm_api_key_env_name.setText(
            str(_cfg_get(cfg, ("autolabel", "llm", "api_key_env_name"), ""))
        )
        self.llm_prompt.setPlainText(str(_cfg_get(cfg, ("autolabel", "llm", "prompt"), "")))
        self.llm_timeout_sec.setValue(
            float(_cfg_get(cfg, ("autolabel", "llm", "timeout_sec"), 60.0))
        )
        self.llm_max_retries.setValue(
            int(_cfg_get(cfg, ("autolabel", "llm", "max_retries"), 2))
        )
        self.llm_retry_backoff_sec.setValue(
            float(_cfg_get(cfg, ("autolabel", "llm", "retry_backoff_sec"), 1.5))
        )
        self.llm_qps_limit.setValue(
            float(_cfg_get(cfg, ("autolabel", "llm", "qps_limit"), 1.0))
        )
        self.llm_max_images.setValue(
            int(_cfg_get(cfg, ("autolabel", "llm", "max_images"), 0))
        )

    def _collect_payload(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name.text().strip(),
            "device": self.device.currentText(),
            "labeled_dir": self.labeled_dir.text(),
            "unlabeled_dir": self.unlabeled_dir.text(),
            "mode": self.mode.currentText(),
            "confidence": self.confidence.value(),
            "batch_size": self.batch_size.value(),
            "visualize": self.visualize.isChecked(),
            "on_conflict": self.on_conflict.currentText(),
            "model_backend": self.model_backend.currentText(),
            "model_onnx": self.model_onnx.text(),
            "llm_base_url": self.llm_base_url.text().strip(),
            "llm_model": self.llm_model.text().strip(),
            "llm_api_key": self.llm_api_key.text().strip(),
            "llm_api_key_env_name": self.llm_api_key_env_name.text().strip(),
            "llm_prompt": self.llm_prompt.toPlainText(),
            "llm_timeout_sec": self.llm_timeout_sec.value(),
            "llm_max_retries": self.llm_max_retries.value(),
            "llm_retry_backoff_sec": self.llm_retry_backoff_sec.value(),
            "llm_qps_limit": self.llm_qps_limit.value(),
            "llm_max_images": self.llm_max_images.value(),
        }

    def _collect_overrides(self) -> list[str]:
        overrides = build_autolabel_overrides_from_payload(self._collect_payload())
        overrides.extend(_parse_override_text(self.extra_overrides.toPlainText()))
        return overrides

    def _save_config(self) -> None:
        config_path = self._require_config_path()
        if not config_path:
            return
        try:
            save_autolabel_config(config_path=config_path, overrides=self._collect_overrides())
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
                save_autolabel_config(config_path=config_path, overrides=self._collect_overrides())
            except ConfigError as exc:
                self._show_error(str(exc))
                return

        self.log_panel.clear()
        self.result_panel.clear()
        self.runner.start_autolabel(
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
        QMessageBox.critical(self, "AutoLabel", message)

    def _show_info(self, message: str) -> None:
        QMessageBox.information(self, "AutoLabel", message)

