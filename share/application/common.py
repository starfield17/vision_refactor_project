"""Common application helpers for train/autolabel frontends."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from share.kernel.kernel import RunResult
from share.kernel.registry import KernelRegistry
from share.kernel.utils.logging import StructuredLogger


def resolve_log_path(cfg: dict[str, Any]) -> Path:
    """Resolve the workspace log path against the configured workdir."""
    workdir = Path(cfg["workspace"]["root"])
    log_file = Path(cfg["workspace"]["log_file"])
    if log_file.is_absolute():
        return log_file
    return workdir / log_file


def read_tail(path: Path | str, max_lines: int = 120) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def format_elapsed(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    secs = ms / 1000.0
    if secs < 60:
        return f"{secs:.1f} s"
    mins = secs / 60.0
    return f"{mins:.1f} min"


def append_override(overrides: list[str], key: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raw = "true" if value else "false"
    else:
        raw = str(value)
    overrides.append(f"{key}={raw}")


def build_train_registry() -> KernelRegistry:
    from share.kernel.trainer.faster_rcnn import run_faster_rcnn_train
    from share.kernel.trainer.yolo import run_yolo_train

    registry = KernelRegistry()
    registry.register_trainer("yolo", run_yolo_train)
    registry.register_trainer("faster_rcnn", run_faster_rcnn_train)
    return registry


def build_autolabel_registry() -> KernelRegistry:
    from share.kernel.autolabel.llm_autolabel import run_llm_autolabel
    from share.kernel.autolabel.model_autolabel import run_model_autolabel

    registry = KernelRegistry()
    registry.register_autolabeler("model", run_model_autolabel)
    registry.register_autolabeler("llm", run_llm_autolabel)
    return registry


def create_logger(cfg: dict[str, Any]) -> StructuredLogger:
    return StructuredLogger(
        log_path=resolve_log_path(cfg),
        level=cfg["workspace"]["log_level"],
    )


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_run_summary(run_dir: Path | str) -> dict[str, Any]:
    run_path = Path(run_dir)
    artifacts_path = run_path / "artifacts.json"
    metrics_path = run_path / "metrics.json"
    resolved_config = run_path / "config.resolved.toml"

    artifacts_payload = _load_json_file(artifacts_path)
    metrics_payload = _load_json_file(metrics_path)
    return {
        "run_dir": str(run_path),
        "resolved_config": str(resolved_config) if resolved_config.exists() else None,
        "artifacts_path": str(artifacts_path) if artifacts_path.exists() else None,
        "artifacts": artifacts_payload,
        "metrics": metrics_payload,
    }


def normalize_run_result(cfg: dict[str, Any], result: RunResult) -> dict[str, Any]:
    summary = read_run_summary(result.run_context.run_dir)
    return {
        "status": result.status,
        "error": result.error,
        "run_id": result.run_context.run_id,
        "elapsed_ms": result.elapsed_ms,
        "run_dir": str(result.run_context.run_dir),
        "resolved_config": summary["resolved_config"],
        "artifacts_path": summary["artifacts_path"],
        "artifacts": summary["artifacts"],
        "log_path": str(resolve_log_path(cfg)),
    }


def safe_open_path(path: Path | str) -> None:
    target = Path(path).expanduser().resolve()
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
        return
    subprocess.Popen(["xdg-open", str(target)])

