"""Application service for autolabel workflows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from share.application.common import (
    append_override,
    build_autolabel_registry,
    create_logger,
    normalize_run_result,
    read_run_summary,
)
from share.config.config_loader import load_config, save_resolved_config
from share.config.editing import persist_config_overrides
from share.kernel.kernel import VisionKernel


AUTOLABEL_EDITABLE_PREFIXES = (
    "workspace",
    "data.labeled_dir",
    "data.unlabeled_dir",
    "train.device",
    "autolabel",
)


def build_autolabel_overrides_from_payload(payload: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    append_override(overrides, "workspace.run_name", payload.get("run_name"))
    append_override(overrides, "train.device", payload.get("device"))
    append_override(overrides, "data.labeled_dir", payload.get("labeled_dir"))
    append_override(overrides, "data.unlabeled_dir", payload.get("unlabeled_dir"))

    append_override(overrides, "autolabel.mode", payload.get("mode"))
    append_override(overrides, "autolabel.confidence", payload.get("confidence"))
    append_override(overrides, "autolabel.batch_size", payload.get("batch_size"))
    append_override(overrides, "autolabel.visualize", payload.get("visualize"))
    append_override(overrides, "autolabel.on_conflict", payload.get("on_conflict"))

    append_override(overrides, "autolabel.model.backend", payload.get("model_backend"))
    append_override(overrides, "autolabel.model.onnx_model", payload.get("model_onnx"))

    append_override(overrides, "autolabel.llm.base_url", payload.get("llm_base_url"))
    append_override(overrides, "autolabel.llm.model", payload.get("llm_model"))
    append_override(overrides, "autolabel.llm.api_key", payload.get("llm_api_key"))
    append_override(
        overrides,
        "autolabel.llm.api_key_env_name",
        payload.get("llm_api_key_env_name"),
    )
    append_override(overrides, "autolabel.llm.prompt", payload.get("llm_prompt"))
    append_override(overrides, "autolabel.llm.timeout_sec", payload.get("llm_timeout_sec"))
    append_override(overrides, "autolabel.llm.max_retries", payload.get("llm_max_retries"))
    append_override(
        overrides,
        "autolabel.llm.retry_backoff_sec",
        payload.get("llm_retry_backoff_sec"),
    )
    append_override(overrides, "autolabel.llm.qps_limit", payload.get("llm_qps_limit"))
    append_override(overrides, "autolabel.llm.max_images", payload.get("llm_max_images"))
    return overrides


def save_autolabel_config(config_path: Path | str, overrides: list[str]) -> dict[str, Any]:
    return persist_config_overrides(
        config_path=Path(config_path),
        overrides=overrides,
        allowed_prefixes=AUTOLABEL_EDITABLE_PREFIXES,
    )


def run_autolabel(
    config_path: Path | str,
    workdir_override: str | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    cfg = load_config(
        config_path=config_file,
        overrides=overrides or [],
        workdir_override=workdir_override,
    )

    logger = create_logger(cfg)
    logger.info("autolabel.service.start", "Autolabel run started", config_path=str(config_file))

    kernel = VisionKernel(cfg=cfg, logger=logger, registry=build_autolabel_registry())
    result = kernel.run_autolabel()

    resolved_path = result.run_context.run_dir / "config.resolved.toml"
    save_resolved_config(cfg, resolved_path)
    shutil.copyfile(resolved_path, Path(cfg["workspace"]["root"]) / "config.resolved.toml")

    logger.info(
        "autolabel.service.done",
        "Autolabel run finished",
        run_id=result.run_context.run_id,
        status=result.status,
        resolved_config=str(resolved_path),
    )
    return normalize_run_result(cfg, result)


def read_autolabel_run_summary(run_dir: Path | str) -> dict[str, Any]:
    return read_run_summary(run_dir)
