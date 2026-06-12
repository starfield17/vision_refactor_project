"""Application helpers for deploy workflows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from common.application.common import append_override, create_logger, normalize_run_result
from common.config.config_loader import load_config, save_resolved_config
from core.deploy.edge_llm import run_edge_llm_deploy
from core.deploy.edge_local import run_edge_local_deploy
from core.deploy.edge_locate_anything import run_edge_locate_anything_deploy
from core.deploy.edge_stream import run_edge_stream_deploy
from core.kernel import VisionKernel
from core.registry import KernelRegistry


def build_edge_overrides_from_payload(payload: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    append_override(overrides, "workspace.run_name", payload.get("run_name"))
    for field in (
        "source_id",
        "mode",
        "source",
        "camera_id",
        "video_path",
        "images_dir",
        "fps_limit",
        "jpeg_quality",
        "confidence",
        "local_model",
        "stats_endpoint",
        "api_key",
        "stats_timeout_sec",
        "save_annotated",
        "max_frames",
        "stream_endpoint",
        "stream_timeout_sec",
        "stream_api_key",
    ):
        append_override(overrides, f"runtime.{field}", payload.get(field))
    for field in (
        "base_url",
        "model",
        "api_key",
        "api_key_env_name",
        "prompt",
        "timeout_sec",
        "max_retries",
        "retry_backoff_sec",
        "qps_limit",
    ):
        append_override(overrides, f"runtime.llm.{field}", payload.get(f"llm_{field}"))
    for field in (
        "model",
        "device",
        "dtype",
        "quantization",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
        "device_map",
        "attn_implementation",
        "generation_mode",
        "max_new_tokens",
        "temperature",
        "prompt_template",
        "nms_iou",
        "default_score",
        "verbose",
        "max_images",
    ):
        append_override(overrides, f"locate_anything.{field}", payload.get(f"locate_anything_{field}"))
    for key, value in payload.items():
        if value is None or key in {"overrides", "config_path", "workdir_override"}:
            continue
        if key.startswith("runtime.") or key.startswith("workspace."):
            raw = "true" if value is True else "false" if value is False else str(value)
            overrides.append(f"{key}={raw}")
    return overrides


def run_deploy_edge(
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
    logger.info("deploy.edge.service.start", "Deploy edge run started", config_path=str(config_file))

    registry = KernelRegistry()
    registry.register_deployer("local", run_edge_local_deploy)
    registry.register_deployer("llm", run_edge_llm_deploy)
    registry.register_deployer("stream", run_edge_stream_deploy)
    registry.register_deployer("locate_anything", run_edge_locate_anything_deploy)

    kernel = VisionKernel(cfg=cfg, logger=logger, registry=registry)
    result = kernel.run_deploy_edge()

    resolved_path = result.run_context.run_dir / "config.resolved.toml"
    save_resolved_config(cfg, resolved_path)
    shutil.copyfile(resolved_path, Path(cfg["workspace"]["root"]) / "config.resolved.toml")

    logger.info(
        "deploy.edge.service.done",
        "Deploy edge run finished",
        run_id=result.run_context.run_id,
        status=result.status,
        resolved_config=str(resolved_path),
    )
    return normalize_run_result(cfg, result)
