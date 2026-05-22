"""Application helpers for deploy workflows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from share.application.common import create_logger, normalize_run_result
from share.config.config_loader import load_config, save_resolved_config
from share.kernel.deploy.edge_llm import run_edge_llm_deploy
from share.kernel.deploy.edge_local import run_edge_local_deploy
from share.kernel.deploy.edge_stream import run_edge_stream_deploy
from share.kernel.kernel import VisionKernel
from share.kernel.registry import KernelRegistry


def build_edge_overrides_from_payload(payload: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    for key, value in payload.items():
        if value is None or key in {"overrides", "config_path", "workdir_override"}:
            continue
        if key.startswith("deploy.edge.") or key.startswith("workspace."):
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
