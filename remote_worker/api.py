"""Remote inference worker API service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common.application.api_common import (
    require_bearer_token,
    resolve_api_token,
    validate_service_security,
)
from common.config.config_loader import save_resolved_config
from common.config.role_schema import role_to_kernel_config
from common.application.node_registration import start_control_plane_heartbeat
from common.types.errors import ConfigError
from core.deploy.remote_server import create_remote_app
from core.infer.factory import create_frame_inferencer
from core.kernel import VisionKernel
from core.registry import KernelRegistry
from common.application.common import create_logger
from remote_worker.config.schema import load_config


def _gpu_summary() -> str:
    try:
        import torch
    except Exception:
        return "unavailable"
    try:
        if not torch.cuda.is_available():
            return "none"
        devices = []
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            devices.append(f"cuda:{index}:{device.name}:{device.total_memory}")
        return ",".join(devices) if devices else "none"
    except Exception:
        return "unknown"


def enrich_capabilities(role_cfg: dict[str, Any]) -> dict[str, Any]:
    capabilities = dict(role_cfg.get("capabilities", {}))
    runtime = role_cfg.get("runtime", {})
    capabilities["protocol"] = str(capabilities.get("protocol") or "frame_http")
    capabilities["model_path"] = str(runtime.get("model") or capabilities.get("model_path") or "")
    capabilities["model_id"] = str(
        capabilities.get("model_id") or f"remote:{Path(capabilities['model_path']).stem}"
    )
    capabilities["device"] = str(
        capabilities.get("device") or role_cfg.get("train", {}).get("device") or "auto"
    )
    if not capabilities.get("gpu") or capabilities["gpu"] == "unknown":
        capabilities["gpu"] = _gpu_summary()
    enriched = dict(role_cfg)
    enriched["capabilities"] = capabilities
    return enriched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote inference worker API")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def create_app(role_cfg: dict[str, Any]):
    from fastapi import Header

    kernel_cfg = role_to_kernel_config(role_cfg, "remote", "remote_worker")
    temp_path = Path(kernel_cfg["workspace"]["root"]) / "tmp" / "remote_worker.kernel.toml"
    save_resolved_config(kernel_cfg, temp_path)
    logger = create_logger(kernel_cfg)
    kernel = VisionKernel(cfg=kernel_cfg, logger=logger, registry=KernelRegistry())
    kernel._ensure_workdir_layout()
    run_ctx = kernel._make_run_context(mode="remote-worker")
    remote_cfg = kernel_cfg["deploy"]["remote"]
    resolved = create_frame_inferencer(
        model_path=Path(remote_cfg["model"]),
        cfg=kernel_cfg,
        confidence=float(remote_cfg["confidence"]),
        default_backend="yolo",
        default_model_id=f"remote:{Path(remote_cfg['model']).stem}",
    )
    app = create_remote_app(
        cfg=kernel_cfg,
        run_id=run_ctx.run_id,
        run_dir=run_ctx.run_dir,
        logger=logger,
        inferencer=resolved.inferencer,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "remote_worker", "run_id": run_ctx.run_id}

    @app.get("/api/v1/status")
    def status(authorization: str | None = Header(default=None)):
        token = resolve_api_token(
            {"api_token": role_cfg["runtime"]["ingest_api_key"], "api_token_env_name": ""}
        )
        denied = require_bearer_token(authorization, token)
        if denied:
            return denied
        return {
            "ok": True,
            "role": "remote",
            "service": "remote_worker",
            "node_id": str(role_cfg["node"]["id"]),
            "endpoint": str(role_cfg["node"].get("endpoint") or ""),
            "workspace_root": str(role_cfg["workspace"]["root"]),
            "run_id": run_ctx.run_id,
        }

    return app


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        role_cfg = load_config(
            Path(args.config).resolve(),
            overrides=args.set,
            workdir_override=str(Path(args.workdir).resolve()) if args.workdir else None,
        )
        role_cfg = enrich_capabilities(role_cfg)
        server_cfg = {
            "host": role_cfg["runtime"]["listen_host"],
            "port": role_cfg["runtime"]["listen_port"],
            "api_token": role_cfg["runtime"]["ingest_api_key"],
            "api_token_env_name": "",
        }
        validate_service_security("remote_worker", server_cfg)
        start_control_plane_heartbeat(
            role_cfg,
            role="remote",
            service="remote_worker",
            server_cfg=server_cfg,
        )
        app = create_app(role_cfg)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except Exception as exc:
        print(f"[RUNTIME ERROR] uvicorn is required: {exc}", file=sys.stderr)
        return 3

    uvicorn.run(
        app,
        host=str(role_cfg["runtime"]["listen_host"]),
        port=int(role_cfg["runtime"]["listen_port"]),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
