"""Train worker API service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common.application.api_common import (
    json_response,
    read_tail,
    require_bearer_token,
    resolve_api_token,
    validate_service_security,
)
from common.application.job_runner import SubprocessJobRunner
from common.application.job_store import JobStore
from common.application.node_registration import (
    build_service_endpoint,
    start_control_plane_heartbeat,
)
from common.application.train_service import build_train_overrides_from_payload
from common.config.config_loader import to_toml
from common.types.errors import ConfigError
from train_worker.config.schema import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train worker API service")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def create_app(role_cfg: dict[str, Any], config_path: Path, workdir_override: str | None = None):
    try:
        from fastapi import FastAPI, Header, Query
    except Exception as exc:
        raise ConfigError(f"fastapi is required for train worker API: {exc}") from exc

    server_cfg = role_cfg["server"]
    token = resolve_api_token(server_cfg)
    store = JobStore(Path(role_cfg["job_store"]["db_path"]))
    store.mark_interrupted_running_jobs()
    log_dir = Path(role_cfg["workspace"]["root"]) / "tmp" / "jobs" / "train_worker"
    runner = SubprocessJobRunner(
        job_store=store,
        log_dir=log_dir,
        worker_module="train_worker.worker",
    )

    app = FastAPI(title="Vision Train Worker API", version="1.0.0")

    def auth(authorization: str | None):
        return require_bearer_token(authorization, token)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "train_worker", "job_db_path": str(store.db_path)}

    @app.get("/api/v1/status")
    def status(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        jobs = store.list_jobs(limit=100)
        active = [job for job in jobs if job["status"] in {"queued", "running"}]
        counts: dict[str, int] = {}
        for job in jobs:
            counts[str(job["status"])] = counts.get(str(job["status"]), 0) + 1
        return {
            "ok": True,
            "role": "train_worker",
            "service": "train_worker",
            "node_id": str(role_cfg["node"]["id"]),
            "endpoint": build_service_endpoint(role_cfg, server_cfg),
            "workspace_root": str(role_cfg["workspace"]["root"]),
            "job_db_path": str(store.db_path),
            "active_job": active[0] if active else None,
            "job_counts": counts,
        }

    @app.post("/api/v1/nodes/register")
    def register_node(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        from common.application.node_registration import register_with_control_plane

        return register_with_control_plane(
            role_cfg,
            role="train_worker",
            service="train_worker",
            server_cfg=server_cfg,
        )

    @app.get("/api/v1/config")
    def get_config(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "config": role_cfg}

    @app.get("/api/v1/config.toml")
    def get_config_toml(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "text": to_toml(role_cfg)}

    @app.post("/api/v1/jobs")
    def submit_train(payload: dict[str, Any], authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        if store.has_active_job():
            return json_response({"ok": False, "error": "busy"}, 409)
        overrides = [str(item) for item in payload.get("overrides", [])]
        overrides.extend(build_train_overrides_from_payload(payload))
        job = store.create_job(
            kind="train",
            payload={
                "config_path": str(config_path),
                "workdir_override": workdir_override,
                "overrides": overrides,
                "request": payload,
            },
        )
        return {"ok": True, "job": runner.start(job)}

    @app.get("/api/v1/jobs")
    def list_jobs(
        limit: int = Query(default=100),
        authorization: str | None = Header(default=None),
    ):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "jobs": store.list_jobs(limit=limit)}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        job = store.get_job(job_id)
        if job is None:
            return json_response({"ok": False, "error": "not_found"}, 404)
        return {"ok": True, "job": job}

    @app.get("/api/v1/jobs/{job_id}/logs")
    def get_job_logs(
        job_id: str,
        tail: int = Query(default=200),
        authorization: str | None = Header(default=None),
    ):
        denied = auth(authorization)
        if denied:
            return denied
        job = store.get_job(job_id)
        if job is None:
            return json_response({"ok": False, "error": "not_found"}, 404)
        return {"ok": True, "job_id": job_id, "text": read_tail(job["log_path"], tail)}

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        ok = runner.cancel(job_id)
        return {"ok": ok, "job": store.get_job(job_id)}

    return app


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None
    try:
        cfg = load_config(config_path, overrides=args.set, workdir_override=workdir_override)
        validate_service_security("train_worker", cfg["server"])
        start_control_plane_heartbeat(
            cfg,
            role="train_worker",
            service="train_worker",
            server_cfg=cfg["server"],
        )
        app = create_app(cfg, config_path=config_path, workdir_override=workdir_override)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except Exception as exc:
        print(f"[RUNTIME ERROR] uvicorn is required: {exc}", file=sys.stderr)
        return 3

    uvicorn.run(app, host=str(cfg["server"]["host"]), port=int(cfg["server"]["port"]), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
