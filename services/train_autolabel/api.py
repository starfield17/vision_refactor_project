"""Train/autolabel backend API service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from share.application.api_common import (
    json_response,
    read_tail,
    require_bearer_token,
    resolve_api_token,
    validate_service_security,
)
from share.application.autolabel_service import (
    build_autolabel_overrides_from_payload,
    save_autolabel_config,
)
from share.application.job_runner import SubprocessJobRunner
from share.application.job_store import JobStore
from share.application.train_service import build_train_overrides_from_payload, save_train_config
from share.config.config_loader import load_config
from share.config.editing import load_merged_user_config
from share.types.errors import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/autolabel API service")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", required=True, help="Path to config.toml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def _payload_to_job_payload(
    payload: dict[str, Any],
    config_path: Path,
    workdir_override: str | None,
    overrides: list[str],
) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "workdir_override": workdir_override,
        "overrides": overrides,
        "request": payload,
    }


def create_app(cfg: dict[str, Any], config_path: Path, workdir_override: str | None = None):
    try:
        from fastapi import FastAPI, Header, Query
    except Exception as exc:
        raise ConfigError(f"fastapi is required for train_autolabel API: {exc}") from exc

    service_cfg = cfg["services"]["train_autolabel"]
    token = resolve_api_token(service_cfg)
    store = JobStore(Path(service_cfg["job_db_path"]))
    store.mark_interrupted_running_jobs()
    log_dir = Path(cfg["workspace"]["root"]) / "tmp" / "jobs" / "train_autolabel"
    runner = SubprocessJobRunner(
        job_store=store,
        log_dir=log_dir,
        worker_module="services.train_autolabel.worker",
    )

    app = FastAPI(title="Vision Train/Autolabel API", version="1.0.0")
    try:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception:
        pass

    def auth(authorization: str | None):
        return require_bearer_token(authorization, token)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "train_autolabel", "job_db_path": str(store.db_path)}

    @app.get("/api/v1/config")
    def get_config(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        try:
            return {"ok": True, "config": load_merged_user_config(config_path)}
        except ConfigError as exc:
            return json_response({"ok": False, "error": "config_error", "detail": str(exc)}, 400)

    @app.patch("/api/v1/config")
    def patch_config(payload: dict[str, Any], authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        area = str(payload.get("area", ""))
        overrides = [str(item) for item in payload.get("overrides", [])]
        try:
            if area == "train":
                updated = save_train_config(config_path=config_path, overrides=overrides)
            elif area == "autolabel":
                updated = save_autolabel_config(config_path=config_path, overrides=overrides)
            else:
                return json_response(
                    {"ok": False, "error": "validation_failed", "detail": "area must be train or autolabel"},
                    400,
                )
        except ConfigError as exc:
            return json_response({"ok": False, "error": "config_error", "detail": str(exc)}, 400)
        return {"ok": True, "config": updated}

    def submit_job(kind: str, payload: dict[str, Any], authorization: str | None):
        denied = auth(authorization)
        if denied:
            return denied
        if store.has_active_job():
            return json_response(
                {"ok": False, "error": "busy", "detail": "another train/autolabel job is active"},
                409,
            )
        extra_overrides = [str(item) for item in payload.get("overrides", [])]
        if kind == "train":
            overrides = extra_overrides + build_train_overrides_from_payload(payload)
        else:
            overrides = extra_overrides + build_autolabel_overrides_from_payload(payload)
        job_payload = _payload_to_job_payload(
            payload=payload,
            config_path=config_path,
            workdir_override=workdir_override,
            overrides=overrides,
        )
        job = store.create_job(kind=kind, payload=job_payload)
        job = runner.start(job)
        return {"ok": True, "job": job}

    @app.post("/api/v1/train/jobs")
    def submit_train(payload: dict[str, Any], authorization: str | None = Header(default=None)):
        return submit_job("train", payload, authorization)

    @app.post("/api/v1/autolabel/jobs")
    def submit_autolabel(payload: dict[str, Any], authorization: str | None = Header(default=None)):
        return submit_job("autolabel", payload, authorization)

    @app.get("/api/v1/jobs")
    def list_jobs(
        limit: int = Query(default=100),
        kind: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "jobs": store.list_jobs(limit=limit, kind=kind)}

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
        return {
            "ok": True,
            "job_id": job_id,
            "log_path": job["log_path"],
            "text": read_tail(job["log_path"], max_lines=tail),
        }

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        ok = runner.cancel(job_id)
        job = store.get_job(job_id)
        return {"ok": ok, "job": job}

    return app


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None
    try:
        cfg = load_config(
            config_path=config_path,
            overrides=args.set,
            workdir_override=workdir_override,
        )
        validate_service_security("train_autolabel", cfg["services"]["train_autolabel"])
        app = create_app(cfg=cfg, config_path=config_path, workdir_override=workdir_override)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except Exception as exc:
        print(f"[RUNTIME ERROR] uvicorn is required: {exc}", file=sys.stderr)
        return 3

    service_cfg = cfg["services"]["train_autolabel"]
    uvicorn.run(
        app,
        host=str(service_cfg["host"]),
        port=int(service_cfg["port"]),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
