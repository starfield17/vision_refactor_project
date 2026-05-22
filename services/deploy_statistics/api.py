"""Deploy/statistics backend API service."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

from share.application.api_common import (
    json_response,
    read_tail,
    require_api_key_or_bearer,
    require_bearer_token,
    resolve_api_token,
    validate_service_security,
)
from share.application.deploy_service import build_edge_overrides_from_payload
from share.application.job_runner import SubprocessJobRunner
from share.application.job_store import JobStore
from share.config.config_loader import load_config
from share.config.editing import load_merged_user_config, persist_config_overrides
from share.kernel.deploy.remote_server import create_remote_app
from share.kernel.infer.local_yolo import LocalYoloInferencer
from share.kernel.statistics.analytics import build_dashboard
from share.kernel.statistics.sqlite_store import get_recent_events, init_stats_db, insert_stats_event
from share.kernel.utils.logging import StructuredLogger
from share.types.errors import ConfigError, DataValidationError
from share.types.stats import StatsEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy/statistics API service")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", required=True, help="Path to config.toml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def _resolve_log_path(cfg: dict[str, Any]) -> Path:
    workdir = Path(cfg["workspace"]["root"])
    log_file = Path(cfg["workspace"]["log_file"])
    if log_file.is_absolute():
        return log_file
    return workdir / log_file


class RemoteRuntime:
    def __init__(self, cfg: dict[str, Any], logger: StructuredLogger) -> None:
        self.cfg = cfg
        self.logger = logger
        self._lock = threading.Lock()
        self._app = None
        self._run_id = ""
        self._run_dir: Path | None = None

    def start(self):
        with self._lock:
            if self._app is not None:
                return {"ok": True, "status": "running", "run_id": self._run_id}
            from share.kernel.kernel import VisionKernel
            from share.kernel.registry import KernelRegistry

            kernel = VisionKernel(cfg=self.cfg, logger=self.logger, registry=KernelRegistry())
            kernel._ensure_workdir_layout()
            run_ctx = kernel._make_run_context(mode="deploy-remote")
            remote_cfg = self.cfg["deploy"]["remote"]
            inferencer = LocalYoloInferencer(
                model_path=Path(remote_cfg["model"]),
                class_names=list(self.cfg["class_map"]["names"]),
                confidence=float(remote_cfg["confidence"]),
                img_size=int(self.cfg["train"]["img_size"]),
                device=str(self.cfg["train"]["device"]),
            )
            self._app = create_remote_app(
                cfg=self.cfg,
                run_id=run_ctx.run_id,
                run_dir=run_ctx.run_dir,
                logger=self.logger,
                inferencer=inferencer,
            )
            self._run_id = run_ctx.run_id
            self._run_dir = run_ctx.run_dir
            return {"ok": True, "status": "running", "run_id": self._run_id}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._app = None
            old_run_id = self._run_id
            self._run_id = ""
            self._run_dir = None
            return {"ok": True, "status": "stopped", "run_id": old_run_id}

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._app is None:
                return {"ok": True, "status": "stopped", "run_id": ""}
            counters = dict(getattr(self._app.state, "remote_counters", {}))
            return {"ok": True, "status": "running", "run_id": self._run_id, **counters}

    def frame(self, payload: dict[str, Any], x_api_key: str | None):
        with self._lock:
            app = self._app
        if app is None:
            return json_response(
                {"ok": False, "error": "remote_not_running"},
                status_code=409,
            )
        endpoint = next(
            route.endpoint for route in app.routes if getattr(route, "path", "") == "/api/v1/frame"
        )
        return endpoint(payload, x_api_key=x_api_key)


class _SimpleRateLimiter:
    def __init__(self, max_per_sec: int) -> None:
        self.max_per_sec = max_per_sec
        self._lock = threading.Lock()
        self._window: dict[tuple[str, int], int] = {}

    def allow(self, source_id: str) -> bool:
        if self.max_per_sec <= 0:
            return True
        now_sec = int(time.time())
        key = (source_id, now_sec)
        with self._lock:
            self._window = {k: v for k, v in self._window.items() if k[1] >= now_sec - 1}
            count = self._window.get(key, 0)
            if count >= self.max_per_sec:
                return False
            self._window[key] = count + 1
            return True


def create_app(cfg: dict[str, Any], config_path: Path, workdir_override: str | None = None):
    try:
        from fastapi import FastAPI, Header, Query
        from fastapi.responses import JSONResponse
    except Exception as exc:
        raise ConfigError(f"fastapi is required for deploy_statistics API: {exc}") from exc

    service_cfg = cfg["services"]["deploy_statistics"]
    token = resolve_api_token(service_cfg)
    store = JobStore(Path(service_cfg["job_db_path"]))
    store.mark_interrupted_running_jobs()
    log_dir = Path(cfg["workspace"]["root"]) / "tmp" / "jobs" / "deploy_statistics"
    runner = SubprocessJobRunner(
        job_store=store,
        log_dir=log_dir,
        worker_module="services.deploy_statistics.worker",
    )

    stats_cfg = cfg["deploy"]["statistics"]
    db_path = Path(stats_cfg["db_path"])
    init_stats_db(db_path)
    logger = StructuredLogger(log_path=_resolve_log_path(cfg), level=cfg["workspace"]["log_level"])
    remote_runtime = RemoteRuntime(cfg=cfg, logger=logger)
    stats_limiter = _SimpleRateLimiter(max_per_sec=int(stats_cfg["rate_limit_per_sec"]))

    app = FastAPI(title="Vision Deploy/Statistics API", version="1.0.0")
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
        return {
            "ok": True,
            "service": "deploy_statistics",
            "job_db_path": str(store.db_path),
            "stats_db_path": str(db_path),
        }

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
        overrides = [str(item) for item in payload.get("overrides", [])]
        try:
            updated = persist_config_overrides(
                config_path=config_path,
                overrides=overrides,
                allowed_prefixes=("workspace", "deploy"),
            )
        except ConfigError as exc:
            return json_response({"ok": False, "error": "config_error", "detail": str(exc)}, 400)
        return {"ok": True, "config": updated}

    @app.post("/api/v1/deploy/edge/jobs")
    def submit_edge(payload: dict[str, Any], authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        if store.has_active_job():
            return json_response(
                {"ok": False, "error": "busy", "detail": "another deploy job is active"},
                409,
            )
        overrides = [str(item) for item in payload.get("overrides", [])]
        overrides.extend(build_edge_overrides_from_payload(payload))
        job_payload = {
            "config_path": str(config_path),
            "workdir_override": workdir_override,
            "overrides": overrides,
            "request": payload,
        }
        job = store.create_job(kind="deploy_edge", payload=job_payload)
        job = runner.start(job)
        return {"ok": True, "job": job}

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

    @app.post("/api/v1/push")
    def push_stats(
        payload: dict[str, Any],
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        expected_api_key = str(stats_cfg["api_key"])
        denied = require_api_key_or_bearer(x_api_key, authorization, expected_api_key)
        if denied:
            return denied
        try:
            event = StatsEvent.from_dict(payload)
        except DataValidationError as exc:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "validation_failed", "detail": str(exc)},
            )
        if not stats_limiter.allow(event.source_id):
            return JSONResponse(status_code=429, content={"ok": False, "error": "rate_limited"})
        insert_stats_event(db_path=db_path, event=event)
        return {"ok": True}

    @app.get("/api/v1/statistics/dashboard")
    def statistics_dashboard(
        limit: int = Query(default=200),
        source_id: str = Query(default=""),
        min_detections: int = Query(default=0),
        authorization: str | None = Header(default=None),
    ):
        denied = auth(authorization)
        if denied:
            return denied
        return {
            "ok": True,
            "dashboard": build_dashboard(
                db_path=db_path,
                limit=limit,
                source_id=source_id,
                min_detections=min_detections,
            ),
        }

    @app.get("/api/v1/statistics/events")
    def statistics_events(
        limit: int = Query(default=200),
        authorization: str | None = Header(default=None),
    ):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "events": get_recent_events(db_path, limit=limit)}

    @app.get("/api/v1/statistics/sources")
    def statistics_sources(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        events = get_recent_events(db_path, limit=2000)
        return {"ok": True, "sources": sorted({str(e["source_id"]) for e in events})}

    @app.post("/api/v1/deploy/remote/start")
    def remote_start(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        try:
            return remote_runtime.start()
        except Exception as exc:
            return json_response({"ok": False, "error": "remote_start_failed", "detail": str(exc)}, 500)

    @app.post("/api/v1/deploy/remote/stop")
    def remote_stop(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return remote_runtime.stop()

    @app.get("/api/v1/deploy/remote/status")
    def remote_status(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return remote_runtime.status()

    @app.post("/api/v1/frame")
    def push_frame(payload: dict[str, Any], x_api_key: str | None = Header(default=None)):
        return remote_runtime.frame(payload, x_api_key=x_api_key)

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
        validate_service_security("deploy_statistics", cfg["services"]["deploy_statistics"])
        app = create_app(cfg=cfg, config_path=config_path, workdir_override=workdir_override)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except Exception as exc:
        print(f"[RUNTIME ERROR] uvicorn is required: {exc}", file=sys.stderr)
        return 3

    service_cfg = cfg["services"]["deploy_statistics"]
    uvicorn.run(
        app,
        host=str(service_cfg["host"]),
        port=int(service_cfg["port"]),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
