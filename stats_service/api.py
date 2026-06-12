"""Statistics service API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common.application.api_common import (
    json_response,
    require_api_key_or_bearer,
    require_bearer_token,
    resolve_api_token,
    validate_service_security,
)
from common.application.node_registration import (
    build_service_endpoint,
    start_control_plane_heartbeat,
)
from common.types.errors import ConfigError, DataValidationError
from common.types.stats import StatsEvent
from core.statistics.analytics import build_dashboard
from core.statistics.sqlite_store import get_recent_events, init_stats_db, insert_stats_event
from stats_service.config.schema import load_config


class SimpleRateLimiter:
    def __init__(self, max_per_sec: int) -> None:
        self.max_per_sec = max_per_sec
        self._window: dict[tuple[str, int], int] = {}

    def allow(self, source_id: str) -> bool:
        if self.max_per_sec <= 0:
            return True
        import time

        now_sec = int(time.time())
        key = (source_id, now_sec)
        self._window = {k: v for k, v in self._window.items() if k[1] >= now_sec - 1}
        count = self._window.get(key, 0)
        if count >= self.max_per_sec:
            return False
        self._window[key] = count + 1
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Statistics API service")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def create_app(role_cfg: dict[str, Any]):
    try:
        from fastapi import FastAPI, Header, Query
        from fastapi.responses import JSONResponse
    except Exception as exc:
        raise ConfigError(f"fastapi is required for statistics API: {exc}") from exc

    token = resolve_api_token(role_cfg["server"])
    stats_cfg = role_cfg["runtime"]
    db_path = Path(stats_cfg["db_path"])
    init_stats_db(db_path)
    limiter = SimpleRateLimiter(max_per_sec=int(stats_cfg["rate_limit_per_sec"]))
    app = FastAPI(title="Vision Statistics API", version="1.0.0")

    def auth(authorization: str | None):
        return require_bearer_token(authorization, token)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "statistics", "stats_db_path": str(db_path)}

    @app.get("/api/v1/status")
    def status(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {
            "ok": True,
            "role": "statistics",
            "service": "stats_service",
            "node_id": str(role_cfg["node"]["id"]),
            "endpoint": build_service_endpoint(role_cfg, role_cfg["server"]),
            "workspace_root": str(role_cfg["workspace"]["root"]),
            "stats_db_path": str(db_path),
        }

    @app.post("/api/v1/nodes/register")
    def register_node(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        from common.application.node_registration import register_with_control_plane

        return register_with_control_plane(
            role_cfg,
            role="statistics",
            service="stats_service",
            server_cfg=role_cfg["server"],
        )

    @app.post("/api/v1/push")
    def push_stats(
        payload: dict[str, Any],
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        denied = require_api_key_or_bearer(x_api_key, authorization, str(stats_cfg["api_key"]))
        if denied:
            return denied
        try:
            event = StatsEvent.from_dict(payload)
        except DataValidationError as exc:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "validation_failed", "detail": str(exc)},
            )
        if not limiter.allow(event.source_id):
            return JSONResponse(status_code=429, content={"ok": False, "error": "rate_limited"})
        insert_stats_event(db_path=db_path, event=event)
        return {"ok": True}

    @app.get("/api/v1/statistics/dashboard")
    def dashboard(
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
    def events(limit: int = Query(default=200), authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "events": get_recent_events(db_path, limit=limit)}

    @app.get("/api/v1/statistics/sources")
    def sources(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        events_payload = get_recent_events(db_path, limit=2000)
        return {"ok": True, "sources": sorted({str(e["source_id"]) for e in events_payload})}

    @app.get("/api/v1/config")
    def config(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "config": role_cfg}

    @app.get("/api/v1/jobs")
    def jobs(authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return {"ok": True, "jobs": []}

    @app.post("/api/v1/jobs")
    def create_job(_payload: dict[str, Any], authorization: str | None = Header(default=None)):
        denied = auth(authorization)
        if denied:
            return denied
        return json_response({"ok": False, "error": "unsupported", "detail": "stats_service has no jobs"}, 400)

    return app


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(
            Path(args.config).resolve(),
            overrides=args.set,
            workdir_override=str(Path(args.workdir).resolve()) if args.workdir else None,
        )
        validate_service_security("statistics", cfg["server"])
        start_control_plane_heartbeat(
            cfg,
            role="statistics",
            service="stats_service",
            server_cfg=cfg["server"],
        )
        app = create_app(cfg)
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
