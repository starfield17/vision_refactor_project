"""Statistics ingest API (Phase 4)."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

from share.config.config_loader import load_config
from share.kernel.statistics.sqlite_store import init_stats_db, insert_stats_event
from share.kernel.utils.logging import StructuredLogger
from share.types.errors import ConfigError, DataValidationError
from share.types.stats import StatsEvent


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


def _resolve_log_path(cfg: dict[str, Any]) -> Path:
    workdir = Path(cfg["workspace"]["root"])
    log_file = Path(cfg["workspace"]["log_file"])
    if log_file.is_absolute():
        return log_file
    return workdir / log_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4 statistics ingest API")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", required=True, help="Path to config.toml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, repeatable.",
    )
    return parser


def create_app(
    db_path: Path,
    logger: StructuredLogger,
    expected_api_key: str,
    rate_limit_per_sec: int,
):
    try:
        from fastapi import FastAPI, Header
        from fastapi.responses import JSONResponse
    except Exception as exc:
        raise ConfigError(f"fastapi is required for statistics API: {exc}") from exc

    limiter = _SimpleRateLimiter(max_per_sec=rate_limit_per_sec)
    app = FastAPI(title="Vision Refactor Statistics API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "storage": "sqlite", "db_path": str(db_path)}

    @app.post("/api/v1/push")
    def push_stats(payload: dict[str, Any], x_api_key: str | None = Header(default=None)):
        if expected_api_key and x_api_key != expected_api_key:
            return JSONResponse(
                status_code=401,
                content={"ok": False, "error": "unauthorized"},
            )

        try:
            event = StatsEvent.from_dict(payload)
        except DataValidationError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "validation_failed",
                    "detail": str(exc),
                },
            )

        if not limiter.allow(event.source_id):
            return JSONResponse(
                status_code=429,
                content={"ok": False, "error": "rate_limited"},
            )

        insert_stats_event(db_path=db_path, event=event)
        logger.info(
            "statistics.api.push.ok",
            "Stats ingested",
            source_id=event.source_id,
            total_detections=event.total_detections,
            latency_ms=event.latency_ms,
        )
        return {"ok": True}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None

    try:
        cfg = load_config(
            config_path=config_path,
            overrides=args.set,
            workdir_override=workdir_override,
        )
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    stats_cfg = cfg["deploy"]["statistics"]
    db_path = Path(stats_cfg["db_path"])
    init_stats_db(db_path)

    logger = StructuredLogger(
        log_path=_resolve_log_path(cfg),
        level=cfg["workspace"]["log_level"],
    )

    try:
        app = create_app(
            db_path=db_path,
            logger=logger,
            expected_api_key=str(stats_cfg["api_key"]),
            rate_limit_per_sec=int(stats_cfg["rate_limit_per_sec"]),
        )
    except ConfigError as exc:
        print(f"[RUNTIME ERROR] {exc}", file=sys.stderr)
        return 3

    try:
        import uvicorn
    except Exception as exc:
        print(f"[RUNTIME ERROR] uvicorn is required: {exc}", file=sys.stderr)
        return 3

    host = str(stats_cfg["public_host"])
    port = int(stats_cfg["api_port"])
    logger.info("statistics.api.start", "Statistics API starting", host=host, port=port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
