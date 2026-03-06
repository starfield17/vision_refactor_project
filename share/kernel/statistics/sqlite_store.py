"""SQLite storage for statistics ingest and UI."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from share.types.stats import StatsEvent


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def init_stats_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stats_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_received_utc TEXT NOT NULL,
                ts_utc TEXT NOT NULL,
                source_id TEXT NOT NULL,
                request_id TEXT NOT NULL DEFAULT '',
                run_id TEXT NOT NULL DEFAULT '',
                model_id TEXT NOT NULL DEFAULT '',
                backend TEXT NOT NULL DEFAULT '',
                transport_mode TEXT NOT NULL DEFAULT '',
                total_detections INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                counts_by_class_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        existing = _table_columns(conn, "stats_events")
        required_columns = {
            "request_id": "TEXT NOT NULL DEFAULT ''",
            "run_id": "TEXT NOT NULL DEFAULT ''",
            "model_id": "TEXT NOT NULL DEFAULT ''",
            "backend": "TEXT NOT NULL DEFAULT ''",
            "transport_mode": "TEXT NOT NULL DEFAULT ''",
        }
        for column, ddl in required_columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE stats_events ADD COLUMN {column} {ddl}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stats_events_ts_utc ON stats_events(ts_utc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stats_events_source_id ON stats_events(source_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stats_events_request_id ON stats_events(request_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stats_events_run_id ON stats_events(run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stats_events_model_id ON stats_events(model_id)"
        )


def insert_stats_event(db_path: Path, event: StatsEvent) -> None:
    payload = event.to_dict()
    now_utc = datetime.now(tz=timezone.utc).isoformat()

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO stats_events (
                ts_received_utc,
                ts_utc,
                source_id,
                request_id,
                run_id,
                model_id,
                backend,
                transport_mode,
                total_detections,
                latency_ms,
                counts_by_class_json,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_utc,
                payload["ts_utc"],
                payload["source_id"],
                payload.get("request_id", ""),
                payload.get("run_id", ""),
                payload.get("model_id", ""),
                payload.get("backend", ""),
                payload.get("transport_mode", ""),
                int(payload["total_detections"]),
                float(payload["latency_ms"]),
                json.dumps(payload["counts_by_class"], ensure_ascii=True),
                json.dumps(payload, ensure_ascii=True),
            ),
        )


def get_overview(db_path: Path) -> dict[str, Any]:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS events_total,
                COALESCE(SUM(total_detections), 0) AS detections_total,
                COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                MAX(ts_utc) AS last_event_ts_utc
            FROM stats_events
            """
        ).fetchone()

        sources = conn.execute("SELECT COUNT(DISTINCT source_id) FROM stats_events").fetchone()

    return {
        "events_total": int(row[0] if row else 0),
        "detections_total": int(row[1] if row else 0),
        "avg_latency_ms": float(row[2] if row else 0.0),
        "last_event_ts_utc": str(row[3]) if row and row[3] is not None else "",
        "source_count": int(sources[0] if sources else 0),
    }


def get_recent_events(db_path: Path, limit: int) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 2000))
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                ts_received_utc,
                ts_utc,
                source_id,
                request_id,
                run_id,
                model_id,
                backend,
                transport_mode,
                total_detections,
                latency_ms,
                counts_by_class_json
            FROM stats_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    events: list[dict[str, Any]] = []
    for row in rows:
        counts_by_class = json.loads(row[11]) if row[11] else {}
        events.append(
            {
                "id": int(row[0]),
                "ts_received_utc": str(row[1]),
                "ts_utc": str(row[2]),
                "source_id": str(row[3]),
                "request_id": str(row[4]),
                "run_id": str(row[5]),
                "model_id": str(row[6]),
                "backend": str(row[7]),
                "transport_mode": str(row[8]),
                "total_detections": int(row[9]),
                "latency_ms": float(row[10]),
                "counts_by_class": counts_by_class,
            }
        )
    return events


def get_class_totals(db_path: Path, limit: int = 500) -> dict[str, int]:
    totals: dict[str, int] = {}
    for event in get_recent_events(db_path, limit=limit):
        for name, count in event["counts_by_class"].items():
            totals[name] = totals.get(name, 0) + int(count)
    return totals
