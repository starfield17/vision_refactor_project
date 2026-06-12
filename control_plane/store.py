"""SQLite store for the control plane."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


class ControlPlaneStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.init()

    def init(self) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    endpoint TEXT NOT NULL DEFAULT '',
                    version TEXT NOT NULL DEFAULT '',
                    last_seen_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    target_role TEXT NOT NULL,
                    target_node_id TEXT NOT NULL DEFAULT '',
                    upstream_endpoint TEXT NOT NULL DEFAULT '',
                    upstream_job_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    finished_at_utc TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    log_path TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_control_jobs_kind ON jobs(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_control_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_control_jobs_node ON jobs(target_node_id)")

    def upsert_node(self, payload: dict[str, Any]) -> dict[str, Any]:
        node_id = str(payload.get("node_id") or payload.get("id") or "")
        role = str(payload.get("role") or "")
        if not node_id:
            raise ValueError("node_id is required")
        if not role:
            raise ValueError("role is required")
        status = str(payload.get("status") or "online")
        endpoint = str(payload.get("endpoint") or "")
        version = str(payload.get("version") or "")
        now = utc_now()
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id, role, status, endpoint, version, last_seen_utc, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    role = excluded.role,
                    status = excluded.status,
                    endpoint = excluded.endpoint,
                    version = excluded.version,
                    last_seen_utc = excluded.last_seen_utc,
                    payload_json = excluded.payload_json
                """,
                (
                    node_id,
                    role,
                    status,
                    endpoint,
                    version,
                    now,
                    json.dumps(payload, ensure_ascii=True),
                ),
            )
        return self.get_node(node_id) or {}

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def list_nodes(self) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM nodes ORDER BY role, node_id").fetchall()
        return [self._row_to_node(row) for row in rows]

    def create_job(
        self,
        *,
        kind: str,
        target_role: str,
        payload: dict[str, Any],
        target_node_id: str = "",
    ) -> dict[str, Any]:
        now = utc_now()
        job_id = uuid.uuid4().hex
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, target_role, target_node_id, status,
                    created_at_utc, updated_at_utc, payload_json
                )
                VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    job_id,
                    kind,
                    target_role,
                    target_node_id,
                    now,
                    now,
                    json.dumps(payload, ensure_ascii=True),
                ),
            )
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError("failed to create control-plane job")
        return job

    def attach_upstream(
        self,
        job_id: str,
        *,
        target_node_id: str,
        upstream_endpoint: str,
        upstream_job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str = "",
        log_path: str = "",
    ) -> dict[str, Any] | None:
        now = utc_now()
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                SET target_node_id = ?,
                    upstream_endpoint = ?,
                    upstream_job_id = ?,
                    status = ?,
                    updated_at_utc = ?,
                    result_json = ?,
                    error = ?,
                    log_path = ?
                WHERE job_id = ?
                """,
                (
                    target_node_id,
                    upstream_endpoint,
                    upstream_job_id,
                    status,
                    now,
                    json.dumps(result or {}, ensure_ascii=True),
                    error,
                    log_path,
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def update_job_status(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str = "",
        log_path: str = "",
        finished: bool = False,
    ) -> dict[str, Any] | None:
        now = utc_now()
        finished_at = now if finished else ""
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    updated_at_utc = ?,
                    finished_at_utc = CASE WHEN ? != '' THEN ? ELSE finished_at_utc END,
                    result_json = CASE WHEN ? != '' THEN ? ELSE result_json END,
                    error = ?,
                    log_path = CASE WHEN ? != '' THEN ? ELSE log_path END
                WHERE job_id = ?
                """,
                (
                    status,
                    now,
                    finished_at,
                    finished_at,
                    json.dumps(result or {}, ensure_ascii=True) if result is not None else "",
                    json.dumps(result or {}, ensure_ascii=True) if result is not None else "",
                    error,
                    log_path,
                    log_path,
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def list_jobs(self, limit: int = 100, kind: str = "", status: str = "") -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        clauses: list[str] = []
        params: list[Any] = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at_utc DESC LIMIT ?",
                (*params, safe_limit),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "node_id": str(row["node_id"]),
            "role": str(row["role"]),
            "status": str(row["status"]),
            "endpoint": str(row["endpoint"]),
            "version": str(row["version"]),
            "last_seen_utc": str(row["last_seen_utc"]),
            "payload": json.loads(str(row["payload_json"] or "{}")),
        }

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": str(row["job_id"]),
            "kind": str(row["kind"]),
            "target_role": str(row["target_role"]),
            "target_node_id": str(row["target_node_id"]),
            "upstream_endpoint": str(row["upstream_endpoint"]),
            "upstream_job_id": str(row["upstream_job_id"]),
            "status": str(row["status"]),
            "created_at_utc": str(row["created_at_utc"]),
            "updated_at_utc": str(row["updated_at_utc"]),
            "finished_at_utc": str(row["finished_at_utc"] or ""),
            "payload": json.loads(str(row["payload_json"] or "{}")),
            "result": json.loads(str(row["result_json"] or "{}")),
            "error": str(row["error"] or ""),
            "log_path": str(row["log_path"] or ""),
        }
