"""SQLite-backed job state for long-running backend services."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_STATUSES = {"succeeded", "failed", "cancelled", "interrupted"}
ACTIVE_STATUSES = {"queued", "running"}


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": str(row["job_id"]),
        "kind": str(row["kind"]),
        "status": str(row["status"]),
        "created_at_utc": str(row["created_at_utc"]),
        "updated_at_utc": str(row["updated_at_utc"]),
        "started_at_utc": str(row["started_at_utc"] or ""),
        "finished_at_utc": str(row["finished_at_utc"] or ""),
        "payload": json.loads(str(row["payload_json"] or "{}")),
        "result": json.loads(str(row["result_json"] or "{}")),
        "error": str(row["error"] or ""),
        "run_id": str(row["run_id"] or ""),
        "run_dir": str(row["run_dir"] or ""),
        "log_path": str(row["log_path"] or ""),
        "worker_pid": int(row["worker_pid"] or 0),
    }


class JobStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.init()

    def init(self) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    started_at_utc TEXT NOT NULL DEFAULT '',
                    finished_at_utc TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    run_id TEXT NOT NULL DEFAULT '',
                    run_dir TEXT NOT NULL DEFAULT '',
                    log_path TEXT NOT NULL DEFAULT '',
                    worker_pid INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at_utc)")

    def mark_interrupted_running_jobs(self) -> int:
        now = utc_now()
        with _connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'interrupted',
                    updated_at_utc = ?,
                    finished_at_utc = ?,
                    error = CASE WHEN error = '' THEN 'Service restarted while job was active' ELSE error END,
                    worker_pid = 0
                WHERE status IN ('queued', 'running')
                """,
                (now, now),
            )
            return int(cur.rowcount)

    def create_job(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        job_id = uuid.uuid4().hex
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, status, created_at_utc, updated_at_utc, payload_json
                )
                VALUES (?, ?, 'queued', ?, ?, ?)
                """,
                (job_id, kind, now, now, json.dumps(payload, ensure_ascii=True)),
            )
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError("failed to create job")
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(self, limit: int = 100, kind: str | None = None) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        with _connect(self.db_path) as conn:
            if kind:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE kind = ?
                    ORDER BY created_at_utc DESC
                    LIMIT ?
                    """,
                    (kind, safe_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at_utc DESC LIMIT ?",
                    (safe_limit,),
                ).fetchall()
        return [_row_to_job(row) for row in rows]

    def has_active_job(self) -> bool:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchone()
        return bool(row and int(row[0]) > 0)

    def mark_running(
        self,
        job_id: str,
        worker_pid: int,
        log_path: Path | str,
    ) -> None:
        now = utc_now()
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    started_at_utc = CASE WHEN started_at_utc = '' THEN ? ELSE started_at_utc END,
                    updated_at_utc = ?,
                    worker_pid = ?,
                    log_path = ?
                WHERE job_id = ?
                """,
                (now, now, int(worker_pid), str(log_path), job_id),
            )

    def update_running_metadata(
        self,
        job_id: str,
        run_id: str = "",
        run_dir: Path | str = "",
        log_path: Path | str = "",
    ) -> None:
        now = utc_now()
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                SET updated_at_utc = ?,
                    run_id = CASE WHEN ? != '' THEN ? ELSE run_id END,
                    run_dir = CASE WHEN ? != '' THEN ? ELSE run_dir END,
                    log_path = CASE WHEN ? != '' THEN ? ELSE log_path END
                WHERE job_id = ?
                """,
                (
                    now,
                    run_id,
                    run_id,
                    str(run_dir),
                    str(run_dir),
                    str(log_path),
                    str(log_path),
                    job_id,
                ),
            )

    def finish_job(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str = "",
        run_id: str = "",
        run_dir: Path | str = "",
        log_path: Path | str = "",
    ) -> None:
        if status not in FINAL_STATUSES:
            raise ValueError(f"invalid final status={status}")
        now = utc_now()
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    updated_at_utc = ?,
                    finished_at_utc = ?,
                    result_json = ?,
                    error = ?,
                    run_id = CASE WHEN ? != '' THEN ? ELSE run_id END,
                    run_dir = CASE WHEN ? != '' THEN ? ELSE run_dir END,
                    log_path = CASE WHEN ? != '' THEN ? ELSE log_path END,
                    worker_pid = 0
                WHERE job_id = ?
                """,
                (
                    status,
                    now,
                    now,
                    json.dumps(result or {}, ensure_ascii=True),
                    error,
                    run_id,
                    run_id,
                    str(run_dir),
                    str(run_dir),
                    str(log_path),
                    str(log_path),
                    job_id,
                ),
            )

    def request_cancel(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None or job["status"] not in ACTIVE_STATUSES:
            return False
        self.finish_job(job_id, status="cancelled", error="Cancelled by user")
        return True
