"""Subprocess job runner used by backend service daemons."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from common.application.job_store import FINAL_STATUSES, JobStore


class SubprocessJobRunner:
    def __init__(self, job_store: JobStore, log_dir: Path | str, worker_module: str) -> None:
        self.job_store = job_store
        self.log_dir = Path(log_dir)
        self.worker_module = worker_module
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def start(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = str(job["job_id"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{job_id}.log"
        log_fp = log_path.open("a", encoding="utf-8")
        args = [
            sys.executable,
            "-m",
            self.worker_module,
            "--job-db",
            str(self.job_store.db_path),
            "--job-id",
            job_id,
        ]
        process = subprocess.Popen(
            args,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_fp.close()
        self.job_store.mark_running(job_id=job_id, worker_pid=int(process.pid), log_path=log_path)
        with self._lock:
            self._processes[job_id] = process
        thread = threading.Thread(
            target=self._monitor_process,
            args=(job_id, process),
            name=f"job-monitor-{job_id}",
            daemon=True,
        )
        thread.start()
        updated = self.job_store.get_job(job_id)
        return updated or job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            process = self._processes.get(job_id)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return self.job_store.request_cancel(job_id)

    def _monitor_process(self, job_id: str, process: subprocess.Popen[str]) -> None:
        exit_code = process.wait()
        with self._lock:
            self._processes.pop(job_id, None)
        job = self.job_store.get_job(job_id)
        if job is None or str(job["status"]) in FINAL_STATUSES:
            return
        if exit_code == 0:
            self.job_store.finish_job(
                job_id,
                status="failed",
                error="Worker exited without writing final job status",
            )
            return
        self.job_store.finish_job(
            job_id,
            status="failed",
            error=f"Worker process exited with code {exit_code}",
        )
