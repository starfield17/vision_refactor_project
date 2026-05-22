"""Worker process for deploy jobs."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from share.application.deploy_service import run_deploy_edge
from share.application.job_store import JobStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy/statistics job worker")
    parser.add_argument("--job-db", required=True)
    parser.add_argument("--job-id", required=True)
    return parser


def _run(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job["payload"])
    config_path = Path(str(payload["config_path"]))
    workdir_override = payload.get("workdir_override")
    if workdir_override is not None:
        workdir_override = str(workdir_override)
    overrides = [str(item) for item in payload.get("overrides", [])]
    kind = str(job["kind"])
    if kind == "deploy_edge":
        return run_deploy_edge(
            config_path=config_path,
            workdir_override=workdir_override,
            overrides=overrides,
        )
    raise ValueError(f"unsupported deploy_statistics job kind={kind}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = JobStore(Path(args.job_db))
    job = store.get_job(args.job_id)
    if job is None:
        print(f"job not found: {args.job_id}")
        return 2
    try:
        summary = _run(job)
        status = "succeeded" if summary.get("status") == "ok" else "failed"
        store.finish_job(
            job_id=args.job_id,
            status=status,
            result=summary,
            error=str(summary.get("error") or ""),
            run_id=str(summary.get("run_id") or ""),
            run_dir=str(summary.get("run_dir") or ""),
            log_path=str(summary.get("log_path") or job.get("log_path") or ""),
        )
        return 0 if status == "succeeded" else 1
    except Exception as exc:
        traceback.print_exc()
        store.finish_job(job_id=args.job_id, status="failed", result={}, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
