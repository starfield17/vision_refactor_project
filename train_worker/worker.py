"""Worker process for train worker jobs."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from common.application.job_store import JobStore
from common.application.train_service import run_train
from common.config.config_loader import apply_overrides
from common.config.role_schema import role_to_kernel_config
from train_worker.config.schema import load_config, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train worker subprocess")
    parser.add_argument("--job-db", required=True)
    parser.add_argument("--job-id", required=True)
    return parser


def _run(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job["payload"])
    config_path = Path(str(payload["config_path"]))
    workdir_override = payload.get("workdir_override")
    role_cfg = load_config(
        config_path=config_path,
        workdir_override=str(workdir_override) if workdir_override is not None else None,
    )
    role_cfg = validate_config(apply_overrides(role_cfg, [str(i) for i in payload.get("overrides", [])]))
    kernel_cfg = role_to_kernel_config(role_cfg, "train", "train_worker")
    temp_path = Path(kernel_cfg["workspace"]["root"]) / "tmp" / "train_worker.kernel.toml"
    from common.config.config_loader import save_resolved_config

    save_resolved_config(kernel_cfg, temp_path)
    return run_train(config_path=temp_path)


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
