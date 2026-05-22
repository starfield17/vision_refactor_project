"""Deploy edge CLI frontend for the deploy/statistics backend service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from share.application.service_client import load_service_connection, submit_job, wait_for_job
from share.types.errors import ConfigError, TransportError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy edge CLI")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=0.0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json-summary", action="store_true")
    return parser


def _print_summary(summary: dict[str, object], json_summary: bool) -> None:
    for key in ("job_id", "run_id", "status", "resolved_config", "artifacts_path"):
        if summary.get(key):
            label = "artifacts" if key == "artifacts_path" else key
            print(f"{label}={summary[key]}")
    if summary.get("error"):
        print(f"error={summary['error']}", file=sys.stderr)
    if json_summary:
        print(json.dumps(summary, ensure_ascii=True))


def _job_to_summary(job: dict[str, object]) -> dict[str, object]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    result_dict = dict(result)  # type: ignore[arg-type]
    status = "ok" if job.get("status") == "succeeded" else str(job.get("status"))
    return {
        **result_dict,
        "job_id": job.get("job_id", ""),
        "job_status": job.get("status", ""),
        "status": result_dict.get("status", status),
        "error": result_dict.get("error") or job.get("error") or None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None
    try:
        api_url, token = load_service_connection(
            config_path=config_path,
            service_name="deploy_statistics",
            workdir_override=workdir_override,
            api_url_override=args.api_url,
            api_token_override=args.api_token,
        )
        job = submit_job(
            api_url=api_url,
            token=token,
            path="/api/v1/deploy/edge/jobs",
            payload={"overrides": list(args.set)},
        )
        if not args.no_wait:
            job = wait_for_job(
                api_url=api_url,
                token=token,
                job_id=str(job["job_id"]),
                poll_sec=float(args.poll_sec),
                timeout_sec=float(args.wait_timeout_sec),
            )
    except (ConfigError, TransportError) as exc:
        print(f"[API ERROR] {exc}", file=sys.stderr)
        return 2

    summary = _job_to_summary(job)
    _print_summary(summary, json_summary=args.json_summary)
    return 0 if summary.get("status") == "ok" and job.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
