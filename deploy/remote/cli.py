"""Deploy remote CLI frontend for the deploy/statistics backend service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from share.application.api_common import get_json, post_json
from share.application.service_client import load_service_connection
from share.types.errors import ConfigError, TransportError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy remote runtime CLI")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json-summary", action="store_true")
    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=["start", "stop", "status"],
        help="Remote runtime command.",
    )
    return parser


def _print_response(payload: dict[str, object], json_summary: bool) -> None:
    for key in ("status", "run_id", "frames_processed", "detections_total", "stats_sent", "stats_failed"):
        if key in payload:
            print(f"{key}={payload[key]}")
    if payload.get("error"):
        print(f"error={payload['error']}", file=sys.stderr)
    if json_summary:
        print(json.dumps(payload, ensure_ascii=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None
    try:
        api_url, token = load_service_connection(
            config_path=config_path,
            service_name="deploy_statistics",
            workdir_override=workdir_override,
            overrides=list(args.set),
            api_url_override=args.api_url,
            api_token_override=args.api_token,
        )
        if args.command == "status":
            payload = get_json(api_url, "/api/v1/deploy/remote/status", token=token)
        else:
            payload = post_json(api_url, f"/api/v1/deploy/remote/{args.command}", {}, token=token)
    except (ConfigError, TransportError) as exc:
        print(f"[API ERROR] {exc}", file=sys.stderr)
        return 2

    _print_response(payload, json_summary=args.json_summary)
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
