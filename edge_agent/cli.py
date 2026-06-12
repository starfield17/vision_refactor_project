"""Edge agent CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.application.deploy_service import run_deploy_edge
from common.config.config_loader import apply_overrides, save_resolved_config
from common.config.role_schema import role_to_kernel_config
from common.types.errors import ConfigError
from edge_agent.config.schema import load_config, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run edge agent locally")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json-summary", action="store_true")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "config"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        role_cfg = load_config(
            Path(args.config).resolve(),
            workdir_override=str(Path(args.workdir).resolve()) if args.workdir else None,
        )
        role_cfg = validate_config(apply_overrides(role_cfg, list(args.set)))
        kernel_cfg = role_to_kernel_config(role_cfg, "edge", "edge_agent")
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2
    if args.command == "config":
        print(json.dumps(role_cfg, ensure_ascii=True, indent=2))
        return 0
    try:
        temp_path = Path(kernel_cfg["workspace"]["root"]) / "tmp" / "edge_agent.kernel.toml"
        save_resolved_config(kernel_cfg, temp_path)
        summary = run_deploy_edge(config_path=temp_path)
    except Exception as exc:
        print(f"[RUNTIME ERROR] {exc}", file=sys.stderr)
        return 1
    for key in ("run_id", "status", "resolved_config", "artifacts_path"):
        if summary.get(key):
            print(f"{key}={summary[key]}")
    if args.json_summary:
        print(json.dumps(summary, ensure_ascii=True))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
