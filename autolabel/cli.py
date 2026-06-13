"""AutoLabel local CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.application.autolabel_service import (
    build_autolabel_overrides_from_payload,
    run_autolabel,
)
from common.config.config_loader import apply_overrides, save_resolved_config
from common.config.role_schema import role_to_kernel_config
from common.types.errors import ConfigError
from autolabel.config.schema import load_config, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local auto-labeling")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json-summary", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--labeled-dir", default=None)
    parser.add_argument("--unlabeled-dir", default=None)
    parser.add_argument("--model-onnx", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = {
        "run_name": args.run_name,
        "mode": args.mode,
        "labeled_dir": args.labeled_dir,
        "unlabeled_dir": args.unlabeled_dir,
        "model_onnx": args.model_onnx,
    }
    overrides = list(args.set) + build_autolabel_overrides_from_payload(payload)
    try:
        role_cfg = load_config(
            Path(args.config).resolve(),
            workdir_override=str(Path(args.workdir).resolve()) if args.workdir else None,
        )
        role_cfg = validate_config(apply_overrides(role_cfg, overrides))
        kernel_cfg = role_to_kernel_config(role_cfg, "autolabel", "autolabel")
        temp_path = Path(kernel_cfg["workspace"]["root"]) / "tmp" / "autolabel.kernel.toml"
        save_resolved_config(kernel_cfg, temp_path)
        summary = run_autolabel(config_path=temp_path)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2
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
