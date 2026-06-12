"""Train worker CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.application.train_service import build_train_overrides_from_payload, run_train
from common.config.config_loader import apply_overrides, save_resolved_config
from common.config.role_schema import role_to_kernel_config
from common.types.errors import ConfigError
from train_worker.config.schema import load_config, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run train worker locally")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json-summary", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = {
        "run_name": args.run_name,
        "dataset_dir": args.dataset_dir,
        "backend": args.backend,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
    }
    overrides = list(args.set) + build_train_overrides_from_payload(payload)
    try:
        role_cfg = load_config(
            Path(args.config).resolve(),
            workdir_override=str(Path(args.workdir).resolve()) if args.workdir else None,
        )
        role_cfg = validate_config(apply_overrides(role_cfg, overrides))
        kernel_cfg = role_to_kernel_config(role_cfg, "train", "train_worker")
        temp_path = Path(kernel_cfg["workspace"]["root"]) / "tmp" / "train_worker.kernel.toml"
        save_resolved_config(kernel_cfg, temp_path)
        summary = run_train(config_path=temp_path)
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
