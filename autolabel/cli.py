"""Autolabel CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from share.application.autolabel_service import (
    build_autolabel_overrides_from_payload,
    run_autolabel,
    save_autolabel_config,
)
from share.config.schema import AUTOLABEL_CONFLICTS, AUTOLABEL_MODES, AUTOLABEL_MODEL_BACKENDS
from share.types.errors import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autolabel CLI")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", required=True, help="Path to config.toml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, repeatable. Example: --set autolabel.mode=model",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Persist granular args/--set values into config.toml before running.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Apply config changes and exit without running autolabel (requires --save-config).",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print a final JSON summary line for machine parsing.",
    )

    parser.add_argument("--run-name", default=None, help="workspace.run_name")
    parser.add_argument("--device", default=None, help="train.device (cpu/cuda:0/mps...)")
    parser.add_argument("--labeled-dir", default=None, help="data.labeled_dir")
    parser.add_argument("--unlabeled-dir", default=None, help="data.unlabeled_dir")

    parser.add_argument(
        "--mode",
        choices=sorted(AUTOLABEL_MODES),
        default=None,
        help="autolabel.mode",
    )
    parser.add_argument("--confidence", type=float, default=None, help="autolabel.confidence")
    parser.add_argument("--batch-size", type=int, default=None, help="autolabel.batch_size")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="autolabel.visualize / --no-visualize",
    )
    parser.add_argument(
        "--on-conflict",
        choices=sorted(AUTOLABEL_CONFLICTS),
        default=None,
        help="autolabel.on_conflict",
    )

    parser.add_argument(
        "--model-backend",
        choices=sorted(AUTOLABEL_MODEL_BACKENDS),
        default=None,
        help="autolabel.model.backend",
    )
    parser.add_argument("--model-onnx", default=None, help="autolabel.model.onnx_model")

    parser.add_argument("--llm-base-url", default=None, help="autolabel.llm.base_url")
    parser.add_argument("--llm-model", default=None, help="autolabel.llm.model")
    parser.add_argument("--llm-api-key", default=None, help="autolabel.llm.api_key")
    parser.add_argument(
        "--llm-api-key-env-name",
        "--llm-api-key-env",
        dest="llm_api_key_env_name",
        default=None,
        help="autolabel.llm.api_key_env_name",
    )
    parser.add_argument("--llm-prompt", default=None, help="autolabel.llm.prompt")
    parser.add_argument("--llm-timeout-sec", type=float, default=None, help="autolabel.llm.timeout_sec")
    parser.add_argument("--llm-max-retries", type=int, default=None, help="autolabel.llm.max_retries")
    parser.add_argument(
        "--llm-retry-backoff-sec", type=float, default=None, help="autolabel.llm.retry_backoff_sec"
    )
    parser.add_argument("--llm-qps-limit", type=float, default=None, help="autolabel.llm.qps_limit")
    parser.add_argument("--llm-max-images", type=int, default=None, help="autolabel.llm.max_images")
    return parser


def _build_payload_from_args(args: argparse.Namespace) -> dict[str, object | None]:
    return {
        "run_name": args.run_name,
        "device": args.device,
        "labeled_dir": args.labeled_dir,
        "unlabeled_dir": args.unlabeled_dir,
        "mode": args.mode,
        "confidence": args.confidence,
        "batch_size": args.batch_size,
        "visualize": args.visualize,
        "on_conflict": args.on_conflict,
        "model_backend": args.model_backend,
        "model_onnx": args.model_onnx,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "llm_api_key": args.llm_api_key,
        "llm_api_key_env_name": args.llm_api_key_env_name,
        "llm_prompt": args.llm_prompt,
        "llm_timeout_sec": args.llm_timeout_sec,
        "llm_max_retries": args.llm_max_retries,
        "llm_retry_backoff_sec": args.llm_retry_backoff_sec,
        "llm_qps_limit": args.llm_qps_limit,
        "llm_max_images": args.llm_max_images,
    }


def _print_summary(summary: dict[str, object], json_summary: bool) -> None:
    if "updated_config" in summary:
        print(f"updated_config={summary['updated_config']}")
    if "run_id" in summary:
        print(f"run_id={summary['run_id']}")
    if "status" in summary:
        print(f"status={summary['status']}")
    if "resolved_config" in summary and summary["resolved_config"]:
        print(f"resolved_config={summary['resolved_config']}")
    if "artifacts_path" in summary and summary["artifacts_path"]:
        print(f"artifacts={summary['artifacts_path']}")
    if summary.get("error"):
        print(f"error={summary['error']}", file=sys.stderr)
    if json_summary:
        print(json.dumps(summary, ensure_ascii=True))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None
    runtime_overrides = list(args.set)
    runtime_overrides.extend(
        build_autolabel_overrides_from_payload(_build_payload_from_args(args))
    )

    if args.config_only and not args.save_config:
        print("[USAGE ERROR] --config-only requires --save-config", file=sys.stderr)
        return 2

    if args.save_config:
        try:
            save_autolabel_config(config_path=config_path, overrides=runtime_overrides)
        except ConfigError as exc:
            print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
            return 2
        _print_summary(
            {
                "status": "ok",
                "updated_config": str(config_path),
                "config_only": bool(args.config_only),
            },
            json_summary=args.json_summary,
        )
        if args.config_only:
            return 0
        runtime_overrides = []

    try:
        summary = run_autolabel(
            config_path=config_path,
            workdir_override=workdir_override,
            overrides=runtime_overrides,
        )
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    _print_summary(summary, json_summary=args.json_summary)
    if summary["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
