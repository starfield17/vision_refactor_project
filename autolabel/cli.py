"""Autolabel CLI entrypoint (Phase 3)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from share.config.config_loader import load_config, save_resolved_config
from share.config.editing import persist_config_overrides
from share.kernel.autolabel.llm_autolabel import run_llm_autolabel
from share.kernel.autolabel.model_autolabel import run_model_autolabel
from share.kernel.kernel import VisionKernel
from share.kernel.registry import KernelRegistry
from share.kernel.trainer.faster_rcnn import run_faster_rcnn_train
from share.kernel.trainer.yolo import run_yolo_train
from share.kernel.utils.logging import StructuredLogger
from share.config.schema import AUTOLABEL_CONFLICTS, AUTOLABEL_MODES, AUTOLABEL_MODEL_BACKENDS
from share.types.errors import ConfigError

AUTOLABEL_EDITABLE_PREFIXES = (
    "workspace",
    "data.labeled_dir",
    "data.unlabeled_dir",
    "train.device",
    "autolabel",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 autolabel CLI")
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
    parser.add_argument("--llm-api-key-env", default=None, help="autolabel.llm.api_key_env")
    parser.add_argument("--llm-prompt", default=None, help="autolabel.llm.prompt")
    parser.add_argument("--llm-timeout-sec", type=float, default=None, help="autolabel.llm.timeout_sec")
    parser.add_argument("--llm-max-retries", type=int, default=None, help="autolabel.llm.max_retries")
    parser.add_argument(
        "--llm-retry-backoff-sec", type=float, default=None, help="autolabel.llm.retry_backoff_sec"
    )
    parser.add_argument("--llm-qps-limit", type=float, default=None, help="autolabel.llm.qps_limit")
    parser.add_argument("--llm-max-images", type=int, default=None, help="autolabel.llm.max_images")
    return parser


def _resolve_log_path(cfg: dict) -> Path:
    workdir = Path(cfg["workspace"]["root"])
    log_file = Path(cfg["workspace"]["log_file"])
    if log_file.is_absolute():
        return log_file
    return workdir / log_file


def _append_override(overrides: list[str], key: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raw = "true" if value else "false"
    else:
        raw = str(value)
    overrides.append(f"{key}={raw}")


def _collect_granular_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    _append_override(overrides, "workspace.run_name", args.run_name)
    _append_override(overrides, "train.device", args.device)
    _append_override(overrides, "data.labeled_dir", args.labeled_dir)
    _append_override(overrides, "data.unlabeled_dir", args.unlabeled_dir)

    _append_override(overrides, "autolabel.mode", args.mode)
    _append_override(overrides, "autolabel.confidence", args.confidence)
    _append_override(overrides, "autolabel.batch_size", args.batch_size)
    _append_override(overrides, "autolabel.visualize", args.visualize)
    _append_override(overrides, "autolabel.on_conflict", args.on_conflict)

    _append_override(overrides, "autolabel.model.backend", args.model_backend)
    _append_override(overrides, "autolabel.model.onnx_model", args.model_onnx)

    _append_override(overrides, "autolabel.llm.base_url", args.llm_base_url)
    _append_override(overrides, "autolabel.llm.model", args.llm_model)
    _append_override(overrides, "autolabel.llm.api_key_env", args.llm_api_key_env)
    _append_override(overrides, "autolabel.llm.prompt", args.llm_prompt)
    _append_override(overrides, "autolabel.llm.timeout_sec", args.llm_timeout_sec)
    _append_override(overrides, "autolabel.llm.max_retries", args.llm_max_retries)
    _append_override(overrides, "autolabel.llm.retry_backoff_sec", args.llm_retry_backoff_sec)
    _append_override(overrides, "autolabel.llm.qps_limit", args.llm_qps_limit)
    _append_override(overrides, "autolabel.llm.max_images", args.llm_max_images)
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None
    runtime_overrides = list(args.set)
    runtime_overrides.extend(_collect_granular_overrides(args))

    if args.config_only and not args.save_config:
        print("[USAGE ERROR] --config-only requires --save-config", file=sys.stderr)
        return 2

    if args.save_config:
        try:
            persist_config_overrides(
                config_path=config_path,
                overrides=runtime_overrides,
                allowed_prefixes=AUTOLABEL_EDITABLE_PREFIXES,
            )
        except ConfigError as exc:
            print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
            return 2
        print(f"updated_config={config_path}")
        if args.config_only:
            return 0
        runtime_overrides = []

    try:
        cfg = load_config(
            config_path=config_path,
            overrides=runtime_overrides,
            workdir_override=workdir_override,
        )
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    log_path = _resolve_log_path(cfg)
    logger = StructuredLogger(log_path=log_path, level=cfg["workspace"]["log_level"])
    logger.info("autolabel.cli.start", "CLI started", config_path=str(config_path))

    registry = KernelRegistry()
    registry.register_trainer("yolo", run_yolo_train)
    registry.register_trainer("faster_rcnn", run_faster_rcnn_train)
    registry.register_autolabeler("model", run_model_autolabel)
    registry.register_autolabeler("llm", run_llm_autolabel)

    kernel = VisionKernel(cfg=cfg, logger=logger, registry=registry)
    result = kernel.run_autolabel()

    resolved_path = result.run_context.run_dir / "config.resolved.toml"
    save_resolved_config(cfg, resolved_path)

    latest_link = Path(cfg["workspace"]["root"]) / "config.resolved.toml"
    shutil.copyfile(resolved_path, latest_link)

    logger.info(
        "autolabel.cli.done",
        "CLI finished",
        run_id=result.run_context.run_id,
        status=result.status,
        resolved_config=str(resolved_path),
    )

    print(f"run_id={result.run_context.run_id}")
    print(f"status={result.status}")
    print(f"resolved_config={resolved_path}")
    print(f"artifacts={result.run_context.run_dir / 'artifacts.json'}")

    if result.status != "ok":
        print(f"error={result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
