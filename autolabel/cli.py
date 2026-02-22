"""Autolabel CLI entrypoint (Phase 3)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from share.config.config_loader import load_config, save_resolved_config
from share.kernel.autolabel.llm_autolabel import run_llm_autolabel
from share.kernel.autolabel.model_autolabel import run_model_autolabel
from share.kernel.kernel import VisionKernel
from share.kernel.registry import KernelRegistry
from share.kernel.trainer.faster_rcnn import run_faster_rcnn_train
from share.kernel.trainer.yolo import run_yolo_train
from share.kernel.utils.logging import StructuredLogger
from share.types.errors import ConfigError


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
    return parser


def _resolve_log_path(cfg: dict) -> Path:
    workdir = Path(cfg["workspace"]["root"])
    log_file = Path(cfg["workspace"]["log_file"])
    if log_file.is_absolute():
        return log_file
    return workdir / log_file


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    workdir_override = str(Path(args.workdir).resolve()) if args.workdir else None

    try:
        cfg = load_config(
            config_path=config_path,
            overrides=args.set,
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
