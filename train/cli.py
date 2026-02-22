"""Train CLI entrypoint (Phase 1)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from share.config.config_loader import load_config, save_resolved_config
from share.config.editing import persist_config_overrides
from share.kernel.kernel import VisionKernel
from share.kernel.registry import KernelRegistry
from share.kernel.trainer.faster_rcnn import run_faster_rcnn_train
from share.kernel.trainer.yolo import run_yolo_train
from share.kernel.utils.logging import StructuredLogger
from share.config.schema import FASTER_RCNN_VARIANTS, QUANTIZE_MODES, TRAIN_BACKENDS
from share.types.errors import ConfigError

TRAIN_EDITABLE_PREFIXES = ("workspace", "data.yolo_dataset_dir", "train", "export")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 train CLI")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", required=True, help="Path to config.toml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, repeatable. Example: --set train.epochs=2",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Persist granular args/--set values into config.toml before running.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Apply config changes and exit without running training (requires --save-config).",
    )

    parser.add_argument("--run-name", default=None, help="workspace.run_name")
    parser.add_argument("--dataset-dir", default=None, help="data.yolo_dataset_dir")

    parser.add_argument(
        "--backend",
        choices=sorted(TRAIN_BACKENDS),
        default=None,
        help="train.backend",
    )
    parser.add_argument("--device", default=None, help="train.device (cpu/cuda:0/mps...)")
    parser.add_argument("--seed", type=int, default=None, help="train.seed")
    parser.add_argument("--epochs", type=int, default=None, help="train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="train.batch_size")
    parser.add_argument("--img-size", type=int, default=None, help="train.img_size")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="train.dry_run / --no-dry-run",
    )

    parser.add_argument("--yolo-weights", default=None, help="train.yolo.weights")
    parser.add_argument(
        "--frcnn-variant",
        choices=sorted(FASTER_RCNN_VARIANTS),
        default=None,
        help="train.faster_rcnn.variant",
    )
    parser.add_argument("--frcnn-lr", type=float, default=None, help="train.faster_rcnn.lr")
    parser.add_argument(
        "--frcnn-momentum", type=float, default=None, help="train.faster_rcnn.momentum"
    )
    parser.add_argument(
        "--frcnn-weight-decay", type=float, default=None, help="train.faster_rcnn.weight_decay"
    )
    parser.add_argument(
        "--frcnn-num-workers", type=int, default=None, help="train.faster_rcnn.num_workers"
    )
    parser.add_argument(
        "--frcnn-max-samples", type=int, default=None, help="train.faster_rcnn.max_samples"
    )

    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export.onnx / --no-export-onnx",
    )
    parser.add_argument("--export-opset", type=int, default=None, help="export.opset")
    parser.add_argument(
        "--export-quantize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export.quantize / --no-export-quantize",
    )
    parser.add_argument(
        "--export-quantize-mode",
        choices=sorted(QUANTIZE_MODES),
        default=None,
        help="export.quantize_mode",
    )
    parser.add_argument(
        "--export-calib-samples", type=int, default=None, help="export.calib_samples"
    )
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
    _append_override(overrides, "data.yolo_dataset_dir", args.dataset_dir)

    _append_override(overrides, "train.backend", args.backend)
    _append_override(overrides, "train.device", args.device)
    _append_override(overrides, "train.seed", args.seed)
    _append_override(overrides, "train.epochs", args.epochs)
    _append_override(overrides, "train.batch_size", args.batch_size)
    _append_override(overrides, "train.img_size", args.img_size)
    _append_override(overrides, "train.dry_run", args.dry_run)

    _append_override(overrides, "train.yolo.weights", args.yolo_weights)
    _append_override(overrides, "train.faster_rcnn.variant", args.frcnn_variant)
    _append_override(overrides, "train.faster_rcnn.lr", args.frcnn_lr)
    _append_override(overrides, "train.faster_rcnn.momentum", args.frcnn_momentum)
    _append_override(overrides, "train.faster_rcnn.weight_decay", args.frcnn_weight_decay)
    _append_override(overrides, "train.faster_rcnn.num_workers", args.frcnn_num_workers)
    _append_override(overrides, "train.faster_rcnn.max_samples", args.frcnn_max_samples)

    _append_override(overrides, "export.onnx", args.export_onnx)
    _append_override(overrides, "export.opset", args.export_opset)
    _append_override(overrides, "export.quantize", args.export_quantize)
    _append_override(overrides, "export.quantize_mode", args.export_quantize_mode)
    _append_override(overrides, "export.calib_samples", args.export_calib_samples)
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
                allowed_prefixes=TRAIN_EDITABLE_PREFIXES,
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
    logger.info("train.cli.start", "CLI started", config_path=str(config_path))

    registry = KernelRegistry()
    registry.register_trainer("yolo", run_yolo_train)
    registry.register_trainer("faster_rcnn", run_faster_rcnn_train)

    kernel = VisionKernel(cfg=cfg, logger=logger, registry=registry)
    result = kernel.run_train()

    resolved_path = result.run_context.run_dir / "config.resolved.toml"
    save_resolved_config(cfg, resolved_path)

    latest_link = Path(cfg["workspace"]["root"]) / "config.resolved.toml"
    shutil.copyfile(resolved_path, latest_link)

    logger.info(
        "train.cli.done",
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
