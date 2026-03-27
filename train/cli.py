"""Train CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from share.application.train_service import (
    build_train_overrides_from_payload,
    run_train,
    save_train_config,
)
from share.config.schema import FASTER_RCNN_VARIANTS, QUANTIZE_MODES, TRAIN_BACKENDS
from share.types.errors import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CLI")
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
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print a final JSON summary line for machine parsing.",
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


def _build_payload_from_args(args: argparse.Namespace) -> dict[str, object | None]:
    return {
        "run_name": args.run_name,
        "dataset_dir": args.dataset_dir,
        "backend": args.backend,
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "dry_run": args.dry_run,
        "yolo_weights": args.yolo_weights,
        "frcnn_variant": args.frcnn_variant,
        "frcnn_lr": args.frcnn_lr,
        "frcnn_momentum": args.frcnn_momentum,
        "frcnn_weight_decay": args.frcnn_weight_decay,
        "frcnn_num_workers": args.frcnn_num_workers,
        "frcnn_max_samples": args.frcnn_max_samples,
        "export_onnx": args.export_onnx,
        "export_opset": args.export_opset,
        "export_quantize": args.export_quantize,
        "export_quantize_mode": args.export_quantize_mode,
        "export_calib_samples": args.export_calib_samples,
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
    runtime_overrides.extend(build_train_overrides_from_payload(_build_payload_from_args(args)))

    if args.config_only and not args.save_config:
        print("[USAGE ERROR] --config-only requires --save-config", file=sys.stderr)
        return 2

    if args.save_config:
        try:
            save_train_config(config_path=config_path, overrides=runtime_overrides)
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
        summary = run_train(
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
