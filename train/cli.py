"""Train CLI frontend for the train/autolabel backend service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from share.application.service_client import (
    load_service_connection,
    patch_config,
    submit_job,
    wait_for_job,
)
from share.application.train_service import build_train_overrides_from_payload
from share.config.schema import FASTER_RCNN_VARIANTS, QUANTIZE_MODES, TRAIN_BACKENDS
from share.types.errors import ConfigError, TransportError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CLI")
    parser.add_argument("--workdir", default=None, help="Override workspace.root")
    parser.add_argument("--config", required=True, help="Path to config.toml")
    parser.add_argument("--api-url", default=None, help="Override train/autolabel service URL")
    parser.add_argument("--api-token", default=None, help="Override train/autolabel service token")
    parser.add_argument("--no-wait", action="store_true", help="Submit job and return immediately.")
    parser.add_argument("--poll-sec", type=float, default=1.0, help="Job polling interval.")
    parser.add_argument("--wait-timeout-sec", type=float, default=0.0, help="0 means no timeout.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--save-config", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--json-summary", action="store_true")

    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--backend", choices=sorted(TRAIN_BACKENDS), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--frcnn-variant", choices=sorted(FASTER_RCNN_VARIANTS), default=None)
    parser.add_argument("--frcnn-lr", type=float, default=None)
    parser.add_argument("--frcnn-momentum", type=float, default=None)
    parser.add_argument("--frcnn-weight-decay", type=float, default=None)
    parser.add_argument("--frcnn-num-workers", type=int, default=None)
    parser.add_argument("--frcnn-max-samples", type=int, default=None)
    parser.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--export-opset", type=int, default=None)
    parser.add_argument("--export-quantize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--export-quantize-mode", choices=sorted(QUANTIZE_MODES), default=None)
    parser.add_argument("--export-calib-samples", type=int, default=None)
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
    for key in ("updated_config", "job_id", "run_id", "status", "resolved_config", "artifacts_path"):
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
    runtime_overrides = list(args.set)
    payload = _build_payload_from_args(args)
    runtime_overrides.extend(build_train_overrides_from_payload(payload))

    if args.config_only and not args.save_config:
        print("[USAGE ERROR] --config-only requires --save-config", file=sys.stderr)
        return 2

    try:
        api_url, token = load_service_connection(
            config_path=config_path,
            service_name="train_autolabel",
            workdir_override=workdir_override,
            api_url_override=args.api_url,
            api_token_override=args.api_token,
        )
        if args.save_config:
            response = patch_config(
                api_url=api_url,
                token=token,
                area="train",
                overrides=runtime_overrides,
            )
            if response.get("ok") is not True:
                raise TransportError(str(response))
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

        job = submit_job(
            api_url=api_url,
            token=token,
            path="/api/v1/train/jobs",
            payload={"overrides": runtime_overrides},
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
