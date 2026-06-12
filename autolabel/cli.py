"""Autolabel CLI frontend for the train/autolabel backend service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from share.application.autolabel_service import build_autolabel_overrides_from_payload
from share.application.service_client import (
    load_service_connection,
    patch_config,
    submit_job,
    wait_for_job,
)
from share.config.schema import AUTOLABEL_CONFLICTS, AUTOLABEL_MODES, AUTOLABEL_MODEL_BACKENDS
from share.types.errors import ConfigError, TransportError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autolabel CLI")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=0.0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--save-config", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--json-summary", action="store_true")

    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--labeled-dir", default=None)
    parser.add_argument("--unlabeled-dir", default=None)
    parser.add_argument("--mode", choices=sorted(AUTOLABEL_MODES), default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--on-conflict", choices=sorted(AUTOLABEL_CONFLICTS), default=None)
    parser.add_argument("--model-backend", choices=sorted(AUTOLABEL_MODEL_BACKENDS), default=None)
    parser.add_argument("--model-onnx", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument(
        "--llm-api-key-env-name",
        "--llm-api-key-env",
        dest="llm_api_key_env_name",
        default=None,
    )
    parser.add_argument("--llm-prompt", default=None)
    parser.add_argument("--llm-timeout-sec", type=float, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=None)
    parser.add_argument("--llm-retry-backoff-sec", type=float, default=None)
    parser.add_argument("--llm-qps-limit", type=float, default=None)
    parser.add_argument("--llm-max-images", type=int, default=None)
    parser.add_argument("--locate-anything-model", default=None)
    parser.add_argument("--locate-anything-device", default=None)
    parser.add_argument("--locate-anything-dtype", default=None)
    parser.add_argument(
        "--locate-anything-quantization",
        choices=["none", "bnb_4bit"],
        default=None,
    )
    parser.add_argument("--locate-anything-bnb-4bit-compute-dtype", default=None)
    parser.add_argument(
        "--locate-anything-bnb-4bit-quant-type",
        choices=["nf4", "fp4"],
        default=None,
    )
    parser.add_argument(
        "--locate-anything-bnb-4bit-use-double-quant",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--locate-anything-device-map", default=None)
    parser.add_argument("--locate-anything-attn-implementation", default=None)
    parser.add_argument(
        "--locate-anything-generation-mode",
        choices=["fast", "slow", "hybrid"],
        default=None,
    )
    parser.add_argument("--locate-anything-max-new-tokens", type=int, default=None)
    parser.add_argument("--locate-anything-temperature", type=float, default=None)
    parser.add_argument("--locate-anything-prompt-template", default=None)
    parser.add_argument("--locate-anything-nms-iou", type=float, default=None)
    parser.add_argument("--locate-anything-default-score", type=float, default=None)
    parser.add_argument("--locate-anything-verbose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--locate-anything-max-images", type=int, default=None)
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
        "locate_anything_model": args.locate_anything_model,
        "locate_anything_device": args.locate_anything_device,
        "locate_anything_dtype": args.locate_anything_dtype,
        "locate_anything_quantization": args.locate_anything_quantization,
        "locate_anything_bnb_4bit_compute_dtype": (
            args.locate_anything_bnb_4bit_compute_dtype
        ),
        "locate_anything_bnb_4bit_quant_type": args.locate_anything_bnb_4bit_quant_type,
        "locate_anything_bnb_4bit_use_double_quant": (
            args.locate_anything_bnb_4bit_use_double_quant
        ),
        "locate_anything_device_map": args.locate_anything_device_map,
        "locate_anything_attn_implementation": args.locate_anything_attn_implementation,
        "locate_anything_generation_mode": args.locate_anything_generation_mode,
        "locate_anything_max_new_tokens": args.locate_anything_max_new_tokens,
        "locate_anything_temperature": args.locate_anything_temperature,
        "locate_anything_prompt_template": args.locate_anything_prompt_template,
        "locate_anything_nms_iou": args.locate_anything_nms_iou,
        "locate_anything_default_score": args.locate_anything_default_score,
        "locate_anything_verbose": args.locate_anything_verbose,
        "locate_anything_max_images": args.locate_anything_max_images,
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
    runtime_overrides.extend(build_autolabel_overrides_from_payload(payload))

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
                area="autolabel",
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
            path="/api/v1/autolabel/jobs",
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
