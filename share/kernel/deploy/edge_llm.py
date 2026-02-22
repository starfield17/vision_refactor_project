"""Edge LLM deploy pipeline for Phase 5."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from share.kernel.llm.client import (
    HTTPCallError,
    build_system_prompt,
    extract_message_content,
    load_api_key,
    parse_llm_detection_payload,
    post_chat_completion,
)
from share.kernel.deploy.edge_common import FpsLimiter, append_stats_snapshot, iter_source_frames
from share.kernel.transport.frame_http import encode_jpeg_base64
from share.kernel.transport.stats_http import push_stats_event
from share.types.detection import Detection
from share.types.errors import DataValidationError, TransportError
from share.types.stats import StatsEvent


def _call_llm_for_frame(
    frame_bgr: Any,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    class_names: list[str],
    class_id_map: dict[str, int],
    confidence: float,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
) -> tuple[list[Detection], float]:
    if not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:  # type: ignore[attr-defined]
        raise DataValidationError("invalid frame payload for llm inference")

    height = int(frame_bgr.shape[0])  # type: ignore[index]
    width = int(frame_bgr.shape[1])  # type: ignore[index]
    if width <= 0 or height <= 0:
        raise DataValidationError("frame size must be positive")

    data_url = f"data:image/jpeg;base64,{encode_jpeg_base64(frame_bgr, jpeg_quality=90)}"
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze this image and output JSON only."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }

    last_error: str | None = None
    infer_started = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            response_json = post_chat_completion(
                url=endpoint,
                payload=payload,
                api_key=api_key,
                timeout_sec=timeout_sec,
            )
            text = extract_message_content(response_json)
            detections = parse_llm_detection_payload(
                response_text=text,
                class_names=class_names,
                class_id_map=class_id_map,
                width=width,
                height=height,
                min_confidence=confidence,
            )
            latency_ms = round((time.perf_counter() - infer_started) * 1000.0, 3)
            return detections, latency_ms
        except HTTPCallError as exc:
            last_error = f"http_error(status={exc.status_code}): {exc.detail[:300]}"
            if attempt < max_retries and exc.retryable:
                time.sleep(retry_backoff_sec * (2**attempt))
                continue
            break
        except DataValidationError as exc:
            last_error = f"response_parse_error: {exc}"
            if attempt < max_retries:
                time.sleep(retry_backoff_sec * (2**attempt))
                continue
            break

    raise DataValidationError(f"LLM frame inference failed after retries, detail={last_error}")


def _save_annotated(frame_bgr: Any, detections: list[Detection], target_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False

    try:
        annotated = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 180, 16), 2)
            cv2.putText(
                annotated,
                f"{det.class_name}:{det.score:.2f}",
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (16, 180, 16),
                1,
                cv2.LINE_AA,
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(target_path), annotated))
    except Exception:
        return False


def run_edge_llm_deploy(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    edge_cfg = cfg["deploy"]["edge"]
    llm_cfg = edge_cfg["llm"]
    source_id = str(edge_cfg["source_id"])

    class_names = list(cfg["class_map"]["names"])
    class_id_map = {str(k): int(v) for k, v in dict(cfg["class_map"]["id_map"]).items()}
    confidence = float(edge_cfg["confidence"])
    api_key = load_api_key(str(llm_cfg["api_key_env"]))
    prompt = build_system_prompt(
        base_prompt=str(llm_cfg["prompt"]),
        class_names=class_names,
        min_conf=confidence,
    )

    run_id = str(run_ctx["run_id"])
    run_dir = Path(str(run_ctx["run_dir"]))
    workdir = Path(cfg["workspace"]["root"])
    logger = run_ctx["logger"]

    save_annotated = bool(edge_cfg["save_annotated"])
    max_frames = int(edge_cfg["max_frames"])
    limiter = FpsLimiter(float(edge_cfg["fps_limit"]))
    endpoint = str(edge_cfg["stats_endpoint"])
    api_key_stats = str(edge_cfg["api_key"])
    timeout_stats_sec = float(edge_cfg["stats_timeout_sec"])
    base_url = str(llm_cfg["base_url"])
    model = str(llm_cfg["model"])
    timeout_sec = float(llm_cfg["timeout_sec"])
    max_retries = int(llm_cfg["max_retries"])
    retry_backoff_sec = float(llm_cfg["retry_backoff_sec"])
    qps_interval = 1.0 / float(llm_cfg["qps_limit"])
    next_call_ts = 0.0

    snapshot_path = run_dir / "stats.jsonl"
    output_dir = workdir / "outputs" / run_id / "annotated_frames"

    attempted = 0
    processed = 0
    detections_total = 0
    llm_failures = 0
    stats_sent = 0
    stats_failed = 0
    annotated_saved = 0
    class_totals: dict[str, int] = {}

    for packet in iter_source_frames(edge_cfg):
        if max_frames > 0 and attempted >= max_frames:
            break
        attempted += 1

        limiter.wait()
        now = time.perf_counter()
        if now < next_call_ts:
            time.sleep(next_call_ts - now)

        try:
            detections, latency_ms = _call_llm_for_frame(
                frame_bgr=packet.frame_bgr,
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=prompt,
                class_names=class_names,
                class_id_map=class_id_map,
                confidence=confidence,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
            )
            next_call_ts = time.perf_counter() + qps_interval
        except DataValidationError as exc:
            llm_failures += 1
            logger.warn(
                "deploy.edge.llm.infer_failed",
                "LLM inference failed for frame",
                run_id=run_id,
                source_id=source_id,
                frame_index=packet.frame_index,
                error=str(exc),
            )
            continue

        counts_by_class: dict[str, int] = {}
        for det in detections:
            counts_by_class[det.class_name] = counts_by_class.get(det.class_name, 0) + 1
            class_totals[det.class_name] = class_totals.get(det.class_name, 0) + 1

        event = StatsEvent.now(
            source_id=source_id,
            total_detections=len(detections),
            counts_by_class=counts_by_class,
            latency_ms=latency_ms,
        )
        append_stats_snapshot(snapshot_path, event)

        if endpoint:
            try:
                push_stats_event(
                    event=event,
                    endpoint=endpoint,
                    api_key=api_key_stats,
                    timeout_sec=timeout_stats_sec,
                )
                stats_sent += 1
            except TransportError as exc:
                stats_failed += 1
                logger.warn(
                    "deploy.edge.llm.stats_push.failed",
                    "Failed to push llm stats event",
                    run_id=run_id,
                    source_id=source_id,
                    frame_index=packet.frame_index,
                    error=str(exc),
                )

        if save_annotated and _save_annotated(packet.frame_bgr, detections, output_dir / packet.frame_name):
            annotated_saved += 1

        detections_total += len(detections)
        processed += 1
        logger.info(
            "deploy.edge.llm.frame",
            "Frame processed by llm mode",
            run_id=run_id,
            source_id=source_id,
            frame_index=packet.frame_index,
            detections=len(detections),
            latency_ms=latency_ms,
            source_path=packet.source_path,
        )

    if attempted == 0:
        raise DataValidationError("no valid frames were processed from deploy source")
    if processed == 0:
        raise DataValidationError("all frames failed in deploy.edge.mode=llm")

    return {
        "mode": "llm",
        "source": str(edge_cfg["source"]),
        "source_id": source_id,
        "stats_snapshot_path": str(snapshot_path),
        "annotated_frames_dir": str(output_dir) if save_annotated else "",
        "stats_endpoint": endpoint,
        "stats": {
            "frames_attempted": attempted,
            "frames_processed": processed,
            "detections_total": detections_total,
            "class_totals": class_totals,
            "llm_failures": llm_failures,
            "stats_sent": stats_sent,
            "stats_failed": stats_failed,
            "annotated_saved": annotated_saved,
            "fps_limit": float(edge_cfg["fps_limit"]),
        },
        "config": {
            "base_url": base_url,
            "model": model,
            "confidence": confidence,
            "timeout_sec": timeout_sec,
            "max_retries": max_retries,
            "retry_backoff_sec": retry_backoff_sec,
            "qps_limit": float(llm_cfg["qps_limit"]),
        },
    }
