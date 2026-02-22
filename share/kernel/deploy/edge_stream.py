"""Edge stream deploy pipeline for Phase 5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from share.kernel.deploy.edge_common import FpsLimiter, append_stats_snapshot, iter_source_frames
from share.kernel.transport.frame_http import decode_jpeg_base64, encode_jpeg_base64, post_json
from share.types.errors import DataValidationError, TransportError
from share.types.stats import StatsEvent


def _save_annotated(frame_bgr: Any, target_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(target_path), frame_bgr))
    except Exception:
        return False


def _safe_counts(raw: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(raw, dict):
        return counts
    for key, value in raw.items():
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed < 0:
            continue
        counts[str(key)] = parsed
    return counts


def run_edge_stream_deploy(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    edge_cfg = cfg["deploy"]["edge"]
    source_id = str(edge_cfg["source_id"])
    run_id = str(run_ctx["run_id"])
    run_dir = Path(str(run_ctx["run_dir"]))
    workdir = Path(cfg["workspace"]["root"])
    logger = run_ctx["logger"]

    save_annotated = bool(edge_cfg["save_annotated"])
    max_frames = int(edge_cfg["max_frames"])
    limiter = FpsLimiter(float(edge_cfg["fps_limit"]))
    jpeg_quality = int(edge_cfg["jpeg_quality"])
    stream_endpoint = str(edge_cfg["stream_endpoint"])
    stream_api_key = str(edge_cfg["stream_api_key"])
    timeout_sec = float(edge_cfg["stream_timeout_sec"])

    if not stream_endpoint:
        raise DataValidationError("deploy.edge.stream_endpoint must not be empty in stream mode")

    snapshot_path = run_dir / "stats.jsonl"
    output_dir = workdir / "outputs" / run_id / "annotated_frames"

    attempts = 0
    processed = 0
    detections_total = 0
    remote_failures = 0
    annotated_saved = 0
    class_totals: dict[str, int] = {}

    for packet in iter_source_frames(edge_cfg):
        if max_frames > 0 and attempts >= max_frames:
            break
        attempts += 1
        limiter.wait()

        payload = {
            "schema_version": 1,
            "source_id": source_id,
            "frame_index": int(packet.frame_index),
            "frame_name": str(packet.frame_name),
            "image_jpeg_base64": encode_jpeg_base64(packet.frame_bgr, jpeg_quality=jpeg_quality),
            "return_annotated": save_annotated,
        }

        try:
            response = post_json(
                endpoint=stream_endpoint,
                payload=payload,
                timeout_sec=timeout_sec,
                api_key=stream_api_key,
            )
        except TransportError as exc:
            remote_failures += 1
            logger.warn(
                "deploy.edge.stream.remote_failed",
                "Failed to push frame to remote",
                run_id=run_id,
                source_id=source_id,
                frame_index=packet.frame_index,
                error=str(exc),
            )
            continue

        if response.get("ok") is not True:
            remote_failures += 1
            logger.warn(
                "deploy.edge.stream.remote_rejected",
                "Remote returned non-ok payload",
                run_id=run_id,
                source_id=source_id,
                frame_index=packet.frame_index,
                remote_payload=str(response)[:500],
            )
            continue

        counts_by_class = _safe_counts(response.get("counts_by_class"))
        total_detections = int(response.get("total_detections", sum(counts_by_class.values())))
        latency_ms = float(response.get("latency_ms", 0.0))
        event = StatsEvent.now(
            source_id=source_id,
            total_detections=total_detections,
            counts_by_class=counts_by_class,
            latency_ms=latency_ms,
        )
        append_stats_snapshot(snapshot_path, event)

        if save_annotated and isinstance(response.get("annotated_jpeg_base64"), str):
            try:
                annotated_frame = decode_jpeg_base64(str(response["annotated_jpeg_base64"]))
                if _save_annotated(annotated_frame, output_dir / packet.frame_name):
                    annotated_saved += 1
            except DataValidationError:
                logger.warn(
                    "deploy.edge.stream.annotated_decode_failed",
                    "Failed to decode annotated frame from remote",
                    run_id=run_id,
                    source_id=source_id,
                    frame_index=packet.frame_index,
                )

        for class_name, class_count in counts_by_class.items():
            class_totals[class_name] = class_totals.get(class_name, 0) + class_count

        detections_total += total_detections
        processed += 1
        logger.info(
            "deploy.edge.stream.frame",
            "Frame streamed to remote",
            run_id=run_id,
            source_id=source_id,
            frame_index=packet.frame_index,
            detections=total_detections,
            latency_ms=latency_ms,
            source_path=packet.source_path,
        )

    if attempts == 0:
        raise DataValidationError("no valid frames were processed from deploy source")
    if processed == 0:
        raise DataValidationError("all frames failed to stream to remote")

    return {
        "mode": "stream",
        "source": str(edge_cfg["source"]),
        "source_id": source_id,
        "stream_endpoint": stream_endpoint,
        "stats_snapshot_path": str(snapshot_path),
        "annotated_frames_dir": str(output_dir) if save_annotated else "",
        "stats": {
            "frames_attempted": attempts,
            "frames_processed": processed,
            "detections_total": detections_total,
            "class_totals": class_totals,
            "remote_failures": remote_failures,
            "annotated_saved": annotated_saved,
            "fps_limit": float(edge_cfg["fps_limit"]),
        },
    }
