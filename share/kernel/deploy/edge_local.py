"""Edge local deploy pipeline for Phase 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from share.kernel.deploy.edge_common import FpsLimiter, append_stats_snapshot, iter_source_frames
from share.kernel.infer.local_yolo import LocalYoloInferencer
from share.kernel.transport.stats_http import push_stats_event
from share.types.errors import DataValidationError, TransportError
from share.types.stats import StatsEvent


def _save_annotated(result: Any, target_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False

    try:
        image = result.plot()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(target_path), image))
    except Exception:
        return False


def run_edge_local_deploy(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    edge_cfg = cfg["deploy"]["edge"]
    source_id = str(edge_cfg["source_id"])
    local_model = Path(edge_cfg["local_model"])

    inferencer = LocalYoloInferencer(
        model_path=local_model,
        class_names=list(cfg["class_map"]["names"]),
        confidence=float(edge_cfg["confidence"]),
        img_size=int(cfg["train"]["img_size"]),
        device=str(cfg["train"]["device"]),
    )

    run_id = str(run_ctx["run_id"])
    run_dir = Path(str(run_ctx["run_dir"]))
    workdir = Path(cfg["workspace"]["root"])
    logger = run_ctx["logger"]

    save_annotated = bool(edge_cfg["save_annotated"])
    max_frames = int(edge_cfg["max_frames"])
    limiter = FpsLimiter(float(edge_cfg["fps_limit"]))
    endpoint = str(edge_cfg["stats_endpoint"])
    api_key = str(edge_cfg["api_key"])
    timeout_sec = float(edge_cfg["stats_timeout_sec"])

    snapshot_path = run_dir / "stats.jsonl"
    output_dir = workdir / "outputs" / run_id / "annotated_frames"

    processed = 0
    detections_total = 0
    stats_sent = 0
    stats_failed = 0
    annotated_saved = 0
    class_totals: dict[str, int] = {}

    frame_iterator = iter_source_frames(edge_cfg)
    for packet in frame_iterator:
        if max_frames > 0 and processed >= max_frames:
            break

        limiter.wait()
        detections, result, latency_ms = inferencer.infer_frame(packet.frame_bgr)
        detections_total += len(detections)

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
                    api_key=api_key,
                    timeout_sec=timeout_sec,
                )
                stats_sent += 1
            except TransportError as exc:
                stats_failed += 1
                logger.warn(
                    "deploy.edge.stats_push.failed",
                    "Failed to push stats event",
                    run_id=run_id,
                    source_id=source_id,
                    frame_index=packet.frame_index,
                    error=str(exc),
                )

        if save_annotated and _save_annotated(result, output_dir / packet.frame_name):
            annotated_saved += 1

        logger.info(
            "deploy.edge.frame",
            "Frame processed",
            run_id=run_id,
            source_id=source_id,
            frame_index=packet.frame_index,
            detections=len(detections),
            latency_ms=latency_ms,
            source_path=packet.source_path,
        )
        processed += 1

    if processed == 0:
        raise DataValidationError("no valid frames were processed from deploy source")

    return {
        "mode": "local",
        "source": str(edge_cfg["source"]),
        "source_id": source_id,
        "model_path": str(local_model),
        "stats_snapshot_path": str(snapshot_path),
        "annotated_frames_dir": str(output_dir) if save_annotated else "",
        "stats_endpoint": endpoint,
        "stats": {
            "frames_processed": processed,
            "detections_total": detections_total,
            "class_totals": class_totals,
            "stats_sent": stats_sent,
            "stats_failed": stats_failed,
            "annotated_saved": annotated_saved,
            "fps_limit": float(edge_cfg["fps_limit"]),
        },
    }
