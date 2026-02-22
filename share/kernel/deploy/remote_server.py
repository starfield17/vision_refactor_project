"""Deploy remote server pipeline for Phase 5."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from share.kernel.deploy.edge_common import append_stats_snapshot
from share.kernel.infer.local_yolo import LocalYoloInferencer
from share.kernel.transport.frame_http import decode_jpeg_base64, encode_jpeg_base64
from share.kernel.transport.stats_http import push_stats_event
from share.types.detection import Detection
from share.types.errors import ConfigError, DataValidationError, TransportError
from share.types.stats import StatsEvent


@dataclass(slots=True)
class _FrameRequest:
    schema_version: int
    source_id: str
    frame_index: int
    frame_name: str
    image_jpeg_base64: str
    return_annotated: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "_FrameRequest":
        try:
            req = cls(
                schema_version=int(payload["schema_version"]),
                source_id=str(payload["source_id"]),
                frame_index=int(payload["frame_index"]),
                frame_name=str(payload["frame_name"]),
                image_jpeg_base64=str(payload["image_jpeg_base64"]),
                return_annotated=bool(payload.get("return_annotated", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"invalid stream frame payload: {exc}") from exc
        req.validate()
        return req

    def validate(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError(
                f"unsupported stream payload schema_version={self.schema_version}"
            )
        if not self.source_id:
            raise DataValidationError("source_id is required")
        if self.frame_index < 0:
            raise DataValidationError("frame_index must be >= 0")
        if not self.frame_name:
            raise DataValidationError("frame_name is required")
        if not self.image_jpeg_base64:
            raise DataValidationError("image_jpeg_base64 is required")


def _safe_counts(detections: list[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return counts


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


def create_remote_app(
    cfg: dict[str, Any],
    run_id: str,
    run_dir: Path,
    logger: Any,
    inferencer: LocalYoloInferencer,
):
    try:
        from fastapi import FastAPI, Header
        from fastapi.responses import JSONResponse
    except Exception as exc:
        raise ConfigError(f"fastapi is required for deploy.remote: {exc}") from exc

    remote_cfg = cfg["deploy"]["remote"]
    stats_endpoint = str(remote_cfg["stats_endpoint"])
    stats_api_key = str(remote_cfg["api_key"])
    stats_timeout_sec = float(remote_cfg["stats_timeout_sec"])
    ingest_api_key = str(remote_cfg["ingest_api_key"])
    max_payload_mb = int(remote_cfg["max_payload_mb"])
    max_payload_len = max_payload_mb * 1024 * 1024
    save_annotated = bool(remote_cfg["save_annotated"])
    jpeg_quality = int(cfg["deploy"]["edge"]["jpeg_quality"])

    snapshot_path = run_dir / "stats.jsonl"
    output_dir = Path(cfg["workspace"]["root"]) / "outputs" / run_id / "annotated_frames"
    counters = {
        "frames_processed": 0,
        "detections_total": 0,
        "stats_sent": 0,
        "stats_failed": 0,
        "annotated_saved": 0,
    }
    counters_lock = threading.Lock()
    infer_lock = threading.Lock()

    app = FastAPI(title="Vision Refactor Remote", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        with counters_lock:
            return {"ok": True, "run_id": run_id, **counters}

    @app.post("/api/v1/frame")
    def push_frame(payload: dict[str, Any], x_api_key: str | None = Header(default=None)):
        if ingest_api_key and x_api_key != ingest_api_key:
            return JSONResponse(status_code=401, content={"ok": False, "error": "unauthorized"})

        try:
            req = _FrameRequest.from_dict(payload)
            if len(req.image_jpeg_base64) > max_payload_len * 2:
                raise DataValidationError("jpeg payload too large")
            frame_bgr = decode_jpeg_base64(req.image_jpeg_base64)
        except DataValidationError as exc:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "validation_failed", "detail": str(exc)},
            )

        with infer_lock:
            detections, result, latency_ms = inferencer.infer_frame(frame_bgr)
        counts_by_class = _safe_counts(detections)
        total_detections = len(detections)
        event = StatsEvent.now(
            source_id=req.source_id,
            total_detections=total_detections,
            counts_by_class=counts_by_class,
            latency_ms=latency_ms,
        )
        append_stats_snapshot(snapshot_path, event)

        if stats_endpoint:
            try:
                push_stats_event(
                    event=event,
                    endpoint=stats_endpoint,
                    api_key=stats_api_key,
                    timeout_sec=stats_timeout_sec,
                )
                with counters_lock:
                    counters["stats_sent"] += 1
            except TransportError as exc:
                with counters_lock:
                    counters["stats_failed"] += 1
                logger.warn(
                    "deploy.remote.stats_push.failed",
                    "Failed to push stats from remote",
                    run_id=run_id,
                    source_id=req.source_id,
                    frame_index=req.frame_index,
                    error=str(exc),
                )

        response: dict[str, Any] = {
            "ok": True,
            "run_id": run_id,
            "source_id": req.source_id,
            "frame_index": req.frame_index,
            "frame_name": req.frame_name,
            "total_detections": total_detections,
            "counts_by_class": counts_by_class,
            "latency_ms": latency_ms,
        }

        if req.return_annotated or save_annotated:
            plotted = result.plot()
            if req.return_annotated:
                response["annotated_jpeg_base64"] = encode_jpeg_base64(
                    plotted,
                    jpeg_quality=jpeg_quality,
                )
            if save_annotated and _save_annotated(plotted, output_dir / req.frame_name):
                with counters_lock:
                    counters["annotated_saved"] += 1

        with counters_lock:
            counters["frames_processed"] += 1
            counters["detections_total"] += total_detections

        logger.info(
            "deploy.remote.frame",
            "Remote frame processed",
            run_id=run_id,
            source_id=req.source_id,
            frame_index=req.frame_index,
            detections=total_detections,
            latency_ms=latency_ms,
        )
        return response

    app.state.remote_counters = counters
    app.state.remote_counters_lock = counters_lock
    return app


def run_remote_deploy(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    remote_cfg = cfg["deploy"]["remote"]
    run_id = str(run_ctx["run_id"])
    run_dir = Path(str(run_ctx["run_dir"]))
    logger = run_ctx["logger"]
    model_path = Path(remote_cfg["model"])

    inferencer = LocalYoloInferencer(
        model_path=model_path,
        class_names=list(cfg["class_map"]["names"]),
        confidence=float(remote_cfg["confidence"]),
        img_size=int(cfg["train"]["img_size"]),
        device=str(cfg["train"]["device"]),
    )

    app = create_remote_app(
        cfg=cfg,
        run_id=run_id,
        run_dir=run_dir,
        logger=logger,
        inferencer=inferencer,
    )

    try:
        import uvicorn
    except Exception as exc:
        raise ConfigError(f"uvicorn is required for deploy.remote: {exc}") from exc

    host = str(remote_cfg["listen_host"])
    port = int(remote_cfg["listen_port"])
    logger.info(
        "deploy.remote.server.start",
        "Deploy remote server starting",
        run_id=run_id,
        host=host,
        port=port,
        model_path=str(model_path),
    )
    uvicorn.run(app, host=host, port=port, log_level="info")

    with app.state.remote_counters_lock:
        counters = dict(app.state.remote_counters)

    return {
        "mode": "remote",
        "source_id": str(remote_cfg["source_id"]),
        "listen_host": host,
        "listen_port": port,
        "model_path": str(model_path),
        "stats_endpoint": str(remote_cfg["stats_endpoint"]),
        "stats_snapshot_path": str(run_dir / "stats.jsonl"),
        "stats": counters,
    }
