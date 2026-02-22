"""Local model inferencer for deploy edge local mode."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from share.types.detection import Detection
from share.types.errors import DataValidationError


class LocalYoloInferencer:
    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        confidence: float,
        img_size: int,
        device: str,
    ) -> None:
        if not model_path.exists():
            raise DataValidationError(f"local model not found: {model_path}")
        self.class_names = class_names
        self.confidence = confidence
        self.img_size = img_size
        self.device = device

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise DataValidationError(f"ultralytics is required for local infer: {exc}") from exc

        self.model = YOLO(str(model_path), task="detect")

    def infer_frame(self, frame_bgr: Any) -> tuple[list[Detection], Any, float]:
        start = time.perf_counter()
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
            stream=False,
            batch=1,
        )
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        if len(results) != 1:
            raise DataValidationError(f"prediction result count mismatch: results={len(results)}")

        result = results[0]
        detections: list[Detection] = []
        boxes = result.boxes

        if boxes is not None:
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                if class_id < 0 or class_id >= len(self.class_names):
                    raise DataValidationError(f"predicted class_id out of range: {class_id}")
                detection = Detection(
                    schema_version=1,
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    score=float(boxes.conf[i].item()),
                    bbox_xyxy=[float(v) for v in boxes.xyxy[i].tolist()],
                )
                detection.validate()
                detections.append(detection)

        return detections, result, latency_ms
