"""ONNX Runtime inferencer for TorchVision Faster R-CNN exports."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.types.detection import Detection
from common.types.errors import DataValidationError


@dataclass(slots=True)
class AnnotatedFrameResult:
    annotated_bgr: Any

    def plot(self) -> Any:
        return self.annotated_bgr


def _resolve_ort_providers(device: str) -> list[str]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise DataValidationError(f"onnxruntime is required for faster_rcnn ONNX infer: {exc}") from exc

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass

    available = set(ort.get_available_providers())
    providers: list[str] = []
    if device.lower().strip().startswith("cuda") and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = sorted(available)
    if not providers:
        raise DataValidationError("onnxruntime has no available execution providers")
    return providers


class LocalFasterRCNNOnnxInferencer:
    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        confidence: float,
        device: str,
    ) -> None:
        if not model_path.exists():
            raise DataValidationError(f"faster_rcnn ONNX model file not found: {model_path}")

        try:
            import onnxruntime as ort
        except Exception as exc:
            raise DataValidationError(f"onnxruntime is required for faster_rcnn ONNX infer: {exc}") from exc

        self.class_names = class_names
        self.confidence = float(confidence)
        self.providers = _resolve_ort_providers(device)
        self.session = ort.InferenceSession(str(model_path), providers=self.providers)
        inputs = self.session.get_inputs()
        if not inputs:
            raise DataValidationError("faster_rcnn ONNX model has no inputs")
        self.input_name = str(inputs[0].name)

    def _parse_outputs(
        self,
        outputs: list[Any],
        frame_bgr: Any,
    ) -> tuple[list[Detection], Any]:
        if len(outputs) < 3:
            raise DataValidationError(
                f"faster_rcnn ONNX expected 3 outputs, got {len(outputs)}"
            )

        boxes = np.asarray(outputs[0])
        labels = np.asarray(outputs[1])
        scores = np.asarray(outputs[2])
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise DataValidationError(f"invalid faster_rcnn boxes output shape: {boxes.shape}")
        if labels.ndim != 1 or scores.ndim != 1:
            raise DataValidationError(
                f"invalid faster_rcnn labels/scores output shapes: {labels.shape}/{scores.shape}"
            )

        height, width = frame_bgr.shape[:2]
        detections: list[Detection] = []
        annotated = frame_bgr.copy()

        for raw_box, raw_label, raw_score in zip(boxes, labels, scores):
            score = float(raw_score)
            if score < self.confidence:
                continue
            score = max(0.0, min(1.0, score))

            class_id = int(raw_label) - 1
            if class_id < 0 or class_id >= len(self.class_names):
                continue

            x1, y1, x2, y2 = [float(v) for v in raw_box.tolist()]
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue

            detection = Detection(
                schema_version=1,
                class_id=class_id,
                class_name=self.class_names[class_id],
                score=score,
                bbox_xyxy=[x1, y1, x2, y2],
            )
            detection.validate()
            detections.append(detection)

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (24, 170, 255), 2)
            cv2.putText(
                annotated,
                f"{detection.class_name}:{detection.score:.2f}",
                (int(x1), max(12, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (24, 170, 255),
                1,
                cv2.LINE_AA,
            )

        return detections, annotated

    def infer_frame(self, frame_bgr: Any) -> tuple[list[Detection], AnnotatedFrameResult, float]:
        if frame_bgr is None or not hasattr(frame_bgr, "shape"):
            raise DataValidationError("invalid frame for faster_rcnn ONNX infer")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb).transpose(2, 0, 1).astype(np.float32) / 255.0
        batched = np.expand_dims(tensor, axis=0)

        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: batched})
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        detections, annotated = self._parse_outputs(outputs=outputs, frame_bgr=frame_bgr)
        return detections, AnnotatedFrameResult(annotated), latency_ms

    def infer_image(self, image_path: Path) -> tuple[list[Detection], Any, Any]:
        original = cv2.imread(str(image_path))
        if original is None:
            raise DataValidationError(f"failed to read image: {image_path}")

        detections, result, _latency_ms = self.infer_frame(original)
        return detections, original, result.plot()
