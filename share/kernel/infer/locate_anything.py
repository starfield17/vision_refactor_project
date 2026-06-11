"""LocateAnything-backed grounding inferencer.

This module keeps NVIDIA LocateAnything behind the same Detection contract used by
YOLO/Faster-RCNN deployment and autolabel flows.  It intentionally queries one class
at a time so the class mapping remains deterministic for downstream YOLO-style labels.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from share.types.detection import Detection
from share.types.errors import DataValidationError


_BOX_TOKEN_RE = re.compile(r"<box>\s*<([0-9]+)>\s*<([0-9]+)>\s*<([0-9]+)>\s*<([0-9]+)>\s*</box>")
_BOX_TEXT_RE = re.compile(
    r"<box>\s*([0-9]+(?:\.[0-9]+)?)\s*[, ]\s*([0-9]+(?:\.[0-9]+)?)\s*[, ]\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*[, ]\s*([0-9]+(?:\.[0-9]+)?)\s*</box>",
    flags=re.IGNORECASE,
)


def _clip_box(box: list[float], width: int, height: int) -> list[float] | None:
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_locate_anything_boxes(answer: str, image_width: int, image_height: int) -> list[list[float]]:
    """Parse LocateAnything ``<box>`` output into pixel-coordinate xyxy boxes.

    The official worker emits quantized integer coordinates in [0, 1000].  A looser
    comma/space-separated variant is accepted so remote wrappers can normalize output
    without breaking this parser.
    """

    boxes: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()

    def add(raw: list[float], normalized_1000: bool) -> None:
        box = raw
        if normalized_1000:
            box = [
                raw[0] / 1000.0 * image_width,
                raw[1] / 1000.0 * image_height,
                raw[2] / 1000.0 * image_width,
                raw[3] / 1000.0 * image_height,
            ]
        clipped = _clip_box(box, width=image_width, height=image_height)
        if clipped is None:
            return
        key = tuple(round(v, 2) for v in clipped)
        if key in seen:
            return
        seen.add(key)
        boxes.append(clipped)

    for match in _BOX_TOKEN_RE.finditer(answer):
        add([float(v) for v in match.groups()], normalized_1000=True)

    for match in _BOX_TEXT_RE.finditer(answer):
        raw = [float(v) for v in match.groups()]
        normalized_1000 = all(0.0 <= v <= 1000.0 for v in raw)
        add(raw, normalized_1000=normalized_1000)

    return boxes


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _nms_by_class(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if iou_threshold <= 0:
        return detections
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(det.class_id != old.class_id or _iou(det, old) <= iou_threshold for old in kept):
            kept.append(det)
    return kept


@dataclass(slots=True)
class LocateAnythingResult:
    frame_bgr: Any
    detections: list[Detection]
    raw_answers: dict[str, str]

    def plot(self) -> Any:
        try:
            import cv2
        except Exception as exc:
            raise DataValidationError(f"opencv is required to plot LocateAnything result: {exc}") from exc

        annotated = self.frame_bgr.copy()
        for det in self.detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
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
        return annotated


class LocateAnythingInferencer:
    """Stateful LocateAnything worker adapted to the project Detection API."""

    def __init__(
        self,
        *,
        model: str,
        class_names: list[str],
        confidence: float = 0.5,
        device: str = "auto",
        dtype: str = "auto",
        generation_mode: str = "hybrid",
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        prompt_template: str = "Locate all the instances that match the following description: {class_name}.",
        nms_iou: float = 0.65,
        default_score: float = 1.0,
        verbose: bool = False,
    ) -> None:
        if not model:
            raise DataValidationError("locate_anything.model must not be empty")
        if generation_mode not in {"fast", "slow", "hybrid"}:
            raise DataValidationError("locate_anything.generation_mode must be fast, slow, or hybrid")
        if "{class_name}" not in prompt_template:
            raise DataValidationError("locate_anything.prompt_template must include {class_name}")

        self.model_id = model
        self.class_names = class_names
        self.confidence = float(confidence)
        self.generation_mode = generation_mode
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.prompt_template = prompt_template
        self.nms_iou = float(nms_iou)
        self.default_score = max(0.0, min(1.0, float(default_score)))
        self.verbose = bool(verbose)

        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except Exception as exc:
            raise DataValidationError(
                "LocateAnything local inference requires torch and transformers. "
                "Install the optional LocateAnything dependencies first."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype == "auto":
            torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        else:
            torch_dtype = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }.get(dtype.lower())
            if torch_dtype is None:
                raise DataValidationError("locate_anything.dtype must be auto, float32, float16, or bfloat16")

        self.torch = torch
        self.dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], *, confidence: float) -> "LocateAnythingInferencer":
        la_cfg = dict(cfg.get("locate_anything") or {})
        return cls(
            model=str(la_cfg.get("model", "nvidia/LocateAnything-3B")),
            class_names=list(cfg["class_map"]["names"]),
            confidence=float(confidence),
            device=str(la_cfg.get("device", cfg.get("train", {}).get("device", "auto"))),
            dtype=str(la_cfg.get("dtype", "auto")),
            generation_mode=str(la_cfg.get("generation_mode", "hybrid")),
            max_new_tokens=int(la_cfg.get("max_new_tokens", 8192)),
            temperature=float(la_cfg.get("temperature", 0.0)),
            prompt_template=str(
                la_cfg.get(
                    "prompt_template",
                    "Locate all the instances that match the following description: {class_name}.",
                )
            ),
            nms_iou=float(la_cfg.get("nms_iou", 0.65)),
            default_score=float(la_cfg.get("default_score", 1.0)),
            verbose=bool(la_cfg.get("verbose", False)),
        )

    def _predict_answer(self, image: Any, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(
            self.device
        )
        pixel_values = inputs["pixel_values"].to(self.dtype)
        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            generation_mode=self.generation_mode,
            temperature=self.temperature,
            do_sample=self.temperature > 0.0,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=self.verbose,
        )
        answer = response[0] if isinstance(response, tuple) else response
        if isinstance(answer, list) and answer:
            answer = answer[0]
        return str(answer)

    def infer_frame(self, frame_bgr: Any) -> tuple[list[Detection], LocateAnythingResult, float]:
        try:
            import cv2
            from PIL import Image
        except Exception as exc:
            raise DataValidationError(f"Pillow and opencv are required for LocateAnything infer: {exc}") from exc

        if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
            raise DataValidationError("invalid frame payload for LocateAnything infer")
        height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        if width <= 0 or height <= 0:
            raise DataValidationError("frame width/height must be positive")

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        start = time.perf_counter()
        detections: list[Detection] = []
        raw_answers: dict[str, str] = {}
        for class_id, class_name in enumerate(self.class_names):
            question = self.prompt_template.format(class_name=class_name)
            answer = self._predict_answer(pil_image, question)
            raw_answers[class_name] = answer
            for box in parse_locate_anything_boxes(answer, image_width=width, image_height=height):
                det = Detection(
                    schema_version=1,
                    class_id=class_id,
                    class_name=class_name,
                    score=self.default_score,
                    bbox_xyxy=box,
                )
                det.validate()
                detections.append(det)

        detections = _nms_by_class(detections, iou_threshold=self.nms_iou)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        result = LocateAnythingResult(
            frame_bgr=frame_bgr,
            detections=detections,
            raw_answers=raw_answers,
        )
        return detections, result, latency_ms

    def infer_image(self, image_path: Path) -> tuple[list[Detection], Any, Any, dict[str, str]]:
        try:
            import cv2
        except Exception as exc:
            raise DataValidationError(f"opencv is required for LocateAnything image infer: {exc}") from exc

        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            raise DataValidationError(f"failed to read image: {image_path}")
        detections, result, _latency_ms = self.infer_frame(frame_bgr)
        return detections, frame_bgr, result.plot(), result.raw_answers
