"""Backend-aware local inferencer construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from share.kernel.model_manifest import resolve_model_identity
from share.types.errors import DataValidationError


@dataclass(slots=True)
class ResolvedFrameInferencer:
    inferencer: Any
    model_meta: dict[str, str]


def create_frame_inferencer(
    *,
    model_path: Path,
    cfg: dict[str, Any],
    confidence: float,
    default_backend: str = "yolo",
    default_model_id: str = "",
) -> ResolvedFrameInferencer:
    model_meta = resolve_model_identity(
        model_path,
        default_backend=default_backend,
        default_model_id=default_model_id or f"{default_backend}:{model_path.stem}",
    )
    backend = model_meta["backend"]
    class_names = list(cfg["class_map"]["names"])
    device = str(cfg["train"]["device"])

    if backend == "yolo":
        from share.kernel.infer.local_yolo import LocalYoloInferencer

        inferencer = LocalYoloInferencer(
            model_path=model_path,
            class_names=class_names,
            confidence=float(confidence),
            img_size=int(cfg["train"]["img_size"]),
            device=device,
        )
        return ResolvedFrameInferencer(inferencer=inferencer, model_meta=model_meta)

    if backend == "faster_rcnn":
        if model_path.suffix.lower() == ".onnx":
            from share.kernel.infer.faster_rcnn_onnx import LocalFasterRCNNOnnxInferencer

            inferencer = LocalFasterRCNNOnnxInferencer(
                model_path=model_path,
                class_names=class_names,
                confidence=float(confidence),
                device=device,
            )
        else:
            from share.kernel.infer.faster_rcnn import LocalFasterRCNNInferencer

            inferencer = LocalFasterRCNNInferencer(
                model_path=model_path,
                class_names=class_names,
                default_variant=str(cfg["train"]["faster_rcnn"]["variant"]),
                confidence=float(confidence),
                device=device,
            )
        return ResolvedFrameInferencer(inferencer=inferencer, model_meta=model_meta)

    raise DataValidationError(f"unsupported model backend for deploy infer: {backend}")
