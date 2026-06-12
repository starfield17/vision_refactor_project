"""Inference backends.

Keep imports lazy so optional/heavy runtimes (torchvision, onnxruntime,
ultralytics, transformers) are only imported by the backend that needs them.
"""

from __future__ import annotations

__all__ = [
    "LocalFasterRCNNInferencer",
    "LocalFasterRCNNOnnxInferencer",
    "LocalYoloInferencer",
    "LocateAnythingInferencer",
    "ResolvedFrameInferencer",
    "create_frame_inferencer",
]


def __getattr__(name: str):
    if name == "LocalFasterRCNNInferencer":
        from .faster_rcnn import LocalFasterRCNNInferencer

        return LocalFasterRCNNInferencer
    if name == "LocalFasterRCNNOnnxInferencer":
        from .faster_rcnn_onnx import LocalFasterRCNNOnnxInferencer

        return LocalFasterRCNNOnnxInferencer
    if name == "LocalYoloInferencer":
        from .local_yolo import LocalYoloInferencer

        return LocalYoloInferencer
    if name == "LocateAnythingInferencer":
        from .locate_anything import LocateAnythingInferencer

        return LocateAnythingInferencer
    if name in {"ResolvedFrameInferencer", "create_frame_inferencer"}:
        from .factory import ResolvedFrameInferencer, create_frame_inferencer

        return {
            "ResolvedFrameInferencer": ResolvedFrameInferencer,
            "create_frame_inferencer": create_frame_inferencer,
        }[name]
    raise AttributeError(name)
