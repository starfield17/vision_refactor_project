"""Inference backends."""

from .faster_rcnn import LocalFasterRCNNInferencer
from .faster_rcnn_onnx import LocalFasterRCNNOnnxInferencer
from .factory import ResolvedFrameInferencer, create_frame_inferencer
from .local_yolo import LocalYoloInferencer

__all__ = [
    "LocalFasterRCNNInferencer",
    "LocalFasterRCNNOnnxInferencer",
    "LocalYoloInferencer",
    "ResolvedFrameInferencer",
    "create_frame_inferencer",
]
