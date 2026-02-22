"""Inference backends."""

from .faster_rcnn import LocalFasterRCNNInferencer
from .local_yolo import LocalYoloInferencer

__all__ = ["LocalFasterRCNNInferencer", "LocalYoloInferencer"]
