"""Local Faster R-CNN inferencer for autolabel/deploy paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

from share.types.detection import Detection
from share.types.errors import DataValidationError


def _build_model(variant: str, num_classes: int) -> FasterRCNN:
    num_classes_with_background = num_classes + 1
    if variant == "resnet50_fpn":
        return fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes_with_background,
        )
    if variant == "resnet50_fpn_v2":
        return fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes_with_background,
        )
    if variant == "mobilenet_v3":
        return fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes_with_background,
        )
    if variant == "resnet18_fpn":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet18",
            weights=None,
            trainable_layers=3,
        )
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        return FasterRCNN(
            backbone=backbone,
            num_classes=num_classes_with_background,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )
    raise DataValidationError(f"unsupported faster_rcnn variant: {variant}")


def _resolve_device(device_str: str) -> torch.device:
    requested = device_str.lower().strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(device_str)
    except Exception:
        return torch.device("cpu")


class LocalFasterRCNNInferencer:
    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        default_variant: str,
        confidence: float,
        device: str,
    ) -> None:
        if not model_path.exists():
            raise DataValidationError(f"faster_rcnn model file not found: {model_path}")

        self.class_names = class_names
        self.confidence = float(confidence)
        self.device = _resolve_device(device)

        checkpoint = torch.load(str(model_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            variant = str(checkpoint.get("variant", default_variant))
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            variant = default_variant
        else:
            raise DataValidationError("invalid faster_rcnn checkpoint format")

        self.model = _build_model(variant=variant, num_classes=len(class_names))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def infer_image(self, image_path: Path) -> tuple[list[Detection], Any, Any]:
        original = cv2.imread(str(image_path))
        if original is None:
            raise DataValidationError(f"failed to read image: {image_path}")

        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0

        with torch.no_grad():
            outputs = self.model([tensor.to(self.device)])
        if not outputs:
            return [], original, original.copy()

        output = outputs[0]
        boxes = output.get("boxes")
        labels = output.get("labels")
        scores = output.get("scores")
        if boxes is None or labels is None or scores is None:
            return [], original, original.copy()

        detections: list[Detection] = []
        annotated = original.copy()
        for idx in range(len(boxes)):
            score = float(scores[idx].item())
            if score < self.confidence:
                continue

            class_id = int(labels[idx].item()) - 1
            if class_id < 0 or class_id >= len(self.class_names):
                continue

            x1, y1, x2, y2 = [float(v) for v in boxes[idx].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue

            det = Detection(
                schema_version=1,
                class_id=class_id,
                class_name=self.class_names[class_id],
                score=score,
                bbox_xyxy=[x1, y1, x2, y2],
            )
            det.validate()
            detections.append(det)

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (24, 170, 255), 2)
            cv2.putText(
                annotated,
                f"{det.class_name}:{det.score:.2f}",
                (int(x1), max(12, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (24, 170, 255),
                1,
                cv2.LINE_AA,
            )

        return detections, original, annotated
