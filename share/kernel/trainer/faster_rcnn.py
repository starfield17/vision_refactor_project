"""Faster R-CNN trainer implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
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
from share.types.label import LabelRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class _Sample:
    image_path: Path
    detections: list[Detection]


class _FasterRCNNDataset(Dataset):
    def __init__(self, samples: list[_Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import cv2

        sample = self.samples[index]
        image_bgr = cv2.imread(str(sample.image_path))
        if image_bgr is None:
            raise DataValidationError(f"failed to read image: {sample.image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
        )

        boxes: list[list[float]] = []
        labels: list[int] = []
        for det in sample.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            x1 = max(0.0, min(float(width), float(x1)))
            y1 = max(0.0, min(float(height), float(y1)))
            x2 = max(0.0, min(float(width), float(x2)))
            y2 = max(0.0, min(float(height), float(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(det.class_id) + 1)  # background class is 0 in TorchVision

        if not boxes:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [1]

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1]),
            "iscrowd": torch.zeros((len(boxes_tensor),), dtype=torch.int64),
        }
        return image_tensor, target


def _collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_label(path: Path) -> LabelRecord:
    return LabelRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _iter_label_files(labeled_dir: Path) -> list[Path]:
    if not labeled_dir.exists():
        raise DataValidationError(f"labeled_dir not found: {labeled_dir}")
    files = sorted(p for p in labeled_dir.rglob("*.json") if p.is_file())
    if not files:
        raise DataValidationError(f"no label json files found under: {labeled_dir}")
    return files


def _collect_samples(
    labeled_dir: Path,
    class_count: int,
    max_samples: int,
    logger: Any,
) -> tuple[list[_Sample], dict[str, int]]:
    samples: list[_Sample] = []
    skipped = {
        "invalid_json": 0,
        "missing_image": 0,
        "empty_detections": 0,
    }

    for label_path in _iter_label_files(labeled_dir):
        try:
            label = _load_label(label_path)
        except Exception:
            skipped["invalid_json"] += 1
            continue

        image_path = Path(label.image_path)
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped["missing_image"] += 1
            continue

        valid_dets: list[Detection] = []
        for det in label.detections:
            if det.class_id < 0 or det.class_id >= class_count:
                continue
            if det.bbox_xyxy[2] <= det.bbox_xyxy[0] or det.bbox_xyxy[3] <= det.bbox_xyxy[1]:
                continue
            valid_dets.append(det)

        if not valid_dets:
            skipped["empty_detections"] += 1
            continue

        samples.append(_Sample(image_path=image_path, detections=valid_dets))
        if max_samples > 0 and len(samples) >= max_samples:
            break

    if not samples:
        raise DataValidationError("no valid labeled samples for faster_rcnn training")

    logger.info(
        "train.faster_rcnn.dataset.ready",
        "Faster R-CNN dataset prepared",
        samples=len(samples),
        skipped=skipped,
    )
    return samples, skipped


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


def _resolve_device(device_str: str, logger: Any) -> torch.device:
    requested = device_str.lower().strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warn(
            "train.faster_rcnn.device.fallback",
            "CUDA requested but unavailable, fallback to CPU",
            requested=device_str,
        )
        return torch.device("cpu")
    try:
        return torch.device(device_str)
    except Exception:
        logger.warn(
            "train.faster_rcnn.device.invalid",
            "Invalid device string, fallback to CPU",
            requested=device_str,
        )
        return torch.device("cpu")


def _train_one_epoch(
    model: FasterRCNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    step_count = 0

    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if not torch.isfinite(losses):
            raise DataValidationError(f"non-finite loss encountered: {losses.item()}")

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        loss_sum += float(losses.item())
        step_count += 1

    return loss_sum / max(step_count, 1)


def run_faster_rcnn_train(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    logger = run_ctx["logger"]
    workdir = Path(cfg["workspace"]["root"])
    run_name = cfg["workspace"]["run_name"]
    model_dir = workdir / "models" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(cfg["class_map"]["names"])
    class_id_map = dict(cfg["class_map"]["id_map"])
    faster_cfg = cfg["train"]["faster_rcnn"]
    dry_run = bool(cfg["train"]["dry_run"])

    labeled_dir = Path(cfg["data"]["labeled_dir"])
    max_samples = int(faster_cfg["max_samples"])
    samples, skipped = _collect_samples(
        labeled_dir=labeled_dir,
        class_count=len(class_names),
        max_samples=max_samples,
        logger=logger,
    )

    labels_snapshot_path = model_dir / "labels.json"
    _write_json(
        labels_snapshot_path,
        {
            "schema_version": 1,
            "names": class_names,
            "id_map": class_id_map,
        },
    )

    device = _resolve_device(str(cfg["train"]["device"]), logger=logger)
    variant = str(faster_cfg["variant"])
    model = _build_model(variant=variant, num_classes=len(class_names)).to(device)

    dataloader = DataLoader(
        _FasterRCNNDataset(samples=samples),
        batch_size=max(1, int(cfg["train"]["batch_size"])),
        shuffle=True,
        num_workers=max(0, int(faster_cfg["num_workers"])),
        collate_fn=_collate_fn,
    )

    train_loss_history: list[float] = []
    if not dry_run:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(faster_cfg["lr"]),
            momentum=float(faster_cfg["momentum"]),
            weight_decay=float(faster_cfg["weight_decay"]),
        )
        epochs = int(cfg["train"]["epochs"])
        for epoch in range(epochs):
            epoch_loss = _train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
            )
            train_loss_history.append(epoch_loss)
            logger.info(
                "train.faster_rcnn.epoch.done",
                "Faster R-CNN epoch finished",
                epoch=epoch + 1,
                epochs=epochs,
                loss=round(epoch_loss, 6),
            )

    model_path = model_dir / "model.pt"
    torch.save(
        {
            "backend": "faster_rcnn",
            "variant": variant,
            "class_names": class_names,
            "state_dict": model.state_dict(),
        },
        model_path,
    )

    detections_total = sum(len(sample.detections) for sample in samples)
    export_messages = [
        "faster_rcnn export path currently uses model.pt for downstream integration",
        "onnx/quantization for faster_rcnn is not implemented in this phase",
    ]
    if bool(cfg["export"]["onnx"]):
        logger.warn(
            "train.faster_rcnn.export.onnx.skipped",
            "ONNX export is skipped for faster_rcnn backend",
        )

    return {
        "backend": "faster_rcnn",
        "dry_run": dry_run,
        "train_note": "dry-run: skipped optimization steps" if dry_run else "train finished",
        "dataset": {
            "labeled_dir": str(labeled_dir),
            "samples_used": len(samples),
            "detections_total": detections_total,
            "skipped": skipped,
        },
        "class_map_validation": {
            "ok": True,
            "class_count": len(class_names),
        },
        "weights": {
            "variant": variant,
            "model_pt": str(model_path),
            "labels_snapshot": str(labels_snapshot_path),
            "device": str(device),
        },
        "metrics": {
            "epochs": int(cfg["train"]["epochs"]),
            "train_loss_history": train_loss_history,
        },
        "export": {
            "onnx_path": None,
            "quantized_onnx_path": None,
            "fp16_onnx_path": None,
            "final_infer_model_path": str(model_path),
            "export_ok": False,
            "quantize_ok": False,
            "quantize_strategy": "not-implemented",
            "messages": export_messages,
        },
    }
