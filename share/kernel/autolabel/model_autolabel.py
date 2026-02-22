"""Model-based autolabel pipeline (Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from share.kernel.infer.faster_rcnn import LocalFasterRCNNInferencer
from share.kernel.media.preview import save_side_by_side_preview
from share.types.detection import Detection
from share.types.errors import DataValidationError
from share.types.label import LabelRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise DataValidationError(f"unlabeled images directory not found: {images_dir}")
    images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise DataValidationError(f"no images found under: {images_dir}")
    return images


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_label(path: Path) -> LabelRecord:
    return LabelRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _dedupe_detections(detections: list[Detection]) -> list[Detection]:
    seen: set[tuple[int, float, float, float, float, float]] = set()
    merged: list[Detection] = []
    for det in detections:
        key = (
            det.class_id,
            round(det.score, 4),
            round(det.bbox_xyxy[0], 2),
            round(det.bbox_xyxy[1], 2),
            round(det.bbox_xyxy[2], 2),
            round(det.bbox_xyxy[3], 2),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(det)
    return merged


def _resolve_label_record(
    label_path: Path,
    incoming: LabelRecord,
    on_conflict: str,
) -> tuple[LabelRecord | None, str]:
    if not label_path.exists():
        return incoming, "created"

    if on_conflict == "skip":
        return None, "skipped"
    if on_conflict == "overwrite":
        return incoming, "overwritten"
    if on_conflict == "merge":
        existing = _load_label(label_path)
        merged = LabelRecord(
            schema_version=1,
            image_path=incoming.image_path,
            source=incoming.source,
            detections=_dedupe_detections([*existing.detections, *incoming.detections]),
        )
        return merged, "merged"

    raise DataValidationError(f"unsupported on_conflict strategy: {on_conflict}")


def _result_to_label(
    result: Any,
    class_names: list[str],
    source: str,
) -> LabelRecord:
    detections: list[Detection] = []

    boxes = result.boxes
    if boxes is not None:
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            if class_id < 0 or class_id >= len(class_names):
                raise DataValidationError(f"predicted class_id out of range: {class_id}")
            xyxy = [float(v) for v in boxes.xyxy[i].tolist()]
            det = Detection(
                schema_version=1,
                class_id=class_id,
                class_name=class_names[class_id],
                score=score,
                bbox_xyxy=xyxy,
            )
            det.validate()
            detections.append(det)

    label = LabelRecord(
        schema_version=1,
        image_path=str(Path(result.path).resolve()),
        source=source,
        detections=detections,
    )
    label.validate()
    return label


def _iter_batches(items: list[Path], batch_size: int) -> list[list[Path]]:
    step = max(1, int(batch_size))
    return [items[i : i + step] for i in range(0, len(items), step)]


def run_model_autolabel(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    class_names = list(cfg["class_map"]["names"])
    model_cfg = cfg["autolabel"]["model"]
    model_backend = str(model_cfg["backend"])
    model_path = Path(model_cfg["onnx_model"])
    if not model_path.exists():
        raise DataValidationError(f"autolabel model not found: {model_path}")

    images_dir = Path(cfg["data"]["unlabeled_dir"])
    labeled_dir = Path(cfg["data"]["labeled_dir"])
    image_paths = _list_images(images_dir)

    run_id = str(run_ctx["run_id"])
    run_outputs_dir = Path(cfg["workspace"]["root"]) / "outputs" / run_id
    annotated_dir = run_outputs_dir / "annotated_frames"

    autolabel_cfg = cfg["autolabel"]
    batch = int(autolabel_cfg["batch_size"])
    conf = float(autolabel_cfg["confidence"])
    visualize = bool(autolabel_cfg["visualize"])
    on_conflict = str(autolabel_cfg["on_conflict"])

    yolo_model = None
    faster_inferencer = None
    if model_backend == "yolo":
        yolo_model = YOLO(str(model_path), task="detect")
    elif model_backend == "faster_rcnn":
        faster_inferencer = LocalFasterRCNNInferencer(
            model_path=model_path,
            class_names=class_names,
            default_variant=str(cfg["train"]["faster_rcnn"]["variant"]),
            confidence=conf,
            device=str(cfg["train"]["device"]),
        )
    else:
        raise DataValidationError(f"unsupported autolabel.model.backend: {model_backend}")

    created = 0
    overwritten = 0
    merged = 0
    skipped = 0
    viz_saved = 0
    total_boxes = 0
    label_files: list[str] = []
    effective_batch_size = 1

    def _persist_result(
        label: LabelRecord,
        preview_payload: tuple[Any, Any] | None,
    ) -> None:
        nonlocal created, overwritten, merged, skipped, viz_saved, total_boxes
        total_boxes += len(label.detections)

        stem = Path(label.image_path).stem
        label_path = labeled_dir / f"{stem}.json"
        resolved, action = _resolve_label_record(label_path, label, on_conflict=on_conflict)

        if resolved is None:
            skipped += 1
            return

        _write_json(label_path, resolved.to_dict())
        label_files.append(str(label_path))

        if action == "created":
            created += 1
        elif action == "overwritten":
            overwritten += 1
        elif action == "merged":
            merged += 1

        if visualize and preview_payload is not None:
            image_name = Path(label.image_path).name
            if save_side_by_side_preview(
                original_bgr=preview_payload[0],
                annotated_bgr=preview_payload[1],
                save_path=annotated_dir / image_name,
                left_title="Original",
                right_title=f"{model_backend} Labels",
            ):
                viz_saved += 1

    if model_backend == "yolo" and yolo_model is not None:
        effective_batch_size = max(1, batch)
        for image_batch in _iter_batches(image_paths, effective_batch_size):
            results = yolo_model.predict(
                source=[str(p) for p in image_batch],
                conf=conf,
                imgsz=int(cfg["train"]["img_size"]),
                device=str(cfg["train"]["device"]),
                verbose=False,
                stream=False,
                batch=effective_batch_size,
            )
            if len(results) != len(image_batch):
                raise DataValidationError(
                    "prediction result count mismatch for batch: "
                    f"inputs={len(image_batch)}, results={len(results)}"
                )

            for image_path, result in zip(image_batch, results):
                label = _result_to_label(
                    result=result,
                    class_names=class_names,
                    source=f"autolabel:model:{model_backend}",
                )
                preview_payload: tuple[Any, Any] | None = None
                if visualize:
                    annotated = result.plot()
                    original = getattr(result, "orig_img", None)
                    if original is None:
                        import cv2

                        original = cv2.imread(str(image_path))
                    if original is not None:
                        preview_payload = (original, annotated)
                _persist_result(label=label, preview_payload=preview_payload)
    elif model_backend == "faster_rcnn" and faster_inferencer is not None:
        for image_path in image_paths:
            detections, original, annotated = faster_inferencer.infer_image(image_path=image_path)
            label = LabelRecord(
                schema_version=1,
                image_path=str(image_path.resolve()),
                source=f"autolabel:model:{model_backend}",
                detections=detections,
            )
            label.validate()
            preview_payload = (original, annotated) if visualize else None
            _persist_result(label=label, preview_payload=preview_payload)
    else:
        raise DataValidationError(f"backend runtime not initialized: {model_backend}")

    stats = {
        "images_total": len(image_paths),
        "labels_created": created,
        "labels_overwritten": overwritten,
        "labels_merged": merged,
        "labels_skipped": skipped,
        "detections_total": total_boxes,
        "visualizations_saved": viz_saved,
    }

    return {
        "mode": "model",
        "model_backend": model_backend,
        "model_path": str(model_path),
        "input_images_dir": str(images_dir),
        "output_labels_dir": str(labeled_dir),
        "run_outputs_dir": str(run_outputs_dir),
        "stats": stats,
        "label_files": label_files,
        "config": {
            "confidence": conf,
            "batch_size": batch,
            "effective_batch_size": effective_batch_size,
            "backend": model_backend,
            "on_conflict": on_conflict,
            "visualize": visualize,
        },
    }
