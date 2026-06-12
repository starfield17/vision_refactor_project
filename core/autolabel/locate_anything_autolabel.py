"""LocateAnything autolabel pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.infer.locate_anything import LocateAnythingInferencer
from core.media.preview import save_side_by_side_preview
from common.types.detection import Detection
from common.types.errors import DataValidationError
from common.types.label import LabelRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(images_dir: Path, max_images: int = 0) -> list[Path]:
    if not images_dir.exists():
        raise DataValidationError(f"unlabeled images directory not found: {images_dir}")
    images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images > 0:
        images = images[:max_images]
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


def run_locate_anything_autolabel(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    images_dir = Path(cfg["data"]["unlabeled_dir"])
    labeled_dir = Path(cfg["data"]["labeled_dir"])
    locate_cfg = dict(cfg.get("locate_anything") or {})
    image_paths = _list_images(images_dir, max_images=int(locate_cfg.get("max_images", 0)))

    run_id = str(run_ctx["run_id"])
    run_outputs_dir = Path(cfg["workspace"]["root"]) / "outputs" / run_id
    annotated_dir = run_outputs_dir / "annotated_frames"
    raw_dir = run_outputs_dir / "locate_anything_raw"

    autolabel_cfg = cfg["autolabel"]
    conf = float(autolabel_cfg["confidence"])
    visualize = bool(autolabel_cfg["visualize"])
    on_conflict = str(autolabel_cfg["on_conflict"])

    inferencer = LocateAnythingInferencer.from_config(cfg, confidence=conf)

    created = 0
    overwritten = 0
    merged = 0
    skipped = 0
    viz_saved = 0
    total_boxes = 0
    raw_saved = 0
    label_files: list[str] = []

    for image_path in image_paths:
        detections, original, annotated, raw_answers = inferencer.infer_image(image_path=image_path)
        label = LabelRecord(
            schema_version=1,
            image_path=str(image_path.resolve()),
            source=f"autolabel:locate_anything:{inferencer.model_id}",
            detections=detections,
        )
        label.validate()
        total_boxes += len(label.detections)

        stem = image_path.stem
        label_path = labeled_dir / f"{stem}.json"
        resolved, action = _resolve_label_record(label_path, label, on_conflict=on_conflict)
        if resolved is None:
            skipped += 1
            continue

        _write_json(label_path, resolved.to_dict())
        label_files.append(str(label_path))
        if action == "created":
            created += 1
        elif action == "overwritten":
            overwritten += 1
        elif action == "merged":
            merged += 1

        _write_json(raw_dir / f"{stem}.json", {"image_path": str(image_path), "answers": raw_answers})
        raw_saved += 1

        if visualize:
            if save_side_by_side_preview(
                original_bgr=original,
                annotated_bgr=annotated,
                save_path=annotated_dir / image_path.name,
                left_title="Original",
                right_title="LocateAnything Labels",
            ):
                viz_saved += 1

    stats = {
        "images_total": len(image_paths),
        "labels_created": created,
        "labels_overwritten": overwritten,
        "labels_merged": merged,
        "labels_skipped": skipped,
        "detections_total": total_boxes,
        "visualizations_saved": viz_saved,
        "raw_answers_saved": raw_saved,
    }

    return {
        "mode": "locate_anything",
        "model_backend": "locate_anything",
        "model_id": inferencer.model_id,
        "input_images_dir": str(images_dir),
        "output_labels_dir": str(labeled_dir),
        "run_outputs_dir": str(run_outputs_dir),
        "stats": stats,
        "label_files": label_files,
        "config": {
            "confidence": conf,
            "effective_batch_size": 1,
            "generation_mode": str(locate_cfg.get("generation_mode", "hybrid")),
            "on_conflict": on_conflict,
            "visualize": visualize,
        },
    }
