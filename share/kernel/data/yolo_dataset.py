"""YOLO dataset scanning and class map validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from share.types.errors import DataValidationError


@dataclass(slots=True)
class YoloDatasetScan:
    dataset_root: Path
    train_images_dir: Path
    val_images_dir: Path
    label_files: int
    image_files: int
    class_histogram: dict[int, int]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _list_label_files(labels_root: Path) -> list[Path]:
    if not labels_root.exists():
        return []
    return sorted(p for p in labels_root.rglob("*.txt") if p.is_file())


def _parse_class_ids(label_path: Path) -> list[int]:
    class_ids: list[int] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise DataValidationError(
                f"Invalid YOLO label format: {label_path}:{line_no} expects 5+ columns"
            )
        try:
            class_ids.append(int(parts[0]))
        except ValueError as exc:
            raise DataValidationError(
                f"Invalid class id at {label_path}:{line_no}, got {parts[0]}"
            ) from exc
    return class_ids


def _detect_split_images_dir(dataset_root: Path, split: str) -> Path:
    candidates = [
        dataset_root / "images" / split,
        dataset_root / split / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return dataset_root / "images" / "train2017"


def scan_and_validate_yolo_dataset(
    dataset_root: Path,
    class_map_names: list[str],
    class_map_id_map: dict[str, int],
) -> YoloDatasetScan:
    if not dataset_root.exists():
        raise DataValidationError(f"Dataset root not found: {dataset_root}")

    labels_root = dataset_root / "labels"
    label_files = _list_label_files(labels_root)
    if not label_files:
        raise DataValidationError(f"No label files found under: {labels_root}")

    valid_ids = set(class_map_id_map.values())
    if len(valid_ids) != len(class_map_names):
        raise DataValidationError("class_map id_map and names length mismatch")

    histogram: Counter[int] = Counter()
    for label_file in label_files:
        for class_id in _parse_class_ids(label_file):
            if class_id not in valid_ids:
                raise DataValidationError(
                    f"Label class_id={class_id} in {label_file} is not defined in class_map"
                )
            histogram[class_id] += 1

    train_images_dir = _detect_split_images_dir(dataset_root, "train2017")
    val_images_dir = _detect_split_images_dir(dataset_root, "val2017")

    image_files = _count_images(train_images_dir)
    if image_files == 0:
        raise DataValidationError(f"No images found under: {train_images_dir}")

    return YoloDatasetScan(
        dataset_root=dataset_root,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        label_files=len(label_files),
        image_files=image_files,
        class_histogram=dict(sorted(histogram.items())),
    )
