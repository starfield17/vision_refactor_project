#!/usr/bin/env python3
"""Download and convert VOC2007 into the project training layouts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
from xml.etree import ElementTree

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_ARCHIVES = {
    "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    "test": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="./work-dir")
    parser.add_argument("--max-train", type=int, default=1500)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--max-unlabeled", type=int, default=30)
    parser.add_argument("--max-deploy", type=int, default=30)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def _download_archives(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, url in VOC_ARCHIVES.items():
        target = raw_dir / Path(url).name
        if not target.exists():
            print(f"download {split}: {url}")
            urlretrieve(url, target)


def _extract_archives(raw_dir: Path) -> None:
    voc_root = raw_dir / "VOCdevkit" / "VOC2007"
    if voc_root.exists():
        return
    for archive in sorted(raw_dir.glob("VOC*.tar")):
        print(f"extract {archive}")
        with tarfile.open(archive) as tar:
            tar.extractall(raw_dir, filter="data")


def _read_split_ids(voc_root: Path, split_name: str, limit: int) -> list[str]:
    split_path = voc_root / "ImageSets" / "Main" / f"{split_name}.txt"
    ids = [
        line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if limit > 0:
        ids = ids[:limit]
    return ids


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _parse_annotation(voc_root: Path, image_id: str) -> tuple[int, int, list[dict[str, object]]]:
    xml_path = voc_root / "Annotations" / f"{image_id}.xml"
    root = ElementTree.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing size in {xml_path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    detections: list[dict[str, object]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in VOC_CLASSES:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        x1 = max(0.0, float(bbox.findtext("xmin", "0")) - 1.0)
        y1 = max(0.0, float(bbox.findtext("ymin", "0")) - 1.0)
        x2 = min(float(width), float(bbox.findtext("xmax", "0")))
        y2 = min(float(height), float(bbox.findtext("ymax", "0")))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = VOC_CLASSES.index(name)
        detections.append(
            {
                "schema_version": 1,
                "class_id": class_id,
                "class_name": name,
                "score": 1.0,
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return width, height, detections


def _write_yolo_label(
    label_path: Path, width: int, height: int, detections: list[dict[str, object]]
) -> None:
    lines: list[str] = []
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]  # type: ignore[index]
        x_center = ((x1 + x2) / 2.0) / float(width)
        y_center = ((y1 + y2) / 2.0) / float(height)
        box_w = (x2 - x1) / float(width)
        box_h = (y2 - y1) / float(height)
        lines.append(
            f"{int(det['class_id'])} {x_center:.8f} {y_center:.8f} {box_w:.8f} {box_h:.8f}"
        )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _convert_split(
    *,
    voc_root: Path,
    image_ids: list[str],
    split_dir_name: str,
    yolo_root: Path,
    labeled_dir: Path | None,
) -> dict[str, int]:
    images_written = 0
    detections_total = 0
    for image_id in image_ids:
        source_image = voc_root / "JPEGImages" / f"{image_id}.jpg"
        target_image = yolo_root / "images" / split_dir_name / f"{image_id}.jpg"
        _copy_or_link(source_image, target_image)
        width, height, detections = _parse_annotation(voc_root, image_id)
        _write_yolo_label(
            yolo_root / "labels" / split_dir_name / f"{image_id}.txt",
            width=width,
            height=height,
            detections=detections,
        )
        if labeled_dir is not None:
            label_record = {
                "schema_version": 1,
                "image_path": str(target_image.resolve()),
                "source": "voc2007",
                "detections": detections,
            }
            label_path = labeled_dir / f"{image_id}.json"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                json.dumps(label_record, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        images_written += 1
        detections_total += len(detections)
    return {"images": images_written, "detections": detections_total}


def _write_dataset_yaml(yolo_root: Path) -> None:
    names_json = json.dumps(VOC_CLASSES, ensure_ascii=True)
    (yolo_root / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {yolo_root.resolve()}",
                "train: images/train2017",
                "val: images/val2017",
                f"nc: {len(VOC_CLASSES)}",
                f"names: {names_json}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _copy_unlabeled_and_deploy(
    *,
    yolo_root: Path,
    unlabeled_dir: Path,
    deploy_dir: Path,
    max_unlabeled: int,
    max_deploy: int,
) -> dict[str, int]:
    source_images = sorted((yolo_root / "images" / "val2017").glob("*.jpg"))
    unlabeled_images = source_images[: max(0, max_unlabeled)]
    deploy_images = source_images[: max(0, max_deploy)]
    for source in unlabeled_images:
        _copy_or_link(source, unlabeled_dir / "images" / source.name)
    for source in deploy_images:
        _copy_or_link(source, deploy_dir / source.name)
    return {"unlabeled_images": len(unlabeled_images), "deploy_images": len(deploy_images)}


def main() -> int:
    args = _parse_args()
    workdir = Path(args.workdir)
    raw_dir = workdir / "datasets" / "raw"
    if not args.skip_download:
        _download_archives(raw_dir)
    _extract_archives(raw_dir)

    voc_root = raw_dir / "VOCdevkit" / "VOC2007"
    yolo_root = workdir / "datasets" / "voc2007_yolo"
    labeled_dir = workdir / "datasets" / "voc2007_labeled"
    unlabeled_dir = workdir / "datasets" / "voc2007_unlabeled"
    deploy_dir = workdir / "datasets" / "voc2007_deploy_images"
    for path in (yolo_root, labeled_dir, unlabeled_dir, deploy_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_ids = _read_split_ids(voc_root, "trainval", args.max_train)
    val_ids = _read_split_ids(voc_root, "test", args.max_val)
    train_stats = _convert_split(
        voc_root=voc_root,
        image_ids=train_ids,
        split_dir_name="train2017",
        yolo_root=yolo_root,
        labeled_dir=labeled_dir,
    )
    val_stats = _convert_split(
        voc_root=voc_root,
        image_ids=val_ids,
        split_dir_name="val2017",
        yolo_root=yolo_root,
        labeled_dir=None,
    )
    _write_dataset_yaml(yolo_root)
    copy_stats = _copy_unlabeled_and_deploy(
        yolo_root=yolo_root,
        unlabeled_dir=unlabeled_dir,
        deploy_dir=deploy_dir,
        max_unlabeled=args.max_unlabeled,
        max_deploy=args.max_deploy,
    )

    summary = {
        "class_names": VOC_CLASSES,
        "class_id_map": {name: idx for idx, name in enumerate(VOC_CLASSES)},
        "yolo_dataset_dir": str(yolo_root.resolve()),
        "labeled_dir": str(labeled_dir.resolve()),
        "unlabeled_dir": str((unlabeled_dir / "images").resolve()),
        "deploy_images_dir": str(deploy_dir.resolve()),
        "train": train_stats,
        "val": val_stats,
        **copy_stats,
    }
    summary_path = workdir / "datasets" / "voc2007_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
