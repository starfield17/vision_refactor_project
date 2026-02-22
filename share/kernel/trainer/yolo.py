"""YOLO trainer for Phase 2."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from share.kernel.data.yolo_dataset import scan_and_validate_yolo_dataset
from share.kernel.export.onnx_export import build_export_artifacts
from share.types.errors import DataValidationError


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _build_data_yaml_content(
    dataset_root: Path,
    train_images_dir: Path,
    val_images_dir: Path,
    class_names: list[str],
) -> str:
    train_rel = str(train_images_dir.relative_to(dataset_root))
    val_rel = str(val_images_dir.relative_to(dataset_root))
    names_json = json.dumps(class_names, ensure_ascii=True)

    return (
        f"path: {dataset_root}\n"
        f"train: {train_rel}\n"
        f"val: {val_rel}\n"
        f"nc: {len(class_names)}\n"
        f"names: {names_json}\n"
    )


def _pick_trained_weight(train_output_dir: Path, fallback_weight: Path) -> Path:
    best = train_output_dir / "weights" / "best.pt"
    last = train_output_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return fallback_weight


def run_yolo_train(cfg: dict[str, Any], run_ctx: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(cfg["data"]["yolo_dataset_dir"])
    weights_path = Path(cfg["train"]["yolo"]["weights"])
    if not weights_path.exists():
        raise DataValidationError(f"YOLO weights file not found: {weights_path}")

    class_names = list(cfg["class_map"]["names"])
    class_id_map = dict(cfg["class_map"]["id_map"])
    scan = scan_and_validate_yolo_dataset(
        dataset_root=dataset_root,
        class_map_names=class_names,
        class_map_id_map=class_id_map,
    )

    workdir = Path(cfg["workspace"]["root"])
    run_name = cfg["workspace"]["run_name"]
    model_dir = workdir / "models" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(str(run_ctx["run_dir"]))
    data_yaml_path = run_dir / "dataset.resolved.yaml"
    data_yaml_path.write_text(
        _build_data_yaml_content(
            dataset_root=scan.dataset_root,
            train_images_dir=scan.train_images_dir,
            val_images_dir=scan.val_images_dir,
            class_names=class_names,
        ),
        encoding="utf-8",
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

    train_output_dir = run_dir / "yolo_train"
    dry_run = bool(cfg["train"]["dry_run"])
    trained_weight = weights_path
    train_note = "dry-run: skipped actual training"

    if not dry_run:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        model.train(
            data=str(data_yaml_path),
            epochs=int(cfg["train"]["epochs"]),
            batch=int(cfg["train"]["batch_size"]),
            imgsz=int(cfg["train"]["img_size"]),
            device=str(cfg["train"]["device"]),
            seed=int(cfg["train"]["seed"]),
            project=str(run_dir),
            name="yolo_train",
            exist_ok=True,
            workers=0,
            verbose=False,
        )
        trained_weight = _pick_trained_weight(train_output_dir, weights_path)
        train_note = "train finished"

    model_pt = model_dir / "model.pt"
    shutil.copyfile(trained_weight, model_pt)

    export_info = build_export_artifacts(
        trained_pt=model_pt,
        model_dir=model_dir,
        cfg_export=cfg["export"],
        device=str(cfg["train"]["device"]),
        logger=run_ctx["logger"],
    )

    return {
        "backend": "yolo",
        "dry_run": dry_run,
        "train_note": train_note,
        "dataset": {
            "root": str(scan.dataset_root),
            "train_images_dir": str(scan.train_images_dir),
            "val_images_dir": str(scan.val_images_dir),
            "image_count": scan.image_files,
            "label_files": scan.label_files,
            "class_histogram": scan.class_histogram,
            "resolved_data_yaml": str(data_yaml_path),
        },
        "class_map_validation": {
            "ok": True,
            "class_count": len(class_names),
        },
        "weights": {
            "init": str(weights_path),
            "trained": str(trained_weight),
            "model_pt": str(model_pt),
            "labels_snapshot": str(labels_snapshot_path),
        },
        "export": {
            "onnx_path": export_info.onnx_path,
            "quantized_onnx_path": export_info.quantized_onnx_path,
            "fp16_onnx_path": export_info.fp16_onnx_path,
            "final_infer_model_path": export_info.final_infer_model_path,
            "export_ok": export_info.export_ok,
            "quantize_ok": export_info.quantize_ok,
            "quantize_strategy": export_info.quantize_strategy,
            "messages": export_info.messages,
        },
    }
