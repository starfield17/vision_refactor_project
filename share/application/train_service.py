"""Application service for training workflows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from share.application.common import (
    append_override,
    build_train_registry,
    create_logger,
    normalize_run_result,
    read_run_summary,
)
from share.config.config_loader import load_config, save_resolved_config
from share.config.editing import persist_config_overrides
from share.kernel.kernel import VisionKernel


TRAIN_EDITABLE_PREFIXES = ("workspace", "data.yolo_dataset_dir", "train", "export")


def build_train_overrides_from_payload(payload: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    append_override(overrides, "workspace.run_name", payload.get("run_name"))
    append_override(overrides, "data.yolo_dataset_dir", payload.get("dataset_dir"))

    append_override(overrides, "train.backend", payload.get("backend"))
    append_override(overrides, "train.device", payload.get("device"))
    append_override(overrides, "train.seed", payload.get("seed"))
    append_override(overrides, "train.epochs", payload.get("epochs"))
    append_override(overrides, "train.batch_size", payload.get("batch_size"))
    append_override(overrides, "train.img_size", payload.get("img_size"))
    append_override(overrides, "train.dry_run", payload.get("dry_run"))

    append_override(overrides, "train.yolo.weights", payload.get("yolo_weights"))
    append_override(overrides, "train.faster_rcnn.variant", payload.get("frcnn_variant"))
    append_override(overrides, "train.faster_rcnn.lr", payload.get("frcnn_lr"))
    append_override(overrides, "train.faster_rcnn.momentum", payload.get("frcnn_momentum"))
    append_override(overrides, "train.faster_rcnn.weight_decay", payload.get("frcnn_weight_decay"))
    append_override(overrides, "train.faster_rcnn.num_workers", payload.get("frcnn_num_workers"))
    append_override(overrides, "train.faster_rcnn.max_samples", payload.get("frcnn_max_samples"))

    append_override(overrides, "export.onnx", payload.get("export_onnx"))
    append_override(overrides, "export.opset", payload.get("export_opset"))
    append_override(overrides, "export.quantize", payload.get("export_quantize"))
    append_override(overrides, "export.quantize_mode", payload.get("export_quantize_mode"))
    append_override(overrides, "export.calib_samples", payload.get("export_calib_samples"))
    return overrides


def save_train_config(config_path: Path | str, overrides: list[str]) -> dict[str, Any]:
    return persist_config_overrides(
        config_path=Path(config_path),
        overrides=overrides,
        allowed_prefixes=TRAIN_EDITABLE_PREFIXES,
    )


def run_train(
    config_path: Path | str,
    workdir_override: str | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    cfg = load_config(
        config_path=config_file,
        overrides=overrides or [],
        workdir_override=workdir_override,
    )

    logger = create_logger(cfg)
    logger.info("train.service.start", "Train run started", config_path=str(config_file))

    kernel = VisionKernel(cfg=cfg, logger=logger, registry=build_train_registry())
    result = kernel.run_train()

    resolved_path = result.run_context.run_dir / "config.resolved.toml"
    save_resolved_config(cfg, resolved_path)
    shutil.copyfile(resolved_path, Path(cfg["workspace"]["root"]) / "config.resolved.toml")

    logger.info(
        "train.service.done",
        "Train run finished",
        run_id=result.run_context.run_id,
        status=result.status,
        resolved_config=str(resolved_path),
    )
    return normalize_run_result(cfg, result)


def read_train_run_summary(run_dir: Path | str) -> dict[str, Any]:
    return read_run_summary(run_dir)

