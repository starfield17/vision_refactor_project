"""Train local tool configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common.config.role_schema import (
    CLASS_MAP_DEFAULT,
    EXPORT_DEFAULT,
    TRAIN_RUNTIME_DEFAULT,
    WORKSPACE_DEFAULT,
    load_role_config,
    validate_class_map,
    validate_train_runtime,
    validate_export,
    validate_workspace,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "workspace": deepcopy(WORKSPACE_DEFAULT),
    "class_map": deepcopy(CLASS_MAP_DEFAULT),
    "data": {
        "yolo_dataset_dir": "../../work-dir/datasets/yolo",
        "labeled_dir": "../../work-dir/datasets/labeled",
    },
    "runtime": deepcopy(TRAIN_RUNTIME_DEFAULT),
    "export": deepcopy(EXPORT_DEFAULT),
}

PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("workspace", "root"),
    ("data", "yolo_dataset_dir"),
    ("data", "labeled_dir"),
    ("runtime", "yolo", "weights"),
)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_workspace(cfg)
    validate_class_map(cfg)
    validate_train_runtime(cfg["runtime"], cfg["data"])
    validate_export(cfg["export"])
    return cfg


def load_config(
    config_path: Path,
    overrides: list[str] | None = None,
    workdir_override: str | None = None,
) -> dict[str, Any]:
    return load_role_config(
        config_path=config_path,
        default_config=DEFAULT_CONFIG,
        validate=validate_config,
        path_fields=PATH_FIELDS,
        overrides=overrides,
        workdir_override=workdir_override,
    )
