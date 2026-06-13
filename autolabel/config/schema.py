"""AutoLabel local tool configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common.config.role_schema import (
    AUTOLABEL_RUNTIME_DEFAULT,
    CLASS_MAP_DEFAULT,
    LOCATE_ANYTHING_DEFAULT,
    TRAIN_RUNTIME_DEFAULT,
    WORKSPACE_DEFAULT,
    load_role_config,
    validate_autolabel_runtime,
    validate_class_map,
    validate_locate_anything,
    validate_workspace,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "workspace": deepcopy(WORKSPACE_DEFAULT),
    "class_map": deepcopy(CLASS_MAP_DEFAULT),
    "data": {
        "labeled_dir": "../../work-dir/datasets/labeled",
        "unlabeled_dir": "../../work-dir/datasets/unlabeled",
    },
    "train": {
        "device": TRAIN_RUNTIME_DEFAULT["device"],
        "img_size": TRAIN_RUNTIME_DEFAULT["img_size"],
        "faster_rcnn": deepcopy(TRAIN_RUNTIME_DEFAULT["faster_rcnn"]),
    },
    "runtime": deepcopy(AUTOLABEL_RUNTIME_DEFAULT),
    "locate_anything": deepcopy(LOCATE_ANYTHING_DEFAULT),
}
DEFAULT_CONFIG["runtime"]["model"]["onnx_model"] = "../../work-dir/models/exp001/model-int8.onnx"

PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("workspace", "root"),
    ("data", "labeled_dir"),
    ("data", "unlabeled_dir"),
    ("runtime", "model", "onnx_model"),
)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_workspace(cfg)
    validate_class_map(cfg)
    validate_autolabel_runtime(cfg["runtime"])
    validate_locate_anything(cfg["locate_anything"])
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
