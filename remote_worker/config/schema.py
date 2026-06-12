"""Remote inference worker configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common.config.role_schema import (
    CLASS_MAP_DEFAULT,
    EDGE_RUNTIME_DEFAULT,
    REMOTE_RUNTIME_DEFAULT,
    TRAIN_RUNTIME_DEFAULT,
    WORKSPACE_DEFAULT,
    load_role_config,
    validate_class_map,
    validate_control_plane_ref,
    validate_node,
    validate_remote_runtime,
    validate_workspace,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "workspace": deepcopy(WORKSPACE_DEFAULT),
    "node": {"id": "remote-001", "role": "remote"},
    "class_map": deepcopy(CLASS_MAP_DEFAULT),
    "train": {
        "device": TRAIN_RUNTIME_DEFAULT["device"],
        "img_size": TRAIN_RUNTIME_DEFAULT["img_size"],
        "faster_rcnn": deepcopy(TRAIN_RUNTIME_DEFAULT["faster_rcnn"]),
    },
    "runtime": deepcopy(REMOTE_RUNTIME_DEFAULT),
    "edge": {"jpeg_quality": EDGE_RUNTIME_DEFAULT["jpeg_quality"]},
    "control_plane": {
        "url": "http://127.0.0.1:7800",
        "api_token": "",
        "api_token_env_name": "VISION_CONTROL_PLANE_API_TOKEN",
        "heartbeat_interval_sec": 15,
    },
}
DEFAULT_CONFIG["runtime"]["model"] = "../../work-dir/models/exp001/model.onnx"

PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("workspace", "root"),
    ("runtime", "model"),
)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_workspace(cfg)
    validate_node(cfg, role="remote")
    validate_class_map(cfg)
    validate_remote_runtime(cfg["runtime"])
    validate_control_plane_ref(cfg)
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
