"""Edge agent configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common.config.role_schema import (
    CLASS_MAP_DEFAULT,
    EDGE_RUNTIME_DEFAULT,
    LOCATE_ANYTHING_DEFAULT,
    TRAIN_RUNTIME_DEFAULT,
    WORKSPACE_DEFAULT,
    load_role_config,
    validate_class_map,
    validate_control_plane_ref,
    validate_edge_runtime,
    validate_job_store,
    validate_locate_anything,
    validate_node,
    validate_server,
    validate_workspace,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "workspace": deepcopy(WORKSPACE_DEFAULT),
    "node": {"id": "edge-001", "role": "edge"},
    "class_map": deepcopy(CLASS_MAP_DEFAULT),
    "train": {
        "device": TRAIN_RUNTIME_DEFAULT["device"],
        "img_size": TRAIN_RUNTIME_DEFAULT["img_size"],
        "faster_rcnn": deepcopy(TRAIN_RUNTIME_DEFAULT["faster_rcnn"]),
    },
    "runtime": deepcopy(EDGE_RUNTIME_DEFAULT),
    "locate_anything": deepcopy(LOCATE_ANYTHING_DEFAULT),
    "server": {
        "host": "127.0.0.1",
        "port": 7813,
        "api_token": "",
        "api_token_env_name": "VISION_EDGE_AGENT_API_TOKEN",
        "advertise_url": "",
    },
    "job_store": {"db_path": "../../work-dir/state/edge_agent_jobs.db"},
    "control_plane": {
        "url": "http://127.0.0.1:7800",
        "api_token": "",
        "api_token_env_name": "VISION_CONTROL_PLANE_API_TOKEN",
        "heartbeat_interval_sec": 15,
    },
}
DEFAULT_CONFIG["runtime"]["images_dir"] = "../../work-dir/datasets/smoke/images"
DEFAULT_CONFIG["runtime"]["local_model"] = "../../work-dir/models/exp001/model-int8.onnx"

PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("workspace", "root"),
    ("runtime", "video_path"),
    ("runtime", "images_dir"),
    ("runtime", "local_model"),
    ("job_store", "db_path"),
)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_workspace(cfg)
    validate_node(cfg, role="edge")
    validate_class_map(cfg)
    validate_edge_runtime(cfg["runtime"])
    validate_locate_anything(cfg["locate_anything"])
    validate_server(cfg)
    validate_job_store(cfg)
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
