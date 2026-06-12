"""Statistics service configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common.config.role_schema import (
    CLASS_MAP_DEFAULT,
    STATISTICS_RUNTIME_DEFAULT,
    WORKSPACE_DEFAULT,
    load_role_config,
    validate_class_map,
    validate_control_plane_ref,
    validate_job_store,
    validate_node,
    validate_server,
    validate_statistics_runtime,
    validate_workspace,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "workspace": deepcopy(WORKSPACE_DEFAULT),
    "node": {"id": "statistics-001", "role": "statistics", "endpoint": ""},
    "class_map": deepcopy(CLASS_MAP_DEFAULT),
    "runtime": deepcopy(STATISTICS_RUNTIME_DEFAULT),
    "server": {
        "host": "127.0.0.1",
        "port": 7803,
        "api_token": "",
        "api_token_env_name": "VISION_STATISTICS_API_TOKEN",
        "advertise_url": "",
    },
    "job_store": {"db_path": "../../work-dir/state/statistics_jobs.db"},
    "control_plane": {
        "url": "http://127.0.0.1:7800",
        "api_token": "",
        "api_token_env_name": "VISION_CONTROL_PLANE_API_TOKEN",
        "heartbeat_interval_sec": 15,
    },
}
DEFAULT_CONFIG["runtime"]["db_path"] = "../../work-dir/stats/stats.db"

PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("workspace", "root"),
    ("runtime", "db_path"),
    ("job_store", "db_path"),
)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_workspace(cfg)
    validate_node(cfg, role="statistics")
    validate_class_map(cfg)
    validate_statistics_runtime(cfg["runtime"])
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
