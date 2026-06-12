"""Control plane configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common.config.role_schema import (
    WORKSPACE_DEFAULT,
    expect_type,
    load_role_config,
    validate_server,
    validate_workspace,
)
from common.types.errors import ConfigError


DEFAULT_CONFIG: dict[str, Any] = {
    "workspace": deepcopy(WORKSPACE_DEFAULT),
    "server": {
        "host": "127.0.0.1",
        "port": 7800,
        "api_token": "",
        "api_token_env_name": "VISION_CONTROL_PLANE_API_TOKEN",
    },
    "storage": {
        "db_path": "../../work-dir/state/control_plane.db",
        "artifact_root": "../../work-dir/artifacts",
        "model_registry": "../../work-dir/models/registry",
    },
    "nodes": {
        "offline_ttl_sec": 45,
    },
}

PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("workspace", "root"),
    ("storage", "db_path"),
    ("storage", "artifact_root"),
    ("storage", "model_registry"),
)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_workspace(cfg)
    validate_server(cfg)
    storage = expect_type(cfg, "storage", dict, "root")
    for key in ("db_path", "artifact_root", "model_registry"):
        value = expect_type(storage, key, str, "storage")
        if not value:
            raise ConfigError(f"storage.{key} must not be empty")
    nodes = expect_type(cfg, "nodes", dict, "root")
    ttl = expect_type(nodes, "offline_ttl_sec", int, "nodes")
    if ttl < 0:
        raise ConfigError("nodes.offline_ttl_sec must be >= 0")
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
