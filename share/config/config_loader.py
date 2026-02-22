"""Load, validate and snapshot configuration."""

from __future__ import annotations

import ast
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from share.config.schema import DEFAULT_CONFIG, deep_merge_dict, validate_config
from share.types.errors import ConfigError


PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("workspace", "root"),
    ("data", "yolo_dataset_dir"),
    ("data", "labeled_dir"),
    ("data", "unlabeled_dir"),
    ("train", "yolo", "weights"),
    ("autolabel", "model", "onnx_model"),
    ("deploy", "edge", "video_path"),
    ("deploy", "edge", "images_dir"),
    ("deploy", "edge", "local_model"),
    ("deploy", "remote", "model"),
    ("deploy", "statistics", "db_path"),
)


def parse_override_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}', expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Invalid override '{item}', empty key")

        target = updated
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = parse_override_value(value.strip())
    return updated


def _get_nested(cfg: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_nested(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur: Any = cfg
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def resolve_path_fields(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = deepcopy(cfg)
    for field_path in PATH_FIELDS:
        current = _get_nested(resolved, field_path)
        if isinstance(current, str):
            _set_nested(resolved, field_path, _resolve_path(current, base_dir))
    return resolved


def _resolve_config_base_dir(config_path: Path) -> Path:
    # For the default layout (work-dir/config.toml), keep historical path semantics:
    # relative paths in config.toml are interpreted from repo root.
    config_dir = config_path.parent.resolve()
    if config_dir.name == "work-dir":
        return config_dir.parent
    return config_dir


def load_config(
    config_path: Path,
    overrides: list[str] | None = None,
    workdir_override: str | None = None,
) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw_cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    merged = deep_merge_dict(DEFAULT_CONFIG, raw_cfg)
    if overrides:
        merged = apply_overrides(merged, overrides)

    if workdir_override:
        merged["workspace"]["root"] = workdir_override

    merged = resolve_path_fields(merged, _resolve_config_base_dir(config_path))
    return validate_config(merged)


def _format_toml_key(key: str) -> str:
    if key.replace("_", "").isalnum() and " " not in key and "-" not in key:
        return key
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        entries = ", ".join(
            f"{_format_toml_key(k)} = {_format_toml_value(v)}" for k, v in value.items()
        )
        return "{" + entries + "}"
    raise ConfigError(f"Unsupported value for TOML serialization: {type(value).__name__}")


def to_toml(cfg: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit_section(prefix: str, section: dict[str, Any]) -> None:
        scalars: list[tuple[str, Any]] = []
        nested: list[tuple[str, dict[str, Any]]] = []

        for key, value in section.items():
            if isinstance(value, dict):
                nested.append((key, value))
            else:
                scalars.append((key, value))

        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars:
            lines.append(f"{_format_toml_key(key)} = {_format_toml_value(value)}")
        if prefix or scalars:
            lines.append("")

        for key, value in nested:
            child = key if not prefix else f"{prefix}.{key}"
            emit_section(child, value)

    emit_section("", cfg)
    return "\n".join(lines).strip() + "\n"


def save_resolved_config(cfg: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_toml(cfg), encoding="utf-8")
