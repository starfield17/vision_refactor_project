"""Helpers for editing user config files from CLI/Web frontends."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Sequence

from share.config.config_loader import apply_overrides, resolve_path_fields, to_toml
from share.config.schema import DEFAULT_CONFIG, deep_merge_dict, validate_config
from share.types.errors import ConfigError


def _resolve_config_base_dir(config_path: Path) -> Path:
    """Mirror load_config() semantics for relative path resolution."""
    config_dir = config_path.parent.resolve()
    if config_dir.name == "work-dir":
        return config_dir.parent
    return config_dir


def _is_allowed_key(key: str, allowed_prefixes: Sequence[str]) -> bool:
    for prefix in allowed_prefixes:
        if key == prefix or key.startswith(f"{prefix}."):
            return True
    return False


def _validate_allowed_overrides(overrides: list[str], allowed_prefixes: Sequence[str]) -> None:
    disallowed: list[str] = []
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}', expected key=value")
        key, _value = item.split("=", 1)
        key = key.strip()
        if not _is_allowed_key(key, allowed_prefixes):
            disallowed.append(key)
    if disallowed:
        listed = ", ".join(sorted(set(disallowed)))
        allowed = ", ".join(sorted(allowed_prefixes))
        raise ConfigError(
            f"Override keys outside editable scope: {listed}. Allowed prefixes: {allowed}"
        )


def load_raw_config(config_path: Path) -> dict[str, Any]:
    """Read user config TOML as-is."""
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw_cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc
    if not isinstance(raw_cfg, dict):
        raise ConfigError(f"Invalid TOML root in {config_path}: expected table")
    return raw_cfg


def load_merged_user_config(config_path: Path) -> dict[str, Any]:
    """Read user config and merge defaults for easier UI rendering."""
    raw_cfg = load_raw_config(config_path)
    return deep_merge_dict(DEFAULT_CONFIG, raw_cfg)


def persist_config_overrides(
    config_path: Path,
    overrides: list[str],
    allowed_prefixes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Apply overrides directly to config.toml and validate the resulting file.

    Returns the validated, path-resolved config dict.
    """
    raw_cfg = load_raw_config(config_path)
    if allowed_prefixes:
        _validate_allowed_overrides(overrides, allowed_prefixes)
    updated_raw = apply_overrides(raw_cfg, overrides)
    merged = deep_merge_dict(DEFAULT_CONFIG, updated_raw)
    resolved = resolve_path_fields(merged, _resolve_config_base_dir(config_path))
    validated = validate_config(resolved)
    config_path.write_text(to_toml(updated_raw), encoding="utf-8")
    return validated
