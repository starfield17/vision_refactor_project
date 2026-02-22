"""Kernel plugin registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from share.types.errors import ConfigError

TrainerFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
AutolabelFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
DeployerFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class KernelRegistry:
    def __init__(self) -> None:
        self._trainers: dict[str, TrainerFn] = {}
        self._autolabelers: dict[str, AutolabelFn] = {}
        self._deployers: dict[str, DeployerFn] = {}

    def register_trainer(self, name: str, fn: TrainerFn) -> None:
        self._trainers[name] = fn

    def get_trainer(self, name: str) -> TrainerFn:
        if name not in self._trainers:
            raise ConfigError(f"No trainer registered for backend '{name}'")
        return self._trainers[name]

    def register_autolabeler(self, name: str, fn: AutolabelFn) -> None:
        self._autolabelers[name] = fn

    def get_autolabeler(self, name: str) -> AutolabelFn:
        if name not in self._autolabelers:
            raise ConfigError(f"No autolabel backend registered for mode '{name}'")
        return self._autolabelers[name]

    def register_deployer(self, name: str, fn: DeployerFn) -> None:
        self._deployers[name] = fn

    def get_deployer(self, name: str) -> DeployerFn:
        if name not in self._deployers:
            raise ConfigError(f"No deploy backend registered for mode '{name}'")
        return self._deployers[name]
