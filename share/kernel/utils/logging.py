"""Structured JSONL logging utilities."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    _LEVEL_PRIORITY = {
        "DEBUG": 10,
        "INFO": 20,
        "WARN": 30,
        "ERROR": 40,
    }

    def __init__(self, log_path: Path, level: str = "INFO") -> None:
        self.log_path = log_path
        normalized = str(level).upper().strip()
        if normalized not in self._LEVEL_PRIORITY:
            normalized = "INFO"
        self.level = normalized
        self._min_level = self._LEVEL_PRIORITY[normalized]
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        level: str,
        event: str,
        message: str,
        **fields: Any,
    ) -> None:
        normalized = str(level).upper().strip()
        level_priority = self._LEVEL_PRIORITY.get(normalized, self._LEVEL_PRIORITY["INFO"])
        if level_priority < self._min_level:
            return
        payload = {
            "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
            "level": normalized,
            "event": event,
            "message": message,
            **fields,
        }
        line = json.dumps(payload, ensure_ascii=True)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")

    def debug(self, event: str, message: str, **fields: Any) -> None:
        self.log("DEBUG", event, message, **fields)

    def info(self, event: str, message: str, **fields: Any) -> None:
        self.log("INFO", event, message, **fields)

    def warn(self, event: str, message: str, **fields: Any) -> None:
        self.log("WARN", event, message, **fields)

    def error(self, event: str, message: str, **fields: Any) -> None:
        self.log("ERROR", event, message, **fields)
