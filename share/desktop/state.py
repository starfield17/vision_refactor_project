"""Desktop page state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PageState:
    status: str = "idle"
    last_result: dict[str, Any] = field(default_factory=dict)

