"""Statistics data contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .errors import DataValidationError


@dataclass(slots=True)
class StatsEvent:
    schema_version: int
    source_id: str
    ts_utc: str
    total_detections: int
    counts_by_class: dict[str, int]
    latency_ms: float

    def validate(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError(f"Unsupported stats schema_version={self.schema_version}")
        if not self.source_id:
            raise DataValidationError("source_id is required")
        if self.total_detections < 0:
            raise DataValidationError("total_detections must be >= 0")
        if self.latency_ms < 0:
            raise DataValidationError("latency_ms must be >= 0")
        for k, v in self.counts_by_class.items():
            if not k:
                raise DataValidationError("class key must be non-empty")
            if v < 0:
                raise DataValidationError("counts_by_class values must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "ts_utc": self.ts_utc,
            "total_detections": self.total_detections,
            "counts_by_class": self.counts_by_class,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def now(
        cls,
        source_id: str,
        total_detections: int,
        counts_by_class: dict[str, int],
        latency_ms: float,
    ) -> "StatsEvent":
        return cls(
            schema_version=1,
            source_id=source_id,
            ts_utc=datetime.now(tz=timezone.utc).isoformat(),
            total_detections=total_detections,
            counts_by_class=counts_by_class,
            latency_ms=latency_ms,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StatsEvent":
        try:
            event = cls(
                schema_version=int(raw["schema_version"]),
                source_id=str(raw["source_id"]),
                ts_utc=str(raw["ts_utc"]),
                total_detections=int(raw["total_detections"]),
                counts_by_class={str(k): int(v) for k, v in dict(raw["counts_by_class"]).items()},
                latency_ms=float(raw["latency_ms"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"Invalid stats payload: {exc}") from exc
        event.validate()
        return event
