"""Label data contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .detection import Detection
from .errors import DataValidationError


@dataclass(slots=True)
class LabelRecord:
    schema_version: int
    image_path: str
    source: str
    detections: list[Detection] = field(default_factory=list)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError(f"Unsupported label schema_version={self.schema_version}")
        if not self.image_path:
            raise DataValidationError("image_path is required")
        if not self.source:
            raise DataValidationError("source is required")
        for detection in self.detections:
            detection.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "image_path": self.image_path,
            "source": self.source,
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LabelRecord":
        try:
            label = cls(
                schema_version=int(raw["schema_version"]),
                image_path=str(raw["image_path"]),
                source=str(raw["source"]),
                detections=[Detection.from_dict(d) for d in raw.get("detections", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"Invalid label payload: {exc}") from exc
        label.validate()
        return label
