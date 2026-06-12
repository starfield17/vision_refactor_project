"""Detection data contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import DataValidationError


@dataclass(slots=True)
class Detection:
    schema_version: int
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: list[float]

    def validate(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError(f"Unsupported detection schema_version={self.schema_version}")
        if self.class_id < 0:
            raise DataValidationError("class_id must be >= 0")
        if not self.class_name:
            raise DataValidationError("class_name is required")
        if not (0.0 <= self.score <= 1.0):
            raise DataValidationError("score must be in [0, 1]")
        if len(self.bbox_xyxy) != 4:
            raise DataValidationError("bbox_xyxy must have 4 numeric values")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "bbox_xyxy": self.bbox_xyxy,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Detection":
        try:
            det = cls(
                schema_version=int(raw["schema_version"]),
                class_id=int(raw["class_id"]),
                class_name=str(raw["class_name"]),
                score=float(raw["score"]),
                bbox_xyxy=[float(x) for x in raw["bbox_xyxy"]],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"Invalid detection payload: {exc}") from exc
        det.validate()
        return det
