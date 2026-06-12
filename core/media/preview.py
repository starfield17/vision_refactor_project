"""Preview helpers for side-by-side original/annotated images."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _resize_to_height(image: Any, target_height: int) -> Any:
    import cv2

    if image.shape[0] == target_height:
        return image
    width = max(1, int(round(image.shape[1] * target_height / image.shape[0])))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)


def _add_title_bar(image: Any, title: str) -> Any:
    import cv2

    bar_height = 36
    titled = cv2.copyMakeBorder(
        image,
        bar_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(24, 24, 24),
    )
    cv2.putText(
        titled,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (236, 236, 236),
        2,
        cv2.LINE_AA,
    )
    return titled


def compose_side_by_side_preview(
    original_bgr: Any,
    annotated_bgr: Any,
    left_title: str = "Original",
    right_title: str = "Labeled",
) -> Any:
    import cv2
    import numpy as np

    if original_bgr is None or annotated_bgr is None:
        raise ValueError("original_bgr and annotated_bgr must not be None")

    original = original_bgr.copy()
    annotated = annotated_bgr.copy()
    target_h = max(original.shape[0], annotated.shape[0])
    original = _resize_to_height(original, target_h)
    annotated = _resize_to_height(annotated, target_h)

    original = _add_title_bar(original, left_title)
    annotated = _add_title_bar(annotated, right_title)

    separator = np.full((original.shape[0], 4, 3), 36, dtype=original.dtype)
    return cv2.hconcat([original, separator, annotated])


def save_side_by_side_preview(
    original_bgr: Any,
    annotated_bgr: Any,
    save_path: Path,
    left_title: str = "Original",
    right_title: str = "Labeled",
) -> bool:
    import cv2

    preview = compose_side_by_side_preview(
        original_bgr=original_bgr,
        annotated_bgr=annotated_bgr,
        left_title=left_title,
        right_title=right_title,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(save_path), preview))
