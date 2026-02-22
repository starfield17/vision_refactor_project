"""Frame source iterators for deploy edge pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from share.types.errors import DataValidationError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class FramePacket:
    frame_index: int
    frame_name: str
    frame_bgr: Any
    source_path: str


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise DataValidationError(f"images_dir not found: {images_dir}")
    images = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise DataValidationError(f"no images found under: {images_dir}")
    return images


def iter_image_frames(images_dir: Path) -> Iterator[FramePacket]:
    try:
        import cv2
    except Exception as exc:
        raise DataValidationError(f"opencv is required for image source: {exc}") from exc

    for index, image_path in enumerate(list_images(images_dir)):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        yield FramePacket(
            frame_index=index,
            frame_name=image_path.name,
            frame_bgr=frame,
            source_path=str(image_path),
        )


def _iter_capture(capture: Any, frame_prefix: str, source_path: str) -> Iterator[FramePacket]:
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        yield FramePacket(
            frame_index=index,
            frame_name=f"{frame_prefix}_{index:06d}.jpg",
            frame_bgr=frame,
            source_path=source_path,
        )
        index += 1


def iter_video_frames(video_path: Path) -> Iterator[FramePacket]:
    if not video_path.exists():
        raise DataValidationError(f"video_path not found: {video_path}")

    try:
        import cv2
    except Exception as exc:
        raise DataValidationError(f"opencv is required for video source: {exc}") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise DataValidationError(f"failed to open video source: {video_path}")

    try:
        yield from _iter_capture(capture, "video_frame", str(video_path))
    finally:
        capture.release()


def iter_camera_frames(camera_id: int) -> Iterator[FramePacket]:
    try:
        import cv2
    except Exception as exc:
        raise DataValidationError(f"opencv is required for camera source: {exc}") from exc

    capture = cv2.VideoCapture(int(camera_id))
    if not capture.isOpened():
        capture.release()
        raise DataValidationError(f"failed to open camera source: camera_id={camera_id}")

    try:
        yield from _iter_capture(capture, "camera_frame", f"camera:{camera_id}")
    finally:
        capture.release()
