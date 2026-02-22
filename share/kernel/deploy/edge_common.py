"""Shared helpers for edge deploy modes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from share.kernel.media.frame_source import (
    FramePacket,
    iter_camera_frames,
    iter_image_frames,
    iter_video_frames,
)
from share.types.errors import ConfigError
from share.types.stats import StatsEvent


class FpsLimiter:
    def __init__(self, fps_limit: float) -> None:
        self.interval = 0.0 if fps_limit <= 0 else 1.0 / fps_limit
        self._next_ts = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.perf_counter()
        if self._next_ts == 0.0:
            self._next_ts = now + self.interval
            return
        if now < self._next_ts:
            time.sleep(self._next_ts - now)
        self._next_ts = max(self._next_ts + self.interval, time.perf_counter())


def iter_source_frames(edge_cfg: dict[str, Any]) -> Iterator[FramePacket]:
    source = str(edge_cfg["source"])
    if source == "images":
        return iter_image_frames(Path(edge_cfg["images_dir"]))
    if source == "video":
        return iter_video_frames(Path(edge_cfg["video_path"]))
    if source == "camera":
        return iter_camera_frames(int(edge_cfg["camera_id"]))
    raise ConfigError(f"unsupported deploy.edge.source={source}")


def append_stats_snapshot(path: Path, event: StatsEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")
