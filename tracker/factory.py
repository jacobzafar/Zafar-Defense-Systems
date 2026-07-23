"""Factory for selecting a tracker backend from config."""

from __future__ import annotations

from typing import Any

from tracker.base import BaseTracker
from tracker.iou_tracker import IoUTracker


def build_tracker(config: dict[str, Any] | None = None) -> BaseTracker:
    config = config or {}
    backend = config.get("backend", "iou")

    if backend == "iou":
        return IoUTracker(
            iou_threshold=config.get("iou_threshold", 0.3),
            max_age=config.get("max_age", 15),
            min_hits_to_confirm=config.get("min_hits_to_confirm", 1),
        )

    if backend == "bytetrack":
        from tracker.bytetrack_adapter import ByteTrackAdapter

        return ByteTrackAdapter()

    raise ValueError(
        f"Unknown tracker backend '{backend}'. Valid options: iou, bytetrack."
    )
