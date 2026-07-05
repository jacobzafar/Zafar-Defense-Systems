"""Tracker interface.

Any tracking backend (IoU-based, ByteTrack, Norfair, etc.) must implement
BaseTracker so the pipeline never depends on a specific tracking library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from detector.base import Detection


@dataclass
class Track:
    """A tracked object with a stable ID across frames."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    age_frames: int
    hits: int
    time_since_update: int
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "age_frames": self.age_frames,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
        }


class BaseTracker(ABC):
    """Common interface for all tracking backends."""

    name: str = "base"

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]:
        """Consume one frame's detections, return current live tracks."""
        raise NotImplementedError

    def reset(self) -> None:
        """Optional hook to clear all track state (e.g. on new video load)."""
        return None
