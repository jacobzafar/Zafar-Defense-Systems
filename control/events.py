"""Pipeline event types.

Observation-only events (frame processed, target tracked, error). No event
type here represents or triggers actuation of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tracker.base import Track


@dataclass
class FrameResult:
    """Everything produced from processing a single frame."""

    frame_index: int
    timestamp: float
    tracks: list[Track]
    detection_count: int
    inference_ms: float
    dropped: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "tracks": [t.as_dict() for t in self.tracks],
            "detection_count": self.detection_count,
            "inference_ms": self.inference_ms,
            "dropped": self.dropped,
            "error": self.error,
        }
