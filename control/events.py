"""Pipeline event types.

Observation-only events (frame processed, target tracked, error). No event
type here represents or triggers actuation of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tracker.base import Track


@dataclass
class FrameResult:
    """Everything produced from processing a single frame.

    `frame` carries the raw (unannotated) BGR frame so consumers (the UI,
    --save-video in the CLI) can render it without re-opening or re-seeking
    the video source — cv2's frame-index seeking is unreliable on
    compressed formats, and re-opening per frame doesn't work for
    non-seekable sources like a webcam or the synthetic demo source. It is
    deliberately excluded from `as_dict()` — the JSONL telemetry log stores
    metadata, not raw pixels.
    """

    frame_index: int
    timestamp: float
    tracks: list[Track]
    detection_count: int
    inference_ms: float
    dropped: bool = False
    error: str | None = None
    decode_ms: float = 0.0
    detector_ms: float = 0.0
    tracker_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0
    frame: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "tracks": [t.as_dict() for t in self.tracks],
            "detection_count": self.detection_count,
            "inference_ms": self.inference_ms,
            "dropped": self.dropped,
            "error": self.error,
            "decode_ms": self.decode_ms,
            "detector_ms": self.detector_ms,
            "tracker_ms": self.tracker_ms,
            "total_ms": self.total_ms,
            "fps": self.fps,
        }
