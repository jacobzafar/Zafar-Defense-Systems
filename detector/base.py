"""Detector interface.

Any detector backend (motion-based, YOLO, RF-DETR, etc.) must implement
BaseDetector so the rest of the pipeline never depends on a specific
detection library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    """A single detected object in a single frame.

    Coordinates are normalized to [0, 1] relative to frame width/height so
    downstream consumers (tracker, UI) never need to know source resolution.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_name": self.class_name,
        }


class BaseDetector(ABC):
    """Common interface for all detector backends."""

    name: str = "base"

    @abstractmethod
    def detect(self, frame) -> list[Detection]:
        """Run detection on a single BGR frame (as returned by cv2.VideoCapture).

        Must return normalized Detection objects (coords in [0, 1]).
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional hook for backends that need a first slow inference call."""
        return None

    def close(self) -> None:
        """Optional hook to release model/resources."""
        return None
