"""Factory for selecting a detector backend from config.

Keeps control/pipeline.py decoupled from any specific detector import.
"""

from __future__ import annotations

from typing import Any

from detector.base import BaseDetector
from detector.motion_detector import MotionDetector


def build_detector(config: dict[str, Any] | None = None) -> BaseDetector:
    config = config or {}
    backend = config.get("backend", "motion")

    if backend == "motion":
        return MotionDetector(
            min_area_px=config.get("min_area_px", 150),
            max_area_fraction=config.get("max_area_fraction", 0.25),
            var_threshold=config.get("var_threshold", 32.0),
            history=config.get("history", 300),
        )

    if backend == "ultralytics":
        from detector.ultralytics_detector import UltralyticsDetector

        return UltralyticsDetector(
            weights_path=config["weights_path"],
            confidence_threshold=config.get("confidence_threshold", 0.35),
            target_classes=config.get("target_classes"),
        )

    if backend == "torchvision":
        from detector.torchvision_detector import TorchvisionDetector

        return TorchvisionDetector(
            confidence_threshold=config.get("confidence_threshold", 0.35),
            target_classes=config.get("target_classes"),
        )

    raise ValueError(
        f"Unknown detector backend '{backend}'. Valid options: motion, ultralytics, torchvision."
    )
