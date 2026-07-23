"""Zero-dependency placeholder detector using OpenCV background subtraction.

This is NOT a drone/aircraft classifier. It flags moving blobs in the frame
so the full pipeline (detect -> track -> UI -> log) can be exercised and
validated end-to-end without requiring a trained model or GPU.

Swap this out for a trained detector (see ultralytics_detector.py for the
adapter pattern) once real weights / a labeled dataset are available.
"""

from __future__ import annotations

import cv2
import numpy as np

from detector.base import BaseDetector, Detection


class MotionDetector(BaseDetector):
    name = "motion_v1"

    def __init__(
        self,
        min_area_px: int = 150,
        max_area_fraction: float = 0.25,
        var_threshold: float = 32.0,
        history: int = 300,
    ) -> None:
        self.min_area_px = min_area_px
        self.max_area_fraction = max_area_fraction
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )

    def detect(self, frame) -> list[Detection]:
        if frame is None:
            return []

        height, width = frame.shape[:2]
        frame_area = float(height * width)

        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_px:
                continue
            if area / frame_area > self.max_area_fraction:
                # Almost certainly a lighting shift / camera jolt, not a target.
                continue

            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(0.95, 0.4 + (area / (frame_area * self.max_area_fraction)))

            detections.append(
                Detection(
                    x1=x / width,
                    y1=y / height,
                    x2=(x + w) / width,
                    y2=(y + h) / height,
                    confidence=round(float(confidence), 3),
                    class_name="moving_object",
                )
            )

        return detections
