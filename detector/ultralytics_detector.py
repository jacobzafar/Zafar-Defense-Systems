"""Optional real-detector adapter (Ultralytics YOLO).

Import is guarded so the rest of the pipeline works even if `ultralytics`
is not installed. Included as a fast path for teams that accept the AGPL-3.0
license (or hold a commercial Ultralytics license); see docs/DECISIONS.md
entry #2 for the license tradeoff discussion before enabling this in a
shipped product.

To switch to a permissively-licensed detector instead (RF-DETR / YOLOX,
Apache-2.0), implement the same BaseDetector interface in a sibling file,
e.g. detector/rfdetr_detector.py, and swap it in via detector/factory.py.
"""

from __future__ import annotations

from typing import Any

from detector.base import BaseDetector, Detection

try:
    from ultralytics import YOLO  # type: ignore

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


class UltralyticsDetector(BaseDetector):
    name = "ultralytics_yolo"

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float = 0.35,
        target_classes: list[str] | None = None,
    ) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "The 'ultralytics' package is not installed. "
                "Install it explicitly with `pip install ultralytics` if "
                "your team has accepted the AGPL-3.0 / commercial license "
                "terms (see docs/DECISIONS.md entry #2)."
            )
        self._model: Any = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes

    def warmup(self) -> None:
        import numpy as np

        dummy = np.zeros((640, 640, 3), dtype="uint8")
        self._model.predict(dummy, verbose=False)

    def detect(self, frame) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self._model.predict(
            frame, conf=self.confidence_threshold, verbose=False
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = result.names.get(cls_id, str(cls_id))
                if self.target_classes and class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append(
                    Detection(
                        x1=x1 / width,
                        y1=y1 / height,
                        x2=x2 / width,
                        y2=y2 / height,
                        confidence=float(box.conf[0]),
                        class_name=class_name,
                    )
                )
        return detections
