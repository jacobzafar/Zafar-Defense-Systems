"""Real (trained) detector adapter using a pretrained torchvision model.

Uses SSDLite320 with a MobileNetV3-Large backbone, pretrained on COCO and
distributed by the torchvision project itself under its BSD-3-Clause
license — no separate/ambiguous weight-file license to review, unlike the
AGPL-3.0 Ultralytics YOLO weights in ultralytics_detector.py (see
docs/DECISIONS.md entry #6 for the full reasoning).

COCO has no "drone" class, so this filters to a small set of COCO classes
that are reasonable visual proxies for a small aerial target in frame:
airplane, bird, kite. It is still not a trained drone classifier, but it
is a real object detector driven by learned features rather than
motion-blob heuristics, which is a meaningful step up for demo
credibility over the default MotionDetector.

Import is guarded so the rest of the pipeline works without torch/
torchvision installed — they are an optional extra, not a core
dependency (see requirements.txt / pyproject.toml `torchvision` extra).
"""

from __future__ import annotations

import cv2

from detector.base import BaseDetector, Detection

try:
    import torch
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

DEFAULT_TARGET_CLASSES = ["airplane", "bird", "kite"]


class TorchvisionDetector(BaseDetector):
    """Pretrained SSDLite MobileNetV3 (COCO), filtered to aerial-proxy classes."""

    name = "torchvision_ssdlite"

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        target_classes: list[str] | None = None,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                "The 'torch' and 'torchvision' packages are not installed. "
                "Install them with `pip install torch torchvision` (see "
                "docs/DECISIONS.md entry #6) to use TorchvisionDetector."
            )
        self.confidence_threshold = confidence_threshold
        self.target_classes = (
            target_classes if target_classes is not None else DEFAULT_TARGET_CLASSES
        )

        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self._categories = weights.meta["categories"]
        self._preprocess = weights.transforms()
        self._model = ssdlite320_mobilenet_v3_large(weights=weights)
        self._model.eval()

    def warmup(self) -> None:
        dummy = torch.zeros(3, 320, 320, dtype=torch.uint8)
        with torch.no_grad():
            self._model([self._preprocess(dummy)])

    def detect(self, frame) -> list[Detection]:
        if frame is None:
            return []

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)

        with torch.no_grad():
            output = self._model([self._preprocess(tensor)])[0]

        detections: list[Detection] = []
        for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
            confidence = float(score)
            if confidence < self.confidence_threshold:
                continue

            class_name = self._categories[int(label)]
            if self.target_classes and class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(
                Detection(
                    x1=x1 / width,
                    y1=y1 / height,
                    x2=x2 / width,
                    y2=y2 / height,
                    confidence=round(confidence, 3),
                    class_name=class_name,
                )
            )

        return detections
