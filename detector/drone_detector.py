"""Drone-specific detector backend — loads weights produced by detector/train.py.

Same architecture as `detector/torchvision_detector.py` (SSDLite320
MobileNetV3), but with the classification head replaced for a single
"drone" foreground class instead of using the stock COCO head — see
`detector/train.py`'s `build_single_class_model()`, which this module's
weights must have come from.

This backend is opt-in and OFF by default (the default detector backend
remains `motion`; see detector/factory.py). It only exists to be selected
explicitly once a maintainer has actually trained drone-specific weights
via detector/train.py — if no weights file is present, it fails at
construction time with a clear, actionable message rather than silently
falling back to something else or pretending to detect anything.

No trained weights ship with this repo. Nothing here has been validated
against real drone footage — see docs/known-limitations.md.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from detector.base import BaseDetector, Detection

try:
    import torch
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class DroneWeightsNotFoundError(FileNotFoundError):
    """Raised when no trained drone-detector weights file is present."""


class DroneDetector(BaseDetector):
    """Single-class ("drone") SSDLite detector loaded from local weights."""

    name = "drone_v1"

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float = 0.35,
        nms_thresh: float = 0.45,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "The 'torch' and 'torchvision' packages are not installed. "
                "Install them with `pip install torch torchvision` to use "
                "DroneDetector."
            )

        weights_file = Path(weights_path) if weights_path else None
        if weights_file is None or not weights_file.exists():
            raise DroneWeightsNotFoundError(
                f"No trained drone-detector weights found at "
                f"'{weights_path or '(no path given)'}'. Train one with "
                f"detector/train.py (see docs/DECISIONS.md and "
                f"detector/datasets/ for the data/training workflow), or "
                f"select a different detector backend (motion / torchvision "
                f"/ ultralytics)."
            )

        self.confidence_threshold = confidence_threshold

        # Local import to avoid a hard dependency on detector.train (and its
        # own guarded torch import chain) for callers that never select
        # this backend.
        from detector.train import build_single_class_model

        self._model = build_single_class_model()
        state_dict = torch.load(weights_file, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.eval()

        # build_single_class_model() only replaces the classification head,
        # so torchvision's ssdlite320_mobilenet_v3_large() factory otherwise
        # leaves its own NMS config in place: score_thresh=0.001,
        # nms_thresh=0.55, topk_candidates=300, detections_per_img=300 (all
        # tuned for 80-class COCO detection). NMS itself runs unconditionally
        # inside the model's own eval-mode forward pass (see
        # torchvision.models.detection.ssd.SSD.postprocess_detections) — it
        # is not missing. But 0.55 is loose for a single dominant-class
        # problem with a dense overlapping anchor grid: measured directly
        # against this model's own real output, the surviving boxes'
        # pairwise IoU clusters at 0.546-0.550, i.e. mechanically just under
        # that cutoff — the "many near-identical boxes" symptom this
        # threshold produces. 0.45 is the plain SSD base class's own
        # (tighter) default in this exact torchvision version, not a value
        # picked to fit this eval set.
        self._model.nms_thresh = nms_thresh

        self._preprocess = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.transforms()

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
            if int(label) != 1:  # 0 = background, 1 = the single "drone" class
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(
                Detection(
                    x1=x1 / width,
                    y1=y1 / height,
                    x2=x2 / width,
                    y2=y2 / height,
                    confidence=confidence,  # unrounded: rounding creates ties that distort AP ranking
                    class_name="drone",
                )
            )

        return detections
