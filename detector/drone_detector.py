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
import numpy as np

from detector.base import BaseDetector, Detection
from detector.input_norm import to_bgr8

try:
    import torch
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class DroneWeightsNotFoundError(FileNotFoundError):
    """Raised when no trained drone-detector weights file is present."""


def _check_training_report(weights_file: Path, input_size: int, architecture: str, anchor_scales) -> None:
    """Refuse weights whose own training_report.json (written next to them
    by detector/train.py) names a different layout. A wrong architecture
    fails load_state_dict anyway, but wrong anchor scales or input size
    load silently and decode every box against the wrong anchors."""
    import json

    report_path = weights_file.parent / "training_report.json"
    if not report_path.exists():
        return
    trained = json.loads(report_path.read_text(encoding="utf-8")).get("config", {})
    expected = {"input_size": input_size, "architecture": architecture, "anchor_scales": anchor_scales}
    defaults = {"input_size": 320, "architecture": "ssdlite", "anchor_scales": None}
    mismatched = {
        key: (trained.get(key, defaults[key]), value)
        for key, value in expected.items()
        if trained.get(key, defaults[key]) != value
    }
    if mismatched:
        raise ValueError(
            f"Detector config does not match how {weights_file} was trained "
            f"(trained vs. configured: {mismatched}, from {report_path})."
        )


PREPROCESS_MODES = ("squeeze", "letterbox")
RESIZE_INTERPOLATIONS = ("bilinear", "area")
# ImageNet mean in RGB, the value the model's normalization maps to ~0.
_LETTERBOX_FILL = (124, 116, 104)


class DroneDetector(BaseDetector):
    """Single-class ("drone") SSDLite detector loaded from local weights."""

    name = "drone_v1"

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float = 0.35,
        nms_thresh: float = 0.45,
        tile_rows: int = 1,
        tile_cols: int = 1,
        tile_overlap: float = 0.2,
        tile_include_full_frame: bool = True,
        input_size: int = 320,
        architecture: str = "ssdlite",
        anchor_scales: list[float] | None = None,
        preprocess: str = "squeeze",
        resize_interp: str = "bilinear",
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

        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f"preprocess must be one of {PREPROCESS_MODES}, got {preprocess!r}")
        if resize_interp not in RESIZE_INTERPOLATIONS:
            raise ValueError(f"resize_interp must be one of {RESIZE_INTERPOLATIONS}, got {resize_interp!r}")
        self.confidence_threshold = confidence_threshold
        # How a frame (or tile) becomes the square model input — DECISIONS #27.
        # "squeeze" + "bilinear" is the original path: the full-resolution
        # tensor goes to the model, whose transform resizes each axis to
        # input_size with bilinear F.interpolate (no antialiasing).
        # "area" resizes on the CPU first with cv2.INTER_AREA (an
        # antialiasing box filter when downscaling). "letterbox" keeps the
        # aspect ratio and pads to a square with the ImageNet mean. A
        # non-square ("adaptive") input is not offered: torchvision's
        # DefaultBoxGenerator stretches anchors with the input's aspect.
        self.preprocess = preprocess
        self.resize_interp = resize_interp

        # Local import to avoid a hard dependency on detector.train (and its
        # own guarded torch import chain) for callers that never select
        # this backend.
        from detector.train import build_single_class_model

        # Must match the input_size the weights were trained at
        # (detector/train.py TrainConfig.input_size; default 320).
        self.input_size = input_size
        self.architecture = architecture
        _check_training_report(weights_file, input_size, architecture, anchor_scales)
        self._model = build_single_class_model(input_size, architecture=architecture, anchor_scales=anchor_scales)
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

        # Tiled inference (detector/tiling.py): 1x1 = the plain full-frame
        # path. Otherwise every overlapping tile (plus, optionally, the
        # whole frame — so large drones cut by tile seams are still seen
        # whole) runs through the model in one batch; boxes are mapped back
        # to frame coordinates and merged across crops with the same NMS
        # threshold, capped at the model's own detections_per_img.
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.tile_overlap = tile_overlap
        self.tile_include_full_frame = tile_include_full_frame

    def warmup(self) -> None:
        dummy = torch.zeros(3, self.input_size, self.input_size, dtype=torch.uint8)
        with torch.no_grad():
            self._model([self._preprocess(dummy)])

    def _prepare(self, rgb):
        """Model input for one RGB crop, and the (sx, sy, ox, oy) that maps
        model-input coordinates back to the crop: crop = (input - o) / s."""
        if self.preprocess == "squeeze" and self.resize_interp == "bilinear":
            return torch.from_numpy(rgb).permute(2, 0, 1), (1.0, 1.0, 0.0, 0.0)
        h, w = rgb.shape[:2]
        size = self.input_size
        if self.preprocess == "squeeze":
            new_w, new_h = size, size
        else:
            scale = size / max(w, h)
            new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        downscale = new_w * new_h < w * h
        interp = cv2.INTER_AREA if self.resize_interp == "area" and downscale else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
        ox = oy = 0
        if (new_w, new_h) != (size, size):
            canvas = np.empty((size, size, 3), np.uint8)
            canvas[:] = _LETTERBOX_FILL
            ox, oy = (size - new_w) // 2, (size - new_h) // 2
            canvas[oy : oy + new_h, ox : ox + new_w] = resized
            resized = canvas
        return torch.from_numpy(resized).permute(2, 0, 1), (new_w / w, new_h / h, float(ox), float(oy))

    @staticmethod
    def _to_crop(out: dict, mapping, crop_w: int, crop_h: int) -> torch.Tensor:
        sx, sy, ox, oy = mapping
        if mapping == (1.0, 1.0, 0.0, 0.0):
            return out["boxes"]
        boxes = (out["boxes"] - torch.tensor([ox, oy, ox, oy])) / torch.tensor([sx, sy, sx, sy])
        return torch.stack(
            [boxes[:, 0].clamp(0, crop_w), boxes[:, 1].clamp(0, crop_h), boxes[:, 2].clamp(0, crop_w), boxes[:, 3].clamp(0, crop_h)],
            dim=1,
        )

    def _run_model(self, rgb, width: int, height: int) -> dict:
        if self.tile_rows * self.tile_cols == 1:
            tensor, mapping = self._prepare(rgb)
            with torch.no_grad():
                out = self._model([self._preprocess(tensor)])[0]
            return {**out, "boxes": self._to_crop(out, mapping, width, height)}

        from torchvision.ops import nms

        from detector.tiling import tile_grid

        crops = tile_grid(width, height, self.tile_rows, self.tile_cols, self.tile_overlap)
        if self.tile_include_full_frame:
            crops = [(0, 0, width, height)] + crops
        prepared = [self._prepare(rgb[y1:y2, x1:x2]) for (x1, y1, x2, y2) in crops]
        with torch.no_grad():
            outputs = self._model([self._preprocess(tensor) for tensor, _ in prepared])

        boxes, scores, labels = [], [], []
        for (x1, y1, x2, y2), (_, mapping), out in zip(crops, prepared, outputs):
            crop_boxes = self._to_crop(out, mapping, x2 - x1, y2 - y1)
            boxes.append(crop_boxes + torch.tensor([x1, y1, x1, y1], dtype=crop_boxes.dtype))
            scores.append(out["scores"])
            labels.append(out["labels"])
        boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
        keep = nms(boxes, scores, self._model.nms_thresh)[: self._model.detections_per_img]
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

    def detect(self, frame) -> list[Detection]:
        if frame is None:
            return []

        frame = to_bgr8(frame)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        output = self._run_model(rgb, width, height)

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
