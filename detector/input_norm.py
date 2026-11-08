"""Normalize whatever a camera or decoder hands over to BGR uint8.

`cv2.VideoCapture` (control/pipeline.py) already covers containers,
codecs and streams (files, RTSP, devices via FFmpeg/V4L2). What it does
not guarantee is the pixel format: mono and thermal sources can arrive
as 2-D or single-channel arrays, raw thermal as 16-bit, some sources as
BGRA, and preprocessed frames as float. Detectors here expect BGR uint8
(detector/base.py), so `to_bgr8` is the one place that conversion lives.

A frame that already is HxWx3 uint8 is returned as-is (same object), so
the existing RGB path is untouched.

16-bit and float frames are stretched per frame between the 0.5th and
99.5th percentile — the usual display mapping for radiometric thermal
data, which uses a small part of the 16-bit range. That is a display
choice, not calibration: absolute temperature is lost, and a detector
trained on visible-light frames has not seen such images (see
DECISIONS #27: the thermal proxy result).
"""

from __future__ import annotations

import cv2
import numpy as np


class UnsupportedFrameError(ValueError):
    """Raised for frames that cannot be interpreted as an image."""


def _stretch_to_uint8(frame: np.ndarray) -> np.ndarray:
    x = frame.astype(np.float32)
    lo, hi = np.percentile(x, (0.5, 99.5))
    if hi <= lo:
        return np.zeros(frame.shape, np.uint8)
    return np.clip((x - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def to_bgr8(frame: np.ndarray) -> np.ndarray:
    """Return `frame` as an HxWx3 BGR uint8 array (see module docstring)."""
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
        raise UnsupportedFrameError(f"Expected a 2-D or 3-D image array, got {getattr(frame, 'shape', type(frame))}")
    if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
        return frame
    if frame.dtype != np.uint8:
        frame = _stretch_to_uint8(frame)
    if frame.ndim == 2 or frame.shape[2] == 1:
        return cv2.cvtColor(frame.reshape(frame.shape[:2]), cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] == 3:
        return frame
    raise UnsupportedFrameError(f"Unsupported channel count {frame.shape[2]}")
