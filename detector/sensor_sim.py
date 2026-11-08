"""Simulated-sensor image transforms: one recipe for eval variants and training.

A "simulated sensor" is an ordered list of ops applied to a BGR uint8
frame, each a dict `{"op": <name>, ...params}`:

- `resize`: `scale` (e.g. 0.5 = a sensor with half the pixels per axis);
  `restore: true` scales back up to the original size (a low-res sensor
  upscaled in the stream), otherwise the frame stays small.
- `noise`: Gaussian read noise `read_sigma` (DN) plus shot noise whose
  variance is `shot` x intensity.
- `motion_blur`: linear kernel `length_frac` of the frame width long, at a
  per-frame random angle.
- `gaussian_blur`: `sigma_frac` of the frame width (optics / defocus).
- `jpeg`: re-encode at `quality` (codec artifacts).
- `color`: fixed `gamma` (out = in^gamma) and per-channel `wb` gains (BGR)
  — one camera's tone curve and white balance, not per-frame jitter.
- `gray`: luminance replicated to 3 channels (a mono sensor).
- `invert`: 255 - x (white-hot / black-hot polarity).

`thermal_proxy` in eval configs is gray + invert + low-res + blur. It is a
crude *appearance* proxy for a low-resolution mono sensor, NOT a thermal
simulation: in real LWIR a drone's contrast comes from heat (motors,
battery), not from inverted visible luminance.

Randomness (noise, blur angle, shake) comes from `frame_rng(seed, key)`,
seeded per frame from its file name, so every run of a variant sees
exactly the same pixels — eval variants are as frozen as the frames.

Geometric ops change the frame size only by uniform scaling; detectors
return normalized boxes, so ground truth in original pixels stays valid.
`shake` (translation) is the one op that moves content: it is applied
separately (`apply_shake`) and returns the offset so the caller shifts
ground truth by exactly the same amount.
"""

from __future__ import annotations

import zlib
from typing import Any

import cv2
import numpy as np

SENSOR_OPS = ("resize", "noise", "motion_blur", "gaussian_blur", "jpeg", "color", "gray", "invert")


class SensorSimError(ValueError):
    """Raised for an unknown op or bad op parameters."""


def frame_rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(key.encode("utf-8"))])


def _resize(image, rng, scale: float, restore: bool = False):
    if not 0.0 < scale <= 4.0:
        raise SensorSimError(f"resize scale must be in (0, 4], got {scale}")
    h, w = image.shape[:2]
    small = cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if restore else small


def _noise(image, rng, read_sigma: float = 4.0, shot: float = 0.0):
    x = image.astype(np.float32)
    sigma = np.sqrt(read_sigma**2 + shot * x)
    return np.clip(x + rng.standard_normal(x.shape, dtype=np.float32) * sigma, 0, 255).astype(np.uint8)


def _motion_blur(image, rng, length_frac: float = 0.004):
    length = max(1, int(round(length_frac * image.shape[1])))
    if length < 2:
        return image
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    angle = float(rng.uniform(0.0, 180.0))
    rot = cv2.getRotationMatrix2D(((length - 1) / 2.0, (length - 1) / 2.0), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def _gaussian_blur(image, rng, sigma_frac: float = 0.001):
    sigma = sigma_frac * image.shape[1]
    return image if sigma <= 0 else cv2.GaussianBlur(image, (0, 0), sigma)


def _jpeg(image, rng, quality: int = 20):
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise SensorSimError("JPEG encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _color(image, rng, gamma: float = 1.0, wb: list[float] | None = None):
    x = image.astype(np.float32) / 255.0
    if wb is not None:
        x = x * np.asarray(wb, np.float32).reshape(1, 1, 3)
    x = np.clip(x, 0.0, 1.0) ** float(gamma)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _gray(image, rng):
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _invert(image, rng):
    return 255 - image


_OPS = {
    "resize": _resize,
    "noise": _noise,
    "motion_blur": _motion_blur,
    "gaussian_blur": _gaussian_blur,
    "jpeg": _jpeg,
    "color": _color,
    "gray": _gray,
    "invert": _invert,
}


def validate_ops(ops: list[dict[str, Any]]) -> None:
    for spec in ops:
        name = spec.get("op")
        if name not in _OPS:
            raise SensorSimError(f"Unknown sensor op {name!r}; valid: {', '.join(SENSOR_OPS)}")


def apply_sensor(image, ops: list[dict[str, Any]], rng: np.random.Generator):
    """Apply `ops` in order to a BGR uint8 frame; returns a new frame."""
    validate_ops(ops)
    for spec in ops:
        params = {k: v for k, v in spec.items() if k != "op"}
        try:
            image = _OPS[spec["op"]](image, rng, **params)
        except TypeError as exc:
            raise SensorSimError(f"Bad parameters for sensor op {spec['op']!r}: {exc}") from exc
    return image


def apply_shake(image, rng: np.random.Generator, amplitude_frac: float):
    """Translate the frame by a per-frame random offset, uniform in
    ±`amplitude_frac` of the frame width (both axes), i.e. independent
    frame-to-frame jitter — a worst case for frame differencing, harsher
    than real handheld shake, which is temporally correlated. Uncovered
    borders replicate the edge pixels (no black bars, which would
    themselves be motion). Returns (frame, (dx, dy)) in pixels."""
    h, w = image.shape[:2]
    amp = amplitude_frac * w
    dx, dy = (int(round(v)) for v in rng.uniform(-amp, amp, size=2))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE), (dx, dy)


# Training-time domain randomization (DECISIONS #27, A2: PREPARED, NOT RUN).
# Each op is applied independently with its probability, parameters drawn
# per image. Every op keeps the image size (resize restores), so box
# targets need no change. No `invert`: polarity-flipped visible light is
# not thermal, and real IR needs real IR training data.
# CAVEAT: these op families overlap the eval variants
# (eval/config/fpv_test_stride8_sim_*.yaml), so a model trained with this
# recipe will look better on those variants partly by construction; real
# cross-sensor generalization must be measured on footage from other
# cameras.
SENSOR_DG_PROBS = {"resize": 0.3, "noise": 0.3, "motion_blur": 0.2, "gaussian_blur": 0.1, "jpeg": 0.3, "color": 0.5, "gray": 0.15}


def sample_sensor_dg(rng: np.random.Generator) -> list[dict[str, Any]]:
    """One random simulated-sensor recipe for a training image."""
    ops: list[dict[str, Any]] = []

    def maybe(name: str) -> bool:
        return bool(rng.random() < SENSOR_DG_PROBS[name])

    if maybe("color"):
        ops.append({"op": "color", "gamma": float(rng.uniform(0.6, 1.8)), "wb": [float(v) for v in rng.uniform(0.75, 1.25, 3)]})
    if maybe("gray"):
        ops.append({"op": "gray"})
    if maybe("resize"):
        ops.append({"op": "resize", "scale": float(rng.uniform(0.35, 1.0)), "restore": True})
    if maybe("motion_blur"):
        ops.append({"op": "motion_blur", "length_frac": float(rng.uniform(0.001, 0.006))})
    if maybe("gaussian_blur"):
        ops.append({"op": "gaussian_blur", "sigma_frac": float(rng.uniform(0.0003, 0.0015))})
    if maybe("noise"):
        ops.append({"op": "noise", "read_sigma": float(rng.uniform(1.0, 10.0)), "shot": float(rng.uniform(0.0, 0.6))})
    if maybe("jpeg"):
        ops.append({"op": "jpeg", "quality": int(rng.integers(15, 96))})
    return ops
