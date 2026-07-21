"""Synthetic in-memory frame source for demo/testing without real footage.

`SyntheticVideoSource` implements the small subset of `cv2.VideoCapture`'s
interface that `Pipeline.run()` needs (`isOpened`/`read`/`get`/`release`),
so the pipeline can treat it exactly like a real video source — no
special-casing anywhere except the one lookup in `Pipeline._open_source`.

It fabricates a short clip of a plain background with a handful of small
moving targets (filled circles) on simple bouncing trajectories, so the
default `MotionDetector` + `IoUTracker` combination has something concrete
to detect and track. This is demo/test scaffolding only — it proves the
pipeline, UI, and logging work end to end, not that detection/tracking
works on real drone footage.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

DEMO_SOURCE_KEYS = frozenset({"demo", "synthetic"})

_BACKGROUND = (18, 22, 26)  # BGR, matches the dark operator UI theme
_TARGET_COLOR = (225, 225, 225)


class SyntheticVideoSource:
    """Fabricates a deterministic short clip of moving targets in memory."""

    def __init__(
        self,
        num_frames: int = 150,
        width: int = 960,
        height: int = 540,
        fps: float = 20.0,
        num_targets: int = 3,
        seed: int = 7,
    ) -> None:
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.fps = fps
        self._frame_index = 0

        rng = random.Random(seed)
        self._targets = [
            {
                "x": rng.uniform(0.1, 0.9) * width,
                "y": rng.uniform(0.1, 0.9) * height,
                "vx": rng.choice([-1, 1]) * rng.uniform(2.5, 5.5),
                "vy": rng.choice([-1, 1]) * rng.uniform(1.5, 4.0),
                "radius": rng.randint(8, 14),
            }
            for _ in range(num_targets)
        ]

    def isOpened(self) -> bool:  # noqa: N802 - matches cv2.VideoCapture's API
        return True

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.num_frames)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._frame_index >= self.num_frames:
            return False, None

        frame = np.full((self.height, self.width, 3), _BACKGROUND, dtype=np.uint8)

        for target in self._targets:
            target["x"] += target["vx"]
            target["y"] += target["vy"]

            if target["x"] <= target["radius"] or target["x"] >= self.width - target["radius"]:
                target["vx"] *= -1
                target["x"] = min(max(target["x"], target["radius"]), self.width - target["radius"])
            if target["y"] <= target["radius"] or target["y"] >= self.height - target["radius"]:
                target["vy"] *= -1
                target["y"] = min(max(target["y"], target["radius"]), self.height - target["radius"])

            cv2.circle(
                frame,
                (int(target["x"]), int(target["y"])),
                target["radius"],
                _TARGET_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )

        self._frame_index += 1
        return True, frame

    def release(self) -> None:
        return None
