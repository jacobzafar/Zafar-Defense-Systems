"""Frame-to-frame camera motion (ego-motion) from sparse optical flow.

A moving camera moves every object in the image. A tracker that predicts
where a target will be next frame must add the camera's motion to the
target's own, or a pan of a few percent of the frame per frame — far more
than a distant drone's own image motion — breaks association
(DECISIONS #27: the IoU tracker's ID switches on Sunny-late's pan).

Method: corners (Shi-Tomasi) on the previous frame at reduced resolution,
tracked into the current frame with pyramidal Lucas-Kanade, then a 4-DoF
similarity (translation, rotation, uniform scale) fitted with RANSAC
(`cv2.estimateAffinePartial2D`). A similarity, not a full homography: for
distant backgrounds and small inter-frame motion it is enough, and it is
far less prone to fitting a degenerate warp when most corners sit on a
small textured region. The target itself is a handful of corners at most;
RANSAC treats it as an outlier.

Every estimate carries a quality verdict. When fewer than `min_inliers`
corners survive, or their share is below `min_inlier_ratio` (sky, blur,
a pan too fast for the flow pyramid), the estimate is marked failed and
identity is returned: the caller then predicts without camera
compensation, and the failure is counted, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EgoMotion:
    """Camera motion previous -> current frame, in full-resolution pixels.

    `matrix` is 2x3: p_current = matrix @ [x, y, 1] for a static point."""

    matrix: np.ndarray
    ok: bool
    num_points: int
    num_inliers: int

    @property
    def translation(self) -> tuple[float, float]:
        return float(self.matrix[0, 2]), float(self.matrix[1, 2])

    def apply(self, x: float, y: float) -> tuple[float, float]:
        m = self.matrix
        return float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2])

    @property
    def scale(self) -> float:
        return float(np.hypot(self.matrix[0, 0], self.matrix[1, 0]))


IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class EgoMotionEstimator:
    def __init__(
        self,
        work_width: int = 480,
        max_corners: int = 400,
        min_inliers: int = 20,
        min_inlier_ratio: float = 0.5,
        ransac_reproj_px: float = 1.5,
    ) -> None:
        self.work_width = work_width
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_reproj_px = ransac_reproj_px
        self._prev: np.ndarray | None = None
        self._prev_scale = 1.0

    def reset(self) -> None:
        self._prev = None

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.work_width / gray.shape[1])
        if scale < 1.0:
            gray = cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return gray, scale

    def update(self, frame: np.ndarray) -> EgoMotion | None:
        """Motion from the previous call's frame to this one; None on the
        first frame of a sequence (nothing to compare against)."""
        gray, scale = self._prepare(frame)
        prev, self._prev, self._prev_scale = self._prev, gray, scale
        if prev is None:
            return None
        if prev.shape != gray.shape:
            return EgoMotion(IDENTITY.copy(), False, 0, 0)

        corners = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8)
        if corners is None or len(corners) < self.min_inliers:
            return EgoMotion(IDENTITY.copy(), False, 0 if corners is None else len(corners), 0)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, corners, None, winSize=(21, 21), maxLevel=4)
        good = status.ravel() == 1
        src, dst = corners[good].reshape(-1, 2), moved[good].reshape(-1, 2)
        if len(src) < self.min_inliers:
            return EgoMotion(IDENTITY.copy(), False, len(src), 0)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_px, maxIters=2000, confidence=0.99
        )
        num_inliers = 0 if inliers is None else int(inliers.sum())
        if matrix is None or num_inliers < self.min_inliers or num_inliers / len(src) < self.min_inlier_ratio:
            return EgoMotion(IDENTITY.copy(), False, len(src), num_inliers)

        # Back to full-resolution pixels: translation scales with 1/scale.
        matrix = matrix.astype(float)
        matrix[:, 2] /= scale
        return EgoMotion(matrix, True, len(src), num_inliers)
