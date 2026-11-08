"""Kalman tracker with camera-motion compensation and two-stage association.

Built for the failure modes the IoU tracker showed on FPV footage
(DECISIONS #27): small, fast-moving drones on a moving camera, where
consecutive boxes often do not overlap at all, and a detector that still
sees the drone in most frames but with a score under the operating
threshold.

- **State**: centre (cx, cy) and velocity (vx, vy) in pixels, constant-
  velocity model; box size is a running average of matched detections.
  Noise scales with the box size (as in SORT/ByteTrack), so a 30 px drone
  and a 300 px one are gated alike.
- **Camera motion**: before each prediction the track is moved by the
  frame-to-frame camera motion from `tracker.ego_motion` (when
  `observe_frame` is called with the frames). If that estimate fails, the
  prediction is uncompensated and its position uncertainty is inflated by
  `ego_fail_std_frac` of the frame width instead — a wider gate, never a
  silent wrong one. Failures are counted in `diagnostics()`.
- **Association** (greedy minimum Mahalanobis distance of the centre,
  chi-square gated, plus an area-ratio gate):
  1. detections with score >= `high_threshold` against all tracks;
  2. detections with `low_threshold` <= score < `high_threshold` against
     the confirmed tracks still unmatched — a low score can extend a
     track, never start one (ByteTrack's idea);
  3. unmatched high detections start tentative tracks.
  Within a stage, `prefer: distance` matches each track to its nearest
  gated detection; `prefer: score` to its highest-scoring gated one
  (distance breaks ties). Several boxes often survive NMS around a small
  drone, and the nearest to a poor prediction is not always the best
  placed (DECISIONS #27: Sunny-late).
- **Confirmation**: a tentative track is confirmed after `confirm_hits`
  matches within its first `confirm_window` frames, else dropped.
- **Output**: confirmed tracks only, and only while they have been
  unmatched for at most `max_coast_output` frames (a coasting box is the
  prediction). Tracks are kept internally up to `max_age` frames so they
  can be re-acquired, but are not shown — no ghost boxes.

Greedy matching instead of Hungarian: with one or a few targets per
frame and gated candidates the two coincide in practice, and it avoids a
scipy dependency.

The harness passes every detection scoring >= `min_detection_score`
(= `low_threshold`) to this tracker instead of only those above the
operating threshold. In the live pipeline the detector's own
`confidence_threshold` must be set to the tracker's `low_threshold` for
stage 2 to see anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from detector.base import Detection
from tracker.base import BaseTracker, Track
from tracker.ego_motion import EgoMotion, EgoMotionEstimator

CHI2_2DOF_99 = 9.21
_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)


@dataclass
class _KTrack:
    track_id: int
    x: np.ndarray  # [cx, cy, vx, vy], pixels
    P: np.ndarray
    w: float
    h: float
    score: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    matched_box: tuple[float, float, float, float] | None = None  # this frame's detection, pixels
    class_name: str = "drone"
    extra: dict = field(default_factory=dict)

    @property
    def size(self) -> float:
        return max(np.sqrt(self.w * self.h), 1.0)


class KalmanTracker(BaseTracker):
    name = "kalman_v1"

    def __init__(
        self,
        high_threshold: float = 0.35,
        low_threshold: float = 0.1,
        confirm_hits: int = 2,
        confirm_window: int = 3,
        max_coast_output: int = 2,
        max_age: int = 15,
        pos_std: float = 1 / 20,
        vel_std: float = 1 / 160,
        meas_std: float = 1 / 20,
        init_vel_std: float = 1.0,
        gate_chi2: float = CHI2_2DOF_99,
        max_area_ratio: float = 4.0,
        ego_motion: bool = True,
        ego_fail_std_frac: float = 0.02,
        size_smoothing: float = 0.5,
        prefer: str = "distance",
    ) -> None:
        if not 0.0 <= low_threshold <= high_threshold:
            raise ValueError(f"Need 0 <= low_threshold <= high_threshold, got {low_threshold}, {high_threshold}")
        if not 1 <= confirm_hits <= confirm_window:
            raise ValueError(f"Need 1 <= confirm_hits <= confirm_window, got {confirm_hits}, {confirm_window}")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_detection_score = low_threshold  # read by eval/harness.py
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.max_coast_output = max_coast_output
        self.max_age = max_age
        self.pos_std = pos_std
        self.vel_std = vel_std
        self.meas_std = meas_std
        # A new track's velocity is unknown: its std is `init_vel_std` box
        # sizes per frame, so a small drone moving about its own length per
        # frame can still be matched on its second frame.
        self.init_vel_std = init_vel_std
        self.gate_chi2 = gate_chi2
        self.max_area_ratio = max_area_ratio
        self.use_ego_motion = ego_motion
        self.ego_fail_std_frac = ego_fail_std_frac
        self.size_smoothing = size_smoothing
        if prefer not in ("distance", "score"):
            raise ValueError(f"prefer must be 'distance' or 'score', got {prefer!r}")
        self.prefer = prefer
        self._estimator = EgoMotionEstimator() if ego_motion else None
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_KTrack] = []
        self._next_id = 1
        self._frame_size: tuple[int, int] = (1, 1)  # (width, height); normalized units until a frame is seen
        self._motion: EgoMotion | None = None
        self._motion_pending = False
        if self._estimator is not None:
            self._estimator.reset()
        self._stats = {"frames": 0, "ego_estimates": 0, "ego_failures": 0, "innovations": [], "camera_shift_px": []}

    # --- frame / camera motion ------------------------------------------------

    def observe_frame(self, frame) -> None:
        """Give the tracker the current frame (before `update`), for its
        size and, if enabled, the camera motion since the previous frame."""
        self._frame_size = (frame.shape[1], frame.shape[0])
        if self._estimator is not None:
            self.apply_camera_motion(self._estimator.update(frame))

    def apply_camera_motion(self, motion: EgoMotion | None) -> None:
        """Set the camera motion to apply at the next `update` (None = first
        frame, nothing to apply). Used directly by eval/track_sweep.py with
        precomputed estimates."""
        self._motion = motion
        self._motion_pending = True

    # --- Kalman steps -----------------------------------------------------------

    def _predict(self, track: _KTrack, motion: EgoMotion | None) -> None:
        s = track.size
        if motion is not None and motion.ok:
            a = motion.matrix[:, :2]
            cx, cy = motion.apply(track.x[0], track.x[1])
            v = a @ track.x[2:]
            track.x = np.array([cx, cy, v[0], v[1]])
            j = np.zeros((4, 4))
            j[:2, :2] = a
            j[2:, 2:] = a
            track.P = j @ track.P @ j.T
            track.w, track.h = track.w * motion.scale, track.h * motion.scale
        q = np.diag([(self.pos_std * s) ** 2] * 2 + [(self.vel_std * s) ** 2] * 2)
        if motion is not None and not motion.ok:
            inflate = (self.ego_fail_std_frac * self._frame_size[0]) ** 2
            q = q + np.diag([inflate, inflate, 0.0, 0.0])
        track.x = _F @ track.x
        track.P = _F @ track.P @ _F.T + q

    def _innovation(self, track: _KTrack, cx: float, cy: float):
        r = np.eye(2) * (self.meas_std * track.size) ** 2
        s_mat = _H @ track.P @ _H.T + r
        y = np.array([cx, cy]) - _H @ track.x
        return y, s_mat

    def _gated_cost(self, track: _KTrack, box) -> float | None:
        x1, y1, x2, y2 = box
        area = max((x2 - x1) * (y2 - y1), 1e-9)
        ratio = area / max(track.w * track.h, 1e-9)
        if not 1.0 / self.max_area_ratio <= ratio <= self.max_area_ratio:
            return None
        y, s_mat = self._innovation(track, (x1 + x2) / 2, (y1 + y2) / 2)
        d2 = float(y @ np.linalg.solve(s_mat, y))
        return d2 if d2 <= self.gate_chi2 else None

    def _correct(self, track: _KTrack, box, score: float) -> None:
        x1, y1, x2, y2 = box
        y, s_mat = self._innovation(track, (x1 + x2) / 2, (y1 + y2) / 2)
        self._stats["innovations"].append(float(np.hypot(*y)) / track.size)
        gain = track.P @ _H.T @ np.linalg.inv(s_mat)
        track.x = track.x + gain @ y
        track.P = (np.eye(4) - gain @ _H) @ track.P
        a = self.size_smoothing
        track.w = a * track.w + (1 - a) * (x2 - x1)
        track.h = a * track.h + (1 - a) * (y2 - y1)
        track.score = score
        track.hits += 1
        track.time_since_update = 0
        track.matched_box = box

    @staticmethod
    def _greedy(costs: list[tuple]) -> list[tuple[int, int]]:
        """Pairs in ascending order of the cost tuples' leading keys; each
        tuple ends with (track index, detection index)."""
        pairs, used_t, used_d = [], set(), set()
        for *_keys, t, d in sorted(costs):
            if t not in used_t and d not in used_d:
                pairs.append((t, d))
                used_t.add(t)
                used_d.add(d)
        return pairs

    # --- main loop ---------------------------------------------------------------

    def update(self, detections: list[Detection]) -> list[Track]:
        width, height = self._frame_size
        motion = self._motion if self._motion_pending else None
        self._motion_pending = False
        self._stats["frames"] += 1
        if motion is not None:
            self._stats["ego_estimates"] += 1
            self._stats["ego_failures"] += int(not motion.ok)
            if motion.ok:
                self._stats["camera_shift_px"].append(float(np.hypot(*motion.translation)))

        for track in self._tracks:
            track.matched_box = None
            track.age += 1
            track.time_since_update += 1
            self._predict(track, motion)

        boxes = [(d.x1 * width, d.y1 * height, d.x2 * width, d.y2 * height) for d in detections]
        high = [i for i, d in enumerate(detections) if d.confidence >= self.high_threshold]
        low = [i for i, d in enumerate(detections) if self.low_threshold <= d.confidence < self.high_threshold]

        def associate(track_idx: list[int], det_idx: list[int]) -> list[tuple[int, int]]:
            costs = []
            for t in track_idx:
                for d in det_idx:
                    cost = self._gated_cost(self._tracks[t], boxes[d])
                    if cost is not None:
                        key = (-detections[d].confidence, cost) if self.prefer == "score" else (cost,)
                        costs.append((*key, t, d))
            return self._greedy(costs)

        all_tracks = list(range(len(self._tracks)))
        first = associate(all_tracks, high)
        for t, d in first:
            self._correct(self._tracks[t], boxes[d], detections[d].confidence)
        matched_t = {t for t, _ in first}
        remaining_confirmed = [t for t in all_tracks if t not in matched_t and self._tracks[t].confirmed]
        for t, d in associate(remaining_confirmed, low):
            self._correct(self._tracks[t], boxes[d], detections[d].confidence)

        matched_d = {d for _, d in first}
        for d in high:
            if d in matched_d:
                continue
            x1, y1, x2, y2 = boxes[d]
            w, h = x2 - x1, y2 - y1
            s = max(np.sqrt(w * h), 1.0)
            p = np.diag([(2 * self.meas_std * s) ** 2] * 2 + [(self.init_vel_std * s) ** 2] * 2)
            self._tracks.append(
                _KTrack(
                    track_id=self._next_id,
                    x=np.array([(x1 + x2) / 2, (y1 + y2) / 2, 0.0, 0.0]),
                    P=p,
                    w=w,
                    h=h,
                    score=detections[d].confidence,
                    matched_box=boxes[d],
                    class_name=detections[d].class_name,
                )
            )
            self._next_id += 1

        kept = []
        for track in self._tracks:
            if not track.confirmed and track.hits >= self.confirm_hits:
                track.confirmed = True
            if not track.confirmed and track.age >= self.confirm_window:
                continue  # never confirmed in its window
            if track.time_since_update > self.max_age:
                continue
            kept.append(track)
        self._tracks = kept

        return [self._as_track(t, width, height) for t in self._tracks if t.confirmed and t.time_since_update <= self.max_coast_output]

    def _as_track(self, track: _KTrack, width: int, height: int) -> Track:
        if track.matched_box is not None:
            x1, y1, x2, y2 = track.matched_box
        else:
            cx, cy = track.x[0], track.x[1]
            x1, y1, x2, y2 = cx - track.w / 2, cy - track.h / 2, cx + track.w / 2, cy + track.h / 2
        return Track(
            track_id=track.track_id,
            x1=x1 / width,
            y1=y1 / height,
            x2=x2 / width,
            y2=y2 / height,
            confidence=track.score,
            class_name=track.class_name,
            age_frames=track.age,
            hits=track.hits,
            time_since_update=track.time_since_update,
            extra={"coasting": track.matched_box is None},
        )

    def diagnostics(self) -> dict:
        """Per-sequence: ego-motion estimates and failures, camera shift, and
        the normalized innovation (|measured - predicted centre| / box size)
        of every matched update — how well prediction anticipated the target."""
        innov = self._stats["innovations"]
        shift = self._stats["camera_shift_px"]
        est = self._stats["ego_estimates"]
        return {
            "frames": self._stats["frames"],
            "ego_motion": self.use_ego_motion,
            "ego_estimates": est,
            "ego_failures": self._stats["ego_failures"],
            "ego_failure_rate": (self._stats["ego_failures"] / est) if est else None,
            "camera_shift_px_median": float(np.median(shift)) if shift else None,
            "camera_shift_px_p95": float(np.percentile(shift, 95)) if shift else None,
            "innovation_median": float(np.median(innov)) if innov else None,
            "innovation_p95": float(np.percentile(innov, 95)) if innov else None,
            "num_updates": len(innov),
        }
