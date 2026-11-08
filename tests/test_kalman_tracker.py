"""tracker/ego_motion.py and tracker/kalman_tracker.py (DECISIONS #27)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from detector.base import Detection  # noqa: E402
from tracker.ego_motion import EgoMotion, EgoMotionEstimator  # noqa: E402
from tracker.factory import build_tracker  # noqa: E402
from tracker.kalman_tracker import KalmanTracker  # noqa: E402

W, H = 640, 360


def _textured(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (H // 8, W // 8), dtype=np.uint8)
    return cv2.cvtColor(cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)


def _shift(image, dx, dy):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)


def _det(cx, cy, size=20, score=0.9):
    return Detection((cx - size / 2) / W, (cy - size / 2) / H, (cx + size / 2) / W, (cy + size / 2) / H, score, "drone")


def test_ego_motion_recovers_a_known_translation():
    est = EgoMotionEstimator(work_width=320)
    scene = _textured()
    assert est.update(scene) is None  # first frame
    motion = est.update(_shift(scene, 12, -7))
    assert motion.ok
    dx, dy = motion.translation
    assert dx == pytest.approx(12, abs=0.7) and dy == pytest.approx(-7, abs=0.7)


def test_ego_motion_reports_failure_on_a_featureless_frame():
    est = EgoMotionEstimator()
    flat = np.full((H, W, 3), 128, np.uint8)
    est.update(flat)
    motion = est.update(flat)
    assert motion is not None and not motion.ok
    assert np.array_equal(motion.matrix, np.array([[1.0, 0, 0], [0, 1.0, 0]]))


def _run(tracker, frames_dets):
    out = []
    for dets in frames_dets:
        tracker.observe_frame(np.zeros((H, W, 3), np.uint8))
        out.append(tracker.update(dets))
    return out


def test_confirmation_needs_m_of_n_and_low_scores_cannot_start_tracks():
    tracker = KalmanTracker(confirm_hits=2, confirm_window=3, ego_motion=False)
    out = _run(tracker, [[_det(100, 100)], [_det(104, 100)], [_det(108, 100)]])
    assert [len(t) for t in out] == [0, 1, 1]
    tracker = KalmanTracker(ego_motion=False)
    out = _run(tracker, [[_det(100, 100, score=0.2)]] * 4)
    assert all(not t for t in out)


def test_low_score_detections_extend_a_confirmed_track():
    tracker = KalmanTracker(confirm_hits=2, confirm_window=3, low_threshold=0.1, ego_motion=False, max_coast_output=0)
    frames = [[_det(100 + 4 * i, 100)] for i in range(3)] + [[_det(100 + 4 * i, 100, score=0.15)] for i in range(3, 8)]
    out = _run(tracker, frames)
    assert all(len(t) == 1 for t in out[1:])
    assert len({t[0].track_id for t in out[1:]}) == 1
    assert out[-1][0].confidence == pytest.approx(0.15)


def test_fast_small_target_keeps_one_id_where_boxes_never_overlap():
    """20 px drone moving 30 px/frame: consecutive boxes have IoU 0, so the
    IoU tracker switches ID every frame; constant velocity keeps it."""
    frames = [[_det(50 + 30 * i, 180)] for i in range(12)]
    kalman = _run(KalmanTracker(ego_motion=False), frames)
    assert len({t.track_id for ts in kalman for t in ts}) == 1
    iou = build_tracker({"backend": "iou"})
    ids = {t.track_id for dets in frames for t in iou.update(dets) if t.time_since_update == 0}
    assert len(ids) == 12


def test_camera_motion_is_applied_to_the_prediction():
    """A static drone on a camera panning 40 px/frame: in the image it moves
    -40 px/frame. With the true camera motion supplied, the prediction is
    exact from the second frame on, even right after a 0-velocity birth."""
    tracker = KalmanTracker(ego_motion=False, confirm_hits=1, confirm_window=1)
    pan = EgoMotion(np.array([[1.0, 0, -40.0], [0, 1.0, 0]]), True, 100, 100)
    ids = set()
    for i in range(8):
        tracker._frame_size = (W, H)
        tracker.apply_camera_motion(pan if i else None)
        tracks = tracker.update([_det(500 - 40 * i, 180)])
        ids |= {t.track_id for t in tracks}
    assert ids == {1}
    assert tracker.diagnostics()["innovation_median"] == pytest.approx(0.0, abs=1e-6)


def test_coasting_is_bounded_and_ghosts_are_not_output():
    tracker = KalmanTracker(ego_motion=False, confirm_hits=1, confirm_window=1, max_coast_output=2, max_age=5)
    out = _run(tracker, [[_det(100, 100)]] + [[]] * 4)
    assert [len(t) for t in out] == [1, 1, 1, 0, 0]
    assert out[1][0].extra["coasting"] is True


def test_failed_ego_motion_is_counted_and_widens_the_gate():
    """An unexplained 30 px jump of a static target (the camera moved, the
    estimate failed): outside the gate of a 3-frame-old track normally,
    inside it when the failure is flagged."""

    def run(flag_failure):
        tracker = KalmanTracker(ego_motion=False, confirm_hits=1, confirm_window=1)
        tracker._frame_size = (W, H)
        for i in range(3):
            tracker.apply_camera_motion(None if i == 0 else EgoMotion(np.array([[1.0, 0, 0], [0, 1.0, 0]]), True, 100, 100))
            tracker.update([_det(300, 180)])
        tracker.apply_camera_motion(EgoMotion(np.array([[1.0, 0, 0], [0, 1.0, 0]]), not flag_failure, 5, 0))
        return tracker, tracker.update([_det(330, 180)])

    tracker, tracks = run(flag_failure=True)
    assert [t.track_id for t in tracks] == [1]
    assert tracker.diagnostics()["ego_failures"] == 1
    _, tracks = run(flag_failure=False)
    assert sorted(t.track_id for t in tracks) == [1, 2]  # no match: the old track coasts, a new one is born


def test_factory_builds_kalman_with_options_and_rejects_bad_ones():
    tracker = build_tracker({"backend": "kalman", "low_threshold": 0.2, "ego_motion": False})
    assert isinstance(tracker, KalmanTracker) and tracker.min_detection_score == 0.2
    with pytest.raises(ValueError):
        KalmanTracker(low_threshold=0.5, high_threshold=0.35)
    with pytest.raises(ValueError):
        KalmanTracker(confirm_hits=4, confirm_window=3)


def test_harness_runs_kalman_with_frames_and_scores_its_output(tmp_path):
    """End to end on the numbered synthetic clip: the tracker gets frames
    (ego-motion diagnostics appear), its output is scored, and the
    compare table renders it."""
    from eval.compare import render_tracks
    from eval.harness import EvalConfig, run_evaluation
    from tests.test_temporal_eval import _numbered_eval_set

    root = _numbered_eval_set(tmp_path / "set", list(range(1, 11)) + [101, 102])
    card = run_evaluation(
        EvalConfig(
            eval_set_dir=str(root),
            detector={"backend": "motion", "min_area_px": 20, "history": 3},
            tracker={"backend": "kalman", "confirm_hits": 1, "confirm_window": 1},
            sequence_grouping="filename_runs",
        )
    )
    assert set(card.tracker_diagnostics) == {"Clip#1-10", "Clip#101-102"}
    assert card.tracker_diagnostics["Clip#1-10"]["ego_estimates"] == 9
    assert card.track_output["operating_point"]["num_gt_boxes"] == 10
    assert card.track_output["false_alarm_rate"]["num_hard_negative_frames"] == 2
    assert "tracker" in card.latency_by_stage
    table = render_tracks([("K", card.as_dict())], sequence="Clip#1-10").strip().splitlines()
    assert all(line.count("|") == table[0].count("|") for line in table)
