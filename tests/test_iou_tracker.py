import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.base import Detection
from tracker.iou_tracker import IoUTracker


def make_detection(x1, y1, x2, y2, conf=0.9, cls="drone"):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_name=cls)


def test_new_detection_creates_new_track():
    tracker = IoUTracker()
    tracks = tracker.update([make_detection(0.1, 0.1, 0.2, 0.2)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 1


def test_same_position_keeps_same_id_across_frames():
    tracker = IoUTracker()
    tracks_1 = tracker.update([make_detection(0.1, 0.1, 0.2, 0.2)])
    tracks_2 = tracker.update([make_detection(0.11, 0.11, 0.21, 0.21)])
    assert tracks_1[0].track_id == tracks_2[0].track_id
    assert tracks_2[0].hits == 2


def test_track_is_dropped_after_max_age_without_updates():
    tracker = IoUTracker(max_age=2)
    tracker.update([make_detection(0.1, 0.1, 0.2, 0.2)])
    tracker.update([])
    tracker.update([])
    tracks = tracker.update([])
    assert tracks == []


def test_far_apart_detections_get_different_ids():
    tracker = IoUTracker()
    tracker.update([make_detection(0.0, 0.0, 0.1, 0.1)])
    tracks = tracker.update([make_detection(0.8, 0.8, 0.9, 0.9)])
    # The first track is still alive (not yet aged past max_age) and the
    # new detection must have been assigned a distinct, new ID.
    track_ids = {t.track_id for t in tracks}
    assert track_ids == {1, 2}


def test_reset_clears_all_tracks():
    tracker = IoUTracker()
    tracker.update([make_detection(0.1, 0.1, 0.2, 0.2)])
    tracker.reset()
    tracks = tracker.update([make_detection(0.1, 0.1, 0.2, 0.2)])
    assert tracks[0].track_id == 1
