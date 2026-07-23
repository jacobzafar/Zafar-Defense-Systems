import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracker.base import Track
from ui.overlay import draw_tracks

_BOX_COLOR_BGR = np.array([66, 145, 255])


def _make_track(track_id: int, x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9) -> Track:
    return Track(
        track_id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence,
        class_name="drone", age_frames=1, hits=1, time_since_update=0,
    )


def test_draw_tracks_returns_a_copy_not_the_original_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = draw_tracks(frame, [])
    assert annotated is not frame
    assert (frame == 0).all()  # untouched


def test_draw_tracks_draws_something_for_each_track():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    tracks = [_make_track(1, 0.1, 0.1, 0.3, 0.3)]
    annotated = draw_tracks(frame, tracks)
    assert annotated.shape == frame.shape
    assert annotated.any()  # no longer all-black now that a box is drawn


def test_label_stays_within_frame_for_a_track_near_the_right_edge():
    """Regression test: a track near the right edge previously had its
    label background drawn past the frame boundary and silently clipped
    by OpenCV there (e.g. the confidence value became unreadable), since
    the label was always anchored at the box's x1 with no clamping. The
    label must now be shifted left to stay fully on-canvas instead.
    """
    width, height = 320, 240
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    # A small box hard against the right edge, like a target exiting frame.
    track = _make_track(6, x1=0.9, y1=0.4, x2=0.98, y2=0.5, confidence=0.41)

    label = f"ID {track.track_id} | {track.class_name} | {track.confidence:.2f}"
    (text_w, _text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x1 = int(track.x1 * width)
    y1 = int(track.y1 * height)
    assert x1 + text_w + 4 > width, "test setup must actually exercise the overflow case"

    annotated = draw_tracks(frame, [track])

    row = y1 - 4  # inside the label background band, above the box's own outline
    filled_columns = np.where((annotated[row] == _BOX_COLOR_BGR).all(axis=-1))[0]

    assert filled_columns.size > 0, "label background should be visible in this row"
    assert filled_columns.min() < x1, "label should be shifted left of the unclamped position"
    assert filled_columns.max() - filled_columns.min() >= text_w, "full label width must be intact, not clipped"
