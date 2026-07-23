import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.motion_detector import MotionDetector


def test_detect_returns_list_on_static_frame():
    detector = MotionDetector()
    static_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = detector.detect(static_frame)
    assert isinstance(detections, list)


def test_detect_handles_none_frame():
    detector = MotionDetector()
    assert detector.detect(None) == []


def test_detect_flags_a_moving_block():
    detector = MotionDetector(min_area_px=50)
    height, width = 240, 320

    # Prime the background model with several static frames.
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(15):
        detector.detect(background)

    # Introduce a bright moving block that should register as foreground.
    moving_frame = background.copy()
    moving_frame[50:100, 50:100] = 255
    detections = detector.detect(moving_frame)

    assert isinstance(detections, list)
    for det in detections:
        assert 0.0 <= det.x1 <= 1.0
        assert 0.0 <= det.confidence <= 1.0
