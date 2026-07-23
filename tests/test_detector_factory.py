import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.factory import build_detector
from detector.motion_detector import MotionDetector


def test_default_backend_is_motion():
    detector = build_detector()
    assert isinstance(detector, MotionDetector)


def test_unknown_backend_raises_with_helpful_message():
    with pytest.raises(ValueError, match="motion, ultralytics, torchvision"):
        build_detector({"backend": "not-a-real-backend"})
