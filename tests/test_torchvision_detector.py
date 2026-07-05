import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from detector.factory import build_detector
from detector.torchvision_detector import TorchvisionDetector


def test_detect_returns_list_on_random_frame():
    detector = TorchvisionDetector(confidence_threshold=0.9)
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    for det in detections:
        assert 0.0 <= det.x1 <= 1.0
        assert 0.0 <= det.confidence <= 1.0


def test_detect_handles_none_frame():
    detector = TorchvisionDetector()
    assert detector.detect(None) == []


def test_target_classes_filter_restricts_output():
    detector = TorchvisionDetector(confidence_threshold=0.0, target_classes=["airplane"])
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    detections = detector.detect(frame)
    assert all(det.class_name == "airplane" for det in detections)


def test_factory_builds_torchvision_backend():
    detector = build_detector({"backend": "torchvision"})
    assert isinstance(detector, TorchvisionDetector)
    assert detector.target_classes == ["airplane", "bird", "kite"]
