import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.synthetic_source import DEMO_SOURCE_KEYS, SyntheticVideoSource


def test_demo_source_keys_include_common_aliases():
    assert "demo" in DEMO_SOURCE_KEYS
    assert "synthetic" in DEMO_SOURCE_KEYS


def test_is_opened_is_always_true():
    source = SyntheticVideoSource(num_frames=3)
    assert source.isOpened()


def test_read_yields_expected_number_of_frames():
    source = SyntheticVideoSource(num_frames=5, width=64, height=48)
    frames_read = 0
    while True:
        ok, frame = source.read()
        if not ok:
            break
        assert frame.shape == (48, 64, 3)
        frames_read += 1
    assert frames_read == 5


def test_read_after_exhaustion_returns_false_none():
    source = SyntheticVideoSource(num_frames=1)
    source.read()
    ok, frame = source.read()
    assert ok is False
    assert frame is None


def test_get_reports_configured_properties():
    source = SyntheticVideoSource(num_frames=10, width=100, height=80, fps=15.0)
    assert source.get(cv2.CAP_PROP_FRAME_WIDTH) == 100.0
    assert source.get(cv2.CAP_PROP_FRAME_HEIGHT) == 80.0
    assert source.get(cv2.CAP_PROP_FPS) == 15.0
    assert source.get(cv2.CAP_PROP_FRAME_COUNT) == 10.0


def test_release_does_not_raise():
    source = SyntheticVideoSource(num_frames=1)
    source.release()  # should not raise


def test_same_seed_is_deterministic():
    a = SyntheticVideoSource(num_frames=5, width=64, height=48, seed=42)
    b = SyntheticVideoSource(num_frames=5, width=64, height=48, seed=42)
    for _ in range(5):
        ok_a, frame_a = a.read()
        ok_b, frame_b = b.read()
        assert ok_a == ok_b
        assert (frame_a == frame_b).all()
