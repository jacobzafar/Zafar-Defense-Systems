import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.synthetic_source import DEMO_SOURCE_KEYS, SyntheticVideoSource

REPO_ROOT = Path(__file__).resolve().parent.parent


def _perceptual_brightness(r: int, g: int, b: int) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def _dominant_pixel_bgr(frame: np.ndarray) -> np.ndarray:
    """The background fills nearly every pixel, so it's the modal row."""
    pixels = frame.reshape(-1, frame.shape[-1])
    uniques, counts = np.unique(pixels, axis=0, return_counts=True)
    return uniques[np.argmax(counts)]


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


def test_background_is_visually_distinct_from_the_dark_ui_chrome():
    """Regression test for the black-screen bug: the demo source's
    background previously had brightness ~23 (BGR (18, 22, 26)), almost
    identical to the operator console's own dark theme (see
    .streamlit/config.toml) — so the "video" rendered visually
    indistinguishable from empty page background, with only the
    detection-box overlay visible. This reads the theme colors directly
    from config so a future background color change can't silently
    reintroduce the same near-invisible frame.
    """
    theme = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())["theme"]

    def _hex_brightness(hex_color: str) -> float:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return _perceptual_brightness(r, g, b)

    chrome_brightness = max(
        _hex_brightness(theme["backgroundColor"]),
        _hex_brightness(theme["secondaryBackgroundColor"]),
    )

    source = SyntheticVideoSource(num_frames=1, width=64, height=48)
    _, frame = source.read()
    b, g, r = (int(c) for c in _dominant_pixel_bgr(frame))
    background_brightness = _perceptual_brightness(r, g, b)

    # A wide margin, not a hair's-width one: the point is that a viewer can
    # immediately tell "there is a frame here," not just that two numbers
    # differ.
    assert background_brightness > chrome_brightness + 30
