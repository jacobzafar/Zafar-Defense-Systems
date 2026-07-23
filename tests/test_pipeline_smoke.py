"""Smoke test: build a tiny synthetic video in memory, run the full pipeline,
assert it produces frame results without raising.
"""

import sys
import tempfile
import tomllib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.pipeline import Pipeline
from detector.motion_detector import MotionDetector
from tracker.iou_tracker import IoUTracker
from ui.overlay import draw_tracks

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_synthetic_video(path: str, num_frames: int = 20, size=(320, 240)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, size)
    width, height = size
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = 20 + i * 5
        cv2.rectangle(frame, (x, 50), (x + 30, 80), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_pipeline_runs_end_to_end_on_synthetic_video():
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = str(Path(tmp_dir) / "synthetic.mp4")
        _write_synthetic_video(video_path)

        pipeline = Pipeline(detector=MotionDetector(min_area_px=50), tracker=IoUTracker())
        results = list(pipeline.run(video_path))

        assert len(results) > 0
        assert all(r.inference_ms >= 0 for r in results)
        assert all(r.error is None or isinstance(r.error, str) for r in results)
        assert all(r.frame is not None for r in results)
        assert all(r.total_ms >= 0 for r in results)


def test_pipeline_runs_end_to_end_on_demo_source():
    """The built-in synthetic demo source needs no file/GPU/camera at all —
    this is the "verify before real footage exists" smoke test."""
    pipeline = Pipeline(detector=MotionDetector(min_area_px=40, var_threshold=16.0), tracker=IoUTracker())
    results = list(pipeline.run("demo"))

    assert len(results) > 0
    assert any(r.detection_count > 0 for r in results)
    assert all(r.frame is not None for r in results)
    assert all(r.fps >= 0 for r in results)


def test_demo_source_renders_as_video_not_a_black_screen():
    """Regression test for a bug where ui/app.py's video pane showed
    detection boxes over what looked like a solid black screen: the demo
    source's frame was real (not None, not corrupt) but its background
    brightness was nearly identical to the operator console's own dark
    theme, so the frame was visually indistinguishable from empty page
    background once rendered. This exercises the exact render sequence
    ui/app.py uses (Pipeline -> draw_tracks -> BGR2RGB) end to end and
    checks the actual displayed pixels, not just that a frame exists.
    """
    theme = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())["theme"]

    def _hex_brightness(hex_color: str) -> float:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return 0.299 * r + 0.587 * g + 0.114 * b

    chrome_brightness = max(
        _hex_brightness(theme["backgroundColor"]),
        _hex_brightness(theme["secondaryBackgroundColor"]),
    )

    pipeline = Pipeline(detector=MotionDetector(min_area_px=40, var_threshold=16.0), tracker=IoUTracker())
    checked_a_frame = False
    for result in pipeline.run("demo"):
        assert result.frame is not None
        annotated = draw_tracks(result.frame, result.tracks)
        displayed = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        pixels = displayed.reshape(-1, displayed.shape[-1])
        uniques, counts = np.unique(pixels, axis=0, return_counts=True)
        r, g, b = (int(c) for c in uniques[np.argmax(counts)])
        assert 0.299 * r + 0.587 * g + 0.114 * b > chrome_brightness + 30
        checked_a_frame = True

    assert checked_a_frame
