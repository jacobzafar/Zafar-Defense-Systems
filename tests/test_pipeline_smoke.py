"""Smoke test: build a tiny synthetic video in memory, run the full pipeline,
assert it produces frame results without raising.
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.pipeline import Pipeline
from detector.motion_detector import MotionDetector
from tracker.iou_tracker import IoUTracker


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
