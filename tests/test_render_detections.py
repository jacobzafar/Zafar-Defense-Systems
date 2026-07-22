"""Tests for tools/render_detections.py, using the zero-dependency
`motion` detector so these stay fast and torch-free (mirrors
tests/test_eval_harness.py's approach) — a separate torch-gated test
exercises the real `drone` backend end-to-end.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.render_detections import (
    FrameRecord,
    load_dataset_frames,
    load_eval_set_frames,
    render_detections,
    select_frames,
)

_MOTION_CONFIG = {"backend": "motion", "min_area_px": 5, "var_threshold": 16.0, "history": 10}


def test_load_dataset_frames_matches_synthetic_fixture(synthetic_drone_dataset):
    frames = load_dataset_frames(synthetic_drone_dataset, target_class="drone")
    assert len(frames) == 6

    with_boxes = [f for f in frames if f.boxes]
    without_boxes = [f for f in frames if not f.boxes]
    assert len(with_boxes) == 3  # "drone" images
    assert len(without_boxes) == 3  # "bird" images have no "drone"-category boxes


def test_load_eval_set_frames_matches_synthetic_fixture(synthetic_eval_set):
    frames = load_eval_set_frames(synthetic_eval_set, target_class="drone")
    assert len(frames) == 8

    with_boxes = [f for f in frames if f.boxes]
    assert len(with_boxes) == 6  # 6 "drone" frames, 2 hard-negative "bird" frames


def test_select_frames_sequential_takes_first_n():
    frames = [FrameRecord(label=str(i), image_path=Path(f"{i}.jpg"), boxes=[(0, 0, i + 1, i + 1)]) for i in range(10)]
    selected = select_frames(frames, num_images=3, selection="sequential")
    assert [f.label for f in selected] == ["0", "1", "2"]


def test_select_frames_extremes_takes_smallest_and_largest():
    # areas: 1, 4, 9, ..., 100 for labels "0".."9"
    frames = [FrameRecord(label=str(i), image_path=Path(f"{i}.jpg"), boxes=[(0, 0, i + 1, i + 1)]) for i in range(10)]
    selected = select_frames(frames, num_images=4, selection="extremes")
    labels = {f.label for f in selected}
    assert labels == {"0", "1", "8", "9"}  # 2 smallest-area + 2 largest-area


def test_select_frames_extremes_excludes_frames_with_no_boxes():
    frames = [
        FrameRecord(label="empty", image_path=Path("empty.jpg"), boxes=[]),
        FrameRecord(label="small", image_path=Path("small.jpg"), boxes=[(0, 0, 2, 2)]),
        FrameRecord(label="large", image_path=Path("large.jpg"), boxes=[(0, 0, 20, 20)]),
    ]
    selected = select_frames(frames, num_images=4, selection="extremes")
    assert {f.label for f in selected} == {"small", "large"}


def test_render_detections_writes_frames_and_manifest(synthetic_eval_set, tmp_path):
    frames = load_eval_set_frames(synthetic_eval_set, target_class="drone")
    output_dir = tmp_path / "renders"

    summary = render_detections(frames, _MOTION_CONFIG, output_dir, fps=1)

    assert summary["num_frames_rendered"] == 8
    assert summary["total_gt_boxes"] == sum(len(f.boxes) for f in frames)
    assert summary["detector_backend"]

    frames_dir = Path(summary["frames_dir"])
    written_frames = sorted(frames_dir.glob("frame_*.jpg"))
    assert len(written_frames) == 8

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest == summary
    assert len(manifest["frames"]) == 8
    assert manifest["frames"][0]["frame_file"] == "frame_0001.jpg"

    # ffmpeg's presence on this machine determines whether an MP4 exists —
    # both are valid outcomes, but the flag and the file's existence must agree.
    assert (summary["video_path"] is not None) == (shutil.which("ffmpeg") is not None)
    if summary["video_path"]:
        assert Path(summary["video_path"]).exists()


def test_render_detections_produces_a_fixed_canvas_size_regardless_of_source_resolution(synthetic_eval_set, tmp_path):
    """DUT Anti-UAV images range from 160x240 to 3744x5616 — every
    rendered frame must letterbox to one consistent canvas so ffmpeg can
    stitch them into a single video stream at all."""
    cv2 = pytest.importorskip("cv2")
    frames = load_eval_set_frames(synthetic_eval_set, target_class="drone")
    output_dir = tmp_path / "renders"

    summary = render_detections(frames, _MOTION_CONFIG, output_dir, fps=1)

    for frame_file in Path(summary["frames_dir"]).glob("frame_*.jpg"):
        image = cv2.imread(str(frame_file))
        height, width = image.shape[:2]
        assert (width, height) == (960, 720)


def test_render_detections_skips_unreadable_images(tmp_path):
    output_dir = tmp_path / "renders"
    frames = [FrameRecord(label="missing", image_path=tmp_path / "does-not-exist.jpg", boxes=[])]

    summary = render_detections(frames, _MOTION_CONFIG, output_dir, fps=1)
    assert summary["num_frames_rendered"] == 0


def test_render_detections_works_with_the_real_drone_backend(synthetic_drone_dataset, tmp_path):
    """Exercises the actual production code path (backend: drone,
    weights_path) end-to-end, not just the zero-dependency motion
    detector — this is what the real DUT Anti-UAV render run uses."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from detector.train import TrainConfig, train

    output_dir = tmp_path / "train_output"
    train(TrainConfig(dataset_dir=str(synthetic_drone_dataset), output_dir=str(output_dir), epochs=1, batch_size=2, val_fraction=0.2, seed=42))
    weights_path = output_dir / "weights.pt"

    frames = load_dataset_frames(synthetic_drone_dataset, target_class="drone")
    summary = render_detections(
        frames,
        {"backend": "drone", "weights_path": str(weights_path), "confidence_threshold": 0.0},
        tmp_path / "renders",
        fps=1,
    )
    assert summary["detector_backend"] == "drone_v1"
    assert summary["num_frames_rendered"] == 6
