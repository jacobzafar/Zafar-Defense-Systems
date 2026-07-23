"""Tests for tools/preannotate.py, using the zero-dependency `motion`
detector so these stay fast and torch-free.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import preannotate  # noqa: E402


def _write_synthetic_video(path: Path, num_frames: int = 12, size=(160, 120)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    width, height = size
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = 10 + i * 5
        cv2.rectangle(frame, (x, 40), (x + 20, 60), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


_DETECTOR_CONFIG = {"backend": "motion", "min_area_px": 20, "var_threshold": 16.0, "history": 30}


def test_preannotate_video_produces_valid_coco_structure(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_synthetic_video(video_path)
    output_dir = tmp_path / "out"

    coco = preannotate.preannotate(str(video_path), output_dir, _DETECTOR_CONFIG)

    assert "images" in coco and "annotations" in coco and "categories" in coco
    assert len(coco["images"]) == 12
    assert all((output_dir / "images" / img["file_name"]).exists() for img in coco["images"])

    image_ids = {img["id"] for img in coco["images"]}
    for ann in coco["annotations"]:
        assert ann["image_id"] in image_ids
        assert len(ann["bbox"]) == 4
        assert 0.0 <= ann["score"] <= 1.0
        assert ann["category_id"] in {c["id"] for c in coco["categories"]}


def test_preannotate_writes_coco_predictions_json(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_synthetic_video(video_path)
    output_dir = tmp_path / "out"

    preannotate.preannotate(str(video_path), output_dir, _DETECTOR_CONFIG)

    json_path = output_dir / "coco_predictions.json"
    assert json_path.exists()
    written = json.loads(json_path.read_text())
    assert written["info"]["detector_backend"] == "motion_v1"
    assert "NOT human-verified" in written["info"]["description"]


def test_preannotate_respects_frame_stride(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_synthetic_video(video_path, num_frames=12)
    output_dir = tmp_path / "out"

    coco = preannotate.preannotate(str(video_path), output_dir, _DETECTOR_CONFIG, frame_stride=3)

    assert len(coco["images"]) == 4  # frames 0, 3, 6, 9


def test_preannotate_respects_max_frames(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_synthetic_video(video_path, num_frames=12)
    output_dir = tmp_path / "out"

    coco = preannotate.preannotate(str(video_path), output_dir, _DETECTOR_CONFIG, max_frames=5)

    assert len(coco["images"]) == 5


def test_preannotate_image_directory_mode(tmp_path):
    images_dir = tmp_path / "frames"
    images_dir.mkdir()
    for i in range(4):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[10:20, 10 + i * 5 : 20 + i * 5] = 255
        cv2.imwrite(str(images_dir / f"img_{i}.jpg"), frame)

    output_dir = tmp_path / "out"
    coco = preannotate.preannotate(str(images_dir), output_dir, _DETECTOR_CONFIG)

    assert len(coco["images"]) == 4
    file_names = {img["file_name"] for img in coco["images"]}
    assert file_names == {f"img_{i}.jpg" for i in range(4)}


def test_bad_video_source_raises_runtime_error(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open"):
        preannotate.preannotate("/nonexistent/video.mp4", tmp_path / "out", _DETECTOR_CONFIG)


def test_build_detector_config_requires_weights_for_drone_backend():
    args = argparse_namespace(detector="drone", preset="default", weights=None, confidence_threshold=None)
    with pytest.raises(ValueError, match="--weights is required"):
        preannotate.build_detector_config(args)


def argparse_namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)
