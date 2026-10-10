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


# --- Ingested-clip mode, Label Studio output, --max-per-frame ---------------

from detector.base import Detection  # noqa: E402


class _StubDetector:
    """Known boxes/scores per frame, so outputs can be checked exactly."""

    name = "stub_v0"

    def detect(self, frame):
        return [
            Detection(x1=0.1, y1=0.2, x2=0.3, y2=0.6, confidence=0.4, class_name="drone"),
            Detection(x1=0.5, y1=0.5, x2=0.6, y2=0.7, confidence=0.9, class_name="drone"),
            Detection(x1=0.0, y1=0.0, x2=0.1, y2=0.1, confidence=0.6, class_name="bird"),
        ]


@pytest.fixture
def stub_detector(monkeypatch):
    monkeypatch.setattr(preannotate, "build_detector", lambda config: _StubDetector())


def test_clip_mode_reads_frames_in_place_and_writes_to_prelabels(ingested_clip, stub_detector):
    coco = preannotate.preannotate(str(ingested_clip), None, {"backend": "stub"})

    out = ingested_clip / "prelabels"
    assert {p.name for p in out.iterdir()} == {"coco_predictions.json", "labelstudio_tasks.json", "labelstudio_config.xml", "README.txt"}
    assert not (out / "images").exists()  # frames are not copied
    frames = json.loads((ingested_clip / "frames.json").read_text())["images"]
    assert [img["file_name"] for img in coco["images"]] == [f["file_name"] for f in frames]
    assert coco["info"]["review_status"] == "unreviewed_predictions"


def test_review_labels_always_offered_with_drone_first(ingested_clip, stub_detector):
    coco = preannotate.preannotate(str(ingested_clip), None, {"backend": "stub"})
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [(1, "drone"), (2, "bird"), (3, "aircraft"), (4, "other")]
    config_xml = (ingested_clip / "prelabels" / "labelstudio_config.xml").read_text()
    for name in ("drone", "bird", "aircraft", "other"):
        assert f'<Label value="{name}"/>' in config_xml


def test_max_per_frame_keeps_the_highest_scoring_boxes(ingested_clip, stub_detector):
    coco = preannotate.preannotate(str(ingested_clip), None, {"backend": "stub"}, max_per_frame=2)
    per_image = {}
    for ann in coco["annotations"]:
        per_image.setdefault(ann["image_id"], []).append(ann["score"])
    assert len(per_image) == 10
    assert all(sorted(scores) == [0.6, 0.9] for scores in per_image.values())


def test_labelstudio_tasks_hold_boxes_as_predictions_in_percent(ingested_clip, stub_detector):
    preannotate.preannotate(str(ingested_clip), None, {"backend": "stub"}, max_per_frame=1)
    tasks = json.loads((ingested_clip / "prelabels" / "labelstudio_tasks.json").read_text())

    assert len(tasks) == 10
    task = tasks[0]
    assert "annotations" not in task  # pre-labels never pose as human annotations
    assert task["data"]["image"] == f"/data/local-files/?d=c0001/images/{task['data']['file_name']}"
    [prediction] = task["predictions"]
    assert prediction["model_version"] == "stub_v0"
    [result] = prediction["result"]
    assert result["value"]["rectanglelabels"] == ["drone"]
    assert result["value"]["x"] == pytest.approx(50.0) and result["value"]["y"] == pytest.approx(50.0)
    assert result["value"]["width"] == pytest.approx(10.0) and result["value"]["height"] == pytest.approx(20.0)
    assert (result["original_width"], result["original_height"]) == (320, 240)


def test_coco_output_matches_cvat_coco_shape(ingested_clip, stub_detector):
    coco = preannotate.preannotate(str(ingested_clip), None, {"backend": "stub"}, max_per_frame=1)
    ann = coco["annotations"][0]
    assert ann["segmentation"] == [] and ann["attributes"] == {"occluded": False}
    assert ann["bbox"] == pytest.approx([160.0, 120.0, 32.0, 48.0])  # 0.5..0.6 x 320, 0.5..0.7 x 240


def test_output_dir_required_for_non_clip_sources(tmp_path, stub_detector):
    images_dir = tmp_path / "frames"
    images_dir.mkdir()
    with pytest.raises(ValueError, match="--output-dir is required"):
        preannotate.preannotate(str(images_dir), None, {"backend": "stub"})


def test_tiling_flags_set_drone_config_and_are_rejected_for_other_backends():
    args = argparse_namespace(
        detector="drone", preset="default", weights="w.pt", confidence_threshold=None,
        tile_rows=3, tile_cols=3, tile_overlap=None, input_size=None,
    )  # fmt: skip
    config = preannotate.build_detector_config(args)
    assert (config["tile_rows"], config["tile_cols"]) == (3, 3) and "tile_overlap" not in config

    args.detector, args.weights = "motion", None
    with pytest.raises(ValueError, match="only applies to --detector drone"):
        preannotate.build_detector_config(args)
