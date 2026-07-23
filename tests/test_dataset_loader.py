import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.datasets.loader import DatasetFormatError, load_manifest_dataset


def test_load_synthetic_dataset_returns_expected_samples(synthetic_drone_dataset):
    samples = load_manifest_dataset(synthetic_drone_dataset)
    assert len(samples) == 6

    drone_samples = [s for s in samples if not s.is_hard_negative]
    hard_negative_samples = [s for s in samples if s.is_hard_negative]
    assert len(drone_samples) == 3
    assert len(hard_negative_samples) == 3

    for sample in samples:
        assert sample.image_path.exists()
        assert sample.width == 64
        assert sample.height == 64


def test_missing_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest_dataset(tmp_path / "does-not-exist")


def test_missing_annotations_file_raises_file_not_found(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="annotations.json"):
        load_manifest_dataset(empty_dir)


def test_malformed_json_raises_dataset_format_error(tmp_path):
    dataset_dir = tmp_path / "bad"
    dataset_dir.mkdir()
    (dataset_dir / "annotations.json").write_text("not valid json{")
    with pytest.raises(DatasetFormatError):
        load_manifest_dataset(dataset_dir)


def test_missing_top_level_keys_raises_dataset_format_error(tmp_path):
    dataset_dir = tmp_path / "bad"
    dataset_dir.mkdir()
    (dataset_dir / "annotations.json").write_text(json.dumps({"images": []}))
    with pytest.raises(DatasetFormatError, match="images.*annotations"):
        load_manifest_dataset(dataset_dir)


def test_annotation_referencing_unknown_image_id_raises(tmp_path):
    dataset_dir = tmp_path / "bad"
    dataset_dir.mkdir()
    (dataset_dir / "annotations.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
                "annotations": [{"image_id": 999, "bbox": [0, 0, 1, 1], "category": "drone"}],
            }
        )
    )
    with pytest.raises(DatasetFormatError, match="unknown image_id"):
        load_manifest_dataset(dataset_dir)
