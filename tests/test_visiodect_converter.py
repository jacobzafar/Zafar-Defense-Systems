"""Tests for detector/datasets/visiodect.py against a tiny synthetic
fixture mimicking VisioDECT's real per-model/per-scenario CSV layout and
its real quirks (a multi-box image, a degenerate box, an orphaned
annotation, a missing label directory, and a non-CSV outlier format) —
not the real ~830MB dataset, which was supplied separately for this pass
(see docs/datasets.md) and is gitignored, never committed.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.datasets.loader import load_manifest_dataset
from detector.datasets.visiodect import VisioDectFormatError, convert_dataset


def _write_image(path: Path, width: int, height: int) -> None:
    np = pytest.importorskip("numpy")
    pil_image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width, 3), dtype="uint8")
    pil_image.fromarray(arr).save(path)


def _write_csv(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)


def _build_raw_dir(raw_dir: Path) -> None:
    """Mirrors the real dataset's quirks in miniature, all under one of
    the six real model names (`Anafi-Extended`) plus one left entirely
    empty (`EFT-E410S`), since `convert_dataset` walks a fixed model list:

    - Evening: 3 images. img1 has one normal box; img2 has two boxes
      (real multi-box images exist); img3's only box is degenerate
      (zero height) so it becomes a hard negative; the csv also has one
      row referencing a 4th, nonexistent image (an orphaned annotation).
    - Cloudy: an image exists but there is no label directory at all.
    - Sunny: an image exists but its only label file is `csv.xlsx`
      (mirrors the one real non-CSV outlier — must be reported, not
      ingested).
    - `EFT-E410S`: left completely empty, mirroring the three real model
      directories that exist only as empty structure.
    """
    for i in (1, 2, 3):
        _write_image(raw_dir / "Anafi-Extended" / "images" / "Evening" / f"img{i}.jpg", 200, 100)
    _write_csv(
        raw_dir / "Anafi-Extended" / "labels" / "evening" / "csv.csv",
        [
            ["Anafi_Extended_Evening", 10, 10, 20, 20, "img1.jpg", 200, 100],
            ["Anafi_Extended_Evening", 50, 50, 30, 10, "img2.jpg", 200, 100],
            ["Anafi_Extended_Evening", 60, 60, 5, 5, "img2.jpg", 200, 100],
            ["Anafi_Extended_Evening", 5, 5, 1, 0, "img3.jpg", 200, 100],
            ["Anafi_Extended_Evening", 1, 1, 5, 5, "does_not_exist.jpg", 200, 100],
        ],
    )

    _write_image(raw_dir / "Anafi-Extended" / "images" / "Cloudy" / "img1.jpg", 200, 100)
    # no labels/cloudy/ directory at all

    _write_image(raw_dir / "Anafi-Extended" / "images" / "Sunny" / "img1.jpg", 200, 100)
    (raw_dir / "Anafi-Extended" / "labels" / "sunny").mkdir(parents=True)
    (raw_dir / "Anafi-Extended" / "labels" / "sunny" / "csv.xlsx").write_bytes(b"not a real xlsx, just a marker")

    # EFT-E410S: nothing at all — mirrors the real empty-skeleton models.


def test_convert_dataset_produces_expected_stats(tmp_path):
    stats = convert_dataset(_raw(tmp_path), tmp_path / "converted")

    assert stats.num_images == 5  # 3 evening + 1 cloudy + 1 sunny
    assert stats.num_drone_instances == 3  # img1 (1 box) + img2 (2 boxes); img3's box is degenerate
    assert stats.num_degenerate_boxes_skipped == 1
    assert stats.num_orphaned_annotations == 1
    assert stats.num_hard_negative_images == 3  # img3, cloudy/img1, sunny/img1


def test_empty_model_directories_are_walked_without_erroring(tmp_path):
    stats = convert_dataset(_raw(tmp_path), tmp_path / "converted")

    eft_stats = [s for s in stats.per_model_scenario if s.model == "EFT-E410S"]
    assert len(eft_stats) == 3  # one entry per scenario, even though nothing exists on disk
    assert all(s.num_images_on_disk == 0 for s in eft_stats)
    assert all(s.annotation_format_available == "none" for s in eft_stats)


def test_missing_label_directory_is_reported_as_format_none(tmp_path):
    stats = convert_dataset(_raw(tmp_path), tmp_path / "converted")

    cloudy_stats = next(s for s in stats.per_model_scenario if s.model == "Anafi-Extended" and s.scenario == "Cloudy")
    assert cloudy_stats.annotation_format_available == "none"
    assert cloudy_stats.num_images_on_disk == 1
    assert cloudy_stats.num_drone_instances == 0


def test_xlsx_annotation_format_is_reported_but_not_ingested(tmp_path):
    stats = convert_dataset(_raw(tmp_path), tmp_path / "converted")

    sunny_stats = next(s for s in stats.per_model_scenario if s.model == "Anafi-Extended" and s.scenario == "Sunny")
    assert sunny_stats.annotation_format_available == "xlsx (not ingested)"
    assert sunny_stats.num_images_on_disk == 1
    assert sunny_stats.num_drone_instances == 0


def test_converted_output_parses_via_the_unified_loader(tmp_path):
    output_dir = tmp_path / "converted"
    convert_dataset(_raw(tmp_path), output_dir)

    samples = load_manifest_dataset(output_dir)
    assert len(samples) == 5
    assert all(s.image_path.exists() for s in samples)

    drone_box_count = sum(1 for s in samples for b in s.boxes if b.category == "drone")
    assert drone_box_count == 3

    hard_negatives = [s for s in samples if s.is_hard_negative]
    assert len(hard_negatives) == 3


def test_box_size_distribution_buckets_correctly(tmp_path):
    stats = convert_dataset(_raw(tmp_path), tmp_path / "converted")

    # img1: 20x20=400px² -> "256-1024"; img2: 30x10=300px² -> "256-1024", 5x5=25px² -> "<256"
    assert stats.box_size_distribution["<16x16 (<256px²)"] == 1
    assert stats.box_size_distribution["16x16-32x32 (256-1024px²)"] == 2


def test_conversion_stats_json_is_written(tmp_path):
    output_dir = tmp_path / "converted"
    stats = convert_dataset(_raw(tmp_path), output_dir)

    on_disk = json.loads((output_dir / "conversion_stats.json").read_text())
    assert on_disk == stats.as_dict()


def test_missing_raw_dir_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_dataset(tmp_path / "does-not-exist", tmp_path / "converted")


def test_malformed_csv_raises_visiodect_format_error(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "Anafi-Extended" / "images" / "Evening" / "img1.jpg", 50, 50)
    _write_csv(raw_dir / "Anafi-Extended" / "labels" / "evening" / "csv.csv", [["not", "enough", "fields"]])

    with pytest.raises(VisioDectFormatError):
        convert_dataset(raw_dir, tmp_path / "converted")


def _raw(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    _build_raw_dir(raw_dir)
    return raw_dir
