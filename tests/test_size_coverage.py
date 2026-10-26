"""Tests for eval/size_coverage.py — the apparent-size + anchor-coverage
breakdown behind docs/fas0-feasibility.md. The real-data numbers need the
gitignored VisioDECT copy; these pin down the mechanics."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from eval import size_coverage  # noqa: E402


def _dataset(tmp_path):
    images = [
        {"id": 1, "file_name": "a.jpg", "width": 1920, "height": 1080, "scenario": "Sunny"},
        {"id": 2, "file_name": "b.jpg", "width": 1920, "height": 1080, "scenario": "Cloudy"},
    ]
    annotations = [
        {"image_id": 1, "bbox": [100, 100, 40, 30], "category": "drone"},  # 2.4% apparent size: unmatchable at stock 320
        {"image_id": 2, "bbox": [500, 300, 600, 450], "category": "drone"},  # ~1/3 of frame: matchable
        {"image_id": 2, "bbox": [0, 0, 50, 50], "category": "bird"},
        {"image_id": 2, "bbox": [0, 0, 0, 5], "category": "drone"},  # degenerate
    ]
    (tmp_path / "annotations.json").write_text(json.dumps({"images": images, "annotations": annotations}))
    return tmp_path


def test_load_boxes_keeps_valid_drones_with_group_and_apparent_size(tmp_path):
    boxes = size_coverage.load_boxes(_dataset(tmp_path), group_by="scenario")
    assert [b["group"] for b in boxes] == ["Sunny", "Cloudy"]
    assert boxes[0]["frac"] == pytest.approx(((40 / 1920) * (30 / 1080)) ** 0.5)


def test_build_report_breaks_down_by_group_and_size(tmp_path):
    layouts = {"320": dict(input_size=320)}
    report = size_coverage.build_report(size_coverage.load_boxes(_dataset(tmp_path), "scenario"), layouts)
    assert set(report["coverage"]) == {"all", "Cloudy", "Sunny"}
    assert report["coverage"]["Sunny"]["320"]["all"]["matchable"] == 0
    assert report["coverage"]["Cloudy"]["320"]["all"]["matchable"] == 1
    assert report["coverage_by_size_fraction"]["2-3%"]["320"] == {"count": 1, "matchable": 0}
    assert "| Cloudy | all | 1 |" in size_coverage.render_markdown(report)


def test_sweep_finds_stock_320_threshold_between_5_and_30_percent():
    result = size_coverage.sweep([0.05, 0.30], samples=20, layouts={"320": dict(input_size=320)})
    assert result["5%"]["320"] == 0.0 and result["30%"]["320"] == 1.0
