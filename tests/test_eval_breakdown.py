"""Tests for eval/breakdown.py: AP/recall by apparent size from a harness
predictions dump, via eval/metrics.py on frame-normalized coordinates."""

import gzip
import json

import pytest

from eval import breakdown


def _dump(path, rows):
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_breakdown_splits_by_apparent_size_and_matches_pixel_iou(tmp_path):
    # 1000x500 frame. Drone A: 25x12.5px -> apparent size 2.5%, found exactly.
    # Drone B: 300x150px -> 30%, missed (only a far-off box).
    rows = [
        {"frame_id": "a", "width": 1000, "height": 500, "gt": [[100, 100, 125, 112.5]], "pred": [[100, 100, 125, 112.5, 0.9]]},
        {"frame_id": "b", "width": 1000, "height": 500, "gt": [[500, 200, 800, 350]], "pred": [[0, 0, 300, 150, 0.8]]},
    ]
    frames = breakdown.load_dump(_dump(tmp_path / "p.jsonl.gz", rows))
    result = breakdown.summarize(frames, threshold=0.5)

    assert result["all"]["count"] == 2 and result["all"]["recall_any_score"] == 0.5
    assert result["2-3%"] == {"count": 1, "ap50": 1.0, "recall_any_score": 1.0, "recall_at_threshold": 1.0}
    assert result["30-100%"]["recall_any_score"] == 0.0 and result["30-100%"]["ap50"] == 0.0


def test_threshold_cuts_operating_recall_only(tmp_path):
    rows = [{"frame_id": "a", "width": 100, "height": 100, "gt": [[10, 10, 20, 20]], "pred": [[10, 10, 20, 20, 0.2]]}]
    result = breakdown.summarize(breakdown.load_dump(_dump(tmp_path / "p.jsonl.gz", rows)), threshold=0.35)
    assert result["all"]["recall_any_score"] == 1.0 and result["all"]["recall_at_threshold"] == 0.0
    assert result["all"]["ap50"] == pytest.approx(1.0)
