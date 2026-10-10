"""Tests for tools/coverage_report.py on generated clips."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import coverage_report  # noqa: E402
import import_reviewed  # noqa: E402
import ingest_footage  # noqa: E402
from tests.conftest import POSITIVE_CLIP_META, synthetic_clip_box, write_synthetic_clip  # noqa: E402

POSITIVE_KEY = ("fpv-quad", "close-lt50m", "daylight-clear", "clean-sky", "positive")
NEGATIVE_META = {
    "drone_type": "none",
    "distance_band": "close-lt50m",
    "lighting": "daylight-overcast",
    "background": "water",
    "is_hard_negative": True,
    "negative_subject": "bird",
}


def _review(clip, tmp_path):
    """Import a CVAT-style export boxing the drone on every frame."""
    frames = json.loads((clip / "frames.json").read_text())["images"]
    annotations = []
    for f in frames:
        x1, y1, x2, y2 = synthetic_clip_box(f["source_frame"])
        annotations.append({"id": f["id"], "image_id": f["id"], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1]})
    export = tmp_path / f"{clip.name}_export.json"
    export.write_text(json.dumps({"categories": [{"id": 1, "name": "drone"}], "images": frames, "annotations": annotations}))
    import_reviewed.import_reviewed(clip, export)


@pytest.fixture
def footage_root(tmp_path):
    """Three clips: a reviewed positive, an unreviewed positive in the same
    cell, and a hard negative."""
    video = write_synthetic_clip(tmp_path / "C0001.mp4")
    root = tmp_path / "clips"
    reviewed = ingest_footage.ingest(video, dict(POSITIVE_CLIP_META), root, clip_id="a-reviewed")
    _review(reviewed, tmp_path)
    ingest_footage.ingest(video, dict(POSITIVE_CLIP_META), root, clip_id="b-unreviewed")
    ingest_footage.ingest(video, dict(NEGATIVE_META), root, clip_id="c-gulls")
    return root


def test_counts_combinations_and_label_status(footage_root):
    report = coverage_report.build_report(footage_root, anchor_check=False)

    assert (report["num_clips"], report["num_positive_clips"], report["num_hard_negative_clips"]) == (3, 2, 1)
    assert report["num_frames"] == 30
    assert report["label_status"] == {"unlabeled": 2, "partially_reviewed": 0, "reviewed": 1}
    positive = report["combos"][POSITIVE_KEY]
    assert (positive["clips"], positive["frames"], positive["reviewed_frames"], positive["drone_boxes"]) == (2, 20, 10, 10)
    negative = report["combos"][("none", "close-lt50m", "daylight-overcast", "water", "hard-negative")]
    assert negative["clips"] == 1 and negative["anchor_check"] == "n/a (hard negative)"


def test_lists_what_is_still_missing(footage_root):
    missing = coverage_report.build_report(footage_root, anchor_check=False)["missing"]

    assert "fpv-quad" not in missing["drone_type"] and "none" not in missing["drone_type"]
    assert missing["distance_band"] == ["medium-50-200m", "far-200-500m", "very-far-gt500m"]
    assert "daylight-clear" not in missing["lighting"] and "daylight-overcast" in missing["lighting"]  # negatives don't count
    assert "bird" not in missing["negative_subject"] and "aircraft" in missing["negative_subject"]


def test_grids_count_positive_frames_only(footage_root):
    grid = coverage_report.build_report(footage_root, anchor_check=False)["grids"]["distance_band x lighting"]
    assert grid["frames"][("close-lt50m", "daylight-clear")] == 20
    assert ("close-lt50m", "daylight-overcast") not in grid["frames"]


def test_anchor_check_is_the_same_measurement_as_eval_anchor_coverage(footage_root):
    pytest.importorskip("torch")
    from eval.anchor_coverage import best_anchor_ious, format_cell, summarize

    report = coverage_report.build_report(footage_root)

    boxes = [synthetic_clip_box(i) for i in range(0, 30, 3)]
    direct = summarize(best_anchor_ious(boxes, [(320, 240)] * len(boxes)), [30.0 * 20.0] * len(boxes))
    assert report["anchor_by_size"] == direct
    # Same cell also holds an unreviewed clip, and says so.
    assert report["combos"][POSITIVE_KEY]["anchor_check"] == format_cell(direct["all"]) + " (+unreviewed clips)"
    assert report["anchor_by_distance"]["close-lt50m"]["drone_boxes"] == 10


def test_never_measures_pre_labels(footage_root):
    pytest.importorskip("torch")
    clip = footage_root / "b-unreviewed"
    (clip / "prelabels").mkdir()
    (clip / "prelabels" / "coco_predictions.json").write_text("{}")
    only_unreviewed = footage_root.parent / "solo"
    only_unreviewed.mkdir()
    clip.rename(only_unreviewed / clip.name)

    report = coverage_report.build_report(only_unreviewed)
    assert report["combos"][POSITIVE_KEY]["anchor_check"] == "unreviewed"
    assert report["anchor_by_size"]["all"]["count"] == 0


def test_markdown_states_reference_and_gaps(footage_root):
    text = coverage_report.render_markdown(coverage_report.build_report(footage_root, anchor_check=False))
    assert coverage_report.DUT_TRAIN_REFERENCE in text
    assert "very-far-gt500m" in text and "skipped (--no-anchor-check)" in text


def test_cli_writes_full_cross_product_csv(footage_root, tmp_path):
    out = tmp_path / "coverage"
    assert coverage_report.main(["--footage-root", str(footage_root), "--output-dir", str(out), "--no-anchor-check"]) == 0
    assert {p.name for p in out.iterdir()} == {"coverage.md", "coverage.json", "coverage_matrix.csv"}

    rows = list(csv.DictReader((out / "coverage_matrix.csv").open()))
    assert len(rows) == 6 * 4 * 7 * 6 + 4 * 7 * 6  # positive drone types + hard negatives, all cells
    captured = [r for r in rows if r["clips"] != "0"]
    assert len(captured) == 2
    assert sum(r["anchor_check"] == "not captured" for r in rows) == len(rows) - 2
    assert json.loads((out / "coverage.json").read_text())["num_clips"] == 3


def test_cli_fails_cleanly_on_missing_root(tmp_path):
    assert coverage_report.main(["--footage-root", str(tmp_path / "nope"), "--no-anchor-check"]) == 1
