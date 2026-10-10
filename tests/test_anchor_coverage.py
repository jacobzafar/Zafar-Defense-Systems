"""Tests for eval/anchor_coverage.py — docs/DECISIONS.md #23's measurement.

The real-data check (reproducing #23's DUT-train table exactly) needs the
gitignored dataset; these pin down the mechanics on known geometry.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from detector.datasets.loader import BoxAnnotation, ImageSample  # noqa: E402
from eval import anchor_coverage  # noqa: E402


def test_anchor_layout_matches_decisions_23():
    small_input, large_input = anchor_coverage.anchor_boxes(320), anchor_coverage.anchor_boxes(640)
    assert len(small_input) == 3234 and len(large_input) == 12828  # #23's "Anchors" column
    sides = ((small_input[:, 2] - small_input[:, 0]) * (small_input[:, 3] - small_input[:, 1])).sqrt()
    assert float(sides.min()) == pytest.approx(64.0)  # smallest anchor 0.2 x 320


def test_anchors_do_not_depend_on_pretrained_weights():
    import torch

    from detector.train import build_single_class_model

    model = build_single_class_model(320).eval()  # pretrained, as DroneDetector/train use it
    images, _ = model.transform([torch.zeros(3, 320, 320)])
    with torch.no_grad():
        features = list(model.backbone(images.tensors).values())
    assert torch.equal(model.anchor_generator(images, features)[0], anchor_coverage.anchor_boxes(320))


def test_box_equal_to_an_anchor_scores_one_after_scaling_into_the_input():
    anchors = anchor_coverage.anchor_boxes(320)
    inside = ((anchors >= 0) & (anchors <= 320)).all(dim=1)  # GT boxes always lie inside the image
    anchor = [float(v) for v in anchors[inside][100]]
    # Same box in a 640x480 image: the model's resize maps it back onto the anchor.
    scaled = (anchor[0] * 2, anchor[1] * 1.5, anchor[2] * 2, anchor[3] * 1.5)
    [best] = anchor_coverage.best_anchor_ious([scaled], [(640, 480)])
    assert best == pytest.approx(1.0, abs=1e-5)


def test_small_drone_is_unmatchable_and_tiling_rescues_a_medium_one():
    frame = (1920, 1080)
    small = (900.0, 500.0, 920.0, 510.0)  # 20x10px: #23's "small" case
    medium = (300.0, 300.0, 396.0, 396.0)  # 96px square, well inside one 3x3 tile

    small_iou, medium_iou = anchor_coverage.best_anchor_ious([small, medium], [frame, frame])
    assert small_iou < anchor_coverage.MATCH_IOU and medium_iou < anchor_coverage.MATCH_IOU

    [tiled_small, tiled_medium] = anchor_coverage.best_anchor_ious([small, medium], [frame, frame], tile_rows=3, tile_cols=3)
    assert tiled_small < anchor_coverage.MATCH_IOU
    assert tiled_medium >= anchor_coverage.MATCH_IOU


def test_summarize_buckets_by_original_area_and_formats_like_decisions_23():
    summary = anchor_coverage.summarize([0.01, 0.2, 0.6, 0.8], [100.0, 2000.0, 20000.0, 30000.0])
    assert summary["small"] == {"count": 1, "matchable": 0, "share_matchable": 0.0, "median_best_iou": 0.01}
    assert summary["large"]["matchable"] == 2 and summary["all"]["count"] == 4
    assert anchor_coverage.format_cell(summary["large"]) == "100.0% (0.700)"
    assert anchor_coverage.format_cell(summary["medium"]) == "0.0% (0.200)"
    assert anchor_coverage.format_cell({"count": 0}) == "n/a (0 boxes)"


def test_check_samples_counts_only_drone_boxes():
    sample = ImageSample(
        image_path="unused.jpg",
        width=320,
        height=320,
        boxes=[
            BoxAnnotation(0, 0, 160, 160, "drone"),
            BoxAnnotation(0, 0, 160, 160, "bird"),
            BoxAnnotation(5, 5, 5, 9, "drone"),  # degenerate: skipped
        ],
    )
    assert anchor_coverage.check_samples([sample])["all"]["count"] == 1
