#!/usr/bin/env python3
"""Anchor-coverage check: can the detector's anchors match these drones at all?

This is docs/DECISIONS.md #23's measurement, made reusable so that our
own footage is measured exactly the way DUT Anti-UAV's train split was
(and the numbers stay comparable). For each ground-truth drone box,
scaled into the model's square input the same way the model's own
transform scales it (each axis independently, to `input_size`), take the
best IoU with any anchor. SSD's matcher makes an anchor a normal positive
only at IoU >= 0.5 (torchvision's SSD `iou_thresh`), so a box whose best
IoU is below 0.5 is "not matchable": training never gives it a usable
target, whatever the epochs or learning rate.

Anchors come from the real model's own anchor generator
(`detector.train.build_single_class_model(..., pretrained=False)` — the
layout does not depend on weights), on the feature maps the model
actually produces at `input_size`.

Tiled inference (detector/tiling.py) can be simulated: a box then also
counts as seen through every tile that fully contains it, scaled from
that tile into the input (plus the full frame, if the detector includes
it), and its best IoU is the best over those views.

Size buckets are COCO's, by box area in the *original* image
(eval/metrics.py's COCO_AREA_RANGES), as in #23.

Usage (reproduces #23's stock-320 row on DUT train):
    python -m eval.anchor_coverage --dataset-dir data/dut-anti-uav/converted/train
    python -m eval.anchor_coverage --dataset-dir ... --input-size 640
    python -m eval.anchor_coverage --dataset-dir ... --tile-rows 3 --tile-cols 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import COCO_AREA_RANGES  # noqa: E402

MATCH_IOU = 0.5  # torchvision SSD's matcher threshold for a normal positive anchor
_IOU_CHUNK = 2048  # rows per box_iou call: bounds memory at 640 (12,828 anchors)


@lru_cache(maxsize=None)
def anchor_boxes(input_size: int = 320):
    """(N, 4) x1y1x2y2 anchors, in input-pixel coordinates, of the
    single-class SSDLite model at `input_size`."""
    import torch

    from detector.train import build_single_class_model

    model = build_single_class_model(input_size, pretrained=False).eval()
    images, _ = model.transform([torch.zeros(3, input_size, input_size)])
    with torch.no_grad():
        features = list(model.backbone(images.tensors).values())
    return model.anchor_generator(images, features)[0]


def size_bucket(area: float) -> str:
    for name, (lo, hi) in COCO_AREA_RANGES.items():
        if lo <= area < hi:
            return name
    return "small"  # negative area can't happen for a valid box; keep total counts honest


def _crops_for_box(box, width, height, tiles, include_full_frame):
    x1, y1, x2, y2 = box
    crops = [(0, 0, width, height)] if include_full_frame or not tiles else []
    crops += [t for t in tiles if t[0] <= x1 and t[1] <= y1 and x2 <= t[2] and y2 <= t[3]]
    if not crops:
        # Straddles every tile seam and no full frame: the detector only
        # ever sees it cut, so use the tile holding most of it, clipped.
        def overlap(t):
            return max(0.0, min(x2, t[2]) - max(x1, t[0])) * max(0.0, min(y2, t[3]) - max(y1, t[1]))

        crops = [max(tiles, key=overlap)]
    return crops


def best_anchor_ious(
    boxes: Iterable[tuple[float, float, float, float]],
    image_sizes: Iterable[tuple[int, int]],
    input_size: int = 320,
    tile_rows: int = 1,
    tile_cols: int = 1,
    tile_overlap: float = 0.2,
    include_full_frame: bool = True,
) -> list[float]:
    """Best anchor IoU for each box (x1, y1, x2, y2, original-image
    pixels), given each box's own image (width, height)."""
    import torch
    from torchvision.ops import box_iou

    from detector.tiling import tile_grid

    anchors = anchor_boxes(input_size)
    tiled = tile_rows * tile_cols > 1
    rows, owners = [], []
    for index, (box, (width, height)) in enumerate(zip(boxes, image_sizes)):
        tiles = tile_grid(width, height, tile_rows, tile_cols, tile_overlap) if tiled else []
        for cx1, cy1, cx2, cy2 in _crops_for_box(box, width, height, tiles, include_full_frame):
            sx, sy = input_size / (cx2 - cx1), input_size / (cy2 - cy1)
            bx1, by1 = max(box[0], cx1), max(box[1], cy1)
            bx2, by2 = min(box[2], cx2), min(box[3], cy2)
            rows.append([(bx1 - cx1) * sx, (by1 - cy1) * sy, (bx2 - cx1) * sx, (by2 - cy1) * sy])
            owners.append(index)

    if not rows:
        return []
    scaled = torch.tensor(rows, dtype=anchors.dtype)
    row_best = torch.cat([box_iou(chunk, anchors).max(dim=1).values for chunk in scaled.split(_IOU_CHUNK)])
    best = torch.zeros(max(owners) + 1, dtype=anchors.dtype)
    best.scatter_reduce_(0, torch.tensor(owners), row_best, reduce="amax")
    return [float(v) for v in best]


def summarize(best_ious: list[float], areas: list[float]) -> dict[str, dict]:
    """Per COCO size bucket (+ "all"): box count, how many are matchable
    (best IoU >= MATCH_IOU), that share, and the median best IoU."""
    groups: dict[str, list[float]] = {name: [] for name in COCO_AREA_RANGES}
    groups["all"] = []
    for value, area in zip(best_ious, areas):
        groups[size_bucket(area)].append(value)
        groups["all"].append(value)
    return {
        name: {
            "count": len(values),
            "matchable": sum(v >= MATCH_IOU for v in values),
            "share_matchable": (sum(v >= MATCH_IOU for v in values) / len(values)) if values else None,
            "median_best_iou": statistics.median(values) if values else None,
        }
        for name, values in groups.items()
    }


def check_samples(samples, target_class: str = "drone", **layout) -> dict[str, dict]:
    """Run the check over `detector.datasets.loader.ImageSample`s' boxes of
    `target_class`. `layout` is passed to `best_anchor_ious`."""
    boxes, sizes, areas = [], [], []
    for sample in samples:
        for b in sample.boxes:
            if b.category != target_class or b.x2 <= b.x1 or b.y2 <= b.y1:
                continue
            boxes.append((b.x1, b.y1, b.x2, b.y2))
            sizes.append((sample.width, sample.height))
            areas.append((b.x2 - b.x1) * (b.y2 - b.y1))
    return summarize(best_anchor_ious(boxes, sizes, **layout), areas)


def format_cell(stats: dict) -> str:
    """#23's table cell format: "24.2% (0.394)" = share matchable (median best IoU)."""
    if not stats["count"]:
        return "n/a (0 boxes)"
    return f"{100 * stats['share_matchable']:.1f}% ({stats['median_best_iou']:.3f})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Unified-layout dir (images/ + annotations.json)")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--tile-rows", type=int, default=1)
    parser.add_argument("--tile-cols", type=int, default=1)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--no-full-frame", action="store_true", help="Tiled: don't also count the full-frame view")
    args = parser.parse_args()

    from detector.datasets.loader import load_manifest_dataset

    summary = check_samples(
        load_manifest_dataset(args.dataset_dir),
        input_size=args.input_size,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        tile_overlap=args.tile_overlap,
        include_full_frame=not args.no_full_frame,
    )
    print(f"Anchors: {len(anchor_boxes(args.input_size))} at input {args.input_size}, tiles {args.tile_rows}x{args.tile_cols}")
    print("| Bucket | Boxes | Share best IoU >= 0.5 (median best IoU) |")
    print("|---|---|---|")
    for name, stats in summary.items():
        print(f"| {name} | {stats['count']} | {format_cell(stats)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
