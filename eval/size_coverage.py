#!/usr/bin/env python3
"""Apparent-size distribution + anchor coverage, broken down by a per-image key.

Wraps eval/anchor_coverage.py (docs/DECISIONS.md #23's measurement,
unchanged) with what a feasibility question needs on top of #23's
per-COCO-bucket table:

- how big the drones look: box width/height in pixels and as a fraction
  of frame width/height (percentiles);
- the share matchable (best anchor IoU >= 0.5) per layout, overall, per
  COCO size bucket, and per value of a per-image key from annotations.json
  (e.g. VisioDECT's "scenario", written by detector/datasets/visiodect.py);
- the share matchable per bin of apparent size, i.e. how large a drone
  must look before the anchors can fit it at all. Apparent size here is
  sqrt(box_w/frame_w * box_h/frame_h): the model resizes each axis to the
  input independently, so both fractions matter, and portrait/landscape
  frames are only comparable this way;
- `--sweep`: the same threshold measured on synthetic boxes (fixed aspect,
  random positions in a 1920x1080 frame), independent of any dataset.

Layouts: stock anchors at 320 and 640, 3x3 tiling (20% overlap, plus the
full frame, as dut_v1's opt-in tiled mode #21) at 320 and 640, and #23's
recommended option B (simulated; not built).

Usage (docs/fas0-feasibility.md):
    python -m eval.size_coverage --dataset-dir data/visiodect/converted-fpv --group-by scenario
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.anchor_coverage import (  # noqa: E402
    MATCH_IOU,
    OPTION_B_640,
    best_anchor_ious,
    format_cell,
    simulated_anchor_boxes,
    summarize,
)


def _option_b(**tiling):
    return lambda: dict(input_size=640, anchors=simulated_anchor_boxes(**OPTION_B_640), **tiling)


def _geometric_scales():
    from detector.ssdlite_stride8 import GEOMETRIC_SCALES

    return GEOMETRIC_SCALES


# name -> layout kwargs for best_anchor_ious (callables are built lazily:
# they need torch).
LAYOUTS = {
    "320": dict(input_size=320),
    "640": dict(input_size=640),
    "320, 3x3 tiles + full": dict(input_size=320, tile_rows=3, tile_cols=3),
    "640, 3x3 tiles + full": dict(input_size=640, tile_rows=3, tile_cols=3),
    "#23 option B (640 + stride-8, 16px), simulated": _option_b(),
    # #26: the built model (detector/ssdlite_stride8.py), its real anchors.
    "stride-8 built (640, geometric scales)": lambda: dict(
        input_size=640, architecture="ssdlite_stride8", anchor_scales=_geometric_scales()
    ),
}
# Bins of apparent size = sqrt(frac_w * frac_h), see module docstring.
SIZE_FRACTION_BINS = [0.0, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 1.01]
SWEEP_FRACTIONS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
PERCENTILES = (5, 25, 50, 75, 95)


def load_boxes(dataset_dir: str | Path, group_by: str | None = None, target_class: str = "drone") -> list[dict]:
    """One dict per valid target box: box, image size, area, group value."""
    raw = json.loads((Path(dataset_dir) / "annotations.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in raw["images"]}
    boxes = []
    for ann in raw["annotations"]:
        x, y, w, h = ann["bbox"]
        if ann.get("category") != target_class or w <= 0 or h <= 0:
            continue
        image = images[ann["image_id"]]
        boxes.append(
            {
                "box": (x, y, x + w, y + h),
                "image_size": (image["width"], image["height"]),
                "area": w * h,
                "w": w,
                "h": h,
                "frac_w": w / image["width"],
                "frac_h": h / image["height"],
                "frac": ((w / image["width"]) * (h / image["height"])) ** 0.5,
                "group": str(image.get(group_by, "?")) if group_by else "all",
            }
        )
    return boxes


def percentiles(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [values[0] if values else float("nan")] * len(PERCENTILES)
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return [cuts[p - 1] for p in PERCENTILES]


def size_bin(frac: float) -> str:
    for lo, hi in zip(SIZE_FRACTION_BINS, SIZE_FRACTION_BINS[1:]):
        if lo <= frac < hi:
            return f"{100 * lo:g}-{100 * min(hi, 1.0):g}%"
    return f">={100 * SIZE_FRACTION_BINS[-2]:g}%"


def _kwargs(layout) -> dict:
    return layout() if callable(layout) else layout


def build_report(boxes: list[dict], layouts: dict[str, dict] = LAYOUTS) -> dict:
    groups = sorted({b["group"] for b in boxes})
    ious = {
        name: best_anchor_ious([b["box"] for b in boxes], [b["image_size"] for b in boxes], **_kwargs(layout))
        for name, layout in layouts.items()
    }

    def subset(pred):
        return [i for i, b in enumerate(boxes) if pred(b)]

    report: dict = {"num_boxes": len(boxes), "sizes": {}, "coverage": {}, "coverage_by_size_fraction": {}}
    for group in ["all", *groups] if len(groups) > 1 else ["all"]:
        idx = subset(lambda b: group == "all" or b["group"] == group)
        report["sizes"][group] = {
            "count": len(idx),
            "image_sizes": sorted({f"{w}x{h}" for w, h in (boxes[i]["image_size"] for i in idx)}),
            **{key: percentiles([boxes[i][key] for i in idx]) for key in ("w", "h", "frac_w", "frac_h", "frac")},
        }
        report["coverage"][group] = {
            name: summarize([values[i] for i in idx], [boxes[i]["area"] for i in idx]) for name, values in ious.items()
        }

    bins: dict[str, list[int]] = {}
    for i, b in enumerate(boxes):
        bins.setdefault(size_bin(b["frac"]), []).append(i)
    order = [size_bin(lo) for lo in SIZE_FRACTION_BINS[:-1]]
    for label in [label for label in order if label in bins]:
        idx = bins[label]
        report["coverage_by_size_fraction"][label] = {
            name: {"count": len(idx), "matchable": sum(values[i] >= MATCH_IOU for i in idx)} for name, values in ious.items()
        }
    return report


def sweep(
    fractions: list[float] = SWEEP_FRACTIONS,
    aspect: float = 1.33,
    frame: tuple[int, int] = (1920, 1080),
    samples: int = 200,
    seed: int = 0,
    layouts: dict[str, dict] = LAYOUTS,
) -> dict[str, dict[str, float]]:
    """Share matchable for synthetic boxes of apparent size `f` (as
    sqrt(frac_w * frac_h)) and pixel aspect w/h `aspect`, at `samples`
    seeded random positions in a `frame`-sized image."""
    import random

    rng = random.Random(seed)
    width, height = frame
    out: dict[str, dict[str, float]] = {}
    for f in fractions:
        side = f * (width * height) ** 0.5  # geometric-mean side in pixels
        bw, bh = side * aspect**0.5, side / aspect**0.5
        boxes = []
        for _ in range(samples):
            x, y = rng.uniform(0, width - bw), rng.uniform(0, height - bh)
            boxes.append((x, y, x + bw, y + bh))
        out[f"{100 * f:g}%"] = {
            name: sum(v >= MATCH_IOU for v in best_anchor_ious(boxes, [frame] * samples, **_kwargs(layout))) / samples
            for name, layout in layouts.items()
        }
    return out


def render_sweep(result: dict[str, dict[str, float]], frame=(1920, 1080), aspect: float = 1.33) -> str:
    layouts = list(next(iter(result.values())))
    side = lambda label: float(label.rstrip("%")) / 100 * (frame[0] * frame[1]) ** 0.5  # noqa: E731
    lines = [f"### Synthetic sweep: share matchable vs apparent size ({frame[0]}x{frame[1]} frame, aspect {aspect})", ""]
    lines += ["| Apparent size | Box px (w x h) | " + " | ".join(layouts) + " |", "|---|---|" + "---|" * len(layouts)]
    for label, by_layout in result.items():
        s = side(label)
        cells = " | ".join(f"{100 * by_layout[name]:.0f}%" for name in layouts)
        lines.append(f"| {label} | {s * aspect**0.5:.0f} x {s / aspect**0.5:.0f} | {cells} |")
    return "\n".join(lines) + "\n"


def render_markdown(report: dict) -> str:
    pct = "/".join(f"p{p}" for p in PERCENTILES)
    lines = [f"Boxes: {report['num_boxes']}", "", f"### Apparent size ({pct})", ""]
    lines += [
        "| Group | Boxes | Frames | Width px | Height px | Width / frame width | Height / frame height | Apparent size |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group, s in report["sizes"].items():
        fmt_px = lambda v: " / ".join(f"{x:.0f}" for x in v)  # noqa: E731
        fmt_fr = lambda v: " / ".join(f"{100 * x:.1f}%" for x in v)  # noqa: E731
        lines.append(
            f"| {group} | {s['count']} | {', '.join(s['image_sizes'])} | {fmt_px(s['w'])} | {fmt_px(s['h'])} "
            f"| {fmt_fr(s['frac_w'])} | {fmt_fr(s['frac_h'])} | {fmt_fr(s['frac'])} |"
        )
    layouts = list(next(iter(report["coverage"].values())))
    lines += ["", "### Share matchable, best anchor IoU >= 0.5 (median best IoU)", ""]
    lines += ["| Group | Bucket | Boxes | " + " | ".join(layouts) + " |", "|---|---|---|" + "---|" * len(layouts)]
    for group, by_layout in report["coverage"].items():
        for bucket in ("all", "small", "medium", "large"):
            first = by_layout[layouts[0]][bucket]
            if not first["count"]:
                continue
            cells = " | ".join(format_cell(by_layout[name][bucket]) for name in layouts)
            lines.append(f"| {group} | {bucket} | {first['count']} | {cells} |")
    lines += ["", "### Share matchable by apparent size, sqrt(frac_w * frac_h)", ""]
    lines += ["| Apparent size | Boxes | " + " | ".join(layouts) + " |", "|---|---|" + "---|" * len(layouts)]
    for label, by_layout in report["coverage_by_size_fraction"].items():
        count = by_layout[layouts[0]]["count"]
        cells = " | ".join(f"{100 * by_layout[name]['matchable'] / count:.1f}%" for name in layouts)
        lines.append(f"| {label} | {count} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", help="Unified-layout dir (images/ + annotations.json)")
    parser.add_argument("--group-by", help="Per-image key in annotations.json to break down by, e.g. scenario")
    parser.add_argument("--output-json", help="Also write the full report as JSON here")
    parser.add_argument("--sweep", action="store_true", help="Also run the synthetic size sweep")
    args = parser.parse_args()
    if not args.dataset_dir and not args.sweep:
        parser.error("give --dataset-dir and/or --sweep")
    if args.sweep:
        print(render_sweep(sweep()))
    if not args.dataset_dir:
        return 0

    report = build_report(load_boxes(args.dataset_dir, args.group_by))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
