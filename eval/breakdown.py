#!/usr/bin/env python3
"""Detection results by group and by apparent size, from harness prediction dumps.

The metric card only splits AP by COCO pixel-area buckets. For a
feasibility question (docs/fas0-feasibility.md) the useful axis is
*apparent size*, sqrt(box_w/frame_w * box_h/frame_h) — the same measure
as eval/size_coverage.py — because the model resizes each frame axis to
its square input independently, so that, not pixels, sets what it sees.

Reads one or more `output_predictions_path` dumps from eval/harness.py
(each optionally labelled, e.g. one per scenario) and reports, per label
and per apparent-size bin:

- AP@0.5: eval/metrics.compute_ap, unchanged, on frame-normalized
  coordinates (x / width, y / height). IoU is invariant to per-axis
  scaling, and a normalized box's area is its apparent size squared, so an
  apparent-size bin [lo, hi) is exactly compute_ap's area_range
  [lo², hi²) with COCO's ignore rules;
- recall@0.5 over every prediction (any score) — whether the detector
  puts a correct box on the drone at all — and at the operating threshold.

Usage:
    python -m eval.breakdown --dump Sunny=data/eval_predictions/fpv_sunny.jsonl.gz \\
        --dump Cloudy=... --dump Evening=... --threshold 0.35
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import FrameEvalData, PredBox, compute_ap, compute_recall_at_threshold  # noqa: E402
from eval.schema import EvalBox  # noqa: E402
from eval.size_coverage import SIZE_FRACTION_BINS, size_bin  # noqa: E402


def load_dump(path: str | Path) -> list[FrameEvalData]:
    """Frames from a harness predictions dump, in frame-normalized coordinates."""
    frames = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            w, h = row["width"], row["height"]
            frames.append(
                FrameEvalData(
                    frame_id=row["frame_id"],
                    gt_boxes=[EvalBox(x1 / w, y1 / h, x2 / w, y2 / h, "drone") for x1, y1, x2, y2 in row["gt"]],
                    pred_boxes=[PredBox(x1 / w, y1 / h, x2 / w, y2 / h, c) for x1, y1, x2, y2, c in row["pred"]],
                )
            )
    return frames


def _apparent(box) -> float:
    return max(0.0, (box.x2 - box.x1) * (box.y2 - box.y1)) ** 0.5


def summarize(frames: list[FrameEvalData], threshold: float | None, bins=SIZE_FRACTION_BINS) -> dict[str, dict]:
    """Per apparent-size bin (+ "all"): GT count, AP@0.5, recall over all
    predictions, recall at `threshold`."""

    def restricted(lo, hi):
        # Recall in a bin: count only GT in the bin (others removed, so they
        # can't be credited). Predictions stay, so matching is unchanged for in-bin GT.
        return [
            FrameEvalData(f.frame_id, [g for g in f.gt_boxes if lo <= _apparent(g) < hi], f.pred_boxes) for f in frames
        ]

    def row(lo, hi):
        sub = restricted(lo, hi)
        n = sum(len(f.gt_boxes) for f in sub)
        if not n:
            return None
        return {
            "count": n,
            "ap50": compute_ap(frames, 0.5, area_range=(lo * lo, hi * hi)),
            "recall_any_score": compute_recall_at_threshold(sub, None)["recall"],
            "recall_at_threshold": compute_recall_at_threshold(sub, threshold)["recall"] if threshold is not None else None,
        }

    out = {"all": row(0.0, float("inf"))}
    for lo, hi in zip(bins, bins[1:]):
        r = row(lo, hi)
        if r:
            out[size_bin(lo)] = r
    return out


def render(results: dict[str, dict[str, dict]], threshold: float | None) -> str:
    lines = [
        f"| Group | Apparent size | Drones | AP@0.5 | Recall, any score | Recall @ {threshold} |",
        "|---|---|---|---|---|---|",
    ]
    for label, bins in results.items():
        for name, r in bins.items():
            at = f"{r['recall_at_threshold']:.3f}" if r["recall_at_threshold"] is not None else "–"
            lines.append(f"| {label} | {name} | {r['count']} | {r['ap50']:.4f} | {r['recall_any_score']:.3f} | {at} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", action="append", required=True, help="[LABEL=]path to a harness predictions dump")
    parser.add_argument("--threshold", type=float, default=0.35, help="Operating confidence threshold")
    parser.add_argument("--output-json", help="Also write the results as JSON here")
    args = parser.parse_args()

    results: dict[str, dict] = {}
    all_frames: list[FrameEvalData] = []
    for spec in args.dump:
        label, _, path = spec.rpartition("=")
        frames = load_dump(path)
        all_frames += frames
        results[label or Path(path).name] = summarize(frames, args.threshold)
    if len(args.dump) > 1:
        results = {"all": summarize(all_frames, args.threshold), **results}
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(render(results, args.threshold), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
