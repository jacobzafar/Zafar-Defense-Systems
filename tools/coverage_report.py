#!/usr/bin/env python3
"""Coverage matrix for our own footage, with an anchor check per cell.

Reads every clip's clip_meta.json under a footage root (written by
tools/ingest_footage.py) and reports what has been captured across drone
type x distance band x lighting x background x (positive / hard negative),
and, more usefully before a shoot, what has not.

Anchor check: for the drones actually present, the share whose best IoU
with any anchor of the current architecture is >= 0.5 (SSD's matcher
threshold), with the median best IoU in parentheses — the exact
measurement of docs/DECISIONS.md #23, via eval/anchor_coverage.py (which
reproduces #23's DUT-train numbers), so the figures are directly
comparable: DUT train scored small 0.0% (0.007), medium 0.7% (0.023),
large 70.3% (0.676) at input 320.

It runs only on human-reviewed labels (<clip>/annotations.json from
tools/import_reviewed.py), never on pre-labels: the detector mostly finds
drones its anchors already fit, so measuring its own boxes would overstate
coverage. Unreviewed clips show "unreviewed".

Usage:
    python tools/coverage_report.py
    python tools/coverage_report.py --footage-root data/own-fpv/clips --output-dir data/own-fpv/coverage
    python tools/coverage_report.py --input-size 640           # anchor check for the prepared 640 layout
    python tools/coverage_report.py --tile-rows 3 --tile-cols 3  # ... or with tiled inference simulated
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from footage_meta import ANNOTATIONS_FILENAME, DIMENSIONS, NEGATIVE_SUBJECTS, load_clip_dirs  # noqa: E402

DEFAULT_FOOTAGE_ROOT = "data/own-fpv/clips"
DUT_TRAIN_REFERENCE = "DUT train, input 320 (#23): small 0.0% (0.007) · medium 0.7% (0.023) · large 70.3% (0.676)"
_NOT_REVIEWED = "unreviewed"


def _clip_boxes(clip_dir: Path):
    """Reviewed drone boxes of a clip as (boxes, image sizes, areas), or
    None if the clip has no reviewed labels."""
    if not (clip_dir / ANNOTATIONS_FILENAME).exists():
        return None
    from detector.datasets.loader import load_manifest_dataset

    boxes, sizes, areas = [], [], []
    for sample in load_manifest_dataset(clip_dir):
        for b in sample.boxes:
            if b.category == "drone" and b.x2 > b.x1 and b.y2 > b.y1:
                boxes.append((b.x1, b.y1, b.x2, b.y2))
                sizes.append((sample.width, sample.height))
                areas.append((b.x2 - b.x1) * (b.y2 - b.y1))
    return boxes, sizes, areas


def _anchor_cell(best_ious: list[float] | None, areas: list[float], any_unreviewed: bool, is_negative: bool, error: str | None) -> str:
    if is_negative:
        return "n/a (hard negative)"
    if error:
        return error
    if best_ious is None:
        return _NOT_REVIEWED
    from eval.anchor_coverage import format_cell, summarize

    stats = summarize(best_ious, areas)["all"]
    cell = format_cell(stats) if stats["count"] else "0 drone boxes"
    return cell + (" (+unreviewed clips)" if any_unreviewed else "")


def build_report(footage_root: str | Path, anchor_check: bool = True, **layout) -> dict:
    """Everything the printed/written report shows, as plain data."""
    clips = load_clip_dirs(footage_root)

    anchor_error = None
    best_by_clip: dict[str, list[float] | None] = {}
    areas_by_clip: dict[str, list[float]] = {}
    if anchor_check:
        try:
            from eval.anchor_coverage import best_anchor_ious
        except ImportError:
            anchor_error = "unavailable (torch not installed)"
    else:
        anchor_error = "skipped (--no-anchor-check)"

    for clip_dir, meta in clips:
        reviewed = _clip_boxes(clip_dir)
        if reviewed is None:
            best_by_clip[meta["clip_id"]] = None
            areas_by_clip[meta["clip_id"]] = []
            continue
        boxes, sizes, areas = reviewed
        areas_by_clip[meta["clip_id"]] = areas
        best_by_clip[meta["clip_id"]] = best_anchor_ious(boxes, sizes, **layout) if not anchor_error else []

    def group_rows(key_fn):
        groups = defaultdict(list)
        for clip_dir, meta in clips:
            groups[key_fn(meta)].append(meta)
        rows = {}
        for key, metas in groups.items():
            reviewed = [m for m in metas if best_by_clip[m["clip_id"]] is not None]
            best = [v for m in reviewed for v in best_by_clip[m["clip_id"]]] if reviewed else None
            areas = [a for m in reviewed for a in areas_by_clip[m["clip_id"]]]
            is_negative = all(m["is_hard_negative"] for m in metas)
            rows[key] = {
                "clips": len(metas),
                "frames": sum(m["num_frames_extracted"] for m in metas),
                "reviewed_frames": sum(m.get("num_frames_reviewed", 0) for m in reviewed),
                "drone_boxes": len(areas),
                "anchor_check": _anchor_cell(best, areas, len(reviewed) < len(metas), is_negative, anchor_error),
                "best_ious": best or [],
                "areas": areas,
            }
        return rows

    dims = list(DIMENSIONS)
    combos = group_rows(lambda m: tuple(m[d] for d in dims) + ("hard-negative" if m["is_hard_negative"] else "positive",))

    positives = [m for _, m in clips if not m["is_hard_negative"]]
    negatives = [m for _, m in clips if m["is_hard_negative"]]
    missing = {
        dim: [v for v in values if v != "none" and not any(m[dim] == v for m in positives)] for dim, values in DIMENSIONS.items()
    }
    missing["negative_subject"] = [s for s in NEGATIVE_SUBJECTS if not any(m.get("negative_subject") == s for m in negatives)]

    def grid(row_dim, col_dim):
        cells = defaultdict(int)
        for m in positives:
            cells[(m[row_dim], m[col_dim])] += m["num_frames_extracted"]
        return {"rows": [v for v in DIMENSIONS[row_dim] if v != "none"], "cols": DIMENSIONS[col_dim], "frames": cells}

    by_distance = group_rows(lambda m: m["distance_band"] if not m["is_hard_negative"] else None)
    by_distance.pop(None, None)

    by_size = None
    if not anchor_error:
        from eval.anchor_coverage import summarize

        all_best = [v for m in positives if best_by_clip[m["clip_id"]] for v in best_by_clip[m["clip_id"]]]
        all_areas = [a for m in positives if best_by_clip[m["clip_id"]] is not None for a in areas_by_clip[m["clip_id"]]]
        by_size = summarize(all_best, all_areas)

    return {
        "footage_root": str(footage_root),
        "layout": layout,
        "num_clips": len(clips),
        "num_positive_clips": len(positives),
        "num_hard_negative_clips": len(negatives),
        "num_frames": sum(m["num_frames_extracted"] for _, m in clips),
        "label_status": {s: sum(m.get("label_status") == s for _, m in clips) for s in ("unlabeled", "partially_reviewed", "reviewed")},
        "combos": combos,
        "missing": missing,
        "grids": {"distance_band x lighting": grid("distance_band", "lighting"), "drone_type x distance_band": grid("drone_type", "distance_band")},
        "anchor_by_distance": by_distance,
        "anchor_by_size": by_size,
        "anchor_error": anchor_error,
    }


def render_markdown(report: dict) -> str:
    from_layout = report["layout"]
    layout_text = (
        f"input {from_layout.get('input_size', 320)}, tiles {from_layout.get('tile_rows', 1)}x{from_layout.get('tile_cols', 1)}"
    )
    lines = [
        "# Own-footage coverage",
        "",
        f"Footage root: `{report['footage_root']}` — {report['num_clips']} clips "
        f"({report['num_positive_clips']} positive, {report['num_hard_negative_clips']} hard-negative), "
        f"{report['num_frames']} frames. Labels: "
        + ", ".join(f"{n} {s}" for s, n in report["label_status"].items())
        + ".",
        "",
        "## Captured combinations",
        "",
        "Anchor check = share of reviewed drone boxes whose best anchor IoU >= 0.5 (median best IoU); "
        f"SSDLite anchors at {layout_text} (dut_v1 as shipped: input 320, tiles 1x1).",
        "",
        "| Drone type | Distance | Lighting | Background | Kind | Clips | Frames | Reviewed frames | Drone boxes | Anchor check |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(report["combos"]):
        row = report["combos"][key]
        lines.append(
            "| " + " | ".join(key) + f" | {row['clips']} | {row['frames']} | {row['reviewed_frames']} | {row['drone_boxes']} | {row['anchor_check']} |"
        )
    if not report["combos"]:
        lines.append("| _(no clips ingested yet)_ | | | | | | | | | |")

    lines += ["", "## Not yet captured (positive clips)", ""]
    for dim, values in report["missing"].items():
        label = "hard-negative subjects" if dim == "negative_subject" else dim
        lines.append(f"- **{label}**: " + (", ".join(values) if values else "all covered"))

    for title, g in report["grids"].items():
        lines += ["", f"## Positive frames: {title}", "", "| | " + " | ".join(g["cols"]) + " |", "|---" * (len(g["cols"]) + 1) + "|"]
        for r in g["rows"]:
            lines.append(f"| {r} | " + " | ".join(str(g["frames"].get((r, c), 0)) for c in g["cols"]) + " |")

    lines += ["", f"## Anchor check ({layout_text})", "", f"Reference: {DUT_TRAIN_REFERENCE}.", ""]
    if report["anchor_error"]:
        lines.append(f"Anchor check {report['anchor_error']}.")
    else:
        from eval.anchor_coverage import format_cell

        lines += ["| Size bucket (COCO, original pixels) | Reviewed drone boxes | Matchable (median best IoU) |", "|---|---|---|"]
        for name, stats in report["anchor_by_size"].items():
            lines.append(f"| {name} | {stats['count']} | {format_cell(stats)} |")
        lines += ["", "| Distance band | Reviewed drone boxes | Anchor check |", "|---|---|---|"]
        for band in DIMENSIONS["distance_band"]:
            if band in report["anchor_by_distance"]:
                row = report["anchor_by_distance"][band]
                lines.append(f"| {band} | {row['drone_boxes']} | {row['anchor_check']} |")
            else:
                lines.append(f"| {band} | 0 | not captured |")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict, output_dir: str | Path) -> None:
    """coverage.md, coverage.json, and coverage_matrix.csv — the full
    cross-product, zero cells included, for filtering what's missing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coverage.md").write_text(render_markdown(report), encoding="utf-8")

    def strip(rows):
        return {(" / ".join(k) if isinstance(k, tuple) else k): {f: v for f, v in r.items() if f not in ("best_ious", "areas")} for k, r in rows.items()}

    serializable = {
        **{k: v for k, v in report.items() if k not in ("combos", "grids", "anchor_by_distance")},
        "combos": strip(report["combos"]),
        "anchor_by_distance": strip(report["anchor_by_distance"]),
    }
    (output_dir / "coverage.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    dims = list(DIMENSIONS)
    with (output_dir / "coverage_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(dims + ["kind", "clips", "frames", "reviewed_frames", "drone_boxes", "anchor_check"])
        for kind in ("positive", "hard-negative"):
            drone_types = ["none"] if kind == "hard-negative" else [d for d in DIMENSIONS["drone_type"] if d != "none"]
            for values in itertools.product(drone_types, *(DIMENSIONS[d] for d in dims[1:])):
                row = report["combos"].get(values + (kind,))
                if row:
                    writer.writerow(list(values) + [kind, row["clips"], row["frames"], row["reviewed_frames"], row["drone_boxes"], row["anchor_check"]])
                else:
                    writer.writerow(list(values) + [kind, 0, 0, 0, 0, "not captured"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--footage-root", default=DEFAULT_FOOTAGE_ROOT)
    parser.add_argument("--output-dir", default=None, help="Also write coverage.md / coverage.json / coverage_matrix.csv here")
    parser.add_argument("--no-anchor-check", action="store_true", help="Skip the anchor check (no torch needed)")
    parser.add_argument("--input-size", type=int, default=320, help="Model input size for the anchor check (default 320 = dut_v1)")
    parser.add_argument("--tile-rows", type=int, default=1)
    parser.add_argument("--tile-cols", type=int, default=1)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    args = parser.parse_args(argv)

    try:
        report = build_report(
            args.footage_root,
            anchor_check=not args.no_anchor_check,
            input_size=args.input_size,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            tile_overlap=args.tile_overlap,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Coverage report failed: {exc}", file=sys.stderr)
        return 1
    print(render_markdown(report))
    if args.output_dir:
        write_outputs(report, args.output_dir)
        print(f"Wrote coverage.md, coverage.json, coverage_matrix.csv to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
