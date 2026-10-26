#!/usr/bin/env python3
"""Split VisioDECT's DJIFPV subset into train / val / test without leakage.

Each scenario (Sunny / Cloudy / Evening) is one continuous video: frame
"(n)" and "(n+1)" differ by a median 0.3-0.6% of the frame width in drone
position, over the same background. A random per-frame split would put
near-duplicates of every test frame into training. So each scenario's
frames, in frame-number order, are cut into `--blocks` contiguous blocks;
whole blocks go to test and val, the rest to train, and `--guard` frames
of train next to every held-out block are dropped entirely.

The rule is fixed in advance and uses no model output: test = blocks 2
and 7, val = block 5 (of 10, 0-based), the same positions in every
scenario. Spread through the clip because drone size drifts along it
(Sunny goes from ~6% to ~1.5% apparent size).

What this does NOT remove: the test frames are still the same clip,
scene, camera and day as training. Results on this split are
in-domain — an upper bound for new footage, not a generalization claim.

Writes unified-layout dirs (images/ symlinked + annotations.json, as
detector/datasets/visiodect.py does) under `--output-dir`/{train,val,test},
plus split_manifest.json recording the exact frame ranges.

Usage (docs/DECISIONS.md #26):
    python tools/split_visiodect_fpv.py --source-dir data/visiodect/converted-fpv \
        --output-dir data/visiodect/fpv-split
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TEST_BLOCKS = (2, 7)
VAL_BLOCKS = (5,)
_FRAME_NUMBER = re.compile(r"\((\d+)\)")


def frame_number(file_name: str) -> int:
    match = _FRAME_NUMBER.search(file_name)
    if not match:
        raise ValueError(f"No '(n)' frame number in {file_name!r}")
    return int(match.group(1))


def assign_splits(
    images: list[dict],
    blocks: int = 10,
    guard: int = 30,
    test_blocks: tuple[int, ...] = TEST_BLOCKS,
    val_blocks: tuple[int, ...] = VAL_BLOCKS,
) -> tuple[dict[int, str], list[dict]]:
    """image id -> "train" / "val" / "test" / "guard", per scenario, and
    one record per scenario block with its frame range and split."""
    by_scenario: dict[str, list[dict]] = {}
    for image in images:
        by_scenario.setdefault(image["scenario"], []).append(image)

    assignment: dict[int, str] = {}
    ranges: list[dict] = []
    for scenario, group in sorted(by_scenario.items()):
        group.sort(key=lambda im: frame_number(im["file_name"]))
        n = len(group)
        bounds = [round(k * n / blocks) for k in range(blocks + 1)]
        block_of = [next(b for b in range(blocks) if bounds[b] <= i < bounds[b + 1]) for i in range(n)]
        split_of_block = {
            b: "test" if b in test_blocks else "val" if b in val_blocks else "train" for b in range(blocks)
        }
        held_out = [i for i in range(n) if split_of_block[block_of[i]] != "train"]
        for i, image in enumerate(group):
            split = split_of_block[block_of[i]]
            if split == "train" and any(abs(i - j) <= guard for j in held_out):
                split = "guard"
            assignment[image["id"]] = split
        for b in range(blocks):
            first, last = group[bounds[b]], group[bounds[b + 1] - 1]
            ranges.append(
                {
                    "scenario": scenario,
                    "block": b,
                    "split": split_of_block[b],
                    "frames": [frame_number(first["file_name"]), frame_number(last["file_name"])],
                }
            )
    return assignment, ranges


def write_split(source_dir: Path, output_dir: Path, blocks: int, guard: int) -> dict[str, dict]:
    raw = json.loads((source_dir / "annotations.json").read_text(encoding="utf-8"))
    assignment, ranges = assign_splits(raw["images"], blocks, guard)
    counts: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        images = [im for im in raw["images"] if assignment[im["id"]] == split]
        ids = {im["id"] for im in images}
        annotations = [a for a in raw["annotations"] if a["image_id"] in ids]
        split_dir = output_dir / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        for image in images:
            link = split_dir / "images" / image["file_name"]
            if not link.exists():
                link.symlink_to((source_dir / "images" / image["file_name"]).resolve())
        (split_dir / "annotations.json").write_text(
            json.dumps({"images": images, "annotations": annotations}, indent=2), encoding="utf-8"
        )
        counts[split] = {
            "images": len(images),
            "drones": len(annotations),
            "empty_images": len(ids - {a["image_id"] for a in annotations}),
        }
    counts["guard_dropped"] = {"images": sum(v == "guard" for v in assignment.values())}
    (output_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "rule": f"{blocks} contiguous blocks per scenario; test={list(TEST_BLOCKS)}, "
                f"val={list(VAL_BLOCKS)}, train=rest minus {guard} guard frames next to held-out blocks",
                "counts": counts,
                "blocks": ranges,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True, help="Unified-layout DJIFPV dir (visiodect.py --models DJIFPV)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--guard", type=int, default=30, help="Train frames dropped next to each held-out block")
    args = parser.parse_args()
    counts = write_split(Path(args.source_dir), Path(args.output_dir), args.blocks, args.guard)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
