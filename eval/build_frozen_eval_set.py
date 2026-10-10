#!/usr/bin/env python3
"""Build a frozen eval set (eval/schema.py's layout) from a unified-schema
single-image dataset directory (detector/datasets/loader.py's layout).

Every image becomes its own single-frame "sequence". This matters for
static-image detection datasets like DUT Anti-UAV's detection subset,
which has no temporal structure between images at all — there is no real
multi-frame track to preserve, so wrapping each image as a trivial
one-frame sequence is the honest representation, not an approximation of
one. A direct consequence: **track continuity is not a meaningful metric
on an eval set built this way** — with exactly one frame per sequence, an
ID switch can never occur, so the metric will trivially read as perfect
continuity regardless of the detector. This is documented in the
generated report, not silently left to look like a real result.

`--max-images N --seed S` keeps a seeded random subset of N images,
drawn exactly the way detector/train.py draws its per-epoch val subset
(`random.Random(seed).sample(samples, N)`), so e.g. DUT's val split can be
turned into an eval set matching the images a training run was monitored
on — used for choosing inference/training settings without touching the
test set.

Usage:
    python -m eval.build_frozen_eval_set \
        --source-dataset-dir data/dut-anti-uav/converted/test \
        --eval-set-dir data/frozen_eval/dut-anti-uav-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.datasets.loader import load_manifest_dataset  # noqa: E402
from eval.schema import compute_manifest_checksum, freeze_eval_set  # noqa: E402


def build_frozen_eval_set(
    source_dataset_dir: str | Path,
    eval_set_dir: str | Path,
    max_images: int | None = None,
    seed: int = 42,
) -> str:
    """Convert `source_dataset_dir` into a frozen eval set at
    `eval_set_dir`, one single-frame sequence per image. Idempotent:
    re-running with the same source only adds missing image symlinks and
    rewrites sequences.json. Returns the resulting sequences.json's
    sha256 checksum (see eval.schema.compute_manifest_checksum).
    """
    samples = load_manifest_dataset(source_dataset_dir)
    if not samples:
        raise ValueError(f"No images found in {source_dataset_dir}")
    if max_images is not None and len(samples) > max_images:
        samples = random.Random(seed).sample(samples, max_images)

    eval_set_dir = Path(eval_set_dir)
    images_dir = eval_set_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    sequences = []
    for index, sample in enumerate(samples, start=1):
        file_name = sample.image_path.name
        symlink_path = images_dir / file_name
        if not symlink_path.exists():
            symlink_path.symlink_to(sample.image_path.resolve())

        boxes = [
            {"bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1], "category": b.category} for b in sample.boxes
        ]
        sequences.append(
            {
                "id": f"img_{index:05d}",
                "frames": [
                    {
                        "file_name": file_name,
                        "width": sample.width,
                        "height": sample.height,
                        "boxes": boxes,
                    }
                ],
            }
        )

    (eval_set_dir / "sequences.json").write_text(
        json.dumps({"sequences": sequences}, indent=2), encoding="utf-8"
    )
    freeze_eval_set(eval_set_dir)
    return compute_manifest_checksum(eval_set_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-dir", required=True, help="Unified-schema dataset dir (images/+annotations.json)")
    parser.add_argument("--eval-set-dir", required=True, help="Where to write images/+sequences.json+.frozen")
    parser.add_argument("--max-images", type=int, help="Keep only a seeded random subset of this many images")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --max-images (default 42, as detector/train.py)")
    args = parser.parse_args()

    checksum = build_frozen_eval_set(args.source_dataset_dir, args.eval_set_dir, args.max_images, args.seed)
    print(f"Frozen eval set written to {args.eval_set_dir}")
    print(f"sequences.json sha256: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
