#!/usr/bin/env python3
"""Interleaved latency benchmark for detector configs.

The harness's latency comes from one pass per config, and on the shared
4-core CPU used here those passes vary by ~20% run to run for the same
work (DECISIONS #27). For latency claims, this runs every config over the
same frames, interleaved round by round (config A, B, C, A, B, C, ...),
so slow periods hit all configs alike, and reports per config the median
over rounds of the mean ms/frame, plus the spread across rounds.

Frames are loaded once, up front; only `detector.detect()` is timed, as in
the harness. Detector configs are taken from harness YAML files (their
`detector` block, with the confidence cutoff left as configured).

Usage:
    python -m eval.bench_latency --frames 60 --rounds 5 \\
        Squeeze=eval/config/fpv_val_stride8_squeeze_bilinear.yaml \\
        Area=eval/config/fpv_val_stride8_squeeze_area.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import yaml  # noqa: E402

from detector.factory import build_detector  # noqa: E402
from eval.schema import load_eval_manifest  # noqa: E402


def load_frames(eval_set_dir: str, count: int) -> list:
    frames = [f for s in load_eval_manifest(eval_set_dir) for f in s.frames]
    step = max(1, len(frames) // count)
    return [cv2.imread(str(f.image_path)) for f in frames[::step][:count]]


def bench(detectors: dict, images: list, rounds: int, warmup: int = 3) -> dict[str, dict[str, float]]:
    for detector in detectors.values():
        for image in images[:warmup]:
            detector.detect(image)
    per_round: dict[str, list[float]] = {name: [] for name in detectors}
    for _ in range(rounds):
        for name, detector in detectors.items():
            start = time.perf_counter()
            for image in images:
                detector.detect(image)
            per_round[name].append((time.perf_counter() - start) * 1000.0 / len(images))
    return {
        name: {
            "median_ms": round(statistics.median(values), 2),
            "min_ms": round(min(values), 2),
            "max_ms": round(max(values), 2),
            "fps": round(1000.0 / statistics.median(values), 2),
        }
        for name, values in per_round.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="LABEL=path/to/harness_config.yaml")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--eval-set-dir", help="Frames to time on (default: the first config's eval_set_dir)")
    args = parser.parse_args()

    detectors, eval_set_dir = {}, args.eval_set_dir
    for spec in args.configs:
        label, _, path = spec.rpartition("=")
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        eval_set_dir = eval_set_dir or raw["eval_set_dir"]
        detectors[label or Path(path).stem] = build_detector(raw["detector"])
    images = load_frames(eval_set_dir, args.frames)

    results = bench(detectors, images, args.rounds)
    print(f"{len(images)} frames from {eval_set_dir}, {args.rounds} interleaved rounds\n")
    print("| Config | ms/frame (median of rounds) | min–max over rounds | FPS |")
    print("|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['median_ms']:.1f} | {r['min_ms']:.1f}–{r['max_ms']:.1f} | {r['fps']:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
