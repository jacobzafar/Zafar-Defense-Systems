#!/usr/bin/env python3
"""One comparison table from several harness metric cards.

Every row comes straight from a metric card JSON (eval/harness.py), so
the results table in docs is generated, not transcribed. Columns are the
ones the temporal / robustness experiment is judged on
(docs/DECISIONS.md #27): AP, operating-point recall overall and by
apparent size, false positives on drone frames and on empty frames,
track quality, and throughput.

Usage:
    python -m eval.compare Baseline=eval/output/fpv_test_stride8_clips_metric_card.json \\
        "JPEG q15=eval/output/fpv_test_stride8_sim_jpeg_metric_card.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SIZE_COLUMNS = ("0-2%", "2-3%", "3-4%", "4-5%", "5-7.5%")


def _f(value, spec=".3f") -> str:
    return "–" if value is None else format(value, spec)


def row(label: str, card: dict) -> str:
    op = card["operating_point"]
    fa = card["false_alarm_rate"]
    fpt = card.get("false_positives_on_target_frames") or {}
    by_size = card.get("recall_by_apparent_size") or {}
    tq = card.get("track_quality") or {}
    tc = card["track_continuity"]
    sizes = [_f((by_size.get(name) or {}).get("recall_at_threshold"), ".2f") for name in SIZE_COLUMNS]
    fa_str = f"{fa['num_false_alarms']}/{fa['num_hard_negative_frames']}" if fa["measurable"] else "–"
    false_ids = tq.get("num_false_track_ids") if tq.get("measurable") else None
    return (
        f"| {label} | {card['ap50']:.4f} | {_f(op['recall'])} | {_f(op['precision'])} | "
        + " | ".join(sizes)
        + f" | {_f(fpt.get('rate_per_frame'))} | {fa_str} | {_f(tq.get('coverage'))} | "
        f"{_f(false_ids, 'd')} | {_f(tc.get('overall_continuity'))} | {card['latency']['fps']:.1f} |"
    )


def render(cards: list[tuple[str, dict]]) -> str:
    header = (
        "| Variant | AP@0.5 | Recall @op | Precision @op | "
        + " | ".join(f"Recall {s}" for s in SIZE_COLUMNS)
        + " | FP / drone frame | False alarms / empty frames | Track coverage | False track IDs | Continuity | FPS |"
    )
    sep = "|" + "---|" * (header.count("|") - 1)
    return "\n".join([header, sep, *(row(label, card) for label, card in cards)]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cards", nargs="+", help="LABEL=path/to/metric_card.json")
    args = parser.parse_args()
    cards = []
    for spec in args.cards:
        label, _, path = spec.rpartition("=")
        cards.append((label or Path(path).stem, json.loads(Path(path).read_text(encoding="utf-8"))))
    print(render(cards), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
