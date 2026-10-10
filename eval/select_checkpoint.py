"""Rank several weight files of the same drone model by threshold-independent AP.

Runs eval/harness.py's `run_evaluation` once per weights file (overriding
only `detector.weights_path` in the given eval config) and ranks them by
AP@0.5, with AP@[0.50:0.95] alongside — so the best training epoch is
measured, not assumed to be the last one. detector/train.py keeps one
`weights_epoch_NNN.pt` per epoch for exactly this.

Pick the checkpoint on a validation eval set (e.g. DUT's val split built
with eval/build_frozen_eval_set.py), then report only the chosen one on
the frozen test set: choosing among checkpoints by their test-set score
leaks the test set into model selection and inflates the reported number.

Usage:
    python -m eval.select_checkpoint --config eval/config/<val_config>.yaml \\
        models/dut_v2/weights_epoch_*.pt [--output-json path]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import EvalConfig, run_evaluation  # noqa: E402


def rank_checkpoints(config: EvalConfig, weights_paths: list[str]) -> list[dict[str, Any]]:
    """Evaluate each weights file; return rows sorted best-first by AP@0.5."""
    rows = []
    for weights_path in weights_paths:
        per_weights_config = copy.deepcopy(config)
        per_weights_config.detector["weights_path"] = str(weights_path)
        card = run_evaluation(per_weights_config)
        rows.append(
            {
                "weights_path": str(weights_path),
                "ap50": card.ap50,
                "ap50_95": card.ap50_95,
                "ap50_small": card.ap50_by_size["small"]["ap"],
                "recall_at_operating_threshold": card.operating_point["recall"],
            }
        )
    return sorted(rows, key=lambda row: row["ap50"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Eval config (its detector.weights_path is overridden per file)")
    parser.add_argument("weights", nargs="+", help="Weights files to compare, e.g. models/dut_v2/weights_epoch_*.pt")
    parser.add_argument("--output-json", help="Optional path to write the ranking as JSON")
    args = parser.parse_args()

    missing = [w for w in args.weights if not Path(w).exists()]
    if missing:
        print(f"Weights file(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    rows = rank_checkpoints(EvalConfig.from_yaml(args.config), args.weights)

    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    print("| Rank | Weights | AP@0.5 | AP@[0.50:0.95] | AP@0.5 small | Recall @ operating threshold |")
    print("|---|---|---|---|---|---|")
    for rank, row in enumerate(rows, start=1):
        print(
            f"| {rank} | `{row['weights_path']}` | {fmt(row['ap50'])} | {fmt(row['ap50_95'])} | "
            f"{fmt(row['ap50_small'])} | {fmt(row['recall_at_operating_threshold'])} |"
        )

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
