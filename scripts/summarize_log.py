#!/usr/bin/env python3
"""Print a summary of a completed run's JSONL telemetry log.

Usage:
    python scripts/summarize_log.py logs/run_20260101_120000.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry.summary import format_summary, summarize_log  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to a run's .jsonl telemetry log")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1

    summary = summarize_log(log_path)
    if not summary.has_data:
        print(f"No frame events found in {log_path} (empty or malformed log).", file=sys.stderr)
        return 1

    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
