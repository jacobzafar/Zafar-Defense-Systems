#!/usr/bin/env python3
"""Dataset acquisition stub — does not download anything.

This deliberately does NOT fetch any dataset. Several of the datasets in
`detector/datasets/manifest.py` require visiting an official source,
accepting usage terms, and/or requesting access/credentials — steps only
a human can complete. This script's job is to make the missing steps
explicit rather than to silently skip them or guess at a URL.

Usage:
    python -m detector.datasets.download --list
    python -m detector.datasets.download --dataset anti-uav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from detector.datasets.manifest import DATASET_REGISTRY, get_manifest, list_datasets  # noqa: E402


class DatasetNotConfiguredError(RuntimeError):
    """Raised when a dataset has no `local_path` configured yet."""


def describe(name: str) -> str:
    manifest = get_manifest(name)
    lines = [
        f"Dataset: {manifest.name}",
        f"  Description : {manifest.description}",
        f"  Classes     : {', '.join(manifest.classes)}",
        f"  License     : {manifest.license.value}",
        f"  License notes: {manifest.license_notes}",
        f"  Source URL  : {manifest.source_url or '(not filled in — see TODO in manifest.py)'}",
        f"  Local path  : {manifest.local_path or '(not configured)'}",
        f"  Available   : {manifest.is_available()}",
    ]
    if manifest.notes:
        lines.append(f"  Notes       : {manifest.notes}")
    return "\n".join(lines)


def require_configured(name: str) -> Path:
    """Raise a clear, actionable error if a dataset isn't set up yet.

    Intended for use by detector/train.py before it attempts to load a
    dataset — fails fast with next steps instead of a confusing loader
    error deep in training code.
    """
    manifest = get_manifest(name)
    if manifest.local_path is None or not manifest.is_available():
        raise DatasetNotConfiguredError(
            f"Dataset '{manifest.name}' is not configured yet.\n"
            f"To use it:\n"
            f"  1. Visit the dataset's official source and review current usage terms.\n"
            f"     ({manifest.license_notes})\n"
            f"  2. Obtain the data (this may require registration/an access request).\n"
            f"  3. Convert it into the unified schema described in "
            f"detector/datasets/loader.py's module docstring "
            f"(images/ + annotations.json).\n"
            f"  4. Set this dataset's `local_path` in "
            f"detector/datasets/manifest.py (or pass an equivalent override "
            f"in your training config) to point at the converted directory."
        )
    return Path(manifest.local_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List known dataset names")
    parser.add_argument("--dataset", default=None, help="Describe one dataset and its setup status")
    args = parser.parse_args()

    if args.list:
        for name in list_datasets():
            print(name)
        return 0

    if args.dataset:
        try:
            print(describe(args.dataset))
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    print(f"Known datasets: {', '.join(list_datasets())}")
    print("Use --dataset <name> to see its setup status, or --list for names only.")
    print(f"({len(DATASET_REGISTRY)} datasets registered; none are downloaded or configured by default.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
