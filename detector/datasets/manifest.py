"""Unified dataset manifest schema for drone-detector fine-tuning.

This module intentionally does NOT ship any dataset, weights, or fetched
metadata. It defines one schema that every source dataset is expected to
be converted into (see `detector/datasets/loader.py`), and a registry
entry per named public dataset with `local_path=None` — i.e. "not
provided" — until a human fills it in.

License fields are deliberately conservative: several of these datasets
are distributed under research/academic-use terms that require visiting
the original source and, in some cases, signing an agreement or
requesting access. Rather than asserting a specific SPDX license this
module cannot verify, the `license` field is set to `LicenseStatus.
UNVERIFIED` unless a maintainer has confirmed otherwise, and
`license_notes` records what's generally understood about typical usage
terms. Treat `license_notes` as a starting point for your own
verification, not a legal conclusion — confirm directly from each
dataset's official source before downloading or using it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LicenseStatus(str, Enum):
    """Coarse license-confidence marker — see module docstring."""

    UNVERIFIED = "UNVERIFIED"  # not yet confirmed by a maintainer against the current source
    RESEARCH_ONLY = "RESEARCH_ONLY_CONFIRMED"  # confirmed academic/research-use-only terms
    PERMISSIVE_CONFIRMED = "PERMISSIVE_CONFIRMED"  # confirmed permissive (e.g. CC-BY, MIT-like) terms


@dataclass
class DatasetManifest:
    """One dataset's identity + where a converted local copy would live.

    `local_path`, when set, must point to a directory containing this
    unified schema's `images/` folder and `annotations.json` (see
    `detector/datasets/loader.py`) — i.e. already converted from whatever
    the source dataset's native format is. This module does not perform
    that conversion; it only records where the result is expected to go.
    """

    name: str
    description: str
    classes: list[str]
    license: LicenseStatus
    license_notes: str
    source_url: str | None  # TODO: fill in from the dataset's current official page — never guessed here
    local_path: Path | None = None  # TODO: set to the converted dataset's directory before training
    notes: str = ""

    def is_available(self) -> bool:
        return self.local_path is not None and Path(self.local_path).exists()


# --------------------------------------------------------------------------
# Registry of named public anti-drone / aerial-object datasets.
#
# All four entries below are infrastructure placeholders: `source_url` and
# `local_path` are intentionally left as TODOs. Nothing here was
# downloaded, scraped, or fetched to produce this file.
# --------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, DatasetManifest] = {
    "anti-uav": DatasetManifest(
        name="Anti-UAV",
        description=(
            "RGB+thermal drone tracking/detection benchmark released alongside the "
            "Anti-UAV challenge series. TODO: confirm current version/edition and "
            "source before use."
        ),
        classes=["drone"],
        license=LicenseStatus.UNVERIFIED,
        license_notes=(
            "Anti-UAV challenge datasets have historically been distributed for "
            "research/academic use, often requiring a request/registration step. "
            "Confirm the current terms directly from the official challenge page "
            "or paper before downloading or using this data."
        ),
        source_url=None,  # TODO: fill in from the current official Anti-UAV challenge page
        local_path=None,  # TODO: point at a locally converted copy (see loader.py schema)
        notes="TODO: verify class taxonomy (single 'drone' class vs. multiple UAV types).",
    ),
    "dut-anti-uav": DatasetManifest(
        name="DUT Anti-UAV",
        description=(
            "Dalian University of Technology anti-UAV detection/tracking dataset. "
            "TODO: confirm current source and access process."
        ),
        classes=["drone"],
        license=LicenseStatus.UNVERIFIED,
        license_notes=(
            "Typically distributed for academic/research use, commonly via a "
            "request to the authors. Confirm current terms from the paper/repo "
            "before use — do not assume redistribution or commercial-use rights."
        ),
        source_url=None,  # TODO: fill in from the current official source
        local_path=None,  # TODO
        notes="TODO: confirm whether bounding-box or segmentation annotations are provided.",
    ),
    "drone-vs-bird": DatasetManifest(
        name="Drone-vs-Bird",
        description=(
            "Drone-vs-Bird Detection Challenge dataset (ICASSP/AVSS challenge "
            "series) — video clips with drones and birds as a hard-negative class "
            "by construction. TODO: confirm current challenge edition/source."
        ),
        classes=["drone", "bird"],
        license=LicenseStatus.UNVERIFIED,
        license_notes=(
            "Challenge datasets of this kind are typically released to registered "
            "participants under challenge-specific terms that can change by "
            "edition/year. Confirm current terms from the official challenge page "
            "before use."
        ),
        source_url=None,  # TODO: fill in from the current official challenge page
        local_path=None,  # TODO
        notes=(
            "This is the primary intended source of structured hard-negative "
            "(bird) examples for detector/train.py — see its --hard-negative "
            "handling."
        ),
    ),
    "visiodect": DatasetManifest(
        name="VisioDECT",
        description="Drone detection dataset. TODO: confirm current source and edition.",
        classes=["drone"],
        license=LicenseStatus.UNVERIFIED,
        license_notes=(
            "Not yet verified by a maintainer of this repo. Confirm the current "
            "license/terms directly from the dataset's official source before "
            "downloading or using it."
        ),
        source_url=None,  # TODO: fill in from the current official source
        local_path=None,  # TODO
        notes="TODO: confirm annotation format and class taxonomy.",
    ),
}


def get_manifest(name: str) -> DatasetManifest:
    key = name.strip().lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {available}.")
    return DATASET_REGISTRY[key]


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)
