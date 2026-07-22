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

Two additional provenance fields exist for exactly this purpose:

- `license_id`: a **verbatim** string copied from the dataset's own
  source (a LICENSE file, a license section in the paper/README, or text
  a maintainer pasted directly from the official source) — or the literal
  string `"UNVERIFIED"` if no such text has been supplied. Never a
  guessed/inferred SPDX identifier.
- `commercial_ok`: `True` only once a maintainer has confirmed, from
  `license_id`'s own text, that commercial use is permitted; `False` if
  confirmed *not* permitted (e.g. explicit research/academic-only terms);
  `None` for "unknown" — the default, and the only honest value before
  `license_id` has been filled in from a verified source.

`detector/train.py`'s `--commercial-only` flag reads `commercial_ok` (via
a dataset's registry entry) to refuse training on any dataset that isn't
explicitly cleared, so a provenance-clean model can be rebuilt later
without re-auditing every dataset by hand.
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
    license_id: str = "UNVERIFIED"  # verbatim text from the source, or "UNVERIFIED" — never guessed
    commercial_ok: bool | None = None  # True/False once confirmed from license_id; None = unknown
    local_path: Path | None = None  # TODO: set to the converted dataset's directory before training
    notes: str = ""

    def is_available(self) -> bool:
        return self.local_path is not None and Path(self.local_path).exists()


# --------------------------------------------------------------------------
# Registry of named public anti-drone / aerial-object datasets.
#
# Three of the four entries below remain infrastructure placeholders:
# `source_url` and `local_path` are intentionally left as TODOs, and
# nothing was downloaded, scraped, or fetched to produce them. `dut-anti-uav`
# is the exception — its `source_url` was verified live against the
# dataset's official GitHub repo (wangdongdut/DUT-Anti-UAV) and its
# detection subset has actually been downloaded and converted for this
# repo's first real fine-tuning/eval pass (see docs/datasets.md). Its
# `license`/`license_id`/`commercial_ok` remain conservative regardless —
# see its `license_notes` for exactly what is and isn't confirmed.
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
            "Dalian University of Technology anti-UAV *detection* subset "
            "(Zhao, Zhang, Li, Wang, 'Vision-based Anti-UAV Detection and "
            "Tracking', IEEE TITS 2022, arXiv:2205.10851). 10,000 static "
            "images, official train/val/test split (5200/2600/2200), single "
            "'UAV' class, Pascal-VOC-style XML annotations (one <annotation> "
            "file per image, <object><name>UAV</name><bndbox> "
            "xmin/ymin/xmax/ymax in absolute pixels)."
        ),
        classes=["drone"],
        license=LicenseStatus.UNVERIFIED,
        license_notes=(
            "The GitHub repo (wangdongdut/DUT-Anti-UAV) contains an Apache-2.0 "
            "LICENSE file (confirmed verbatim from "
            "raw.githubusercontent.com/wangdongdut/DUT-Anti-UAV/master/LICENSE "
            "— standard Apache License 2.0 text), but that repo holds only a "
            "README and an example image; the actual detection-subset images "
            "and annotations are hosted externally (Google Drive / Baidu Pan "
            "links in the README) with no explicit license statement attached "
            "to them there. Whether the repo's Apache-2.0 grant is intended to "
            "cover the externally-hosted dataset itself is NOT confirmed — "
            "this is exactly the ambiguity `license_id`/`commercial_ok` exist "
            "to avoid guessing about. Pending a maintainer pasting the "
            "authoritative verbatim text (paper data-availability statement, "
            "author correspondence, or an explicit statement on the dataset "
            "download page) before either field is set to anything but "
            "'unverified'/unknown."
        ),
        source_url="https://github.com/wangdongdut/DUT-Anti-UAV",  # verified live 2026-07-22
        license_id="UNVERIFIED",  # see license_notes — repo LICENSE found, dataset-content license not confirmed
        commercial_ok=None,
        local_path=None,  # set per-split by detector/datasets/dut_anti_uav.py once converted; never local by default
        notes=(
            "14 detectors benchmarked on this exact test split in the source "
            "paper (Table II): SSD-VGG16 mAP 0.632 @ 33.2 FPS, Faster-RCNN "
            "ResNet50/ResNet18/VGG16 mAP 0.653/0.605/0.633, Cascade-RCNN "
            "ResNet50 (best) mAP 0.683, YOLOX-ResNet18 (fastest) 53.7 FPS. "
            "Paper does not state the IoU convention behind their single "
            "'mAP' column explicitly (P-R curves are shown separately for "
            "IoU=0.5 and IoU=0.75), so treat this as the closest available "
            "published reference, not a guaranteed apples-to-apples match "
            "against this repo's AP@0.5 definition."
        ),
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
