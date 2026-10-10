"""Shared vocabulary and on-disk layout for our own captured footage.

Used by tools/ingest_footage.py (writes it), tools/preannotate.py and
tools/import_reviewed.py (add to a clip), and tools/coverage_report.py
(reads it). Vocabulary follows docs/coverage-matrix.md's dimensions so the
two stay comparable.

One directory per ingested clip, in the unified layout
detector/datasets/loader.py reads:

    <footage_root>/<clip_id>/
        images/<clip_id>_f<source_frame:06d>.jpg
        frames.json        # {"images": [...]} — loader's image entries, no annotations
        clip_meta.json     # per-clip metadata sidecar (this module's schema)
        prelabels/         # tools/preannotate.py output: detector predictions, NOT ground truth
        annotations.json   # only once human-reviewed labels are imported (tools/import_reviewed.py)

annotations.json is deliberately not written at ingest: the loader reads an
image with no boxes as a hard negative, so an unlabeled drone clip with an
empty annotations.json would silently train as "no drone here".
"""

from __future__ import annotations

import json
from pathlib import Path

CLIP_META_FILENAME = "clip_meta.json"
FRAMES_FILENAME = "frames.json"
PRELABELS_DIRNAME = "prelabels"
ANNOTATIONS_FILENAME = "annotations.json"
SCHEMA_VERSION = 1

# drone_type "none" is reserved for hard-negative clips (no drone in frame).
DRONE_TYPES = ["fpv-quad", "multirotor-quad", "multirotor-hex-oct", "fixed-wing", "micro-nano", "other", "none"]
DISTANCE_BANDS = ["close-lt50m", "medium-50-200m", "far-200-500m", "very-far-gt500m"]
LIGHTING = ["daylight-clear", "daylight-overcast", "dusk-dawn", "night", "rain", "fog-haze", "snow"]
BACKGROUNDS = ["clean-sky", "urban-buildings", "foliage-trees", "terrain-mountains", "water", "mixed-cluttered"]
NEGATIVE_SUBJECTS = ["bird", "aircraft", "insect", "empty-sky", "other"]

DIMENSIONS: dict[str, list[str]] = {
    "drone_type": DRONE_TYPES,
    "distance_band": DISTANCE_BANDS,
    "lighting": LIGHTING,
    "background": BACKGROUNDS,
}


class ClipMetaError(ValueError):
    """Raised when clip metadata is missing, malformed, or inconsistent."""


def validate_meta(meta: dict) -> None:
    for key, allowed in DIMENSIONS.items():
        if meta.get(key) not in allowed:
            raise ClipMetaError(f"{key}={meta.get(key)!r} is not one of {allowed}")
    if not isinstance(meta.get("is_hard_negative"), bool):
        raise ClipMetaError("is_hard_negative must be true or false")
    if meta["is_hard_negative"] and meta["drone_type"] != "none":
        raise ClipMetaError("a hard-negative clip has no drone: use drone_type 'none'")
    if not meta["is_hard_negative"] and meta["drone_type"] == "none":
        raise ClipMetaError("drone_type 'none' is only for hard-negative clips (--hard-negative)")
    subject = meta.get("negative_subject")
    if subject is not None and subject not in NEGATIVE_SUBJECTS:
        raise ClipMetaError(f"negative_subject={subject!r} is not one of {NEGATIVE_SUBJECTS}")


def load_clip_dirs(footage_root: str | Path) -> list[tuple[Path, dict]]:
    """Every (clip_dir, metadata) under `footage_root`, sorted by clip id."""
    root = Path(footage_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Footage root not found: {root}")
    clips = []
    for meta_path in sorted(root.glob(f"*/{CLIP_META_FILENAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            validate_meta(meta)
        except ClipMetaError as exc:
            raise ClipMetaError(f"{meta_path}: {exc}") from exc
        clips.append((meta_path.parent, meta))
    return clips
