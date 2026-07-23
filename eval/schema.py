"""Frozen evaluation-set schema and loader.

This is a deliberately separate schema from `detector/datasets/loader.py`:
that loader is single-image-oriented (right for classification-style
fine-tuning), while evaluation needs ordered *sequences* of frames so the
track-continuity metric (see `eval/metrics.py`) can measure ID stability
across time, matching the sequence/tracking style of the named public
anti-UAV benchmarks this repo's `detector/datasets/manifest.py` registers.

Directory layout expected under `eval_set_dir`:

    <eval_set_dir>/
        .frozen                 # empty marker file — see freeze_eval_set()
        images/
            <file_name>          # referenced by sequences.json
        sequences.json

    sequences.json:
    {
      "sequences": [
        {
          "id": "seq_001",
          "frames": [
            {
              "file_name": "seq_001_0001.jpg", "width": 1920, "height": 1080,
              "boxes": [{"bbox": [x, y, w, h], "category": "drone"}]
            }
          ]
        }
      ]
    }

A frame with no "drone"-category box (either an empty `boxes` list, or
only other categories like "bird"/"clutter") is a hard-negative frame —
used by the false-alarm-rate metric.

**Frozen means frozen.** `assert_not_for_training()` is a hard technical
guard, not just a convention: `detector/train.py` calls it before loading
any dataset, and refuses to proceed if the target directory contains a
`.frozen` marker file. Mark a real eval set frozen with `freeze_eval_set()`
once it's finalized, and never point `detector/train.py`'s `dataset_dir`
at it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

FROZEN_MARKER_FILENAME = ".frozen"


class EvalSetFrozenError(RuntimeError):
    """Raised when something tries to train on a frozen evaluation set."""


class EvalSetFormatError(ValueError):
    """Raised when sequences.json doesn't match the expected schema."""


class EvalSetChecksumMismatchError(ValueError):
    """Raised when a frozen eval set's contents don't match its recorded checksum."""


@dataclass
class EvalBox:
    x1: float
    y1: float
    x2: float
    y2: float
    category: str

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class EvalFrame:
    sequence_id: str
    frame_index: int
    image_path: Path
    width: int
    height: int
    boxes: list[EvalBox] = field(default_factory=list)

    @property
    def gt_drone_boxes(self) -> list[EvalBox]:
        return [b for b in self.boxes if b.category == "drone"]

    @property
    def is_hard_negative(self) -> bool:
        return len(self.gt_drone_boxes) == 0


@dataclass
class EvalSequence:
    sequence_id: str
    frames: list[EvalFrame]


def is_frozen(dataset_dir: str | Path) -> bool:
    return (Path(dataset_dir) / FROZEN_MARKER_FILENAME).exists()


def freeze_eval_set(eval_set_dir: str | Path) -> None:
    """Mark a directory as a frozen evaluation set (idempotent)."""
    (Path(eval_set_dir) / FROZEN_MARKER_FILENAME).touch()


def assert_not_for_training(dataset_dir: str | Path) -> None:
    """Raise `EvalSetFrozenError` if `dataset_dir` is a frozen eval set.

    Called by `detector/train.py` before loading any dataset — this is the
    hard technical enforcement behind "the eval set must never be used for
    training," not just a documentation warning.
    """
    if is_frozen(dataset_dir):
        raise EvalSetFrozenError(
            f"'{dataset_dir}' is marked frozen (contains {FROZEN_MARKER_FILENAME}) — "
            f"it is reserved for evaluation only (see eval/README.md) and must never "
            f"be used as a training dataset_dir."
        )


def compute_manifest_checksum(eval_set_dir: str | Path) -> str:
    """SHA-256 of sequences.json's raw bytes.

    Detects structural/label tampering (added/removed/modified boxes or
    frames) cheaply. It does NOT hash image pixel data — swapping an image
    file's contents while keeping its file name would not be caught. Treat
    this as a lightweight integrity check, not a cryptographic guarantee
    the images themselves are unchanged.
    """
    manifest_path = Path(eval_set_dir) / "sequences.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"sequences.json not found in {eval_set_dir}")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def verify_checksum(eval_set_dir: str | Path, expected_checksum: str) -> None:
    actual = compute_manifest_checksum(eval_set_dir)
    if actual != expected_checksum:
        raise EvalSetChecksumMismatchError(
            f"Eval set at {eval_set_dir} does not match its recorded checksum "
            f"(expected {expected_checksum}, got {actual}). The frozen eval set "
            f"may have been modified — re-verify before trusting any comparison "
            f"against previous results."
        )


def load_eval_manifest(eval_set_dir: str | Path) -> list[EvalSequence]:
    eval_set_dir = Path(eval_set_dir)
    if not eval_set_dir.exists():
        raise FileNotFoundError(f"Eval set directory not found: {eval_set_dir}")

    manifest_path = eval_set_dir / "sequences.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"sequences.json not found in {eval_set_dir}. Expected an eval set in the "
            f"layout documented in eval/schema.py's module docstring."
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalSetFormatError(f"Could not parse {manifest_path}: {exc}") from exc

    if "sequences" not in raw:
        raise EvalSetFormatError(f"{manifest_path} is missing the required 'sequences' key.")

    images_dir = eval_set_dir / "images"
    sequences: list[EvalSequence] = []
    for seq_entry in raw["sequences"]:
        try:
            sequence_id = seq_entry["id"]
            frame_entries = seq_entry["frames"]
        except KeyError as exc:
            raise EvalSetFormatError(f"Malformed sequence entry in {manifest_path}: missing {exc}") from exc

        frames: list[EvalFrame] = []
        for frame_index, frame_entry in enumerate(frame_entries):
            try:
                file_name = frame_entry["file_name"]
                width = frame_entry["width"]
                height = frame_entry["height"]
            except KeyError as exc:
                raise EvalSetFormatError(f"Malformed frame entry in sequence '{sequence_id}': missing {exc}") from exc

            boxes = []
            for box_entry in frame_entry.get("boxes", []):
                try:
                    x, y, w, h = box_entry["bbox"]
                    category = box_entry["category"]
                except (KeyError, ValueError) as exc:
                    raise EvalSetFormatError(f"Malformed box in sequence '{sequence_id}': {exc}") from exc
                boxes.append(EvalBox(x1=x, y1=y, x2=x + w, y2=y + h, category=category))

            frames.append(
                EvalFrame(
                    sequence_id=sequence_id,
                    frame_index=frame_index,
                    image_path=images_dir / file_name,
                    width=width,
                    height=height,
                    boxes=boxes,
                )
            )

        sequences.append(EvalSequence(sequence_id=sequence_id, frames=frames))

    return sequences
