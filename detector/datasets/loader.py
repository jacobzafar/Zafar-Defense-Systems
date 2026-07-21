"""Loader for the unified dataset schema used by detector/train.py.

Every source dataset in `detector/datasets/manifest.py` is expected to be
converted (by a human, outside this codebase — no such converter is
implemented here) into this one directory layout before training:

    <local_path>/
        images/
            <file_name>            # referenced by annotations.json's "file_name"
            ...
        annotations.json

`annotations.json` is a deliberately small, COCO-inspired schema — not
full COCO — so this loader has no dependency on `pycocotools` or any
other new package:

    {
      "images": [
        {"id": 1, "file_name": "0001.jpg", "width": 1920, "height": 1080}
      ],
      "annotations": [
        {"image_id": 1, "bbox": [x, y, w, h], "category": "drone"}
      ]
    }

An image with no annotations at all is a valid, expected case: a
hard-negative / background-only sample (e.g. a bird or clutter frame with
no drone present). `category` values other than "drone" (e.g. "bird",
"clutter") are kept as labeled hard negatives rather than dropped, so
`detector/train.py` can handle them explicitly instead of silently
ignoring them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BoxAnnotation:
    """One labeled box in an image's own pixel coordinate space."""

    x1: float
    y1: float
    x2: float
    y2: float
    category: str


@dataclass
class ImageSample:
    """One image plus its boxes (empty = hard negative / background-only)."""

    image_path: Path
    width: int
    height: int
    boxes: list[BoxAnnotation] = field(default_factory=list)

    @property
    def is_hard_negative(self) -> bool:
        """True if this image has no positive ("drone") boxes at all."""
        return not any(b.category == "drone" for b in self.boxes)


class DatasetFormatError(ValueError):
    """Raised when annotations.json doesn't match the expected schema."""


def load_manifest_dataset(dataset_dir: str | Path) -> list[ImageSample]:
    """Load a converted dataset directory into a list of `ImageSample`.

    Raises `FileNotFoundError` if the directory or annotations file is
    missing, and `DatasetFormatError` if the JSON doesn't match the
    expected shape — both with a message naming the exact path, so a
    misconfigured `local_path` in a `DatasetManifest` fails clearly rather
    than with a confusing KeyError deep in training code.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    annotations_path = dataset_dir / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"annotations.json not found in {dataset_dir}. Expected a converted "
            f"dataset directory — see detector/datasets/loader.py's module docstring "
            f"for the required layout."
        )

    try:
        raw = json.loads(annotations_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"Could not parse {annotations_path}: {exc}") from exc

    if "images" not in raw or "annotations" not in raw:
        raise DatasetFormatError(
            f"{annotations_path} is missing required top-level keys 'images'/'annotations'."
        )

    images_dir = dataset_dir / "images"
    samples_by_id: dict[int, ImageSample] = {}
    for image_entry in raw["images"]:
        try:
            image_id = image_entry["id"]
            file_name = image_entry["file_name"]
            width = image_entry["width"]
            height = image_entry["height"]
        except KeyError as exc:
            raise DatasetFormatError(f"Malformed image entry in {annotations_path}: missing {exc}") from exc

        samples_by_id[image_id] = ImageSample(
            image_path=images_dir / file_name, width=width, height=height
        )

    for ann_entry in raw["annotations"]:
        try:
            image_id = ann_entry["image_id"]
            x, y, w, h = ann_entry["bbox"]
            category = ann_entry["category"]
        except (KeyError, ValueError) as exc:
            raise DatasetFormatError(f"Malformed annotation entry in {annotations_path}: {exc}") from exc

        if image_id not in samples_by_id:
            raise DatasetFormatError(
                f"Annotation references unknown image_id {image_id} in {annotations_path}."
            )

        samples_by_id[image_id].boxes.append(
            BoxAnnotation(x1=x, y1=y, x2=x + w, y2=y + h, category=category)
        )

    return list(samples_by_id.values())
