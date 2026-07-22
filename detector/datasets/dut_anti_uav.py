#!/usr/bin/env python3
"""Concrete converter for the DUT Anti-UAV detection subset.

The first dataset in `detector/datasets/manifest.py`'s registry with an
actual converter behind it — see `docs/datasets.md` for full provenance
notes (verified source, real per-split counts, license status).

Converts the dataset's native per-split layout, as distributed by the
official source (train.zip/val.zip/test.zip, each containing `img/` and
`xml/`):

    <raw_split_dir>/img/00001.jpg
    <raw_split_dir>/xml/00001.xml   # Pascal-VOC-style, one <annotation> per image

into this repo's unified schema (see `detector/datasets/loader.py`):

    <output_dir>/images/00001.jpg   # symlinked, not copied — these are large, gitignored trees
    <output_dir>/annotations.json

The source XML's single object class is literally named `"UAV"` — mapped
to this repo's `"drone"` category. Any other category value would be kept
as-is (this repo's hard-negative convention — see loader.py), though none
have been observed in this dataset. The `train` split does contain 3
genuine background-only images (XML files with zero `<object>` tags at
all — verified: `data/dut-anti-uav/raw/train/xml/{00579,00639,00724}.xml`);
`val` and `test` have none. These count as real hard negatives in the
converted output, distinct from the degenerate-box case below.

One data-quality issue was found and is handled explicitly rather than
silently: a small number of boxes in the raw XML are degenerate (zero
width or height, e.g. xmin == xmax) — these are skipped with a count in
the printed/written stats, not silently kept as a zero-area training
target or allowed to crash the converter.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SOURCE_CLASS = "UAV"
TARGET_CLASS = "drone"
SMALL_OBJECT_AREA_PX = 1024.0  # COCO's small-object convention (32*32) — matches eval/metrics.py's default


class DutAnnotationFormatError(ValueError):
    """Raised when a DUT XML annotation file doesn't match the expected VOC-style schema."""


@dataclass
class ConversionStats:
    split: str
    num_images: int
    num_drone_instances: int
    num_hard_negative_images: int
    num_small_instances: int
    num_degenerate_boxes_skipped: int
    small_object_area_threshold_px: float = SMALL_OBJECT_AREA_PX

    def as_dict(self) -> dict:
        return asdict(self)

    def print_summary(self) -> None:
        pct = (self.num_small_instances / self.num_drone_instances * 100) if self.num_drone_instances else 0.0
        print(
            f"[{self.split}] images={self.num_images}  "
            f"drone_instances={self.num_drone_instances}  "
            f"hard_negative_images={self.num_hard_negative_images}  "
            f"small_instances={self.num_small_instances} "
            f"({pct:.1f}% of instances, area<{self.small_object_area_threshold_px:.0f}px²)  "
            f"degenerate_boxes_skipped={self.num_degenerate_boxes_skipped}"
        )


def _parse_one_xml(xml_path: Path) -> tuple[int, int, list[tuple[float, float, float, float, str]]]:
    """Returns (width, height, boxes); boxes is [(x1, y1, x2, y2, category), ...]
    with `category` already mapped ("UAV" -> "drone").
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise DutAnnotationFormatError(f"Could not parse {xml_path}: {exc}") from exc

    size = root.find("size")
    if size is None:
        raise DutAnnotationFormatError(f"{xml_path} missing <size>")
    try:
        width = int(size.find("width").text)
        height = int(size.find("height").text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DutAnnotationFormatError(f"{xml_path} has a malformed <size>: {exc}") from exc

    boxes: list[tuple[float, float, float, float, str]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_el is None or bndbox is None:
            raise DutAnnotationFormatError(f"{xml_path} has an <object> missing <name>/<bndbox>")
        try:
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise DutAnnotationFormatError(f"{xml_path} has a malformed <bndbox>: {exc}") from exc

        category = TARGET_CLASS if name_el.text == SOURCE_CLASS else name_el.text
        boxes.append((x1, y1, x2, y2, category))

    return width, height, boxes


def convert_split(raw_split_dir: str | Path, output_dir: str | Path) -> ConversionStats:
    """Convert one raw DUT Anti-UAV split directory into the unified schema.

    Raises `FileNotFoundError` if the expected `img/`/`xml/` layout isn't
    present, or `DutAnnotationFormatError` if an XML file doesn't match
    the expected VOC-style schema.
    """
    raw_split_dir = Path(raw_split_dir)
    img_dir = raw_split_dir / "img"
    xml_dir = raw_split_dir / "xml"
    if not img_dir.is_dir() or not xml_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{raw_split_dir}' to contain both 'img/' and 'xml/' subdirectories "
            f"(DUT Anti-UAV's native per-split layout) — see detector/datasets/dut_anti_uav.py."
        )

    xml_paths = sorted(xml_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No .xml annotation files found in {xml_dir}")

    output_dir = Path(output_dir)
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    images_out = []
    annotations_out = []
    num_drone_instances = 0
    num_hard_negative_images = 0
    num_small_instances = 0
    num_degenerate_skipped = 0

    for image_id, xml_path in enumerate(xml_paths, start=1):
        stem = xml_path.stem
        image_path = img_dir / f"{stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Annotation {xml_path} has no matching image at {image_path}")

        width, height, raw_boxes = _parse_one_xml(xml_path)
        file_name = f"{stem}.jpg"
        images_out.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        symlink_path = out_images_dir / file_name
        if not symlink_path.exists():
            symlink_path.symlink_to(image_path.resolve())

        has_drone_box = False
        for x1, y1, x2, y2, category in raw_boxes:
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                num_degenerate_skipped += 1
                continue
            annotations_out.append({"image_id": image_id, "bbox": [x1, y1, w, h], "category": category})
            if category == TARGET_CLASS:
                has_drone_box = True
                num_drone_instances += 1
                if w * h < SMALL_OBJECT_AREA_PX:
                    num_small_instances += 1

        if not has_drone_box:
            num_hard_negative_images += 1

    (output_dir / "annotations.json").write_text(
        json.dumps({"images": images_out, "annotations": annotations_out}, indent=2), encoding="utf-8"
    )

    stats = ConversionStats(
        split=raw_split_dir.name,
        num_images=len(images_out),
        num_drone_instances=num_drone_instances,
        num_hard_negative_images=num_hard_negative_images,
        num_small_instances=num_small_instances,
        num_degenerate_boxes_skipped=num_degenerate_skipped,
    )
    (output_dir / "conversion_stats.json").write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Path to one raw split dir (contains img/ and xml/)")
    parser.add_argument("--output-dir", required=True, help="Where to write images/ + annotations.json")
    args = parser.parse_args()

    stats = convert_split(args.raw_dir, args.output_dir)
    stats.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
