#!/usr/bin/env python3
"""Concrete converter for the VisioDECT drone-detection dataset.

VisioDECT ships as one directory per UAV model, each with `images/<Scenario>/`
(Title-case: Evening/Cloudy/Sunny) and `labels/<scenario>/` (lowercase),
the latter holding up to three parallel annotation formats
(`csv.csv`, `txt/` YOLO-style, `voc/` Pascal-VOC XML) for the same boxes.
This converter reads `csv.csv` — verified to be the only format present
for every populated model/scenario combination in the copy this was
written against (`txt/`/`voc/` are missing or incomplete for several
combinations; see docs/datasets.md for the exact counts):

    <raw_dir>/<Model>/images/<Scenario>/<file>.jpg
    <raw_dir>/<Model>/labels/<scenario>/csv.csv
        # one row per box, no header, comma-separated:
        # <class_name>,<xmin>,<ymin>,<width>,<height>,<file_name>,<img_width>,<img_height>
        # (absolute pixels; class_name is always "<Model>_<Scenario>" in the
        # copy this was verified against — every row is a positive drone
        # instance, there is no hard-negative/other class in this dataset)

into this repo's unified schema (see `detector/datasets/loader.py`):

    <output_dir>/images/<file>.jpg   # symlinked, not copied
    <output_dir>/annotations.json

Three real data-quality/completeness issues were found in the copy this
was written against and are handled explicitly, not silently or by
guessing:

1. **Not all six documented UAV models have data.** Only `Anafi-Extended`,
   `DJIFPV`, and `DJIPhantom` contained any images or labels; `EFT-E410S`,
   `Mavic_Air`, and `Mavic_Enterprise` existed only as empty directory
   skeletons (`images/<Scenario>/` and `labels/<scenario>/{voc,txt}/`
   present, zero files inside any of them). All six are still walked —
   an absent/empty model contributes zero images and is reported as such
   in `ConversionStats`, not treated as an error.
2. **Some annotations reference images that don't exist anywhere in the
   copy** (`DJIPhantom/labels/evening/csv.csv` has 1200 rows; zero image
   files exist under `DJIPhantom/images/Evening/`). These rows are
   skipped and counted as `num_orphaned_annotations`, never fabricated
   into a training target for a nonexistent file.
3. **One scenario's annotations are in a different, unsupported format**:
   `DJIPhantom/labels/sunny/csv.xlsx` (Excel, not `csv.csv`) — the only
   non-CSV annotation file found anywhere in the dataset. Not parsed (no
   spreadsheet-parsing dependency is added for one outlier covering 203
   images); `DJIPhantom`/`Sunny` is reported with
   `annotation_format_available="xlsx (not ingested)"` and contributes
   images-on-disk but zero annotated images to the converted output.

Images present on disk with zero matching CSV rows are kept as valid
hard-negative samples (0 boxes), consistent with `detector/datasets/
loader.py`'s convention — though note (recorded here rather than
asserted as fact) this dataset gives no way to distinguish "no drone
visible in this frame" from "this frame was simply never labeled";
treat that gap as unresolved, not as confirmed hard negatives.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TARGET_CLASS = "drone"
SMALL_OBJECT_AREA_PX = 1024.0  # COCO's small-object convention (32*32) — matches eval/metrics.py's default

# The six UAV models VisioDECT documents (per this converter's own real
# run against the copy it was verified against — see docs/datasets.md for
# which ones actually had data). Walked unconditionally so a maintainer's
# fuller copy is picked up automatically without a code change.
UAV_MODELS = ["Anafi-Extended", "DJIFPV", "DJIPhantom", "EFT-E410S", "Mavic_Air", "Mavic_Enterprise"]
# (image folder name, label folder name) — VisioDECT's own casing differs between them.
SCENARIOS = [("Evening", "evening"), ("Cloudy", "cloudy"), ("Sunny", "sunny")]

_BOX_SIZE_BUCKETS = [
    ("<16x16 (<256px²)", 0.0, 256.0),
    ("16x16-32x32 (256-1024px²)", 256.0, 1024.0),
    ("32x32-64x64 (1024-4096px²)", 1024.0, 4096.0),
    ("64x64-128x128 (4096-16384px²)", 4096.0, 16384.0),
    (">=128x128 (>=16384px²)", 16384.0, float("inf")),
]


class VisioDectFormatError(ValueError):
    """Raised when a VisioDECT csv.csv row doesn't match the expected schema."""


@dataclass
class ModelScenarioStats:
    model: str
    scenario: str
    num_images_on_disk: int
    num_images_annotated: int  # images with >=1 valid (non-degenerate, on-disk) box
    num_drone_instances: int
    num_orphaned_annotations: int  # csv rows whose image file does not exist on disk
    num_degenerate_boxes_skipped: int
    annotation_format_available: str  # "csv", "xlsx (not ingested)", or "none"


@dataclass
class ConversionStats:
    per_model_scenario: list[ModelScenarioStats]
    num_images: int
    num_drone_instances: int
    num_hard_negative_images: int
    num_orphaned_annotations: int
    num_degenerate_boxes_skipped: int
    box_size_distribution: dict[str, int]
    small_object_area_threshold_px: float = SMALL_OBJECT_AREA_PX

    def as_dict(self) -> dict:
        return asdict(self)

    def print_summary(self) -> None:
        print("=== VisioDECT conversion summary (per model x scenario) ===")
        for s in self.per_model_scenario:
            print(
                f"  {s.model:18s} {s.scenario:8s}  on_disk={s.num_images_on_disk:5d}  "
                f"annotated={s.num_images_annotated:5d}  drone_instances={s.num_drone_instances:5d}  "
                f"orphaned_annotations={s.num_orphaned_annotations:4d}  "
                f"degenerate_skipped={s.num_degenerate_boxes_skipped:3d}  "
                f"format={s.annotation_format_available}"
            )
        print(
            f"\nTOTAL: images={self.num_images}  drone_instances={self.num_drone_instances}  "
            f"hard_negative_images={self.num_hard_negative_images}  "
            f"orphaned_annotations={self.num_orphaned_annotations}  "
            f"degenerate_boxes_skipped={self.num_degenerate_boxes_skipped}"
        )
        print("\nBox-size distribution (of all drone instances, by area):")
        for label, count in self.box_size_distribution.items():
            pct = (count / self.num_drone_instances * 100) if self.num_drone_instances else 0.0
            print(f"  {label:32s} {count:5d}  ({pct:.1f}%)")


def _bucket_for_area(area: float) -> str:
    for label, lo, hi in _BOX_SIZE_BUCKETS:
        if lo <= area < hi:
            return label
    return _BOX_SIZE_BUCKETS[-1][0]


@dataclass
class _ParsedCsv:
    boxes_by_file: dict[str, list[tuple[float, float, float, float]]] = field(default_factory=dict)
    size_by_file: dict[str, tuple[int, int]] = field(default_factory=dict)


def _parse_csv_rows(csv_path: Path) -> _ParsedCsv:
    """Groups rows by file_name (a handful of real images in this dataset
    have more than one box) and keeps each row's own img_width/img_height
    — real per-image values already present in the source data, so the
    converter never needs to open an annotated image just to learn its
    dimensions.
    """
    parsed = _ParsedCsv()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv_module.reader(handle):
            if len(row) != 8:
                raise VisioDectFormatError(f"{csv_path}: expected 8 fields, got {len(row)}: {row}")
            _class_name, xmin, ymin, w, h, file_name, img_w, img_h = row
            try:
                box = (float(xmin), float(ymin), float(w), float(h))
                size = (int(img_w), int(img_h))
            except ValueError as exc:
                raise VisioDectFormatError(f"{csv_path}: non-numeric field in row {row}") from exc
            parsed.boxes_by_file.setdefault(file_name, []).append(box)
            parsed.size_by_file[file_name] = size
    return parsed


def _read_image_size(image_path: Path) -> tuple[int, int]:
    """Only called for images with zero CSV rows (so their size can't be
    read from the annotation data itself) — lazily imported so converting
    the (usually large majority) annotated portion never requires Pillow.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            f"'{image_path}' has no annotation row to read its dimensions from, and the "
            f"optional 'Pillow' package (pulled in by the torchvision extra) isn't "
            f"installed to open it directly. Install it with `pip install torch torchvision` "
            f"or `pip install Pillow`."
        ) from exc
    with Image.open(image_path) as img:
        return img.size


def _convert_one_combo(
    model: str, image_scenario: str, label_scenario: str, raw_dir: Path, out_images_dir: Path, next_image_id: int,
) -> tuple[ModelScenarioStats, list[dict], list[dict], int]:
    """Returns (stats, images_out, annotations_out, next_image_id)."""
    images_dir = raw_dir / model / "images" / image_scenario
    labels_dir = raw_dir / model / "labels" / label_scenario

    image_files = sorted(images_dir.glob("*.jpg")) if images_dir.is_dir() else []

    csv_path = labels_dir / "csv.csv"
    if csv_path.exists():
        annotation_format = "csv"
        parsed = _parse_csv_rows(csv_path)
    else:
        parsed = _ParsedCsv()
        non_csv = list(labels_dir.glob("csv.*")) if labels_dir.is_dir() else []
        annotation_format = f"{non_csv[0].suffix.lstrip('.')} (not ingested)" if non_csv else "none"

    images_out: list[dict] = []
    annotations_out: list[dict] = []
    num_drone_instances = 0
    num_images_annotated = 0
    num_degenerate_skipped = 0

    on_disk_names = {p.name for p in image_files}
    for image_path in image_files:
        image_id = next_image_id
        next_image_id += 1

        if image_path.name in parsed.size_by_file:
            width, height = parsed.size_by_file[image_path.name]
        else:
            width, height = _read_image_size(image_path)

        images_out.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        symlink_path = out_images_dir / image_path.name
        if not symlink_path.exists():
            symlink_path.symlink_to(image_path.resolve())

        has_box = False
        for xmin, ymin, w, h in parsed.boxes_by_file.get(image_path.name, []):
            if w <= 0 or h <= 0:
                num_degenerate_skipped += 1
                continue
            annotations_out.append(
                {"image_id": image_id, "bbox": [xmin, ymin, w, h], "category": TARGET_CLASS}
            )
            num_drone_instances += 1
            has_box = True
        if has_box:
            num_images_annotated += 1

    num_orphaned = sum(1 for file_name in parsed.boxes_by_file if file_name not in on_disk_names)

    stats = ModelScenarioStats(
        model=model,
        scenario=image_scenario,
        num_images_on_disk=len(image_files),
        num_images_annotated=num_images_annotated,
        num_drone_instances=num_drone_instances,
        num_orphaned_annotations=num_orphaned,
        num_degenerate_boxes_skipped=num_degenerate_skipped,
        annotation_format_available=annotation_format,
    )
    return stats, images_out, annotations_out, next_image_id


def convert_dataset(raw_dir: str | Path, output_dir: str | Path) -> ConversionStats:
    """Convert the whole VisioDECT tree (all models, all scenarios) into
    one combined unified-schema directory. There is no official train/
    val/test split shipped with this dataset, unlike DUT Anti-UAV, so
    (per this task's scope) this produces one directory, not a per-split
    one — splitting is a decision for whoever actually trains on it.

    Raises `FileNotFoundError` if `raw_dir` itself doesn't exist, or
    `VisioDectFormatError` if a `csv.csv` file doesn't match the expected
    8-column schema. An individual model/scenario with no images or no
    label file at all is not an error — see the module docstring.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"VisioDECT raw directory not found: {raw_dir}")

    output_dir = Path(output_dir)
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    per_model_scenario: list[ModelScenarioStats] = []
    next_image_id = 1

    for model in UAV_MODELS:
        for image_scenario, label_scenario in SCENARIOS:
            stats, images_out, annotations_out, next_image_id = _convert_one_combo(
                model, image_scenario, label_scenario, raw_dir, out_images_dir, next_image_id
            )
            per_model_scenario.append(stats)
            all_images.extend(images_out)
            all_annotations.extend(annotations_out)

    (output_dir / "annotations.json").write_text(
        json.dumps({"images": all_images, "annotations": all_annotations}, indent=2), encoding="utf-8"
    )

    num_drone_instances = sum(s.num_drone_instances for s in per_model_scenario)
    annotated_image_ids = {a["image_id"] for a in all_annotations}
    box_size_distribution = {label: 0 for label, _, _ in _BOX_SIZE_BUCKETS}
    for ann in all_annotations:
        _x, _y, w, h = ann["bbox"]
        box_size_distribution[_bucket_for_area(w * h)] += 1

    stats = ConversionStats(
        per_model_scenario=per_model_scenario,
        num_images=len(all_images),
        num_drone_instances=num_drone_instances,
        num_hard_negative_images=len(all_images) - len(annotated_image_ids),
        num_orphaned_annotations=sum(s.num_orphaned_annotations for s in per_model_scenario),
        num_degenerate_boxes_skipped=sum(s.num_degenerate_boxes_skipped for s in per_model_scenario),
        box_size_distribution=box_size_distribution,
    )
    (output_dir / "conversion_stats.json").write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="Path to the VisioDECT root (contains one dir per UAV model)")
    parser.add_argument("--output-dir", required=True, help="Where to write images/ + annotations.json")
    args = parser.parse_args()

    stats = convert_dataset(args.raw_dir, args.output_dir)
    stats.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
