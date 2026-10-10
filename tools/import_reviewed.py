#!/usr/bin/env python3
"""Bring human-reviewed labels back from CVAT or Label Studio into a clip.

Reads an annotation tool's export and writes `<clip>/annotations.json` in
the unified schema (detector/datasets/loader.py), which makes the clip
directory a loadable dataset and lets tools/coverage_report.py run its
anchor check on it. Updates clip_meta.json's label_status.

Accepted exports:
- CVAT "COCO 1.0" (`annotations/instances_default.json`, or the bare JSON).
  CVAT does not record which frames a person actually looked at, so this
  trusts the export: mark the CVAT job completed only after every frame
  was reviewed.
- Label Studio "JSON" export. Only `annotations` (human-submitted) are
  read, never `predictions` (the pre-labels); a task with no submitted,
  non-cancelled annotation counts as unreviewed.

Every frame of the clip must be covered, or the import is refused: a frame
missing from annotations.json is merely excluded from training, but a
frame present with no boxes trains as a hard negative, so "partially
reviewed" must not be confused with "reviewed, no drone". --allow-partial
writes only the reviewed frames.

Usage:
    python tools/import_reviewed.py --clip data/own-fpv/clips/<clip_id> --export instances_default.json
    python tools/import_reviewed.py --clip data/own-fpv/clips/<clip_id> --export project-3-export.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from footage_meta import ANNOTATIONS_FILENAME, CLIP_META_FILENAME, FRAMES_FILENAME  # noqa: E402


class ImportReviewedError(ValueError):
    """Raised when an export can't be safely mapped onto the clip."""


def _basename(path_or_url: str) -> str:
    parsed = urlparse(path_or_url)
    if "d" in parse_qs(parsed.query):  # Label Studio local files: /data/local-files/?d=<path>
        path_or_url = parse_qs(parsed.query)["d"][0]
    else:
        path_or_url = parsed.path or path_or_url
    return Path(unquote(path_or_url)).name


def _boxes_from_coco(export: dict) -> dict[str, list[dict]]:
    names = {c["id"]: c["name"] for c in export["categories"]}
    file_by_id = {img["id"]: _basename(img["file_name"]) for img in export["images"]}
    boxes: dict[str, list[dict]] = {name: [] for name in file_by_id.values()}
    for ann in export["annotations"]:
        boxes[file_by_id[ann["image_id"]]].append({"bbox": [float(v) for v in ann["bbox"]], "category": names[ann["category_id"]]})
    return boxes


def _boxes_from_labelstudio(tasks: list) -> dict[str, list[dict]]:
    boxes: dict[str, list[dict]] = {}
    for task in tasks:
        submitted = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not submitted:
            continue  # only predictions, or skipped: not reviewed
        annotation = max(submitted, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        file_name = task["data"].get("file_name") or _basename(task["data"]["image"])
        frame_boxes = []
        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue
            value, width, height = result["value"], result["original_width"], result["original_height"]
            if value.get("rotation"):
                raise ImportReviewedError(f"{file_name}: rotated boxes are not supported by the unified schema")
            frame_boxes.append(
                {
                    "bbox": [
                        value["x"] * width / 100.0,
                        value["y"] * height / 100.0,
                        value["width"] * width / 100.0,
                        value["height"] * height / 100.0,
                    ],
                    "category": value["rectanglelabels"][0],
                }
            )
        boxes[file_name] = frame_boxes
    return boxes


def load_export(export_path: str | Path) -> tuple[str, dict[str, list[dict]]]:
    """(tool name, {frame file name: [{"bbox": xywh, "category": str}]})."""
    raw = json.loads(Path(export_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and {"images", "annotations", "categories"} <= raw.keys():
        if raw.get("info", {}).get("review_status") == "unreviewed_predictions":
            raise ImportReviewedError(f"{export_path} is tools/preannotate.py's own pre-label file, not a reviewed export")
        return "coco", _boxes_from_coco(raw)
    if isinstance(raw, list):
        return "labelstudio", _boxes_from_labelstudio(raw)
    raise ImportReviewedError(f"{export_path}: not a COCO export or a Label Studio JSON export")


def import_reviewed(clip_dir: str | Path, export_path: str | Path, allow_partial: bool = False) -> dict:
    """Write `<clip>/annotations.json` from a reviewed export. Returns it."""
    clip_dir = Path(clip_dir)
    frames = json.loads((clip_dir / FRAMES_FILENAME).read_text(encoding="utf-8"))["images"]
    tool, reviewed = load_export(export_path)

    known = {f["file_name"] for f in frames}
    unknown = sorted(set(reviewed) - known)
    if unknown:
        raise ImportReviewedError(f"Export has {len(unknown)} frame(s) not in this clip, e.g. {unknown[:3]} — wrong clip?")
    missing = [f["file_name"] for f in frames if f["file_name"] not in reviewed]
    if missing and not allow_partial:
        raise ImportReviewedError(
            f"{len(missing)} of {len(frames)} frames have no reviewed labels (e.g. {missing[:3]}). Finish "
            f"review, or pass --allow-partial to write only the reviewed frames."
        )

    images, annotations = [], []
    for frame in frames:
        if frame["file_name"] not in reviewed:
            continue
        images.append({k: frame[k] for k in ("id", "file_name", "width", "height")})
        for box in reviewed[frame["file_name"]]:
            x, y, w, h = box["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(frame["width"]), x + w), min(float(frame["height"]), y + h)
            annotations.append({"image_id": frame["id"], "bbox": [x1, y1, x2 - x1, y2 - y1], "category": box["category"]})

    dataset = {
        "info": {
            "label_source": f"human-reviewed ({tool} export: {Path(export_path).name})",
            "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "images": images,
        "annotations": annotations,
    }
    (clip_dir / ANNOTATIONS_FILENAME).write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    meta_path = clip_dir / CLIP_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["label_status"] = "reviewed" if not missing else "partially_reviewed"
    meta["num_frames_reviewed"] = len(images)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", required=True, help="Ingested clip directory")
    parser.add_argument("--export", required=True, help="CVAT COCO 1.0 JSON or Label Studio JSON export")
    parser.add_argument("--allow-partial", action="store_true", help="Write only the frames that were reviewed")
    args = parser.parse_args(argv)
    try:
        dataset = import_reviewed(args.clip, args.export, args.allow_partial)
    except (ImportReviewedError, FileNotFoundError, KeyError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    drones = sum(a["category"] == "drone" for a in dataset["annotations"])
    print(f"Wrote {len(dataset['images'])} reviewed frames, {drones} drone boxes, to {Path(args.clip) / ANNOTATIONS_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
