#!/usr/bin/env python3
"""Auto-annotation (pre-labeling) tool — NOT a substitute for human review.

Runs a chosen detector backend over a video file, a directory of images,
or a clip directory from tools/ingest_footage.py, and emits its detections
as pre-labels for human correction, in two import formats:

- `coco_predictions.json`: COCO instances, shaped like CVAT's own "COCO 1.0"
  export, so CVAT (and other COCO tools) import it as-is.
- `labelstudio_tasks.json` + `labelstudio_config.xml`: Label Studio tasks
  carrying the boxes as *predictions* (Label Studio's own pre-annotation
  slot — shown to the annotator, never counted as annotations).

For a clip directory, frames are read in place (not copied) and output goes
to `<clip>/prelabels/` unless --output-dir is given.

Every emitted box is a detector *prediction*, not a verified ground-truth
label — this tool does not label data itself. Each annotation carries a
"score" field (the detector's confidence) so a reviewer can see how
confident the model was, and every image should be reviewed before any
of these boxes are treated as ground truth. Reviewed labels come back
into the dataset only through tools/import_reviewed.py, which reads human
annotations and never these predictions. See docs/data-capture-runbook.md.

Pre-label quality with today's best detector is low: dut_v1 at the 0.35
threshold emits ~144 boxes/frame on DUT val, and its top 3 per frame cover
13% of drones (18% with 3x3 tiling). Use --max-per-frame so the annotator
deletes a few boxes rather than a hundred; expect pre-labels to start
saving real time only once a detector has been retrained on our footage.

Usage:
    python tools/preannotate.py --source path/to/video.mp4 --output-dir path/to/output
    python tools/preannotate.py --source path/to/frames_dir --output-dir path/to/output --detector torchvision
    python tools/preannotate.py --source path/to/video.mp4 --output-dir out --preset fast --frame-stride 5
    python tools/preannotate.py --source data/own-fpv/clips/<clip_id> --detector drone \
        --weights models/dut_v1/weights.pt --max-per-frame 3 [--tile-rows 3 --tile-cols 3]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from config.loader import ConfigError, get_preset  # noqa: E402
from detector.factory import build_detector  # noqa: E402
from footage_meta import CLIP_META_FILENAME, FRAMES_FILENAME, PRELABELS_DIRNAME  # noqa: E402

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Always offered to the annotator, in this order (so "drone" is category 1),
# whatever the detector emits: non-drone classes are the loader's labeled
# hard negatives (detector/datasets/loader.py). Detector classes outside this
# list (e.g. the motion backend's "moving_object") are appended after.
REVIEW_LABELS = ["drone", "bird", "aircraft", "other"]

PRELABEL_NOTICE = (
    "Auto-generated pre-labels from tools/preannotate.py — detector "
    "predictions, NOT human-verified ground truth. Review every box "
    "before treating this as a training or eval label."
)
LABEL_STUDIO_URL_PREFIX = "/data/local-files/?d="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Video file, directory of images, or an ingested clip directory")
    parser.add_argument("--output-dir", default=None, help="Where to write pre-labels (required unless --source is a clip directory)")
    parser.add_argument("--preset", default="default", help="Named config preset (see config/presets.yaml)")
    parser.add_argument("--detector", default=None, choices=["motion", "ultralytics", "torchvision", "drone"], help="Override the preset's detector backend")
    parser.add_argument("--weights", default=None, help="Weights path, required if --detector ultralytics or drone")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Override the detector's confidence threshold")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame of a video (default: every frame)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames")
    parser.add_argument("--max-per-frame", type=int, default=None, help="Keep only the N highest-scoring boxes per frame")
    parser.add_argument("--tile-rows", type=int, default=None, help="Tiled inference rows (drone backend; see detector/tiling.py)")
    parser.add_argument("--tile-cols", type=int, default=None, help="Tiled inference columns (drone backend)")
    parser.add_argument("--tile-overlap", type=float, default=None, help="Tile overlap fraction (drone backend; default 0.2)")
    parser.add_argument("--input-size", type=int, default=None, help="Model input size the weights were trained at (drone backend)")
    parser.add_argument(
        "--ls-document-root",
        default=None,
        help="Label Studio LOCAL_FILES_DOCUMENT_ROOT that image URLs are made relative to "
        "(default: the clip's parent dir for a clip source, else --output-dir)",
    )
    return parser.parse_args()


def _iter_video_frames(source: str, frame_stride: int, max_frames: int | None):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_index = 0
    processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                yield f"frame_{frame_index:06d}.jpg", frame
                processed += 1
                if max_frames and processed >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()


def _iter_image_dir_frames(source_dir: Path, max_frames: int | None):
    paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    for count, path in enumerate(paths):
        if max_frames and count >= max_frames:
            break
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Skipping unreadable image: {path}", file=sys.stderr)
            continue
        yield path.name, frame


def build_detector_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = get_preset(args.preset)
    detector_config = dict(preset.detector)
    if args.detector:
        detector_config["backend"] = args.detector
    if detector_config.get("backend") in ("ultralytics", "drone"):
        weights_path = args.weights or detector_config.get("weights_path")
        if not weights_path:
            raise ValueError(f"--weights is required when --detector {detector_config['backend']}")
        detector_config["weights_path"] = weights_path
    if args.confidence_threshold is not None:
        detector_config["confidence_threshold"] = args.confidence_threshold
    for key in ("tile_rows", "tile_cols", "tile_overlap", "input_size"):
        value = getattr(args, key, None)
        if value is not None:
            if detector_config.get("backend") != "drone":
                raise ValueError(f"--{key.replace('_', '-')} only applies to --detector drone")
            detector_config[key] = value
    return detector_config


def _iter_clip_frames(clip_dir: Path, max_frames: int | None):
    """Frames of an ingested clip, in frames.json order, read in place."""
    entries = json.loads((clip_dir / FRAMES_FILENAME).read_text(encoding="utf-8"))["images"]
    for count, entry in enumerate(entries):
        if max_frames and count >= max_frames:
            break
        frame = cv2.imread(str(clip_dir / "images" / entry["file_name"]))
        if frame is None:
            print(f"Skipping unreadable image: {entry['file_name']}", file=sys.stderr)
            continue
        yield entry["file_name"], frame


def is_clip_dir(path: Path) -> bool:
    return path.is_dir() and (path / CLIP_META_FILENAME).exists() and (path / FRAMES_FILENAME).exists()


def labelstudio_config(labels: list[str]) -> str:
    """Label Studio labeling config matching the tasks' from_name/to_name."""
    label_tags = "\n".join(f'    <Label value="{name}"/>' for name in labels)
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{label_tags}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def to_labelstudio_tasks(coco: dict[str, Any], image_urls: dict[str, str], model_version: str) -> list[dict[str, Any]]:
    """COCO pre-labels -> Label Studio tasks with boxes under "predictions"
    (percent coordinates), never under "annotations"."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    tasks = []
    for image in coco["images"]:
        width, height = image["width"], image["height"]
        results = []
        for ann in by_image.get(image["id"], []):
            x, y, w, h = ann["bbox"]
            results.append(
                {
                    "id": f"p{ann['id']}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "score": ann["score"],
                    "value": {
                        "x": 100.0 * x / width,
                        "y": 100.0 * y / height,
                        "width": 100.0 * w / width,
                        "height": 100.0 * h / height,
                        "rotation": 0,
                        "rectanglelabels": [names[ann["category_id"]]],
                    },
                }
            )
        tasks.append(
            {
                "data": {"image": image_urls[image["file_name"]], "file_name": image["file_name"]},
                "predictions": [
                    {
                        "model_version": model_version,
                        "score": max((r["score"] for r in results), default=0.0),
                        "result": results,
                    }
                ],
            }
        )
    return tasks


def preannotate(
    source: str,
    output_dir: Path | None,
    detector_config: dict[str, Any],
    frame_stride: int = 1,
    max_frames: int | None = None,
    max_per_frame: int | None = None,
    ls_document_root: Path | None = None,
) -> dict[str, Any]:
    """Run `detector_config`'s backend over `source`, writing COCO and Label
    Studio pre-label files into `output_dir`.

    A video or image-directory source has its frames written to
    `output_dir/images/`; an ingested clip directory is read in place and
    `output_dir` defaults to `<clip>/prelabels/`.

    Returns the COCO dict that was written (useful for tests).
    """
    source_path = Path(source)
    clip_mode = is_clip_dir(source_path)
    if output_dir is None:
        if not clip_mode:
            raise ValueError("--output-dir is required unless --source is an ingested clip directory")
        output_dir = source_path / PRELABELS_DIRNAME
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if clip_mode:
        images_dir = source_path / "images"
        frame_iter = _iter_clip_frames(source_path, max_frames)
        ls_document_root = Path(ls_document_root) if ls_document_root else source_path.parent
    else:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            frame_iter = _iter_image_dir_frames(source_path, max_frames)
        else:
            frame_iter = _iter_video_frames(source, frame_stride, max_frames)
        ls_document_root = Path(ls_document_root) if ls_document_root else output_dir

    detector = build_detector(detector_config)

    category_ids: dict[str, int] = {name: i + 1 for i, name in enumerate(REVIEW_LABELS)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_urls: dict[str, str] = {}
    next_image_id = 1
    next_annotation_id = 1

    for file_name, frame in frame_iter:
        height, width = frame.shape[:2]
        if not clip_mode:
            cv2.imwrite(str(images_dir / file_name), frame)
        image_path = (images_dir / file_name).resolve()
        try:
            image_urls[file_name] = LABEL_STUDIO_URL_PREFIX + image_path.relative_to(ls_document_root.resolve()).as_posix()
        except ValueError:
            image_urls[file_name] = LABEL_STUDIO_URL_PREFIX + image_path.as_posix()

        image_id = next_image_id
        next_image_id += 1
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        detections = sorted(detector.detect(frame), key=lambda d: d.confidence, reverse=True)
        if max_per_frame is not None:
            detections = detections[:max_per_frame]
        for detection in detections:
            if detection.class_name not in category_ids:
                category_ids[detection.class_name] = len(category_ids) + 1

            x1 = detection.x1 * width
            y1 = detection.y1 * height
            box_w = (detection.x2 - detection.x1) * width
            box_h = (detection.y2 - detection.y1) * height

            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[detection.class_name],
                    "segmentation": [],
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "attributes": {"occluded": False},
                    "score": detection.confidence,
                }
            )
            next_annotation_id += 1

    backend_name = getattr(detector, "name", type(detector).__name__)
    model_version = backend_name
    if detector_config.get("weights_path"):
        model_version += f" ({detector_config['weights_path']})"
    if detector_config.get("tile_rows", 1) * detector_config.get("tile_cols", 1) > 1:
        model_version += f" tiled {detector_config['tile_rows']}x{detector_config['tile_cols']}"

    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "description": PRELABEL_NOTICE,
            "review_status": "unreviewed_predictions",
            "detector_backend": backend_name,
            "model_version": model_version,
            "max_per_frame": max_per_frame,
            "source": str(source),
        },
        "categories": [{"id": cat_id, "name": name, "supercategory": ""} for name, cat_id in category_ids.items()],
        "images": images,
        "annotations": annotations,
    }

    (output_dir / "coco_predictions.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    tasks = to_labelstudio_tasks(coco, image_urls, model_version)
    (output_dir / "labelstudio_tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (output_dir / "labelstudio_config.xml").write_text(labelstudio_config(list(category_ids)), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        PRELABEL_NOTICE
        + "\n\nCVAT: Actions > Upload annotations > COCO 1.0 > coco_predictions.json\n"
        "Label Studio: Settings > Labeling Interface > paste labelstudio_config.xml; "
        "then Import labelstudio_tasks.json\n"
        "After review, bring labels back with tools/import_reviewed.py (see docs/data-capture-runbook.md).\n",
        encoding="utf-8",
    )
    return coco


def main() -> int:
    args = parse_args()

    try:
        detector_config = build_detector_config(args)
    except (ConfigError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    ls_root = Path(args.ls_document_root) if args.ls_document_root else None

    try:
        coco = preannotate(
            args.source, output_dir, detector_config, args.frame_stride, args.max_frames, args.max_per_frame, ls_root
        )
    except (RuntimeError, ImportError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    written_to = output_dir or Path(args.source) / PRELABELS_DIRNAME
    num_images = len(coco["images"])
    print(f"Wrote {num_images} images and {len(coco['annotations'])} pre-label annotations to {written_to}")
    print(f"Categories: {[c['name'] for c in coco['categories']]}")
    if num_images and args.max_per_frame is None and len(coco["annotations"]) / num_images > 10:
        print(
            f"Warning: {len(coco['annotations']) / num_images:.0f} boxes/frame — consider --max-per-frame "
            "so reviewers aren't deleting a flood of boxes.",
            file=sys.stderr,
        )
    print("These are unverified detector predictions — review before use as ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
