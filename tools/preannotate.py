#!/usr/bin/env python3
"""Auto-annotation (pre-labeling) tool — NOT a substitute for human review.

Runs a chosen detector backend over a video file or a directory of images
and emits its detections as pre-labels in standard COCO instances format
(images + categories + annotations), ready to import into CVAT, Label
Studio, or any COCO-compatible annotation tool for human correction.

Every emitted box is a detector *prediction*, not a verified ground-truth
label — this tool does not label data itself. Each annotation carries a
"score" field (the detector's confidence) so a reviewer can see how
confident the model was, and every image should be reviewed before any
of these boxes are treated as ground truth. See docs/coverage-matrix.md
for tracking what footage conditions still need coverage once real
labeled data starts accumulating.

Usage:
    python tools/preannotate.py --source path/to/video.mp4 --output-dir path/to/output
    python tools/preannotate.py --source path/to/frames_dir --output-dir path/to/output --detector torchvision
    python tools/preannotate.py --source path/to/video.mp4 --output-dir out --preset fast --frame-stride 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from config.loader import ConfigError, get_preset  # noqa: E402
from detector.factory import build_detector  # noqa: E402

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Video file path, or a directory of image files")
    parser.add_argument("--output-dir", required=True, help="Where to write images/ + coco_predictions.json")
    parser.add_argument("--preset", default="default", help="Named config preset (see config/presets.yaml)")
    parser.add_argument("--detector", default=None, choices=["motion", "ultralytics", "torchvision", "drone"], help="Override the preset's detector backend")
    parser.add_argument("--weights", default=None, help="Weights path, required if --detector ultralytics or drone")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Override the detector's confidence threshold")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame of a video (default: every frame)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames")
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
    return detector_config


def preannotate(source: str, output_dir: Path, detector_config: dict[str, Any], frame_stride: int = 1, max_frames: int | None = None) -> dict[str, Any]:
    """Run `detector_config`'s backend over `source`, writing a COCO-format
    predictions file + copied/extracted image frames into `output_dir`.

    Returns the COCO dict that was written (useful for tests).
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(detector_config)

    source_path = Path(source)
    if source_path.is_dir():
        frame_iter = _iter_image_dir_frames(source_path, max_frames)
    else:
        frame_iter = _iter_video_frames(source, frame_stride, max_frames)

    category_ids: dict[str, int] = {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    for file_name, frame in frame_iter:
        height, width = frame.shape[:2]
        cv2.imwrite(str(images_dir / file_name), frame)

        image_id = next_image_id
        next_image_id += 1
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        for detection in detector.detect(frame):
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
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "score": detection.confidence,
                }
            )
            next_annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cat_id, "name": name} for name, cat_id in category_ids.items()],
        "info": {
            "description": (
                "Auto-generated pre-labels from tools/preannotate.py — detector "
                "predictions, NOT human-verified ground truth. Review every box "
                "before treating this as a training or eval label."
            ),
            "detector_backend": getattr(detector, "name", type(detector).__name__),
        },
    }

    output_path = output_dir / "coco_predictions.json"
    output_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return coco


def main() -> int:
    args = parse_args()

    try:
        detector_config = build_detector_config(args)
    except (ConfigError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)

    try:
        coco = preannotate(args.source, output_dir, detector_config, args.frame_stride, args.max_frames)
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote {len(coco['images'])} images and {len(coco['annotations'])} pre-label annotations to {output_dir}")
    print(f"Categories seen: {[c['name'] for c in coco['categories']]}")
    print("These are unverified detector predictions — review before use as ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
