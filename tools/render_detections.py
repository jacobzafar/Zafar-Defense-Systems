#!/usr/bin/env python3
"""Render detector predictions + ground truth onto real images — visual
proof, not a metric.

Runs a chosen detector backend over images from a unified-schema dataset
(detector/datasets/loader.py) or a frozen eval set (eval/schema.py),
draws ground-truth boxes in one color and predicted boxes (with
confidence) in another onto a copy of each image, writes the annotated
frames to an output directory, and stitches them into an MP4 if `ffmpeg`
is on PATH.

**Strictly read-only with respect to whatever it's pointed at.** This
only calls `detector.detect()` (inference) and `cv2.imwrite()` (drawing
output elsewhere) — it never writes back to the source dataset/eval set
and never trains anything. Pointing it at a frozen eval set does not
violate eval/README.md's "never train on the eval set" rule; nothing
here calls detector/train.py or touches model weights.

Frame selection: `--selection sequential` takes the first N images in
directory order; `--selection extremes` instead takes the N/2 images
with the largest ground-truth box (by pixel area) and the N/2 with the
smallest, deliberately avoiding a cherry-picked "easy" sample — this is
meant to show both where a detector succeeds and where it fails.

Usage:
    python tools/render_detections.py \
        --source-type eval_set --source-dir data/frozen_eval/dut-anti-uav-test \
        --detector drone --weights models/dut_v1/weights.pt \
        --output-dir renders/dut_test_sample --num-images 100 --selection extremes
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from detector.datasets.loader import load_manifest_dataset  # noqa: E402
from detector.factory import build_detector  # noqa: E402
from eval.schema import load_eval_manifest  # noqa: E402

_GT_COLOR = (60, 200, 60)  # BGR green
_PRED_COLOR = (0, 140, 255)  # BGR orange
_CANVAS_SIZE = (960, 720)  # (width, height) — DUT images vary wildly in resolution (160x240 to 3744x5616);
# a fixed, letterboxed canvas is what makes ffmpeg able to stitch them into one video stream at all.


@dataclass
class FrameRecord:
    label: str
    image_path: Path
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def max_box_area(self) -> float:
        return max(((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in self.boxes), default=0.0)


def load_dataset_frames(source_dir: str | Path, target_class: str) -> list[FrameRecord]:
    samples = load_manifest_dataset(source_dir)
    return [
        FrameRecord(
            label=sample.image_path.stem,
            image_path=sample.image_path,
            boxes=[(b.x1, b.y1, b.x2, b.y2) for b in sample.boxes if b.category == target_class],
        )
        for sample in samples
    ]


def load_eval_set_frames(source_dir: str | Path, target_class: str) -> list[FrameRecord]:
    sequences = load_eval_manifest(source_dir)
    records = []
    for sequence in sequences:
        for frame in sequence.frames:
            records.append(
                FrameRecord(
                    label=f"{sequence.sequence_id}_{frame.frame_index:04d}",
                    image_path=frame.image_path,
                    boxes=[(b.x1, b.y1, b.x2, b.y2) for b in frame.boxes if b.category == target_class],
                )
            )
    return records


def select_frames(frames: list[FrameRecord], num_images: int, selection: str) -> list[FrameRecord]:
    if selection == "sequential":
        return frames[:num_images]

    # "extremes": half the smallest ground-truth boxes, half the largest —
    # excludes frames with no target-class box at all (nothing to rank by).
    with_boxes = [f for f in frames if f.boxes]
    ranked = sorted(with_boxes, key=lambda f: f.max_box_area)
    half = num_images // 2
    remainder = num_images - half
    smallest = ranked[:half]
    largest = ranked[-remainder:] if remainder > 0 else []

    seen: set[Path] = set()
    selected: list[FrameRecord] = []
    for record in smallest + largest:
        if record.image_path not in seen:
            seen.add(record.image_path)
            selected.append(record)
    return selected


def _letterbox(image, target_size: tuple[int, int]):
    target_w, target_h = target_size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = cv2.copyMakeBorder(
        resized,
        top=(target_h - new_h) // 2,
        bottom=target_h - new_h - (target_h - new_h) // 2,
        left=(target_w - new_w) // 2,
        right=target_w - new_w - (target_w - new_w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    return canvas


def _draw_frame(image, gt_boxes: list[tuple[float, float, float, float]], detections: list[Any]):
    annotated = image.copy()
    height, width = annotated.shape[:2]

    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), _GT_COLOR, 2)

    for det in detections:
        x1, y1 = int(det.x1 * width), int(det.y1 * height)
        x2, y2 = int(det.x2 * width), int(det.y2 * height)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _PRED_COLOR, 2)
        label = f"{det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 6)), (x1 + text_w + 4, y1), _PRED_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 2, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(annotated, "GT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(annotated, "Pred", (60, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _PRED_COLOR, 2, cv2.LINE_AA)
    return annotated


def render_detections(
    frames: list[FrameRecord],
    detector_config: dict[str, Any],
    output_dir: str | Path,
    fps: int = 1,
) -> dict[str, Any]:
    """Run inference + drawing over `frames`, writing annotated JPEGs (and
    an MP4 if `ffmpeg` is on PATH) into `output_dir`. Returns a summary
    dict — also written as output_dir/manifest.json — with per-frame
    counts and the resolved output paths.
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(detector_config)

    manifest_frames = []
    total_gt = 0
    total_pred = 0

    for index, record in enumerate(frames, start=1):
        image = cv2.imread(str(record.image_path))
        if image is None:
            print(f"Skipping unreadable image: {record.image_path}", file=sys.stderr)
            continue

        detections = detector.detect(image)
        annotated = _draw_frame(image, record.boxes, detections)
        annotated = _letterbox(annotated, _CANVAS_SIZE)

        frame_name = f"frame_{index:04d}.jpg"
        cv2.imwrite(str(frames_dir / frame_name), annotated)

        total_gt += len(record.boxes)
        total_pred += len(detections)
        manifest_frames.append(
            {
                "frame_file": frame_name,
                "label": record.label,
                "source_image": str(record.image_path),
                "num_gt_boxes": len(record.boxes),
                "num_pred_boxes": len(detections),
                "gt_box_area_px": record.max_box_area,
            }
        )

    video_path = None
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin and manifest_frames:
        video_path = output_dir / "detections.mp4"
        result = subprocess.run(
            [
                ffmpeg_bin, "-y", "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%04d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(video_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ffmpeg failed (exit {result.returncode}): {result.stderr[-500:]}", file=sys.stderr)
            video_path = None

    summary = {
        "detector_backend": getattr(detector, "name", type(detector).__name__),
        "num_frames_rendered": len(manifest_frames),
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "frames_dir": str(frames_dir),
        "video_path": str(video_path) if video_path else None,
        "ffmpeg_available": ffmpeg_bin is not None,
        "frames": manifest_frames,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-type", required=True, choices=["dataset", "eval_set"])
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--detector", required=True, choices=["motion", "ultralytics", "torchvision", "drone"])
    parser.add_argument("--weights", default=None, help="Required for --detector ultralytics or drone")
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--target-class", default="drone", help="Ground-truth category to draw/rank by")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--selection", choices=["sequential", "extremes"], default="sequential")
    parser.add_argument("--fps", type=int, default=1, help="Frame rate for the stitched MP4, if produced")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    detector_config: dict[str, Any] = {"backend": args.detector}
    if args.detector in ("ultralytics", "drone"):
        if not args.weights:
            print(f"--weights is required for --detector {args.detector}", file=sys.stderr)
            return 1
        detector_config["weights_path"] = args.weights
    if args.confidence_threshold is not None:
        detector_config["confidence_threshold"] = args.confidence_threshold

    loader = load_dataset_frames if args.source_type == "dataset" else load_eval_set_frames
    try:
        frames = loader(args.source_dir, args.target_class)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not frames:
        print(f"No images found in {args.source_dir}", file=sys.stderr)
        return 1

    selected = select_frames(frames, args.num_images, args.selection)

    try:
        summary = render_detections(selected, detector_config, args.output_dir, fps=args.fps)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Rendered {summary['num_frames_rendered']} frames ({summary['total_gt_boxes']} GT boxes, "
          f"{summary['total_pred_boxes']} predicted boxes) to {summary['frames_dir']}")
    if summary["video_path"]:
        print(f"MP4 written to {summary['video_path']}")
    elif not summary["ffmpeg_available"]:
        print("ffmpeg not found on PATH — frames written, no MP4 produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
