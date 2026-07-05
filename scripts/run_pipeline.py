#!/usr/bin/env python3
"""Headless CLI runner: video in -> detect -> track -> log, no UI required.

Usage:
    python scripts/run_pipeline.py --source path/to/video.mp4
    python scripts/run_pipeline.py --source 0                     # webcam
    python scripts/run_pipeline.py --source video.mp4 --save-video out.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from control.pipeline import Pipeline  # noqa: E402
from detector.factory import build_detector  # noqa: E402
from telemetry.logger import EventLogger  # noqa: E402
from tracker.factory import build_tracker  # noqa: E402
from ui.overlay import draw_tracks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the detection/tracking pipeline headlessly.")
    parser.add_argument("--source", required=True, help="Video file path, RTSP URL, or webcam index (e.g. 0)")
    parser.add_argument("--detector", default="motion", choices=["motion", "ultralytics", "torchvision"])
    parser.add_argument("--tracker", default="iou", choices=["iou", "bytetrack"])
    parser.add_argument("--weights", default=None, help="Weights path, required if --detector ultralytics")
    parser.add_argument("--save-video", default=None, help="Optional path to write an annotated output video")
    parser.add_argument("--log-dir", default="logs", help="Directory for JSONL run logs")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for smoke tests)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source: str | int = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    detector_config = {"backend": args.detector}
    if args.detector == "ultralytics":
        if not args.weights:
            print("--weights is required when --detector ultralytics", file=sys.stderr)
            return 1
        detector_config["weights_path"] = args.weights

    detector = build_detector(detector_config)
    tracker = build_tracker({"backend": args.tracker})
    logger = EventLogger(log_dir=args.log_dir)

    pipeline = Pipeline(detector=detector, tracker=tracker, logger=logger)

    video_writer = None
    capture_for_dims = cv2.VideoCapture(source)
    frame_w = int(capture_for_dims.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(capture_for_dims.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = capture_for_dims.get(cv2.CAP_PROP_FPS) or 25.0
    capture_for_dims.release()

    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, (frame_w, frame_h))

    raw_capture = cv2.VideoCapture(source)

    print(f"[run_pipeline] source={args.source} detector={args.detector} tracker={args.tracker}")
    print(f"[run_pipeline] logging to {logger.log_path}")

    frame_count = 0
    try:
        for result in pipeline.run(source):
            print(
                f"frame={result.frame_index:05d} "
                f"detections={result.detection_count} "
                f"tracks={len(result.tracks)} "
                f"inference_ms={result.inference_ms} "
                f"dropped={result.dropped}"
                + (f" error={result.error}" if result.error else "")
            )

            if video_writer is not None:
                ok, raw_frame = raw_capture.read()
                if ok:
                    annotated = draw_tracks(raw_frame, result.tracks)
                    video_writer.write(annotated)

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                pipeline.stop()
                break
    finally:
        raw_capture.release()
        if video_writer is not None:
            video_writer.release()
        logger.close()

    print(f"[run_pipeline] done. {frame_count} frames processed. Log: {logger.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
