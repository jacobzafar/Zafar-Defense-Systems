#!/usr/bin/env python3
"""Headless CLI runner: video in -> detect -> track -> log, no UI required.

Usage:
    python scripts/run_pipeline.py --source path/to/video.mp4
    python scripts/run_pipeline.py --source 0                     # webcam
    python scripts/run_pipeline.py --source demo                  # synthetic demo clip, no file needed
    python scripts/run_pipeline.py --source video.mp4 --preset fast
    python scripts/run_pipeline.py --source video.mp4 --save-video out.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from config.loader import ConfigError, get_preset  # noqa: E402
from control.pipeline import Pipeline  # noqa: E402
from detector.factory import build_detector  # noqa: E402
from telemetry.logger import EventLogger  # noqa: E402
from telemetry.summary import format_summary, summarize_log  # noqa: E402
from tracker.factory import build_tracker  # noqa: E402
from ui.overlay import draw_tracks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the detection/tracking pipeline headlessly.")
    parser.add_argument(
        "--source",
        required=True,
        help='Video file path, RTSP URL, webcam index (e.g. 0), or "demo" for the built-in synthetic clip',
    )
    parser.add_argument("--preset", default="default", help="Named config preset (see config/presets.yaml)")
    parser.add_argument("--detector", default=None, choices=["motion", "ultralytics", "torchvision"], help="Override the preset's detector backend")
    parser.add_argument("--tracker", default=None, choices=["iou", "bytetrack"], help="Override the preset's tracker backend")
    parser.add_argument("--weights", default=None, help="Weights path, required if --detector ultralytics")
    parser.add_argument("--save-video", default=None, help="Optional path to write an annotated output video")
    parser.add_argument("--log-dir", default=None, help="Directory for JSONL run logs (defaults to the preset's)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for smoke tests)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source: str | int = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    try:
        preset = get_preset(args.preset)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    detector_config = dict(preset.detector)
    if args.detector:
        detector_config["backend"] = args.detector
    if detector_config.get("backend") == "ultralytics":
        weights_path = args.weights or detector_config.get("weights_path")
        if not weights_path:
            print("--weights is required when --detector ultralytics", file=sys.stderr)
            return 1
        detector_config["weights_path"] = weights_path

    tracker_config = dict(preset.tracker)
    if args.tracker:
        tracker_config["backend"] = args.tracker

    log_dir = args.log_dir or preset.logging.get("log_dir", "logs")

    try:
        detector = build_detector(detector_config)
        tracker = build_tracker(tracker_config)
    except ImportError as exc:
        print(f"Missing optional dependency for the selected backend: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(f"Invalid detector/tracker config: {exc}", file=sys.stderr)
        return 1

    logger = EventLogger(log_dir=log_dir)
    pipeline = Pipeline(detector=detector, tracker=tracker, logger=logger)

    video_writer = None
    print(f"[run_pipeline] source={args.source} preset={args.preset} detector={detector_config.get('backend')} tracker={tracker_config.get('backend')}")
    print(f"[run_pipeline] logging to {logger.log_path}")

    frame_count = 0
    failure: str | None = None
    try:
        for result in pipeline.run(source):
            print(
                f"frame={result.frame_index:05d} "
                f"detections={result.detection_count} "
                f"tracks={len(result.tracks)} "
                f"total_ms={result.total_ms} "
                f"fps={result.fps} "
                f"dropped={result.dropped}"
                + (f" error={result.error}" if result.error else "")
            )

            if args.save_video and result.frame is not None:
                if video_writer is None:
                    height, width = result.frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (width, height))
                annotated = draw_tracks(result.frame, result.tracks)
                video_writer.write(annotated)

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                pipeline.stop()
                break
    except RuntimeError as exc:
        failure = str(exc)
    finally:
        if video_writer is not None:
            video_writer.release()
        logger.close()

    if failure is not None:
        print(f"[run_pipeline] failed to run pipeline: {failure}", file=sys.stderr)
        return 1

    print(f"[run_pipeline] done. {frame_count} frames processed.")
    print(format_summary(summarize_log(logger.log_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
