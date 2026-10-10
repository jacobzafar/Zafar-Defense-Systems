#!/usr/bin/env python3
"""Ingest one captured video into a per-clip raw dataset directory.

Extracts frames at a fixed sampling rate (default 5 FPS: consecutive
frames at 30-60 FPS are near-duplicates that cost annotation time and add
little) into the unified layout detector/datasets/loader.py reads, and
records the clip's capture conditions in a `clip_meta.json` sidecar that
tools/coverage_report.py aggregates. See tools/footage_meta.py for the
layout and vocabulary, and docs/data-capture-runbook.md for the workflow.

No labels are created here — not even empty ones (see footage_meta.py for
why). Frames are named `<clip_id>_f<source frame index>.jpg`, so every
frame is globally unique across clips and traceable to the exact frame of
the source video.

Usage:
    python tools/ingest_footage.py --video DJI_0042.MP4 --drone-type fpv-quad \\
        --distance-band medium-50-200m --lighting daylight-clear --background clean-sky
    python tools/ingest_footage.py --video gulls.mov --hard-negative --negative-subject bird \\
        --drone-type none --distance-band close-lt50m --lighting daylight-overcast --background water
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from footage_meta import (  # noqa: E402
    BACKGROUNDS,
    CLIP_META_FILENAME,
    DISTANCE_BANDS,
    DRONE_TYPES,
    FRAMES_FILENAME,
    LIGHTING,
    NEGATIVE_SUBJECTS,
    SCHEMA_VERSION,
    validate_meta,
)

DEFAULT_FOOTAGE_ROOT = "data/own-fpv/clips"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


class IngestError(RuntimeError):
    """Raised when a video can't be ingested as asked."""


def default_clip_id(video_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", video_path.stem).strip("-").lower() or "clip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest(
    video: str | Path,
    metadata: dict,
    footage_root: str | Path = DEFAULT_FOOTAGE_ROOT,
    clip_id: str | None = None,
    sample_fps: float = 5.0,
    jpeg_quality: int = 95,
) -> Path:
    """Extract frames + write frames.json and clip_meta.json. Returns the
    clip directory. Refuses to overwrite an existing clip."""
    video = Path(video)
    if not video.is_file():
        raise IngestError(f"Video not found: {video}")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise IngestError(f"Unsupported video extension {video.suffix!r}; expected one of {sorted(VIDEO_EXTENSIONS)}")
    validate_meta(metadata)
    if sample_fps <= 0:
        raise IngestError(f"--fps must be > 0, got {sample_fps}")

    clip_id = clip_id or default_clip_id(video)
    clip_dir = Path(footage_root) / clip_id
    if clip_dir.exists():
        raise IngestError(f"Clip already ingested: {clip_dir} (pass a different --clip-id, or delete it first)")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IngestError(f"Could not open video: {video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        capture.release()
        raise IngestError(f"Video reports no frame rate, can't sample by time: {video}")

    # Build in a sibling temp dir and rename at the end, so an interrupted
    # ingest never leaves a half-written clip that looks complete.
    staging = clip_dir.with_name(f".{clip_id}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)

    period = 1.0 / sample_fps
    images, index, next_tick, width, height = [], 0, 0.0, None, None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Keep the first frame at or after each 1/sample_fps tick, by
            # timestamp; streamed because container frame counts are unreliable.
            if sample_fps >= source_fps or index / source_fps + 1e-9 >= next_tick:
                height, width = frame.shape[:2]
                file_name = f"{clip_id}_f{index:06d}.jpg"
                if not cv2.imwrite(str(staging / "images" / file_name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise IngestError(f"Failed to write frame {file_name}")
                images.append({"id": len(images) + 1, "file_name": file_name, "width": width, "height": height, "source_frame": index})
                next_tick += period
            index += 1
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        capture.release()

    if not images:
        shutil.rmtree(staging, ignore_errors=True)
        raise IngestError(f"No frames could be decoded from {video}")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        **metadata,
        "source": {
            "file_name": video.name,
            "sha256": _sha256(video),
            "fps": source_fps,
            "num_frames_decoded": index,
            "duration_s": round(index / source_fps, 3),
            "width": width,
            "height": height,
        },
        "sample_fps": sample_fps,
        "num_frames_extracted": len(images),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_status": "unlabeled",
    }
    (staging / FRAMES_FILENAME).write_text(json.dumps({"images": images}, indent=2), encoding="utf-8")
    (staging / CLIP_META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    staging.rename(clip_dir)
    return clip_dir


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Source video (mp4/mov/...)")
    parser.add_argument("--footage-root", default=DEFAULT_FOOTAGE_ROOT, help=f"Where clip dirs go (default: {DEFAULT_FOOTAGE_ROOT})")
    parser.add_argument("--clip-id", default=None, help="Clip directory name (default: from the video file name)")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second to extract (default: 5)")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--drone-type", required=True, choices=DRONE_TYPES)
    parser.add_argument("--distance-band", required=True, choices=DISTANCE_BANDS)
    parser.add_argument("--lighting", required=True, choices=LIGHTING)
    parser.add_argument("--background", required=True, choices=BACKGROUNDS)
    parser.add_argument("--hard-negative", action="store_true", help="No drone in this clip (requires --drone-type none)")
    parser.add_argument("--negative-subject", choices=NEGATIVE_SUBJECTS, default=None, help="What the hard negative shows")
    parser.add_argument("--capture-date", default=None, help="YYYY-MM-DD the footage was shot")
    parser.add_argument("--notes", default="", help="Free text: location, altitude, camera, anything unusual")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    metadata = {
        "drone_type": args.drone_type,
        "distance_band": args.distance_band,
        "lighting": args.lighting,
        "background": args.background,
        "is_hard_negative": args.hard_negative,
        "negative_subject": args.negative_subject,
        "capture_date": args.capture_date,
        "notes": args.notes,
    }
    try:
        clip_dir = ingest(args.video, metadata, args.footage_root, args.clip_id, args.fps, args.jpeg_quality)
    except (IngestError, ValueError) as exc:
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1
    meta = json.loads((clip_dir / CLIP_META_FILENAME).read_text(encoding="utf-8"))
    print(
        f"Ingested {meta['num_frames_extracted']} frames ({meta['source']['duration_s']}s at "
        f"{meta['source']['fps']:.2f} FPS, sampled at {args.fps} FPS) into {clip_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
