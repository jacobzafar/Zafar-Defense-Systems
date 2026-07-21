"""Summarize a completed run's JSONL telemetry log.

Reads back what `telemetry.logger.EventLogger` wrote — this module never
runs during the pipeline loop itself, only afterward, so it has no
performance impact on a live run. Used by the CLI (`scripts/run_pipeline.py`,
`scripts/summarize_log.py`) and the Streamlit results panel so both share
one definition of "what a run summary means."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    """Aggregate totals derived from one run's JSONL log."""

    log_path: str
    source: str | None = None
    detector: str | None = None
    tracker: str | None = None
    total_frames: int = 0
    total_detections: int = 0
    unique_track_count: int = 0
    dropped_frames: int = 0
    started_at: float | None = None
    ended_at: float | None = None
    duration_seconds: float = 0.0
    avg_fps: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.total_frames > 0 or self.dropped_frames > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "log_path": self.log_path,
            "source": self.source,
            "detector": self.detector,
            "tracker": self.tracker,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "unique_track_count": self.unique_track_count,
            "dropped_frames": self.dropped_frames,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "avg_fps": self.avg_fps,
        }


def summarize_log(log_path: str | Path) -> RunSummary:
    """Parse a JSONL run log into a `RunSummary`.

    Tolerant of a missing file or malformed lines (returns an empty/partial
    summary rather than raising) — this runs after a pipeline run, and a
    summary that can't be computed shouldn't itself crash the caller.
    """
    log_path = Path(log_path)
    summary = RunSummary(log_path=str(log_path))
    if not log_path.exists():
        return summary

    track_ids: set[int] = set()
    fps_values: list[float] = []

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("event_type")
            payload = event.get("payload") or {}
            timestamp = event.get("timestamp")

            if timestamp is not None:
                if summary.started_at is None:
                    summary.started_at = timestamp
                summary.ended_at = timestamp

            if event_type == "run_started":
                summary.source = payload.get("source")
                summary.detector = payload.get("detector")
                summary.tracker = payload.get("tracker")
            elif event_type == "frame_processed":
                summary.total_frames += 1
                summary.total_detections += payload.get("detection_count", 0) or 0
                for track in payload.get("tracks") or []:
                    track_id = track.get("track_id")
                    if track_id is not None:
                        track_ids.add(track_id)
                fps = payload.get("fps")
                if fps:
                    fps_values.append(fps)
            elif event_type == "frame_dropped":
                summary.dropped_frames += 1

    summary.unique_track_count = len(track_ids)
    if summary.started_at is not None and summary.ended_at is not None:
        summary.duration_seconds = round(summary.ended_at - summary.started_at, 3)
    if fps_values:
        summary.avg_fps = round(sum(fps_values) / len(fps_values), 2)

    return summary


def format_summary(summary: RunSummary) -> str:
    """Render a `RunSummary` as the plain-text block used by the CLI tools."""
    lines = [
        f"Run summary — {summary.log_path}",
        f"  source            : {summary.source}",
        f"  detector          : {summary.detector}",
        f"  tracker           : {summary.tracker}",
        f"  total frames      : {summary.total_frames}",
        f"  total detections  : {summary.total_detections}",
        f"  unique tracks     : {summary.unique_track_count}",
        f"  dropped frames    : {summary.dropped_frames}",
        f"  duration (s)      : {summary.duration_seconds}",
        f"  avg fps           : {summary.avg_fps}",
    ]
    return "\n".join(lines)
