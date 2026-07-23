"""Frozen evaluation harness.

Runs a chosen detector (and, for track continuity, a tracker) over a
held-out, config-declared eval set and produces a `MetricCard`: AP@0.5,
small-object recall, false-alarm rate, latency, and track continuity —
see eval/metrics.py for exact metric definitions.

This harness never modifies the eval set and never trains anything. If
the eval set directory is not marked frozen (see eval/schema.py), this
prints a warning but still runs — freezing is a maintainer action
(`eval.schema.freeze_eval_set`) taken once an eval set is finalized, not
a hard precondition for every experimental run.

Usage:
    python -m eval.harness --config path/to/eval_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import yaml  # noqa: E402

from detector.factory import build_detector  # noqa: E402
from eval.metrics import (  # noqa: E402
    FrameEvalData,
    PredBox,
    compute_ap50,
    compute_false_alarm_rate,
    compute_latency_stats,
    compute_small_object_recall,
    compute_track_continuity,
)
from eval.report import render_report_md  # noqa: E402
from eval.schema import (  # noqa: E402
    EvalSetChecksumMismatchError,
    compute_manifest_checksum,
    is_frozen,
    load_eval_manifest,
    verify_checksum,
)
from tracker.factory import build_tracker  # noqa: E402


class EvalConfigError(ValueError):
    """Raised when an eval config is malformed."""


@dataclass
class EvalConfig:
    eval_set_dir: str
    detector: dict[str, Any]
    tracker: dict[str, Any] = field(default_factory=lambda: {"backend": "iou"})
    iou_threshold: float = 0.5
    small_object_area_px: float = 1024.0
    track_continuity_iou_threshold: float = 0.3
    expected_checksum: str | None = None
    output_json_path: str = "eval/output/metric_card.json"
    output_report_path: str = "eval/output/REPORT.md"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known_fields = set(cls.__dataclass_fields__)
        unknown = set(raw) - known_fields
        if unknown:
            raise EvalConfigError(f"Unknown key(s) in {path}: {', '.join(sorted(unknown))}")
        return cls(**raw)


@dataclass
class MetricCard:
    eval_set_dir: str
    detector_backend: str
    tracker_backend: str
    num_sequences: int
    num_frames: int
    ap50: float
    small_object_recall: dict[str, Any]
    false_alarm_rate: dict[str, Any]
    latency: dict[str, float]
    track_continuity: dict[str, Any]
    manifest_checksum: str
    evaluated_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_evaluation(config: EvalConfig) -> MetricCard:
    eval_set_dir = Path(config.eval_set_dir)

    if not is_frozen(eval_set_dir):
        print(
            f"WARNING: {eval_set_dir} is not marked frozen (no .frozen marker file). "
            f"Results from an eval set that can still change are not safely comparable "
            f"across runs — see eval/README.md.",
            file=sys.stderr,
        )

    manifest_checksum = compute_manifest_checksum(eval_set_dir)
    if config.expected_checksum:
        verify_checksum(eval_set_dir, config.expected_checksum)

    sequences = load_eval_manifest(eval_set_dir)
    if not sequences:
        raise ValueError(f"No sequences found in eval set: {eval_set_dir}")

    detector = build_detector(config.detector)

    all_frame_eval_data: list[FrameEvalData] = []
    latencies_ms: list[float] = []
    tracks_by_sequence: dict[str, list[list]] = {}

    for sequence in sequences:
        tracker = build_tracker(config.tracker)
        per_frame_tracks = []

        for frame in sequence.frames:
            image = cv2.imread(str(frame.image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read eval image: {frame.image_path}")

            start = time.perf_counter()
            detections = detector.detect(image)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            pred_boxes = [
                PredBox(
                    x1=d.x1 * frame.width,
                    y1=d.y1 * frame.height,
                    x2=d.x2 * frame.width,
                    y2=d.y2 * frame.height,
                    confidence=d.confidence,
                )
                for d in detections
            ]
            all_frame_eval_data.append(
                FrameEvalData(
                    frame_id=f"{frame.sequence_id}#{frame.frame_index}",
                    gt_boxes=frame.gt_drone_boxes,
                    pred_boxes=pred_boxes,
                )
            )

            tracks = tracker.update(detections)
            per_frame_tracks.append(tracks)

        tracks_by_sequence[sequence.sequence_id] = per_frame_tracks

    ap50 = compute_ap50(all_frame_eval_data, iou_threshold=config.iou_threshold)
    small_object_recall = compute_small_object_recall(
        all_frame_eval_data, small_area_threshold_px=config.small_object_area_px, iou_threshold=config.iou_threshold
    )
    false_alarm_rate = compute_false_alarm_rate(all_frame_eval_data)
    latency = compute_latency_stats(latencies_ms)
    track_continuity = compute_track_continuity(
        sequences, tracks_by_sequence, iou_threshold=config.track_continuity_iou_threshold
    )

    return MetricCard(
        eval_set_dir=str(eval_set_dir),
        detector_backend=getattr(detector, "name", type(detector).__name__),
        tracker_backend=config.tracker.get("backend", "iou"),
        num_sequences=len(sequences),
        num_frames=len(all_frame_eval_data),
        ap50=ap50,
        small_object_recall=small_object_recall,
        false_alarm_rate=false_alarm_rate,
        latency=latency,
        track_continuity=track_continuity,
        manifest_checksum=manifest_checksum,
        evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def write_outputs(metric_card: MetricCard, config: EvalConfig) -> None:
    json_path = Path(config.output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metric_card.as_dict(), indent=2), encoding="utf-8")

    report_path = Path(config.output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report_md(metric_card, config), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen evaluation harness.")
    parser.add_argument("--config", required=True, help="Path to a YAML EvalConfig file")
    args = parser.parse_args()

    try:
        config = EvalConfig.from_yaml(args.config)
    except (EvalConfigError, FileNotFoundError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    try:
        metric_card = run_evaluation(config)
    except EvalSetChecksumMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_outputs(metric_card, config)
    print(json.dumps(metric_card.as_dict(), indent=2))
    print(f"\nReport written to {config.output_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
