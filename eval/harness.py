"""Frozen evaluation harness.

Runs a chosen detector (and, for track continuity, a tracker) over a
held-out, config-declared eval set and produces a `MetricCard`:
threshold-independent AP@0.5, AP@[0.50:0.95] and AP@0.5 by object size;
recall/precision and small-object recall at the operating confidence
threshold; false-alarm rate, latency, and track continuity —
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
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import yaml  # noqa: E402

from detector.factory import build_detector  # noqa: E402
from detector.sensor_sim import apply_sensor, apply_shake, frame_rng, validate_ops  # noqa: E402
from eval.breakdown import summarize as summarize_by_apparent_size  # noqa: E402
from eval.metrics import (  # noqa: E402
    FrameEvalData,
    PredBox,
    compute_ap50,
    compute_ap50_95,
    compute_ap_by_size,
    compute_false_alarm_rate,
    compute_fp_on_target_frames,
    compute_latency_stats,
    compute_recall_at_threshold,
    compute_small_object_recall,
    compute_track_continuity,
    compute_track_quality,
)
from eval.report import render_report_md  # noqa: E402
from eval.schema import (  # noqa: E402
    EvalSetChecksumMismatchError,
    compute_manifest_checksum,
    is_frozen,
    load_eval_manifest,
    verify_checksum,
)
from eval.sequences import SEQUENCE_GROUPINGS, apply_grouping  # noqa: E402
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
    # Optional gzipped JSONL, one line per frame: size, GT boxes and every
    # raw prediction (no cutoff), for breakdowns the metric card doesn't
    # carry (eval/breakdown.py). Off by default: tiled runs emit ~300
    # boxes/frame.
    output_predictions_path: str | None = None
    # Temporal view (eval/sequences.py): "none" = sequences.json as stored;
    # "filename_runs" = regroup frames into clips by "(n)" frame numbers,
    # without touching the manifest. Detector and tracker reset per sequence.
    sequence_grouping: str = "none"
    # Simulated sensor (detector/sensor_sim.py): ordered ops applied to each
    # frame after loading, outside the latency timer; seeded per frame.
    sensor_sim: list[dict[str, Any]] | None = None
    # Synthetic camera shake: {"amplitude_frac": 0.02, "seed": 0}. Ground
    # truth is shifted with the frame.
    shake: dict[str, Any] | None = None
    sensor_seed: int = 0

    def __post_init__(self) -> None:
        if self.sequence_grouping not in SEQUENCE_GROUPINGS:
            raise EvalConfigError(
                f"sequence_grouping must be one of {', '.join(SEQUENCE_GROUPINGS)}, got {self.sequence_grouping!r}"
            )
        if self.sensor_sim:
            validate_ops(self.sensor_sim)
        if self.shake is not None and "amplitude_frac" not in self.shake:
            raise EvalConfigError("shake needs an 'amplitude_frac'")

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
    ap50_95: float
    ap50_by_size: dict[str, dict[str, Any]]
    operating_point: dict[str, Any]
    small_object_recall: dict[str, Any]
    false_alarm_rate: dict[str, Any]
    latency: dict[str, float]
    track_continuity: dict[str, Any]
    manifest_checksum: str
    evaluated_at: str
    # Added for the temporal / robustness experiment (docs/DECISIONS.md #27);
    # defaults keep older callers (and cards) valid.
    false_positives_on_target_frames: dict[str, Any] = field(default_factory=dict)
    recall_by_apparent_size: dict[str, Any] = field(default_factory=dict)
    track_quality: dict[str, Any] = field(default_factory=dict)
    # The tracker's returned boxes scored as the system's output (see
    # track_output_metrics), and per-sequence tracker diagnostics (e.g.
    # ego-motion failures) when the tracker offers them.
    track_output: dict[str, Any] = field(default_factory=dict)
    tracker_diagnostics: dict[str, Any] = field(default_factory=dict)
    latency_by_stage: dict[str, dict[str, float]] = field(default_factory=dict)
    sequence_grouping: str = "none"
    sensor_sim: list[dict[str, Any]] | None = None
    shake: dict[str, Any] | None = None

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

    sequences = apply_grouping(load_eval_manifest(eval_set_dir), config.sequence_grouping)
    if not sequences:
        raise ValueError(f"No sequences found in eval set: {eval_set_dir}")

    # AP must be threshold-independent: rank *every* detection the model
    # emits, then integrate the full PR curve. So the detector is built with
    # its confidence cutoff disabled, and the configured
    # `confidence_threshold` is applied here afterwards, only to the
    # operating-point metrics (small-object recall, false-alarm rate) and
    # to what the tracker sees — the same detections the live pipeline
    # would act on. Before this, the cutoff was applied inside
    # detector.detect(), so AP only ever saw detections >= 0.35. See
    # docs/DECISIONS.md entry #19.
    operating_threshold = config.detector.get("confidence_threshold")
    detector_config = dict(config.detector)
    if operating_threshold is not None:
        detector_config["confidence_threshold"] = 0.0
    detector = build_detector(detector_config)

    def at_operating_point(items):
        if operating_threshold is None:
            return list(items)
        return [item for item in items if item.confidence >= operating_threshold]

    all_frame_eval_data: list[FrameEvalData] = []
    frame_sizes: list[tuple[int, int]] = []
    latencies_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    tracks_by_sequence: dict[str, list[list]] = {}
    tracker_diagnostics: dict[str, Any] = {}
    evaluated_sequences = []

    for sequence in sequences:
        tracker = build_tracker(config.tracker)
        reset = getattr(detector, "reset", None)
        if callable(reset):
            reset()
        per_frame_tracks = []
        evaluated_frames = []

        for frame in sequence.frames:
            image = cv2.imread(str(frame.image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read eval image: {frame.image_path}")
            image, frame = simulate_sensor(image, frame, config)
            evaluated_frames.append(frame)

            start = time.perf_counter()
            detections = detector.detect(image)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            for stage, ms in (getattr(detector, "last_stage_ms", None) or {}).items():
                stage_ms.setdefault(stage, []).append(ms)

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
            frame_sizes.append((frame.width, frame.height))
            all_frame_eval_data.append(
                FrameEvalData(
                    frame_id=f"{frame.sequence_id}#{frame.frame_index}",
                    gt_boxes=frame.gt_drone_boxes,
                    pred_boxes=pred_boxes,
                )
            )

            # A tracker with `min_detection_score` (e.g. kalman's low threshold)
            # gets the detections down to that score instead of only the
            # operating point; one with `observe_frame` gets the frame first.
            start = time.perf_counter()
            observe = getattr(tracker, "observe_frame", None)
            if callable(observe):
                observe(image)
            min_score = getattr(tracker, "min_detection_score", None)
            tracker_input = (
                [d for d in detections if d.confidence >= min_score]
                if min_score is not None
                else at_operating_point(detections)
            )
            tracks = tracker.update(tracker_input)
            stage_ms.setdefault("tracker", []).append((time.perf_counter() - start) * 1000.0)
            # Snapshot: trackers update Track objects in place, so storing the
            # live objects would give every frame the tracks' final positions.
            per_frame_tracks.append([copy.copy(track) for track in tracks])

        tracks_by_sequence[sequence.sequence_id] = per_frame_tracks
        diagnostics = getattr(tracker, "diagnostics", None)
        if callable(diagnostics):
            tracker_diagnostics[sequence.sequence_id] = diagnostics()
        evaluated_sequences.append(replace(sequence, frames=evaluated_frames))

    if config.output_predictions_path:
        write_predictions(config.output_predictions_path, all_frame_eval_data, frame_sizes)

    operating_frames = [
        FrameEvalData(frame_id=f.frame_id, gt_boxes=f.gt_boxes, pred_boxes=at_operating_point(f.pred_boxes))
        for f in all_frame_eval_data
    ]

    ap50 = compute_ap50(all_frame_eval_data, iou_threshold=config.iou_threshold)
    ap50_95 = compute_ap50_95(all_frame_eval_data)
    ap50_by_size = compute_ap_by_size(all_frame_eval_data, iou_threshold=config.iou_threshold)
    operating_point = compute_recall_at_threshold(
        all_frame_eval_data, operating_threshold, iou_threshold=config.iou_threshold
    )
    small_object_recall = compute_small_object_recall(
        operating_frames, small_area_threshold_px=config.small_object_area_px, iou_threshold=config.iou_threshold
    )
    false_alarm_rate = compute_false_alarm_rate(operating_frames)
    latency = compute_latency_stats(latencies_ms)
    track_continuity = compute_track_continuity(
        evaluated_sequences, tracks_by_sequence, iou_threshold=config.track_continuity_iou_threshold
    )
    track_quality = compute_track_quality(
        evaluated_sequences, tracks_by_sequence, iou_threshold=config.track_continuity_iou_threshold
    )
    fp_on_target = compute_fp_on_target_frames(operating_frames, iou_threshold=config.iou_threshold)
    recall_by_apparent_size = summarize_by_apparent_size(
        normalized_frames(all_frame_eval_data, frame_sizes), operating_threshold
    )

    return MetricCard(
        eval_set_dir=str(eval_set_dir),
        detector_backend=getattr(detector, "name", type(detector).__name__),
        tracker_backend=config.tracker.get("backend", "iou"),
        num_sequences=len(sequences),
        num_frames=len(all_frame_eval_data),
        ap50=ap50,
        ap50_95=ap50_95,
        ap50_by_size=ap50_by_size,
        operating_point=operating_point,
        small_object_recall=small_object_recall,
        false_alarm_rate=false_alarm_rate,
        latency=latency,
        track_continuity=track_continuity,
        manifest_checksum=manifest_checksum,
        evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        false_positives_on_target_frames=fp_on_target,
        recall_by_apparent_size=recall_by_apparent_size,
        track_quality=track_quality,
        track_output=track_output_metrics(evaluated_sequences, tracks_by_sequence, config.iou_threshold),
        tracker_diagnostics=tracker_diagnostics,
        latency_by_stage={stage: compute_latency_stats(values) for stage, values in stage_ms.items()},
        sequence_grouping=config.sequence_grouping,
        sensor_sim=config.sensor_sim,
        shake=config.shake,
    )


def simulate_sensor(image, frame, config: EvalConfig):
    """Apply the config's simulated sensor and shake to one loaded frame;
    returns the frame image and its EvalFrame (GT shifted with any shake).
    Seeded by file name, so a variant is identical on every run."""
    if not config.sensor_sim and config.shake is None:
        return image, frame
    key = frame.image_path.name
    if config.sensor_sim:
        image = apply_sensor(image, config.sensor_sim, frame_rng(config.sensor_seed, key))
    if config.shake is not None:
        rng = frame_rng(config.shake.get("seed", 0), "shake:" + key)
        # Offsets are drawn in the (possibly resized) image; GT stays in original pixels.
        sx, sy = frame.width / image.shape[1], frame.height / image.shape[0]
        image, (dx, dy) = apply_shake(image, rng, config.shake["amplitude_frac"])
        frame = replace(frame, boxes=[_shift_box(b, dx * sx, dy * sy, frame.width, frame.height) for b in frame.boxes])
        frame = replace(frame, boxes=[b for b in frame.boxes if b.area > 0])
    return image, frame


def track_output_metrics(
    sequences: list, tracks_by_sequence: dict[str, list[list]], iou_threshold: float = 0.5
) -> dict[str, Any]:
    """The tracker's returned boxes, scored as what an operator would see:
    every returned track is a prediction (no further cutoff), matched to GT
    at `iou_threshold` like a detection. Recall/precision, false positives
    on drone frames, false alarms on empty frames, recall by apparent size,
    and recall/precision per sequence. Empty when every sequence has one
    frame (a tracker's output then says nothing beyond the detector's)."""
    if not any(len(s.frames) > 1 for s in sequences):
        return {}
    frames, sizes, per_sequence = [], [], {}
    for sequence in sequences:
        seq_frames = []
        for frame, tracks in zip(sequence.frames, tracks_by_sequence.get(sequence.sequence_id, [])):
            seq_frames.append(
                FrameEvalData(
                    frame_id=f"{sequence.sequence_id}#{frame.frame_index}",
                    gt_boxes=frame.gt_drone_boxes,
                    pred_boxes=[
                        PredBox(t.x1 * frame.width, t.y1 * frame.height, t.x2 * frame.width, t.y2 * frame.height, t.confidence)
                        for t in tracks
                    ],
                )
            )
            sizes.append((frame.width, frame.height))
        frames += seq_frames
        op = compute_recall_at_threshold(seq_frames, None, iou_threshold=iou_threshold)
        per_sequence[sequence.sequence_id] = {
            "recall": op["recall"],
            "precision": op["precision"],
            "num_gt_boxes": op["num_gt_boxes"],
            "num_false_positives": op["num_false_positives"],
        }
    return {
        "operating_point": compute_recall_at_threshold(frames, None, iou_threshold=iou_threshold),
        "false_positives_on_target_frames": compute_fp_on_target_frames(frames, iou_threshold=iou_threshold),
        "false_alarm_rate": compute_false_alarm_rate(frames),
        "recall_by_apparent_size": summarize_by_apparent_size(normalized_frames(frames, sizes), None),
        "per_sequence": per_sequence,
    }


def _shift_box(box, dx: float, dy: float, width: int, height: int):
    return replace(
        box,
        x1=min(max(box.x1 + dx, 0.0), width),
        y1=min(max(box.y1 + dy, 0.0), height),
        x2=min(max(box.x2 + dx, 0.0), width),
        y2=min(max(box.y2 + dy, 0.0), height),
    )


def normalized_frames(frames: list[FrameEvalData], sizes: list[tuple[int, int]]) -> list[FrameEvalData]:
    """Frame-normalized copies (x / width, y / height), as eval/breakdown.py expects."""
    out = []
    for frame, (w, h) in zip(frames, sizes):
        out.append(
            FrameEvalData(
                frame_id=frame.frame_id,
                gt_boxes=[replace(g, x1=g.x1 / w, y1=g.y1 / h, x2=g.x2 / w, y2=g.y2 / h) for g in frame.gt_boxes],
                pred_boxes=[PredBox(p.x1 / w, p.y1 / h, p.x2 / w, p.y2 / h, p.confidence) for p in frame.pred_boxes],
            )
        )
    return out


def write_predictions(path: str | Path, frames: list[FrameEvalData], sizes: list[tuple[int, int]]) -> None:
    """Gzipped JSONL: {"frame_id", "width", "height", "gt": [[x1,y1,x2,y2]],
    "pred": [[x1,y1,x2,y2,confidence]]} per frame, pixels, all predictions."""
    import gzip

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for frame, (width, height) in zip(frames, sizes):
            row = {
                "frame_id": frame.frame_id,
                "width": width,
                "height": height,
                "gt": [[round(v, 2) for v in (b.x1, b.y1, b.x2, b.y2)] for b in frame.gt_boxes],
                "pred": [[round(p.x1, 2), round(p.y1, 2), round(p.x2, 2), round(p.y2, 2), round(p.confidence, 6)] for p in frame.pred_boxes],
            }
            handle.write(json.dumps(row) + "\n")


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
