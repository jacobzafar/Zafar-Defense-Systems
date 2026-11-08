#!/usr/bin/env python3
"""Choose tracker parameters on val by replaying cached detections.

The detector dominates harness run time (~1 s/frame at 640 on this CPU),
but a tracker grid only needs the detections, which a harness run already
dumped (`output_predictions_path`: every raw prediction). This replays
those detections through each tracker config, clip by clip, and scores
the tracker's returned boxes exactly as the harness does
(harness.track_output_metrics, compute_track_quality,
compute_track_continuity). Camera motion is estimated once per frame
(on the same simulated-sensor/shaken pixels the harness saw) and reused
for every config with `ego_motion: true`.

Selection only — run on val, never on test. The chosen config is then
run once through the real harness (which also times it).

Usage:
    python -m eval.track_sweep --config eval/config/fpv_val_stride8_squeeze_bilinear.yaml \\
        --config eval/config/fpv_val_stride8_shake.yaml --grid eval/config/kalman_sweep_grid.yaml \\
        --output-json eval/output/kalman_sweep_val.json
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import yaml  # noqa: E402

from detector.base import Detection  # noqa: E402
from eval.harness import EvalConfig, simulate_sensor, track_output_metrics  # noqa: E402
from eval.metrics import compute_track_continuity, compute_track_quality  # noqa: E402
from eval.schema import EvalBox, load_eval_manifest  # noqa: E402
from eval.sequences import apply_grouping  # noqa: E402
from tracker.ego_motion import EgoMotionEstimator  # noqa: E402
from tracker.factory import build_tracker  # noqa: E402


def load_replay(config: EvalConfig):
    """Sequences (GT as in the dump, i.e. shifted by any shake), per-frame
    detections from the dump, and per-frame camera motion."""
    if not config.output_predictions_path or not Path(config.output_predictions_path).exists():
        raise FileNotFoundError(f"Run the harness with this config first: no dump at {config.output_predictions_path}")
    rows = {}
    with gzip.open(config.output_predictions_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["frame_id"]] = row

    sequences = apply_grouping(load_eval_manifest(config.eval_set_dir), config.sequence_grouping)
    replay = []
    for sequence in sequences:
        estimator = EgoMotionEstimator()
        frames, detections, motions = [], [], []
        for frame in sequence.frames:
            row = rows[f"{sequence.sequence_id}#{frame.frame_index}"]
            w, h = row["width"], row["height"]
            frames.append(replace(frame, boxes=[EvalBox(*b, "drone") for b in row["gt"]]))
            detections.append([Detection(x1 / w, y1 / h, x2 / w, y2 / h, c, "drone") for x1, y1, x2, y2, c in row["pred"]])
            image, _ = simulate_sensor(cv2.imread(str(frame.image_path)), frame, config)
            motions.append((estimator.update(image), (image.shape[1], image.shape[0])))
        replay.append((replace(sequence, frames=frames), detections, motions))
    return replay


def run_config(
    replay, tracker_config: dict[str, Any], iou_threshold: float, continuity_iou: float, operating_threshold: float
) -> dict[str, Any]:
    sequences, tracks_by_sequence, diagnostics = [], {}, {}
    kalman = tracker_config.get("backend") == "kalman"
    for sequence, detections, motions in replay:
        # Camera motion comes from the cache, not from a per-config estimator.
        tracker = build_tracker({**tracker_config, "ego_motion": False} if kalman else tracker_config)
        use_motion = tracker_config.get("ego_motion", True)
        # As the harness: a tracker without min_detection_score sees the operating point.
        min_score = getattr(tracker, "min_detection_score", operating_threshold)
        per_frame = []
        for dets, (motion, size) in zip(detections, motions):
            if kalman:
                tracker._frame_size = size
                tracker.apply_camera_motion(motion if use_motion else None)
            tracks = tracker.update([d for d in dets if d.confidence >= min_score])
            per_frame.append([replace(t) for t in tracks])
        tracks_by_sequence[sequence.sequence_id] = per_frame
        sequences.append(sequence)
        diagnostics[sequence.sequence_id] = tracker.diagnostics() if kalman else {}
    output = track_output_metrics(sequences, tracks_by_sequence, iou_threshold)
    quality = compute_track_quality(sequences, tracks_by_sequence, continuity_iou)
    continuity = compute_track_continuity(sequences, tracks_by_sequence, continuity_iou)
    op = output["operating_point"]
    recall, precision = op["recall"] or 0.0, op["precision"] or 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / (recall + precision) if recall + precision else 0.0,
        "fp_per_drone_frame": output["false_positives_on_target_frames"]["rate_per_frame"],
        "false_alarms_empty": output["false_alarm_rate"]["num_false_alarms"],
        "continuity": continuity["overall_continuity"],
        "false_track_ids": quality["num_false_track_ids"],
        "frames_to_first_track_mean": quality["frames_to_first_track_mean"],
        "per_sequence_recall": {k: v["recall"] for k, v in output["per_sequence"].items()},
        "innovation_median": {k: d.get("innovation_median") for k, d in diagnostics.items()},
    }


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", required=True, help="Harness config(s) whose dump to replay")
    parser.add_argument("--grid", required=True, help="YAML: tracker key -> list of values (backend included)")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    candidates = [c for g in grid["grids"] for c in expand_grid(g)]
    replays = {}
    for path in args.config:
        config = EvalConfig.from_yaml(path)
        replays[Path(path).stem] = (load_replay(config), config)
        print(f"loaded {path}", file=sys.stderr)

    results = []
    for candidate in candidates:
        per_set = {
            name: run_config(
                replay, candidate, cfg.iou_threshold, cfg.track_continuity_iou_threshold, cfg.detector["confidence_threshold"]
            )
            for name, (replay, cfg) in replays.items()
        }
        results.append({"tracker": candidate, "sets": per_set, "mean_f1": sum(r["f1"] for r in per_set.values()) / len(per_set)})
    results.sort(key=lambda r: (-r["mean_f1"], sum(s["false_track_ids"] for s in r["sets"].values())))

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results[:10]:
        sets = " | ".join(f"{n}: R {s['recall']:.3f} P {s['precision']:.3f} cont {s['continuity']:.3f}" for n, s in r["sets"].items())
        print(f"{r['mean_f1']:.4f}  {r['tracker']}  {sets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
