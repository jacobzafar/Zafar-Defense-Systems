"""Metric computations for the frozen evaluation harness (eval/harness.py).

Every function here is pure (no I/O, no detector/tracker execution) so it
can be unit-tested with small, hand-computed examples independent of any
model — see tests/test_eval_metrics.py, which includes worked-by-hand
expected values for the AP@0.5 calculation specifically.

Metric definitions (fixed here so results are comparable run to run):

- **AP@0.5** (single-class "drone" average precision at IoU=0.5): standard
  precision/recall curve with all-point (COCO/VOC2010+-style) precision
  envelope interpolation. Reported as "AP@0.5", not "mAP", since this is a
  single-class problem — mAP is only meaningful with class averaging.
  Threshold-independent: callers must pass *every* detection the model
  emits (no confidence cutoff) — a cutoff truncates the PR curve and can
  only lower AP. eval/harness.py does this; see docs/DECISIONS.md #19.
- **AP@[0.50:0.95]**: mean of the above over IoU 0.50, 0.55, ..., 0.95
  (COCO's primary metric; all-point interpolation per threshold).
- **AP@0.5 by size**: the above restricted to COCO's small (<32²px),
  medium (32²-96²px) and large (>=96²px) GT boxes, with COCO's ignore
  rules for out-of-range boxes and predictions (see compute_ap).
- **Recall/precision at the operating threshold**: at the detector
  config's `confidence_threshold` — a deployment operating point,
  reported separately from (never instead of) the threshold-independent AP.
- **Small-object recall**: recall restricted to ground-truth boxes with
  pixel area below `small_area_threshold_px` (default 1024 = 32x32,
  COCO's own "small object" convention), at a fixed confidence threshold
  (whatever predictions the caller passed in were already filtered to).
- **False-alarm rate**: predicted boxes raised on frames with zero
  ground-truth "drone" boxes (i.e. hard-negative frames — background,
  bird, or clutter only), reported as false alarms per hard-negative
  frame. Not bounded to [0, 1]; a frame can have multiple false alarms.
- **Latency**: wall-clock detector.detect() time per frame, in
  milliseconds — mean/median/p95 and derived FPS (1000/mean_ms).
- **Track continuity**: 1 - (ID switches / frames where a GT drone was
  present), across one or more sequences. 1.0 = the tracker never
  switched IDs while a target was continuously present; lower is worse.
  A gap where the GT drone leaves and later returns does not itself count
  as a switch — only a change in track ID while GT presence is
  uninterrupted counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval.schema import EvalBox, EvalSequence


@dataclass
class PredBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def iou(a: Any, b: Any) -> float:
    """IoU between two objects exposing .x1/.y1/.x2/.y2 in the same units."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class FrameEvalData:
    """One frame's ground truth + predictions, in matching pixel-coordinate space."""

    frame_id: str
    gt_boxes: list[EvalBox] = field(default_factory=list)
    pred_boxes: list[PredBox] = field(default_factory=list)


# COCO's object-size buckets, in pixel area of the box in the original
# image. Half-open [lo, hi) so "small" matches compute_small_object_recall
# and the DUT converter's own `area < 1024` small-instance count exactly.
COCO_AREA_RANGES: dict[str, tuple[float, float]] = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}

AP50_95_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def _box_area(box: Any) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _iou_matrix(preds: list[PredBox], gts: list[EvalBox]) -> np.ndarray:
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    p = np.array([[b.x1, b.y1, b.x2, b.y2] for b in preds], dtype=float)
    g = np.array([[b.x1, b.y1, b.x2, b.y2] for b in gts], dtype=float)
    inter_w = np.clip(np.minimum(p[:, None, 2], g[None, :, 2]) - np.maximum(p[:, None, 0], g[None, :, 0]), 0, None)
    inter_h = np.clip(np.minimum(p[:, None, 3], g[None, :, 3]) - np.maximum(p[:, None, 1], g[None, :, 1]), 0, None)
    inter = inter_w * inter_h
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    area_g = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where((inter > 0) & (union > 0), inter / np.where(union > 0, union, 1.0), 0.0)


def compute_ap(
    frames: list[FrameEvalData],
    iou_threshold: float = 0.5,
    area_range: tuple[float, float] | None = None,
) -> float:
    """Single-class, threshold-independent AP at one IoU threshold.

    Every prediction in `frames` is ranked by confidence (no cutoff — pass
    them all) and the full PR curve is integrated with all-point
    interpolation of the backward-max precision envelope. Each prediction
    is greedily matched, in descending-score order, to the unmatched GT
    box in its frame with the highest IoU >= `iou_threshold`.

    `area_range=(lo, hi)` restricts evaluation to GT boxes with
    lo <= area < hi using COCO's ignore rules: out-of-range GT boxes don't
    count toward recall, a prediction matched to one is ignored (neither
    TP nor FP), and an unmatched prediction whose own area is out of range
    is ignored. Matching prefers in-range GT boxes over ignored ones.

    Returns 0.0 if there are no (in-range) GT boxes at all — AP is
    undefined there; 0.0 is a JSON-safe default rather than NaN. Callers
    that need to tell "undefined" from "measured zero" (e.g.
    compute_ap_by_size) check the GT count themselves.
    """
    lo, hi = area_range if area_range is not None else (-float("inf"), float("inf"))

    def in_range(box: Any) -> bool:
        return lo <= _box_area(box) < hi

    scores: list[float] = []
    outcomes: list[int] = []  # 1 = TP, 0 = FP, -1 = ignored
    total_gt = 0

    for frame in frames:
        gt_in_range = np.array([in_range(gt) for gt in frame.gt_boxes], dtype=bool)
        total_gt += int(gt_in_range.sum())
        if not frame.pred_boxes:
            continue

        order = sorted(range(len(frame.pred_boxes)), key=lambda i: frame.pred_boxes[i].confidence, reverse=True)
        preds = [frame.pred_boxes[i] for i in order]
        ious = _iou_matrix(preds, frame.gt_boxes)
        gt_matched = np.zeros(len(frame.gt_boxes), dtype=bool)

        for row, pred in enumerate(preds):
            outcome = None
            for want_in_range in (True, False):
                candidates = (~gt_matched) & (gt_in_range == want_in_range) & (ious[row] >= iou_threshold)
                if candidates.any():
                    best = int(np.argmax(np.where(candidates, ious[row], -1.0)))
                    gt_matched[best] = True
                    outcome = 1 if want_in_range else -1
                    break
            if outcome is None:
                outcome = 0 if in_range(pred) else -1
            scores.append(pred.confidence)
            outcomes.append(outcome)

    if total_gt == 0:
        return 0.0

    scores_arr = np.asarray(scores, dtype=float)
    outcomes_arr = np.asarray(outcomes, dtype=int)
    keep = outcomes_arr >= 0
    scores_arr, outcomes_arr = scores_arr[keep], outcomes_arr[keep]
    if len(scores_arr) == 0:
        return 0.0

    # Stable sort: ties keep frame order, matching the original implementation.
    global_order = np.argsort(-scores_arr, kind="stable")
    tps = (outcomes_arr[global_order] == 1).astype(float)
    fps = 1.0 - tps

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    envelope = np.maximum.accumulate(precisions[::-1])[::-1]
    recall_steps = np.diff(np.concatenate(([0.0], recalls)))
    return float(np.sum(recall_steps * envelope))


def compute_ap50(frames: list[FrameEvalData], iou_threshold: float = 0.5) -> float:
    """Single-class AP at the given IoU threshold (default 0.5) — see compute_ap."""
    return compute_ap(frames, iou_threshold=iou_threshold)


def compute_ap50_95(frames: list[FrameEvalData]) -> float:
    """COCO-style AP@[0.50:0.95]: mean of compute_ap over IoU thresholds
    0.50, 0.55, ..., 0.95. (All-point interpolation per threshold, not
    pycocotools' 101-point sampling — values can differ slightly from
    pycocotools in the third decimal.)"""
    return float(np.mean([compute_ap(frames, iou_threshold=t) for t in AP50_95_IOU_THRESHOLDS]))


def compute_ap_by_size(frames: list[FrameEvalData], iou_threshold: float = 0.5) -> dict[str, dict[str, Any]]:
    """AP per COCO size bucket (small/medium/large, see COCO_AREA_RANGES).
    `ap` is None (not 0.0) for a bucket with no GT boxes — unmeasurable,
    not measured-as-zero."""
    result: dict[str, dict[str, Any]] = {}
    for name, (lo, hi) in COCO_AREA_RANGES.items():
        num_gt = sum(1 for f in frames for gt in f.gt_boxes if lo <= _box_area(gt) < hi)
        result[name] = {
            "ap": compute_ap(frames, iou_threshold=iou_threshold, area_range=(lo, hi)) if num_gt else None,
            "num_gt_boxes": num_gt,
            "area_range_px": [lo, hi if hi != float("inf") else None],
        }
    return result


def compute_recall_at_threshold(
    frames: list[FrameEvalData], confidence_threshold: float | None, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """Recall and precision at a fixed operating confidence threshold —
    what a deployed detector actually reports — kept separate from the
    threshold-independent AP. `confidence_threshold=None` means no cutoff
    (e.g. a backend without a confidence score threshold)."""
    total_gt = 0
    tp = 0
    fp = 0
    for frame in frames:
        total_gt += len(frame.gt_boxes)
        preds = [
            p for p in frame.pred_boxes if confidence_threshold is None or p.confidence >= confidence_threshold
        ]
        preds.sort(key=lambda p: p.confidence, reverse=True)
        ious = _iou_matrix(preds, frame.gt_boxes)
        gt_matched = np.zeros(len(frame.gt_boxes), dtype=bool)
        for row in range(len(preds)):
            candidates = (~gt_matched) & (ious[row] >= iou_threshold)
            if candidates.any():
                gt_matched[int(np.argmax(np.where(candidates, ious[row], -1.0)))] = True
                tp += 1
            else:
                fp += 1
    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "recall": (tp / total_gt) if total_gt else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "num_gt_boxes": total_gt,
        "num_true_positives": tp,
        "num_false_positives": fp,
    }


def compute_small_object_recall(
    frames: list[FrameEvalData], small_area_threshold_px: float = 1024.0, iou_threshold: float = 0.5
) -> dict[str, Any]:
    total_small_gt = 0
    matched_small_gt = 0

    for f in frames:
        preds_sorted = sorted(f.pred_boxes, key=lambda p: p.confidence, reverse=True)
        matched_idxs: set[int] = set()
        for pred in preds_sorted:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(f.gt_boxes):
                if idx in matched_idxs:
                    continue
                score = iou(pred, gt)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                matched_idxs.add(best_idx)

        for idx, gt in enumerate(f.gt_boxes):
            if gt.area < small_area_threshold_px:
                total_small_gt += 1
                if idx in matched_idxs:
                    matched_small_gt += 1

    recall = (matched_small_gt / total_small_gt) if total_small_gt > 0 else 0.0
    return {
        "recall": recall,
        "num_small_gt_boxes": total_small_gt,
        "num_small_gt_matched": matched_small_gt,
        "area_threshold_px": small_area_threshold_px,
    }


def compute_false_alarm_rate(frames: list[FrameEvalData]) -> dict[str, Any]:
    """`rate_per_frame` (and `measurable`) is `None`/`False`, not `0.0`,
    when there are zero hard-negative frames in `frames` — an eval set
    with no hard negatives says nothing about false-alarm behavior, and a
    `0.0` there would silently read as "measured zero false alarms"
    rather than "never tested." Callers must check `measurable` before
    trusting `rate_per_frame`.
    """
    hard_negative_frames = [f for f in frames if len(f.gt_boxes) == 0]
    num_false_alarms = sum(len(f.pred_boxes) for f in hard_negative_frames)
    measurable = len(hard_negative_frames) > 0
    return {
        "measurable": measurable,
        "rate_per_frame": (num_false_alarms / len(hard_negative_frames)) if measurable else None,
        "num_hard_negative_frames": len(hard_negative_frames),
        "num_false_alarms": num_false_alarms,
        "note": None
        if measurable
        else "No hard-negative (zero-GT) frames in this eval set — false-alarm rate is not measurable here.",
    }


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
    arr = np.asarray(latencies_ms, dtype=float)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


def compute_track_continuity(
    sequences: list[EvalSequence],
    tracks_by_sequence: dict[str, list[list]],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Score ID stability against GT across one or more sequences.

    `tracks_by_sequence[sequence_id]` must be a list of per-frame
    `tracker.base.Track` lists, one entry per frame in that sequence, in
    the same order as `EvalSequence.frames` (normalized [0,1] coordinates,
    matching Track's own convention — converted to pixel space here using
    each frame's width/height before matching against ground truth).

    `measurable` is `False` when every sequence has at most 1 frame — an
    ID switch is structurally impossible to observe with only one frame
    per sequence (e.g. an eval set built by wrapping independent static
    images as trivial one-frame "sequences" — see
    eval.build_frozen_eval_set). In that case `overall_continuity` and
    every per-sequence `continuity` are `None`, not `1.0`: a `1.0` there
    would silently read as "measured perfect continuity" rather than
    "never actually tested."
    """
    measurable = any(len(sequence.frames) > 1 for sequence in sequences)

    total_switches = 0
    total_gt_present_frames = 0
    per_sequence: list[dict[str, Any]] = []

    for sequence in sequences:
        tracks_per_frame = tracks_by_sequence.get(sequence.sequence_id, [])
        switches = 0
        gt_present = 0
        current_id = None

        for frame, tracks in zip(sequence.frames, tracks_per_frame):
            gt_boxes = frame.gt_drone_boxes
            if not gt_boxes:
                current_id = None
                continue

            gt_present += 1
            gt_box = gt_boxes[0]
            best_iou = 0.0
            best_track = None
            for track in tracks:
                candidate = PredBox(
                    x1=track.x1 * frame.width,
                    y1=track.y1 * frame.height,
                    x2=track.x2 * frame.width,
                    y2=track.y2 * frame.height,
                    confidence=track.confidence,
                )
                score = iou(candidate, gt_box)
                if score > best_iou:
                    best_iou = score
                    best_track = track

            if best_track is not None and best_iou >= iou_threshold:
                if current_id is not None and best_track.track_id != current_id:
                    switches += 1
                current_id = best_track.track_id
            else:
                current_id = None

        total_switches += switches
        total_gt_present_frames += gt_present
        per_sequence.append(
            {
                "sequence_id": sequence.sequence_id,
                "id_switches": switches,
                "gt_present_frames": gt_present,
                "continuity": (1.0 - switches / gt_present) if measurable and gt_present > 0 else None,
            }
        )

    overall_continuity = (
        1.0 - (total_switches / total_gt_present_frames)
        if measurable and total_gt_present_frames > 0
        else None
    )
    return {
        "measurable": measurable,
        "overall_continuity": overall_continuity,
        "total_id_switches": total_switches,
        "total_gt_present_frames": total_gt_present_frames,
        "per_sequence": per_sequence,
        "note": None
        if measurable
        else (
            "Every sequence in this eval set has at most 1 frame — an ID switch is "
            "structurally impossible to observe, so track continuity is not measurable here."
        ),
    }
