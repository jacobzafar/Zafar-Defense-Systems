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


def compute_ap50(frames: list[FrameEvalData], iou_threshold: float = 0.5) -> float:
    """Single-class AP at the given IoU threshold (default 0.5).

    Returns 0.0 if there are no ground-truth boxes at all across `frames`
    (AP is undefined in that case; 0.0 is a safe, JSON-serializable
    default rather than NaN).
    """
    total_gt = sum(len(f.gt_boxes) for f in frames)
    if total_gt == 0:
        return 0.0

    gt_matched: dict[str, list[bool]] = {f.frame_id: [False] * len(f.gt_boxes) for f in frames}
    gt_by_frame: dict[str, list[EvalBox]] = {f.frame_id: f.gt_boxes for f in frames}

    all_preds: list[tuple[float, str, PredBox]] = [
        (pred.confidence, f.frame_id, pred) for f in frames for pred in f.pred_boxes
    ]
    all_preds.sort(key=lambda item: item[0], reverse=True)

    tps = np.zeros(len(all_preds))
    fps = np.zeros(len(all_preds))

    for i, (_confidence, frame_id, pred) in enumerate(all_preds):
        gts = gt_by_frame[frame_id]
        matched = gt_matched[frame_id]
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if matched[idx]:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched[best_idx] = True
            tps[i] = 1
        else:
            fps[i] = 1

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    envelope = precisions.copy()
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])

    ap = 0.0
    prev_recall = 0.0
    for i in range(len(recalls)):
        ap += (recalls[i] - prev_recall) * envelope[i]
        prev_recall = recalls[i]

    return float(ap)


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
