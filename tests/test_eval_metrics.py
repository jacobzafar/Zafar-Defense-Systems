"""Unit tests for eval/metrics.py, with hand-computed expected values for
AP@0.5 specifically — see the comment on test_ap50_matches_hand_computed_example.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    FrameEvalData,
    PredBox,
    compute_ap50,
    compute_false_alarm_rate,
    compute_latency_stats,
    compute_small_object_recall,
    compute_track_continuity,
    iou,
)
from eval.schema import EvalBox, EvalFrame, EvalSequence


def test_iou_of_identical_boxes_is_one():
    a = EvalBox(x1=0, y1=0, x2=10, y2=10, category="drone")
    assert iou(a, a) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    a = EvalBox(x1=0, y1=0, x2=10, y2=10, category="drone")
    b = EvalBox(x1=100, y1=100, x2=110, y2=110, category="drone")
    assert iou(a, b) == 0.0


def test_ap50_matches_hand_computed_example():
    # 2 GT boxes, 3 predictions in confidence order: TP (perfect match GT1),
    # FP (matches nothing), TP (perfect match GT2).
    # By hand: tps=[1,0,1], fps=[0,1,0] -> cum_tp=[1,1,2], cum_fp=[0,1,1]
    # recalls=[0.5,0.5,1.0], precisions=[1.0,0.5,0.667]
    # envelope (backward max) = [1.0, 0.667, 0.667]
    # AP = (0.5-0)*1.0 + (0.5-0.5)*0.667 + (1.0-0.5)*0.667 = 0.5 + 0 + 0.3335 = 0.8333...
    gt1 = EvalBox(x1=0, y1=0, x2=10, y2=10, category="drone")
    gt2 = EvalBox(x1=100, y1=100, x2=110, y2=110, category="drone")
    pred1 = PredBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9)
    pred2 = PredBox(x1=200, y1=200, x2=210, y2=210, confidence=0.8)
    pred3 = PredBox(x1=100, y1=100, x2=110, y2=110, confidence=0.7)

    frame = FrameEvalData(frame_id="f1", gt_boxes=[gt1, gt2], pred_boxes=[pred1, pred2, pred3])
    ap = compute_ap50([frame])

    assert abs(ap - 0.8333333333) < 1e-6


def test_ap50_is_one_for_a_perfect_detector():
    gt = EvalBox(x1=0, y1=0, x2=10, y2=10, category="drone")
    pred = PredBox(x1=0, y1=0, x2=10, y2=10, confidence=0.99)
    frame = FrameEvalData(frame_id="f1", gt_boxes=[gt], pred_boxes=[pred])
    assert compute_ap50([frame]) == 1.0


def test_ap50_is_zero_with_no_predictions():
    gt = EvalBox(x1=0, y1=0, x2=10, y2=10, category="drone")
    frame = FrameEvalData(frame_id="f1", gt_boxes=[gt], pred_boxes=[])
    assert compute_ap50([frame]) == 0.0


def test_ap50_is_zero_with_no_ground_truth_at_all():
    frame = FrameEvalData(frame_id="f1", gt_boxes=[], pred_boxes=[PredBox(0, 0, 10, 10, 0.9)])
    assert compute_ap50([frame]) == 0.0


def test_small_object_recall_counts_only_boxes_under_threshold():
    small_gt = EvalBox(x1=0, y1=0, x2=10, y2=10, category="drone")  # area 100
    large_gt = EvalBox(x1=20, y1=20, x2=120, y2=120, category="drone")  # area 10000
    small_pred = PredBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9)
    frame = FrameEvalData(frame_id="f1", gt_boxes=[small_gt, large_gt], pred_boxes=[small_pred])

    result = compute_small_object_recall([frame], small_area_threshold_px=1024)
    assert result["num_small_gt_boxes"] == 1
    assert result["num_small_gt_matched"] == 1
    assert result["recall"] == 1.0


def test_small_object_recall_is_zero_with_no_small_boxes():
    large_gt = EvalBox(x1=0, y1=0, x2=100, y2=100, category="drone")
    frame = FrameEvalData(frame_id="f1", gt_boxes=[large_gt], pred_boxes=[])
    result = compute_small_object_recall([frame], small_area_threshold_px=1024)
    assert result["num_small_gt_boxes"] == 0
    assert result["recall"] == 0.0


def test_false_alarm_rate_only_counts_hard_negative_frames():
    hard_negative_with_fa = FrameEvalData(frame_id="fa1", gt_boxes=[], pred_boxes=[PredBox(0, 0, 5, 5, 0.5)])
    hard_negative_clean = FrameEvalData(frame_id="fa2", gt_boxes=[], pred_boxes=[])
    positive_frame = FrameEvalData(
        frame_id="pos1",
        gt_boxes=[EvalBox(0, 0, 10, 10, "drone")],
        pred_boxes=[PredBox(0, 0, 10, 10, 0.9), PredBox(50, 50, 60, 60, 0.4)],
    )

    result = compute_false_alarm_rate([hard_negative_with_fa, hard_negative_clean, positive_frame])
    assert result["num_hard_negative_frames"] == 2
    assert result["num_false_alarms"] == 1
    assert result["rate_per_frame"] == 0.5


def test_false_alarm_rate_is_zero_with_no_hard_negatives():
    positive_frame = FrameEvalData(frame_id="pos1", gt_boxes=[EvalBox(0, 0, 10, 10, "drone")], pred_boxes=[])
    result = compute_false_alarm_rate([positive_frame])
    assert result["num_hard_negative_frames"] == 0
    assert result["rate_per_frame"] == 0.0


def test_latency_stats_computes_mean_and_fps():
    stats = compute_latency_stats([10.0, 20.0, 30.0])
    assert stats["mean_ms"] == 20.0
    assert stats["median_ms"] == 20.0
    assert stats["fps"] == 50.0


def test_latency_stats_handles_empty_list():
    stats = compute_latency_stats([])
    assert stats == {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}


class _FakeTrack:
    def __init__(self, track_id, x1, y1, x2, y2, confidence=0.9):
        self.track_id = track_id
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.confidence = confidence


def _frame(sequence_id, index, width, height, gt_boxes):
    return EvalFrame(
        sequence_id=sequence_id, frame_index=index, image_path=Path("/dev/null"),
        width=width, height=height, boxes=gt_boxes,
    )


def test_track_continuity_is_one_when_id_never_switches():
    gt_box = EvalBox(x1=10, y1=10, x2=20, y2=20, category="drone")
    frames = [_frame("seq1", i, 100, 100, [gt_box]) for i in range(3)]
    sequence = EvalSequence(sequence_id="seq1", frames=frames)

    # Track ID stays 1 in every frame, always overlapping GT well.
    same_track = [_FakeTrack(1, 0.1, 0.1, 0.2, 0.2)]
    tracks_by_sequence = {"seq1": [same_track, same_track, same_track]}

    result = compute_track_continuity([sequence], tracks_by_sequence, iou_threshold=0.3)
    assert result["overall_continuity"] == 1.0
    assert result["total_id_switches"] == 0


def test_track_continuity_detects_an_id_switch():
    gt_box = EvalBox(x1=10, y1=10, x2=20, y2=20, category="drone")
    frames = [_frame("seq1", i, 100, 100, [gt_box]) for i in range(3)]
    sequence = EvalSequence(sequence_id="seq1", frames=frames)

    tracks_by_sequence = {
        "seq1": [
            [_FakeTrack(1, 0.1, 0.1, 0.2, 0.2)],
            [_FakeTrack(1, 0.1, 0.1, 0.2, 0.2)],
            [_FakeTrack(2, 0.1, 0.1, 0.2, 0.2)],  # ID switch on the 3rd frame
        ]
    }

    result = compute_track_continuity([sequence], tracks_by_sequence, iou_threshold=0.3)
    assert result["total_id_switches"] == 1
    assert abs(result["overall_continuity"] - (1 - 1 / 3)) < 1e-9


def test_track_continuity_ignores_gaps_where_gt_is_absent():
    gt_present = EvalBox(x1=10, y1=10, x2=20, y2=20, category="drone")
    frames = [
        _frame("seq1", 0, 100, 100, [gt_present]),
        _frame("seq1", 1, 100, 100, []),  # GT absent — a gap, not a switch
        _frame("seq1", 2, 100, 100, [gt_present]),
    ]
    sequence = EvalSequence(sequence_id="seq1", frames=frames)

    tracks_by_sequence = {
        "seq1": [
            [_FakeTrack(1, 0.1, 0.1, 0.2, 0.2)],
            [],
            [_FakeTrack(2, 0.1, 0.1, 0.2, 0.2)],  # different ID, but GT was absent in between
        ]
    }

    result = compute_track_continuity([sequence], tracks_by_sequence, iou_threshold=0.3)
    assert result["total_id_switches"] == 0
    assert result["overall_continuity"] == 1.0
