"""Tests for eval/report.py's Markdown rendering, focused on the
"not measurable" states (false-alarm rate, track continuity) that must
never be silently rendered as a misleading 0.0/1.0 — see eval/metrics.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import EvalConfig, MetricCard
from eval.report import render_report_md


def _base_metric_card(**overrides) -> MetricCard:
    defaults = dict(
        eval_set_dir="data/frozen_eval/dut-anti-uav-test",
        detector_backend="drone_v1",
        tracker_backend="iou",
        num_sequences=3,
        num_frames=3,
        ap50=0.5,
        ap50_95=0.25,
        ap50_by_size={
            "small": {"ap": 0.4, "num_gt_boxes": 2, "area_range_px": [0.0, 1024.0]},
            "medium": {"ap": 0.6, "num_gt_boxes": 1, "area_range_px": [1024.0, 9216.0]},
            "large": {"ap": 0.8, "num_gt_boxes": 1, "area_range_px": [9216.0, None]},
        },
        operating_point={
            "confidence_threshold": 0.35,
            "iou_threshold": 0.5,
            "recall": 0.6667,
            "precision": 0.5,
            "num_gt_boxes": 3,
            "num_true_positives": 2,
            "num_false_positives": 2,
        },
        small_object_recall={"recall": 0.5, "num_small_gt_matched": 1, "num_small_gt_boxes": 2, "area_threshold_px": 1024.0},
        false_alarm_rate={
            "measurable": False,
            "rate_per_frame": None,
            "num_hard_negative_frames": 0,
            "num_false_alarms": 0,
            "note": "No hard-negative (zero-GT) frames in this eval set — false-alarm rate is not measurable here.",
        },
        latency={"mean_ms": 10.0, "median_ms": 10.0, "p95_ms": 12.0, "fps": 100.0},
        track_continuity={
            "measurable": False,
            "overall_continuity": None,
            "total_id_switches": 0,
            "total_gt_present_frames": 3,
            "per_sequence": [
                {"sequence_id": f"img_{i}", "id_switches": 0, "gt_present_frames": 1, "continuity": None} for i in range(3)
            ],
            "note": (
                "Every sequence in this eval set has at most 1 frame — an ID switch is "
                "structurally impossible to observe, so track continuity is not measurable here."
            ),
        },
        manifest_checksum="0" * 64,
        evaluated_at="2026-07-22T00:00:00",
    )
    defaults.update(overrides)
    return MetricCard(**defaults)


def _config() -> EvalConfig:
    return EvalConfig(eval_set_dir="data/frozen_eval/dut-anti-uav-test", detector={"backend": "drone"})


def test_not_measurable_metrics_render_explicitly_not_as_zero_or_perfect():
    report = render_report_md(_base_metric_card(), _config())

    assert "**not measurable**" in report
    assert "No hard-negative" in report
    assert "structurally impossible to observe" in report
    # Must not render as if it were an actual measured 0.0 rate or 1.0 continuity.
    assert "0.0000 / frame" not in report
    assert "| Track continuity | 1.0000 |" not in report


def test_not_measurable_track_continuity_collapses_the_per_sequence_table():
    """2200 single-frame sequences must not produce 2200 identical
    "not measurable" table rows — that's real output this repo generated
    once and is exactly the noise this collapsed rendering avoids."""
    report = render_report_md(_base_metric_card(), _config())
    per_sequence_rows = [line for line in report.splitlines() if line.startswith("| img_")]
    assert per_sequence_rows == []  # not one row per sequence, regardless of how many there are
    assert "single-frame sequences omitted" in report


def test_measurable_metrics_still_render_numeric_values_and_a_per_sequence_table():
    metric_card = _base_metric_card(
        false_alarm_rate={"measurable": True, "rate_per_frame": 0.25, "num_hard_negative_frames": 4, "num_false_alarms": 1, "note": None},
        track_continuity={
            "measurable": True,
            "overall_continuity": 0.9,
            "total_id_switches": 1,
            "total_gt_present_frames": 10,
            "per_sequence": [{"sequence_id": "seq_001", "id_switches": 1, "gt_present_frames": 10, "continuity": 0.9}],
            "note": None,
        },
    )
    report = render_report_md(metric_card, _config())

    assert "0.2500 / frame" in report
    assert "0.9000" in report
    assert "not measurable" not in report
    assert "| seq_001 | 1 | 10 | 0.9000 |" in report


def test_report_renders_threshold_independent_ap_separately_from_the_operating_point():
    by_size = dict(_base_metric_card().ap50_by_size)
    by_size["large"] = {"ap": None, "num_gt_boxes": 0, "area_range_px": [9216.0, None]}
    report = render_report_md(_base_metric_card(ap50_by_size=by_size), _config())

    assert "AP@[0.50:0.95] | 0.2500" in report
    assert "AP@0.5 — small | 0.4000" in report
    assert "AP@0.5 — large | **not measurable** (no GT boxes)" in report
    assert "Recall @ conf ≥ 0.35 | 0.6667 (2/3)" in report
