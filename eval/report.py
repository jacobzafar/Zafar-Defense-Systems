"""Render a `MetricCard` (eval/harness.py) as a human-readable Markdown report."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.harness import EvalConfig, MetricCard


def render_report_md(metric_card: "MetricCard", config: "EvalConfig") -> str:
    sor = metric_card.small_object_recall
    far = metric_card.false_alarm_rate
    lat = metric_card.latency
    cont = metric_card.track_continuity

    far_str = (
        f"{far['rate_per_frame']:.4f} / frame"
        if far["measurable"]
        else f"**not measurable** ({far['note']})"
    )
    cont_str = (
        f"{cont['overall_continuity']:.4f}"
        if cont["measurable"] and cont["overall_continuity"] is not None
        else f"**not measurable** ({cont['note']})"
    )

    lines = [
        "# Evaluation Report",
        "",
        f"- Evaluated at: {metric_card.evaluated_at}",
        f"- Eval set: `{metric_card.eval_set_dir}`",
        f"- Eval set manifest checksum (sequences.json sha256): `{metric_card.manifest_checksum}`",
        f"- Detector backend: `{metric_card.detector_backend}`",
        f"- Tracker backend: `{metric_card.tracker_backend}`",
        f"- Sequences: {metric_card.num_sequences}  |  Frames: {metric_card.num_frames}",
        "",
        "**This eval set is frozen and must never be used for training** — see eval/README.md.",
        "",
        "## Metric card",
        "",
        "| Metric | Value | Definition |",
        "|---|---|---|",
        f"| AP@{config.iou_threshold} (single-class \"drone\") | {metric_card.ap50:.4f} | Average precision at IoU={config.iou_threshold}, all-point interpolation. |",
        f"| Small-object recall | {sor['recall']:.4f} ({sor['num_small_gt_matched']}/{sor['num_small_gt_boxes']}) | Recall on GT boxes with area < {sor['area_threshold_px']:.0f}px² (COCO \"small\" convention). |",
        f"| False-alarm rate | {far_str} | {far['num_false_alarms']} false detections across {far['num_hard_negative_frames']} hard-negative (no-drone) frames. |",
        f"| Latency (mean) | {lat['mean_ms']:.2f} ms | Wall-clock detector.detect() time per frame, this run's hardware only. |",
        f"| Latency (p95) | {lat['p95_ms']:.2f} ms | |",
        f"| Throughput | {lat['fps']:.2f} FPS | 1000 / mean latency. |",
        f"| Track continuity | {cont_str} | 1 - (ID switches / frames with GT present); {cont['total_id_switches']} switches across {cont['total_gt_present_frames']} frames. |",
        "",
        "## Per-sequence track continuity",
        "",
    ]

    if not cont["measurable"]:
        lines += [
            f"Not measurable — {cont['note']} "
            f"({len(cont['per_sequence'])} single-frame sequences omitted from a per-sequence table; "
            f"every one has 0 ID switches by construction, which is not evidence of anything.)",
            "",
        ]
    else:
        lines += ["| Sequence | ID switches | Frames w/ GT present | Continuity |", "|---|---|---|---|"]
        for seq in cont["per_sequence"]:
            continuity_str = f"{seq['continuity']:.4f}" if seq["continuity"] is not None else "n/a (no GT present)"
            lines.append(f"| {seq['sequence_id']} | {seq['id_switches']} | {seq['gt_present_frames']} | {continuity_str} |")
        lines.append("")

    lines += [
        "",
        "## Baseline comparison",
        "",
        "_Paste published baseline numbers below for comparison. Nothing in this row is",
        "measured by this repo — fill it in manually from the source paper/benchmark you",
        "are comparing against, and cite it._",
        "",
        "| Source | AP@0.5 | Small-object recall | False-alarm rate | FPS | Notes |",
        "|---|---|---|---|---|---|",
        "| _(paste published baseline here)_ | | | | | |",
        f"| This run (`{metric_card.detector_backend}`) | {metric_card.ap50:.4f} | {sor['recall']:.4f} | "
        f"{far_str if far['measurable'] else 'not measurable'} | {lat['fps']:.2f} | |",
        "",
    ]

    return "\n".join(lines)
