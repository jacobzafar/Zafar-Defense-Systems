import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry.summary import format_summary, summarize_log


def _write_log(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def test_summarize_missing_file_returns_empty_summary(tmp_path):
    summary = summarize_log(tmp_path / "does-not-exist.jsonl")
    assert summary.total_frames == 0
    assert not summary.has_data


def test_summarize_typical_run(tmp_path):
    log_path = tmp_path / "run.jsonl"
    _write_log(
        log_path,
        [
            {"event_type": "run_started", "timestamp": 100.0, "payload": {"source": "demo", "detector": "motion_v1", "tracker": "iou_v1"}},
            {"event_type": "frame_processed", "timestamp": 100.1, "payload": {"detection_count": 2, "tracks": [{"track_id": 1}, {"track_id": 2}], "fps": 30.0}},
            {"event_type": "frame_processed", "timestamp": 100.2, "payload": {"detection_count": 1, "tracks": [{"track_id": 1}], "fps": 32.0}},
            {"event_type": "frame_dropped", "timestamp": 100.3, "payload": {"error": "boom"}},
            {"event_type": "run_stopped", "timestamp": 100.4, "payload": {}},
        ],
    )

    summary = summarize_log(log_path)
    assert summary.has_data
    assert summary.source == "demo"
    assert summary.detector == "motion_v1"
    assert summary.tracker == "iou_v1"
    assert summary.total_frames == 2
    assert summary.total_detections == 3
    assert summary.unique_track_count == 2
    assert summary.dropped_frames == 1
    assert summary.duration_seconds == round(100.4 - 100.0, 3)
    assert summary.avg_fps == 31.0


def test_summarize_tolerates_malformed_lines(tmp_path):
    log_path = tmp_path / "run.jsonl"
    log_path.write_text(
        'not-json\n{"event_type": "frame_processed", "timestamp": 1.0, "payload": {"detection_count": 1, "tracks": []}}\n'
    )
    summary = summarize_log(log_path)
    assert summary.total_frames == 1
    assert summary.total_detections == 1


def test_format_summary_includes_key_fields(tmp_path):
    log_path = tmp_path / "run.jsonl"
    _write_log(
        log_path,
        [{"event_type": "run_started", "timestamp": 1.0, "payload": {"source": "demo"}}],
    )
    summary = summarize_log(log_path)
    text = format_summary(summary)
    assert "source" in text
    assert "demo" in text
