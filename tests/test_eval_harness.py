"""End-to-end tests for eval/harness.py against the synthetic eval-set
fixture, using the zero-dependency `motion` detector so these stay fast
and torch-free. Never treat these numbers as a real accuracy result —
see eval/README.md.
"""

import json
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import EvalConfig, EvalConfigError, run_evaluation, write_outputs
from eval.schema import EvalSetChecksumMismatchError, compute_manifest_checksum


def _make_config(eval_set_dir: Path, output_dir: Path, **overrides) -> EvalConfig:
    defaults = dict(
        eval_set_dir=str(eval_set_dir),
        detector={"backend": "motion", "min_area_px": 20, "var_threshold": 16.0, "history": 50},
        tracker={"backend": "iou"},
        output_json_path=str(output_dir / "metric_card.json"),
        output_report_path=str(output_dir / "REPORT.md"),
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)


def test_run_evaluation_produces_a_complete_metric_card(synthetic_eval_set, tmp_path):
    config = _make_config(synthetic_eval_set, tmp_path / "out")
    metric_card = run_evaluation(config)

    assert metric_card.num_sequences == 1
    assert metric_card.num_frames == 8
    assert 0.0 <= metric_card.ap50 <= 1.0
    assert metric_card.small_object_recall["num_small_gt_boxes"] > 0
    assert metric_card.false_alarm_rate["num_hard_negative_frames"] == 2
    assert metric_card.latency["mean_ms"] >= 0.0
    assert "overall_continuity" in metric_card.track_continuity
    assert len(metric_card.manifest_checksum) == 64


def test_write_outputs_creates_json_and_report(synthetic_eval_set, tmp_path):
    output_dir = tmp_path / "out"
    config = _make_config(synthetic_eval_set, output_dir)
    metric_card = run_evaluation(config)
    write_outputs(metric_card, config)

    json_path = Path(config.output_json_path)
    report_path = Path(config.output_report_path)
    assert json_path.exists()
    assert report_path.exists()

    written = json.loads(json_path.read_text())
    assert written["num_frames"] == 8

    report_text = report_path.read_text()
    assert "Baseline comparison" in report_text
    assert "frozen and must never be used for training" in report_text


def test_checksum_mismatch_is_caught(synthetic_eval_set, tmp_path):
    config = _make_config(synthetic_eval_set, tmp_path / "out", expected_checksum="0" * 64)
    with pytest.raises(EvalSetChecksumMismatchError):
        run_evaluation(config)


def test_correct_checksum_passes(synthetic_eval_set, tmp_path):
    checksum = compute_manifest_checksum(synthetic_eval_set)
    config = _make_config(synthetic_eval_set, tmp_path / "out", expected_checksum=checksum)
    run_evaluation(config)  # should not raise


def test_non_frozen_eval_set_warns_but_still_runs(tmp_path):
    from tests.conftest import build_synthetic_eval_set

    unfrozen_dir = build_synthetic_eval_set(tmp_path / "unfrozen", freeze=False)
    config = _make_config(unfrozen_dir, tmp_path / "out")
    metric_card = run_evaluation(config)  # should not raise, just warn
    assert metric_card.num_frames == 8


def test_eval_config_from_yaml(tmp_path, synthetic_eval_set):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            eval_set_dir: {synthetic_eval_set}
            detector:
              backend: motion
            """
        )
    )
    config = EvalConfig.from_yaml(config_path)
    assert config.eval_set_dir == str(synthetic_eval_set)
    assert config.iou_threshold == 0.5  # default preserved


def test_eval_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("eval_set_dir: /tmp/x\ndetector: {backend: motion}\nnot_a_real_field: 1\n")
    with pytest.raises(EvalConfigError, match="Unknown key"):
        EvalConfig.from_yaml(config_path)


def test_missing_eval_set_raises_file_not_found(tmp_path):
    config = _make_config(tmp_path / "does-not-exist", tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        run_evaluation(config)
