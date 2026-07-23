import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.schema import (
    EvalSetChecksumMismatchError,
    EvalSetFormatError,
    assert_not_for_training,
    compute_manifest_checksum,
    freeze_eval_set,
    is_frozen,
    load_eval_manifest,
    verify_checksum,
)


def test_load_synthetic_eval_set_returns_expected_structure(synthetic_eval_set):
    sequences = load_eval_manifest(synthetic_eval_set)
    assert len(sequences) == 1
    assert len(sequences[0].frames) == 8

    drone_frames = [f for f in sequences[0].frames if not f.is_hard_negative]
    hard_negative_frames = [f for f in sequences[0].frames if f.is_hard_negative]
    assert len(drone_frames) == 6
    assert len(hard_negative_frames) == 2

    for frame in sequences[0].frames:
        assert frame.image_path.exists()


def test_synthetic_eval_set_is_frozen_by_default(synthetic_eval_set):
    assert is_frozen(synthetic_eval_set)


def test_freeze_eval_set_is_idempotent(tmp_path):
    freeze_eval_set(tmp_path)
    freeze_eval_set(tmp_path)  # must not raise
    assert is_frozen(tmp_path)


def test_assert_not_for_training_raises_on_frozen_set(synthetic_eval_set):
    with pytest.raises(RuntimeError, match="must never be used as a training"):
        assert_not_for_training(synthetic_eval_set)


def test_assert_not_for_training_allows_non_frozen_dir(tmp_path):
    assert_not_for_training(tmp_path)  # should not raise — no .frozen marker


def test_checksum_is_stable_across_reloads(synthetic_eval_set):
    checksum_a = compute_manifest_checksum(synthetic_eval_set)
    checksum_b = compute_manifest_checksum(synthetic_eval_set)
    assert checksum_a == checksum_b
    assert len(checksum_a) == 64  # sha256 hex digest


def test_verify_checksum_passes_for_correct_checksum(synthetic_eval_set):
    checksum = compute_manifest_checksum(synthetic_eval_set)
    verify_checksum(synthetic_eval_set, checksum)  # should not raise


def test_verify_checksum_fails_for_wrong_checksum(synthetic_eval_set):
    with pytest.raises(EvalSetChecksumMismatchError):
        verify_checksum(synthetic_eval_set, "0" * 64)


def test_verify_checksum_detects_manifest_edit(synthetic_eval_set):
    checksum = compute_manifest_checksum(synthetic_eval_set)
    manifest_path = synthetic_eval_set / "sequences.json"
    manifest_path.write_text(manifest_path.read_text() + "\n")  # trivial edit
    with pytest.raises(EvalSetChecksumMismatchError):
        verify_checksum(synthetic_eval_set, checksum)


def test_missing_sequences_json_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="sequences.json"):
        load_eval_manifest(tmp_path)


def test_malformed_sequences_json_raises_format_error(tmp_path):
    (tmp_path / "sequences.json").write_text("not valid json{")
    with pytest.raises(EvalSetFormatError):
        load_eval_manifest(tmp_path)
