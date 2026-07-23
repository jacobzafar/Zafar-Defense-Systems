"""Plumbing tests for detector/train.py.

These only prove the fine-tuning loop runs end-to-end (loads a tiny
synthetic dataset, trains a couple of steps, writes weights + a report)
without a GPU or any real drone data. They are not a benchmark and must
never be read as one — see the "note" field in the report itself and
eval/ for the frozen evaluation harness that produces real metrics.
"""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from detector.datasets.manifest import DATASET_REGISTRY, DatasetManifest, LicenseStatus
from detector.train import CommercialLicenseRequiredError, TrainConfig, train
from eval.build_frozen_eval_set import build_frozen_eval_set
from eval.schema import EvalSetFrozenError


def test_train_config_from_yaml_round_trips(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            dataset_dir: /tmp/does-not-matter
            output_dir: /tmp/does-not-matter-out
            epochs: 3
            seed: 7
            """
        )
    )
    config = TrainConfig.from_yaml(config_path)
    assert config.epochs == 3
    assert config.seed == 7
    assert config.target_class == "drone"  # default preserved


def test_train_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset_dir: /tmp/x\noutput_dir: /tmp/y\nnot_a_real_field: 1\n")
    with pytest.raises(ValueError, match="Unknown key"):
        TrainConfig.from_yaml(config_path)


def test_train_runs_end_to_end_on_synthetic_dataset(synthetic_drone_dataset, tmp_path):
    output_dir = tmp_path / "train_output"
    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        val_fraction=0.2,
        seed=42,
    )

    report = train(config)

    assert (output_dir / "weights.pt").exists()
    assert (output_dir / "training_report.json").exists()
    assert report["num_train_images"] == 5
    assert report["num_val_images"] == 1
    assert report["num_drone_boxes_train"] > 0
    assert report["num_hard_negative_images_train"] > 0
    assert len(report["epoch_losses"]) == 1
    assert report["final_loss"] is not None
    assert "benchmark" in report["note"].lower()


def test_commercial_only_without_dataset_name_raises(synthetic_drone_dataset, tmp_path):
    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        commercial_only=True,
    )
    with pytest.raises(CommercialLicenseRequiredError, match="dataset_name"):
        train(config)


def test_commercial_only_rejects_unconfirmed_dataset(synthetic_drone_dataset, tmp_path):
    """Every registry entry defaults commercial_ok=None (unknown) — see
    tests/test_dataset_manifest.py — so --commercial-only must refuse
    even a real, registered dataset until a maintainer confirms it."""
    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        dataset_name="dut-anti-uav",
        commercial_only=True,
    )
    with pytest.raises(CommercialLicenseRequiredError, match="commercial_ok"):
        train(config)


def test_commercial_only_allows_a_confirmed_dataset(synthetic_drone_dataset, tmp_path, monkeypatch):
    fake_manifest = DatasetManifest(
        name="Fake Cleared Dataset",
        description="test fixture",
        classes=["drone"],
        license=LicenseStatus.PERMISSIVE_CONFIRMED,
        license_notes="test fixture",
        source_url="https://example.invalid/fake",
        license_id="Fake-Permissive-1.0",
        commercial_ok=True,
    )
    monkeypatch.setitem(DATASET_REGISTRY, "fake-cleared-dataset", fake_manifest)

    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        dataset_name="fake-cleared-dataset",
        commercial_only=True,
    )
    report = train(config)
    assert report["num_train_images"] > 0


def test_train_refuses_a_frozen_eval_set_built_from_a_real_dataset(synthetic_drone_dataset, tmp_path):
    """End-to-end proof (not just eval.schema's own unit test) that
    detector/train.py cannot be pointed at an eval set that went through
    the real build_frozen_eval_set() path used to register the DUT
    Anti-UAV test split — see docs/datasets.md / eval/README.md."""
    frozen_eval_set_dir = tmp_path / "frozen_eval"
    build_frozen_eval_set(synthetic_drone_dataset, frozen_eval_set_dir)

    config = TrainConfig(dataset_dir=str(frozen_eval_set_dir), output_dir=str(tmp_path / "out"), epochs=1)
    with pytest.raises(EvalSetFrozenError, match="reserved for evaluation only"):
        train(config)


def test_grad_clipping_prevents_nan_loss_that_an_unclipped_run_hits(synthetic_drone_dataset, tmp_path):
    """Regression test for a real bug: a first fine-tuning run against
    DUT Anti-UAV's real train split diverged to NaN weights within one
    epoch at this script's previous learning_rate default (0.005) — the
    freshly-initialized single-class head produces large early gradients
    a tiny synthetic smoke test never ran long enough to expose. This
    reproduces the same failure mode (a too-high learning rate) on the
    synthetic fixture instead, and proves grad_clip_max_norm actually
    prevents it rather than merely existing as an unused config field.
    See docs/DECISIONS.md for the full diagnosis on real data.
    """
    unclipped_config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(tmp_path / "unclipped"),
        epochs=10,
        batch_size=2,
        val_fraction=0.2,
        seed=42,
        learning_rate=0.5,
        grad_clip_max_norm=1e12,  # effectively disabled
    )
    unclipped_report = train(unclipped_config)
    assert any(loss != loss for loss in unclipped_report["epoch_losses"])  # NaN != NaN

    clipped_config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(tmp_path / "clipped"),
        epochs=10,
        batch_size=2,
        val_fraction=0.2,
        seed=42,
        learning_rate=0.5,
        grad_clip_max_norm=1.0,
    )
    clipped_report = train(clipped_config)
    assert all(loss == loss for loss in clipped_report["epoch_losses"])  # no NaN
    assert clipped_report["final_loss"] is not None


def test_train_raises_clearly_on_empty_dataset(tmp_path):
    import json

    dataset_dir = tmp_path / "empty_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "images").mkdir()
    (dataset_dir / "annotations.json").write_text(json.dumps({"images": [], "annotations": []}))

    config = TrainConfig(dataset_dir=str(dataset_dir), output_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="No images found"):
        train(config)
