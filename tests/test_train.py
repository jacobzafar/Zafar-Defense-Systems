"""Plumbing tests for detector/train.py.

These only prove the fine-tuning loop runs end-to-end (loads a tiny
synthetic dataset, trains a couple of steps, writes weights + a report)
without a GPU or any real drone data. They are not a benchmark and must
never be read as one — see the "note" field in the report itself and
eval/ for the frozen evaluation harness that produces real metrics.
"""

import json
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


def test_device_auto_resolves_to_cuda_only_when_available(monkeypatch):
    import torch

    from detector.train import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cpu") == "cpu"  # explicit config value overrides auto-detection


def test_checkpoint_written_every_epoch_and_interrupted_run_resumes(synthetic_drone_dataset, tmp_path, monkeypatch):
    """Simulates a Colab disconnect: a 2-epoch run is killed during epoch 2.
    Re-running the same config must resume from the epoch-1 checkpoint
    (not restart), and end with exactly the same per-epoch losses and
    weights as a run that was never interrupted."""
    import torch

    import detector.train as train_module

    def make_config(output_dir):
        return TrainConfig(
            dataset_dir=str(synthetic_drone_dataset),
            output_dir=str(output_dir),
            epochs=2,
            batch_size=2,
            val_fraction=0.2,
            seed=42,
            val_dataset_dir=str(synthetic_drone_dataset),
            val_max_images=3,
        )

    uninterrupted = train(make_config(tmp_path / "uninterrupted"))

    interrupted_dir = tmp_path / "interrupted"
    real_evaluate = train_module.evaluate_on_val
    calls = {"n": 0}

    def evaluate_then_disconnect_on_epoch_2(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated Colab disconnect")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(train_module, "evaluate_on_val", evaluate_then_disconnect_on_epoch_2)
    with pytest.raises(KeyboardInterrupt):
        train(make_config(interrupted_dir))
    monkeypatch.setattr(train_module, "evaluate_on_val", real_evaluate)

    checkpoint = torch.load(interrupted_dir / "checkpoint.pt", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert {"model_state", "optimizer_state"} <= set(checkpoint)
    assert not (interrupted_dir / "weights.pt").exists()
    partial_report = json.loads((interrupted_dir / "training_report.json").read_text())
    assert partial_report["status"] == "in_progress"
    assert partial_report["epochs_completed"] == 1

    resumed = train(make_config(interrupted_dir))

    assert resumed["resumed_from_epoch"] == 1
    assert resumed["status"] == "complete"
    assert resumed["epoch_losses"] == uninterrupted["epoch_losses"]
    assert [v["epoch"] for v in resumed["val_per_epoch"]] == [1, 2]
    assert resumed["val_per_epoch"] == uninterrupted["val_per_epoch"]
    resumed_weights = torch.load(interrupted_dir / "weights.pt")
    uninterrupted_weights = torch.load(tmp_path / "uninterrupted" / "weights.pt")
    for key, value in uninterrupted_weights.items():
        assert torch.equal(resumed_weights[key], value), key


def test_val_metrics_report_ap_and_raw_score_spread(synthetic_drone_dataset, tmp_path):
    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        val_dataset_dir=str(synthetic_drone_dataset),
    )
    report = train(config)

    (val,) = report["val_per_epoch"]
    assert val["epoch"] == 1
    assert val["num_val_images"] == 6
    assert 0.0 <= val["ap50"] <= 1.0
    assert val["num_raw_scores"] > 0
    assert val["raw_score_min"] <= val["raw_score_max"]
    assert val["raw_score_spread"] == pytest.approx(val["raw_score_max"] - val["raw_score_min"], abs=1e-5)


def test_resume_refuses_a_checkpoint_from_different_settings(synthetic_drone_dataset, tmp_path):
    output_dir = tmp_path / "out"
    base = dict(dataset_dir=str(synthetic_drone_dataset), output_dir=str(output_dir), epochs=1, batch_size=2)
    train(TrainConfig(**base))

    with pytest.raises(ValueError, match="different settings"):
        train(TrainConfig(**{**base, "epochs": 2, "learning_rate": 0.01}))


def test_val_dataset_dir_refuses_the_frozen_test_set(synthetic_drone_dataset, tmp_path):
    frozen_eval_set_dir = tmp_path / "frozen_eval"
    build_frozen_eval_set(synthetic_drone_dataset, frozen_eval_set_dir)

    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        val_dataset_dir=str(frozen_eval_set_dir),
    )
    with pytest.raises(EvalSetFrozenError):
        train(config)


def test_input_size_changes_only_the_resize_and_keeps_relative_anchors():
    import torch

    from detector.train import build_single_class_model

    model = build_single_class_model(input_size=640).eval()
    assert model.transform.fixed_size == (640, 640)
    images, _ = model.transform([torch.zeros(3, 1080, 1920)])
    assert tuple(images.tensors.shape[-2:]) == (640, 640)
    features = list(model.backbone(images.tensors).values())
    assert tuple(features[0].shape[-2:]) == (40, 40)  # 20x20 at the stock 320
    anchors = model.anchor_generator(images, features)[0]
    sides = ((anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])).sqrt()
    assert float(sides.min()) == pytest.approx(0.2 * 640)  # smallest anchor stays 0.2 of the input side


def test_train_at_a_non_default_input_size_and_detect_at_the_same_size(synthetic_drone_dataset, tmp_path):
    import numpy as np

    from detector.drone_detector import DroneDetector

    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset), output_dir=str(tmp_path / "out"), epochs=1, batch_size=2, input_size=160
    )
    report = train(config)
    assert report["config"]["input_size"] == 160

    detector = DroneDetector(weights_path=report["weights_path"], confidence_threshold=0.0, input_size=160)
    assert detector._model.transform.fixed_size == (160, 160)
    assert isinstance(detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)), list)


def test_resume_refuses_a_different_input_size_but_treats_old_checkpoints_as_320(synthetic_drone_dataset, tmp_path):
    import torch

    output_dir = tmp_path / "out"
    base = dict(dataset_dir=str(synthetic_drone_dataset), output_dir=str(output_dir), batch_size=2)
    train(TrainConfig(**base, epochs=1))

    with pytest.raises(ValueError, match="input_size"):
        train(TrainConfig(**base, epochs=2, input_size=640))

    # A checkpoint written before input_size existed has no such key: it was a 320 run.
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    del checkpoint["config"]["input_size"]
    torch.save(checkpoint, checkpoint_path)
    report = train(TrainConfig(**base, epochs=2))
    assert report["resumed_from_epoch"] == 1
