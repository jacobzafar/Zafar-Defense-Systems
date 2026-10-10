"""Plumbing test for eval/select_checkpoint.py: per-epoch weights written
by detector/train.py can each be evaluated and ranked. Synthetic data —
the AP values here mean nothing beyond "the ranking is well-formed"."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from detector.train import TrainConfig, train
from eval.harness import EvalConfig
from eval.select_checkpoint import rank_checkpoints


def test_per_epoch_weights_are_kept_and_ranked(synthetic_drone_dataset, synthetic_eval_set, tmp_path):
    output_dir = tmp_path / "train_out"
    report = train(
        TrainConfig(dataset_dir=str(synthetic_drone_dataset), output_dir=str(output_dir), epochs=2, batch_size=2)
    )
    epoch_weights = report["epoch_weights_paths"]
    assert [Path(p).name for p in epoch_weights] == ["weights_epoch_001.pt", "weights_epoch_002.pt"]
    assert all(Path(p).exists() for p in epoch_weights)

    config = EvalConfig(
        eval_set_dir=str(synthetic_eval_set),
        detector={"backend": "drone", "weights_path": "overridden", "confidence_threshold": 0.35},
    )
    rows = rank_checkpoints(config, epoch_weights)

    assert sorted(row["weights_path"] for row in rows) == sorted(epoch_weights)
    assert rows[0]["ap50"] >= rows[1]["ap50"]
    assert config.detector["weights_path"] == "overridden"  # caller's config not mutated
