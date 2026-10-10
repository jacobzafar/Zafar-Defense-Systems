import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.build_frozen_eval_set import build_frozen_eval_set
from eval.schema import compute_manifest_checksum, is_frozen, load_eval_manifest


def test_build_frozen_eval_set_wraps_each_image_as_a_one_frame_sequence(synthetic_drone_dataset, tmp_path):
    eval_set_dir = tmp_path / "frozen_eval"
    checksum = build_frozen_eval_set(synthetic_drone_dataset, eval_set_dir)

    assert is_frozen(eval_set_dir)
    assert checksum == compute_manifest_checksum(eval_set_dir)

    sequences = load_eval_manifest(eval_set_dir)
    assert len(sequences) == 6  # synthetic_drone_dataset has 6 images
    assert all(len(seq.frames) == 1 for seq in sequences)
    assert all(seq.frames[0].image_path.exists() for seq in sequences)

    drone_sequences = [s for s in sequences if not s.frames[0].is_hard_negative]
    hard_negative_sequences = [s for s in sequences if s.frames[0].is_hard_negative]
    assert len(drone_sequences) == 3
    assert len(hard_negative_sequences) == 3


def test_build_frozen_eval_set_is_idempotent(synthetic_drone_dataset, tmp_path):
    eval_set_dir = tmp_path / "frozen_eval"
    first_checksum = build_frozen_eval_set(synthetic_drone_dataset, eval_set_dir)
    second_checksum = build_frozen_eval_set(synthetic_drone_dataset, eval_set_dir)
    assert first_checksum == second_checksum


def test_raises_on_empty_source_dataset(tmp_path):
    empty_dataset_dir = tmp_path / "empty"
    (empty_dataset_dir / "images").mkdir(parents=True)
    (empty_dataset_dir / "annotations.json").write_text(json.dumps({"images": [], "annotations": []}))

    with pytest.raises(ValueError, match="No images found"):
        build_frozen_eval_set(empty_dataset_dir, tmp_path / "frozen_eval")


def test_max_images_keeps_the_same_seeded_subset_train_py_validates_on(synthetic_drone_dataset, tmp_path):
    import random

    from detector.datasets.loader import load_manifest_dataset

    eval_set_dir = tmp_path / "subset"
    build_frozen_eval_set(synthetic_drone_dataset, eval_set_dir, max_images=3, seed=42)

    expected = [s.image_path.name for s in random.Random(42).sample(load_manifest_dataset(synthetic_drone_dataset), 3)]
    sequences = load_eval_manifest(eval_set_dir)
    assert [seq.frames[0].image_path.name for seq in sequences] == expected
