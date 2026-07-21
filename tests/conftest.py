"""Shared pytest fixtures."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_synthetic_drone_dataset(dataset_dir: Path) -> Path:
    """Write a tiny dataset in the unified schema (detector/datasets/loader.py)
    into `dataset_dir`: a few "drone" positives and a few hard-negative
    ("bird") images. Real pixels, real (trivial) annotations — used to
    smoke-test the dataset loader and detector/train.py without any real
    drone data.
    """
    np = pytest.importorskip("numpy")
    pil_image = pytest.importorskip("PIL.Image")

    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)

    images = []
    annotations = []
    for i in range(6):
        width, height = 64, 64
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        file_name = f"img_{i}.jpg"
        if i % 2 == 0:
            arr[20:30, 20:30] = 255
            annotations.append({"image_id": i + 1, "bbox": [20, 20, 10, 10], "category": "drone"})
        else:
            arr[5:15, 40:55] = 128
            annotations.append({"image_id": i + 1, "bbox": [40, 5, 15, 10], "category": "bird"})
        pil_image.fromarray(arr).save(images_dir / file_name)
        images.append({"id": i + 1, "file_name": file_name, "width": width, "height": height})

    (dataset_dir / "annotations.json").write_text(json.dumps({"images": images, "annotations": annotations}))
    return dataset_dir


@pytest.fixture
def synthetic_drone_dataset(tmp_path):
    return build_synthetic_drone_dataset(tmp_path / "synthetic_dataset")


@pytest.fixture(scope="module")
def synthetic_drone_dataset_module_scoped(tmp_path_factory):
    return build_synthetic_drone_dataset(tmp_path_factory.mktemp("synthetic_dataset"))
