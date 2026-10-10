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


def build_synthetic_eval_set(eval_set_dir: Path, freeze: bool = True) -> Path:
    """Write a tiny frozen-eval-set fixture (eval/schema.py's layout) into
    `eval_set_dir`: one sequence with a moving "drone" square across its
    first frames, then a couple of hard-negative ("bird") frames. Used to
    smoke-test eval/harness.py without any real drone footage or a
    trained model — the zero-dependency `motion` detector is enough to
    exercise the full harness.
    """
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    images_dir = eval_set_dir / "images"
    images_dir.mkdir(parents=True)

    frames = []
    for i in range(8):
        width, height = 128, 96
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        file_name = f"seq_001_{i:03d}.jpg"
        boxes = []
        if i < 6:
            x = 10 + i * 10
            arr[30:40, x : x + 10] = 255
            boxes.append({"bbox": [x, 30, 10, 10], "category": "drone"})
        else:
            arr[5:15, 80:95] = 180
            boxes.append({"bbox": [80, 5, 15, 10], "category": "bird"})
        cv2.imwrite(str(images_dir / file_name), arr)
        frames.append({"file_name": file_name, "width": width, "height": height, "boxes": boxes})

    (eval_set_dir / "sequences.json").write_text(json.dumps({"sequences": [{"id": "seq_001", "frames": frames}]}))

    if freeze:
        from eval.schema import freeze_eval_set

        freeze_eval_set(eval_set_dir)

    return eval_set_dir


@pytest.fixture
def synthetic_eval_set(tmp_path):
    return build_synthetic_eval_set(tmp_path / "synthetic_eval_set")


# --- Own-footage pipeline (tools/ingest_footage.py and friends) ------------

SYNTHETIC_CLIP_FPS = 15.0
SYNTHETIC_CLIP_SECONDS = 2
SYNTHETIC_CLIP_SIZE = (320, 240)


def synthetic_clip_box(frame_index: int) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2) of the dark "drone" in frame `frame_index` of the clip
    written by `write_synthetic_clip` — the ground truth a reviewer would draw."""
    x = 20 + frame_index * 4
    return (x, 100, x + 30, 120)


def write_synthetic_clip(path: Path) -> Path:
    """A 2-second, 15 FPS, 320x240 mp4: a dark rectangle crossing a sky-blue
    background. Generated, so the pipeline is testable without real footage."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    width, height = SYNTHETIC_CLIP_SIZE
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), SYNTHETIC_CLIP_FPS, (width, height))
    for i in range(int(SYNTHETIC_CLIP_FPS * SYNTHETIC_CLIP_SECONDS)):
        frame = np.full((height, width, 3), (200, 170, 120), dtype=np.uint8)
        x1, y1, x2, y2 = synthetic_clip_box(i)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
        writer.write(frame)
    writer.release()
    return path


POSITIVE_CLIP_META = {
    "drone_type": "fpv-quad",
    "distance_band": "close-lt50m",
    "lighting": "daylight-clear",
    "background": "clean-sky",
    "is_hard_negative": False,
}


@pytest.fixture
def synthetic_video(tmp_path):
    return write_synthetic_clip(tmp_path / "C0001.mp4")


@pytest.fixture
def ingested_clip(tmp_path, synthetic_video):
    """A clip dir as tools/ingest_footage.py writes it (10 frames at 5 FPS)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import ingest_footage

    return ingest_footage.ingest(synthetic_video, dict(POSITIVE_CLIP_META), tmp_path / "clips", clip_id="c0001", sample_fps=5.0)
