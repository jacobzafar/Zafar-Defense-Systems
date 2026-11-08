import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from detector.drone_detector import DroneDetector, DroneWeightsNotFoundError
from detector.factory import build_detector
from detector.train import TrainConfig, train


def test_raises_clear_error_when_weights_path_missing():
    with pytest.raises(DroneWeightsNotFoundError, match="no path given"):
        DroneDetector(weights_path="")


def test_raises_clear_error_when_weights_file_does_not_exist(tmp_path):
    with pytest.raises(DroneWeightsNotFoundError, match="No trained drone-detector weights"):
        DroneDetector(weights_path=str(tmp_path / "does-not-exist.pt"))


def test_factory_surfaces_the_same_clear_error():
    with pytest.raises(DroneWeightsNotFoundError):
        build_detector({"backend": "drone"})


@pytest.fixture(scope="module")
def trained_weights_path(tmp_path_factory, synthetic_drone_dataset_module_scoped):
    output_dir = tmp_path_factory.mktemp("drone_train_output")
    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset_module_scoped),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        val_fraction=0.2,
        seed=42,
    )
    train(config)
    return output_dir / "weights.pt"


def test_loads_trained_weights_and_runs_detect(trained_weights_path):
    import numpy as np

    detector = DroneDetector(weights_path=str(trained_weights_path), confidence_threshold=0.0)
    frame = (np.random.rand(64, 64, 3) * 255).astype("uint8")

    detections = detector.detect(frame)

    assert isinstance(detections, list)
    for det in detections:
        assert det.class_name == "drone"
        assert 0.0 <= det.confidence <= 1.0
        assert 0.0 <= det.x1 <= 1.0


def test_detect_handles_none_frame(trained_weights_path):
    detector = DroneDetector(weights_path=str(trained_weights_path))
    assert detector.detect(None) == []


def test_factory_builds_drone_backend_with_valid_weights(trained_weights_path):
    detector = build_detector({"backend": "drone", "weights_path": str(trained_weights_path)})
    assert isinstance(detector, DroneDetector)
    assert detector.name == "drone_v1"


def test_nms_thresh_defaults_to_a_tighter_value_than_the_stock_coco_factory(trained_weights_path):
    """Regression guard for the "near-identical boxes" bug: build_single_class_model()
    only swaps the classification head, so torchvision's ssdlite320_mobilenet_v3_large()
    factory's own nms_thresh=0.55 (tuned for 80-class COCO) would otherwise carry
    through untouched. DroneDetector must actively set a tighter value on the
    underlying model, not just leave the COCO default in place."""
    detector = DroneDetector(weights_path=str(trained_weights_path))
    assert detector._model.nms_thresh == pytest.approx(0.45)
    assert detector._model.nms_thresh < 0.55


def test_nms_thresh_is_configurable(trained_weights_path):
    detector = DroneDetector(weights_path=str(trained_weights_path), nms_thresh=0.3)
    assert detector._model.nms_thresh == pytest.approx(0.3)


def test_factory_passes_nms_thresh_through(trained_weights_path):
    detector = build_detector({"backend": "drone", "weights_path": str(trained_weights_path), "nms_thresh": 0.25})
    assert detector._model.nms_thresh == pytest.approx(0.25)


class _FakeTileModel:
    """Returns one box at crop-local (10, 10, 20, 20) per input crop, with a
    score that identifies the crop's position in the batch."""

    nms_thresh = 0.45
    detections_per_img = 300

    def __init__(self):
        self.crop_shapes = []

    def __call__(self, images):
        import torch

        self.crop_shapes = [tuple(img.shape[1:]) for img in images]
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
                "scores": torch.tensor([0.9 - 0.01 * i]),
                "labels": torch.tensor([1]),
            }
            for i in range(len(images))
        ]


def test_tiled_detect_maps_tile_boxes_back_to_frame_coordinates(trained_weights_path):
    import numpy as np

    from detector.tiling import tile_grid

    detector = DroneDetector(weights_path=str(trained_weights_path), confidence_threshold=0.0, tile_rows=2, tile_cols=2)
    fake = _FakeTileModel()
    detector._model = fake
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    tiles = tile_grid(100, 50, 2, 2, 0.2)
    assert fake.crop_shapes == [(50, 100)] + [(y2 - y1, x2 - x1) for x1, y1, x2, y2 in tiles]  # full frame first
    # The full-frame box and the first tile's box coincide (both at offset 0,0)
    # and merge under NMS; the other three tiles' boxes are offset and survive.
    expected = sorted([(10, 10)] + [(x1 + 10, y1 + 10) for x1, y1, _, _ in tiles[1:]])
    got = sorted((round(d.x1 * 100), round(d.y1 * 50)) for d in detections)
    assert got == expected


def test_factory_passes_tiling_options_through(trained_weights_path):
    from detector.factory import build_detector

    detector = build_detector(
        {"backend": "drone", "weights_path": str(trained_weights_path), "tile_rows": 3, "tile_cols": 3, "tile_overlap": 0.25}
    )
    assert (detector.tile_rows, detector.tile_cols, detector.tile_overlap, detector.tile_include_full_frame) == (3, 3, 0.25, True)


class _FakeFixedBoxModel:
    """Returns one box at fixed model-input coordinates and records inputs."""

    def __init__(self, box):
        self.box = box
        self.shapes = []

    def __call__(self, images):
        import torch

        self.shapes = [tuple(img.shape) for img in images]
        return [{"boxes": torch.tensor([self.box]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])} for _ in images]


def test_letterbox_pads_to_square_and_maps_boxes_back_through_the_padding(trained_weights_path):
    import numpy as np

    detector = DroneDetector(weights_path=str(trained_weights_path), confidence_threshold=0.0, preprocess="letterbox")
    size = detector.input_size  # 320
    # 640x320 frame -> scale 0.5 -> 320x160 content, padded by 80 rows top and bottom.
    fake = _FakeFixedBoxModel([100.0, 100.0, 120.0, 110.0])
    detector._model = fake
    [d] = detector.detect(np.zeros((320, 640, 3), np.uint8))
    assert fake.shapes == [(3, size, size)]
    assert (d.x1 * 640, d.y1 * 320, d.x2 * 640, d.y2 * 320) == pytest.approx((200, 40, 240, 60))


def test_area_squeeze_resizes_on_cpu_and_maps_per_axis(trained_weights_path):
    import numpy as np

    detector = DroneDetector(weights_path=str(trained_weights_path), confidence_threshold=0.0, resize_interp="area")
    fake = _FakeFixedBoxModel([32.0, 160.0, 64.0, 192.0])
    detector._model = fake
    [d] = detector.detect(np.zeros((640, 1280, 3), np.uint8))
    assert fake.shapes == [(3, 320, 320)]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((0.1, 0.5, 0.2, 0.6))


def test_default_path_hands_the_full_resolution_frame_to_the_model(trained_weights_path):
    import numpy as np

    detector = DroneDetector(weights_path=str(trained_weights_path), confidence_threshold=0.0)
    fake = _FakeFixedBoxModel([10.0, 10.0, 20.0, 20.0])
    detector._model = fake
    [d] = detector.detect(np.zeros((50, 100, 3), np.uint8))
    assert fake.shapes == [(3, 50, 100)]
    assert (d.x1, d.y1) == pytest.approx((0.1, 0.2))


def test_detect_accepts_single_channel_frames(trained_weights_path):
    import numpy as np

    detector = DroneDetector(weights_path=str(trained_weights_path), confidence_threshold=0.0)
    fake = _FakeFixedBoxModel([10.0, 10.0, 20.0, 20.0])
    detector._model = fake
    detector.detect(np.zeros((50, 100), np.uint16))
    assert fake.shapes == [(3, 50, 100)]


def test_invalid_preprocess_options_are_rejected(trained_weights_path):
    with pytest.raises(ValueError, match="preprocess"):
        DroneDetector(weights_path=str(trained_weights_path), preprocess="adaptive")
    with pytest.raises(ValueError, match="resize_interp"):
        DroneDetector(weights_path=str(trained_weights_path), resize_interp="lanczos")


def test_tile_size_grids_portrait_and_landscape_frames_per_frame(trained_weights_path):
    import numpy as np

    from detector.tiling import tile_grid, tiles_for_size

    detector = DroneDetector(weights_path=str(trained_weights_path), confidence_threshold=0.0, tile_size=100)
    fake = _FakeTileModel()
    detector._model = fake
    for h, w in ((120, 300), (300, 120)):
        detector.detect(np.zeros((h, w, 3), np.uint8))
        rows, cols = tiles_for_size(h, 100, 0.2), tiles_for_size(w, 100, 0.2)
        expected = [(h, w)] + [(y2 - y1, x2 - x1) for x1, y1, x2, y2 in tile_grid(w, h, rows, cols, 0.2)]
        assert fake.crop_shapes == expected
