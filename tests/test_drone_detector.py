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
