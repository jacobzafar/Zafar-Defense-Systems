"""Tests for the stride-8 SSDLite variant (detector/ssdlite_stride8.py,
docs/DECISIONS.md #26) and the training-side guards added with it:
content-based eval leakage check, early stopping, horizontal flip and the
provenance block. Plumbing only — no real data, not a benchmark."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from detector.datasets.loader import load_manifest_dataset  # noqa: E402
from detector.ssdlite_stride8 import GEOMETRIC_SCALES, build_stride8_model  # noqa: E402
from detector.train import (  # noqa: E402
    CommercialLicenseRequiredError,
    DataLeakageError,
    TrainConfig,
    _DroneTrainingDataset,
    _should_stop_early,
    build_single_class_model,
    train,
)
from eval import anchor_coverage  # noqa: E402
from eval.build_frozen_eval_set import build_frozen_eval_set  # noqa: E402


def test_stride8_model_has_seven_levels_from_80x80_at_640():
    model = build_stride8_model(640, pretrained=False).eval()
    images, _ = model.transform([torch.zeros(3, 1080, 1920)])
    features = list(model.backbone(images.tensors).values())
    assert [tuple(f.shape[-2:]) for f in features] == [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5), (3, 3), (2, 2)]
    assert features[1].shape[1] == 672  # the stock SSDLite's first (stride-16) map is unchanged
    assert len(model.anchor_generator(images, features)[0]) == 51228


def test_stride8_reuses_the_stock_pretrained_modules():
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    torch.manual_seed(0)
    stock = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    torch.manual_seed(0)
    model = build_stride8_model(640, pretrained=False)
    stock_backbone = sum(p.numel() for p in stock.backbone.parameters())
    assert sum(p.numel() for p in model.backbone.parameters()) == stock_backbone
    for level in range(6):  # stock regression levels 0-5 become levels 1-6
        a = stock.head.regression_head.module_list[level].state_dict()
        b = model.head.regression_head.module_list[level + 1].state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_stride8_anchor_scales_are_validated_and_stock_rejects_them():
    with pytest.raises(ValueError, match="8 values"):
        build_stride8_model(640, anchor_scales=(0.1, 0.5), pretrained=False)
    with pytest.raises(ValueError, match="only supported"):
        build_single_class_model(320, pretrained=False, anchor_scales=list(GEOMETRIC_SCALES))
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_single_class_model(320, pretrained=False, architecture="yolo")


def test_real_stride8_anchors_equal_the_geometric_simulation():
    """The built model's anchors are the layout docs/fas0-feasibility.md simulated (74.7% on FPV)."""
    import numpy as np
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    from torchvision.models.detection.image_list import ImageList

    real = anchor_coverage.anchor_boxes(640, "ssdlite_stride8", GEOMETRIC_SCALES)
    generator = DefaultBoxGenerator([[2, 3]] * 7, scales=list(np.geomspace(0.025, 0.95, 8)))
    images = ImageList(torch.zeros(1, 3, 640, 640), [(640, 640)])
    simulated = generator(images, [torch.zeros(1, 1, m, m) for m in (80, 40, 20, 10, 5, 3, 2)])[0]
    assert torch.allclose(real, simulated, atol=1e-3)


def test_stride8_matches_a_3_percent_drone_the_stock_model_cannot():
    frame = (1920, 1080)
    box = (900.0, 500.0, 958.0, 543.0)  # 58x43 px: ~3% apparent size
    [stock] = anchor_coverage.best_anchor_ious([box], [frame], input_size=640)
    [stride8] = anchor_coverage.best_anchor_ious(
        [box], [frame], input_size=640, architecture="ssdlite_stride8", anchor_scales=GEOMETRIC_SCALES
    )
    assert stock < anchor_coverage.MATCH_IOU <= stride8


def _stride8_config(dataset, out, **extra):
    return TrainConfig(
        dataset_dir=str(dataset),
        output_dir=str(out),
        epochs=1,
        batch_size=2,
        input_size=160,
        architecture="ssdlite_stride8",
        **extra,
    )


def test_stride8_trains_and_loads_in_drone_detector(synthetic_drone_dataset, tmp_path):
    import numpy as np

    from detector.drone_detector import DroneDetector
    from detector.factory import build_detector

    report = train(_stride8_config(synthetic_drone_dataset, tmp_path / "out"))
    assert report["config"]["architecture"] == "ssdlite_stride8"
    detector = build_detector(
        {
            "backend": "drone",
            "weights_path": report["weights_path"],
            "confidence_threshold": 0.0,
            "input_size": 160,
            "architecture": "ssdlite_stride8",
        }
    )
    assert isinstance(detector, DroneDetector)
    assert isinstance(detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)), list)


def test_drone_detector_refuses_weights_trained_with_another_layout(synthetic_drone_dataset, tmp_path):
    from detector.drone_detector import DroneDetector

    report = train(_stride8_config(synthetic_drone_dataset, tmp_path / "out"))
    with pytest.raises(ValueError, match="architecture"):
        DroneDetector(weights_path=report["weights_path"], input_size=160)  # stock architecture
    other_scales = [0.02, 0.04, 0.07, 0.12, 0.2, 0.34, 0.56, 0.95]
    with pytest.raises(ValueError, match="anchor_scales"):
        DroneDetector(
            weights_path=report["weights_path"], input_size=160, architecture="ssdlite_stride8", anchor_scales=other_scales
        )


def test_resume_refuses_a_different_architecture(synthetic_drone_dataset, tmp_path):
    out = tmp_path / "out"
    train(TrainConfig(dataset_dir=str(synthetic_drone_dataset), output_dir=str(out), epochs=1, batch_size=2, input_size=160))
    resumed = _stride8_config(synthetic_drone_dataset, out)
    resumed.epochs = 2
    with pytest.raises(ValueError, match="architecture"):
        train(resumed)


def test_leakage_guard_refuses_training_images_found_in_a_frozen_eval_set(synthetic_drone_dataset, tmp_path):
    eval_dir = tmp_path / "frozen"
    build_frozen_eval_set(synthetic_drone_dataset, eval_dir)
    # A copy, not the same path: the guard compares file contents.
    copy = tmp_path / "copy"
    shutil.copytree(synthetic_drone_dataset, copy)
    with pytest.raises(DataLeakageError, match="frozen eval set"):
        train(_stride8_config(copy, tmp_path / "out", leakage_guard_eval_dirs=[str(eval_dir)]))


def test_leakage_guard_passes_disjoint_data_and_is_reported(synthetic_drone_dataset, tmp_path):
    from tests.conftest import build_synthetic_eval_set

    eval_dir = build_synthetic_eval_set(tmp_path / "frozen")
    report = train(_stride8_config(synthetic_drone_dataset, tmp_path / "out", leakage_guard_eval_dirs=[str(eval_dir)]))
    assert report["leakage_guard_images_checked"] == 8


def test_should_stop_early_counts_epochs_since_the_best_val_ap():
    history = [{"epoch": 1, "ap50": 0.1}, {"epoch": 2, "ap50": 0.3}, {"epoch": 3, "ap50": 0.2}, {"epoch": 4, "ap50": 0.3}]
    assert not _should_stop_early(history, 3)  # ties don't reset: best is epoch 2, 2 epochs ago
    assert _should_stop_early(history, 2)
    assert not _should_stop_early(history, None)


def test_early_stopping_ends_training_and_reports_best_epoch(synthetic_drone_dataset, tmp_path, monkeypatch):
    import detector.train as train_module

    aps = iter([0.5, 0.4, 0.3, 0.2, 0.1])
    monkeypatch.setattr(train_module, "evaluate_on_val", lambda *a, **k: {"ap50": next(aps)})
    config = TrainConfig(
        dataset_dir=str(synthetic_drone_dataset),
        output_dir=str(tmp_path / "out"),
        val_dataset_dir=str(synthetic_drone_dataset),
        epochs=5,
        batch_size=2,
        input_size=160,
        early_stopping_patience=2,
    )
    report = train(config)
    assert report["epochs_completed"] == 3
    assert report["stopped_early"] is True
    assert report["best_val_epoch"] == 1
    best = torch.load(report["best_weights_path"])
    epoch1 = torch.load(tmp_path / "out" / "weights_epoch_001.pt")
    assert all(torch.equal(best[k], epoch1[k]) for k in best)


def test_hflip_mirrors_boxes(synthetic_drone_dataset):
    samples = load_manifest_dataset(synthetic_drone_dataset)
    sample = next(s for s in samples if any(b.category == "drone" for b in s.boxes))
    plain_image, plain = _DroneTrainingDataset([sample], "drone")[0]
    flipped_image, flipped = _DroneTrainingDataset([sample], "drone", hflip_prob=1.0)[0]
    width = plain_image.shape[-1]
    assert torch.equal(flipped_image, plain_image.flip(-1))
    expected = torch.stack(
        [width - plain["boxes"][:, 2], plain["boxes"][:, 1], width - plain["boxes"][:, 0], plain["boxes"][:, 3]], dim=1
    )
    assert torch.allclose(flipped["boxes"], expected)


def test_visiodect_feasibility_run_is_research_only(synthetic_drone_dataset, tmp_path):
    """VisioDECT is commercial_ok=None: --commercial-only refuses it, and a
    research run's report says it is not commercially cleared."""
    config = _stride8_config(synthetic_drone_dataset, tmp_path / "out", dataset_name="visiodect", purpose="research")
    report = train(config)
    provenance = report["provenance"]
    assert provenance["license_id"] == "UNVERIFIED"
    assert provenance["commercially_cleared"] is False
    assert json.loads((tmp_path / "out" / "training_report.json").read_text())["provenance"]["purpose"] == "research"

    config.commercial_only = True
    config.output_dir = str(tmp_path / "out2")
    with pytest.raises(CommercialLicenseRequiredError):
        train(config)


def test_stride8_fpv_config_is_a_research_run_with_the_verified_layout():
    config = TrainConfig.from_yaml(Path(__file__).resolve().parent.parent / "config" / "stride8_fpv.yaml")
    assert config.architecture == "ssdlite_stride8"
    assert tuple(config.anchor_scales) == GEOMETRIC_SCALES
    assert config.input_size == 640
    assert config.dataset_name == "visiodect" and not config.commercial_only
    assert config.leakage_guard_eval_dirs and config.early_stopping_patience


def test_stock320_control_differs_from_stride8_only_in_architecture():
    root = Path(__file__).resolve().parent.parent / "config"
    stride8 = TrainConfig.from_yaml(root / "stride8_fpv.yaml").__dict__
    control = TrainConfig.from_yaml(root / "stock320_fpv_control.yaml").__dict__
    differing = {k for k in stride8 if stride8[k] != control[k]}
    assert differing == {"architecture", "anchor_scales", "input_size", "output_dir", "purpose"}
