#!/usr/bin/env python3
"""YAML-configurable fine-tuning script for the drone-specific detector.

Starts from the same pretrained backbone as `detector/torchvision_detector.py`
(SSDLite320 MobileNetV3, COCO-pretrained) and replaces its classification
head with a single-class ("drone") head, so the resulting weights are
loadable by `detector/drone_detector.py`.

This is infrastructure: it does not ship any dataset or pretrained
drone-specific weights, and running it against real data is the caller's
responsibility (see detector/datasets/manifest.py + loader.py for the
expected dataset layout). This script has been exercised end-to-end in
this repo only against a tiny synthetic fixture
(tests/test_train.py) to prove the training loop itself runs — that is a
plumbing check, not a claim about detection accuracy on real drone
footage. Never treat this script's own loss numbers as a benchmark result;
see eval/ for the frozen evaluation harness that is.

Structured hard-negative handling: any image whose boxes, after filtering
to `target_class`, are empty (i.e. it had none, or only had boxes labeled
as something else — e.g. "bird"/"clutter") is kept in the training set as
a zero-object target. This is the standard hard-negative-mining mechanism
for object detectors: the loss for such an image penalizes any false
positive the model raises on it, teaching it to suppress detections on
birds/clutter without needing a second output class for them.

Usage:
    python detector/train.py --config path/to/train_config.yaml

Requires the optional `torchvision` extra (torch + torchvision) — see
requirements.txt / pyproject.toml. Not part of the core install.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from detector.datasets.loader import ImageSample, load_manifest_dataset  # noqa: E402

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
    from torchvision.transforms.functional import to_tensor
    from PIL import Image

    _TRAIN_DEPS_AVAILABLE = True
except ImportError:
    _TRAIN_DEPS_AVAILABLE = False


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str
    target_class: str = "drone"
    hard_negative_categories: list[str] = field(default_factory=lambda: ["bird", "clutter"])
    val_fraction: float = 0.2
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 0.005
    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known_fields = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known_fields
        if unknown:
            raise ValueError(f"Unknown key(s) in {path}: {', '.join(sorted(unknown))}")
        return cls(**raw)


def _require_train_deps() -> None:
    if not _TRAIN_DEPS_AVAILABLE:
        raise ImportError(
            "detector/train.py requires the optional 'torchvision' extra "
            "(torch + torchvision + Pillow). Install with: "
            "pip install torch torchvision"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _DroneTrainingDataset(Dataset):
    """Wraps ImageSample[] for torchvision detection training.

    Boxes are filtered to `target_class` only — everything else
    (hard-negative categories, or simply absent) contributes a zero-object
    training target for that image, per the module's hard-negative
    handling described above.
    """

    def __init__(self, samples: list[ImageSample], target_class: str) -> None:
        self.samples = samples
        self.target_class = target_class

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = to_tensor(image)

        boxes = [[b.x1, b.y1, b.x2, b.y2] for b in sample.boxes if b.category == self.target_class]
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return image_tensor, {"boxes": boxes_tensor, "labels": labels_tensor}


def _collate(batch):
    return tuple(zip(*batch))


def build_single_class_model():
    """SSDLite320 MobileNetV3, COCO-pretrained backbone, head replaced for
    a single foreground class ("drone") + background."""
    _require_train_deps()
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)

    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    num_classes = 2  # background + drone
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer=torch.nn.BatchNorm2d
    )
    return model


def _freeze_batchnorm(model) -> None:
    """Keep BatchNorm layers in eval mode (using their pretrained running
    stats) during fine-tuning.

    Standard practice when fine-tuning a pretrained detector with small
    batch sizes: recomputing batch statistics from a handful of images is
    unstable, and a batch of size 1 (an unavoidable last batch for some
    dataset sizes) makes BatchNorm raise outright (`nn.BatchNorm2d`
    requires >1 sample per channel in training mode). Freezing BN avoids
    both problems and matches torchvision's own detection fine-tuning
    guidance.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def _split_train_val(samples: list[ImageSample], val_fraction: float, seed: int) -> tuple[list[ImageSample], list[ImageSample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_count = int(len(shuffled) * val_fraction)
    return shuffled[val_count:], shuffled[:val_count]


def train(config: TrainConfig) -> dict[str, Any]:
    """Run fine-tuning per `config`, returning the training report dict.

    Also writes `weights.pt` and `training_report.json` into
    `config.output_dir`. Contains only measurements from this actual run —
    no invented metrics.
    """
    _require_train_deps()
    set_seed(config.seed)

    samples = load_manifest_dataset(config.dataset_dir)
    if not samples:
        raise ValueError(f"No images found in dataset dir: {config.dataset_dir}")

    train_samples, val_samples = _split_train_val(samples, config.val_fraction, config.seed)
    hard_negative_count = sum(1 for s in train_samples if s.is_hard_negative)
    drone_box_count = sum(1 for s in train_samples for b in s.boxes if b.category == config.target_class)

    device = torch.device(config.device)
    model = build_single_class_model().to(device)
    model.train()
    _freeze_batchnorm(model)

    train_loader = DataLoader(
        _DroneTrainingDataset(train_samples, config.target_class),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=len(train_samples) > config.batch_size,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)

    epoch_losses: list[float] = []
    for _epoch in range(config.epochs):
        running_loss = 0.0
        num_batches = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        epoch_losses.append(running_loss / max(num_batches, 1))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "weights.pt"
    torch.save(model.state_dict(), weights_path)

    report = {
        "config": asdict(config),
        "dataset_dir": str(config.dataset_dir),
        "num_train_images": len(train_samples),
        "num_val_images": len(val_samples),
        "num_drone_boxes_train": drone_box_count,
        "num_hard_negative_images_train": hard_negative_count,
        "epoch_losses": [round(loss_value, 6) for loss_value in epoch_losses],
        "final_loss": round(epoch_losses[-1], 6) if epoch_losses else None,
        "weights_path": str(weights_path),
        "torch_version": torch.__version__,
        "seed": config.seed,
        "device": config.device,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": (
            "Training-loss metrics only. This is not a benchmark of detection "
            "accuracy — see eval/ for the frozen evaluation harness that "
            "produces mAP/recall/false-alarm/latency numbers."
        ),
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML TrainConfig file")
    args = parser.parse_args()

    try:
        config = TrainConfig.from_yaml(args.config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    try:
        report = train(config)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
