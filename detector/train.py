#!/usr/bin/env python3
"""YAML-configurable fine-tuning script for the drone-specific detector.

Starts from the same pretrained backbone as `detector/torchvision_detector.py`
(SSDLite320 MobileNetV3, COCO-pretrained) and replaces its classification
head with a single-class ("drone") head, so the resulting weights are
loadable by `detector/drone_detector.py`.

This is infrastructure: it does not ship any dataset or pretrained
drone-specific weights, and running it against real data is the caller's
responsibility (see detector/datasets/manifest.py + loader.py for the
expected dataset layout). Its own loss numbers are training diagnostics,
never a benchmark result — see eval/ for the frozen evaluation harness
that produces one.

Structured hard-negative handling: any image whose boxes, after filtering
to `target_class`, are empty (i.e. it had none, or only had boxes labeled
as something else — e.g. "bird"/"clutter") is kept in the training set as
a zero-object target. This is the standard hard-negative-mining mechanism
for object detectors: the loss for such an image penalizes any false
positive the model raises on it, teaching it to suppress detections on
birds/clutter without needing a second output class for them.

Gradient clipping (`grad_clip_max_norm`, default 10.0) is applied every
step. This was not a defensive default added speculatively: a first real
fine-tuning run against DUT Anti-UAV's train split (5200 real images) at
this script's previous `learning_rate` default (0.005) diverged to NaN
weights within the first epoch — the freshly-initialized single-class
head produces large early gradients that a tiny synthetic 6-image smoke
test never ran long enough to expose. Clipping alone reduced the blow-up
but the loss still oscillated without clearly converging; lowering the
default `learning_rate` to 0.001 alongside it produced a stable,
monotonically-behaved loss curve over a real 300-batch check. See
docs/DECISIONS.md for the full diagnosis.

Interruption-safe: after every epoch a `checkpoint.pt` (model weights,
optimizer state, completed-epoch count, per-epoch history, RNG states) is
written atomically into `output_dir`, together with an up-to-date
`training_report.json` and that epoch's own `weights_epoch_NNN.pt` (kept,
not overwritten, so the best epoch can be picked afterwards with
eval/select_checkpoint.py instead of assuming the last one is best). Re-running the same command with the same
`output_dir` resumes from that checkpoint instead of restarting — the
reason this exists is that Colab sessions disconnect mid-run, and losing
every completed epoch with them is not acceptable. On Colab, point
`output_dir` at a mounted Google Drive path, or the checkpoint is lost
along with the VM.

Device: `device: auto` (the default) uses CUDA when
`torch.cuda.is_available()`, else CPU; any explicit value ("cpu",
"cuda", "cuda:1", ...) overrides that.

Per-epoch validation: if `val_dataset_dir` is set (e.g. DUT Anti-UAV's
own official val split), each epoch ends with a quick evaluation on it
(optionally a fixed, seeded subset of `val_max_images`), logging AP@0.5
and the min/max/spread of the raw (pre-confidence-filter) "drone" scores.
The raw-score spread is there to show directly whether the
classification head leaves the collapsed ~0.43-0.45 band measured in
docs/DECISIONS.md entry #18. These are tuning diagnostics on the val
split, not the frozen-test-set benchmark eval/ produces.

Usage:
    python detector/train.py --config path/to/train_config.yaml

Requires the optional `torchvision` extra (torch + torchvision) — see
requirements.txt / pyproject.toml. Not part of the core install.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from detector.datasets.loader import ImageSample, load_manifest_dataset  # noqa: E402
from detector.datasets.manifest import get_manifest  # noqa: E402
from eval.schema import assert_not_for_training  # noqa: E402

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
    from torchvision.transforms.functional import to_tensor
    from PIL import Image

    _TRAIN_DEPS_AVAILABLE = True
except ImportError:
    _TRAIN_DEPS_AVAILABLE = False


class CommercialLicenseRequiredError(RuntimeError):
    """Raised when --commercial-only is set but the dataset isn't cleared."""


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str
    target_class: str = "drone"
    hard_negative_categories: list[str] = field(default_factory=lambda: ["bird", "clutter"])
    val_fraction: float = 0.2
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 0.001
    grad_clip_max_norm: float = 10.0
    seed: int = 42
    input_size: int = 320  # square input the model resizes every image to; see build_single_class_model
    device: str = "auto"  # "auto" = cuda if torch.cuda.is_available() else cpu; any other value is used as-is
    val_dataset_dir: str | None = None  # separate held-out split evaluated after every epoch (e.g. DUT's own val/)
    val_max_images: int | None = None  # evaluate a fixed, seeded subset of this many val images (None = all)
    dataset_name: str | None = None  # key into detector.datasets.manifest.DATASET_REGISTRY, for --commercial-only
    commercial_only: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known_fields = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known_fields
        if unknown:
            raise ValueError(f"Unknown key(s) in {path}: {', '.join(sorted(unknown))}")
        return cls(**raw)


def _assert_commercial_clearance(config: "TrainConfig") -> None:
    """Enforce --commercial-only: refuse to train unless the dataset's
    registry entry has an explicitly confirmed commercial_ok=True.

    This is how a provenance-clean model gets rebuilt later without
    re-auditing every dataset by hand — see
    detector/datasets/manifest.py's module docstring for what
    `commercial_ok` means and why it defaults to unknown (None) rather
    than a guess.
    """
    if not config.commercial_only:
        return

    if not config.dataset_name:
        raise CommercialLicenseRequiredError(
            "--commercial-only requires 'dataset_name' to be set in the training "
            "config, naming a key in detector.datasets.manifest.DATASET_REGISTRY, "
            "so its commercial_ok field can be checked."
        )

    manifest = get_manifest(config.dataset_name)
    if manifest.commercial_ok is not True:
        raise CommercialLicenseRequiredError(
            f"--commercial-only is set, but dataset '{config.dataset_name}' has "
            f"commercial_ok={manifest.commercial_ok!r} (license_id={manifest.license_id!r}). "
            f"Only datasets with an explicitly confirmed commercial_ok=True may be "
            f"used for a --commercial-only training run. See its license_notes in "
            f"detector/datasets/manifest.py and docs/datasets.md."
        )


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


def build_single_class_model(input_size: int = 320):
    """SSDLite MobileNetV3, COCO-pretrained backbone, head replaced for
    a single foreground class ("drone") + background.

    `input_size` sets the square size every image is resized to (stock:
    320). torchvision hard-codes 320 in the factory, but the size lives
    only in `model.transform`; anchors are generated per forward pass as
    fractions of the input (scales 0.2-0.95), so the same weights and
    anchor layout work at any size — at 640 the feature maps double
    (40x40 first map) and the smallest anchor is 128px. Weights trained at
    one size should be evaluated at that size (DroneDetector's
    `input_size`).
    """
    _require_train_deps()
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    if input_size != 320:
        stock = model.transform
        model.transform = GeneralizedRCNNTransform(
            input_size,
            input_size,
            stock.image_mean,
            stock.image_std,
            size_divisible=1,
            fixed_size=(input_size, input_size),
        )

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


CHECKPOINT_FILENAME = "checkpoint.pt"


def epoch_weights_filename(epoch: int) -> str:
    return f"weights_epoch_{epoch:03d}.pt"

# Fields whose change makes a resumed run no longer the same run. `epochs`
# is deliberately absent (raising it is how a finished run is extended),
# as are `device` (resume a CPU run on a GPU and vice versa) and dataset
# paths (mount points differ between Colab sessions and local machines).
_RESUME_MUST_MATCH = (
    "target_class", "val_fraction", "batch_size", "learning_rate", "grad_clip_max_norm", "seed", "input_size"
)

# Mirrors DroneDetector's default nms_thresh (detector/drone_detector.py)
# so the per-epoch val AP@0.5 is computed the same way eval/harness.py
# computes it for a trained model.
_VAL_NMS_THRESH = 0.45


_TRAIN_CONFIG_DEFAULTS = {
    name: f.default for name, f in TrainConfig.__dataclass_fields__.items() if f.default is not MISSING
}


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _save_checkpoint_atomically(state: dict[str, Any], path: Path) -> None:
    """Write to a temp file then rename, so a disconnect mid-save leaves the
    previous epoch's checkpoint intact instead of a truncated file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def evaluate_on_val(model, samples: list[ImageSample], target_class: str, device) -> dict[str, Any]:
    """Quick per-epoch evaluation: threshold-independent AP@0.5 over every
    raw "drone" detection (nms_thresh 0.45, no confidence cutoff — same as
    eval/harness.py) plus min/max/spread of those raw scores. Leaves the model in train mode
    with BatchNorm frozen, as the training loop expects."""
    from eval.metrics import FrameEvalData, PredBox, compute_ap50
    from eval.schema import EvalBox

    original_nms_thresh = model.nms_thresh
    model.nms_thresh = _VAL_NMS_THRESH
    model.eval()

    frames: list[FrameEvalData] = []
    raw_scores: list[float] = []
    try:
        with torch.no_grad():
            for sample in samples:
                image_tensor = to_tensor(Image.open(sample.image_path).convert("RGB")).to(device)
                output = model([image_tensor])[0]
                pred_boxes = []
                for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                    if int(label) != 1:
                        continue
                    confidence = float(score)
                    raw_scores.append(confidence)
                    x1, y1, x2, y2 = (float(v) for v in box)
                    pred_boxes.append(PredBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence))
                gt_boxes = [
                    EvalBox(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, category=b.category)
                    for b in sample.boxes
                    if b.category == target_class
                ]
                frames.append(FrameEvalData(frame_id=str(sample.image_path), gt_boxes=gt_boxes, pred_boxes=pred_boxes))
    finally:
        model.nms_thresh = original_nms_thresh
        model.train()
        _freeze_batchnorm(model)

    return {
        "num_val_images": len(samples),
        "num_gt_boxes": sum(len(f.gt_boxes) for f in frames),
        "ap50": round(compute_ap50(frames), 6),
        "num_raw_scores": len(raw_scores),
        "raw_score_min": round(min(raw_scores), 6) if raw_scores else None,
        "raw_score_max": round(max(raw_scores), 6) if raw_scores else None,
        "raw_score_spread": round(max(raw_scores) - min(raw_scores), 6) if raw_scores else None,
    }


def _split_train_val(samples: list[ImageSample], val_fraction: float, seed: int) -> tuple[list[ImageSample], list[ImageSample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_count = int(len(shuffled) * val_fraction)
    return shuffled[val_count:], shuffled[:val_count]


def train(config: TrainConfig) -> dict[str, Any]:
    """Run fine-tuning per `config`, returning the training report dict.

    Writes `checkpoint.pt` and `training_report.json` into
    `config.output_dir` after every epoch, and `weights.pt` once all
    epochs are done; resumes from an existing `checkpoint.pt` there. The
    report contains only measurements from this actual run (across any
    resumes) — no invented metrics.
    """
    _require_train_deps()
    set_seed(config.seed)

    _assert_commercial_clearance(config)
    assert_not_for_training(config.dataset_dir)
    samples = load_manifest_dataset(config.dataset_dir)
    if not samples:
        raise ValueError(f"No images found in dataset dir: {config.dataset_dir}")

    train_samples, val_samples = _split_train_val(samples, config.val_fraction, config.seed)
    hard_negative_count = sum(1 for s in train_samples if s.is_hard_negative)
    drone_box_count = sum(1 for s in train_samples for b in s.boxes if b.category == config.target_class)

    val_eval_samples: list[ImageSample] = []
    if config.val_dataset_dir:
        assert_not_for_training(config.val_dataset_dir)  # never tune against the frozen test set
        val_eval_samples = load_manifest_dataset(config.val_dataset_dir)
        if config.val_max_images is not None and len(val_eval_samples) > config.val_max_images:
            val_eval_samples = random.Random(config.seed).sample(val_eval_samples, config.val_max_images)

    resolved_device = resolve_device(config.device)
    device = torch.device(resolved_device)
    model = build_single_class_model(config.input_size).to(device)
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

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    weights_path = output_dir / "weights.pt"

    epoch_losses: list[float] = []
    val_history: list[dict[str, Any]] = []
    start_epoch = 0
    resumed_from_epoch: int | None = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mismatched = {
            # A key missing from an older checkpoint means it was written
            # before that field existed, i.e. with the field's default.
            key: (checkpoint["config"].get(key, _TRAIN_CONFIG_DEFAULTS.get(key)), getattr(config, key))
            for key in _RESUME_MUST_MATCH
            if checkpoint["config"].get(key, _TRAIN_CONFIG_DEFAULTS.get(key)) != getattr(config, key)
        }
        if mismatched:
            raise ValueError(
                f"Checkpoint {checkpoint_path} was written by a run with different settings "
                f"(checkpoint vs. now: {mismatched}). Use a fresh output_dir, or delete the "
                f"checkpoint to deliberately restart."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _restore_rng_state(checkpoint["rng_state"])
        start_epoch = checkpoint["epoch"]
        resumed_from_epoch = start_epoch
        epoch_losses = list(checkpoint["epoch_losses"])
        val_history = list(checkpoint["val_history"])
        print(f"Resuming from {checkpoint_path}: {start_epoch} epoch(s) already completed.", file=sys.stderr)

    def build_report(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "config": asdict(config),
            "dataset_dir": str(config.dataset_dir),
            "num_train_images": len(train_samples),
            "num_val_images": len(val_samples),
            "num_drone_boxes_train": drone_box_count,
            "num_hard_negative_images_train": hard_negative_count,
            "epochs_completed": len(epoch_losses),
            "resumed_from_epoch": resumed_from_epoch,
            "epoch_losses": [round(loss_value, 6) for loss_value in epoch_losses],
            "final_loss": round(epoch_losses[-1], 6) if epoch_losses else None,
            "val_dataset_dir": config.val_dataset_dir,
            "val_per_epoch": val_history,
            "checkpoint_path": str(checkpoint_path),
            "epoch_weights_paths": [str(output_dir / epoch_weights_filename(e)) for e in range(1, len(epoch_losses) + 1)],
            "weights_path": str(weights_path) if status == "complete" else None,
            "torch_version": torch.__version__,
            "seed": config.seed,
            "device": resolved_device,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": (
                "Training-loss metrics, plus (if val_per_epoch is non-empty) quick "
                "per-epoch AP@0.5 / raw-score diagnostics on a held-out val split. "
                "This is not a benchmark of detection accuracy — see eval/ for the "
                "frozen evaluation harness that produces mAP/recall/false-alarm/latency "
                "numbers on the test set."
            ),
        }

    report_path = output_dir / "training_report.json"
    for epoch in range(start_epoch, config.epochs):
        running_loss = 0.0
        num_batches = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        epoch_losses.append(running_loss / max(num_batches, 1))

        if val_eval_samples:
            val_metrics = evaluate_on_val(model, val_eval_samples, config.target_class, device)
            val_history.append({"epoch": epoch + 1, **val_metrics})

        _save_checkpoint_atomically(model.state_dict(), output_dir / epoch_weights_filename(epoch + 1))
        _save_checkpoint_atomically(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": _rng_state(),
                "epoch_losses": epoch_losses,
                "val_history": val_history,
                "config": asdict(config),
            },
            checkpoint_path,
        )
        report_path.write_text(json.dumps(build_report("in_progress"), indent=2), encoding="utf-8")
        val_summary = f", val {val_history[-1]}" if val_eval_samples else ""
        print(f"Epoch {epoch + 1}/{config.epochs}: loss {epoch_losses[-1]:.6f}{val_summary}", file=sys.stderr)

    torch.save(model.state_dict(), weights_path)
    report = build_report("complete")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML TrainConfig file")
    parser.add_argument(
        "--commercial-only",
        action="store_true",
        help=(
            "Refuse to train unless the dataset (config's 'dataset_name', looked up in "
            "detector.datasets.manifest.DATASET_REGISTRY) has an explicitly confirmed "
            "commercial_ok=True. Overrides 'commercial_only' in the config file if passed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Overrides the config's output_dir — e.g. a mounted Google Drive path on Colab, "
            "so checkpoints survive the VM. Re-run with the same value to resume."
        ),
    )
    args = parser.parse_args()

    try:
        config = TrainConfig.from_yaml(args.config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    if args.commercial_only:
        config.commercial_only = True
    if args.output_dir:
        config.output_dir = args.output_dir

    try:
        report = train(config)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except CommercialLicenseRequiredError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
