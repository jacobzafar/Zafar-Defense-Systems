"""Load and validate named config presets (config/presets.yaml).

This module only builds and validates the plain config *dicts* that
`detector.factory.build_detector()` / `tracker.factory.build_tracker()`
already accept — it does not duplicate or bypass those factories, so they
stay the single source of truth for what backends and keys exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PRESETS_PATH = Path(__file__).resolve().parent / "presets.yaml"

# Kept in sync with detector.factory / tracker.factory by hand — these are
# the backend identifiers those factories recognize.
VALID_DETECTOR_BACKENDS = frozenset({"motion", "ultralytics", "torchvision"})
VALID_TRACKER_BACKENDS = frozenset({"iou", "bytetrack"})

_NUMERIC_DETECTOR_KEYS = ("min_area_px", "max_area_fraction", "var_threshold", "history", "confidence_threshold")
_NUMERIC_TRACKER_KEYS = ("iou_threshold", "max_age", "min_hits_to_confirm")


class ConfigError(ValueError):
    """Raised when a preset or config dict fails validation."""


@dataclass
class AppConfig:
    """A named, validated detector/tracker/logging configuration."""

    name: str
    detector: dict[str, Any] = field(default_factory=dict)
    tracker: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    debug: bool = False


def load_presets(path: str | Path = PRESETS_PATH) -> dict[str, AppConfig]:
    """Load every preset from a presets YAML file, validating each one."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Preset file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc

    presets_raw = raw.get("presets") or {}
    if not presets_raw:
        raise ConfigError(f"No presets found in {path}")

    return {name: _build_and_validate(name, body or {}) for name, body in presets_raw.items()}


def get_preset(name: str, presets: dict[str, AppConfig] | None = None) -> AppConfig:
    """Look up one preset by name, loading the default catalog if needed."""
    presets = presets if presets is not None else load_presets()
    if name not in presets:
        available = ", ".join(sorted(presets)) or "(none)"
        raise ConfigError(f"Unknown preset '{name}'. Valid options: {available}.")
    return presets[name]


def _build_and_validate(name: str, body: dict[str, Any]) -> AppConfig:
    detector_cfg = dict(body.get("detector") or {})
    tracker_cfg = dict(body.get("tracker") or {})
    logging_cfg = dict(body.get("logging") or {})
    debug = bool(body.get("debug", False))

    detector_backend = detector_cfg.get("backend", "motion")
    if detector_backend not in VALID_DETECTOR_BACKENDS:
        raise ConfigError(
            f"Preset '{name}': unknown detector backend '{detector_backend}'. "
            f"Valid options: {', '.join(sorted(VALID_DETECTOR_BACKENDS))}."
        )

    tracker_backend = tracker_cfg.get("backend", "iou")
    if tracker_backend not in VALID_TRACKER_BACKENDS:
        raise ConfigError(
            f"Preset '{name}': unknown tracker backend '{tracker_backend}'. "
            f"Valid options: {', '.join(sorted(VALID_TRACKER_BACKENDS))}."
        )

    for key in _NUMERIC_DETECTOR_KEYS:
        if key in detector_cfg and not isinstance(detector_cfg[key], (int, float)):
            raise ConfigError(f"Preset '{name}': detector.{key} must be a number.")

    for key in _NUMERIC_TRACKER_KEYS:
        if key in tracker_cfg and not isinstance(tracker_cfg[key], (int, float)):
            raise ConfigError(f"Preset '{name}': tracker.{key} must be a number.")

    return AppConfig(name=name, detector=detector_cfg, tracker=tracker_cfg, logging=logging_cfg, debug=debug)
