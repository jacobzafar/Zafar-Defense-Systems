import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import AppConfig, ConfigError, get_preset, load_presets

REPO_PRESETS_PATH = Path(__file__).resolve().parent.parent / "config" / "presets.yaml"


def test_repo_presets_load_and_include_expected_names():
    presets = load_presets(REPO_PRESETS_PATH)
    assert {"default", "demo", "debug", "fast"} <= set(presets)
    for preset in presets.values():
        assert isinstance(preset, AppConfig)


def test_debug_preset_has_debug_flag_set():
    presets = load_presets(REPO_PRESETS_PATH)
    assert presets["debug"].debug is True
    assert presets["default"].debug is False


def test_get_preset_returns_named_preset():
    presets = load_presets(REPO_PRESETS_PATH)
    preset = get_preset("demo", presets)
    assert preset.name == "demo"
    assert preset.detector["backend"] == "motion"


def test_get_preset_unknown_name_raises():
    presets = load_presets(REPO_PRESETS_PATH)
    with pytest.raises(ConfigError, match="Unknown preset"):
        get_preset("does-not-exist", presets)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_presets(tmp_path / "missing.yaml")


def test_unknown_detector_backend_raises(tmp_path):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(
        textwrap.dedent(
            """
            presets:
              broken:
                detector:
                  backend: not-a-real-backend
            """
        )
    )
    with pytest.raises(ConfigError, match="unknown detector backend"):
        load_presets(bad_path)


def test_unknown_tracker_backend_raises(tmp_path):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(
        textwrap.dedent(
            """
            presets:
              broken:
                tracker:
                  backend: not-a-real-tracker
            """
        )
    )
    with pytest.raises(ConfigError, match="unknown tracker backend"):
        load_presets(bad_path)


def test_non_numeric_detector_field_raises(tmp_path):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(
        textwrap.dedent(
            """
            presets:
              broken:
                detector:
                  backend: motion
                  min_area_px: "not-a-number"
            """
        )
    )
    with pytest.raises(ConfigError, match="must be a number"):
        load_presets(bad_path)


def test_empty_presets_file_raises(tmp_path):
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("presets: {}\n")
    with pytest.raises(ConfigError, match="No presets found"):
        load_presets(empty_path)
