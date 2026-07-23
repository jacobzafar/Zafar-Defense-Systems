"""End-to-end tests for the Streamlit operator console, using Streamlit's
own `AppTest` harness (executes the real script in-process, no browser).

These exercise exactly the bug class this app is prone to: state left over
across reruns, widget defaults not matching a preset, and unhandled
exceptions surfacing as a raw traceback instead of the app's own error
panel. Kept to a handful of cases — fast and deterministic (the "demo"
source is fully synthetic, no file or network I/O beyond writing to a
tmp_path log directory).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def _start_button(at):
    return next(b for b in at.sidebar.button if b.label == "Start run")


def _reset_button(at):
    return next(b for b in at.sidebar.button if b.label == "Reset")


def test_initial_load_defaults_to_ready_with_demo_mode():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert list(at.exception) == []
    assert at.session_state["app_state"] == "ready"
    assert at.session_state["preset_name"] == "default"
    assert at.session_state["detector_backend"] == "motion"
    assert at.session_state["source_kind_widget"] == "Demo mode (synthetic)"


def test_preset_switch_updates_backend_and_thresholds():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.sidebar.selectbox(key="preset_name").set_value("fast").run(timeout=30)

    assert list(at.exception) == []
    assert at.session_state["min_area_px"] == 200  # matches config/presets.yaml's "fast" preset


def test_full_demo_run_completes_and_populates_summary(tmp_path):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.session_state["log_dir"] = str(tmp_path)

    _start_button(at).click().run(timeout=60)

    assert list(at.exception) == []
    assert at.session_state["app_state"] == "completed"
    kpi = at.session_state["last_kpi"]
    assert kpi["frame_count"] > 0
    summary = at.session_state["last_summary"]
    assert summary.source == "demo"
    assert summary.total_frames == kpi["frame_count"]


def test_reset_clears_results_but_keeps_source_ready(tmp_path):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.session_state["log_dir"] = str(tmp_path)

    _start_button(at).click().run(timeout=60)
    assert at.session_state["app_state"] == "completed"

    _reset_button(at).click().run(timeout=30)

    assert list(at.exception) == []
    assert at.session_state["last_summary"] is None
    # Demo mode is still selected, so the app should fall back to "ready", not get stuck.
    assert at.session_state["app_state"] == "ready"


def test_missing_optional_dependency_is_a_friendly_error_not_a_crash(tmp_path):
    # Uses "ultralytics" specifically: unlike "torchvision", it is never
    # installed by either of this repo's dependency states (core-only or
    # core+torchvision-extra), so this stays deterministic either way.
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.session_state["log_dir"] = str(tmp_path)

    at.sidebar.selectbox(key="detector_backend").set_value("ultralytics").run(timeout=30)
    _start_button(at).click().run(timeout=30)

    assert list(at.exception) == []
    assert at.session_state["app_state"] == "error"
    assert at.session_state["error_kind"] == "missing_dependency"


def test_drone_backend_with_no_weights_is_a_friendly_error_not_a_crash(tmp_path):
    # Requires torch: without it, DroneDetector raises ImportError before
    # ever checking for weights, which is also correctly handled (as
    # "missing_dependency") but is a different branch than this test targets.
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.session_state["log_dir"] = str(tmp_path)

    at.sidebar.selectbox(key="detector_backend").set_value("drone").run(timeout=30)
    _start_button(at).click().run(timeout=30)

    assert list(at.exception) == []
    assert at.session_state["app_state"] == "error"
    assert at.session_state["error_kind"] == "missing_weights"


def test_upload_mode_disables_run_until_a_file_is_provided():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.sidebar.radio(key="source_kind_widget").set_value("Upload video").run(timeout=30)

    assert list(at.exception) == []
    assert at.session_state["app_state"] == "idle"
    assert _start_button(at).disabled is True


def test_header_shows_the_active_preset():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    header_markdown = [m.value for m in at.markdown if 'class="zds-header"' in m.value]
    assert header_markdown
    assert "PRESET" in header_markdown[0]
    assert "default" in header_markdown[0]


def test_ultralytics_backend_shows_agpl_licensing_warning():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.sidebar.selectbox(key="detector_backend").set_value("ultralytics").run(timeout=30)

    assert list(at.exception) == []
    warning_texts = [w.value for w in at.sidebar.warning]
    assert any("AGPL" in text for text in warning_texts)


def test_motion_backend_shows_no_licensing_warning():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert list(at.exception) == []
    assert list(at.sidebar.warning) == []


def test_track_lifecycle_events_appear_in_event_log(tmp_path):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.session_state["log_dir"] = str(tmp_path)
    at.sidebar.number_input(key="max_frames_widget").set_value(10).run(timeout=30)

    _start_button(at).click().run(timeout=60)

    assert list(at.exception) == []
    event_lines = at.session_state["event_lines"]
    assert any("ACQUIRED" in line for line in event_lines)
