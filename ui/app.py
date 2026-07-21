"""Streamlit operator console.

Run with: streamlit run ui/app.py

Dark, panel-based operator dashboard for the observation-only detection/
tracking pipeline: pick a source (upload, the built-in synthetic demo
clip, or a server-side webcam), pick a config preset and backends, run,
and watch live KPIs / annotated frames / a structured event log, then
review a run summary and export the JSONL telemetry log. There is no
control here that can command an effector — no such abstraction exists
anywhere in this repo.

State model: this page follows Streamlit's normal rerun-on-interaction
model (no background threads), so a run is a single blocking loop that
live-updates placeholders as it goes — the same approach as before, just
with a much richer render per frame. Because of that, there is no true
live "stop mid-run" control; use the "Max frames" limiter to bound a run
instead (documented in the sidebar and in README.md).
"""

from __future__ import annotations

import html
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import streamlit as st

from config.loader import ConfigError, load_presets
from control.pipeline import Pipeline
from detector.factory import build_detector
from telemetry.logger import EventLogger
from telemetry.summary import summarize_log
from tracker.factory import build_tracker
from ui import theme
from ui.overlay import draw_tracks

DETECTOR_BACKENDS = ["motion", "torchvision", "ultralytics", "drone"]
TRACKER_BACKENDS = ["iou", "bytetrack"]

_ERROR_HINTS = {
    "missing_dependency": (
        "A selected backend needs an optional dependency that isn't installed in this "
        "environment. Install it (see the sidebar note or README.md) or switch back to the "
        "default motion detector / IoU tracker."
    ),
    "missing_weights": (
        "The drone-specific backend has no trained weights at the given path yet. Train one "
        "with detector/train.py (see docs/DECISIONS.md), or switch to a different backend."
    ),
    "bad_source": (
        "The video source could not be opened or read. Check that the file isn't corrupt, "
        "try re-encoding to H.264 MP4, or confirm the webcam index is correct and available."
    ),
    "unexpected": "An unexpected error occurred. Check the terminal running Streamlit for the full traceback.",
}


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def _load_presets_safely():
    try:
        return load_presets(), None
    except ConfigError as exc:
        return {}, str(exc)


def _apply_preset(name: str, presets: dict) -> None:
    preset = presets.get(name)
    if preset is None:
        return
    st.session_state.detector_backend = preset.detector.get("backend", "motion")
    st.session_state.tracker_backend = preset.tracker.get("backend", "iou")
    st.session_state.debug_mode = preset.debug
    st.session_state.min_area_px = int(preset.detector.get("min_area_px", 150))
    st.session_state.max_area_fraction = float(preset.detector.get("max_area_fraction", 0.25))
    st.session_state.var_threshold = float(preset.detector.get("var_threshold", 32.0))
    st.session_state.history = int(preset.detector.get("history", 300))
    st.session_state.confidence_threshold = float(preset.detector.get("confidence_threshold", 0.35))
    st.session_state.iou_threshold = float(preset.tracker.get("iou_threshold", 0.3))
    st.session_state.max_age = int(preset.tracker.get("max_age", 15))
    st.session_state.min_hits_to_confirm = int(preset.tracker.get("min_hits_to_confirm", 1))
    st.session_state.log_dir = preset.logging.get("log_dir", "logs")


def _init_state(presets: dict) -> None:
    if st.session_state.get("_initialized"):
        return
    st.session_state._initialized = True
    st.session_state.app_state = "idle"
    st.session_state.last_summary = None
    st.session_state.last_kpi = None
    st.session_state.error_message = None
    st.session_state.error_kind = None
    st.session_state.log_path = None
    st.session_state.event_lines = []
    st.session_state.last_frame = None
    st.session_state.run_name_used = None
    st.session_state.source_kind_widget = "Demo mode (synthetic)"
    st.session_state.preset_name = "default" if "default" in presets else next(iter(presets), "default")
    _apply_preset(st.session_state.preset_name, presets)


def _reset_run_state() -> None:
    st.session_state.app_state = "idle"
    st.session_state.last_summary = None
    st.session_state.last_kpi = None
    st.session_state.error_message = None
    st.session_state.error_kind = None
    st.session_state.log_path = None
    st.session_state.event_lines = []
    st.session_state.last_frame = None
    st.session_state.run_name_used = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def _render_sidebar(presets: dict, preset_error: str | None) -> dict:
    st.sidebar.markdown("### Mission setup")

    if preset_error:
        st.sidebar.error(f"Preset catalog failed to load: {preset_error}")

    preset_names = sorted(presets) or ["default"]
    st.sidebar.selectbox(
        "Config preset",
        preset_names,
        key="preset_name",
        on_change=lambda: _apply_preset(st.session_state.preset_name, presets),
        help="Loads a tuned starting point for the detector/tracker settings below. "
        "You can still adjust anything after picking one.",
    )

    st.sidebar.radio(
        "Video source",
        ["Demo mode (synthetic)", "Upload video", "Webcam (server-side)"],
        key="source_kind_widget",
    )
    source_kind = st.session_state.source_kind_widget

    uploaded_file = None
    if source_kind == "Upload video":
        uploaded_file = st.sidebar.file_uploader(
            "Video file", type=["mp4", "avi", "mov", "mkv"], key="uploaded_file_widget"
        )
    elif source_kind == "Webcam (server-side)":
        st.sidebar.number_input("Webcam index", min_value=0, max_value=10, value=0, key="webcam_index_widget")
        st.sidebar.caption(
            "Opens a camera attached to the machine running this Streamlit server — "
            "not your browser's camera."
        )
    else:
        st.sidebar.caption(
            "Uses a built-in synthetic clip (a few moving shapes on a plain background). "
            "For verifying the UI, pipeline, and logs before real footage exists — not a "
            "substitute for real drone footage."
        )

    st.sidebar.selectbox("Detector backend", DETECTOR_BACKENDS, key="detector_backend")
    st.sidebar.selectbox("Tracker backend", TRACKER_BACKENDS, key="tracker_backend")

    with st.sidebar.expander("Advanced settings", expanded=False):
        if st.session_state.detector_backend == "motion":
            st.number_input(
                "Min blob size (px²)", min_value=1, key="min_area_px",
                help="Minimum motion-blob size counted as a detection. Lower = more sensitive, more false positives.",
            )
            st.slider(
                "Max blob size (fraction of frame)", 0.01, 1.0, key="max_area_fraction",
                help="Ignore blobs larger than this fraction of the frame — usually a lighting change or camera jolt.",
            )
            st.number_input(
                "Motion sensitivity threshold", min_value=1.0, key="var_threshold",
                help="Lower = detects subtler motion, but more background noise.",
            )
            st.number_input(
                "Background learning frames", min_value=1, key="history",
                help="How many frames the background model learns from.",
            )
        else:
            st.slider(
                "Detection confidence threshold", 0.0, 1.0, key="confidence_threshold",
                help="Minimum model confidence required to count as a detection.",
            )
            if st.session_state.detector_backend == "ultralytics":
                st.text_input(
                    "Weights path (.pt)", key="weights_path",
                    help="Path to a trained YOLO weights file on this machine. Required for this backend.",
                )
            elif st.session_state.detector_backend == "drone":
                st.text_input(
                    "Drone weights path (.pt)", key="weights_path",
                    help="Path to weights produced by detector/train.py. Required for this backend — "
                    "there are no trained drone-specific weights shipped with this repo.",
                )
            dep = {
                "torchvision": "torch/torchvision",
                "ultralytics": "ultralytics",
                "drone": "torch/torchvision",
            }[st.session_state.detector_backend]
            st.caption(f"Requires the optional `{dep}` package(s) to be installed.")

        st.divider()

        if st.session_state.tracker_backend == "iou":
            st.slider(
                "Track match threshold (IoU)", 0.05, 0.9, key="iou_threshold",
                help="How much overlap is required to keep the same track ID between frames.",
            )
            st.number_input(
                "Max frames to keep a lost track", min_value=1, key="max_age",
                help="How many frames a track survives without a matching detection before it's dropped.",
            )
            st.number_input(
                "Frames to confirm a new track", min_value=1, key="min_hits_to_confirm",
                help="How many consecutive detections are needed before a track is shown as confirmed.",
            )
        else:
            st.caption("ByteTrack has no extra parameters here; requires the `trackers`/`supervision` packages.")

        st.divider()
        st.checkbox("Debug mode (per-frame timing + raw internals)", key="debug_mode")

    st.sidebar.text_input("Run name (optional)", key="run_name_input", placeholder="e.g. rooftop-test-1")
    st.sidebar.number_input(
        "Max frames (0 = unlimited)", min_value=0, value=0, key="max_frames_widget",
        help="Caps how many frames this run processes. There is no live mid-run stop control "
        "in this synchronous demo UI — use this limiter instead of a Stop button.",
    )

    can_run = source_kind != "Upload video" or uploaded_file is not None

    col_run, col_reset = st.sidebar.columns(2)
    run_clicked = col_run.button("Start run", type="primary", disabled=not can_run, width="stretch")
    reset_clicked = col_reset.button("Reset", width="stretch")

    if source_kind == "Upload video" and uploaded_file is not None:
        size_kb = len(uploaded_file.getvalue()) / 1024
        source_label = f"{uploaded_file.name} ({size_kb:.0f} KB)"
    elif source_kind == "Webcam (server-side)":
        source_label = f"Webcam index {st.session_state.webcam_index_widget} (server-side camera)"
    elif source_kind == "Upload video":
        source_label = None
    else:
        source_label = "Synthetic demo clip (moving targets, no file needed)"

    return {
        "source_kind": source_kind,
        "uploaded_file": uploaded_file,
        "can_run": can_run,
        "source_label": source_label,
        "run_clicked": run_clicked,
        "reset_clicked": reset_clicked,
        "max_frames": st.session_state.max_frames_widget,
        "run_name": st.session_state.run_name_input,
    }


def _build_configs() -> tuple[dict, dict]:
    detector_config: dict = {"backend": st.session_state.detector_backend}
    if st.session_state.detector_backend == "motion":
        detector_config.update(
            min_area_px=st.session_state.min_area_px,
            max_area_fraction=st.session_state.max_area_fraction,
            var_threshold=st.session_state.var_threshold,
            history=st.session_state.history,
        )
    else:
        detector_config.update(confidence_threshold=st.session_state.confidence_threshold)
        if st.session_state.detector_backend in ("ultralytics", "drone"):
            detector_config["weights_path"] = st.session_state.get("weights_path", "")

    tracker_config: dict = {"backend": st.session_state.tracker_backend}
    if st.session_state.tracker_backend == "iou":
        tracker_config.update(
            iou_threshold=st.session_state.iou_threshold,
            max_age=st.session_state.max_age,
            min_hits_to_confirm=st.session_state.min_hits_to_confirm,
        )

    return detector_config, tracker_config


def _materialize_source(sidebar: dict) -> tuple[str | int | None, str | None]:
    """Turn the sidebar's source selection into a real source + a validation error, if any."""
    kind = sidebar["source_kind"]
    if kind == "Demo mode (synthetic)":
        return "demo", None
    if kind == "Webcam (server-side)":
        return int(st.session_state.webcam_index_widget), None

    uploaded_file = sidebar["uploaded_file"]
    if uploaded_file is None:
        return None, "No file uploaded."
    data = uploaded_file.getvalue()
    if len(data) == 0:
        return None, "Uploaded file is empty (0 bytes) — choose a different file."
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name, None


# --------------------------------------------------------------------------
# Header / KPIs
# --------------------------------------------------------------------------

def _state_badge_kind(state: str) -> str:
    return {
        "idle": "idle",
        "ready": "ready",
        "running": "running",
        "completed": "completed",
        "error": "error",
    }.get(state, "neutral")


def _render_header() -> None:
    state = st.session_state.app_state
    badges = [
        theme.badge(state.upper(), _state_badge_kind(state)),
        theme.badge(f"DET · {st.session_state.detector_backend}", "neutral"),
        theme.badge(f"TRK · {st.session_state.tracker_backend}", "neutral"),
    ]
    if st.session_state.get("run_name_used"):
        badges.append(theme.badge(f"RUN · {st.session_state.run_name_used}", "neutral"))

    st.markdown(
        '<div class="zds-header">'
        '<div><div class="zds-title">Zafar Defense Systems — Detection / Tracking Console</div>'
        '<div class="zds-subtitle">Observation-only drone detection &amp; tracking MVP — '
        "no jamming, no kinetic effects, no autonomous engagement.</div></div>"
        f'<div class="zds-badges">{"".join(badges)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _kpi_html(frame_count, detections_total, active_tracks, dropped_total, fps, detector_name, tracker_name, source_label) -> str:
    cards = [
        theme.kpi_card("State", st.session_state.app_state.capitalize()),
        theme.kpi_card("Frames processed", str(frame_count)),
        theme.kpi_card("Detections", str(detections_total)),
        theme.kpi_card("Active tracks", str(active_tracks)),
        theme.kpi_card("Dropped frames", str(dropped_total)),
        theme.kpi_card("Throughput", f"{fps:.1f} fps" if fps else "—"),
        theme.kpi_card("Detector", detector_name),
        theme.kpi_card("Tracker", tracker_name),
        theme.kpi_card("Source", source_label or "—"),
    ]
    return theme.kpi_row(cards)


# --------------------------------------------------------------------------
# Run execution
# --------------------------------------------------------------------------

def _execute_run(source, source_label, detector_config, tracker_config, log_dir, run_name, max_frames, debug_mode, placeholders) -> None:
    kpi_ph, frame_ph, debug_ph, event_ph = placeholders

    _reset_run_state()
    st.session_state.app_state = "running"

    resolved_run_name = (run_name or "").strip() or time.strftime("run_%Y%m%d_%H%M%S")
    st.session_state.run_name_used = resolved_run_name

    try:
        logger = EventLogger(log_dir=log_dir, run_name=resolved_run_name)
    except OSError as exc:
        st.session_state.app_state = "error"
        st.session_state.error_kind = "unexpected"
        st.session_state.error_message = f"Could not create the log directory/file: {exc}"
        return

    try:
        detector = build_detector(detector_config)
        tracker = build_tracker(tracker_config)
    except ImportError as exc:
        st.session_state.app_state = "error"
        st.session_state.error_kind = "missing_dependency"
        st.session_state.error_message = str(exc)
        logger.close()
        return
    except FileNotFoundError as exc:
        # Covers detector.drone_detector.DroneWeightsNotFoundError.
        st.session_state.app_state = "error"
        st.session_state.error_kind = "missing_weights"
        st.session_state.error_message = str(exc)
        logger.close()
        return
    except (KeyError, ValueError) as exc:
        st.session_state.app_state = "error"
        st.session_state.error_kind = "unexpected"
        st.session_state.error_message = f"Invalid detector/tracker configuration: {exc}"
        logger.close()
        return

    pipeline = Pipeline(detector=detector, tracker=tracker, logger=logger)

    frame_count = 0
    detections_total = 0
    dropped_total = 0
    track_ids: set[int] = set()
    last_fps = 0.0

    try:
        for result in pipeline.run(source):
            frame_count += 1
            detections_total += result.detection_count
            dropped_total += int(result.dropped)
            for t in result.tracks:
                track_ids.add(t.track_id)
            last_fps = result.fps or last_fps

            if result.frame is not None:
                annotated = draw_tracks(result.frame, result.tracks)
                st.session_state.last_frame = annotated
                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

            kpi_ph.markdown(
                _kpi_html(frame_count, detections_total, len(result.tracks), dropped_total, last_fps, detector.name, tracker.name, source_label),
                unsafe_allow_html=True,
            )

            if result.dropped:
                line = theme.event_line("error", "DROPPED", f"frame {result.frame_index} — {result.error or 'unknown error'}")
            else:
                line = theme.event_line(
                    "neutral", "FRAME",
                    f"#{result.frame_index} · {result.detection_count} det · {len(result.tracks)} trk · {result.total_ms:.1f} ms total",
                )
            st.session_state.event_lines.append(line)
            st.session_state.event_lines = st.session_state.event_lines[-40:]
            event_ph.markdown(_event_log_html(), unsafe_allow_html=True)

            if debug_mode:
                debug_ph.markdown(
                    theme.panel(
                        "Frame timing (last frame)",
                        f"decode {result.decode_ms:.1f} ms &middot; detect {result.detector_ms:.1f} ms &middot; "
                        f"track {result.tracker_ms:.1f} ms &middot; total {result.total_ms:.1f} ms",
                    ),
                    unsafe_allow_html=True,
                )

            if max_frames and frame_count >= max_frames:
                pipeline.stop()
                break

        st.session_state.app_state = "completed"
    except RuntimeError as exc:
        st.session_state.app_state = "error"
        st.session_state.error_kind = "bad_source"
        st.session_state.error_message = str(exc)
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the operator, not a raw traceback
        st.session_state.app_state = "error"
        st.session_state.error_kind = "unexpected"
        st.session_state.error_message = str(exc)
    finally:
        logger.close()

    st.session_state.log_path = str(logger.log_path)
    st.session_state.last_summary = summarize_log(logger.log_path)
    st.session_state.last_kpi = {
        "frame_count": frame_count,
        "detections_total": detections_total,
        "dropped_total": dropped_total,
        "unique_tracks": len(track_ids),
        "fps": last_fps,
        "source_label": source_label,
    }


# --------------------------------------------------------------------------
# State-dependent rendering
# --------------------------------------------------------------------------

def _error_panel_html() -> str:
    kind = st.session_state.error_kind
    message = st.session_state.error_message or "Unknown error."
    hint = _ERROR_HINTS.get(kind, "")
    body = f"<b>{html.escape(message)}</b>"
    if hint:
        body += f'<br><span style="color:rgba(230,237,243,0.6)">{html.escape(hint)}</span>'
    return theme.panel("Run failed", body)


def _event_log_html() -> str:
    lines = st.session_state.event_lines
    body = "".join(lines) if lines else '<span style="color:rgba(230,237,243,0.4)">No events yet — start a run to see live output.</span>'
    return theme.panel("Event log", f'<div class="event-log">{body}</div>')


def _summary_panel_html() -> str:
    summary = st.session_state.last_summary
    if summary is None:
        return theme.panel("Run summary", "No summary available.")

    note = ""
    if summary.total_detections == 0:
        note = (
            '<br><i style="color:rgba(230,237,243,0.55)">No detections in this run — this can be '
            "expected depending on footage and detector sensitivity, not necessarily a bug.</i>"
        )

    body = (
        f"run: <b>{html.escape(st.session_state.run_name_used or '—')}</b><br>"
        f"source: {html.escape(str(summary.source or '—'))}<br>"
        f"detector: {html.escape(str(summary.detector or '—'))} &middot; tracker: {html.escape(str(summary.tracker or '—'))}<br>"
        f"frames: {summary.total_frames} &middot; detections: {summary.total_detections}<br>"
        f"unique tracks: {summary.unique_track_count} &middot; dropped: {summary.dropped_frames}<br>"
        f"duration: {summary.duration_seconds}s &middot; avg fps: {summary.avg_fps}<br>"
        f'log: <code style="font-size:0.75rem">{html.escape(summary.log_path)}</code>'
        f"{note}"
    )
    return theme.panel("Run summary", body)


def _render_telemetry_export(telemetry_ph) -> None:
    log_path = st.session_state.get("log_path")
    if not log_path:
        telemetry_ph.empty()
        return
    path = Path(log_path)
    with telemetry_ph.container():
        st.markdown(theme.panel("Telemetry", f"log file: <code>{html.escape(str(path))}</code>"), unsafe_allow_html=True)
        if path.exists():
            st.download_button(
                "Download telemetry log (.jsonl)",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/jsonl",
                key="download_log_button",
            )


def _render_state_panels(placeholders: dict, source_label: str | None) -> None:
    kpi_ph, frame_ph, info_ph, event_ph, telemetry_ph = (
        placeholders["kpi"], placeholders["frame"], placeholders["info"], placeholders["event"], placeholders["telemetry"]
    )
    state = st.session_state.app_state

    if state == "idle":
        kpi_ph.empty()
        frame_ph.markdown(
            theme.empty_state("No feed yet", "Choose a video source in the sidebar — Demo mode needs nothing else — then press Start run."),
            unsafe_allow_html=True,
        )
        info_ph.markdown(theme.panel("Status", "Waiting for a video source."), unsafe_allow_html=True)
        event_ph.markdown(_event_log_html(), unsafe_allow_html=True)
        telemetry_ph.empty()

    elif state == "ready":
        kpi_ph.empty()
        frame_ph.markdown(
            theme.empty_state("Ready to run", f"Source selected: {html.escape(source_label or '—')}. Press Start run in the sidebar to begin."),
            unsafe_allow_html=True,
        )
        info_ph.markdown(
            theme.panel("Status", f"Source ready: <b>{html.escape(source_label or '—')}</b><br>Press Start run to begin."),
            unsafe_allow_html=True,
        )
        event_ph.markdown(_event_log_html(), unsafe_allow_html=True)
        telemetry_ph.empty()

    elif state == "completed":
        kpi = st.session_state.last_kpi or {}
        kpi_ph.markdown(
            _kpi_html(
                kpi.get("frame_count", 0), kpi.get("detections_total", 0), kpi.get("unique_tracks", 0),
                kpi.get("dropped_total", 0), kpi.get("fps", 0.0), st.session_state.detector_backend,
                st.session_state.tracker_backend, kpi.get("source_label", source_label),
            ),
            unsafe_allow_html=True,
        )
        if st.session_state.last_frame is not None:
            frame_ph.image(cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
        else:
            frame_ph.markdown(theme.empty_state("Run complete", "No frames were available to preview."), unsafe_allow_html=True)
        info_ph.markdown(_summary_panel_html(), unsafe_allow_html=True)
        event_ph.markdown(_event_log_html(), unsafe_allow_html=True)
        _render_telemetry_export(telemetry_ph)

    elif state == "error":
        kpi_ph.empty()
        if st.session_state.last_frame is not None:
            frame_ph.image(cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
        else:
            frame_ph.markdown(theme.empty_state("Run failed", "See the error panel for details."), unsafe_allow_html=True)
        info_ph.markdown(_error_panel_html(), unsafe_allow_html=True)
        event_ph.markdown(_event_log_html(), unsafe_allow_html=True)
        _render_telemetry_export(telemetry_ph)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.set_page_config(page_title="Zafar Defense Systems — Detection/Tracking Console", layout="wide", initial_sidebar_state="expanded")
theme.inject_theme()

_presets, _preset_error = _load_presets_safely()
_init_state(_presets)

_sidebar = _render_sidebar(_presets, _preset_error)

if _sidebar["reset_clicked"]:
    _reset_run_state()
    st.rerun()

if st.session_state.app_state == "idle" and _sidebar["can_run"]:
    st.session_state.app_state = "ready"
elif st.session_state.app_state == "ready" and not _sidebar["can_run"]:
    st.session_state.app_state = "idle"

_render_header()

kpi_placeholder = st.empty()
frame_col, info_col = st.columns([2, 1])
frame_placeholder = frame_col.empty()
info_placeholder = info_col.empty()
debug_placeholder = st.empty()
event_placeholder = st.empty()
telemetry_placeholder = st.empty()

_placeholders = {
    "kpi": kpi_placeholder,
    "frame": frame_placeholder,
    "info": info_placeholder,
    "event": event_placeholder,
    "telemetry": telemetry_placeholder,
}

if _sidebar["run_clicked"] and _sidebar["can_run"]:
    source, source_error = _materialize_source(_sidebar)
    if source_error is not None:
        st.session_state.app_state = "error"
        st.session_state.error_kind = "unexpected"
        st.session_state.error_message = source_error
    else:
        detector_config, tracker_config = _build_configs()
        _execute_run(
            source,
            _sidebar["source_label"],
            detector_config,
            tracker_config,
            st.session_state.log_dir,
            _sidebar["run_name"],
            _sidebar["max_frames"],
            st.session_state.debug_mode,
            (kpi_placeholder, frame_placeholder, debug_placeholder, event_placeholder),
        )

_render_state_panels(_placeholders, _sidebar["source_label"])

st.caption(
    "Observation-only: no jamming, no kinetic effects, no autonomous engagement, no attack "
    "logic, no effectors exist anywhere in this codebase."
)
