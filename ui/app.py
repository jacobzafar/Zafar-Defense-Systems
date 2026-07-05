"""Minimal Streamlit operator UI.

Run with: streamlit run ui/app.py

Loads a video file, runs the detection/tracking pipeline, shows overlaid
boxes/IDs/confidence, pipeline status, and a scrolling event log. This is
an observation dashboard only — there is no control here that can command
an effector, because no such abstraction exists anywhere in this repo.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import streamlit as st

from control.pipeline import Pipeline
from detector.factory import build_detector
from telemetry.logger import EventLogger
from tracker.factory import build_tracker
from ui.overlay import draw_tracks

st.set_page_config(page_title="Zafar Defense Systems — Detection/Tracking MVP", layout="wide")
st.title("Detection / Tracking MVP — Operator View")
st.caption(
    "Observation-only prototype: video in, aerial object detection, target "
    "tracking, simple logging. No jamming, no kinetic effects, no autonomy."
)

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    detector_backend = st.selectbox("Detector backend", ["motion", "torchvision", "ultralytics"], index=0)
    tracker_backend = st.selectbox("Tracker backend", ["iou", "bytetrack"], index=0)
    run_button = st.button("Run pipeline", type="primary", disabled=uploaded_file is None)

status_placeholder = st.empty()
frame_placeholder = st.empty()
metrics_placeholder = st.empty()
log_placeholder = st.empty()

if run_button and uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    detector = build_detector({"backend": detector_backend})
    tracker = build_tracker({"backend": tracker_backend})
    logger = EventLogger(log_dir="logs", run_name="streamlit_run")
    pipeline = Pipeline(detector=detector, tracker=tracker, logger=logger)

    status_placeholder.info("Pipeline running...")
    event_lines: list[str] = []

    try:
        for result in pipeline.run(video_path):
            capture_check = cv2.VideoCapture(video_path)
            capture_check.set(cv2.CAP_PROP_POS_FRAMES, result.frame_index)
            ok, raw_frame = capture_check.read()
            capture_check.release()

            if ok:
                annotated = draw_tracks(raw_frame, result.tracks)
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

            metrics_placeholder.markdown(
                f"**Frame:** {result.frame_index} | "
                f"**Detections:** {result.detection_count} | "
                f"**Active tracks:** {len(result.tracks)} | "
                f"**Inference:** {result.inference_ms} ms | "
                f"**Dropped:** {result.dropped}"
            )

            event_lines.append(
                f"frame={result.frame_index} tracks={len(result.tracks)} "
                f"dets={result.detection_count} err={result.error}"
            )
            log_placeholder.code("\n".join(event_lines[-20:]))

        status_placeholder.success(f"Pipeline finished. Log written to {logger.log_path}")
    finally:
        logger.close()
elif uploaded_file is None:
    status_placeholder.warning("Upload a video file to begin.")
