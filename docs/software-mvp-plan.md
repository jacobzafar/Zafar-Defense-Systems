# Software MVP Plan — Detection/Tracking Pipeline

Status: implemented in this pass.
Scope: video in -> detect -> track -> overlay/UI -> log. No jamming, no
kinetic effects, no autonomy, no weapon/engagement/effector logic anywhere
in this repo.

## 1. What exists vs. what's missing

This sandbox has no network access, so the actual GitHub repo
(jacobzafar/Zafar-Defense-Systems, then named Zafar-Defense-Systems-AB)
could not be cloned or inspected. See docs/DECISIONS.md entry #1. This
MVP is built standalone here and is
meant to be copied into the real repo. Nothing is assumed to pre-exist.

## 2. Target architecture (summary)

video source (file/webcam/RTSP)
        |
        v
  detector.BaseDetector          -> Detection[] (normalized 0..1 coords)
        |
        v
  tracker.BaseTracker            -> Track[] (stable IDs over time)
        |
        v
  control.Pipeline               -> orchestrates the above, per-frame loop
        |           |
        v           v
  ui.overlay    telemetry.EventLogger (JSONL)
        |
        v
  ui.app (Streamlit) or scripts/run_pipeline.py (headless CLI)

Full detail in docs/architecture.md.

## 3. Module responsibilities

- detector/  - frame -> normalized Detection objects. Ships with a
  zero-dependency MotionDetector (OpenCV background subtraction) so the
  pipeline runs with no ML framework installed. A guarded, optional
  UltralyticsDetector adapter is included for later upgrade (licensing
  note: docs/DECISIONS.md entry #2).
- tracker/   - Detection[] -> Track[] with persistent IDs. Ships with a
  dependency-free IoUTracker. An optional adapter for Roboflow's
  Apache-2.0 `trackers` package sits behind the same interface.
- control/   - pipeline loop, a small state machine
  (IDLE -> RUNNING -> STOPPED/ERROR), and event types. No actuation, no
  effector code, no targeting/engagement logic exists anywhere in this
  module or repo.
- ui/        - minimal Streamlit app: load a video, run the pipeline, show
  overlaid boxes/IDs/confidence, pipeline status, and a live event log.
- telemetry/ - JSONL logger for detections, tracks, dropped frames, and
  system events, one line per record.
- scripts/   - headless CLI runner for environments without a browser.
- tests/     - fast unit tests for tracker/detector logic, no GPU or video
  file required.

## 4. Assumptions & constraints

- Python 3.10+, OpenCV for video I/O and drawing.
- The default detector (MotionDetector) is a placeholder to validate the
  pipeline end-to-end; it is NOT a real drone classifier. Swapping in a
  trained model is the expected next step (docs/known-limitations.md).
- Dependency install and the full pipeline (tests, headless CLI run,
  Streamlit UI boot) have since been verified end-to-end in a real
  environment; see docs/known-limitations.md for the one dependency
  change that came out of that pass (`opencv-python-headless`).
- control/ is explicitly observation-only: it only ever emits events and
  state, never commands that could drive an effector.

## 5. Build order (this pass)

1. detector/ base interface + MotionDetector + optional adapter stub.
2. tracker/ base interface + IoUTracker + optional adapter stub.
3. telemetry/ JSONL logger.
4. control/ pipeline + state machine + events.
5. ui/ Streamlit app + overlay drawing helper.
6. scripts/run_pipeline.py headless runner.
7. tests/ for tracker/detector logic.
8. docs/architecture.md, docs/DECISIONS.md, docs/known-limitations.md.
9. Root README.md, requirements.txt, pyproject.toml, Makefile,
   config/example.yaml.
