# Zafar Defense Systems — Detection/Tracking MVP

Early R&D MVP for a modular drone **observation** system: video in, aerial
object detection, target tracking, a simple operator UI, and simple
logging.

**This is strictly observation-only software.** No jamming, no kinetic
effects, no weaponization, no autonomous engagement, no attack logic, no
effectors, no target neutralization exist anywhere in this codebase, and
none of the interfaces below are designed to be extended toward that.

## What's in here

```
detector/    Frame -> Detection[]. Default backend is a zero-dependency
             motion detector (OpenCV background subtraction) — a
             placeholder, not a real drone classifier. Optional Ultralytics
             YOLO adapter included (see licensing note below).
tracker/     Detection[] -> Track[] with persistent IDs. Default backend is
             a small dependency-free greedy IoU tracker. Optional ByteTrack
             adapter (Roboflow `trackers`, Apache-2.0) included.
control/     Pipeline orchestration, a small state machine, and event
             types. Observation-only — no actuation code exists here.
ui/          Minimal Streamlit operator view: load a video, see overlaid
             boxes/IDs/confidence, pipeline status, live event log.
telemetry/   JSONL event logger (detections, tracks, dropped frames,
             system events).
scripts/     Headless CLI runner (no browser required).
tests/       Unit + smoke tests, no GPU or real video file required.
docs/        Architecture, decisions log, known limitations, this plan.
config/      Example YAML config for detector/tracker backend selection.
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the operator UI (upload a video file in the browser):
streamlit run ui/app.py

# Or run headless on a video file:
python scripts/run_pipeline.py --source path/to/video.mp4

# Or on a webcam:
python scripts/run_pipeline.py --source 0

# Run tests:
python -m pytest tests/ -v
```

Logs are written as JSONL to `logs/<run_name>.jsonl` — one JSON object per
line, easy to `tail -f`, `jq`, or load with
`pandas.read_json(path, lines=True)`.

## Swapping in a real detector/tracker

Both are selected via a config dict passed to
`detector.factory.build_detector()` / `tracker.factory.build_tracker()` —
see `config/example.yaml`. To add a new backend, implement
`detector.base.BaseDetector` or `tracker.base.BaseTracker` in a new file
and register it in the corresponding factory. Nothing else in the codebase
needs to change.

**Licensing note:** the optional Ultralytics YOLO detector backend
(`detector/ultralytics_detector.py`) is AGPL-3.0 and requires a commercial
license from Ultralytics for closed-source production use — it is not
enabled by default. See `docs/DECISIONS.md` for the full reasoning and for
permissively-licensed alternatives (RF-DETR, YOLOX — both Apache-2.0).

## Known limitations

See `docs/known-limitations.md`. Headline items: the default detector is a
motion-blob detector (not a drone classifier), the default tracker has no
re-identification across occlusion, and nothing in this repo has been
benchmarked for real-world FPS or accuracy.

## Documents

- `docs/software-mvp-plan.md` — implementation plan for this pass.
- `docs/architecture.md` — data flow, module boundaries, interfaces.
- `docs/DECISIONS.md` — key technical/licensing decisions and why.
- `docs/known-limitations.md` — what this MVP does not do yet.
