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
             placeholder, not a real drone classifier. Optional real
             (trained) backends: a pretrained torchvision SSDLite
             detector (permissive license, recommended for demo clips)
             and an Ultralytics YOLO adapter (see licensing note below).
tracker/     Detection[] -> Track[] with persistent IDs. Default backend is
             a small dependency-free greedy IoU tracker. Optional ByteTrack
             adapter (Roboflow `trackers`, Apache-2.0) included.
control/     Pipeline orchestration, a state machine, event types, and the
             built-in synthetic demo source. Observation-only — no
             actuation code exists here.
ui/          Streamlit operator console: dark theme, KPI/status panels,
             live annotated video, structured event log, run summary,
             telemetry export. See "The operator console" below.
telemetry/   JSONL event logger (detections, tracks, dropped frames,
             system events) + a run-summary utility shared by the CLI
             and the UI.
scripts/     Headless CLI runner + a standalone log-summary tool.
tests/       Unit, smoke, and UI tests — no GPU, real video file, or
             browser required (the UI is tested via Streamlit's own
             in-process `AppTest` harness).
docs/        Architecture, decisions log, known limitations, this plan,
             the demo gap analysis this pass was scoped from.
config/      Named, validated presets (demo/default/debug/fast) plus a
             hand-written config reference.
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the operator console (opens with Demo mode selected — no video needed):
streamlit run ui/app.py

# Or run headless against the built-in synthetic demo clip:
python scripts/run_pipeline.py --source demo --preset demo

# Or headless on a real video file:
python scripts/run_pipeline.py --source path/to/video.mp4

# Or with the real pretrained detector (needs: pip install torch torchvision):
python scripts/run_pipeline.py --source path/to/video.mp4 --detector torchvision

# Or on a webcam:
python scripts/run_pipeline.py --source 0

# Summarize a completed run's log:
python scripts/summarize_log.py logs/run_20260101_120000.jsonl

# Run tests:
python -m pytest tests/ -v
```

`make demo` runs the same synthetic demo clip headlessly for a 30-second
sanity check with no arguments and no video file.

Logs are written as JSONL to `<log_dir>/<run_name>.jsonl` (`log_dir`
defaults to `logs/`, or a preset's own subfolder, e.g. `logs/demo/`) — one
JSON object per line, easy to `tail -f`, `jq`, or load with
`pandas.read_json(path, lines=True)`.

## The operator console

`streamlit run ui/app.py` opens a dark, panel-based console (see
`.streamlit/config.toml` for the base theme, `ui/theme.py` for the
badges/KPI-card/panel styling layer):

- **Header** — product title, a run-state badge (idle/ready/running/
  completed/error), and the active detector/tracker/run-name badges.
- **Sidebar ("Mission setup")** — pick a config preset, a video source
  (Demo mode, file upload, or a server-side webcam index), the detector/
  tracker backend, an "Advanced settings" expander for thresholds and a
  debug toggle, a max-frames limiter, and Start run / Reset.
- **Main panel** — a KPI row (state, frames, detections, active tracks,
  dropped frames, throughput, detector/tracker/source), the live annotated
  video (or a clear empty/ready/error state when there's nothing to show
  yet), a structured event log, a run summary after completion, and a
  telemetry panel to inspect/download the JSONL log.

**No true live "Stop" control.** The UI runs a single blocking loop per
run (same execution model as before, just far richer per-frame rendering)
— there is no background thread to interrupt mid-run from a button click.
Use the **Max frames** limiter in the sidebar to bound a run instead.

## Demo mode

`control/synthetic_source.py` fabricates a short in-memory clip — a
plain background with a few small moving shapes on simple bouncing
trajectories — so the detector/tracker/UI/logging chain can be exercised
and demoed with **no video file, camera, GPU, or network access**. Select
it from the UI sidebar ("Demo mode (synthetic)", the default) or pass
`--source demo` (alias: `synthetic`) to the CLI.

**This proves the pipeline works, not that detection/tracking works on
real drone footage.** The synthetic shapes are easy for the default
motion detector to pick up; real aerial footage will behave very
differently. Treat a clean demo-mode run as "the software runs
end-to-end," not as "the detector is validated."

## Config presets

`config/presets.yaml` (loaded and validated by `config/loader.py`) ships
four named presets, selectable from the CLI (`--preset`) or the UI
sidebar:

| Preset | Purpose |
|---|---|
| `default` | The factories' own built-in defaults — safe general baseline. |
| `demo` | Tuned for the synthetic demo source's small, fast-moving shapes. |
| `debug` | Same as `default`, with the debug flag on (extra timing panel in the UI). |
| `fast` | Optimized for throughput: fewer/larger contours, quicker track turnover. |

Presets set the initial detector/tracker backend and thresholds; every
value is still editable afterward in the UI's "Advanced settings"
expander or via CLI flags (`--detector`, `--tracker` override just the
backend). Invalid presets/values fail with a clear message
(`config.loader.ConfigError`) instead of a raw traceback.
`config/example.yaml` remains a plain reference for hand-writing a one-off
config dict in the shape the factories expect.

## Swapping in a real detector/tracker

Both are selected via a config dict passed to
`detector.factory.build_detector()` / `tracker.factory.build_tracker()` —
see `config/presets.yaml`. To add a new backend, implement
`detector.base.BaseDetector` or `tracker.base.BaseTracker` in a new file
and register it in the corresponding factory. Nothing else in the codebase
needs to change.

**Licensing note:** the optional Ultralytics YOLO detector backend
(`detector/ultralytics_detector.py`) is AGPL-3.0 and requires a commercial
license from Ultralytics for closed-source production use — it is not
enabled by default. The optional torchvision detector backend
(`detector/torchvision_detector.py`, `--detector torchvision`) has no such
issue: it uses a pretrained SSDLite MobileNetV3 model distributed by the
torchvision project under BSD-3-Clause, and is the recommended "real
detector" path for demo clips today. See `docs/DECISIONS.md` for the full
reasoning, including entry #6 on the torchvision backend and its
limitations (COCO has no "drone" class — it filters to `airplane`,
`bird`, `kite` as visual proxies).

## Known limitations

See `docs/known-limitations.md` for the full list. Headline items:

- **The default detector (`motion`) is a placeholder motion-blob
  detector, not a drone classifier**, and will false-positive on any
  motion (birds, trees, camera shake). The optional `torchvision`/
  `ultralytics` backends are real trained detectors but are not
  drone-specific either (see the licensing note above).
- **The default tracker has no re-identification** across occlusion —
  a target that's fully occluded longer than `max_age` frames gets a new
  ID when it reappears.
- **Nothing in this repo has been benchmarked on real drone footage
  yet.** Timing/FPS numbers surfaced in the UI and CLI are real
  measurements of *this* run on *this* machine, not general performance
  claims — see `docs/known-limitations.md` for what has and hasn't been
  measured.
- **No true live "Stop" control** in the UI (see "The operator console"
  above) — only a pre-run frame limiter.
- **The webcam option in the UI opens a camera on the server**, not the
  viewer's browser — there is no browser-side camera capture (that would
  need an extra dependency like `streamlit-webrtc`, not currently used).

## Recommended next steps before real field testing

1. **Record/source real drone footage** and run it through both the
   `demo` and `default`/`fast` presets to see how the placeholder motion
   detector actually behaves on real backgrounds (clutter, camera shake,
   lighting) — expect to need per-clip threshold tuning via "Advanced
   settings" or a new preset.
2. **Decide on a real detector path**: either accept the `torchvision`
   backend's COCO proxy classes (`airplane`/`bird`/`kite`) as good enough
   for now, or invest in a drone-specific model (fine-tuning, or
   resolving the Ultralytics AGPL/commercial-license question — see
   `docs/DECISIONS.md` #2 and #6).
3. **Validate tracking stability on real footage** — the IoU tracker has
   no re-identification, so occlusion-heavy footage may need the
   ByteTrack adapter (`tracker/bytetrack_adapter.py`, requires
   `trackers`+`supervision`) instead.
4. **Write a demo runbook** with the exact clips, presets, and expected
   on-screen results for a live demo — reduces operator error under
   pressure and is not yet in this repo.
5. **Re-run the full test suite and a demo-mode smoke run** after any of
   the above changes (`pytest tests/ -v`, `make demo`) before presenting.

## Documents

- `docs/software-mvp-plan.md` — implementation plan for the original pass.
- `docs/architecture.md` — data flow, module boundaries, interfaces.
- `docs/DECISIONS.md` — key technical/licensing decisions and why.
- `docs/known-limitations.md` — what this MVP does not do yet.
- `docs/demo-gap-analysis.md` — the gap analysis this UI/demo pass was scoped from.
