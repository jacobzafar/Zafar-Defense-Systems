# Architecture

## Data flow

```
                 +-------------------+
 video source -->| cv2.VideoCapture  |
 (file/webcam/   | or the built-in   |
  RTSP), or      | SyntheticVideo    |
  "demo"         | Source (demo mode)|
                 +---------+---------+
                            |
                            v BGR frame (kept on FrameResult.frame)
                 +-------------------+
                 |  detector.detect  |  -> list[Detection] (normalized 0..1)
                 +---------+---------+
                            |
                            v
                 +-------------------+
                 |  tracker.update   |  -> list[Track] (stable IDs)
                 +---------+---------+
                            |
              +-------------+-------------+
              v                           v
   +---------------------+     +-----------------------+
   |  ui.overlay.draw_    |     | telemetry.EventLogger |
   |  tracks (annotate)   |     | (JSONL append)        |
   +----------+-----------+     +-----------+-----------+
              v                             v
   +---------------------+       +-----------------------+
   | ui.app (Streamlit)  |        | telemetry.summary      |
   | or headless CLI     |        | .summarize_log()       |
   +---------------------+        | (used by both UI + CLI)|
                                   +-----------------------+
```

`control.pipeline.Pipeline.run()` is the single orchestration point: it owns
the video capture loop (`Pipeline._open_source` resolves either a real
`cv2.VideoCapture` or the synthetic demo source), calls the detector then
the tracker per frame, times each stage, handles per-frame exceptions
without killing the run, logs every frame, and yields a `FrameResult`
(including the raw decoded frame, so neither the UI nor the CLI's
`--save-video` path needs to re-open or re-seek the source) that both the
CLI and the Streamlit UI consume the same way.

## Module boundaries

| Module | Owns | Must not contain |
|---|---|---|
| `detector/` | Frame -> Detection[] | Tracking logic, UI code, logging |
| `tracker/` | Detection[] -> Track[] | Detection model code, UI code |
| `control/` | Orchestration, state, events, the synthetic demo source | Any detector/tracker implementation details, any actuation/effector code |
| `ui/` | Rendering, operator interaction, theme/CSS (`ui/theme.py`) | Business logic (detection/tracking math) |
| `telemetry/` | Structured logging + run summarization | Detection/tracking logic |
| `scripts/` | CLI entry points | Reusable library logic (delegates to the modules above) |
| `config/` | Named preset catalog + validation (`config/loader.py`) | Backend construction itself — it builds config dicts, the factories still own what those dicts mean |

## Interfaces

- `detector.base.BaseDetector.detect(frame) -> list[Detection]`
- `tracker.base.BaseTracker.update(detections) -> list[Track]`

Both are swappable via `detector/factory.py` and `tracker/factory.py`
respectively, selected by a config dict (see `config/presets.yaml`,
loaded via `config/loader.py`). No other module imports a concrete
detector/tracker class directly — they all go through the factories, so
changing backend is a one-line config change.

## Explicit non-goals (enforced by absence, not by a flag)

There is no effector abstraction, no "engage" or "fire" method, no command
channel to any external actuator anywhere in this codebase. Extending this
system toward any kinetic or jamming capability would require adding new
modules, not flipping a switch in existing ones.
