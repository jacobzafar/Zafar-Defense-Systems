# Architecture

## Data flow

```
                 +-------------------+
 video source -->|  cv2.VideoCapture |
 (file/webcam/   +---------+---------+
  RTSP)                    |
                            v BGR frame
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
   +----------+-----------+     +-----------------------+
              v
   +---------------------+
   | ui.app (Streamlit)  |
   | or headless CLI     |
   +---------------------+
```

`control.pipeline.Pipeline.run()` is the single orchestration point: it owns
the video capture loop, calls the detector then the tracker per frame,
handles per-frame exceptions without killing the run, logs every frame, and
yields a `FrameResult` that both the CLI and the Streamlit UI consume the
same way.

## Module boundaries

| Module | Owns | Must not contain |
|---|---|---|
| `detector/` | Frame -> Detection[] | Tracking logic, UI code, logging |
| `tracker/` | Detection[] -> Track[] | Detection model code, UI code |
| `control/` | Orchestration, state, events | Any detector/tracker implementation details, any actuation/effector code |
| `ui/` | Rendering, operator interaction | Business logic (detection/tracking math) |
| `telemetry/` | Structured logging | Detection/tracking logic |
| `scripts/` | CLI entry points | Reusable library logic (delegates to the modules above) |

## Interfaces

- `detector.base.BaseDetector.detect(frame) -> list[Detection]`
- `tracker.base.BaseTracker.update(detections) -> list[Track]`

Both are swappable via `detector/factory.py` and `tracker/factory.py`
respectively, selected by a config dict (see `config/example.yaml`). No
other module imports a concrete detector/tracker class directly — they all
go through the factories, so changing backend is a one-line config change.

## Explicit non-goals (enforced by absence, not by a flag)

There is no effector abstraction, no "engage" or "fire" method, no command
channel to any external actuator anywhere in this codebase. Extending this
system toward any kinetic or jamming capability would require adding new
modules, not flipping a switch in existing ones.
