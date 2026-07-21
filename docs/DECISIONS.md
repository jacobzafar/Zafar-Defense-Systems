# Decisions log

## 1. Repo built standalone, not cloned from GitHub

The sandbox this was built in has no network egress, so
`github.com/jacobzafar/Zafar-Defense-Systems-AB` could not be reached to
inspect existing content or push changes. This entire MVP was built fresh
and is meant to be copied/merged into that repository manually. If the
real repo already contains files with the same names, review for conflicts
before merging.

## 2. Default detector is a placeholder motion detector, not a trained model

`detector.motion_detector.MotionDetector` uses OpenCV background
subtraction (MOG2) to flag moving blobs. It is not a drone/aircraft
classifier and will flag any moving object (birds, cars in frame, waving
trees). It exists so the full pipeline is runnable and testable on day one
without a GPU, trained weights, or a licensing decision.

An optional `detector.ultralytics_detector.UltralyticsDetector` adapter is
included but **not enabled by default**, because Ultralytics YOLO ships
under AGPL-3.0 and requires a separate commercial license for closed-source
production use. That is a business decision, not a technical one — do not
enable this backend in a shipped product without that license question
being resolved by whoever owns commercial terms for this project.

Permissively-licensed alternatives (RF-DETR, YOLOX — both Apache-2.0) were
identified in the earlier repo-sprint research and are the recommended
next real detector to wire in via the same `BaseDetector` interface.

## 3. Default tracker is a simple IoU tracker; ByteTrack is optional

`tracker.iou_tracker.IoUTracker` is a small, dependency-free, greedy IoU
tracker. It has no re-identification/appearance model, so a track that is
fully occluded for more than `max_age` frames will get a new ID when it
reappears. This is an accepted MVP limitation.

`tracker.bytetrack_adapter.ByteTrackAdapter` wraps Roboflow's `trackers`
package (Apache-2.0, clean-room re-implementations of ByteTrack/SORT/
OC-SORT/BoT-SORT) rather than the original FoundationVision/ByteTrack repo,
because the latter has an open, unresolved issue about some files retaining
a "Megvii, all rights reserved" header despite the repo's overall MIT
license. Roboflow's package avoids that ambiguity entirely.

## 4. Logging format: JSONL, not a database

Chosen for zero setup cost and easy inspection (`tail -f`, `jq`, or
`pandas.read_json(..., lines=True)`). If log volume or query needs grow
beyond what flat files support, replacing `telemetry.logger.EventLogger`
with a SQLite- or Postgres-backed implementation behind the same
`log_event()` call signature is a contained change.

## 5. UI: Streamlit, not a custom web frontend

Chosen for speed of delivery in an MVP timeframe. It is a real limitation:
Streamlit reruns the whole script on each interaction, which caps how
interactive/responsive the operator view can be. If the UI needs to become
a real always-on operator console, plan to replace `ui/app.py` with a
proper frontend (e.g. a small FastAPI + websocket backend and a JS/React
frontend) — the `Pipeline`/`FrameResult` interfaces underneath don't need
to change for that migration.

## 6. Added a real detector backend: pretrained torchvision SSDLite (COCO)

Follow-up to `docs/demo-gap-analysis.md`, which flagged the placeholder
motion detector as the single biggest credibility risk for a customer
demo. This adds `detector/torchvision_detector.py` /
`TorchvisionDetector`, wired into `detector/factory.py` as
`backend: torchvision`, alongside the existing `motion` and `ultralytics`
backends — no changes to `BaseDetector`, the tracker, control, UI
rendering, or telemetry were needed.

**Why this backend, over the already-present Ultralytics adapter:**
`detector/ultralytics_detector.py` already existed but is AGPL-3.0 and
needs a commercial license for closed-source production use (entry #2
above) — an unresolved business decision, not something to default a demo
onto. `torchvision.models.detection.ssdlite320_mobilenet_v3_large` is
maintained and distributed by the torchvision project itself under
BSD-3-Clause, with pretrained COCO weights hosted directly on
`download.pytorch.org` — there is no separate weight-file license to
review, unlike third-party weight redistributions. It is also small and
fast enough for a CPU-only demo laptop: the pretrained weights are a
~13MB download, and inference measured ~58ms/frame (~17 FPS) on a 640x480
frame on CPU in this environment.

**Dependency handling:** `torch`/`torchvision` are treated exactly like
the existing Ultralytics dependency — optional, guarded imports (see the
`try/except ImportError` in `detector/torchvision_detector.py`), listed as
commented-out install instructions in `requirements.txt` and as a
`torchvision` extra in `pyproject.toml`, not part of the core
`requirements.txt` install. **The default detector backend remains
`motion`** — a demo laptop that only runs `pip install -r
requirements.txt` still starts reliably with zero heavy dependencies;
`torchvision` is opt-in via `--detector torchvision` (CLI) or the
Streamlit backend dropdown, intended to be the backend actually used
against curated demo clips once `torch`/`torchvision` are installed.

**Limitations that remain:** COCO has no "drone" class. The detector is
configured (`DEFAULT_TARGET_CLASSES` in `detector/torchvision_detector.py`)
to keep only `airplane`, `bird`, and `kite` detections as visual proxies
for a small aerial target — a real learned-feature detector, but still
not a drone-specific classifier. It will still miss drones that don't
resemble those classes and may fire on real birds/kites/planes in frame.
It is also a materially heavier dependency than the zero-dependency motion
detector (a multi-hundred-MB `torch` install versus none). Fine-tuning on
real drone imagery, or licensing a drone-specific model, is the next step
if detection accuracy on the actual demo clips proves insufficient — see
`docs/known-limitations.md`.

## 7. UI/demo-readiness pass: operator console, demo mode, presets, and two bug fixes

Follow-up to `docs/demo-gap-analysis.md`. Scope: make the app demo-ready
*before* real drone footage exists, without redesigning the tracker or
the module boundaries.

**Two pre-existing bugs fixed as part of this pass, not new features:**

1. `control/state.py`'s transition table didn't allow `IDLE -> ERROR`. A
   video source that failed to open would crash with a confusing
   `ValueError` about the state machine instead of the intended
   `RuntimeError` about the bad source — this was flagged but deliberately
   left alone in the prior detector-integration pass as out of scope; it's
   in scope now under "robustness/error handling." Fixed by adding
   `IDLE: {RUNNING, ERROR}` and `STOPPED: {..., ERROR}` to the transition
   table (`tests/test_state_machine.py` covers this directly).
2. `ui/app.py` used to re-open a **new** `cv2.VideoCapture` and seek by
   frame index on every single frame just to fetch the raw frame for
   display, on top of the capture already opened inside
   `Pipeline.run()`. Frame-index seeking is unreliable on compressed
   video in OpenCV, and this doubled I/O per frame. Fixed at the source:
   `control.events.FrameResult` now carries the raw decoded frame
   (`FrameResult.frame`, excluded from JSONL serialization), so the UI
   and the CLI's `--save-video` path both use the frame the pipeline
   already decoded — no second capture, no seeking, and it works
   uniformly for files, webcams, and the new synthetic source (none of
   which support arbitrary re-seeking).

**Demo mode (`control/synthetic_source.py`):** a small `SyntheticVideoSource`
implementing just the `isOpened`/`read`/`get`/`release` surface
`Pipeline.run()` needs, so it's a drop-in alternative to
`cv2.VideoCapture` selected by the source string `"demo"`/`"synthetic"` —
no special-casing anywhere else in the pipeline, UI, or CLI. It fabricates
a deterministic (seeded) clip of a few bouncing shapes so the full
detect/track/log/UI chain can be exercised with zero video files, camera
access, GPU, or network calls. This is explicitly a plumbing check, not a
detection-quality check — documented as such in the UI, README, and
`docs/known-limitations.md` so it can't be mistaken for validation on real
footage.

**Config presets (`config/presets.yaml` + `config/loader.py`):** adds
`pyyaml` as a *core* dependency (previously unused — `config/example.yaml`
described a config shape nothing actually loaded). The loader only builds
and validates the plain config dicts the existing
`detector.factory`/`tracker.factory` already accept; it doesn't duplicate
or bypass them, so backend/key validity stays defined in exactly one
place. Four presets (`default`/`demo`/`debug`/`fast`) ship as a starting
point; invalid presets fail with a `ConfigError` naming the problem
instead of a raw exception from deep in the factories.

**UI: still Streamlit, now with a real dark theme + a custom panel layer.**
`.streamlit/config.toml` sets Streamlit's own supported dark base theme
(this is why `.streamlit/` is no longer blanket-`.gitignore`d — only
`.streamlit/secrets.toml` is now, since that's the file that should never
be committed); `ui/theme.py` adds badges/KPI cards/panels on top via CSS
classes this app fully controls, rather than fighting Streamlit's
internal (and unstable across versions) generated class names. Setting
`gatherUsageStats = false` in that same file also resolves a decision
flagged as open in `docs/demo-gap-analysis.md` (whether Streamlit's
anonymous telemetry ping is acceptable for a demo room) by turning it off
by default.

**No true live "Stop" control, by design, not by oversight.** The UI
still runs one blocking loop per invocation (same execution model as
before). Interrupting that loop mid-run from a button click would need a
background thread and a polled cancellation flag — a real architectural
change to the UI's execution model, which was out of scope for this pass
("do not redesign the UI" beyond the requested panel/theme work). The
sidebar's "Max frames" limiter is the practical substitute; this is called
out explicitly in the UI copy, README, and known-limitations so it reads
as an documented constraint, not a missing feature.

**Verified, not just written:** every piece above was exercised directly
in this pass — `pytest` (49 passed with the optional `torch` extra
installed, 1 skipped without it, including new state-machine, synthetic
source, config loader, telemetry summary, and UI tests), the CLI against
both a real file and demo mode with `--save-video`/`--preset`, `make
demo`, `scripts/summarize_log.py`, and the Streamlit app driven
end-to-end (initial load, preset switching, a full run, reset, and both
the "missing optional dependency" and "bad source" error paths) via
Streamlit's own `AppTest` harness — see `tests/test_ui_app.py`.
