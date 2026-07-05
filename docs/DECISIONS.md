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
