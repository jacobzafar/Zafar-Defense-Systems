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

## 8. Licensing hygiene audit: keep AGPL out of the default install path

Prompted by an acquisition-due-diligence-style review: confirm nothing
AGPL/GPL is in the default/shipped path, not just believe it.

**Finding:** the architecture already isolated this correctly before this
pass — `detector/ultralytics_detector.py` (AGPL-3.0, entry #2) was already
an optional `pyproject.toml` extra, not a core dependency, and
`detector/factory.py` already only imports it lazily (inside the
`if backend == "ultralytics":` branch), so the package is never imported
unless a caller explicitly selects that backend. No code change was
needed to fix an isolation gap — there wasn't one.

**What this pass added is the verification tooling that was missing:**
`scripts/license_audit.py` parses `pyproject.toml`'s core dependencies and
every optional extra against a small hand-maintained license registry,
and fails (exit 1) if a copyleft (AGPL/GPL-family) package is found in the
*core* dependency list; copyleft in an opt-in extra is reported but does
not fail the audit, since those are never installed or imported by
default. `docs/licenses.md` is the human-readable counterpart — one table
for core dependencies (all permissive: MIT/BSD-3-Clause/Apache-2.0), one
for extras (including the AGPL-3.0 `ultralytics` extra and why it's
acceptable there specifically).

**Limitation:** the audit's license registry is hand-maintained, not
pulled from PyPI metadata or an SBOM tool — it only knows about packages
already listed in it, and reports anything else as `UNKNOWN` rather than
silently assuming it's safe. Update the registry (and `docs/licenses.md`)
whenever a dependency is added or changed;
`tests/test_license_audit.py` covers the parsing/flagging logic itself,
not whether the registry is currently exhaustive or accurate.

## 9. Drone-specific detector: infrastructure only, no data or weights included

Adds a real path toward a drone-specific (not COCO-proxy) detector, kept
strictly to plumbing — no dataset was downloaded, no license was
asserted without a caveat, and no weights were invented.

**`detector/datasets/`:** `manifest.py` defines one schema
(`DatasetManifest`) and a registry entry for each of the four named
public sets (Anti-UAV, DUT Anti-UAV, Drone-vs-Bird, VisioDECT). Every
entry's `source_url` and `local_path` are `None` with a `TODO` — no URL
was guessed, consistent with this assistant's standing instruction to
never generate/guess URLs it isn't confident about. Every entry's
`license` field is `LicenseStatus.UNVERIFIED` with a `license_notes` field
describing *typical* terms for that kind of dataset (usually
research/academic-use, often gated behind a request/registration step) —
this is deliberately not an asserted SPDX license, because it hasn't been
confirmed against the current official source. `loader.py` implements one
small, dependency-free loader for a unified COCO-inspired
`images/`+`annotations.json` layout that every source dataset is expected
to be converted into by a human — this repo does not attempt per-dataset
format parsing for formats that haven't been verified. `download.py` is a
stub: it describes what a maintainer needs to do (visit the source, review
current terms, convert to the unified schema, set `local_path`) and raises
a clear `DatasetNotConfiguredError` rather than fetching anything.

**`detector/train.py`:** YAML-configurable fine-tuning of the same
SSDLite320 MobileNetV3 backbone `detector/torchvision_detector.py` already
uses, with its classification head replaced for a single "drone"
foreground class. Structured hard-negative handling: images whose boxes
(after filtering to `target_class`) are empty — because they had none, or
only had a different labeled category like "bird"/"clutter" — are kept as
zero-object training targets, the standard mechanism for teaching a
detector to suppress false positives on a known confusable class, rather
than silently dropping those images or inventing a second output class
for them. Deterministic seeding (`random`/`numpy`/`torch`) for
reproducibility. Writes `weights.pt` + a `training_report.json` containing
only measurements from that actual run (loss curve, dataset counts,
config, seed, versions) — it deliberately does not compute or claim any
accuracy metric (mAP, recall, etc.); that is exclusively the frozen
evaluation harness's job (Priority 4 / `eval/`), so training-time numbers
can never be mistaken for a validated benchmark result.

**Real bug found and fixed while proving this actually runs:** the first
version of `train.py` crashed on a batch of size 1
(`ValueError: Expected more than 1 value per channel when training`) —
`nn.BatchNorm2d` requires more than one sample per channel in training
mode, and a dataset size not evenly divisible by the batch size produces
exactly such a batch. Fixed by freezing BatchNorm layers to eval mode
during fine-tuning (`_freeze_batchnorm`), which is also standard practice
for fine-tuning a pretrained detector on a small dataset — recomputing
batch statistics from a handful of images is unstable regardless of the
crash. Verified end-to-end in this environment: built a tiny synthetic
dataset (6 images, half labeled "drone", half labeled "bird" as a hard
negative), ran `train()` for real, loaded the resulting `weights.pt` via
the new `detector/drone_detector.DroneDetector`, and confirmed `detect()`
runs and returns well-formed `Detection` objects. This proves the
plumbing, not detection accuracy — the model in that smoke test saw six
tiny synthetic images for one epoch.

**`detector/drone_detector.py`:** new `BaseDetector` backend, registered
in `detector/factory.py` as `backend: "drone"`, added to
`config/loader.py`'s `VALID_DETECTOR_BACKENDS`, `scripts/run_pipeline.py`'s
`--detector` choices, and the UI's detector dropdown — **default backend
stays `motion`**, this is purely an additional opt-in option. If no
weights file exists at the configured path, it raises
`DroneWeightsNotFoundError` (a `FileNotFoundError` subclass) with a
message naming the exact path and pointing at `detector/train.py`, rather
than silently falling back to anything. The CLI and UI both catch this
specifically (`error_kind: "missing_weights"` in the UI) and show that
message directly instead of a raw traceback.

**What remains, honestly:** no dataset has been downloaded or converted,
no drone-specific weights exist anywhere in this repo, and nothing has
been validated against real drone footage. This entry is infrastructure
that becomes useful the moment a maintainer supplies real, license-checked
data — it does not itself claim to detect drones any better than the
existing placeholders.

## 10. Frozen evaluation harness: the only place accuracy numbers come from

Adds `eval/` — a harness that runs a chosen detector (and tracker, for
continuity) over a held-out eval set and produces a metric card: AP@0.5,
small-object recall, false-alarm rate, latency, and track continuity. See
`eval/README.md` for the full policy; this entry records the design
decisions and what was actually verified.

**A separate schema from `detector/datasets/`, on purpose.** That
loader (Priority 3) is single-image-oriented, right for classification-
style fine-tuning. Track continuity needs ordered *sequences* of frames
with a stable ground-truth identity across time — closer to how the named
public anti-UAV benchmarks are actually structured (short tracking
sequences, not an unordered image bag). `eval/schema.py` defines its own
`EvalSequence`/`EvalFrame` schema for this reason, not to duplicate
`detector/datasets/loader.py` gratuitously.

**"Frozen" is enforced technically, not just documented.** A `.frozen`
marker file in an eval set directory is checked by
`eval.schema.assert_not_for_training()`, which `detector/train.py` now
calls before loading any dataset — training refuses to start with a clear
`EvalSetFrozenError` if the target directory is a frozen eval set. A
manifest checksum (sha256 of `sequences.json`) can additionally be
recorded in an eval config's `expected_checksum` and is verified on every
harness run, catching accidental edits to a supposedly-frozen set. Neither
guard hashes image pixel data — a swapped image file with the same name
would not be caught; this is a lightweight integrity check, not a
cryptographic one.

**Metric definitions were fixed explicitly, not left implicit,** because
"mAP" and "small object" mean different things in different papers:
AP@0.5 uses standard all-point (COCO/VOC2010+-style) precision-envelope
interpolation for a single class (reported as "AP@0.5," not "mAP" — mAP
implies averaging across classes, which doesn't apply to a single-class
"drone" problem). Small-object recall uses COCO's own small-object
convention (area < 32²=1024px², overridable). False-alarm rate is
reported as false detections per hard-negative frame (not bounded to
[0,1] — a frame can have more than one false alarm). Track continuity is
`1 - (ID switches / frames where GT was continuously present)`; a gap
where the ground-truth target leaves and later returns is not itself
counted as a switch. All of this is written into `eval/metrics.py`'s
module docstring so it stays the single definition source.

**Verified, including by hand.** The AP@0.5 implementation was checked
against a hand-computed 2-GT/3-prediction example (expected 0.8333...,
`tests/test_eval_metrics.py::test_ap50_matches_hand_computed_example`)
before anything else was built on top of it, plus perfect-detector (AP=1)
and no-ground-truth (AP=0) edge cases. The full harness was run
end-to-end against a synthetic 8-frame eval sequence (6 "drone" frames +
2 hard-negative "bird" frames) using the zero-dependency `motion`
detector — chosen deliberately so this harness's own tests need no
torch/GPU — producing real (if unremarkable, given 8 synthetic frames)
numbers, a written JSON metric card, and a rendered `REPORT.md` with a
working "Baseline comparison" section. The `.frozen`-marker training
guard and the checksum-mismatch guard were both exercised directly
(attempting to train on the frozen synthetic eval set raises
`EvalSetFrozenError`; a deliberately wrong checksum raises
`EvalSetChecksumMismatchError`).

**What remains, honestly:** no real eval set exists in this repo. The
synthetic fixture proves the harness's mechanics, not any backend's real
accuracy — see `eval/README.md`'s "What this harness does NOT do" section
and `docs/known-limitations.md`. `eval/output/` (the harness's generated
JSON/report) is gitignored specifically so a stale or synthetic-smoke-test
result can never be mistaken for a current, validated one.

## 11. Auto-annotation tool and a coverage-tracking template

Adds `tools/preannotate.py` and `docs/coverage-matrix.md`. Neither labels
any data — both are infrastructure for a human-in-the-loop workflow.

**`tools/preannotate.py`** runs whichever detector backend is selected
(via the same `config/presets.yaml` + `detector.factory` path everything
else uses — no separate detector-selection logic was introduced) over a
video file or a directory of images, and writes its predictions in
standard COCO instances format (`images`/`annotations`/`categories`),
with each annotation carrying a `score` field so a reviewer can see the
detector's confidence. `categories` are built dynamically from whatever
`class_name` values the chosen backend actually returns (`"drone"`,
`"moving_object"`, `"bird"`, etc.) rather than hardcoding "drone" — this
tool is meant to work with any backend, including the placeholder motion
detector, not just a future drone-specific one. The output JSON's `info`
field states explicitly that these are unverified predictions, not
ground truth. Verified end-to-end against a synthetic video (video mode)
and a directory of extracted frames (image-directory mode), including
`--frame-stride` and `--max-frames` handling, using the zero-dependency
`motion` detector — no torch required for this tool's own tests.

**`docs/coverage-matrix.md`** is a template for tracking what real
footage has actually been collected across drone type (including FPV),
lighting/weather, background, range, angle, speed, and sensor modality
(EO/IR). It ships with every checkbox unchecked and an empty scenario
log — no coverage is claimed. The design favors a single append-only
"scenario log" table (one row per real, sourced clip) over a giant
sparse cross-product matrix across all dimensions, which would be
impractical to fill in and easy to leave stale.

**What remains:** no footage has been run through `tools/preannotate.py`
in anger yet (only the synthetic smoke test), and the coverage matrix has
no real entries. Both become useful the moment real footage exists —
this entry is about the tooling, not about any coverage already achieved.
