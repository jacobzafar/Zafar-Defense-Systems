# Decisions log

## 1. Repo built standalone, not cloned from GitHub

The sandbox this was built in has no network egress, so
`github.com/jacobzafar/Zafar-Defense-Systems` (at the time,
`Zafar-Defense-Systems-AB` — the repo was later renamed) could not be
reached to inspect existing content or push changes. This entire MVP was
built fresh and is meant to be copied/merged into that repository
manually. If the real repo already contains files with the same names,
review for conflicts before merging.

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

## 12. Demo readiness pass: UI polish and a demo runbook

**The per-frame video re-seek bug was already fixed** in the earlier
UI-rebuild pass (entry #7) — `ui/app.py` uses `FrameResult.frame` directly
and has not reopened/re-seeked a `cv2.VideoCapture` since. Re-verified by
inspection before starting this priority; no code change was needed for
that specific item.

**What was actually still rough, and fixed here:** the live event log
only ever showed raw per-frame counts (`FRAME #42 · 2 det · 2 trk ·
5.3ms`) — exactly the gap flagged as a remaining recommended improvement
in entry #7 and never acted on. Added operator-facing narrative events:
`ACQUIRED`/`LOST` lines when a track ID appears/disappears between
frames, interleaved with the existing per-frame line rather than
replacing it. Also added a `PRESET` badge to the header (previously only
detector/tracker were shown, not which preset was active) and an inline
AGPL-3.0 licensing warning in the sidebar when the `ultralytics` backend
is selected — a direct, operator-facing connection to the licensing work
in entry #8, rather than that information only living in docs. All three
verified via `AppTest` (`tests/test_ui_app.py`): the preset badge renders,
the warning appears only for `ultralytics` (not `motion`), and
`ACQUIRED` events appear in the event log for a short demo-mode run (a
150-frame run was checked manually too — the events are there, just
scrolled out of the log's last-40-lines window by frame 150, which is
expected, not a bug).

**`docs/demo-runbook.md`** is new: a concrete pre-demo checklist, the
demo script itself, and — the part most runbooks skip — the specific
failure modes this app is actually known to have (no live Stop control,
backend/dependency mismatches, the webcam option being server-side not
browser-side, Streamlit's upload size limit) written from what this repo
has actually verified about its own behavior, not generic advice.

## 13. Housekeeping: repo rename cleanup, final install/test verification

The GitHub repository was renamed from `Zafar-Defense-Systems-AB` to
`Zafar-Defense-Systems` (confirmed via `gh repo view`, not assumed — the
local git remote still points at the old URL, which continues to work
because GitHub redirects renamed-repo URLs automatically; this pass did
not touch git remote configuration, only documentation text). Updated the
stale `-AB` references in `docs/README.md` (title and clone
instructions) and left the historical mentions in `docs/DECISIONS.md`
entry #1 and `docs/software-mvp-plan.md` intact but annotated, since they
describe what was true at the time rather than the current name.

Final verification for this whole backlog pass, from a completely fresh
`.venv`: `pip install -r requirements.txt` alone (no torch) installs in
~35s with no ML framework in the dependency tree; `detector.factory.
build_detector()`'s default remains `MotionDetector`; the full test suite
passes both without the optional `torch`/`torchvision` extra (120 passed,
4 skipped) and with it installed (135 passed); `scripts/license_audit.py`
still passes; `make demo` still runs cleanly end-to-end; and `python -m
py_compile` succeeds across all 61 `.py` files in the repo.

## 14. Real bug found fine-tuning on actual DUT Anti-UAV data: NaN weights from an unclipped, too-high learning rate

`detector/train.py` had, until now, only ever been exercised end-to-end
against a 6-image synthetic fixture (entry #9). The first real
fine-tuning run against DUT Anti-UAV's actual train split (5200 real
images, `config/dut_train.yaml`, 2 epochs) exposed a latent bug that
tiny fixture could never have caught: loss diverged to `NaN` partway
through the first epoch, and the saved `weights.pt` came back with 234 of
476 tensors containing `NaN` — a genuinely broken model, not just a bad
score. This was caught before evaluating anything against it (loading
`weights.pt` and checking `torch.isnan(...).any()` per-tensor first,
precisely to avoid running eval against a corrupted model and reporting
whatever falls out).

**Diagnosis, not a guess:** re-ran the training loop directly (bypassing
`train()`, printing per-batch loss) against the real data. Loss exploded
from ~6 to the hundreds within the first ~20 batches — classic gradient
explosion. The freshly-initialized single-class classification head
(everything else is COCO-pretrained) produces large early gradients that
a 1-epoch, 6-image, 3-batch smoke test never runs long enough to expose.
Confirmed the fix empirically before applying it: gradient clipping
(`max_norm=10.0`) alone kept the loss finite over 300 real batches but
still oscillating (4 to 35, not clearly converging); adding it *together
with* lowering `learning_rate` from this script's old default of 0.005 to
0.001 produced a stable, non-exploding curve over the same 300 batches.

**Fix, applied as durable code, not a one-off config workaround:**
`TrainConfig` gained `grad_clip_max_norm: float = 10.0`, applied via
`torch.nn.utils.clip_grad_norm_()` every step in `train()`'s loop — a
standard, low-risk stabilization for any future real fine-tuning run, not
specific to this dataset. `learning_rate`'s default also moved from 0.005
to 0.001, since 0.005 is now demonstrated to reliably diverge on real
data and was never validated for anything beyond the tiny synthetic
smoke test. `tests/test_train.py`'s
`test_grad_clipping_prevents_nan_loss_that_an_unclipped_run_hits`
reproduces the same failure mode (too-high learning rate) on the fast
synthetic fixture and proves clipping actually prevents it — a real
regression test, not just a config field that exists unused.

**Result:** the re-run (same config, `grad_clip_max_norm=10.0`,
`learning_rate=0.001`) produced a finite, monotonically-decreasing loss
curve (5.18 → 4.49 across 2 epochs) and a `weights.pt` with zero
NaN/Inf tensors, verified directly. See `eval/REPORT.md` for what this
model actually detects — training loss decreasing is not itself an
accuracy claim.

## 15. First real dataset end-to-end: license provenance, ingestion, fine-tuning, evaluation

Summary of this whole pass (entries #14 above covers one bug found along
the way in detail). Every prior mention of "drone" detection accuracy in
this repo was either a COCO-proxy class or a synthetic fixture; this pass
is the first time a real public dataset went all the way through
license-provenance tracking, conversion, fine-tuning, and the frozen
evaluation harness.

**License provenance came first, deliberately.** `DatasetManifest` gained
`license_id` (verbatim source text, or `"UNVERIFIED"` — never guessed)
and `commercial_ok` (`True`/`False`/`None`=unknown), plus a
`detector/train.py --commercial-only` flag that refuses any dataset
without an explicit `commercial_ok=True`. DUT Anti-UAV's own GitHub repo
carries a real Apache-2.0 `LICENSE` file (verified byte-for-byte from
`raw.githubusercontent.com`), but that governs the repo's own contents
(a README and one image) — the dataset itself is hosted externally with
no license statement attached directly to it, so `license_id`/
`commercial_ok` stay `UNVERIFIED`/unknown pending a maintainer confirming
the dataset's own terms. See `docs/datasets.md`.

**Ingestion (`detector/datasets/dut_anti_uav.py`):** a concrete
Pascal-VOC-XML → unified-schema converter, unit-tested against a
synthetic fixture and then actually run against the real, downloaded
dataset (10,000 images across the official 5200/2600/2200 train/val/test
split — every count cross-checked against the source paper's own text
and table, which agree). Two real data-quality findings surfaced and
handled explicitly rather than silently: 3 genuine background-only images
in `train`, 1 degenerate (zero-area) box in `val`.

**Splits kept as-is, frozen eval enforced technically
(`eval/build_frozen_eval_set.py`):** DUT's own test split became the
frozen eval set (one single-frame "sequence" per image, since the
detection subset has no temporal structure) — no re-splitting, so this
run's AP@0.5 is comparable to the paper's own benchmark protocol. The
`.frozen` guard was proven against this exact real directory, not just a
schema-level unit test: pointing `detector/train.py` at it raises
`EvalSetFrozenError` before any data loads.

**Fine-tuning hit a real bug (entry #14): fixed, not worked around.**

**Evaluation surfaced a second real bug, fixed the same way — verify,
don't assume.** DUT's test split has zero hard-negative images and no
temporal structure, so false-alarm rate and track continuity are
structurally unmeasurable on it. `eval/metrics.py` previously returned
`0.0`/`1.0` for exactly this case — both looked like real, good results.
Fixed to return an explicit `measurable: False` plus a reason;
`eval/report.py` renders that as "not measurable," and collapses the
per-sequence table (which would otherwise print one identical
"not measurable" row per image — 2200 of them) into a single summary
line.

**Actual result, honestly weak, as intended for a first pass:**
AP@0.5 = 0.1895, small-object recall = 0.2524, ~60ms/frame (16.6 FPS) on
CPU — well below the paper's own published baselines (0.40–0.68 mAP) for
fully-trained detectors on this identical split, consistent with 2 epochs
on CPU being nowhere near convergence. See `eval/REPORT.md` for the full
metric card and cited baseline comparison (Zhao et al., IEEE TITS 2022,
arXiv:2205.10851, Table II).

**What remains, honestly:** the dataset's commercial-use status is still
unresolved (blocks `--commercial-only`, not general research use); only
2 epochs were run (the model is very likely undertrained, not
representative of this architecture's ceiling on this data); the
tracking subset (20 real sequences, which would make track continuity
actually measurable) has not been downloaded; and the other three
registry entries (Anti-UAV, Drone-vs-Bird, VisioDECT) remain pure
infrastructure. See `docs/known-limitations.md`.

## 16. Real bug found in the UI's video pane: demo frames rendered as an apparent black screen

The operator console's video pane appeared to show a black screen with
only the detection-box overlay visible, no video underneath — reported
as top priority, since a demo that visibly shows no video undermines the
whole console. Diagnosed with real repro scripts before changing
anything, not by guesswork:

- Traced the full frame path end to end (`Pipeline.run()` ->
  `detector.detect()` -> `draw_tracks()` -> `cv2.cvtColor(..., BGR2RGB)`
  -> `st.image(..., channels="RGB")`) for all four detector backends
  (`motion`, `torchvision`, `drone`, and the synthetic demo path),
  checking array identity/dtype/mean brightness at each step. No
  in-place mutation of the frame by any detector, no BGR/RGB channel
  swap, no dtype issue — every array was a correctly-shaped, correctly-
  converted `uint8` frame at every stage.
- Also re-verified entry #7's earlier per-frame re-seek fix is still in
  place and not implicated: `ui/app.py` never opens or seeks a
  `cv2.VideoCapture` itself; it only reads `FrameResult.frame`, and
  `Pipeline._open_source` opens exactly one capture per run.
- Built a real test video from actual DUT Anti-UAV images
  (`cv2.VideoWriter`, since no video file previously existed in this
  repo to test the UI's non-demo path against) and ran it through the
  same path: real footage rendered at its true brightness (frame means
  ~110-170/255 across sampled frames) — the array-level pipeline was
  never the problem.
- The actual root cause was in `control/synthetic_source.py`'s demo
  background color: `_BACKGROUND = (18, 22, 26)` (BGR), chosen (per its
  own prior comment) to "match the dark operator UI theme." Measured
  perceptual brightness (`0.299R + 0.587G + 0.114B`): background ~22.7,
  versus the console's own configured theme in `.streamlit/config.toml`
  — `backgroundColor #0b0f14` (~14.4) and `secondaryBackgroundColor
  #121821` (~23.2). The demo frame's brightness sat inside the app
  chrome's own brightness range, so it was visually indistinguishable
  from empty page background — confirmed by rendering an actual frame to
  PNG and inspecting it directly, not just by comparing numbers. This
  was a real content/contrast defect, not a data-plumbing bug: "Demo
  mode (synthetic)" is the UI's default, zero-config source, so it's the
  first thing anyone sees.
- **Fix:** changed `_BACKGROUND` to `(80, 78, 74)` (BGR, perceptual
  brightness ~77) — a neutral mid-tone slate gray, clearly distinguishable
  from the theme chrome while still a plain, deliberately-unrealistic
  backdrop (per `docs/known-limitations.md`, this source is a plumbing
  check, not a detection-quality one).
- **Regression coverage, at two levels:** `tests/test_synthetic_source.py`
  asserts the demo background's brightness directly against the theme
  colors read from `.streamlit/config.toml` (so a future color change
  that reintroduces near-invisible content fails immediately, without
  a hardcoded magic threshold divorced from the actual theme);
  `tests/test_pipeline_smoke.py` adds an end-to-end version of the same
  check that runs the exact render sequence `ui/app.py` uses. Both tests
  fail against the old color and pass against the fix (checked directly,
  not assumed). Full suite: 170 passed, 0 failed.

## 17. Operator console redesign: presentable for grant reviewers, still zero fabricated data

Follow-up to entry #16, once the video pane actually rendered. Scope was
explicitly visual/informational-hierarchy only — no new metrics invented,
no effector UI, same underlying pipeline data throughout.

**Visual hierarchy, four tiers instead of one flat KPI wall.** The old
single 9-card `kpi-row` mixed live instantaneous state (fps, active
tracks), cumulative run totals (frames, detections), and static config
(detector/tracker name) at identical visual weight. Replaced with:
header badges (config context: preset/detector/tracker/run name — state
removed from here, no longer duplicated) → a new primary "status strip"
(`theme.stat_tile`/`stat_row`) with exactly the four live readouts asked
for — State, Throughput, Latency, Active tracks — in large tabular-numeral
type → a smaller secondary row for cumulative run totals (frames,
detections, unique tracks, dropped, source) → the live-feed/status panels
→ the event log. Always rendered (even idle/ready, as an explicit "—"),
so the layout doesn't jump when a run starts.

**Latency was real but hidden.** `FrameResult.total_ms` was already
computed every frame by `control/pipeline.py` and only ever shown inside
the opt-in debug panel. It's now one of the four primary tiles,
always visible — `_execute_run` tracks `last_latency_ms` the same way it
already tracked `last_fps`, carried into `last_kpi` for the
completed/error views. No new computation, just surfaced.

**"Active tracks" now means one thing in every state.** The previous
single KPI card was live "tracks this instant" while running but silently
became "unique tracks all-run" once completed (same label, different
meaning). Split into two honest, separately-labeled numbers: "Active
tracks" is 0 whenever nothing is running (true — a finished run has
nothing currently active) and "Unique tracks" (secondary row) is the
cumulative distinct-ID count `RunSummary` already tracked.

**One accent color, actually restrained to detections/alerts — audited by
screenshot, not just by writing a CSS comment.** `ui/theme.py` gained a
`--zds-accent` custom property (`#ff9142`); `ui/overlay.py`'s detection
box color now derives from the same value. First pass also put a
decorative accent border on all four primary stat tiles regardless of
content — caught by actually looking at a rendered screenshot (see
below), not by re-reading the CSS, since a color rule is easy to eyeball
as "fine" and hard to notice is semantically wrong. Fixed: tiles get a
neutral border; only "Active tracks" borrows the accent, and only when
`active_tracks > 0` (it's the one primary-strip number that *is* a live
detection count). Green/red stay reserved for unambiguous success/failure
(`ACQUIRED`/`COMPLETED` vs. `DROPPED`/`ERROR`), matching the existing
state-badge palette rather than introducing a second meaning for either.

**Event log: real events kept, routine noise de-emphasized, nothing
deleted.** `ACQUIRED`/`LOST`/`DROPPED` got dedicated badge kinds (green /
accent / red) instead of borrowing unrelated state-badge kinds (`LOST`
previously reused the `running` amber, `ACQUIRED` the `completed` green
by coincidence of color, not by a named relationship). The per-frame
`FRAME` line — real telemetry, but the same numbers already visible in
the status strip — gets a muted/smaller style (`.event-line-frame`) so it
recedes visually instead of being deleted; a one-line legend was added
above the log explaining what each kind means, for an audience seeing
this console for the first time.

**Verified by actually running it, not just reading the diff.** Installed
Playwright + Chromium into the local venv (dev-only, `.venv` is
gitignored, not a project dependency), launched the real Streamlit server,
and drove it end to end: idle → set max-frames → start a demo run →
completed, plus the `missing_dependency` error path. This is what caught
two real bugs neither code review nor the test suite would have:

1. **The header/title was clipped by Streamlit's own fixed toolbar.**
   Measured precisely rather than eyeballed: `header[data-testid=
   "stHeader"]` bottom edge at y=60px, `.zds-title` top at y=44px — a
   16px overlap, painting over the top of the title and header badges.
   Pre-existing (the padding value was untouched from the original UI
   pass), just never visually verified before. Fixed by increasing
   `.block-container`'s `padding-top` from `1.75rem` to `4.5rem`
   (confirmed by re-measuring: 28px clear gap after the fix).
2. **Detection labels near the right frame edge were clipped.**
   `ui/overlay.py`'s label was always anchored at the box's `x1` with no
   bounds check, so a track near the right edge had its confidence value
   drawn past the frame boundary and silently cut off by OpenCV's
   clipping. Fixed by clamping the label's x-position to stay on-canvas
   (`tests/test_overlay.py`, new file, covers this directly — verified to
   fail against the old unclamped math before confirming the fix).

**What was deliberately not touched:** no fake gauges, coordinates, or
capabilities added — every number on screen still traces to a real field
on `FrameResult`/`RunSummary`/session state. No weapon/effector UI of any
kind, consistent with every prior entry in this log. Full suite: 173
passed, 0 failed (170 from entry #16 + 3 new `tests/test_overlay.py`
cases).

## 18. Diagnosing the "98-193 near-identical boxes" symptom: three hypotheses tested against real data, one confirmed

Entry #15/`eval/REPORT.md` left the fine-tuned `drone_v1` detector's
flood of near-identical, low-confidence boxes attributed to
"undertrained, not fundamentally broken" — a reasonable inference from
the loss curve, but an inference, not a direct measurement. This pass
tested three concrete, more-specific hypotheses against the actual model
and its actual predictions before touching anything, per the explicit
instruction to report before changing.

**(a) Is NMS missing or misconfigured?** No. `detector/train.py`'s
`build_single_class_model()` only swaps `SSDLite320_MobileNet_V3_Large`'s
classification head — everything else, including postprocessing, is the
stock model. Verified directly on the loaded instance:
`score_thresh=0.001, nms_thresh=0.55, topk_candidates=300,
detections_per_img=300` — all inherited from torchvision's
`ssdlite320_mobilenet_v3_large()` factory (its own defaults, tuned for
80-class COCO detection, not this codebase's choice). NMS
(`torchvision.models.detection.ssd.SSD.postprocess_detections` →
`box_ops.batched_nms`) runs unconditionally in the model's own eval-mode
forward pass. It is not missing.

But the *symptom* — "near-identical overlapping boxes" — has a precise,
measurable mechanism: sampling 15 real test images and computing the
maximum pairwise IoU among every image's surviving (confidence ≥ 0.35)
boxes gave **0.546-0.550 in every single image** — clustered mechanically
just under the 0.55 NMS cutoff. That is exactly what "near-identical" a
0.55 IoU threshold permits, not evidence NMS silently failed.

Fixed anyway, as a real (if modest) improvement: `nms_thresh` tightened
to **0.45** — the plain `SSD` base class's own tighter default in this
exact torchvision version (not a value picked to fit this eval set) —
now set explicitly in `DroneDetector.__init__` (`detector/drone_detector.py`)
instead of silently inheriting the COCO-tuned factory's looser value, and
exposed through `detector/factory.py`'s config dict
(`tests/test_drone_detector.py` covers both the new default and that it's
configurable). Re-evaluated on the *same, unretrained* `models/dut_v1/`
weights — zero retraining: AP@0.5 moved **0.1895 → 0.1914** (+0.0019,
noise, not the "may move substantially" the hypothesis predicted if NMS
were actually wrong), while small-object recall **dropped** 0.2524 →
0.2300 (214 → 195 GT boxes matched) — tightening NMS occasionally
suppresses a redundant-but-lucky box that happened to be a small GT
box's best match, a real and measured trade-off, not hidden here. Net:
NMS was a real, principled thing to tighten, but not the fix.

**(b) Optimization/classification collapse?** Confirmed, decisively, by
direct measurement rather than inferring it from `training_report.json`'s
two epoch-loss numbers alone. Sampled every raw (pre-confidence-filter)
"drone"-class prediction score across the same 15 real test images:
**2,299 scores, all within [0.4277, 0.4531]** — a band 0.025 wide, 100%
inside a single bin of a ten-bin [0,1] histogram. The classification head
is not discriminating drone-vs-background by spatial location at all; it
emits a near-constant, barely-above-the-0.35-threshold score almost
everywhere. This, not NMS, is the real reason so many boxes cross the
confidence cutoff — nearly every anchor scores similarly, and NMS (working
correctly, per (a)) can only suppress the subset whose IoU with each
other exceeds its threshold.

**(c) Is the learning rate too high?** The premise in this task's
instructions (`config/dut_train.yaml` uses 0.005) does not match the
file: it already uses `learning_rate: 0.001`, the fix already applied in
entry #14 for the original NaN-divergence incident (0.005 was the *old*
default before that fix). Corrected and tested anyway as a real,
controlled comparison rather than skipped: `config/dut_train_lr0005.yaml`
— identical dataset/split/seed(42)/epochs(2), `learning_rate: 0.0005`
only, writing to `models/dut_v1_lr0005/` so the original run and config
are untouched. Result: **worse, not better.** Final training loss 4.500
vs. 4.487 (no meaningful difference — both are just the epoch-2 average
after the same 1,300-batch budget); evaluated with the same
`nms_thresh=0.45`: AP@0.5 **0.1606** (vs. 0.1914), small-object recall
**0.1297**, 110/848 (vs. 0.2300, 195/848); the same narrow-band collapse
persisted, just at a different constant (`[0.4651, 0.4871]`). A lower
learning rate makes *less* progress in the same fixed step budget — it
cannot fix a freshly-initialized classification head that hasn't been
trained long enough, and this run demonstrates that rather than assuming
it.

**Which hypothesis the evidence supports: (b), decisively — this is
undertraining, evidenced directly (a literal narrow-band score
histogram), not inferred from a loss curve, and with two plausible
alternative explanations (NMS misconfiguration, this specific LR change)
actually tested and ruled out rather than assumed away.** Neither (a)'s
fix nor (c)'s comparison meaningfully moves the number; `nms_thresh=0.45`
is kept as the new default on its own principled merits, and
`models/dut_v1_lr0005/` plus its config are kept for the record (a real
negative result), not deleted.

**The real number, current best configuration** (`models/dut_v1/weights.pt`,
lr=0.001, 2 epochs, `nms_thresh=0.45`), from a fresh
`python -m eval.harness --config eval/config/dut_anti_uav.yaml` run
against the same frozen 2200-image DUT Anti-UAV test split as entry #15
(see `eval/REPORT.md` for the full metric card and restored baseline
table):

| Metric | This run | Published baseline range (Zhao et al., Table II) |
|---|---|---|
| AP@0.5 | **0.1914** | 0.400 (YOLOX-ResNet18, fastest) — 0.683 (Cascade-RCNN-ResNet50, best) |
| Small-object recall | **0.2300** (195/848) | not reported by the paper |
| Latency (mean) | **62.03 ms** (16.12 FPS, CPU) | not directly comparable — no GPU in this environment |

Still well below every published baseline — this pass does not claim
otherwise. What changed is *why*: that gap is now attributable to a
specifically-evidenced collapsed classification head after only ~1,300
optimizer steps on a freshly-initialized head, with NMS and this
particular LR change directly tested and ruled out as the explanation,
rather than "probably just needs more training." The next lever remains
what `docs/STATUS.md` §6 already said: train longer — this pass is the
evidence for why that's the right call, not a substitute for doing it.

**Verified, not assumed:** every number above comes from an actual
`eval.harness`/`detector/train.py` run in this pass, cross-checked
against a from-scratch diagnostic script (pairwise-IoU and score-histogram
measurements) independent of the harness itself. Full test suite: run
below, before committing.

## 19. AP@0.5 was computed after a 0.35 confidence cutoff: audit and fix

**Audit (before changing anything).** `compute_ap50` (`eval/metrics.py`)
itself was correct — all-point PR-envelope integration over whatever
predictions it is handed, no cutoff of its own. But it was never handed
all predictions:

- `eval/harness.py` built predictions from `detector.detect(image)`, and
  `DroneDetector.detect` (`detector/drone_detector.py`) drops every score
  below `confidence_threshold` — 0.35, from `eval/config/dut_anti_uav.yaml`
  via `detector/factory.py`. So every AP in `eval/REPORT.md` up to this
  point was integrated over a PR curve truncated at score 0.35.
- `detector/train.py`'s per-epoch `evaluate_on_val` (added in the
  resumable-training commit) copied the same 0.35 cutoff deliberately,
  to match the harness — so the dut_v2 Colab run's per-epoch val AP
  (~0.09-0.10, best 0.128 at epoch 3) carries the same understatement.
- `DroneDetector` also rounded scores to 3 decimals, creating ranking
  ties that `compute_ap50` broke by frame order.
- Not a bug: the model's own `score_thresh=0.001` and 300-detections-per-
  image cap (standard; COCO itself caps at 100 detections per image).

A cutoff can only lower AP: removing the lowest-scored predictions
truncates the PR curve's tail (lost recall) and can only lower the
backward-max precision envelope before it, never raise it. So the old
numbers are lower bounds on the true threshold-independent AP.

**Fix.** The harness now builds the detector with its cutoff disabled
(`confidence_threshold: 0.0`), computes AP over every raw detection, and
applies the configured `confidence_threshold` itself — only to the
operating-point metrics (small-object recall, false-alarm rate) and to
what the tracker sees. `evaluate_on_val` drops its cutoff too, and
`DroneDetector` no longer rounds scores (the UI formats them itself).
Hand-computed regression test: two GT boxes with a TP at 0.9, an FP at
0.8, and a TP at 0.2 give AP 0.8333 over all detections vs. 0.5 cut off
at 0.35 (`tests/test_eval_metrics.py`); plus a harness-level test that a
perfect detector scoring only 0.2 now gets AP 1.0 (was 0.0) while its
operating-point recall at 0.35 stays 0.
