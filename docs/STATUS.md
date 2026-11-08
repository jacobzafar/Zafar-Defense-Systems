# Project status

**Zafar Defense Systems — Anti-UAV Detection Software MVP**
*Generated from the actual state of `feature/temporal-robustness` (branched from `feature/software-mvp-scaffold`); refreshed 2026-11-08 after `docs/DECISIONS.md` #27. §§2–4 on DUT Anti-UAV are unchanged since #24; the FPV feasibility work (#25–#27) is in §2b. Observation-only: this repository contains no jamming, kinetic, or effector/turret-actuation logic anywhere, and none is planned in this branch.*

This document describes what actually exists and runs in this repository
today — not a roadmap, not a pitch. Every number below is either read
directly from a file in this repo (`eval/REPORT.md`, test output, git
history) or from an actual command run while writing this document. Where
something has not been measured, that is stated explicitly rather than
estimated.

---

## 1. What exists and runs today

| Component | What it does |
|---|---|
| `control/pipeline.py` | Orchestrates detect → track → log for a video source, frame by frame. |
| `control/synthetic_source.py` | Deterministic synthetic video source ("demo mode") — proves the pipeline runs with zero camera/GPU/network. |
| `control/state.py` | A small explicit state machine (IDLE/RUNNING/STOPPED/ERROR) gating pipeline execution. |
| `detector/motion_detector.py` | **Default** backend. OpenCV background-subtraction (MOG2) blob detector — flags motion, not drones specifically. Zero extra dependencies. |
| `detector/torchvision_detector.py` | Pretrained SSDLite MobileNetV3 (COCO), used as a proxy by keeping `airplane`/`bird`/`kite` classes. Real learned features, not a drone classifier. |
| `detector/ultralytics_detector.py` | Optional YOLO adapter. **AGPL-3.0** — opt-in only, never imported unless explicitly selected. |
| `detector/drone_detector.py` | Single-class "drone" SSDLite, loads weights from `detector/train.py` (real fine-tuned weights in `models/dut_v1/`, gitignored — see §2). Optional **tiled inference** (overlapping crops + full frame, merged by NMS; **off by default**) and configurable `input_size`. |
| `detector/train.py` | YAML-configured fine-tuning script for the drone-specific detector. Deterministic seeding, gradient clipping, structured hard-negative handling; resumes after interruption (per-epoch checkpoint + kept per-epoch weights), CUDA auto-detect, per-epoch validation AP. |
| `detector/datasets/` | Unified dataset schema + registry (4 named public datasets) + license-provenance tracking (`license_id`/`commercial_ok`) + a concrete DUT Anti-UAV converter. |
| `detector/ssdlite_stride8.py` | SSDLite with an added stride-8 level and geometric anchors (`architecture: ssdlite_stride8`, #26). Opt-in; the stock layout stays the default. |
| `detector/input_norm.py`, `detector/sensor_sim.py` | Frame normalization (gray / 1-channel / BGRA / 16-bit / float → BGR8) and seeded simulated-sensor transforms used by the eval variants and the prepared `augment: sensor_dg` training recipe (#27). |
| `tracker/iou_tracker.py` | **Default** backend. Dependency-free greedy IoU tracker. No re-identification. Coasts stale boxes for 15 frames (measured as ghost boxes in #27). |
| `tracker/kalman_tracker.py`, `tracker/ego_motion.py` | Opt-in `kalman` backend (#27): camera-motion-compensated Kalman tracker with two-stage association. **Evaluated in the harness only; not wired into the CLI, UI or presets**, and the live detector threshold is unchanged. |
| `tracker/bytetrack_adapter.py` | Optional Apache-2.0 ByteTrack/SORT family wrapper (Roboflow `trackers` package). |
| `eval/` | The **only** place test-set accuracy numbers are produced: a frozen-eval-set harness computing threshold-independent AP@0.5, AP@[.50:.95], AP by object size, recall/precision at the operating threshold, small-object recall, false-alarm rate, latency, track continuity. Since #27 also: clip view of frame-numbered sets, simulated-sensor and shake variants, false positives on drone frames, recall by apparent size, track quality, the tracker's output scored as the system's output. `eval/select_checkpoint.py` ranks training epochs on validation; `eval/track_sweep.py` chooses tracker parameters on val. |
| `tools/preannotate.py` | Runs a detector over video/images, emits COCO-format pre-labels for human review — not a labeling tool itself. |
| `tools/render_detections.py` | Draws predictions vs. ground truth onto real images/video for visual inspection — read-only, inference + drawing only. |
| `telemetry/logger.py` | JSONL structured event logging per pipeline run. |
| `ui/app.py` | Streamlit operator console — live run view, preset/backend selection, event log, demo mode. |
| `scripts/run_pipeline.py` | CLI entry point: `python scripts/run_pipeline.py --source <file\|demo> --detector {motion,torchvision,ultralytics,drone} --tracker {iou,bytetrack}`. |
| `scripts/license_audit.py` | Fails the build if a copyleft (AGPL/GPL) package ends up in the core (non-extra) dependency list. |

Run it: `make install && make demo` (zero-dependency default path), or
`make run-ui` for the Streamlit console. `pip install torch torchvision`
unlocks `--detector torchvision`/`drone`.

---

## 2. What is PROVEN — with real numbers

Source: **`eval/REPORT.md`** (its "Summary" section is the
self-contained read), generated by `eval/harness.py` against the real,
frozen DUT Anti-UAV detection-subset test split (2,200 real images,
2,245 drones; official split from Zhao et al., IEEE TITS 2022,
arXiv:2205.10851), using `detector/drone_detector.py` with weights
fine-tuned by `detector/train.py` for 2 epochs on the same dataset's
5,200-image train split. No setting was chosen on the test set.

| Threshold-independent AP@0.5 | Overall | Large | Medium | Small | CPU speed |
|---|---|---|---|---|---|
| Standard inference | **0.1929** | 0.7294 | 0.0218 | 0.0016 | 59.4 ms/frame, 16.8 FPS |
| 3×3 tiled inference (opt-in, no retraining) | **0.3058** | 0.8049 | 0.2899 | 0.0020 | 356.0 ms/frame, 2.8 FPS |
| Published baselines, same split | 0.400 (YOLOX-ResNet18) – 0.683 (Cascade-RCNN-ResNet50) mAP | — | — | — | GPU |

**Read on this result:** the model is good on large drones, usable on
medium drones only with tiling, and effectively blind to small drones.
The small-drone gap has a measured root cause, not a guess: SSDLite's
smallest anchor is 0.2 of the input side (64px at 320), and **0% of
small drones in the training data have any anchor they overlap at
IoU ≥ 0.5** — at 320 or 640 input, or with 3×3 tiling — so the model
never receives a usable training signal for them (`docs/DECISIONS.md`
#23). That is an architecture floor; more epochs, a different learning
rate, or a larger input do not remove it. **At the 0.35 operating
threshold the model is not yet deployable:** its scores are collapsed
into ~0.43-0.47 (#18), so the threshold filters almost nothing (~141
false positives per frame standard, ~300 tiled).

Real bugs found and fixed on the way to these numbers, rather than worked
around:
- First fine-tuning diverged to `NaN` weights at the old default
  learning rate — fixed with gradient clipping + a lower default (#14).
- False-alarm rate and track continuity were reported as `0.0` / `1.0`
  when actually unmeasurable — now an explicit not-measurable state (#15).
- The "near-identical boxes" symptom: NMS present but loose (tightened),
  a lower learning rate made it worse, score-head collapse confirmed by
  direct measurement (#18).
- AP was computed after the 0.35 confidence cutoff, which can only
  understate it; now computed over every detection. For this model it
  changed nothing (all scores were above 0.35), but it would have
  understated any model whose scores spread out (#19).

---

## 2b. FPV feasibility on VisioDECT (#25–#27) — research only

All of this is **feasibility on public data**: VisioDECT DJIFPV (license
`UNVERIFIED`, every model `commercially_cleared: false`), held-out
blocks of the **same clips** used for training (in-domain, an upper bound
for new footage). The real validation is the project's own data, planned
for January 2027.

- **#26, stride-8 detector, FPV-trained (pass with constraints).** Frozen
  FPV test (720 frames, 662 drones, 58 empty): AP@0.5 **0.8627**, recall
  0.829 / precision 0.888 at the 0.35 operating threshold, 0 false alarms
  on 58 empty frames (a THIN figure). 7.4 FPS on this CPU, below the
  12 FPS real-time floor.
- **#27, temporal tracking — Kalman tracker: keep (as feasibility).**
  Scored on what the tracker shows (IoU 0.5), test: recall **0.882** at
  precision **0.890**, vs. 0.829 / 0.888 for the detector alone and
  0.846 / 0.187 for the current IoU tracker's output. Under synthetic
  shake, track continuity 0.785 vs. 0.330 for the IoU tracker. Cost
  ~5 ms/frame. The 77 → 0 false alarms on empty frames is against the
  **old IoU tracker's output** (ghost boxes); the detector alone was
  already at 0. Parameters were chosen on val; test was run once.
- **Where it does not help:** Sunny-late (small drones against empty
  sky) stays at recall 0.425. Image-based camera-motion estimation fails
  on every frame there — there is nothing in the sky to measure against.
  A deployed PTZ/gimbal mount's own pan/tilt/IMU telemetry is the
  intended replacement. The thermal-proxy variant is a zero result: real
  IR needs real IR data.
- **Measured and dropped or deferred (#27):** letterbox / adaptive input
  (dropped); tiled inference (dropped on val — recovers small drones but
  ~40× the false positives; code option kept); area resize (opt-in, 13%
  faster, accuracy-neutral); `prefer: score` box choice (opt-in, could
  not be validated on this data); motion cue + fusion and the
  `sensor_dg` training recipe (prepared/deferred to January).

---

## 3. What is NOT proven, or not measurable — and why

- **False-alarm rate: not measurable on DUT, THIN on FPV, never against
  real clutter.** DUT Anti-UAV's test split has **zero hard-negative
  (no-drone) images**. The FPV test set has 58 empty frames from two
  clips; no birds, planes or clouds-as-clutter exist in any eval set, so
  the false-alarm cost of temporal tracking against real clutter — and
  the value of track confirmation — is **unmeasured** (#27).
- **Track continuity: measurable on FPV clips only (#27), not on DUT.** The
  detection subset is a bag of independent static images with no
  temporal relationship between them; the frozen eval set wraps each as
  its own single-frame "sequence" (the honest representation of that,
  not an approximation of a real sequence). An ID switch cannot occur
  with one frame per sequence, so this metric is undefined here, not
  perfect. DUT Anti-UAV's separate **tracking subset** (20 real
  multi-frame sequences) would make this measurable — **it has not been
  downloaded or ingested in this branch.**
- **IR/night imagery: not evaluated.** No infrared/thermal data has been
  sourced, converted, or evaluated anywhere in this repo. #27's
  "thermal proxy" (inverted low-res mono) is not IR and scored ~0.
- **FPV-class targets: evaluated in-domain on VisioDECT only (§2b).**
  Not on own footage, not on other cameras: cross-sensor robustness was
  measured only on *simulated* sensor variants of the same frames (#27).
  Small-drone AP@0.5 on DUT remains 0.002 with the stock model (§2).
- **Real-world (non-benchmark) footage: never evaluated.** Every real
  number above comes from a static-image research benchmark, not live
  camera footage, a real deployment environment, or footage resembling
  an actual operating scenario. `docs/coverage-matrix.md` (drone type ×
  lighting/weather × background × range/angle/speed × EO/IR) remains an
  empty template — zero rows logged.
- **`ui/app.py`'s live pipeline has never been run against the
  fine-tuned `drone` backend on real footage** — only against synthetic
  demo frames and, separately, the batch eval harness. Live-pipeline
  behavior (per-frame latency under Streamlit's rerun model, tracker
  interaction) with this specific model is unverified.

---

## 4. Dataset and license status

Full detail: **`docs/datasets.md`**. Summary:

| Dataset | Status | License (`license_id`) | `commercial_ok` |
|---|---|---|---|
| DUT Anti-UAV (detection subset) | **Downloaded, converted, in active use** | `UNVERIFIED` | `unknown` |
| VisioDECT | **Ingested; DJIFPV trained on and evaluated for research/feasibility only (#26, #27)** — see `docs/datasets.md` | `UNVERIFIED` — no license file found anywhere in the supplied archive | `unknown` |
| Anti-UAV | Registry placeholder only, no data | `UNVERIFIED` | `unknown` |
| Drone-vs-Bird | Registry placeholder only, no data | `UNVERIFIED` | `unknown` |

**Flag for commercial use / acquisition due diligence:** DUT Anti-UAV's
own GitHub repository (`wangdongdut/DUT-Anti-UAV`) carries a real,
verified Apache-2.0 `LICENSE` file — but that repository contains only a
README and one example image. The actual dataset (images + annotations
this model was trained and evaluated on) is hosted externally on Google
Drive / Baidu Pan, with **no license statement found attached to the
dataset content itself**. Whether the Apache-2.0 grant is intended to
cover the dataset is not confirmed. Until a maintainer supplies the
authoritative text (the paper's data-availability statement, direct
author correspondence, or an explicit statement on the download page),
this dataset's `commercial_ok` stays `unknown` — and
`detector/train.py --commercial-only` (added this pass specifically for
this purpose) will refuse to train on it. **Practical effect: today's
research/development use is consistent with the dataset's evident
academic purpose, but no model trained on it should be sold, licensed,
or shipped commercially until this is resolved.**

Core runtime dependencies are all permissive (MIT/BSD-3-Clause/
Apache-2.0 — see `docs/licenses.md`); `scripts/license_audit.py` enforces
that no copyleft package can enter the core install path. The only
copyleft dependency in the repo (`ultralytics`, AGPL-3.0) is an opt-in
extra, never installed or imported by default.

---

## 5. Repository facts

- **Tests:** 320 passing, 0 failing (with the optional
  `torch`/`torchvision` extra installed; some are skipped without it).
  `python -m pytest tests/ -q`, run 2026-11-08.
- **Branch:** `feature/temporal-robustness`, 8 commits on top of
  `feature/software-mvp-scaffold` (which is in sync with its origin).
  Not merged: the Kalman tracker is not integrated into the product yet.
- **License audit:** passing (`scripts/license_audit.py`) — no copyleft
  package in the core install path.

---

## 6. Next: own data in January

`docs/DECISIONS.md` #27 holds the full plan. In short: own footage with
hard negatives (birds, clouds, shake) to measure temporal false-alarm
cost and track confirmation; camera-motion compensation from PTZ/gimbal
telemetry, then motion cue + fusion; retraining on own data (`sensor_dg`
augmentation, letterbox, tile/native-scale training for small objects);
cross-sensor tests on real other cameras including real IR; real-time
on GPU/target hardware. Wiring `kalman` into the CLI/UI/presets (and
lowering the live detector threshold to match it) is a deliberate later
step, together with that retraining; until then the live detector stays
the validated baseline.

The section below is the status as of #24 (DUT Anti-UAV), kept for the
record; its lever 1 (own FPV footage) and lever 3 (stride-8) were since
taken up on VisioDECT in #25–#26.

## 6a. Detection work as of #24: a finished chapter, and what reopens it

The detection work is **locked at a deliberate stopping point**
(`docs/DECISIONS.md` #24), not abandoned. It ends with a corrected,
reproducible evaluation; a measured root cause for the remaining gap;
an opt-in no-retraining improvement (tiling, 0.19 → 0.31); and prepared
but unrun GPU experiments (640 vs. 320 input, `config/dut_train_640.yaml`
/ `config/dut_train_320_control.yaml`, marked "PREPARED, NOT RUN").

Next levers, in order — each deferred by choice:

**1. The project's own FPV footage.** This is the data the product
actually has to work on. Before any model change, measure its drone-size
distribution with the same anchor-coverage check (`docs/DECISIONS.md`
#23): if its targets are mostly medium/large in frame, the current
architecture (with tiling where latency allows) may be adequate; if they
are mostly small, lever 3 becomes necessary. Footage with drone-free
frames would also make false-alarm rate measurable for the first time.

**2. Resolve DUT Anti-UAV's commercial license status** (§4). A business/
legal task that blocks any commercial use of a model trained on it,
regardless of accuracy.

**3. Only if lever 1 shows small targets matter: a stride-8 or FPN
architecture.** `docs/DECISIONS.md` #23 option B (stride-8 level + 16px
anchors in SSDLite at 640; simulated to make ~24% of small / ~75% of
medium drones matchable, from 0% / 0.7%; est. 1-2 days + GPU training)
or option D (FPN detector, the family behind the published baselines;
est. 3-5 days, heavier on CPU).
