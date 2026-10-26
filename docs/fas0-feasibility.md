# Fas 0 feasibility: can we detect small, fast FPV drones with what we have?

**Question.** Our positioning is detecting small, fast FPV drones from the
ground. Is that reachable with the current detector architecture
(SSDLite / MobileNetV3, `models/dut_v1`), measured on real
ground-perspective FPV footage we already have?

**Short answer: no.** On 3,439 real FPV drones:

- **The deployed, real-time detector (untiled) finds none of them above
  noise: AP@0.5 = 0.002.** Not one of the drones fits its anchors (0.0%),
  so it cannot learn them however long it trains.
- **With 3×3 tiling (6× slower, 2.9 FPS) it finds drones that cover at
  least ~5% of the frame**, with AP 0.93 at that size. Only 21% of the
  FPV drones in this footage are that large, so overall AP@0.5 is 0.22.
  On a typical wide camera that is a drone within about 3–4 m.
- Both modes raise 180–300 false alarms on every empty frame.

The stride-8 change recommended in DECISIONS #23 (option B) is required.
Even with it, range beyond a few tens of metres depends on the lens as
much as on the model.

Everything here is measurement on existing weights; nothing was trained.
Verdict at the bottom.

## What was measured

**Data.** VisioDECT's `DJIFPV` class, the only FPV airframe in any
dataset we hold: 3,600 frames, 3,439 labelled drones, filmed from the
ground with a hand-held phone, mostly against open sky (sometimes trees,
floodlight masts or a building at the edge). It was built as its own
subset in the repo's unified schema (commands at the end).

VisioDECT gives each frame a scenario label (Sunny / Cloudy / Evening)
and **nothing else**: no distance, altitude, speed or camera metadata in
our copy. The scenarios differ more in camera set-up than in light:

| Scenario | Frames | Drones | Frame size | Note |
|---|---|---|---|---|
| Cloudy | 1,200 | 1,112 | 2400×1080 landscape | 88 frames unlabelled |
| Evening | 1,200 | 1,127 | 2400×1080 landscape | Frames look like full daylight, not dusk. 73 unlabelled |
| Sunny | 1,200 | 1,200 | 1,081 portrait 720×1280, 119 at 1920×1080 | Portrait framing makes drones cover more of the frame width |

The 161 unlabelled frames looked empty in a spot check of six. They are
used as hard negatives, which makes false alarms per empty frame
measurable here for the first time. The dataset does not say whether
they were checked or simply not labelled.

**Three measurements.**

1. **Apparent size** of every drone.
2. **Anchor coverage.** The same check as DECISIONS #23
   (`eval/anchor_coverage.py`). A drone is *matchable* if any of the
   model's anchors overlaps it at IoU ≥ 0.5, the threshold SSD training
   uses. An unmatchable drone never gets a usable training target. This
   is a limit on what a model *can learn*, not on what it outputs.
   Layouts measured:
   - stock anchors at 320 and 640 input;
   - 3×3 tiling (simulated, as dut_v1's opt-in mode, #21);
   - #23's option B (simulated with torchvision's own anchor generator;
     it reproduces #23's DUT figures exactly).
3. **Actual detection.** dut_v1 (trained on DUT Anti-UAV, never shown
   VisioDECT) through the frozen eval harness, untiled and 3×3 tiled. The
   harness's raw predictions were also broken down by apparent size
   (`eval/breakdown.py`).

**"Apparent size"** is sqrt(box width / frame width × box height / frame
height): the linear share of the frame the drone covers. The model
resizes each axis of the frame to its square input independently, so
this, not pixel count, decides what the model sees. **Filming at higher
resolution does not change it; only distance and lens do.**

## 1. How big FPV drones look in this footage

| | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|
| Width, px | 28 | 39 | 55 | 67 | 80 |
| Height, px | 23 | 33 | 41 | 49 | 60 |
| **Apparent size, all** | 1.6% | 2.4% | **3.4%** | 4.6% | 6.3% |
| Cloudy | 1.6% | 1.9% | 2.2% | 2.8% | 3.8% |
| Evening | 2.3% | 2.9% | 3.5% | 3.8% | 5.3% |
| Sunny | 1.5% | 3.6% | 5.1% | 6.0% | 6.8% |

By COCO's pixel-area buckets this footage looks "medium" (83% of boxes
are 32²–96² px). The frames are large, though (1080–2400 px), so
relative to the frame **every FPV drone here is small: 95% cover under
6.3% of the frame.**

**Implied distance (estimate, not measured).** No camera data was
supplied. A DJI FPV is about 0.3 m across. A typical phone main-camera
field of view (~70° horizontal) puts the median Cloudy drone at about
13 m, the smallest 5% at about 20 m, and the median Evening drone at
about 8 m. Treat this footage as **close range, roughly 5–20 m**. Real
engagement ranges (50–500 m) are not in this data.

## 2. Can the current anchors match them? (#23's measurement)

Share of drones with an anchor at IoU ≥ 0.5 (median best IoU in brackets):

| Scenario | Drones | 320 (dut_v1) | 640 | 320, 3×3 tiles + full frame | 640, 3×3 tiles + full frame | #23 option B (simulated) |
|---|---|---|---|---|---|---|
| **All** | 3,439 | **0.0%** (0.028) | **0.0%** (0.028) | 14.9% (0.188) | 15.1% (0.188) | 42.5% (0.482) |
| Cloudy | 1,112 | 0.0% (0.013) | 0.0% (0.013) | 0.1% (0.085) | 0.1% (0.085) | 59.3% (0.527) |
| Evening | 1,127 | 0.0% (0.031) | 0.0% (0.031) | 2.9% (0.208) | 3.5% (0.208) | 50.7% (0.501) |
| Sunny | 1,200 | 0.0% (0.066) | 0.0% (0.066) | 39.9% (0.425) | 40.1% (0.425) | 19.2% (0.454) |

**Size thresholds.** The sweep places synthetic boxes at random
positions in a 1920×1080 frame, so the result does not depend on this
dataset:

| Apparent size | Box at 1080p | 320 / 640 | 3×3 tiled | Option B (#23 as written) | Option B, geometric scales |
|---|---|---|---|---|---|
| 1.5% | 25×19 | – | – | – | 0% |
| 2% | 33×25 | 0% | 0% | 70% | 71% |
| 3% | 50×37 | 0% | 0% | 92% | 100% |
| 5% | 83×62 | 0% | 0% | 100% | 90% |
| 7.5% | 125×94 | 0% | 98% | 100% | 74% |
| 10% | 166×125 | 0% | 88% | **0%** | 84% |
| 12.5% | 208×156 | 0% | 78% | **0%** | 69% |
| 15% and up | ≥249×187 | 100% | 100% | 100% | 52–67% |

- **Untiled, the current model can only learn drones covering at least
  ~15% of the frame** (about 250×190 px at 1080p). 640 input changes
  nothing, because anchors scale with the input, as #22 and #23
  predicted. No FPV drone in this footage is that large.
- **3×3 tiling** lowers that to ~7.5%, with gaps between 7.5% and 15%.
- **#23 option B as written has a scale gap.** Its 7 anchor levels are
  spaced linearly from 0.025 to 0.95, which leaves ~4–14% uncovered. That
  is why it scores only 19% on Sunny. With the same levels but
  **geometrically spaced scales** (0.025, 0.042, 0.071, 0.119, 0.2,
  0.336, 0.565, 0.95):
  - FPV coverage rises from 42.5% to **74.7%** (Cloudy 68.9%, Evening
    93.9%, Sunny 62.1%);
  - no size from 2% upward falls below 52%.

  This was a quick check, not committed code. Whoever builds option B
  should use geometric spacing.
- **Option B's floor is ~2% apparent size** (about 33×25 px at 1080p).

## 3. Actual detection: dut_v1 on the FPV frames

Frozen harness, dut_v1 weights, 0.35 operating threshold. AP is
threshold-independent. Configs: `eval/config/visiodect_fpv*.yaml`.

| Mode | AP@0.5 | AP@[.5:.95] | Small / Medium / Large AP (COCO px buckets) | Recall @0.35 | TP / FP @0.35 | False alarms per empty frame | Speed (4-core CPU) |
|---|---|---|---|---|---|---|---|
| **Untiled (deployed, real-time)** | **0.0022** | 0.0003 | 0.001 / 0.015 / 0.003 | 0.257 | 885 / 630,021 | 183 | ~12–26 FPS |
| 3×3 tiled + full frame | **0.2237** | 0.0666 | 0.001 / 0.339 / 0.564 | 0.477 | 1,642 / 1,078,358 | 300 (cap) | ~2.9 FPS |
| *For reference: DUT test (#21), untiled / tiled* | *0.193 / 0.306* | | *0.002 / 0.022 / 0.729 and 0.002 / 0.290 / 0.805* | | | | |

Per scenario, AP@0.5:

| Scenario | Untiled | 3×3 tiled | Tiled recall @0.35 |
|---|---|---|---|
| Sunny | 0.0058 | **0.5595** | 0.704 |
| Evening | 0.0004 | 0.0423 | 0.417 |
| Cloudy | 0.0017 | 0.0143 | 0.294 |

**By apparent size** (all scenarios pooled; AP restricted to that size
with COCO's ignore rules):

| Apparent size | Drones | Untiled AP@0.5 | Tiled AP@0.5 | Tiled recall |
|---|---|---|---|---|
| 0–2% | 482 | 0.001 | 0.001 | 0.18 |
| 2–3% | 907 | 0.003 | 0.043 | 0.30 |
| 3–4% | 1,004 | 0.006 | 0.061 | 0.36 |
| 4–5% | 319 | 0.070 | 0.318 | 0.71 |
| **5–7.5%** | 701 | 0.126 | **0.925** | 0.97 |
| 7.5–15% | 26 | ~0.03 | ~0.82 | ~0.92 |

What this says:

- **Untiled, the detector is noise on FPV drones at every size.** It
  scores 0.002 overall, and the best size bin is 0.13. Its 25.7% recall
  is the same box spam seen on DUT (#18: ~175 boxes per frame, all scored
  0.43–0.47), and recall below ~4% apparent size (~18–20%) is chance hits
  from that spam.
- **Tiled, there is a sharp threshold at ~5% apparent size.** Above it,
  drones are found and ranked well (AP 0.93). At 4–5% it is partial
  (0.32). Below 4%, where 70% of these FPV drones sit, it fails (≤0.06).
  The scenario spread is the same threshold seen through the data:
  - Sunny scores 0.56 because 53% of its drones are ≥5%;
  - Cloudy scores 0.014 because 4 of its 1,112 drones are.
- **The trained model reaches below its anchor-matchable floor** (~5%
  tiled, against 7.5% for anchors), because box regression can shrink a
  larger anchor onto a smaller drone. So anchor coverage overstates the
  hard limit somewhat. It does not change the conclusion: the median FPV
  drone (3.4%) is still below the detection floor.
- **Both modes are unusable at the operating point:** 183–300 false
  alarms on every empty frame, precision 0.0015. That is the score
  collapse documented in #18 and a separate problem from size. Per-size
  AP measures whether drones of that size are found and ranked; it does
  not mean the detector is quiet.

## 4. What this means in metres

For a 16:9 frame, the distance at which a drone covers a given apparent
size is roughly `span / (2·tan(FOV/2) · 0.87 · size)`. Below, for a
5-inch FPV quad (~0.25 m across):

| Detector | Size floor | Basis | 80° lens (typical wide camera) | 20° lens (~5× zoom) |
|---|---|---|---|---|
| Current, untiled (real-time) | ~12% (est.) | Not reached in this data; tiled floor × ~2.6 tile magnification | **~1.5 m** | ~7 m |
| Current, 3×3 tiled (2.9 FPS) | **~5%** | **Measured** (AP 0.93 above, ≤0.06 below 4%) | ~3.5 m | ~16 m |
| Option B, geometric scales (not built) | ~2% | Simulated anchors only: an upper bound | ~9 m | ~41 m |
| Option B + 3×3 tiling (not built) | ~0.8% (est.) | Assumes tiling's ~2.6× gain carries over | ~20 m | ~100 m |

These distances are estimates from assumed optics. The option-B rows are
anchor floors (what a retrained model *could* learn), not achieved
detection; a 2% drone is 25–35 px of mostly sky.

## 5. Answer

**At what apparent sizes / distances are FPV drones detectable with what
we have?**
- Real-time (untiled): at no size present in this footage. By estimate,
  only at ≥ ~12% of the frame, i.e. within about 1.5 m on a wide lens or
  7 m on a 20° lens.
- With 3×3 tiling at 2.9 FPS: at ≥ ~5% of the frame (measured), i.e.
  about 3.5 m on a wide lens or 16 m on a 20° lens.

**How much of the real FPV footage falls in that range?**
- Untiled: effectively none. 26 of 3,439 drones (0.8%) are ≥7.5%, and
  even those are not ranked above noise.
- Tiled: 21% (727 of 3,439). That is 53% of Sunny's portrait frames,
  7.4% of Evening and 0.4% of Cloudy, even though this footage is itself
  close range (~5–20 m, estimated).

**Does the positioning hold?**
- **Not as-is.**
- **Not "at close range" either.** The range where the current model
  works is 1.5 m real-time, or 3.5 m at 2.9 FPS, on a normal lens. That
  is not an operational range, and the false-alarm rate makes it
  unusable even there.
- **The stride-8 architecture change is required.** Use #23 option B,
  but with geometric anchor scales. Fixing the score collapse (#18) is
  also required.
- **Even then, range is bound by the optics.** Option B's floor (~2% of
  the frame) puts a 0.25 m quad at roughly 9 m on a wide lens and 40 m on
  a 20° lens. Reaching 100 m or more needs both the stride-8 model and a
  narrow field of view (zoom, tiling, or a cued narrow sensor). No
  footage we hold shows an FPV drone beyond ~20 m, so that remains
  untested.

## Caveats

- **Narrow data.** One FPV airframe (DJI FPV, ~0.3 m), one phone camera
  set-up, mostly clean sky. Small racing or strike quads, cluttered
  backgrounds and real distances are not represented. Our own footage
  (`docs/data-capture-runbook.md`) is what closes that gap; its
  shoot-day thresholds (~11–17% untiled, 4–7% tiled) agree with these.
- **Speed not tested.** These are still frames. Motion blur at FPV speeds
  will make detection harder, not easier.
- **Distances are estimates** from assumed optics (sections 1 and 4).
- **Trained on DUT only.** dut_v1 never saw VisioDECT. A model trained on
  FPV data would do better within its anchor limits, but not below them.
- **License.** VisioDECT's license is UNVERIFIED (`docs/datasets.md`). It
  is used here for measurement only, never for training.
- **Latency is approximate.** It is only indicative: the full-set
  untiled run shared the CPU with other work, so speeds are quoted from
  the per-scenario runs.

## Reproduce

```
# FPV subset (unified schema; images/ symlinked) and eval sets
python detector/datasets/visiodect.py --raw-dir data/visiodect/raw \
    --output-dir data/visiodect/converted-fpv --models DJIFPV
python -m eval.build_frozen_eval_set --source-dataset-dir data/visiodect/converted-fpv \
    --eval-set-dir data/frozen_eval/visiodect-fpv
#   per scenario: split converted-fpv/annotations.json by its "scenario" key into
#   data/visiodect/converted-fpv-{sunny,cloudy,evening}, then build
#   data/frozen_eval/visiodect-fpv-{sunny,cloudy,evening} the same way

# Sections 1-2: sizes and anchor coverage
python -m eval.anchor_coverage --dataset-dir data/visiodect/converted-fpv
python -m eval.size_coverage --dataset-dir data/visiodect/converted-fpv --group-by scenario --sweep

# Section 3: detection (full set, then per scenario with prediction dumps)
python -m eval.harness --config eval/config/visiodect_fpv.yaml
python -m eval.harness --config eval/config/visiodect_fpv_tiled3x3.yaml
for s in sunny cloudy evening; do
  python -m eval.harness --config eval/config/visiodect_fpv_$s.yaml
  python -m eval.harness --config eval/config/visiodect_fpv_tiled3x3_$s.yaml
done
P=data/eval_predictions
python -m eval.breakdown --dump Sunny=$P/visiodect_fpv_tiled3x3_sunny.jsonl.gz \
    --dump Cloudy=$P/visiodect_fpv_tiled3x3_cloudy.jsonl.gz --dump Evening=$P/visiodect_fpv_tiled3x3_evening.jsonl.gz
```

---

**Verdict: Fas 0 fails.** The current architecture cannot detect FPV
drones at any operational range. The deciding number: **0.0% of 3,439
real FPV drones fit the deployed model's anchors, and its AP@0.5 on them
is 0.002.** Even 6×-slower tiling only reaches drones covering ≥5% of the
frame (~3.5 m on a wide lens), which is 21% of this already close-range
footage. The stride-8 change (#23 option B, geometric scales) is a
prerequisite, not an option.
