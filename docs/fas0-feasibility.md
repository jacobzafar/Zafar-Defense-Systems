# Fas 0 feasibility: can we detect small, fast FPV drones with what we have?

**Question.** Our positioning is detecting small, fast FPV drones from the
ground. Is that reachable with the current detector architecture
(SSDLite / MobileNetV3, `models/dut_v1`), measured on real
ground-perspective FPV footage we already have?

> **Update, DECISIONS #26 (section 6):** the stride-8 model with geometric
> anchors, trained on FPV footage, reaches **AP@0.5 0.863** on 720
> held-out FPV frames, with recall 0.83 at the 0.35 threshold, at 7.4 FPS
> on the same CPU. Verdict: **pass with constraints**. It is not real-time
> on CPU, the result is in-domain only, no false-alarm rate can be stated
> yet, and range is still bound by the optics. Sections 1–5 below are the
> original Fas 0 measurement of the *existing* model.

**Short answer (existing model): no.** On 3,439 real FPV drones:

- **The deployed, real-time detector (untiled) finds none of them above
  noise: AP@0.5 = 0.002.** Not one of the drones fits its anchors (0.0%).
  (Section 6 shows that this does not make them unlearnable: the same
  architecture trained on FPV reaches AP 0.70, but never above the 0.35
  threshold.)
- **With 3×3 tiling (6× slower, 2.9 FPS) it finds drones that cover at
  least ~5% of the frame**, with AP 0.93 at that size. Only 21% of the
  FPV drones in this footage are that large, so overall AP@0.5 is 0.22.
  On a typical wide camera that is a drone within about 3–4 m.
- Both modes raise 180–300 false alarms on every empty frame.

The stride-8 change recommended in DECISIONS #23 (option B) is required.
Even with it, range beyond a few tens of metres depends on the lens as
much as on the model.

Sections 1–5 are measurement on existing weights. Section 6 is the
trained stride-8 experiment. Verdicts at the bottom.

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
   uses. An unmatchable drone gets only a forced match to its single best
   anchor, a poor training target. Section 6 shows this makes such drones
   *hard* to learn rather than impossible.
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

## 6. The stride-8 experiment (DECISIONS #26)

**What was built.** #23 option B with geometric anchor scales
(`detector/ssdlite_stride8.py`, `config/stride8_fpv.yaml`): a stride-8
level added to SSDLite, 640 input, 7 levels, scales `geomspace(0.025,
0.95, 8)`. A control (`config/stock320_fpv_control.yaml`) is the stock
architecture at 320, otherwise trained identically. Both were trained on
a Colab GPU.

**Anchors, verified on the built model before training:** 74.7% of the 3,439
FPV drones are matchable (Sunny 61.9%, Cloudy 68.9%, Evening 93.9%), equal
to the simulation in section 2. No size from 2% to 30% has zero coverage.
Above ~14% coverage is lower than linear spacing (46–78%).

**Split.** DJIFPV is three continuous videos, so a random split leaks
near-duplicate frames. Each scenario is cut into 10 contiguous blocks:
test = blocks 2 and 7, val = block 5, and 30 guard frames are dropped next
to each held-out block (`tools/split_visiodect_fpv.py`). The test set has
720 frames, 662 drones and 58 empty frames, with median apparent size 3.3%
(full set 3.4%). A content-hash guard refuses training on any test image.

**Results.** All rows measured in this repo, on the same 720 frames and
the same 4-core CPU (`eval/config/fpv_test_*.yaml`, `eval.breakdown`).
AP is threshold-independent; recall and false alarms are at 0.35.

| Model | AP@0.5 | AP@[.5:.95] | AP small / medium | Recall @0.35 | Small recall @0.35 | False alarms, 58 empty frames | FPS |
|---|---|---|---|---|---|---|---|
| dut_v1 untiled | 0.0013 | 0.0002 | 0.0004 / 0.0089 | 0.181 | 0.171 | 190 per frame | 15.6 |
| dut_v1 3×3 tiled | 0.1956 | 0.0561 | 0.0015 / 0.2905 | 0.369 | 0.165 | 300 per frame | 3.1 |
| Stock 320, FPV-trained (control) | 0.7033 | 0.2583 | 0.5347 / 0.8205 | 0.000 | 0.000 | 0 (no detections at all) | 15.6 |
| **Stride-8 640, FPV-trained** | **0.8627** | **0.4476** | 0.5685 / 0.9774 | **0.829** | **0.658** | **0 (THIN)** | **7.4** |

No test drone is COCO-large (≥ 96² px). Stride-8's precision at 0.35 is
0.888 (549 true / 69 false positives, all on frames that contain a drone).

Per scenario, AP@0.5 (recall @0.35):

| Scenario (drones) | dut_v1 untiled | dut_v1 tiled | Control | Stride-8 |
|---|---|---|---|---|
| Sunny (240) | 0.0037 | 0.5200 | 0.6696 (0.00) | 0.6541 (0.69) |
| Cloudy (231) | 0.0005 | 0.0008 | 0.6401 (0.00) | 0.9772 (0.84) |
| Evening (191) | 0.0011 | 0.0064 | 0.9198 (0.00) | 1.0000 (0.99) |

By apparent size, AP@0.5 (stride-8 recall @0.35):

| Apparent size | Drones | dut_v1 tiled | Control | Stride-8 |
|---|---|---|---|---|
| 0–2% | 100 | 0.0003 | 0.6612 | 0.9585 (0.83) |
| 2–3% | 149 | 0.0101 | 0.6241 | 0.6595 (0.82) |
| 3–4% | 268 | 0.0503 | 0.7393 | 0.9391 (0.75) |
| 4–5% | 25 | 0.0075 | 0.9104 | 0.9600 (0.96) |
| 5–7.5% | 120 | 1.0000 | 0.9754 | 1.0000 (1.00) |

What this says:

- **The architecture fix plus FPV training gives a detector that works on
  this FPV footage**, where the deployed model gave noise (0.0013 → 0.863).
- **Most of the gain is FPV training.** The control, with 0% of drones
  anchor-matchable, reaches 0.703 of stride-8's 0.863. So section 2's 0%
  meant *hard to learn*, not *unlearnable*: coverage measures difficulty,
  not a hard limit.
- **Stride-8 adds two things: the operating point and a real small-object
  advantage.**
  - *Operating point:* at the deployed 0.35 threshold the control detects
    nothing (all its scores are lower); stride-8 finds 83% of drones.
  - *Small objects:* the two size views disagree. By COCO pixel area the
    AP-small gap is small (0.569 vs 0.535). By apparent size (share of the
    frame, which is what the model's input sees and what sets range), it is
    large: 0.96 vs 0.66 at 0–2% and 0.94 vs 0.74 at 3–4%. The apparent-size
    view is the operationally relevant one. The COCO-small bucket mixes in
    60 Sunny drones that are few pixels on 720×1280 frames but ≥3% of the
    frame, so it mostly reflects the Sunny issue below.
  - *Soft spot:* the 2–3% band is 0.66 overall, almost entirely from Sunny:
    0.98 on Cloudy's 129 drones vs 0.10 on Sunny's 11.
- **Stride-8 beats 3×3 tiling on both axes at once:** AP@0.5 0.86 vs 0.20
  *and* 7.4 vs 3.1 FPS on this CPU. Tiling as a workaround for dut_v1 is
  now obsolete. (Tiling *combined* with stride-8 remains an untested range
  lever, section 4.)
- **Sunny: a likely preprocessing artifact, to investigate.** Stride-8 sits
  just under the control there (0.654 vs 0.670), with recall @0.35 of 0.39
  on 3–4% drones. Every Sunny test frame is 720×1280 portrait, squeezed
  into a square 640 input, which distorts the drone's shape relative to the
  2400×1080 landscape frames the other scenarios use. The likely fix is an
  aspect-preserving resize (letterboxing). That is a preprocessing issue to
  investigate, not evidence of a fundamental weakness. It is untested, and
  nothing has been fixed.
- **Evening's perfect score has a narrow denominator.** AP 1.0000 / recall
  0.99 comes from 240 frames: 191 drones and 49 empty frames from one clip.
  Every drone is 3.0–4.2% of the frame (median 3.7%), inside stride-8's
  best-covered band (93.9% anchor-matchable), with no COCO-small drone. It
  is a narrow, easy subset: never quote the 1.00 without that context.
- **Not real-time on CPU.** 7.4 FPS is 1.6× below the 12 FPS floor; the
  stock 320 model runs at 15.6 FPS on the same machine. Nearly all of the cost is the
  640 input. Deploying it needs a GPU or edge accelerator, or optimization.
- **No false-alarm rate yet.** 0 false alarms on 58 empty frames, all from
  these clips (none from Sunny), is a THIN figure. No rate will be stated
  until real hard negatives (birds, planes, clutter) are collected in Fas 1.
- **Range stays optics-bound.** The floor is ~2% apparent size: a 0.25 m
  quad is matchable to ~9 m on a wide lens and ~40 m at 5× zoom
  (section 4). The architecture fix moves the floor; it does not remove
  the need for narrow optics at real engagement ranges.

## Caveats

- **Narrow data.** One FPV airframe (DJI FPV, ~0.3 m), one phone camera
  set-up, mostly clean sky. Small racing or strike quads, cluttered
  backgrounds and real distances are not represented. Our own footage
  (`docs/data-capture-runbook.md`) is what closes that gap; its
  shoot-day thresholds (~11–17% untiled, 4–7% tiled) agree with these.
- **Speed not tested.** These are still frames. Motion blur at FPV speeds
  will make detection harder, not easier.
- **Distances are estimates** from assumed optics (sections 1 and 4).
- **Trained on DUT only (sections 1–5).** dut_v1 never saw VisioDECT.
  Section 6's models did, on held-out blocks of the *same* clips, so their
  numbers are in-domain upper bounds.
- **License.** VisioDECT's license is UNVERIFIED (`docs/datasets.md`).
  Sections 1–5 use it for measurement only. Section 6 trains on it as a
  research/feasibility experiment: those weights are research-only and
  refused by `--commercial-only`.
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

---

**Verdict after the stride-8 experiment (#26): pass with constraints.**
- **Capability:** on wide optics the stride-8 model trained on FPV data
  recovers a real mid-range capability on this footage: AP@0.5 0.863 and
  recall 0.83 at the operating threshold, on drones a median 3.3% of the
  frame (~10 m on a phone lens). Most of it comes from FPV training; the
  architecture adds a usable operating point and a real small-object
  advantage by apparent size. It also makes 3×3 tiling obsolete, beating it on
  AP and FPS at once.
- **Not yet sellable:**
  - it is not real-time on CPU (7.4 FPS, 1.6× below the 12 FPS floor);
  - it is proven only in-domain on VisioDECT, not on our own data;
  - no false-alarm rate can be stated until real hard negatives are collected;
  - the weights are research-only (VisioDECT license UNVERIFIED);
  - range is still set by the optics: ~9 m wide, ~40 m at 5× zoom for a 0.25 m quad.
- **Optics requirement:** reaching operational ranges still needs a narrow
  field of view (zoom, tiling or a cued narrow sensor) in addition to this model.
- **Next:** temporal / motion-based detection is the next architecture track,
  to be tested on own fixed-camera data in Fas 1 (DECISIONS #26).
