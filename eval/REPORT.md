# Evaluation Report

- Evaluated at: 2026-10-10T10:14:06 (corrected-AP re-run; see "AP correction" below)
- Eval set: `data/frozen_eval/dut-anti-uav-test`
- Eval set manifest checksum (sequences.json sha256): `d7d58f3a2e22d3aec8814ae47f38ee145a735f53c4e9cab16c8732a1eaa2244c`
- Detector backend: `drone_v1` (`models/dut_v1/weights.pt`, lr=0.001, 2 epochs; `nms_thresh=0.45`; operating `confidence_threshold=0.35`)
- Tracker backend: `iou`
- Sequences: 2200  |  Frames: 2200  |  GT drone boxes: 2245

**This eval set is frozen and must never be used for training** — see eval/README.md.

## Summary: where the detection work stands (locked 2026-10-10)

This section is the self-contained read for someone who did not write
this code. Everything below it is the supporting detail; the full
reasoning is in `docs/DECISIONS.md` entries #19-#24.

**What was evaluated.** A single-class "drone" detector — SSDLite
(MobileNetV3 backbone, COCO-pretrained), fine-tuned for 2 epochs on DUT
Anti-UAV's 5,200-image train split (`models/dut_v1`) — on that dataset's
frozen 2,200-image test split (2,245 drones). The test set has never been
used to train or to choose any setting; settings were chosen on a
separate 500-image validation subset.

**1. Accuracy (threshold-independent AP@0.5, frozen test set, no retraining):**

| | AP@0.5 | Large (≥96²px) | Medium (32²-96²px) | Small (<32²px) | Speed (4-core CPU) |
|---|---|---|---|---|---|
| Standard inference | 0.19 (0.1929) | 0.73 | 0.02 | 0.002 | 59 ms/frame, 16.8 FPS |
| 3×3 tiled inference (opt-in) | **0.31 (0.3058)** | **0.80** | **0.29** | **0.002** | 356 ms/frame, 2.8 FPS |
| Published baselines, same split (Zhao et al. 2022) | 0.40-0.68 | not reported | not reported | not reported | GPU, not comparable |

AP here is computed over every raw detection, ranked by score — an
earlier version applied the 0.35 confidence cutoff first, which can only
understate AP; that bug was found and fixed (#19). Tiling splits each
frame into 9 overlapping crops plus the whole frame and merges the
results; it needs no retraining (#21). The tiled mode exists in code but
is **off by default**.

**2. Root cause of the small-drone gap — an architecture floor, not a
tuning problem (#21, #23).** SSDLite's smallest anchor box (the template
sizes the model can match an object to) is fixed at 0.2 of the input
side: 64px at the standard 320×320 input. A small drone in a 1920×1080
frame shrinks to roughly 5×9px at that input. Measured on all 5,243 drone
boxes in the *train* split: **0% of small drones (and 0.7% of medium) have
any anchor they overlap at IoU ≥ 0.5** — the threshold the training
matcher uses — so the model never gets a usable training target for
them. This holds at 320 and at 640 input (anchors scale with the input),
and with 3×3 tiling (simulated). A stride-8 feature map with 16px anchors
at 640 input would make ~24% of small and ~75% of medium drones matchable
(simulated with torchvision's own anchor generator). It is consistent
with training longer barely helping — dut_v2 (20 epochs) sat at ~0.09-0.10
validation AP@0.5 for most epochs (best single epoch 0.128) vs. dut_v1's
0.093 on the same images; dut_v2's figures come from the Colab run's log
under the old cutoff, not re-measured here — and with tiling helping
medium drones (which it pushes into anchor range) but not small ones.

**3. Not deployable at the operating point yet.** At the 0.35 confidence
threshold the model's scores are collapsed into a narrow ~0.43-0.47 band
(#18), so the threshold filters almost nothing: ~141 false positives per
frame untiled, ~300 (the per-frame cap) tiled; precision 0.003 / 0.002.
Tiling improves *ranking* — which is what AP measures — but not the
operating point, and it drops the CPU frame rate from ~16.8 to ~2.8 FPS.
False-alarm rate on drone-free frames and track continuity remain
unmeasurable: DUT's test split has no drone-free images and no video
sequences.

**4. What is ready but not run.** Configurable input size and a matched
640 vs. 320 training pair (`config/dut_train_640.yaml`,
`config/dut_train_320_control.yaml`, #22) — prepared, not run, needs a
GPU. Per the anchor analysis, 640 alone is not expected to fix small
drones.

**5. Next levers — both deferred by choice, not abandoned:**
(a) **the project's own FPV footage** — the data the product actually has
to work on; its drone-size distribution should be measured with the same
anchor-coverage check (#23) before choosing a model change, since it
decides whether the small-object floor even matters for it;
(b) **if still needed then, a stride-8 or FPN-based architecture**
(#23 options B and D, est. ~1-2 and ~3-5 days plus GPU training).

## Metric card

| Metric | Value | Definition |
|---|---|---|
| AP@0.5 (single-class "drone") | 0.1929 | Threshold-independent: every raw detection ranked by score (no confidence cutoff), all-point interpolated PR integral at IoU=0.5. |
| AP@[0.50:0.95] | 0.1074 | Mean AP over IoU 0.50, 0.55, …, 0.95 (COCO primary metric; all-point interpolation per threshold). |
| AP@0.5 — small | 0.0016 | 848 GT boxes with 0 ≤ area < 1024 px² (COCO size bucket); threshold-independent. |
| AP@0.5 — medium | 0.0218 | 836 GT boxes with 1024 ≤ area < 9216 px² (COCO size bucket); threshold-independent. |
| AP@0.5 — large | 0.7294 | 561 GT boxes with area ≥ 9216 px² (COCO size bucket); threshold-independent. |
| Recall @ conf ≥ 0.35 | 0.4548 (1021/2245) | Operating point, separate from AP: fraction of GT boxes matched (IoU ≥ 0.5) by detections at or above the configured confidence threshold. |
| Precision @ conf ≥ 0.35 | 0.0033 | 310970 false positives at that same operating point. |
| Small-object recall @ conf ≥ 0.35 | 0.2300 (195/848) | Operating point: recall on GT boxes with area < 1024px² (COCO "small" convention). |
| False-alarm rate | **not measurable** (No hard-negative (zero-GT) frames in this eval set — false-alarm rate is not measurable here.) | 0 false detections across 0 hard-negative (no-drone) frames. |
| Latency (mean) | 90.82 ms | Wall-clock detector.detect() time per frame, this run's hardware only. **Not a clean measurement this pass** — see note below. |
| Latency (p95) | 141.15 ms | |
| Throughput | 11.01 FPS | 1000 / mean latency. |
| Track continuity | **not measurable** (Every sequence in this eval set has at most 1 frame — an ID switch is structurally impossible to observe, so track continuity is not measurable here.) | 1 - (ID switches / frames with GT present); 0 switches across 2200 frames. |
| Source | AP@0.5 | AP@[0.50:0.95] | Small-object recall | False-alarm rate | FPS | Notes |

**Latency note:** this pass ran the dut_v1 and lr=0.0005 evaluations
concurrently on a 4-core CPU, and the harness now materializes every raw
detection (~140/frame) rather than only those above 0.35, so 90.82 ms is
not a clean per-frame latency. The previous clean single-run measurement
(entry #18 pass, same weights and hardware) was **62.03 ms mean / 87.54 ms
p95 / 16.12 FPS**; that remains the latency figure to quote.

## Small-object work (docs/DECISIONS.md #20-#21)

Settings are chosen on the DUT val500 subset; the frozen test set is
reported once per chosen setting. FP counts are at the 0.35 operating
threshold (DUT test has no hard-negative frames, so false-alarm rate per
empty frame is not measurable; FP per frame is the stand-in). Latency is
each run alone on this 4-core CPU.

| dut_v1 on frozen test | AP@0.5 | AP@[.50:.95] | Small | Medium | Large | Recall@0.35 | TP / FP @0.35 | FP / frame | ms/frame (p95) | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| Untiled, 320 (reference) | 0.1929 | 0.1074 | 0.0016 | 0.0218 | 0.7294 | 0.4548 | 1,021 / 310,970 | 141.3 | 59.39 (82.53) | 16.84 |
| Tiled 3×3 + full frame, 320, no retraining | 0.3058 | 0.1758 | 0.0020 | 0.2899 | 0.8049 | 0.6423 | 1,442 / 658,558 | 299.3 | 355.97 (403.99) | 2.81 |

Tiling fixes medium drones (13×), not small ones: SSDLite's smallest
anchor is 0.2 of the input side (64px at 320), and a small drone is still
only ~14×25px inside a 3×3 tile. Kept as an opt-in mode
(`eval/config/dut_anti_uav_tiled3x3.yaml`), not the default, because of
the 6× latency cost and doubled FPs at the operating threshold.

## AP correction (docs/DECISIONS.md entry #19)

**The audit found a real confidence-cutoff bug:** `eval/harness.py`
built its PR curve from `detector.detect()`, which drops every score
below the operating `confidence_threshold` (0.35), and `detector/train.py`'s
per-epoch val AP copied the same cutoff. AP is now computed over every
raw detection (threshold-independent); 0.35 applies only to the
operating-point rows. `DroneDetector` also stopped rounding scores to 3
decimals (rounding created ranking ties).

**For these two models the cutoff changed nothing** — every one of their
raw detections scores above 0.35 (the collapsed band from entry #18), so
nothing was being cut. The small AP change below comes entirely from
removing score rounding, measured by re-applying the cutoff to the same
detections:

| Model (frozen test set) | Old reported AP@0.5 | AP@0.5 cut at 0.35, unrounded | **Corrected AP@0.5** | AP@[0.50:0.95] | Raw detections < 0.35 | Raw score range |
|---|---|---|---|---|---|---|
| `dut_v1` | 0.1914 | 0.1929 | **0.1929** | 0.1074 | 0 of 311,991 | 0.4274 – 0.4694 |
| `dut_v1_lr0005` | 0.1606 | 0.1613 | **0.1613** | 0.0874 | 0 of 230,815 | 0.4649 – 0.5007 |

### AP@0.5 by object size (COCO buckets, threshold-independent)

| Model | Small (< 32² px) | Medium (32²–96² px) | Large (≥ 96² px) |
|---|---|---|---|
| GT boxes in frozen test set | 848 (38%) | 836 (37%) | 561 (25%) |
| `dut_v1` | 0.0016 | 0.0218 | 0.7294 |
| `dut_v1_lr0005` | 0.0002 | 0.0115 | 0.6356 |

Nearly all of the headline AP comes from large drones. Small and medium
drones are effectively undetected in a ranked sense: some are found (small
recall at 0.35 is 195/848), but buried among ~311k false positives
(precision 0.0033 at 0.35), which the size-restricted AP integrates over.

### The dut_v2 val number (~0.09-0.10) in context

`dut_v2`'s weights are **not reachable in this environment** (no
`models/dut_v2/`, no mounted Drive, no `checkpoint.pt` anywhere on disk),
so dut_v2 has **not** been evaluated on the frozen test set here and no
dut_v2 number in this report is measured by this pass.

What can be measured: `dut_v1` on the *same* 500-image seeded DUT val
subset dut_v2's per-epoch evaluation used (`config/dut_train_v2.yaml`:
seed 42, `val_max_images: 500` — assuming the Colab run used that config
and the same converted val split):

| Model (DUT val, 500-image seeded subset) | AP@0.5 | AP@[0.50:0.95] | Small AP@0.5 | Medium | Large |
|---|---|---|---|---|---|
| GT boxes in subset | 503 | | 261 (52%) | 178 (35%) | 64 (13%) |
| `dut_v1` (measured this pass) | 0.0934 | 0.0546 | 0.0023 | 0.0166 | 0.6879 |
| `dut_v2`, per-epoch (Colab run, as reported to me; old cutoff) | ~0.09–0.10, best 0.128 (epoch 3) | — | — | — | — |

The val subset is much more small-object-heavy than the test set (52% vs
38% small, 13% vs 25% large), which on its own takes `dut_v1` from 0.19
(test) to 0.09 (val). The dut_v2 per-epoch values were computed with the
0.35 cutoff, so they are **lower bounds**; with dut_v2's scores spread
across ~0.98 some detections fell below 0.35, so its true val AP is at
least that and possibly higher — by how much cannot be known without the
weights.

### Checkpoint selection

Not possible for this dut_v2 run: `detector/train.py` kept only the
latest `checkpoint.pt` (overwritten each epoch), so per-epoch weights —
including epoch 3's — were never saved, on Drive or anywhere. Only the
final epoch's weights exist. Training now keeps
`weights_epoch_NNN.pt` per epoch, and `eval/select_checkpoint.py` ranks
them by corrected AP; select on val, then report the chosen one on test.

## Previous pass: the box-flood diagnosis (docs/DECISIONS.md entry #18)

_AP figures in this section are from before the #19 correction above (scores were rounded to 3 decimals; the 0.35 cutoff removed nothing for these models). Kept as the historical record._

Three hypotheses for the "98-193 near-identical boxes at confidence ~0.43"
symptom were tested against real data before touching anything:

1. **NMS missing/misconfigured?** No — verified directly on the loaded
   model (`score_thresh=0.001, nms_thresh=0.55, topk_candidates=300,
   detections_per_img=300`, all inherited unchanged from torchvision's
   stock `ssdlite320_mobilenet_v3_large()` factory). NMS runs
   unconditionally inside the model's own eval-mode forward pass. But
   empirically, surviving boxes' pairwise IoU clustered at 0.546-0.550 —
   mechanically just under that 0.55 cutoff. `nms_thresh` tightened to
   0.45 (the plain SSD base class's own default, not a value fit to this
   eval set) and re-evaluated on the *same, unretrained* weights:
   AP@0.5 moved 0.1895 → 0.1914 (+0.0019 — noise, not "substantial"),
   while small-object recall actually **dropped** 0.2524 → 0.2300
   (214 → 195 matched). NMS tuning is not the fix, in either direction.
2. **Optimization/classification collapse?** Confirmed, decisively. 2,299
   raw predicted scores sampled across 15 real test images all fell
   within **[0.4277, 0.4531]** — a 0.025-wide band, 100% inside one
   histogram bin out of ten. The classification head is not
   discriminating drone-vs-background by location at all.
3. **Learning rate too high (config's stated 0.005)?** The premise was
   wrong — `config/dut_train.yaml` already uses `learning_rate: 0.001`
   (0.005 was the *original* default that caused a NaN divergence, fixed
   in a previous pass; see entry #14). Tested anyway, as a real
   controlled comparison: same seed (42), same 2 epochs, `learning_rate:
   0.0005` (`config/dut_train_lr0005.yaml`, `models/dut_v1_lr0005/`).
   Result: **worse**, not better — final loss 4.500 vs. 4.487 (no
   meaningful difference), AP@0.5 **0.1606** vs. 0.1914, small-object
   recall **0.1297** (110/848) vs. 0.2300, and the same narrow-band score
   collapse persisted (`[0.4651, 0.4871]`). A lower learning rate makes
   *less* progress in the same fixed step budget (1,300 optimizer steps
   over 2 epochs) — it does not fix a collapsed, freshly-initialized
   classification head.

**Conclusion the evidence actually supports:** this is undertraining, not
a configuration bug — consistent with this document's previous read, but
now backed by a direct, narrow-band score histogram and a real
negative-result LR comparison rather than inference from the loss curve
alone. `nms_thresh=0.45` is kept as the new default (a more principled
value for a single dominant-class problem, and a small net-neutral-to-positive
change on the headline metric) but is not, and was never expected to be,
a fix for the underlying collapse.

## Baseline comparison

Published baseline numbers below are from **Zhao, Zhang, Li, Wang,
"Vision-based Anti-UAV Detection and Tracking," IEEE Transactions on
Intelligent Transportation Systems, 2022 (arXiv:2205.10851), Table II** —
the same paper that introduced this dataset and this exact train/test
split. All 14 detectors listed there were retrained on this dataset's
training split and evaluated on its test split, same as this run. Two
caveats, stated rather than glossed over:

1. **Metric definition may not be exactly comparable.** The paper reports
   a single "mAP" column per detector without stating the IoU convention
   in the visible text (its Fig. 6 separately plots P-R curves at
   IoU=0.5 and IoU=0.75, suggesting "mAP" may be a COCO-style average
   across thresholds, not AP@0.5 specifically). Treat the comparison
   below as directional, not a certified apples-to-apples number.
2. **FPS is not comparable hardware.** This run measured latency on this
   environment's CPU (no GPU available — see `docs/known-limitations.md`);
   the paper does not state its own hardware, but published FPS in the
   50s-60s range for a lightweight detector like YOLOX-ResNet18 is not
   plausible on CPU, so their numbers almost certainly come from a GPU.
   Small-object recall and false-alarm rate are not reported by the
   paper at all (TODO if a source for them is ever found).

| Source | AP@0.5 / mAP | Small-object recall | False-alarm rate | FPS | Notes |
|---|---|---|---|---|---|
| SSD-VGG16 (paper, Table II) | 0.632 | TODO | TODO | 33.2 | Closest one-stage baseline to this run's SSDLite320 architecture. |
| Faster-RCNN-ResNet50 (paper, Table II) | 0.653 | TODO | TODO | 12.8 | Two-stage; user-requested comparison point. |
| Faster-RCNN-ResNet18 (paper, Table II) | 0.605 | TODO | TODO | 19.4 | |
| Cascade-RCNN-ResNet50 (paper, Table II, best mAP) | 0.683 | TODO | TODO | 10.7 | Best detector in the paper's own benchmark. |
| YOLOX-ResNet18 (paper, Table II, fastest) | 0.400 | TODO | TODO | 53.7 | Fastest detector in the paper's own benchmark. |
| This run (`drone_v1`, 2 epochs, CPU, lr=0.001, nms=0.45) | 0.1929 (AP@[.50:.95] 0.1074) | 0.2300 | not measurable | 16.12 | Corrected AP (#19); FPS from the clean #18 run. |
| lr=0.0005 comparison (2 epochs, CPU, nms=0.45) | 0.1613 (AP@[.50:.95] 0.0874) | 0.1297 | not measurable | 17.45 | Worse, not better — see entry #18. Corrected AP (#19). |
| This run, 3×3 tiled inference (same weights, opt-in) | 0.3058 (AP@[.50:.95] 0.1758) | 0.3809 (323/848) | not measurable | 2.81 | #21. Small-object *recall* rises at 0.35, but small AP stays 0.002 — recall bought among ~300 FP/frame. |

**Read on this result (locked 2026-10-10):** still below every
published baseline. The earlier read here — "train longer" — has been
tested and superseded: a 20-epoch run was roughly level with the 2-epoch
model on validation for most epochs (best epoch 0.128 vs. 0.093), and the
anchor analysis (#23) shows why — small drones have
no matchable anchor in this architecture, so more epochs cannot teach
them. Tiling closes part of the gap (0.19 → 0.31) without retraining.
The remaining gap is architectural for small drones and a collapsed
score head for the operating point; see the Summary at the top of this
report.
