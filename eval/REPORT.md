# Evaluation Report

- Evaluated at: 2026-10-10T10:14:06 (corrected-AP re-run; see "AP correction" below)
- Eval set: `data/frozen_eval/dut-anti-uav-test`
- Eval set manifest checksum (sequences.json sha256): `d7d58f3a2e22d3aec8814ae47f38ee145a735f53c4e9cab16c8732a1eaa2244c`
- Detector backend: `drone_v1` (`models/dut_v1/weights.pt`, lr=0.001, 2 epochs; `nms_thresh=0.45`; operating `confidence_threshold=0.35`)
- Tracker backend: `iou`
- Sequences: 2200  |  Frames: 2200  |  GT drone boxes: 2245

**This eval set is frozen and must never be used for training** — see eval/README.md.

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

**Read on this result, updated:** still well below every published
baseline, and still, honestly, "proof the pipeline works end-to-end, not
a working detector" — but that phrase now rests on a specific, verified
mechanism (a collapsed classification head, evidenced directly) and two
ruled-out alternative explanations (NMS, this specific LR change), not
just "2 epochs is probably not enough." The single clearest next step
remains what §6 of `docs/STATUS.md` already said before this pass: train
longer. This pass adds the evidence for *why* that's the right call
rather than a guess.
