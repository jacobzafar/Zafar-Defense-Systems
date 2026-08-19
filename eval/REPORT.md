# Evaluation Report

- Evaluated at: 2026-07-23T10:25:16
- Eval set: `data/frozen_eval/dut-anti-uav-test`
- Eval set manifest checksum (sequences.json sha256): `d7d58f3a2e22d3aec8814ae47f38ee145a735f53c4e9cab16c8732a1eaa2244c`
- Detector backend: `drone_v1` (`models/dut_v1/weights.pt`, lr=0.001, 2 epochs; `nms_thresh=0.45` — see below)
- Tracker backend: `iou`
- Sequences: 2200  |  Frames: 2200

**This eval set is frozen and must never be used for training** — see eval/README.md.

## Metric card

| Metric | Value | Definition |
|---|---|---|
| AP@0.5 (single-class "drone") | 0.1914 | Average precision at IoU=0.5, all-point interpolation. |
| Small-object recall | 0.2300 (195/848) | Recall on GT boxes with area < 1024px² (COCO "small" convention). |
| False-alarm rate | **not measurable** (No hard-negative (zero-GT) frames in this eval set — false-alarm rate is not measurable here.) | 0 false detections across 0 hard-negative (no-drone) frames. |
| Latency (mean) | 62.03 ms | Wall-clock detector.detect() time per frame, this run's hardware only. |
| Latency (p95) | 87.54 ms | |
| Throughput | 16.12 FPS | 1000 / mean latency. |
| Track continuity | **not measurable** (Every sequence in this eval set has at most 1 frame — an ID switch is structurally impossible to observe, so track continuity is not measurable here.) | 1 - (ID switches / frames with GT present); 0 switches across 2200 frames. |

## Per-sequence track continuity

Not measurable — Every sequence in this eval set has at most 1 frame — an ID switch is structurally impossible to observe, so track continuity is not measurable here. (2200 single-frame sequences omitted from a per-sequence table; every one has 0 ID switches by construction, which is not evidence of anything.)

## What changed since the last report, and why (docs/DECISIONS.md entry #18 has the full diagnosis)

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
| This run (`drone_v1`, 2 epochs, CPU, lr=0.001, nms=0.45) | 0.1914 | 0.2300 | not measurable | 16.12 | See "What changed" above and docs/DECISIONS.md entry #18. |
| lr=0.0005 comparison (2 epochs, CPU, nms=0.45) | 0.1606 | 0.1297 | not measurable | 17.45 | Worse, not better — see entry #18. Kept for the record, not recommended. |

**Read on this result, updated:** still well below every published
baseline, and still, honestly, "proof the pipeline works end-to-end, not
a working detector" — but that phrase now rests on a specific, verified
mechanism (a collapsed classification head, evidenced directly) and two
ruled-out alternative explanations (NMS, this specific LR change), not
just "2 epochs is probably not enough." The single clearest next step
remains what §6 of `docs/STATUS.md` already said before this pass: train
longer. This pass adds the evidence for *why* that's the right call
rather than a guess.
