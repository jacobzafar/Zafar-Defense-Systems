# Evaluation Report

- Evaluated at: 2026-07-22T18:29:03
- Eval set: `data/frozen_eval/dut-anti-uav-test`
- Eval set manifest checksum (sequences.json sha256): `d7d58f3a2e22d3aec8814ae47f38ee145a735f53c4e9cab16c8732a1eaa2244c`
- Detector backend: `drone_v1`
- Tracker backend: `iou`
- Sequences: 2200  |  Frames: 2200

**This eval set is frozen and must never be used for training** — see eval/README.md.

## Metric card

| Metric | Value | Definition |
|---|---|---|
| AP@0.5 (single-class "drone") | 0.1895 | Average precision at IoU=0.5, all-point interpolation. |
| Small-object recall | 0.2524 (214/848) | Recall on GT boxes with area < 1024px² (COCO "small" convention). |
| False-alarm rate | **not measurable** (No hard-negative (zero-GT) frames in this eval set — false-alarm rate is not measurable here.) | 0 false detections across 0 hard-negative (no-drone) frames. |
| Latency (mean) | 60.27 ms | Wall-clock detector.detect() time per frame, this run's hardware only. |
| Latency (p95) | 84.34 ms | |
| Throughput | 16.59 FPS | 1000 / mean latency. |
| Track continuity | **not measurable** (Every sequence in this eval set has at most 1 frame — an ID switch is structurally impossible to observe, so track continuity is not measurable here.) | 1 - (ID switches / frames with GT present); 0 switches across 2200 frames. |

## Per-sequence track continuity

Not measurable — Every sequence in this eval set has at most 1 frame — an ID switch is structurally impossible to observe, so track continuity is not measurable here. (2200 single-frame sequences omitted from a per-sequence table; every one has 0 ID switches by construction, which is not evidence of anything.)


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
| This run (`drone_v1`, 2 epochs, CPU) | 0.1895 | 0.2524 | not measurable | 16.59 | See "Real numbers vs. baseline" in this repo's summary for the honest read. |
