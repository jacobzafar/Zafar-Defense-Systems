# Dataset inventory

Generated from `detector/datasets/manifest.py`'s `DATASET_REGISTRY`. This
is the single source of truth — if this table and the registry ever
disagree, the registry wins and this file is stale.

See that module's docstring for what `license_id` / `commercial_ok` mean:
`license_id` is a **verbatim** string from the dataset's own source (or
the literal `"UNVERIFIED"`), never a guessed SPDX identifier;
`commercial_ok` is `True`/`False` only once a maintainer has confirmed it
from that verbatim text, `None` (unknown) otherwise. `detector/train.py`'s
`--commercial-only` flag refuses to train on any dataset whose
`commercial_ok` isn't explicitly `True`.

| Key | Name | Classes | `license` | `license_id` | `commercial_ok` | `source_url` | `local_path` |
|---|---|---|---|---|---|---|---|
| `anti-uav` | Anti-UAV | drone | UNVERIFIED | UNVERIFIED | unknown | _(unset)_ | _(unset)_ |
| `dut-anti-uav` | DUT Anti-UAV | drone | UNVERIFIED | UNVERIFIED | unknown | https://github.com/wangdongdut/DUT-Anti-UAV | set per-split, see below |
| `drone-vs-bird` | Drone-vs-Bird | drone, bird | UNVERIFIED | UNVERIFIED | unknown | _(unset)_ | _(unset)_ |
| `visiodect` | VisioDECT | drone | UNVERIFIED | UNVERIFIED | unknown | _(unset)_ | _(unset)_ |

## `dut-anti-uav` — the one dataset actually in use

This is the first dataset in this repo's history to move past
infrastructure-only status. What's real, and what isn't, as of this pass:

**Confirmed (verified against the primary source, 2026-07-22):**
- Official repo: https://github.com/wangdongdut/DUT-Anti-UAV (raw README
  fetched directly from `raw.githubusercontent.com`, not paraphrased).
- Detection subset: 10,000 static images total, official train/val/test
  split of 5200/2600/2200 images (5243/2621/2245 UAV objects — some
  images have 2+ objects). Every one of these counts was verified twice:
  once by the paper's own text (Zhao et al., IEEE TITS 2022,
  arXiv:2205.10851, Section III.A and Table I) and once by actually
  downloading all three splits and counting the extracted files directly
  — both agree exactly.
- Annotation format: Pascal-VOC-style XML, one file per image
  (`<annotation><object><name>UAV</name><bndbox>xmin/ymin/xmax/ymax`,
  absolute pixel coordinates). Single class, literally named `"UAV"` in
  the source data — mapped to this repo's `"drone"` category by the
  converter (`detector/datasets/dut_anti_uav.py`).

**Real conversion output** (`python -m detector.datasets.dut_anti_uav
--raw-dir data/dut-anti-uav/raw/<split> --output-dir
data/dut-anti-uav/converted/<split>`, then re-verified by loading the
converted output back through `detector/datasets/loader.py`):

| Split | Images | Drone instances | Hard-negative images | Small instances (<32×32=1024px²) | Degenerate boxes skipped |
|---|---|---|---|---|---|
| train | 5200 | 5243 | 3 | 2723 (51.9%) | 0 |
| val | 2600 | 2620 | 0 | 1401 (53.5%) | 1 |
| test | 2200 | 2245 | 0 | 848 (37.8%) | 0 |

Two real data-quality findings, handled explicitly rather than silently:
- **3 genuine background-only images in `train`** (`00579.xml`, `00639.xml`,
  `00724.xml` — zero `<object>` tags in the raw XML, not an artifact of
  conversion). `val`/`test` have none. These are real hard negatives the
  false-alarm-rate metric can use, just very few of them, and only in the
  training split.
- **1 degenerate (zero-area) box in `val`** (`00991.xml`: box
  `(1056,443)-(1059,443)` — zero height) — skipped by the converter with
  a counted warning rather than kept as a zero-area training target or
  silently dropped along with the rest of that image's valid boxes.
- The repository itself carries an Apache-2.0 `LICENSE` file (fetched
  verbatim from
  `raw.githubusercontent.com/wangdongdut/DUT-Anti-UAV/master/LICENSE` —
  confirmed to be the standard Apache License 2.0 text, byte-for-byte).

**NOT confirmed — this is the actual blocker, not a formality:**
The Apache-2.0 file governs the GitHub repository's own contents (a
README and one example image — the dataset itself is not stored in that
repo). The actual images/annotations are hosted externally on Google
Drive / Baidu Pan links, and no explicit license statement was found
attached to *those*. Academic datasets distributed this way very
frequently carry research-only terms distinct from whatever license
covers the authors' code repository, so assuming the Apache-2.0 grant
extends to the dataset content would be a guess, not a verification —
exactly what `license_id`/`commercial_ok` exist to prevent. Both remain
`UNVERIFIED`/`unknown` in the registry pending a maintainer pasting the
authoritative text (the paper's data-availability statement, direct
author correspondence, or an explicit statement on the download page
itself).

**Practical effect today:** this dataset is used below for fine-tuning
and evaluation (research/development use, consistent with its evident
purpose as a published academic benchmark), but `detector/train.py
--commercial-only` will refuse it until that confirmation lands — see
`detector/train.py`'s `_assert_commercial_clearance`.

**Local layout** (gitignored — see `.gitignore`; nothing under `data/` or
`models/` is committed):

```
data/dut-anti-uav/raw/{train,val,test}/{img,xml}/...       # as downloaded, VOC XML
data/dut-anti-uav/converted/{train,val,test}/images/       # this repo's unified schema
data/dut-anti-uav/converted/{train,val,test}/annotations.json
data/frozen_eval/dut-anti-uav-test/                        # frozen copy of the test split, see eval/README.md
```

**The frozen eval set** (`data/frozen_eval/dut-anti-uav-test/`) was built
from `data/dut-anti-uav/converted/test/` via
`eval.build_frozen_eval_set.build_frozen_eval_set()` — one single-frame
"sequence" per image (the detection subset has no temporal structure
between images, so this is the honest representation, not an
approximation of a real multi-frame sequence; see that module's
docstring for what this means for the track-continuity metric). Marked
frozen (`.frozen` marker file); its `sequences.json` sha256 as of this
build is `d7d58f3a2e22d3aec8814ae47f38ee145a735f53c4e9cab16c8732a1eaa2244c`
(2200 sequences/images, matching the test split's real count above).
`detector/train.py`'s `.frozen` guard was confirmed against this exact
directory, not just a synthetic fixture — pointing `TrainConfig.dataset_dir`
at it raises `EvalSetFrozenError` before any data loading happens
(`tests/test_train.py::test_train_refuses_a_frozen_eval_set_built_from_a_real_dataset`
covers the same code path with a fast synthetic fixture, since the real
~275MB test split isn't something CI should need on disk).

`local_path` in the registry is intentionally left `None` at the Python
level — it is a single field, and this dataset has three splits with
different roles (train / eval-frozen-test), so the actual paths are wired
directly into the training and eval configs (`config/dut_train.yaml`,
`eval/config/dut_anti_uav.yaml`) rather than forced through one field
that can't represent "which split for what purpose."
