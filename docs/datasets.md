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

## `visiodect` — ingested, not yet trained on

A copy was supplied for this pass (not downloaded by this repo) and
ingested via a new converter (`detector/datasets/visiodect.py`). Per the
task this was done under: made available and reported on, **not** trained
on — no config/training run touches it yet.

**Real structure, verified by inspecting the extracted archive directly
before writing any loader code** (not assumed from documentation): one
directory per UAV model, each with `images/<Scenario>/` (Title-case:
Evening/Cloudy/Sunny) and `labels/<scenario>/` (lowercase) holding up to
three parallel annotation formats for the same boxes — `csv.csv` (one row
per box: `class_name,xmin,ymin,width,height,file_name,img_width,img_height`,
absolute pixels, no header), a `txt/` directory (YOLO-style, normalized),
and a `voc/` directory (Pascal-VOC XML). All three formats were spot-checked
against each other for one image and agree exactly. **CSV is the only
format present for every populated model/scenario** — `txt`/`voc` are
missing entirely for some scenarios and badly incomplete for others (e.g.
`Anafi-Extended/Evening`: 1200 images, csv rows for all 1200, but only 801
`txt` files and none at all under `Anafi-Extended/labels/cloudy/`) — so the
converter reads `csv.csv` exclusively.

**Coverage actually supplied — six UAV models documented, three populated:**

| UAV model | Data present? |
|---|---|
| Anafi-Extended | Yes |
| DJIFPV | Yes |
| DJIPhantom | Yes (partial — see below) |
| EFT-E410S | **No — empty directory skeleton only** (`images/<Scenario>/`, `labels/<scenario>/{voc,txt}/` all exist, zero files in any of them) |
| Mavic_Air | **No — empty directory skeleton only**, same as above |
| Mavic_Enterprise | **No — empty directory skeleton only**, same as above |

**Real conversion output** (`python detector/datasets/visiodect.py
--raw-dir data/visiodect/raw --output-dir data/visiodect/converted`, then
re-verified by loading the converted output back through
`detector/datasets/loader.py`):

| Model | Scenario | Images on disk | Annotated | Drone instances | Orphaned annotations | Degenerate skipped | Format |
|---|---|---|---|---|---|---|---|
| Anafi-Extended | Evening | 1200 | 1200 | 1200 | 0 | 0 | csv |
| Anafi-Extended | Cloudy | 1200 | 1200 | 1202 | 0 | 2 | csv |
| Anafi-Extended | Sunny | 1071 | 989 | 1053 | 0 | 0 | csv |
| DJIFPV | Evening | 1200 | 1127 | 1127 | 0 | 0 | csv |
| DJIFPV | Cloudy | 1200 | 1112 | 1112 | 0 | 0 | csv |
| DJIFPV | Sunny | 1200 | 1200 | 1200 | 0 | 0 | csv |
| DJIPhantom | Evening | **0** | 0 | 0 | **1200** | 0 | csv |
| DJIPhantom | Cloudy | 900 | 900 | 901 | 0 | 0 | csv |
| DJIPhantom | Sunny | 203 | 0 | 0 | 0 | 0 | **xlsx (not ingested)** |
| EFT-E410S / Mavic_Air / Mavic_Enterprise | all | 0 | 0 | 0 | 0 | 0 | none |
| **TOTAL** | | **8174** | | **7795** | **1200** | **2** | |

446 images (8174 - annotated-image count) are hard negatives (zero boxes)
in the converted output.

**Box-size distribution of all 7795 drone instances** (same area buckets
as `eval/metrics.py`'s small-object convention, 1024px²=32x32):

| Bucket | Count | % |
|---|---|---|
| <16x16 (<256px²) | 46 | 0.6% |
| 16x16-32x32 (256-1024px²) | 1126 | 14.4% |
| 32x32-64x64 (1024-4096px²) | 4700 | 60.3% |
| 64x64-128x128 (4096-16384px²) | 1560 | 20.0% |
| >=128x128 (>=16384px²) | 363 | 4.7% |

**Read on this, directly answering "what coverage does this actually
give us":** only half the documented UAV models (3 of 6) have any data in
the copy supplied, and all three scenarios (sunny/cloudy/evening) are
present only for those three. Despite including an "FPV" model category
(`DJIFPV`), the box-size distribution skews toward *larger*, not smaller,
targets than `dut-anti-uav`: only ~15% of instances fall under the same
1024px² "small" threshold here (46+1126 of 7795), vs. 37.8-53.5% for
`dut-anti-uav`'s three splits. **This dataset, as currently supplied, is
not primarily a source of very-small/distant-target coverage** — it adds
model-type and lighting-condition diversity (three additional real
airframes, three lighting conditions DUT Anti-UAV doesn't label
separately), not the small-object signal one might assume from "FPV" being
present.

**Three real data-quality/completeness issues found, handled explicitly
by the converter, not silently:**
1. Three of six model directories are empty structure only (above).
2. `DJIPhantom/labels/evening/csv.csv` has 1200 annotation rows for image
   files that do not exist anywhere in the supplied copy — skipped and
   counted (`num_orphaned_annotations`), never fabricated into a training
   target for a nonexistent file.
3. `DJIPhantom/labels/sunny/csv.xlsx` is the only non-CSV annotation file
   in the whole archive (203 images) — not parsed; no spreadsheet-parsing
   dependency was added for one outlier. Images are still counted as
   present on disk; they carry no boxes in the converted output.

**License: UNVERIFIED, `commercial_ok` unknown — and will stay that way
until someone supplies more than this archive.** The full extracted tree
was searched for a LICENSE file, README, or any attribution/terms text of
any kind — **none exists in the copy supplied for this pass.** This is a
stronger negative than `dut-anti-uav`'s situation (which at least has a
repo-level Apache-2.0 file, even though its scope is disputed): here there
is nothing to cite at all. Per `detector/datasets/manifest.py`'s own
rules, `license_id` stays the literal string `"UNVERIFIED"` and
`commercial_ok` stays `None` — not a guess, not an inference from the
dataset's evident research origin. `detector/train.py --commercial-only`
will refuse this dataset exactly as it refuses `dut-anti-uav` today.

**Local layout** (gitignored, same as every other dataset under `data/`):

```
data/visiodect/raw/<Model>/images/<Scenario>/*.jpg
data/visiodect/raw/<Model>/labels/<scenario>/csv.csv   # + txt/, voc/ where present (unused by the converter)
data/visiodect/converted/images/                        # this repo's unified schema, symlinked not copied
data/visiodect/converted/annotations.json
data/visiodect/converted/conversion_stats.json          # the real per-model/scenario table above, machine-readable
```
