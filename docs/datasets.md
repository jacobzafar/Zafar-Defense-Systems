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
  — both agree exactly. See `docs/known-limitations.md` and the
  converter's own printed stats for the full breakdown (small-object
  fraction, one degenerate zero-area box found in `val`).
- Annotation format: Pascal-VOC-style XML, one file per image
  (`<annotation><object><name>UAV</name><bndbox>xmin/ymin/xmax/ymax`,
  absolute pixel coordinates). Single class, literally named `"UAV"` in
  the source data — mapped to this repo's `"drone"` category by the
  converter (`scripts/convert_dut_anti_uav.py`).
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

`local_path` in the registry is intentionally left `None` at the Python
level — it is a single field, and this dataset has three splits with
different roles (train / eval-frozen-test), so the actual paths are wired
directly into the training and eval configs (`config/dut_train.yaml`,
`eval/config/dut_anti_uav.yaml`) rather than forced through one field
that can't represent "which split for what purpose."
