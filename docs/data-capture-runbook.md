# Data-capture runbook: own FPV footage

Practical guide for drone filming sessions: what to shoot, and how to turn
the files into reviewed training data the same day. Context: detection on
public data is locked (`eval/REPORT.md`); our own footage is the next
lever. Its first job is to answer one question (`docs/DECISIONS.md`
#23-#24): **are our drones, as our camera sees them, inside or below the
current architecture's anchor floor?**

"Footage" here means video **of** our FPV drone, filmed by an observing
camera (ideally the camera type we would deploy). It does not mean the
drone's own onboard feed.

Observation only. Fly under the operator's normal permits and local
aviation rules, and avoid filming identifiable bystanders.

## 1. Before the shoot

```bash
python tools/coverage_report.py          # what's captured, what's missing
```

Bring: the observing camera (note make, model, resolution and zoom/focal
length), a tripod, a way to know distance (rangefinder, paced or marked
distances, or GPS/OSD telemetry), and a printed shot list (section 2).
Set the camera to its highest frame rate at 1080p or above. Do not use
digital stabilisation or beauty or HDR filters.

## 2. What to shoot

**One clip = one set of conditions.** Each clip gets a single
drone_type / distance band / lighting / background value at ingest. If the
drone flies through several distance bands, cut the clip per band
afterwards, or fly *holds* at each distance. 20-40 s per clip is plenty:
at the default 5 FPS that is 100-200 frames to review.

**Expectation, so the shoot is planned around it.** The current detector
(SSDLite, input 320) can only learn drones that fill a large part of the
frame. In a simulation with the same anchors (`eval/anchor_coverage.py`),
a drone is matchable once it is about **11% of frame width** (square
shape; about 17% for a flat, side-on FPV shape). With 3×3 tiling the
figure is about **4-7%**. This is a fraction of the frame, so **filming in
4K does not help**. Only distance and zoom do. As a rough guide for a
5-inch quad (about 0.25 m across) on an 80° lens, that is about 1.3 m
untiled and about 3.5 m tiled. With about 5× zoom (20° lens) it is about
6 m and 17 m. These are estimates from assumed optics. The coverage
report's anchor check measures the real figure. So shoot **every distance
band anyway**: the far bands are what tell us whether a stride-8 or FPN
model (#23 B/D) is needed. Also shoot extra close-range material, which is
the only footage this model can learn from today. Record the approximate
distance in `--notes` (e.g. `range_m=8`), because the bands are coarse at
the close end.

### Positive shots (drone in frame), by priority

| # | Distance band | What | Why |
|---|---|---|---|
| 1 | `close-lt50m` | Holds at roughly 3, 5, 10, 20, 40 m (one clip each) | Spans the predicted anchor cutoff, which the coverage report then measures |
| 2 | `medium-50-200m` | Holds at roughly 75 and 150 m, plus slow passes | Realistic engagement range; expected to be under the floor |
| 3 | `far-200-500m` | Holds at 250 and 400 m, if visible at all | Sets how far the problem extends; often a few pixels |
| 4 | `very-far-gt500m` | Only if the drone is visible on screen | Documents the limit |

For each distance, vary in this order when time allows:

- **Angle:** ground camera looking up (main case), level/side-on, and
  looking down from a height (hill or roof). Put the angle in `--notes`.
- **Motion:** hover, slow lateral pass, fast pass, approach and recede.
  FPV speed is a known gap (`docs/coverage-matrix.md`).
- **Camera:** mostly tripod-locked (the tracker assumes a static camera,
  `docs/known-limitations.md`), plus some handheld or panning clips. Note
  which in `--notes`.

**Backgrounds** (`--background`), most valuable first: `clean-sky`,
`foliage-trees` (the drone against treelines is the hard case),
`urban-buildings`, `terrain-mountains`, `water`, `mixed-cluttered`. The
easy wins are clear sky and a treeline from the same spot, by tilting the
camera.

**Lighting** (`--lighting`): `daylight-clear` and `daylight-overcast`
first, then a `dusk-dawn` session (low contrast, backlit sun). Shoot
`night`, `rain`, `fog-haze` and `snow` when the chance comes up. The sensor
is visible-light only. Note IR in `--notes` if a thermal camera is ever
used.

### Hard negatives (no drone anywhere in frame)

Drone-free footage lets us measure the **false-alarm rate** for the first
time (DUT has none). The drone must be on the ground or out of frame for
the entire clip. Ingest with `--hard-negative --drone-type none
--negative-subject ...`:

| Subject | Where / how |
|---|---|
| `bird` | Gulls over water or a landfill, crows and pigeons in parks, birds of prey over fields. The most important negative. Get close and far. |
| `aircraft` | Planes on approach paths, helicopters, light aircraft, at varied distances |
| `insect` | Insects passing close to the lens (look like a small dark blob at range) |
| `empty-sky` | The same backgrounds as the positive clips with nothing flying: clouds, treelines moving in wind, power lines, rooftops |
| `other` | Kites, balloons, plastic bags in wind (record what in `--notes`) |

Rule of thumb: about 1 minute of hard negatives for every 3 minutes of
drone footage, on the **same backgrounds** as the positives. Otherwise the
model learns "background X means drone".

### Hold-out for evaluation

Keep whole sessions (a day, or a location) out of training for
evaluation, never individual frames. Neighbouring frames of one clip are
near-duplicates, and splitting them leaks test data into training.
Assigning held-out clips is a manual decision for now. Nothing in the
tools enforces it yet.

## 3. After the shoot: the exact commands

Copy the card off first. Keep the original files: ingest records their
SHA-256 and never modifies them. Run from the repo root, inside `.venv`
(torch is only needed for `preannotate` with the drone backend and for the
anchor check).

**Step 1: ingest each clip** (one line per clip; a shot log written on
site makes this a quick transcription job):

```bash
python tools/ingest_footage.py --video /media/card/DCIM/C0012.MP4 --clip-id 2026-10-20-close-05m-sky \
    --drone-type fpv-quad --distance-band close-lt50m --lighting daylight-clear --background clean-sky \
    --capture-date 2026-10-20 --notes "range_m=5, ground looking up, tripod, Sony ZV-E10 24mm"
python tools/ingest_footage.py --video /media/card/DCIM/C0031.MP4 --clip-id 2026-10-20-gulls-harbour \
    --hard-negative --drone-type none --negative-subject bird \
    --distance-band close-lt50m --lighting daylight-clear --background water --capture-date 2026-10-20
```

Output goes to `data/own-fpv/clips/<clip-id>/` (gitignored, like all of
`data/`). `--fps 5` is the default. Use `--fps 2` for long, static clips.

**Step 2: pre-label with the current best detector** (dut_v1, untiled or
tiled):

```bash
for clip in data/own-fpv/clips/*/; do
  python tools/preannotate.py --source "$clip" --detector drone --weights models/dut_v1/weights.pt \
      --max-per-frame 3 --tile-rows 3 --tile-cols 3
done
```

This writes `<clip>/prelabels/`: `coco_predictions.json` (CVAT),
`labelstudio_tasks.json` + `labelstudio_config.xml` (Label Studio). These
are **predictions, not labels**. Be realistic: on DUT val, dut_v1's top 3
boxes per frame cover only 13% of drones (18% tiled). For the first
session, a reviewer will mostly be deleting wrong boxes and drawing new
ones. In CVAT it is usually faster to **skip the pre-labels and annotate
in track mode**: box the drone on a few keyframes and let interpolation
fill in the rest. Pre-labels become worthwhile after a detector has been
retrained on this footage. Tiling costs about 6× the CPU time (about 0.36
s/frame instead of 0.06).

**Step 3: review in an annotation tool.** Labels: `drone`, plus `bird`,
`aircraft` and `other` for visible distractors. Box only what is actually
visible. A frame with no drone stays empty: that is a valid label (hard
negative). Use one of the two tools:

- **CVAT:** create a task. Labels are `drone`, `bird`, `aircraft`,
  `other`. Upload the files from `<clip>/images/`. Optionally go to
  *Actions → Upload annotations → COCO 1.0* and choose
  `<clip>/prelabels/coco_predictions.json`. Review every frame and mark
  the job completed. Then *Export task dataset → COCO 1.0* and unzip
  `annotations/instances_default.json`.
- **Label Studio:**
  ```bash
  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/data/own-fpv/clips" label-studio start
  ```
  Create a project and paste `<clip>/prelabels/labelstudio_config.xml`
  under *Settings → Labeling Interface → Code*. Under *Settings → Cloud
  Storage → Add Source Storage → Local files*, add the absolute path of
  `<clip>/images` (do not sync). *Import* `labelstudio_tasks.json`. The
  boxes appear as predictions. Correct each task and **Submit** it. Then
  *Export → JSON*.

**Step 4: bring the reviewed labels back:**

```bash
python tools/import_reviewed.py --clip data/own-fpv/clips/2026-10-20-close-05m-sky --export instances_default.json
```

This writes `<clip>/annotations.json`, which the training loader reads.
It refuses the pre-label file itself, and it refuses a clip where some
frames were not reviewed (`--allow-partial` keeps only the reviewed
ones). Only Label Studio's submitted *annotations* are read, never its
predictions.

**Step 5: coverage and the anchor answer:**

```bash
python tools/coverage_report.py --output-dir data/own-fpv/coverage
python tools/coverage_report.py --tile-rows 3 --tile-cols 3   # same, with tiling simulated
```

The anchor-check column and the per-size / per-distance tables use
exactly #23's measurement. On DUT train, fed through this same report,
they reproduce #23's numbers: small 0.0% (0.007), medium 0.7% (0.023),
large 70.3% (0.676). How to read them:

- **Close and medium bands mostly matchable:** the current architecture
  (plus tiling where latency allows) may be enough for our scenario.
  Retrain on our footage first.
- **Mostly 0% beyond a few metres:** our real footage hits the same
  floor as DUT. #23 option B (stride-8 + 16px anchors at 640) or D (FPN)
  is needed. Re-run with `--input-size 640` to confirm 640 alone does
  not fix it.

`coverage_matrix.csv` lists every combination, including empty ones.
Filter for `not captured` to plan the next shoot. Clips without reviewed
labels show `unreviewed` in the anchor column. They are never measured
from pre-labels, because the detector mostly finds the drones its anchors
already fit.

## 4. Not covered yet

- Merging reviewed clips into one training directory: `detector/train.py`
  takes a single `dataset_dir`, and each clip is its own unified-layout
  directory.
- Marking held-out evaluation clips, and building a frozen eval set from
  them (`eval/build_frozen_eval_set.py` targets DUT).
- `docs/coverage-matrix.md` (the manual scenario log) is superseded for
  our own footage by `tools/coverage_report.py`, but has not been folded
  into it.
