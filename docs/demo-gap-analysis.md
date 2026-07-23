# Demo Gap Analysis — MVP Demo v1

Source of truth: MVP demo v1 specification (video in -> detection -> tracking
-> visualization -> logging, observation-only, recorded video as primary
input). This document compares that spec against the real state of this
repository as of commit `ee07485` (branch `feature/software-mvp-scaffold`).

No application code was changed to produce this analysis.

> **Status update:** a follow-up UI/demo-readiness pass (see
> `docs/DECISIONS.md` entry #7) has since addressed several items from
> this analysis: the `IDLE -> ERROR` state-machine bug (§3.4's "hard
> failure" concern) is fixed, the UI's per-frame video re-seek (§3.5) is
> gone (frames now flow through `FrameResult.frame`), a built-in synthetic
> demo source now exists so the app can be exercised before real footage
> is available (§3.1/§6 are still open for *real* footage, but the
> "nothing to test with" gap is closed), config presets now exist and are
> loaded/validated (§3.3), and the UI status/event log (§3.4) is
> substantially richer. **§3.1 — no real demo clips — remains the top
> open item**; the rest of this document is left as-is as the original
> point-in-time analysis.

## 1. Current repository state

Verified by actually running the code (not just reading it): `pip install -r
requirements.txt`, `pytest tests/ -v` (9/9 passed), `scripts/run_pipeline.py`
against a synthetic in-memory video, and `streamlit run ui/app.py` (booted,
served HTTP 200, no import errors).

```
control/     Pipeline orchestration (Pipeline.run loop), a 4-state state
             machine (IDLE/RUNNING/STOPPED/ERROR), FrameResult event type.
detector/    BaseDetector interface. Default backend: MotionDetector
             (OpenCV MOG2 background subtraction) — flags any moving blob,
             not specifically an aerial target. Optional UltralyticsDetector
             adapter exists but is unused/unwired (AGPL licensing question
             unresolved, see docs/DECISIONS.md #2).
tracker/     BaseTracker interface. Default backend: IoUTracker (greedy
             IoU matching, no re-identification). Optional ByteTrack
             adapter exists but requires extra packages not in
             requirements.txt.
ui/          Streamlit app (ui/app.py): file upload, backend selectors,
             live frame display with overlay, a text status line, and a
             scrolling text "event log" (last 20 lines, frame/track/
             detection counts).
telemetry/   JSONL logger, one file per run, timestamps + event payloads.
scripts/     Headless CLI runner with --source/--detector/--tracker/
             --save-video/--max-frames flags.
tests/       9 unit/smoke tests, all synthetic (a moving white rectangle on
             a black frame) — none run against real footage.
docs/        architecture.md, DECISIONS.md, known-limitations.md,
             software-mvp-plan.md — all describe the pipeline accurately.
config/      example.yaml exists as a reference only; nothing in the
             codebase actually loads it (confirmed: no `yaml` import
             anywhere in the repo, no PyYAML dependency).
```

**Not present anywhere in the repo:** any video file, any `clips/` or
`test_data/` directory, any dataset, any CI config, any demo runbook.

## 2. What already satisfies the MVP demo spec

| Spec item | Status | Evidence |
|---|---|---|
| #1 Recorded video as primary input | **Met** | `cv2.VideoCapture` accepts file paths in both `scripts/run_pipeline.py` and `ui/app.py` (file upload); webcam index also works as the optional secondary path. |
| #2 Visual marking of a detected target | **Mechanically met** | `ui/overlay.draw_tracks` draws a bounding box + label on every track. Whether it looks convincing depends entirely on which clips are used (see §3). |
| #3 Tracking with a stable ID | **Mechanically met** | `IoUTracker` assigns and persists a `track_id`; unit-tested against a synthetically moving box across 20+ frames with a consistent ID. Never validated against real footage. |
| #4 UI shows bbox / ID / confidence / status / log | **Partially met** | Bbox, ID, confidence are all in the overlay label. "Status" is a single text line (`st.info`/`st.success`/`st.warning`) with no idle/running/stopped/error distinction. "Event log" is a scrolling text block of raw per-frame counts, not operator-legible events (e.g. no "target acquired" / "target lost" language). |
| #5 Simple log per run with timestamps | **Met** | `telemetry.logger.EventLogger` writes JSONL with `run_started`, `frame_processed` (includes full track state), `frame_dropped`, `run_stopped`, each timestamped. |
| #7 (bullets 1–2) starts reliably, runs end-to-end | **Met, with one caveat** | Verified clean start and full run in this session. Caveat: if the video path is invalid, `Pipeline.run()` raises a raw `RuntimeError` that neither `scripts/run_pipeline.py` nor `ui/app.py` catches — the CLI would print a Python traceback and the Streamlit app would show its default red exception box. Not "silent crash," but not a clean operator-facing message either. |
| #9 Positioned as observation/decision-support, not fielded system | **Met at the doc level** | README, DECISIONS.md, and known-limitations.md are explicit and consistent about this framing. No code implies otherwise (no effector abstraction exists). |
| #10 Value chain wired end-to-end | **Met** | video -> detect -> track -> overlay -> UI/CLI -> JSONL log is a real, working chain, not a mockup. |

## 3. What is still missing

Ordered roughly by how blocking each item is to a credible demo.

### 3.1 No real test footage (the actual blocker)
There are zero video assets in this repo. Every test and every manual run
so far has used a synthetic rectangle sliding across a black frame. This
means:
- Detection/tracking quality against real aerial footage is **completely
  unvalidated**.
- `MotionDetector`'s default thresholds (`min_area_px=150`,
  `var_threshold=32.0`, `history=300`) and `IoUTracker`'s defaults
  (`iou_threshold=0.3`, `max_age=15`) are untuned guesses.
- Spec requirement #6 ("small set of selected real test videos") and the
  "good enough" bar in #7 ("detects targets in the main demo clips",
  "maintains visually credible tracking") cannot be evaluated at all until
  real clips exist.

Nothing else in this list can be properly finished until this is resolved.

### 3.2 Detector is generic motion detection, not aerial-target-aware
`MotionDetector` will flag *any* moving pixel blob — a bird, a passing
car, a waving tree, a camera jolt. It has no concept of "aerial target."
For v1 this is an accepted, documented placeholder (see
`docs/DECISIONS.md` #2), and the spec's "good enough" bar does not require
a trained classifier. But it does mean **demo credibility now depends
entirely on clip selection**: clips need a static or near-static camera,
a clean/uncluttered background (sky), and minimal irrelevant motion. This
is a clip-curation constraint, not (necessarily) a code gap — but it is a
real risk if the wrong clips are chosen.

### 3.3 No parameter tuning workflow
`config/example.yaml` documents the tunable parameters but nothing loads
it — `scripts/run_pipeline.py` only accepts a fixed set of CLI flags
(`--detector`, `--tracker`, no way to pass `min_area_px`, `var_threshold`,
`iou_threshold`, etc. without editing code). Once real clips exist, each
one will likely need different thresholds, and there's currently no clean,
reproducible way to store "these are the settings for clip X."

### 3.4 UI status and event log are thin
- Status is a single sentence, not a clear operator-facing pipeline state
  (idle / running / stopped / error) matching the state machine that
  already exists in `control/state.py`.
- The in-UI "event log" is raw frame/track/detection counts
  (`frame=12 tracks=2 dets=2 err=None`), not narrative operator events
  ("Target #1 acquired", "Target #1 lost after occlusion"). The JSONL file
  log is fine for engineering purposes but isn't what gets read out loud
  in a demo.
- No Stop control in the UI — `Pipeline.stop()` exists but nothing calls
  it from `ui/app.py`. If something looks wrong mid-demo, the only recourse
  is closing the browser tab.

### 3.5 UI re-seeks the video file every frame
`ui/app.py`'s render loop opens a **new** `cv2.VideoCapture` and calls
`.set(cv2.CAP_PROP_POS_FRAMES, ...)` on every single frame just to fetch
the raw frame for display, in addition to the `VideoCapture` already
opened internally by `Pipeline.run()`. Two known real-world risks:
1. Frame-accurate seeking via `CAP_PROP_POS_FRAMES` is unreliable on
   compressed formats (h264/mp4) in OpenCV — it can land on the nearest
   keyframe rather than the exact frame, causing visible overlay/frame
   misalignment.
2. Opening a new `VideoCapture` per frame is wasteful and will slow down
   playback smoothness as clip length/resolution grows — a live demo
   should not visibly stutter.

This has not caused a visible problem yet only because it hasn't been
exercised against a real, longer, compressed video file.

### 3.6 No demo runbook
There is no single document that says "for the demo, run exactly these
commands/clicks, in this order, and here is what should happen." Given
spec #7's bar of "starts reliably without manual debugging" and #8's
emphasis on a "repeatable demo," this is a real gap — right now,
reliability depends on the presenter remembering the right CLI flags or
UI clicks from memory.

### 3.7 No CI / automated check that the demo path still works
Not explicitly required by the spec, but worth naming: there's no
automated guard that `scripts/run_pipeline.py --source <demo-clip>` keeps
working as the code changes. Low priority relative to the items above, but
cheap to add later.

## 4. Recommended implementation order

Biased toward the fastest path to a credible, repeatable demo — not toward
architectural completeness.

1. **Source and curate 2–4 real demo clips.** This unblocks everything
   else and is a decision only you can make (see §5). Prefer: mostly
   static camera, one clearly visible aerial target, relatively clean sky
   background, a few seconds to ~30s each, varying difficulty (one "easy"
   clip that will clearly work, one or two more representative ones).
2. **Run the existing pipeline against each clip as-is, unmodified**, and
   record what actually happens (false positives, missed detections, ID
   switches, tracking dropouts). This is a diagnostic pass, not a coding
   pass — it tells you exactly which of §3.2/§3.3's risks are real for
   your actual clips versus theoretical.
3. **Tune detector/tracker parameters per clip** based on step 2's
   findings. Decide then whether informal per-clip CLI flags are enough
   or whether wiring up `config/*.yaml` loading (§3.3) is worth the extra
   half-day for reproducibility.
4. **Fix whatever step 2 actually surfaces as broken** — likely candidates
   are motion-detector sensitivity (false positives on background
   clutter) and tracker ID stability (switches on brief occlusion/overlap).
   Don't pre-fix hypothetical problems; fix what the real clips show.
5. **Add graceful error handling at the CLI/UI entry points** (§3.4/known
   gap: unhandled `RuntimeError` on bad source) so a mis-clicked file or
   bad path fails with a clear one-line message instead of a stack trace
   or Streamlit's default red exception box.
6. **Fix the per-frame re-seek in `ui/app.py`** (§3.5) — read the raw
   frame once per iteration from a single capture object shared with (or
   parallel to, but advancing in lockstep with) the pipeline, rather than
   reopening/seeking a new capture every frame. This is the one item in
   this list that's a pure correctness/robustness fix rather than a
   tuning or content task.
7. **Light UI polish**: surface the actual pipeline state
   (idle/running/stopped/error) clearly, reword the live event log into
   operator-facing language, add a Stop button wired to `Pipeline.stop()`.
8. **Write the one-page demo runbook**: exact steps for both the UI path
   and the CLI fallback path, expected runtime, what "success" looks like
   on screen, and what to do if something goes wrong live.
9. *(Optional, only if time allows and clip results from step 2 are weak)*
   Revisit whether a real lightweight detector (the existing
   `UltralyticsDetector` adapter, or a permissively-licensed alternative)
   is worth wiring in before the demo instead of after. Treat this as a
   fallback, not a default plan — the spec explicitly does not require
   maximum detection performance for v1.

## 5. Risks, blockers, and decisions you need to make

- **No demo clips exist yet — this is the hard blocker.** Everything from
  step 2 onward in §4 is stalled until clips are chosen. Decide: source
  (your own recordings vs. licensed stock footage vs. public-domain
  drone footage), how many, and what difficulty spread.
- **Clip storage policy.** `.gitignore` already excludes `*.mp4/*.avi/*.mov`,
  so raw clips won't land in git by default. Decide how the demo operator
  actually gets the clips onto the machine on demo day: a local
  untracked folder with a documented path, a separate cloud link, or
  git-lfs if you want them versioned. This needs a decision before the
  runbook (step 8) can be written concretely.
- **Motion-detector clutter sensitivity is a clip-selection constraint,
  not just a code issue.** If the only available footage has a moving
  camera or a busy background, the current detector will likely produce
  visible false positives live. Decide whether to constrain clip selection
  to avoid this, or invest in a real detector before the demo (§4 step 9).
- **Licensing decision, only if you go down the real-detector path.**
  `UltralyticsDetector` is AGPL-3.0 and needs a commercial license for
  closed-source use (already flagged in `docs/DECISIONS.md` #2). Only
  relevant if step 9 in §4 gets triggered.
- **Confidence score framing.** `MotionDetector`'s "confidence" is a
  synthetic, area-based heuristic, not a calibrated detection probability.
  Decide how (or whether) to explain this if a technical stakeholder asks
  what the percentage means.
- **Streamlit telemetry ping.** By default Streamlit reports anonymous
  usage stats on start ("Collecting usage statistics..."), which implies
  outbound network activity. Decide whether that's acceptable for the
  demo environment or should be disabled
  (`--browser.gatherUsageStats false`) for an offline/controlled room.
- **Live webcam fallback.** Spec marks live camera as optional/secondary.
  Decide now whether it's worth rehearsing at all for this demo, or
  explicitly out of scope, so it doesn't quietly become an unplanned
  fallback live on stage.

## 6. Definition of "done" for the next implementation phase

The next phase is done when all of the following are true:

1. 2–4 real, curated demo clips are in place, with a documented local path
   and a clear note on how to obtain/place them (not necessarily committed
   to git).
2. Running the CLI (`scripts/run_pipeline.py`) against every chosen clip,
   end to end, produces: no crash, no unhandled traceback, a plausible
   detection on the intended target for the majority of frames, and a
   track ID that stays stable for a visually convincing stretch of the
   clip.
3. The Streamlit UI path has been rehearsed at least once, start to
   finish, for at least one clip: upload -> run -> watch overlay, status,
   and event log update live -> see the final JSONL log path reported.
4. Per-clip tuned parameters (if any were needed) are written down
   somewhere reproducible — either in versioned YAML or explicitly in the
   runbook — not left as one-off manual edits.
5. A one-page demo runbook exists and has been followed literally at least
   once by someone re-running it from the doc alone (not from memory).
6. Invalid input (bad file path / corrupt file) fails with a clear,
   one-line operator-facing message on both the CLI and the UI path,
   rather than a raw stack trace.
