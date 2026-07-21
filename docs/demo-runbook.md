# Demo Runbook

Exact steps for a clean, repeatable, end-to-end demo of the operator
console — plus the specific failure modes this app is known to have and
how to avoid hitting them live. Written from what has actually been
verified about this app's behavior (see `docs/DECISIONS.md`), not
aspirational.

## Before demo day (once, when preparing)

1. Fresh install check:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python -m pytest tests/ -v
   ```
   All tests should pass (one module — `tests/test_torchvision_detector.py`
   — skips if you haven't installed the optional `torch`/`torchvision`
   extra; that's expected, not a failure).
2. Run the license audit if anything in `requirements.txt`/`pyproject.toml`
   changed since the last demo: `python scripts/license_audit.py`.
3. Decide which detector backend you're demoing (see "Choosing a backend"
   below) and install its optional dependency now, not day-of:
   ```bash
   pip install torch torchvision   # only if demoing --detector torchvision
   ```
4. If demoing on a real video file: confirm it opens cleanly ahead of
   time —
   ```bash
   python scripts/run_pipeline.py --source path/to/your_clip.mp4 --max-frames 5
   ```
   If this fails, fix it now (re-encode to H.264 MP4 is the most common
   fix — see "Known failure modes" below), not live.
5. Do one full dry run of the exact demo script below, start to finish,
   on the actual machine you'll present from.

## Choosing a backend

- **Safest default: `motion` detector, `demo` or `default` preset.** Zero
  extra dependencies, always available, fastest to set up. Downside: it's
  a motion-blob detector, not a drone classifier — pick clips with a
  static-ish camera and a clean background so it looks credible (see
  `docs/known-limitations.md`).
- **`torchvision` backend**: a real object detector, needs `torch`/
  `torchvision` installed ahead of time (see above). Slower per-frame
  (~50-100ms on CPU) — use `--max-frames`/the UI's "Max frames" limiter
  so a demo run has a predictable, short runtime.
- **`drone` backend**: only demo this if you have actually trained
  weights via `detector/train.py` on real data. If you don't, it will
  fail immediately with "No trained drone-detector weights found" — do
  not select it live on a hope.
- **`ultralytics` backend**: AGPL-3.0 — the UI now shows a licensing
  warning when you pick it. Only demo this if your audience needs to see
  it specifically and the licensing conversation is welcome; otherwise
  avoid it to keep the demo focused.

## The demo script (~5-10 minutes)

1. **Start clean.** Close any old Streamlit browser tabs from a previous
   run (stale JS state can look like nothing is happening). Start fresh:
   ```bash
   source .venv/bin/activate
   streamlit run ui/app.py
   ```
   Open the URL it prints in a fresh browser tab.
2. **Narrate the header.** Point out the state badge (should read
   `READY` — Demo mode is selected by default), the preset/detector/
   tracker badges, and the subtitle: "observation-only... no jamming, no
   kinetic effects, no autonomous engagement." This framing matters —
   say it before anyone asks.
3. **Set a frame limit before you press anything.** In the sidebar, set
   "Max frames" to something short and predictable — 100-150 is a good
   ~2-5 second demo run. **There is no live Stop button** (see "Known
   failure modes" below) — the frame limit is how you control run length.
4. **Press "Start run"** with Demo mode still selected. Narrate live as
   it runs: the KPI row updating, the annotated video feed, the event log
   showing `FRAME` lines and `ACQUIRED`/`LOST` target events.
5. **Let it finish** (it will, on its own, once it hits the frame limit
   or the clip ends). Show the "Run summary" panel and the telemetry
   download button — this is the JSONL log, one line per event, real
   audit trail.
6. **If demoing real footage**, switch the sidebar to "Upload video",
   choose your pre-verified clip (see prep step 4), set a sane frame
   limit if the clip is long, and repeat steps 4-5.
7. **Close on the boundary, not just the features.** Repeat: no jamming,
   no weaponization, no autonomous engagement, no effectors anywhere in
   this codebase — point at `README.md`'s opening lines if asked to prove
   it.

## Known failure modes (and how to avoid them)

- **No live "Stop" button.** A run is one blocking loop; there is no way
  to interrupt it mid-run from the UI. Always set "Max frames" before
  pressing Start — don't rely on being able to stop a long/unlimited run
  if it goes on too long. (See `docs/known-limitations.md`.)
- **Selecting `torchvision`/`ultralytics`/`drone` without the dependency
  installed** fails immediately with a friendly "missing optional
  dependency" error — not a crash, but also not a good look mid-demo.
  Install ahead of time (prep step 3), or just don't select those
  backends live if you haven't.
- **Selecting `drone` without trained weights** fails immediately with
  "No trained drone-detector weights found" — this backend has no
  weights in this repo by default (see `docs/DECISIONS.md` entry #9).
  Don't select it unless you've actually trained something.
- **Uploading a corrupt or unusually-encoded video file** surfaces as a
  clear "could not be opened or read" error rather than a crash, but
  it's still a bad live moment. Pre-verify every real clip you plan to
  use (prep step 4). If a clip fails, re-encoding to H.264 MP4 is the
  most common fix:
  ```bash
  ffmpeg -i input.mov -c:v libx264 -pix_fmt yuv420p output.mp4
  ```
- **Streamlit's default upload size limit (200MB)** will reject larger
  files with its own error. Trim demo clips to a short, focused segment
  well under that.
- **The "Webcam (server-side)" option opens a camera on the machine
  running Streamlit, not the presenter's laptop camera via the browser.**
  In most demo environments (cloud sandbox, remote server) there is no
  camera attached at all, and selecting this will fail with a
  webcam-open error. Don't select it unless you've confirmed a real
  camera is attached to the exact machine running the server.
- **Logs accumulate across runs** (`logs/`, gitignored). Not a failure
  mode, but if you've run many rehearsals, consider clearing old ones
  before a real demo so a fresh telemetry download doesn't confuse you
  with an old run's file.
- **A stale browser tab from a previous code change** can show an
  outdated layout or silently-broken widget state. Always open a fresh
  tab (or hard-refresh) right before presenting.

## If something breaks live

1. Stay calm — narrate what you're doing, don't go silent.
2. If the app shows its own error panel (red `ERROR` badge): read the
   message aloud, it's designed to be clear. Click **Reset** in the
   sidebar, switch back to Demo mode, and continue — this is the fastest
   recovery path and doubles as a demonstration that the app fails
   gracefully instead of crashing.
3. If the whole Streamlit process appears to have hung: it's still a
   single-request-at-a-time architecture — a very long unlimited run
   without a frame limit is the most likely cause. Ctrl+C the terminal
   and restart with `streamlit run ui/app.py`; this is exactly why prep
   step 5 (a full dry run) and always setting "Max frames" matter.
4. Fall back to the headless CLI if the UI itself is the problem — it
   exercises the identical pipeline and is simpler to reason about live:
   ```bash
   python scripts/run_pipeline.py --source demo --preset demo --max-frames 100
   ```

## After the demo

- `python scripts/summarize_log.py logs/<the run you want to show>.jsonl`
  if someone asks for a written summary afterward.
- Clean up `logs/` if it's grown large across rehearsals — it's
  gitignored, so this is just local housekeeping, not a repo change.
