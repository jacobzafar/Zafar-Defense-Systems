# Known limitations

- **Default detection is still a placeholder.** `MotionDetector` (the
  default backend) detects moving blobs via background subtraction, not
  drones specifically. It will false-positive on any motion (birds, trees,
  camera shake) and false-negative on stationary or very slow-moving
  aerial objects.
- **The new `torchvision` detector backend is a real object detector, but
  not a drone classifier either.** It's a pretrained SSDLite MobileNetV3
  model trained on COCO, which has no "drone" class — it's configured to
  keep `airplane`, `bird`, and `kite` detections as visual proxies for a
  small aerial target. This will still miss actual drones that don't
  resemble those classes, and may fire on real birds/kites/planes in
  frame. Training or fine-tuning on real drone data (or licensing a
  drone-specific model) remains the clear next step for detection
  accuracy; see `docs/DECISIONS.md` entry #6.
- **No re-identification in the default tracker.** `IoUTracker` will assign
  a new ID to a target that was fully occluded for longer than `max_age`
  frames (default 15). Upgrading to `ByteTrackAdapter` mitigates this
  somewhat but still lacks appearance-based re-ID.
- **No camera-motion compensation.** Both the detector and tracker assume a
  roughly static camera. A panning/moving camera will produce a lot of
  spurious motion detections with `MotionDetector`.
- **Streamlit UI is single-run, single-user, and has no true live "Stop"
  control.** No concurrent sessions, no persistence between runs beyond
  the JSONL logs on disk. A run is one blocking loop that live-updates
  the page as it goes — there is no background thread for a button click
  to interrupt mid-run. Use the sidebar's "Max frames" limiter to bound a
  run instead of relying on a stop button.
- **The UI's webcam option opens a camera on the machine running the
  Streamlit server, not the viewer's browser.** True browser-side webcam
  capture would need an additional dependency (e.g. `streamlit-webrtc`)
  that isn't part of this repo.
- **The built-in synthetic demo source (`control/synthetic_source.py`,
  `--source demo` / the UI's "Demo mode") proves the pipeline runs
  end-to-end, not that detection/tracking works on real footage.** Its
  moving shapes are easy for `MotionDetector` to pick up; treat a clean
  demo-mode run as a software-plumbing check, not a detection-quality
  result.
- **No performance/throughput claims are made or implied for real demo
  footage.** Actual FPS depends entirely on hardware, resolution, and
  which detector backend is selected. Data points measured in this
  development environment: the `torchvision` backend at ~58ms/frame
  (~17 FPS) on a 640x480 frame on CPU; the `motion` backend at well over
  100 FPS on small synthetic/test clips. Both are informative for *this*
  environment only, not a guarantee for other hardware or resolutions —
  the UI and CLI now surface a live/measured FPS per run so this can be
  checked directly on your own footage instead of taken on faith. No real
  drone footage has been used to benchmark accuracy anywhere in this
  repo's history.
- **Verified installable and runnable.** `pip install -r requirements.txt`,
  `pytest tests/ -v` (49 passed with the optional `torch`/`torchvision`
  extra installed, 1 skipped without it), `scripts/run_pipeline.py`
  against both a real (synthetic-file) video and the built-in demo
  source, `make demo`, `scripts/summarize_log.py`, and the Streamlit UI
  (booted, and driven end-to-end via Streamlit's own `AppTest` harness —
  initial load, preset switching, a full demo run, reset, and both error
  paths) have all been exercised in this pass. Note:
  `requirements.txt`/`pyproject.toml` pin `opencv-python-headless` rather
  than `opencv-python` — the GUI build requires `libGL.so.1`, which is
  absent on many servers/containers, and nothing in this codebase calls
  `cv2.imshow`/highgui.
- **No dataset included.** You will need to source or record drone footage
  and, if training a real detector, licensed/labeled data — neither is
  included here.
- **The `drone` detector backend (`detector/drone_detector.py`) has no
  trained weights and will refuse to run until you supply some.** This is
  intentional infrastructure (see `docs/DECISIONS.md` entry #9), not a
  finished model: `detector/datasets/` has a unified schema and a
  registry of four named public datasets (Anti-UAV, DUT Anti-UAV,
  Drone-vs-Bird, VisioDECT), but every entry's `source_url` and
  `local_path` are unset — no data has been downloaded, and every
  dataset's `license` field is marked `UNVERIFIED` (not asserted as any
  specific SPDX license) pending a maintainer confirming current terms
  from the official source. `detector/train.py` has only ever been run
  in this repo against a 6-image synthetic smoke-test fixture
  (`tests/test_train.py`) — that proves the fine-tuning loop itself
  works, not that the resulting model detects anything real. Blocked on:
  real, license-checked drone imagery being provided.
- **No real accuracy numbers exist for any backend.** `eval/` (see
  `docs/DECISIONS.md` entry #10) is a real, tested evaluation harness —
  AP@0.5, small-object recall, false-alarm rate, latency, track
  continuity — but no real eval set ships with this repo, only a tiny
  8-frame synthetic fixture used purely to test the harness's own
  mechanics (`tests/test_eval_harness.py`). Any metric card or
  `eval/REPORT.md` you see in this environment was generated from that
  synthetic fixture, not real drone footage — `eval/output/` is
  gitignored specifically so a stale/synthetic result can never look like
  a current, validated one. Blocked on: a real, frozen, license-checked
  eval set (see `eval/README.md` for the required layout) being provided.
- **No coverage data exists yet.** `docs/coverage-matrix.md` is an empty
  template (drone type × lighting/weather × background × range/angle/
  speed × EO/IR) — every checkbox is unchecked and its scenario log has
  no rows. `tools/preannotate.py` (auto-annotation for human review) has
  only been run against a synthetic test video, never real footage.
  Blocked on: real footage being collected/sourced and run through it.
