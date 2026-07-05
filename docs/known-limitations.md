# Known limitations

- **Detection is a placeholder.** `MotionDetector` detects moving blobs via
  background subtraction, not drones specifically. It will false-positive
  on any motion (birds, trees, camera shake) and false-negative on
  stationary or very slow-moving aerial objects. Replacing it with a
  trained model (RF-DETR/YOLOX recommended, see `docs/DECISIONS.md`) is the
  clear next step and requires labeled drone data.
- **No re-identification in the default tracker.** `IoUTracker` will assign
  a new ID to a target that was fully occluded for longer than `max_age`
  frames (default 15). Upgrading to `ByteTrackAdapter` mitigates this
  somewhat but still lacks appearance-based re-ID.
- **No camera-motion compensation.** Both the detector and tracker assume a
  roughly static camera. A panning/moving camera will produce a lot of
  spurious motion detections with `MotionDetector`.
- **Streamlit UI is single-run, single-user.** No concurrent sessions,
  no persistence between runs beyond the JSONL logs on disk.
- **No performance/throughput claims are made or implied.** Actual FPS
  depends entirely on hardware and which detector backend is selected; this
  has not been benchmarked in this environment (no GPU, no sample drone
  footage available here).
- **Verified installable and runnable.** `pip install -r requirements.txt`,
  `pytest tests/ -v` (9 passed), `scripts/run_pipeline.py` against a
  synthetic video, and `streamlit run ui/app.py` have all been exercised
  end-to-end. Note: `requirements.txt`/`pyproject.toml` pin
  `opencv-python-headless` rather than `opencv-python` — the GUI build
  requires `libGL.so.1`, which is absent on many servers/containers, and
  nothing in this codebase calls `cv2.imshow`/highgui.
- **No dataset included.** You will need to source or record drone footage
  and, if training a real detector, licensed/labeled data — neither is
  included here.
