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
- **Streamlit UI is single-run, single-user.** No concurrent sessions,
  no persistence between runs beyond the JSONL logs on disk.
- **No performance/throughput claims are made or implied for real demo
  footage.** Actual FPS depends entirely on hardware, resolution, and
  which detector backend is selected. One data point: the `torchvision`
  backend measured ~58ms/frame (~17 FPS) on a 640x480 frame on CPU in this
  development environment — informative, not a guarantee for other
  hardware or resolutions. No real drone footage has been used to
  benchmark accuracy in this environment.
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
