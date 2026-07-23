"""Pipeline orchestration: video source -> detector -> tracker -> log.

This module intentionally contains zero actuation, targeting, or
engagement logic. It only reads frames, runs detection/tracking, logs
results, and yields structured results for a UI or CLI to render. Nothing
here can command an effector — there is no effector abstraction at all.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator

import cv2

from control.events import FrameResult
from control.state import PipelineState, StateMachine
from control.synthetic_source import DEMO_SOURCE_KEYS, SyntheticVideoSource
from detector.base import BaseDetector
from telemetry.logger import EventLogger
from tracker.base import BaseTracker

_FPS_WINDOW = 30


class Pipeline:
    def __init__(
        self,
        detector: BaseDetector,
        tracker: BaseTracker,
        logger: EventLogger | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.logger = logger
        self.state = StateMachine()

    @staticmethod
    def _open_source(source: str | int):
        """Resolve `source` into a capture-like object (isOpened/read/release/get).

        A source string matching `control.synthetic_source.DEMO_SOURCE_KEYS`
        ("demo"/"synthetic") builds an in-memory synthetic clip instead of
        opening a real video — used for demo mode. Everything else is
        handed to `cv2.VideoCapture` unchanged (file path, RTSP URL, or an
        integer webcam index).
        """
        if isinstance(source, str) and source.strip().lower() in DEMO_SOURCE_KEYS:
            return SyntheticVideoSource()
        return cv2.VideoCapture(source)

    def run(self, source: str | int) -> Iterator[FrameResult]:
        """Process a video source frame by frame, yielding FrameResult.

        `source` is anything `cv2.VideoCapture` accepts (a file path, an
        RTSP URL, or an integer webcam index), or the special value "demo"
        / "synthetic" for the built-in synthetic demo clip.
        """
        capture = self._open_source(source)
        if not capture.isOpened():
            self.state.transition(PipelineState.ERROR)
            if self.logger:
                self.logger.log_event("inference_error", error=f"Could not open source: {source}")
            raise RuntimeError(f"Could not open video source: {source}")

        self.detector.warmup()
        self.state.transition(PipelineState.RUNNING)
        if self.logger:
            self.logger.log_event(
                "run_started",
                source=str(source),
                detector=getattr(self.detector, "name", type(self.detector).__name__),
                tracker=getattr(self.tracker, "name", type(self.tracker).__name__),
            )

        frame_index = 0
        frame_end_times: deque[float] = deque(maxlen=_FPS_WINDOW)
        try:
            while self.state.is_running():
                frame_start = time.perf_counter()

                decode_start = frame_start
                try:
                    ok, frame = capture.read()
                except Exception as exc:  # noqa: BLE001 - a corrupt/unreadable stream ends the run, not the process
                    if self.logger:
                        self.logger.log_event(
                            "inference_error",
                            frame_index=frame_index,
                            error=f"Frame decode failed, stopping run: {exc}",
                        )
                    break
                decode_ms = (time.perf_counter() - decode_start) * 1000.0

                if not ok:
                    break

                detector_ms = 0.0
                tracker_ms = 0.0
                try:
                    detect_start = time.perf_counter()
                    detections = self.detector.detect(frame)
                    detector_ms = (time.perf_counter() - detect_start) * 1000.0

                    track_start = time.perf_counter()
                    tracks = self.tracker.update(detections)
                    tracker_ms = (time.perf_counter() - track_start) * 1000.0

                    dropped = False
                    error = None
                except Exception as exc:  # noqa: BLE001 - we want to keep the loop alive
                    detections = []
                    tracks = []
                    dropped = True
                    error = str(exc)
                    if self.logger:
                        self.logger.log_event(
                            "frame_dropped", frame_index=frame_index, error=error
                        )

                total_ms = (time.perf_counter() - frame_start) * 1000.0
                frame_end_times.append(time.perf_counter())
                fps = 0.0
                if len(frame_end_times) >= 2:
                    span = frame_end_times[-1] - frame_end_times[0]
                    if span > 0:
                        fps = (len(frame_end_times) - 1) / span

                result = FrameResult(
                    frame_index=frame_index,
                    timestamp=time.time(),
                    tracks=tracks,
                    detection_count=len(detections),
                    inference_ms=round(detector_ms + tracker_ms, 2),
                    dropped=dropped,
                    error=error,
                    decode_ms=round(decode_ms, 2),
                    detector_ms=round(detector_ms, 2),
                    tracker_ms=round(tracker_ms, 2),
                    total_ms=round(total_ms, 2),
                    fps=round(fps, 2),
                    frame=frame,
                )

                if self.logger:
                    result_payload = result.as_dict()
                    result_payload.pop("frame_index", None)
                    self.logger.log_event(
                        "frame_processed", frame_index=frame_index, **result_payload
                    )

                yield result
                frame_index += 1
        finally:
            capture.release()
            if self.state.state is PipelineState.RUNNING:
                self.state.transition(PipelineState.STOPPED)
            if self.logger:
                self.logger.log_event("run_stopped", frame_index=frame_index)

    def stop(self) -> None:
        if self.state.is_running():
            self.state.transition(PipelineState.STOPPED)
