"""Pipeline orchestration: video source -> detector -> tracker -> log.

This module intentionally contains zero actuation, targeting, or
engagement logic. It only reads frames, runs detection/tracking, logs
results, and yields structured results for a UI or CLI to render. Nothing
here can command an effector — there is no effector abstraction at all.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from control.events import FrameResult
from control.state import PipelineState, StateMachine
from detector.base import BaseDetector
from telemetry.logger import EventLogger
from tracker.base import BaseTracker


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

    def run(self, source: str | int) -> Iterator[FrameResult]:
        """Process a video source frame by frame, yielding FrameResult.

        `source` is anything cv2.VideoCapture accepts: a file path, an RTSP
        URL, or an integer webcam index.
        """
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            self.state.transition(PipelineState.ERROR)
            if self.logger:
                self.logger.log_event("inference_error", error=f"Could not open source: {source}")
            raise RuntimeError(f"Could not open video source: {source}")

        self.detector.warmup()
        self.state.transition(PipelineState.RUNNING)
        if self.logger:
            self.logger.log_event("run_started", source=str(source))

        frame_index = 0
        try:
            while self.state.is_running():
                ok, frame = capture.read()
                if not ok:
                    break

                start = time.perf_counter()
                try:
                    detections = self.detector.detect(frame)
                    tracks = self.tracker.update(detections)
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

                inference_ms = (time.perf_counter() - start) * 1000.0

                result = FrameResult(
                    frame_index=frame_index,
                    timestamp=time.time(),
                    tracks=tracks,
                    detection_count=len(detections),
                    inference_ms=round(inference_ms, 2),
                    dropped=dropped,
                    error=error,
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
