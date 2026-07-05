"""Optional adapter for Roboflow's Apache-2.0 `trackers` package (ByteTrack).

Import is guarded so the pipeline works without this dependency installed.
This is the recommended upgrade path from IoUTracker once re-identification
across brief occlusions matters (see docs/DECISIONS.md entry #3 for why
Roboflow `trackers` was chosen over the original FoundationVision/ByteTrack
repo — license clarity).

Install with: pip install trackers supervision
"""

from __future__ import annotations

from detector.base import Detection
from tracker.base import BaseTracker, Track

try:
    import supervision as sv  # type: ignore
    from trackers import ByteTrackTracker  # type: ignore

    _TRACKERS_AVAILABLE = True
except ImportError:
    _TRACKERS_AVAILABLE = False


class ByteTrackAdapter(BaseTracker):
    name = "bytetrack_roboflow"

    def __init__(self) -> None:
        if not _TRACKERS_AVAILABLE:
            raise ImportError(
                "The 'trackers' and 'supervision' packages are not installed. "
                "Install them with `pip install trackers supervision` to use "
                "ByteTrackAdapter."
            )
        self._tracker = ByteTrackTracker()

    def reset(self) -> None:
        self._tracker = ByteTrackTracker()

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            sv_detections = sv.Detections.empty()
        else:
            xyxy = [[d.x1, d.y1, d.x2, d.y2] for d in detections]
            confidence = [d.confidence for d in detections]
            sv_detections = sv.Detections(
                xyxy=__import__("numpy").array(xyxy),
                confidence=__import__("numpy").array(confidence),
            )

        tracked = self._tracker.update(sv_detections)

        tracks: list[Track] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            class_name = (
                detections[0].class_name if detections else "unknown"
            )
            tracks.append(
                Track(
                    track_id=tracker_id,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=conf,
                    class_name=class_name,
                    age_frames=-1,
                    hits=-1,
                    time_since_update=0,
                )
            )
        return tracks
