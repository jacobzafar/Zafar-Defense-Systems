"""Dependency-free greedy IoU tracker.

Simple by design: matches each frame's detections to existing tracks by
bounding-box IoU (greedy, highest-IoU-first), ages out tracks that go
unmatched for too long, and creates new tracks for unmatched detections.

This has no re-identification / appearance model, so it will lose the ID
of a target that is fully occluded for more than `max_age` frames. That's
an accepted limitation for the MVP (see docs/known-limitations.md); the
same BaseTracker interface can later be swapped for ByteTrack/BoT-SORT
without touching control/pipeline.py.
"""

from __future__ import annotations

from detector.base import Detection
from tracker.base import BaseTracker, Track


def _iou(a: Detection, b: Track) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class IoUTracker(BaseTracker):
    name = "iou_v1"

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits_to_confirm: int = 1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits_to_confirm = min_hits_to_confirm
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched_detections = list(range(len(detections)))
        unmatched_track_ids = list(self._tracks.keys())

        # Greedy matching: compute all pairs, sort by IoU descending, assign.
        candidate_pairs: list[tuple[float, int, int]] = []
        for d_idx in unmatched_detections:
            for t_id in unmatched_track_ids:
                iou = _iou(detections[d_idx], self._tracks[t_id])
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, d_idx, t_id))
        candidate_pairs.sort(key=lambda p: p[0], reverse=True)

        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()

        for _iou_score, d_idx, t_id in candidate_pairs:
            if d_idx in matched_detections or t_id in matched_tracks:
                continue
            matched_detections.add(d_idx)
            matched_tracks.add(t_id)

            det = detections[d_idx]
            track = self._tracks[t_id]
            track.x1, track.y1, track.x2, track.y2 = det.x1, det.y1, det.x2, det.y2
            track.confidence = det.confidence
            track.class_name = det.class_name
            track.age_frames += 1
            track.hits += 1
            track.time_since_update = 0

        # New tracks for unmatched detections.
        for d_idx in unmatched_detections:
            if d_idx in matched_detections:
                continue
            det = detections[d_idx]
            track = Track(
                track_id=self._next_id,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                confidence=det.confidence,
                class_name=det.class_name,
                age_frames=1,
                hits=1,
                time_since_update=0,
            )
            self._tracks[self._next_id] = track
            self._next_id += 1

        # Age unmatched tracks; drop ones that exceeded max_age.
        for t_id in unmatched_track_ids:
            if t_id in matched_tracks:
                continue
            track = self._tracks[t_id]
            track.age_frames += 1
            track.time_since_update += 1

        self._tracks = {
            t_id: t for t_id, t in self._tracks.items() if t.time_since_update <= self.max_age
        }

        return [t for t in self._tracks.values() if t.hits >= self.min_hits_to_confirm]
