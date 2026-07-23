"""Drawing helpers: render tracks onto a frame for display."""

from __future__ import annotations

import cv2

from tracker.base import Track

_BOX_COLOR = (66, 145, 255)  # BGR for #ff9142 — the console's one detection/alert accent (ui/theme.py's --zds-accent)
_TEXT_COLOR = (14, 17, 21)  # dark text reads cleanly on the accent-colored label background


def draw_tracks(frame, tracks: list[Track]):
    """Draw bounding boxes, track IDs, and confidence onto a copy of frame."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for track in tracks:
        x1 = int(track.x1 * width)
        y1 = int(track.y1 * height)
        x2 = int(track.x2 * width)
        y2 = int(track.y2 * height)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)

        label = f"ID {track.track_id} | {track.class_name} | {track.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Anchor the label to the box's left edge, but keep it clamped inside
        # the frame — a track near the right edge would otherwise have its
        # confidence value drawn past the frame boundary and clipped off.
        label_x1 = max(0, min(x1, width - text_w - 4))
        cv2.rectangle(
            annotated, (label_x1, max(0, y1 - text_h - 8)), (label_x1 + text_w + 4, y1), _BOX_COLOR, -1
        )
        cv2.putText(
            annotated,
            label,
            (label_x1 + 2, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return annotated
