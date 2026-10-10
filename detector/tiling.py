"""Sliced ("tiled") inference helpers for small-object detection.

SSDLite320 resizes every input to 320x320, so a ~30px drone in a
1920x1080 frame shrinks to roughly 5x9px before the network sees it.
Running the same detector on overlapping crops of the frame and mapping
the detections back (SAHI-style slicing) gives each small object several
times more input pixels without retraining. See docs/DECISIONS.md #21.
"""

from __future__ import annotations


def tile_grid(width: int, height: int, rows: int, cols: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return `rows*cols` (x1, y1, x2, y2) crops covering a width x height
    frame, adjacent tiles overlapping by `overlap` (a fraction of the tile
    size, in [0, 1)) so an object on a seam appears whole in at least one
    tile. The last row/column ends exactly at the frame edge."""
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1, got {rows}x{cols}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    def spans(length: int, count: int) -> list[tuple[int, int]]:
        if count == 1:
            return [(0, length)]
        tile = length / (count - (count - 1) * overlap)
        step = tile * (1.0 - overlap)
        result = []
        for i in range(count):
            start = int(round(i * step))
            end = length if i == count - 1 else int(round(i * step + tile))
            result.append((start, end))
        return result

    return [(x1, y1, x2, y2) for (y1, y2) in spans(height, rows) for (x1, x2) in spans(width, cols)]
