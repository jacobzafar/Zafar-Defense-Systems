"""Tests for detector/tiling.py's tile grid (pure geometry)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.tiling import tile_grid


def test_single_tile_is_the_whole_frame():
    assert tile_grid(1920, 1080, 1, 1, 0.2) == [(0, 0, 1920, 1080)]


def test_2x2_with_20_percent_overlap_hand_computed():
    # width 100: tile = 100 / (2 - 0.2) = 55.56, step = 44.44 -> (0, 56), (44, 100)
    # height 50: tile = 27.78, step = 22.22 -> (0, 28), (22, 50)
    assert tile_grid(100, 50, 2, 2, 0.2) == [(0, 0, 56, 28), (44, 0, 100, 28), (0, 22, 56, 50), (44, 22, 100, 50)]


def test_3x3_tiles_cover_the_frame_and_overlap():
    tiles = tile_grid(1920, 1080, 3, 3, 0.2)
    assert len(tiles) == 9
    xs = sorted({(x1, x2) for x1, _, x2, _ in tiles})
    assert xs[0][0] == 0 and xs[-1][1] == 1920
    for (a1, a2), (b1, b2) in zip(xs, xs[1:]):
        assert b1 < a2  # adjacent columns overlap
        assert (a2 - b1) == pytest.approx(0.2 * (a2 - a1), abs=2)


@pytest.mark.parametrize("rows,cols,overlap", [(0, 2, 0.2), (2, 2, 1.0), (2, 2, -0.1)])
def test_invalid_grid_raises(rows, cols, overlap):
    with pytest.raises(ValueError):
        tile_grid(100, 100, rows, cols, overlap)


def test_tiles_for_size_picks_the_fewest_tiles_no_larger_than_the_target():
    from detector.tiling import tile_grid, tiles_for_size

    assert tiles_for_size(640, 640, 0.2) == 1
    assert tiles_for_size(2400, 640, 0.2) == 5
    assert tiles_for_size(1080, 640, 0.2) == 2
    for length in (720, 1080, 1280, 2400):
        n = tiles_for_size(length, 640, 0.2)
        spans = tile_grid(length, 1, 1, n, 0.2)
        assert max(x2 - x1 for x1, _, x2, _ in spans) <= 641
        if n > 1:
            assert length / ((n - 1) - (n - 2) * 0.2) > 640  # one fewer would be too big
