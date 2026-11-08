import numpy as np
import pytest

from detector.input_norm import UnsupportedFrameError, to_bgr8


def test_bgr8_frame_is_returned_unchanged_as_the_same_object():
    frame = np.zeros((4, 5, 3), np.uint8)
    assert to_bgr8(frame) is frame


@pytest.mark.parametrize("shape", [(4, 5), (4, 5, 1)])
def test_gray_frames_become_three_identical_channels(shape):
    frame = np.arange(20, dtype=np.uint8).reshape(shape)
    out = to_bgr8(frame)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8
    assert np.array_equal(out[..., 0], out[..., 2])


def test_bgra_drops_alpha():
    frame = np.zeros((2, 2, 4), np.uint8)
    frame[..., 0], frame[..., 3] = 7, 255
    out = to_bgr8(frame)
    assert out.shape == (2, 2, 3) and out[0, 0, 0] == 7


def test_16bit_thermal_is_stretched_between_percentiles():
    frame = np.linspace(30000, 31000, 10000, dtype=np.uint16).reshape(100, 100)
    out = to_bgr8(frame)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_flat_and_invalid_frames():
    assert to_bgr8(np.full((3, 3), 5.0, np.float32)).max() == 0
    with pytest.raises(UnsupportedFrameError):
        to_bgr8(np.zeros((3, 3, 2), np.uint8))
    with pytest.raises(UnsupportedFrameError):
        to_bgr8(np.zeros((0, 3), np.uint8))
