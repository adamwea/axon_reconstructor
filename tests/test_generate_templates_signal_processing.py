import numpy as np
import pytest

from axon_reconstructor.pipeline.signal_processing import get_time_derivative


def test_get_time_derivative_seconds_axis0():
    x = np.array([0.0, 1.0, 3.0, 6.0], dtype=float)
    # sampling_rate=1 Hz -> delta_t = 1 second
    d = get_time_derivative(x, sampling_rate=1, unit="seconds", axis=0)
    assert np.allclose(d, np.array([1.0, 2.0, 3.0]))


def test_get_time_derivative_ms_axis1_2d():
    # 2 channels x 4 samples; differentiate along time axis (axis=1)
    x = np.array([[0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 2.0, 2.0]], dtype=float)
    # sampling_rate=1000 Hz -> delta_t = 1 ms when unit="ms"
    d = get_time_derivative(x, sampling_rate=1000, unit="ms", axis=1)
    assert d.shape == (2, 3)
    # Values are per-millisecond (not per-second).
    assert np.allclose(d[0], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(d[1], np.array([2.0, 0.0, 0.0]))


def test_get_time_derivative_invalid_unit():
    x = np.zeros((3, 3), dtype=float)
    with pytest.raises(ValueError, match="Invalid unit"):
        get_time_derivative(x, unit="minutes")
