"""Small, dependency-minimal signal processing helpers.

This module intentionally stays NumPy-only so it can be imported and tested
without pulling in the heavier pipeline stack.
"""

from __future__ import annotations

import numpy as np


def get_time_derivative(
    x: np.ndarray,
    *,
    unit: str = "seconds",
    sampling_rate: float = 10_000,
    axis: int = 0,
) -> np.ndarray:
    """Compute the discrete time derivative along `axis`.

    Parameters
    ----------
    x:
        Array containing signals.
    unit:
        Either "seconds" or "ms". Controls the delta used in the finite
        difference.
    sampling_rate:
        Samples per second.
    axis:
        Axis along which the time derivative is computed.

    Returns
    -------
    np.ndarray
        `np.diff(x, axis=axis) / delta_t`
    """

    if unit == "seconds":
        delta_t = 1.0 / float(sampling_rate)
    elif unit == "ms":
        delta_t = 1.0 / (float(sampling_rate) / 1000.0)
    else:
        raise ValueError("Invalid unit. Choose either 'seconds' or 'ms'.")

    return np.diff(x, axis=axis) / delta_t
