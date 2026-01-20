"""Lightweight numerical helpers for template processing.

This module is intentionally dependency-minimal (NumPy only) so it can be
imported and unit-tested without bringing in the heavier template extraction
stack (e.g., spikeinterface).
"""

from __future__ import annotations

import numpy as np


def get_time_derivative(
    merged_template: np.ndarray,
    *,
    unit: str = "seconds",
    sampling_rate: float = 10_000,
    axis: int = 0,
) -> np.ndarray:
    """Compute the discrete time derivative along `axis`.

    Parameters
    ----------
    merged_template:
        Array containing signals. Common convention in this codebase is that
        time is one axis and channels is the other.
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
        `np.diff(merged_template, axis=axis) / delta_t`
    """

    if unit == "seconds":
        delta_t = 1.0 / float(sampling_rate)
    elif unit == "ms":
        delta_t = 1.0 / (float(sampling_rate) / 1000.0)
    else:
        raise ValueError("Invalid unit. Choose either 'seconds' or 'ms'.")

    return np.diff(merged_template, axis=axis) / delta_t
