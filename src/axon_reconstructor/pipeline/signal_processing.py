"""Public pipeline API for lightweight signal processing helpers.

Historically, template processing helpers lived in various internal modules.
Tests and pipeline code import these via `axon_reconstructor.pipeline.signal_processing`.

We keep this module dependency-minimal (NumPy only) and avoid importing from the
deprecated internal bundle.
"""

from __future__ import annotations

from typing import Any


def get_time_derivative(
    merged_template: "Any",
    *,
    unit: str = "seconds",
    sampling_rate: float = 10_000,
    axis: int = 0,
) -> "Any":
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

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required for get_time_derivative") from e

    if unit == "seconds":
        delta_t = 1.0 / float(sampling_rate)
    elif unit == "ms":
        delta_t = 1.0 / (float(sampling_rate) / 1000.0)
    else:
        raise ValueError("Invalid unit. Choose either 'seconds' or 'ms'.")

    return np.diff(merged_template, axis=axis) / delta_t
