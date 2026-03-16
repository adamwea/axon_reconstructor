"""Compatibility wrappers for stg1 runtime.

Canonical runtime ownership has moved to
`MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.runtime`.
"""

from __future__ import annotations

from MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.runtime import (  # noqa: F401
    build_concatenated_recording,
)

__all__ = ["build_concatenated_recording"]
