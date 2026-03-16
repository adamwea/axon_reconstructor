"""Compatibility wrappers for stg1 concatenation utilities.

Canonical ownership has moved to
`MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.runtime`.
"""

from __future__ import annotations

from MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.runtime import (  # noqa: F401
    _process_rec_segment_for_concatenation,
    find_common_electrodes_from_segments,
)

__all__ = ["find_common_electrodes_from_segments", "_process_rec_segment_for_concatenation"]
