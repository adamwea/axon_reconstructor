"""Compatibility wrappers for stg1 plugin discovery utility.

Canonical ownership has moved to
`MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.runtime`.
"""

from __future__ import annotations

from MEA_Analysis.IPNAnalysis.multiseg_utils.preprocess_multiseg_h5.runtime import _ensure_maxwell_hdf5_plugin_path  # noqa: F401

__all__ = ["_ensure_maxwell_hdf5_plugin_path"]
