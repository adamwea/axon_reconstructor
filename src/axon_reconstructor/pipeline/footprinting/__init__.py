"""Footprinting step.

This package computes footprint (peak-to-peak) maps from per-unit templates and
writes curated/uncurated PDFs and JSON summaries.

Primary entry points are re-exported from .main for compatibility.
"""

from __future__ import annotations

from .main import FootprintingInputs, FootprintingOutputs, run_footprinting

__all__ = [
    "FootprintingInputs",
    "FootprintingOutputs",
    "run_footprinting",
]
