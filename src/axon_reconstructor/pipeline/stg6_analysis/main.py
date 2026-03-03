"""Analysis step public API.

This stage builds convenient summary artifacts for quick inspection.

Current focus: per-unit summary grids that combine reconstruction + templates + waveforms visuals.
"""

from __future__ import annotations

from .constants import ANALYSIS_OUTPUTS_DIRNAME
from .runner import AnalysisInputs, AnalysisOutputs, analyze_units

__all__ = [
    "ANALYSIS_OUTPUTS_DIRNAME",
    "AnalysisInputs",
    "AnalysisOutputs",
    "analyze_units",
]
