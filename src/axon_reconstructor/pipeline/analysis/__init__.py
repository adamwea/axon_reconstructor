"""Analysis stage.

Generates post-hoc summary artifacts combining outputs from earlier stages.
"""

from __future__ import annotations

from .cross_well import main as run_cross_well_analysis
from .main import ANALYSIS_OUTPUTS_DIRNAME, AnalysisInputs, AnalysisOutputs, analyze_units

__all__ = [
    "ANALYSIS_OUTPUTS_DIRNAME",
    "AnalysisInputs",
    "AnalysisOutputs",
    "analyze_units",
    "run_cross_well_analysis",
]
