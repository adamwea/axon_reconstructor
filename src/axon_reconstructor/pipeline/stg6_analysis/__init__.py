"""Analysis stage.

Generates post-hoc summary artifacts combining outputs from earlier stages.
"""

from __future__ import annotations

from .cross_well import main as run_cross_well_analysis
from .analysis_deck import run_with_args as run_analysis_deck
from .main import ANALYSIS_OUTPUTS_DIRNAME, AnalysisInputs, AnalysisOutputs, analyze_units

__all__ = [
    "ANALYSIS_OUTPUTS_DIRNAME",
    "AnalysisInputs",
    "AnalysisOutputs",
    "analyze_units",
    "run_analysis_deck",
    "run_cross_well_analysis",
]
