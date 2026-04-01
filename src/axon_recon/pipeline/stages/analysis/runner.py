from __future__ import annotations

from .core import run_analysis_stage_core
from .models.inputs import AnalysisInputs
from .models.results import AnalysisResult


def run_analysis_stage(inputs: AnalysisInputs) -> AnalysisResult:
	return run_analysis_stage_core(inputs)
