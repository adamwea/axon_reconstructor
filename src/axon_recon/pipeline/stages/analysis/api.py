from __future__ import annotations

from .models.inputs import AnalysisInputs
from .models.results import AnalysisResult
from .runner import run_analysis_stage


def run_analysis(inputs: AnalysisInputs) -> AnalysisResult:
	return run_analysis_stage(inputs)
