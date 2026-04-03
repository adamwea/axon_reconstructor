from __future__ import annotations

from .models.inputs import PreprocessInputs
from .models.results import PreprocessResult
from .runner import run_preprocess_stage


def run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
	return run_preprocess_stage(inputs)
