from __future__ import annotations

from .models.inputs import ReconstructionInputs
from .models.results import ReconstructionResult
from .runner import run_reconstruct_stage


def run_reconstruct(inputs: ReconstructionInputs) -> ReconstructionResult:
	return run_reconstruct_stage(inputs)

