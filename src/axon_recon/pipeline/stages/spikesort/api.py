from __future__ import annotations

from .models.inputs import SpikesortInputs
from .models.results import SpikesortResult
from .runner import run_spikesort_stage


def run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
	return run_spikesort_stage(inputs)
