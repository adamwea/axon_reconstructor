from __future__ import annotations

from .models.inputs import InitInputs
from .phases.copy_src_to_scratch import run_init_copy_src_to_scratch_phase
from .runner import run_init_stage


def run_init_copy_src_to_scratch(inputs: InitInputs) -> dict[str, object]:
	"""Single-target entrypoint for the init `copy_src_to_scratch` phase.

	Takes a pre-built `InitInputs` and returns the phase's summary payload.
	Multi-target / runtime-config-driven invocation flows through
	`pipeline/runner.py:run_init_copy_src_to_scratch_from_runtime`.
	"""

	return run_init_copy_src_to_scratch_phase(inputs)


__all__ = [
	"run_init_copy_src_to_scratch",
	"run_init_stage",
]
