from __future__ import annotations

from .models.inputs import CleanupInputs
from .phases.wipe_src_scratch import run_cleanup_wipe_src_scratch_phase
from .runner import run_cleanup_stage


def run_cleanup_wipe_src_scratch(inputs: CleanupInputs) -> dict[str, object]:
	"""Single-target entrypoint for the cleanup `wipe_src_scratch` phase.

	Takes a pre-built `CleanupInputs` and returns the phase's summary payload.
	Multi-target / runtime-config-driven invocation flows through
	`pipeline/runner.py:run_cleanup_wipe_src_scratch_from_runtime`.
	"""

	return run_cleanup_wipe_src_scratch_phase(inputs)


__all__ = [
	"run_cleanup_wipe_src_scratch",
	"run_cleanup_stage",
]
