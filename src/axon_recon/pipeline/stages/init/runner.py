from __future__ import annotations

import logging
from typing import Any

from ...execution.results import MultiTargetStageResult, TargetStageResult

from .config import InitStageConfig


LOGGER = logging.getLogger("axon_recon.init.runner")


def run_init_stage(
	stage_config: InitStageConfig,
	*,
	targets: list[Any] | None = None,
) -> MultiTargetStageResult:
	"""Run the init stage.

	Slice 4 scaffolds this stage with no phases. When the stage is disabled
	OR its `phase_sequence` is empty (the defaults today), the runner is a
	clean no-op that returns an empty `MultiTargetStageResult`. Once slice 5
	moves `copy_src_to_scratch` here the `NotImplementedError` branch will
	be replaced with real per-phase wiring.
	"""

	del targets  # currently unused; reserved for slice 5 when a real phase moves in

	if not stage_config.enabled or not stage_config.phase_sequence:
		LOGGER.debug(
			"init stage no-op enabled=%s phase_sequence=%s",
			bool(stage_config.enabled),
			tuple(stage_config.phase_sequence),
		)
		return MultiTargetStageResult(
			stage="init",
			total_targets=0,
			succeeded_targets=0,
			failed_targets=0,
			target_results=[],
		)

	# Unreachable until slice 5 moves `copy_src_to_scratch` in. Raising here
	# (rather than silently returning a no-op) protects against an operator
	# enabling the stage via YAML before a phase has been registered: the
	# error makes the half-wired state visible instead of silently passing.
	raise NotImplementedError("init stage has no phases registered yet")


# Re-export the result type so callers can compose against this module without
# pulling in the execution.results path explicitly. Kept here rather than in
# __init__ so the import surface stays tight.
__all__ = ["MultiTargetStageResult", "TargetStageResult", "run_init_stage"]
