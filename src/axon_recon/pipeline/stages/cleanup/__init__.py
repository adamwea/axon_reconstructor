"""Cleanup stage package.

Holds end-of-run cleanup phases. Slice 6 of `phase_roster_cleanup_plan` moved
`wipe_src_scratch` here from the preprocess stage so its on-disk semantics
are owned by a dedicated stage with its own force-restart rmtree scope, its
own opt-in toggle, and its own per-target idempotence guarantees.

Long-term roadmap (per `phase_roster_cleanup_plan.md` slice 6 open question 2
and `tech_debt.md` §"Finalize the phase roster"): the scattered per-stage
`cleanup_*` phases (`spikesort.cleanup_concat_binary`,
`spikesort.cleanup_analyzers`, `reconstruct.clear_templates_cache`) will be
consolidated into this stage so a user enabling end-of-run hygiene only has
one toggle to flip.
"""

from .api import run_cleanup_stage, run_cleanup_wipe_src_scratch
from .config import CleanupStageConfig, parse_cleanup_stage_config
from .models.inputs import (
	DEFAULT_CLEANUP_PHASE_SEQUENCE,
	CleanupInputs,
	CleanupPhasesConfig,
	CleanupWipeSrcScratchPhaseConfig,
)


__all__ = [
	"DEFAULT_CLEANUP_PHASE_SEQUENCE",
	"CleanupInputs",
	"CleanupPhasesConfig",
	"CleanupStageConfig",
	"CleanupWipeSrcScratchPhaseConfig",
	"parse_cleanup_stage_config",
	"run_cleanup_stage",
	"run_cleanup_wipe_src_scratch",
]
