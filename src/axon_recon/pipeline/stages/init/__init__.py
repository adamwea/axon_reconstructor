"""Init stage package.

Holds once-per-data-config setup phases. Slice 5 of `phase_roster_cleanup_plan`
moved `copy_src_to_scratch` here from the preprocess stage so iteration on
preprocess does not pay the copy phase's overhead on every run.
"""

from .api import run_init_copy_src_to_scratch, run_init_stage
from .config import InitStageConfig, parse_init_stage_config
from .models.inputs import (
	DEFAULT_INIT_PHASE_SEQUENCE,
	InitCopySrcToScratchPhaseConfig,
	InitInputs,
	InitPhasesConfig,
)


__all__ = [
	"DEFAULT_INIT_PHASE_SEQUENCE",
	"InitCopySrcToScratchPhaseConfig",
	"InitInputs",
	"InitPhasesConfig",
	"InitStageConfig",
	"parse_init_stage_config",
	"run_init_copy_src_to_scratch",
	"run_init_stage",
]
