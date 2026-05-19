"""Init stage package.

Scaffolded in `phase_roster_cleanup_plan.md` slice 4 with no phases yet. Slice
5 will move `copy_src_to_scratch` here. Until then the stage is disabled by
default and any invocation is a no-op (see `run_init_stage`).
"""

from .config import InitStageConfig, parse_init_stage_config
from .runner import run_init_stage


__all__ = [
	"InitStageConfig",
	"parse_init_stage_config",
	"run_init_stage",
]
