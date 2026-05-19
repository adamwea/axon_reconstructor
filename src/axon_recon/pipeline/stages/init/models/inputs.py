from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# The init stage's default phase sequence. Single-phase today; slice 5 moved
# `copy_src_to_scratch` here from preprocess. Future once-per-data-config setup
# phases (data fetch, globus pulls, etc.) would extend this tuple — that's the
# stage's design intent per `phase_roster_cleanup_plan.md` §1.D.
DEFAULT_INIT_PHASE_SEQUENCE: tuple[str, ...] = (
	"copy_src_to_scratch",
)


@dataclass(frozen=True)
class InitCopySrcToScratchPhaseConfig:
	"""Per-phase config for `init.copy_src_to_scratch`.

	The actual H5 copy flows through `resolve_copy_src_to_scratch_input_path`
	during execution-target selection; this dataclass only carries the YAML
	knobs that affect the phase's summary-writing and enable/skip decisions.
	"""

	enabled: bool = False
	requires_use_scratch_root: bool = False
	summary_json_relpath: str = "context/copy_src_to_scratch_summary.json"
	resource_class: str | None = None


@dataclass(frozen=True)
class InitPhasesConfig:
	copy_src_to_scratch: InitCopySrcToScratchPhaseConfig = field(
		default_factory=InitCopySrcToScratchPhaseConfig
	)


@dataclass(frozen=True)
class InitInputs:
	"""Per-target inputs for the init stage's phases.

	Mirrors the slim slice of `PreprocessInputs` that `run_copy_src_to_scratch_core`
	actually consumes — h5/source_h5/stream_id for identification, the scratch
	flag for the core payload, the mea_output_root for resolving the summary
	path, plus the standard force_restart + phases bundle.
	"""

	h5_path: Path
	stream_id: str
	mea_output_root: Path
	source_h5_path: Path | None = None
	copied_to_scratch: bool = False
	output_rel_root: str = "init_outputs"
	force_restart: bool = False
	replot: bool = False
	phase_sequence: tuple[str, ...] = DEFAULT_INIT_PHASE_SEQUENCE
	phases: InitPhasesConfig = field(default_factory=InitPhasesConfig)
