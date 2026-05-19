from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# The cleanup stage's default phase sequence. Single-phase today; slice 6 moved
# `wipe_src_scratch` here from preprocess. The long-term roadmap (per
# `phase_roster_cleanup_plan.md` §6 open question 2 + `tech_debt.md`
# "Finalize the phase roster") is to consolidate the scattered per-stage
# `cleanup_*` phases (spikesort.cleanup_concat_binary, spikesort.cleanup_analyzers,
# reconstruct.clear_templates_cache) into this stage so the user gets a single
# opt-in stage for end-of-run hygiene.
DEFAULT_CLEANUP_PHASE_SEQUENCE: tuple[str, ...] = (
	"wipe_src_scratch",
)


@dataclass(frozen=True)
class CleanupWipeSrcScratchPhaseConfig:
	"""Per-phase config for `cleanup.wipe_src_scratch`.

	Carries the YAML knobs that control wipe enablement, dry-run staging,
	and the scratch-root requirement check. The actual file removal happens
	inside `core/wipe_src_scratch.py`.
	"""

	enabled: bool = False
	dry_run: bool = False
	requires_use_scratch_root: bool = False
	summary_json_relpath: str = "context/wipe_src_scratch_summary.json"
	resource_class: str | None = None


@dataclass(frozen=True)
class CleanupPhasesConfig:
	wipe_src_scratch: CleanupWipeSrcScratchPhaseConfig = field(
		default_factory=CleanupWipeSrcScratchPhaseConfig
	)


@dataclass(frozen=True)
class CleanupInputs:
	"""Per-target inputs for the cleanup stage's phases.

	Mirrors the slim slice of inputs `run_wipe_src_scratch_core` actually
	consumes — h5/source_h5/stream_id for identification, the scratch flag
	for the core payload, the mea_output_root for resolving the summary
	path, plus the standard force_restart + phases bundle.
	"""

	h5_path: Path
	stream_id: str
	mea_output_root: Path
	source_h5_path: Path | None = None
	copied_to_scratch: bool = False
	output_rel_root: str = "cleanup_outputs"
	force_restart: bool = False
	replot: bool = False
	phase_sequence: tuple[str, ...] = DEFAULT_CLEANUP_PHASE_SEQUENCE
	phases: CleanupPhasesConfig = field(default_factory=CleanupPhasesConfig)
