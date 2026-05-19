"""analysis.unitmatch phase orchestrator (scaffold).

Slice 1 of `unitmatch_phase_plan.md`: wires the analysis-stage
`unitmatch` phase into the pipeline but does no real work. When the
phase runs (`enabled: true` in YAML), it returns a scaffold marker
documenting that the slice-2+ logic is not yet implemented. When
`enabled: false`, the orchestrator returns a noop marker.

Subsequent slices fill in:
- Slice 2: chip-well group discovery + path resolution.
- Slice 3: `unitlink.match(...)` invocation + output landing.
- Slice 4: `--targets` → chip-well group mapping.
- Slice 5: enable in YAML + smoke run on real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .compute_metrics import (
	_print_analysis_aggregate,
	_target_datasets_override_from_args,
)


def run_analysis_unitmatch(
	*,
	dataset_index: int,
	dataset_id: str | None,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> dict[str, Any]:
	"""Per-target entry point for the analysis.unitmatch phase (scaffold).

	Returns ``{"status": "noop", "phase": "unitmatch", ...}`` when the
	phase is disabled in YAML or when slice-1 scaffolding is still the
	current implementation. Subsequent slices replace the body.
	"""

	_ = h5_path, output_rel_root, force_restart

	enabled = bool(getattr(stage_config, "unitmatch_enabled", False))
	return {
		"status": "noop" if not enabled else "scaffold",
		"phase": "unitmatch",
		"dataset_index": int(dataset_index),
		"dataset_id": dataset_id,
		"stream_id": str(stream_id),
		"mea_output_root": str(mea_output_root),
		"enabled": enabled,
		"reason": (
			"phase disabled in YAML"
			if not enabled
			else (
				"unitmatch_phase_plan slice 1 is scaffold-only; "
				"group discovery + unitlink.match invocation arrive in slices 2-3"
			)
		),
	}


def _run_unitmatch_from_args(args: argparse.Namespace) -> int:
	"""CLI handler for `axon-recon stages analysis.unitmatch`.

	Routes through `pipeline/runner.run_analysis_unitmatch_from_runtime`
	(slice 1 scaffold). Subsequent slices keep this entry point stable;
	only the runtime function's body changes.
	"""

	from ....runner import run_analysis_unitmatch_from_runtime

	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_analysis_aggregate(
		run_analysis_unitmatch_from_runtime(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			replot_override=(True if bool(getattr(args, "replot", False)) else None),
			task_allocation_override=getattr(args, "task_allocation_override", None),
		)
	)


__all__ = ["run_analysis_unitmatch", "_run_unitmatch_from_args"]
