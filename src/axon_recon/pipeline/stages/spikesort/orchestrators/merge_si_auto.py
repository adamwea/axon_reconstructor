from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ....execution.results import MultiTargetStageResult
from ..models.results import SpikesortMergeResult
from ..runner import run_spikesort_merge_stage
from .merge_units import (
	_print_spikesort_merge_aggregate,
	_with_standalone_merge_phase_stage_config,
)


def run_spikesort_merge_si_auto(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
	force_replot: bool = False,
) -> SpikesortMergeResult:
	return run_spikesort_merge_stage(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=_with_standalone_merge_phase_stage_config(
			stage_config,
			phase_prefix="merge_si_auto",
			merge_sequence=("auto_merge",),
			method_enabled_field="auto_merge_enabled",
			assert_field="cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace",
		),
		force_restart=force_restart,
		force_replot=force_replot,
	)


def run_spikesort_merge_si_auto_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	stage_name: str = "spikesort.merge_si_auto",
) -> MultiTargetStageResult:
	from ....runner import run_spikesort_merge_from_runtime as run_spikesort_merge_runtime

	return run_spikesort_merge_runtime(
		config_path=str(config_path),
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		stage_config_transformer=lambda stage_config: _with_standalone_merge_phase_stage_config(
			stage_config,
			phase_prefix="merge_si_auto",
			merge_sequence=("auto_merge",),
			method_enabled_field="auto_merge_enabled",
			assert_field="cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace",
		),
		debug_phase_label="merge_si_auto",
		debug_enabled_attr="merge_si_auto_debug_mode_enabled",
		debug_limit_datasets_attr="merge_si_auto_debug_limit_datasets",
		debug_limit_wells_attr="merge_si_auto_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="merge_si_auto_debug_limit_wells_per_dataset",
		stage_name=stage_name,
	)


def _run_merge_si_auto_from_args(args: argparse.Namespace) -> int:
	return _print_spikesort_merge_aggregate(
		run_spikesort_merge_si_auto_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)