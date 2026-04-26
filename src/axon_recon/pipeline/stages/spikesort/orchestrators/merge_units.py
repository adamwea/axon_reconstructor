from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ....execution.results import MultiTargetStageResult
from ..models.results import SpikesortMergeResult
from ..runner import run_spikesort_merge_stage


def run_spikesort_merge_units(
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
		stage_config=stage_config,
		force_restart=force_restart,
		force_replot=force_replot,
	)


def run_spikesort_merge_units_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	merge_sequence_override: tuple[str, ...] | list[str] | None = None,
	stage_name: str = "spikesort.merge",
) -> MultiTargetStageResult:
	from ....runner import run_spikesort_merge_from_runtime as run_spikesort_merge_runtime

	return run_spikesort_merge_runtime(
		config_path=str(config_path),
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		merge_sequence_override=merge_sequence_override,
		stage_name=stage_name,
	)


def _print_spikesort_merge_aggregate(agg: object) -> int:
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		target = item.target
		if item.status == "ok" and item.result is not None:
			merge_out_dir = getattr(item.result, "merge_out_dir", None)
			summary_json = getattr(item.result, "summary_json", None)
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=ok "
				f"merge_out_dir={merge_out_dir} "
				f"summary={summary_json}"
			)
		else:
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0


def _run_merge_units_from_args(args: argparse.Namespace) -> int:
	return _print_spikesort_merge_aggregate(
		run_spikesort_merge_units_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_merge_slay_from_args(args: argparse.Namespace) -> int:
	return _print_spikesort_merge_aggregate(
		run_spikesort_merge_units_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
			merge_sequence_override=("SLAy",),
			stage_name="spikesort.merge.slay",
		)
	)


def _run_merge_auto_merge_from_args(args: argparse.Namespace) -> int:
	return _print_spikesort_merge_aggregate(
		run_spikesort_merge_units_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
			merge_sequence_override=("auto_merge",),
			stage_name="spikesort.merge.auto_merge",
		)
	)