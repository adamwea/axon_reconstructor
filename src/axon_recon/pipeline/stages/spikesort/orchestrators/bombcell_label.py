from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ....execution.results import MultiTargetStageResult
from ..models.results import SpikesortBombcellResult
from ..runner import run_spikesort_bombcell_label_stage
from .sort import _target_datasets_override_from_args


def run_spikesort_bombcell_label(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortBombcellResult:
	return run_spikesort_bombcell_label_stage(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def run_spikesort_bombcell_label_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	stage_name: str = "spikesort.bombcell_label",
) -> MultiTargetStageResult:
	from ....runner import (
		run_spikesort_bombcell_label_from_runtime as run_spikesort_bombcell_runtime,
	)

	return run_spikesort_bombcell_runtime(
		config_path=str(config_path),
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		stage_name=stage_name,
	)


def _print_spikesort_bombcell_aggregate(agg: object) -> int:
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		target = item.target
		if item.status == "ok" and item.result is not None:
			bombcell_out_dir = getattr(item.result, "bombcell_out_dir", None)
			summary_json = getattr(item.result, "summary_json", None)
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=ok "
				f"bombcell_out_dir={bombcell_out_dir} "
				f"summary={summary_json}"
			)
		else:
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0


def _run_bombcell_label_from_args(args: argparse.Namespace) -> int:
	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_spikesort_bombcell_aggregate(
		run_spikesort_bombcell_label_from_runtime(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)