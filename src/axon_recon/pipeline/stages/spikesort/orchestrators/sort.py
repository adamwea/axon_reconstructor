from __future__ import annotations

import argparse

from ....execution.results import MultiTargetStageResult
from ..models.inputs import SpikesortInputs
from ..models.results import SpikesortResult
from ..runner import run_spikesort_stage


def run_spikesort_sort(inputs: SpikesortInputs) -> SpikesortResult:
	return run_spikesort_stage(inputs)


def run_spikesort_sort_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	from ....runner import run_spikesort_sort_from_runtime as run_spikesort_sort_runtime

	return run_spikesort_sort_runtime(
		config_path=str(config_path),
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def _print_spikesort_aggregate(agg: object) -> int:
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		target = item.target
		if item.status == "ok" and item.result is not None:
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=ok "
				f"spikesort_out_dir={item.result.spikesort_out_dir} "
				f"outputs={len(item.result.outputs)}"
			)
		else:
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0


def _run_sort_from_args(args: argparse.Namespace) -> int:
	return _print_spikesort_aggregate(
		run_spikesort_sort_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)