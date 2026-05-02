from __future__ import annotations

import argparse
import logging

from axon_recon.runtime_config import RuntimeConfig

from ....execution.results import MultiTargetStageResult
from ..config import parse_spikesort_stage_config
from ..models.inputs import SpikesortInputs
from ..models.results import SpikesortResult
from ..runner import run_spikesort_stage


LOGGER = logging.getLogger("axon_recon.spikesort.cli")


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


def _debug_outputs_enabled_for_config(config_path: str) -> bool:
	runtime_cfg = RuntimeConfig.load(config_path)
	return bool(parse_spikesort_stage_config(runtime_config=runtime_cfg).debug_outputs)


def _emit_spikesort_aggregate(agg: object, *, debug_outputs: bool) -> int:
	lines = [
		f"stage: {agg.stage}",
		f"targets_total: {agg.total_targets}",
		f"targets_succeeded: {agg.succeeded_targets}",
		f"targets_failed: {agg.failed_targets}",
	]
	for item in agg.target_results:
		target = item.target
		if item.status == "ok" and item.result is not None:
			result_dir = (
				getattr(item.result, "spikesort_out_dir", None)
				or getattr(item.result, "bombcell_out_dir", None)
				or getattr(item.result, "merge_out_dir", None)
				or getattr(item.result, "summary_json", "")
			)
			lines.append(
				f"target[{target.dataset_index}:{target.stream_id}] status=ok "
				f"out_dir={result_dir} "
				f"outputs={len(item.result.outputs)}"
			)
		else:
			lines.append(
				f"target[{target.dataset_index}:{target.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	for line in lines:
		LOGGER.info(line)
		if bool(debug_outputs):
			print(line)
	return 0


def _print_spikesort_aggregate(agg: object) -> int:
	return _emit_spikesort_aggregate(agg, debug_outputs=True)


def _run_sort_from_args(args: argparse.Namespace) -> int:
	config_path = str(args.config)
	agg = run_spikesort_sort_from_runtime(
		config_path=config_path,
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
	)
	return _emit_spikesort_aggregate(
		agg,
		debug_outputs=_debug_outputs_enabled_for_config(config_path),
	)