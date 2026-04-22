from __future__ import annotations

import argparse
from typing import Callable

from ....execution.results import MultiTargetStageResult
from ..models.inputs import PreprocessInputs


PreprocessPhaseRunner = Callable[[PreprocessInputs], dict[str, object]]
PreprocessPhaseRuntimeRunner = Callable[..., MultiTargetStageResult]


def run_selected_preprocess_phase(*, inputs: PreprocessInputs, phase_name: str) -> dict[str, object]:
	from ..runner import _run_preprocess_selected_phase

	return _run_preprocess_selected_phase(inputs, selected_phase=str(phase_name))


def run_preprocess_phase_from_runtime(
	*,
	phase_name: str,
	runner_fn: PreprocessPhaseRunner,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	from ....runner import _run_preprocess_substage_from_runtime

	return _run_preprocess_substage_from_runtime(
		config_path=str(config_path),
		stage_name=f"preprocess.{str(phase_name)}",
		runner_fn=runner_fn,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def print_preprocess_aggregate(agg: object) -> int:
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		target = item.target
		if item.status != "ok" or item.result is None:
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
			continue
		result = item.result
		if isinstance(result, dict):
			print(
				f"target[{target.dataset_index}:{target.stream_id}] status=ok "
				f"phase={result.get('phase', agg.stage)} preprocess_out_dir={result.get('preprocess_out_dir', None)} "
				f"summary={result.get('summary_json', None)}"
			)
			continue
		print(
			f"target[{target.dataset_index}:{target.stream_id}] status=ok "
			f"preprocess_out_dir={result.preprocess_out_dir} "
			f"outputs={len(result.outputs)}"
		)
	return 0


def run_preprocess_phase_from_args(
	args: argparse.Namespace,
	*,
	runtime_runner: PreprocessPhaseRuntimeRunner,
) -> int:
	return print_preprocess_aggregate(
		runtime_runner(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)