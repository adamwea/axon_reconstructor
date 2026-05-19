from __future__ import annotations

import argparse
from typing import Callable

from ....execution.results import MultiTargetStageResult
from ..models.inputs import PreprocessInputs


PreprocessPhaseRunner = Callable[[PreprocessInputs], dict[str, object]]
PreprocessPhaseRuntimeRunner = Callable[..., MultiTargetStageResult]


def _parse_target_dataset_indices(raw: object) -> list[int] | None:
	if raw is None:
		return None
	items = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
	parsed: list[int] = []
	seen: set[int] = set()
	for item in items:
		for token in str(item).split(","):
			text = str(token).strip()
			if not text:
				continue
			try:
				value = int(text)
			except Exception as exc:
				raise ValueError(f"Invalid dataset index for --target-datasets: {text!r}") from exc
			if value < 0:
				raise ValueError(f"Dataset indices for --target-datasets must be >= 0, got {value}")
			if value in seen:
				continue
			seen.add(value)
			parsed.append(value)
	if not parsed:
		raise ValueError("--target-datasets requires at least one dataset index")
	return parsed


def _target_datasets_override_from_args(args: argparse.Namespace) -> list[int] | None:
	try:
		return _parse_target_dataset_indices(getattr(args, "target_datasets", None))
	except ValueError as exc:
		raise SystemExit(str(exc)) from exc


def run_selected_preprocess_phase(*, inputs: PreprocessInputs, phase_name: str) -> dict[str, object]:
	from ..runner import _run_preprocess_selected_phase

	return _run_preprocess_selected_phase(inputs, selected_phase=str(phase_name))


def run_preprocess_phase_from_runtime(
	*,
	phase_name: str,
	runner_fn: PreprocessPhaseRunner,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
) -> MultiTargetStageResult:
	from ....runner import _run_preprocess_substage_from_runtime

	return _run_preprocess_substage_from_runtime(
		config_path=str(config_path),
		stage_name=f"preprocess.{str(phase_name)}",
		runner_fn=runner_fn,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)


def print_preprocess_aggregate(agg: object) -> int:
	from ....execution.results import stage_aggregate_summary_lines

	for line in stage_aggregate_summary_lines(agg):
		print(line)
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
	target_datasets_override = _target_datasets_override_from_args(args)
	return print_preprocess_aggregate(
		runtime_runner(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			replot_override=(True if bool(getattr(args, "replot", False)) else None),
		)
	)