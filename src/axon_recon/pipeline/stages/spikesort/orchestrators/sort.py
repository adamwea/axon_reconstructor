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


def run_spikesort_sort_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
) -> MultiTargetStageResult:
	from ....runner import run_spikesort_sort_from_runtime as run_spikesort_sort_runtime

	return run_spikesort_sort_runtime(
		config_path=str(config_path),
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)


def _debug_outputs_enabled_for_config(config_path: str) -> bool:
	runtime_cfg = RuntimeConfig.load(config_path)
	return bool(parse_spikesort_stage_config(runtime_config=runtime_cfg).debug_outputs)


def _emit_spikesort_aggregate(agg: object, *, debug_outputs: bool) -> int:
	from ....execution.results import (
		stage_aggregate_exit_code,
		stage_aggregate_summary_lines,
	)

	lines = list(stage_aggregate_summary_lines(agg))
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
	return stage_aggregate_exit_code(agg)


def _print_spikesort_aggregate(agg: object) -> int:
	return _emit_spikesort_aggregate(agg, debug_outputs=True)


def _run_sort_from_args(args: argparse.Namespace) -> int:
	config_path = str(args.config)
	target_datasets_override = _target_datasets_override_from_args(args)
	agg = run_spikesort_sort_from_runtime(
		config_path=config_path,
		limit_segments_override=getattr(args, "limit_segments", None),
		limit_datasets_override=getattr(args, "limit_datasets", None),
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		replot_override=(True if bool(getattr(args, "replot", False)) else None),
	)
	return _emit_spikesort_aggregate(
		agg,
		debug_outputs=_debug_outputs_enabled_for_config(config_path),
	)