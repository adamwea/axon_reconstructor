from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from ....execution.results import MultiTargetStageResult
from ..models.results import SpikesortMergeResult
from ..runner import run_spikesort_merge_stage
from .sort import _target_datasets_override_from_args


def _with_merge_sequence_override(stage_config: Any, merge_sequence: tuple[str, ...] | list[str]) -> Any:
	normalized_sequence = tuple(str(token).strip() for token in tuple(merge_sequence) if str(token).strip())
	if not normalized_sequence:
		return stage_config
	return _with_stage_config_overrides(stage_config, merge_sequence=normalized_sequence)


def _with_stage_config_overrides(stage_config: Any, **replace_kwargs: Any) -> Any:
	if getattr(stage_config, "__dataclass_fields__", None) is not None:
		return replace(stage_config, **replace_kwargs)
	for field_name, field_value in replace_kwargs.items():
		setattr(stage_config, field_name, field_value)
	return stage_config


def _with_standalone_merge_phase_stage_config(
	stage_config: Any,
	*,
	phase_prefix: str,
	merge_sequence: tuple[str, ...] | list[str],
	method_enabled_field: str | None = None,
) -> Any:
	phase_enabled = bool(getattr(stage_config, f"{phase_prefix}_enabled", False))
	replace_kwargs: dict[str, Any] = {
		"merge_sequence": tuple(str(token).strip() for token in tuple(merge_sequence) if str(token).strip()),
		"merge_units_enabled": bool(phase_enabled),
		"merge_rel_output_root": str(getattr(stage_config, f"{phase_prefix}_rel_output_root")),
		"merge_delete_outputs_on_force_restart": bool(
			getattr(stage_config, f"{phase_prefix}_delete_outputs_on_force_restart")
		),
		"merge_force_restart": bool(getattr(stage_config, f"{phase_prefix}_force_restart")),
		"merge_force_replot": bool(getattr(stage_config, f"{phase_prefix}_force_replot")),
	}
	phase_runtime_overrides = getattr(stage_config, "merge_phase_runtime_overrides", None)
	if isinstance(phase_runtime_overrides, dict):
		phase_override_payload = phase_runtime_overrides.get(phase_prefix, None)
		if isinstance(phase_override_payload, dict):
			replace_kwargs.update(dict(phase_override_payload))
	if method_enabled_field is not None:
		replace_kwargs[method_enabled_field] = bool(phase_enabled)
	return _with_stage_config_overrides(stage_config, **replace_kwargs)


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
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	merge_sequence_override: tuple[str, ...] | list[str] | None = None,
	stage_name: str = "spikesort.merge",
) -> MultiTargetStageResult:
	from ....runner import run_spikesort_merge_from_runtime as run_spikesort_merge_runtime

	return run_spikesort_merge_runtime(
		config_path=str(config_path),
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
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
	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_spikesort_merge_aggregate(
		run_spikesort_merge_units_from_runtime(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_merge_slay_from_args(args: argparse.Namespace) -> int:
	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_spikesort_merge_aggregate(
		run_spikesort_merge_units_from_runtime(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
			merge_sequence_override=("SLAy",),
			stage_name="spikesort.merge.slay",
		)
	)


