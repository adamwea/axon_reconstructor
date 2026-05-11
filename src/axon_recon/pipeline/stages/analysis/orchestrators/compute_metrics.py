from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ....execution.results import MultiTargetStageResult
from ..models.results import AnalysisResult
from ..runner import run_analysis_compute_metrics_stage


def run_analysis_compute_metrics(
	*,
	dataset_index: int,
	dataset_id: str | None,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> AnalysisResult:
	return run_analysis_compute_metrics_stage(
		dataset_index=dataset_index,
		dataset_id=dataset_id,
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def run_analysis_compute_metrics_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	from ....runner import run_analysis_compute_metrics_from_runtime as run_runtime

	return run_runtime(
		config_path=str(config_path),
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		task_allocation_override=task_allocation_override,
	)


def _target_datasets_override_from_args(args: argparse.Namespace) -> list[int] | None:
	raw = getattr(args, "target_datasets", None)
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
				raise SystemExit(f"Invalid dataset index for --target-datasets: {text!r}") from exc
			if value < 0:
				raise SystemExit(f"Dataset indices must be >= 0, got {value}")
			if value in seen:
				continue
			seen.add(value)
			parsed.append(value)
	return parsed if parsed else None


def _print_analysis_aggregate(agg: MultiTargetStageResult) -> int:
	import logging

	LOGGER = logging.getLogger("axon_recon.analysis.cli")
	LOGGER.info("stage: %s", agg.stage)
	LOGGER.info("targets_total: %s", agg.total_targets)
	LOGGER.info("targets_succeeded: %s", agg.succeeded_targets)
	LOGGER.info("targets_failed: %s", agg.failed_targets)
	for item in agg.target_results:
		target = item.target
		if item.status != "ok" or item.result is None:
			LOGGER.info(
				"target[%s:%s] status=error error=%s",
				target.dataset_index,
				target.stream_id,
				item.error or "unknown",
			)
			continue
		result = item.result
		LOGGER.info(
			"target[%s:%s] status=ok analysis_out_dir=%s manifest=%s outputs=%s",
			target.dataset_index,
			target.stream_id,
			getattr(result, "analysis_out_dir", None),
			getattr(result, "manifest_json", None),
			len(getattr(result, "outputs", {}) or {}),
		)
	return 0


def _run_compute_metrics_from_args(args: argparse.Namespace) -> int:
	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_analysis_aggregate(
		run_analysis_compute_metrics_from_runtime(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
			task_allocation_override=getattr(args, "task_allocation_override", None),
		)
	)
