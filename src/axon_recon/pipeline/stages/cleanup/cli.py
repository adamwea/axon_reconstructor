from __future__ import annotations

import argparse
import logging


LOGGER = logging.getLogger("axon_recon.cleanup.cli")


def _emit_cleanup_aggregate(agg: object) -> int:
	from ...execution.results import stage_aggregate_summary_lines

	for line in stage_aggregate_summary_lines(agg):
		LOGGER.info(line)
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
		if isinstance(result, dict):
			LOGGER.info(
				"target[%s:%s] status=ok phase=%s cleanup_out_dir=%s summary=%s",
				target.dataset_index,
				target.stream_id,
				result.get("phase", agg.stage),
				result.get("cleanup_out_dir", None),
				result.get("summary_json", None),
			)
			continue
		LOGGER.info(
			"target[%s:%s] status=ok",
			target.dataset_index,
			target.stream_id,
		)
	return 0 if agg.failed_targets == 0 else 1


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
				raise SystemExit(f"Dataset indices for --target-datasets must be >= 0, got {value}")
			if value in seen:
				continue
			seen.add(value)
			parsed.append(value)
	if not parsed:
		raise SystemExit("--target-datasets requires at least one dataset index")
	return parsed


def _run_wipe_src_scratch_from_args(args: argparse.Namespace) -> int:
	"""CLI handler for `axon-recon stages cleanup.wipe_src_scratch`."""

	from ...runner import run_cleanup_wipe_src_scratch_from_runtime

	config_path = str(getattr(args, "config", None) or "")
	if not config_path:
		raise SystemExit("--config is required for cleanup.wipe_src_scratch")
	target_datasets_override = _target_datasets_override_from_args(args)
	agg = run_cleanup_wipe_src_scratch_from_runtime(
		config_path=config_path,
		limit_datasets_override=getattr(args, "limit_datasets", None),
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		task_allocation_override=getattr(args, "task_allocation_override", None),
	)
	return _emit_cleanup_aggregate(agg)


def _run_from_args(args: argparse.Namespace) -> int:
	"""CLI entry point for the full cleanup stage (`axon-recon stages cleanup`).

	Walks `stage_config.phase_sequence` and dispatches each phase via
	`run_cleanup_from_runtime`. With slice 6's single phase configured, this
	is equivalent to running `cleanup.wipe_src_scratch` end-to-end; the
	sequence abstraction lets future end-of-run cleanup phases slot in
	without a CLI change.
	"""

	from ...runner import run_cleanup_from_runtime

	config_path = str(getattr(args, "config", None) or "")
	if not config_path:
		raise SystemExit("--config is required for the cleanup stage")
	target_datasets_override = _target_datasets_override_from_args(args)
	agg = run_cleanup_from_runtime(
		config_path=config_path,
		limit_datasets_override=getattr(args, "limit_datasets", None),
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		task_allocation_override=getattr(args, "task_allocation_override", None),
	)
	return _emit_cleanup_aggregate(agg)
