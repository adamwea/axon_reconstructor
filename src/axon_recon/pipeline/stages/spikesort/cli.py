from __future__ import annotations

import argparse

from .orchestrators import (
	_run_bombcell_label_from_args as _run_bombcell_label_orchestrator_from_args,
	_run_bootstrap_concat_binary_from_args as _run_bootstrap_concat_binary_orchestrator_from_args,
	_run_cleanup_concat_binary_from_args as _run_cleanup_concat_binary_orchestrator_from_args,
	_run_concat_analyzer_from_args as _run_concat_analyzer_orchestrator_from_args,
	_run_merge_slay_from_args as _run_merge_slay_orchestrator_from_args,
	_run_merge_units_from_args as _run_merge_units_orchestrator_from_args,
	_run_restore_sorter_output_from_args as _run_restore_sorter_output_orchestrator_from_args,
	_run_snapshot_sorter_output_from_args as _run_snapshot_sorter_output_orchestrator_from_args,
	_run_sort_from_args,
	_run_summarize_sort_from_args as _run_summarize_sort_orchestrator_from_args,
)
from .orchestrators.sort import _target_datasets_override_from_args


def register_spikesort_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("spikesort", aliases=["spikesorting"], help="Run spikesort stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Recompute spikesort outputs for each target")
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	parser.add_argument(
		"--target-dataset",
		"--target-datasets",
		nargs="+",
		default=None,
		dest="target_datasets",
		help=(
			"Target specific 0-based dataset indices, for example --target-dataset 0 or "
			"--target-datasets 0,2,8"
		),
	)
	parser.set_defaults(handler=_run_sort_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	from ...runner import run_spikesort_from_runtime
	from .orchestrators.sort import _debug_outputs_enabled_for_config, _emit_spikesort_aggregate

	config_path = str(args.config)
	target_datasets_override = _target_datasets_override_from_args(args)

	return _emit_spikesort_aggregate(
		run_spikesort_from_runtime(
			config_path=config_path,
				limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
			task_allocation_override=getattr(args, "task_allocation_override", None),
		),
		debug_outputs=_debug_outputs_enabled_for_config(config_path),
	)


def _run_merge_from_args(args: argparse.Namespace) -> int:
	return _run_merge_units_orchestrator_from_args(args)


def _run_merge_slay_from_args(args: argparse.Namespace) -> int:
	return _run_merge_slay_orchestrator_from_args(args)


def _run_summarize_sort_from_args(args: argparse.Namespace) -> int:
	return _run_summarize_sort_orchestrator_from_args(args)


def _run_snapshot_sorter_output_from_args(args: argparse.Namespace) -> int:
	return _run_snapshot_sorter_output_orchestrator_from_args(args)


def _run_concat_analyzer_from_args(args: argparse.Namespace) -> int:
	return _run_concat_analyzer_orchestrator_from_args(args)


def _run_restore_sorter_output_from_args(args: argparse.Namespace) -> int:
	return _run_restore_sorter_output_orchestrator_from_args(args)


def _run_bombcell_from_args(args: argparse.Namespace) -> int:
	return _run_bombcell_label_orchestrator_from_args(args)


def _run_bootstrap_concat_binary_from_args(args: argparse.Namespace) -> int:
	return _run_bootstrap_concat_binary_orchestrator_from_args(args)


def _run_cleanup_concat_binary_from_args(args: argparse.Namespace) -> int:
	return _run_cleanup_concat_binary_orchestrator_from_args(args)
