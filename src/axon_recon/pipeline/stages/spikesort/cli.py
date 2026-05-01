from __future__ import annotations

import argparse

from .orchestrators import (
	_run_bombcell_label_from_args as _run_bombcell_label_orchestrator_from_args,
	_run_bootstrap_concat_binary_from_args as _run_bootstrap_concat_binary_orchestrator_from_args,
	_run_cleanup_concat_binary_from_args as _run_cleanup_concat_binary_orchestrator_from_args,
	_run_merge_si_auto_from_args as _run_merge_si_auto_orchestrator_from_args,
	_run_merge_slay_from_args as _run_merge_slay_orchestrator_from_args,
	_run_merge_unitmatch_from_args as _run_merge_unitmatch_orchestrator_from_args,
	_run_merge_units_from_args as _run_merge_units_orchestrator_from_args,
	_run_sort_from_args,
	_run_summarize_sort_from_args as _run_summarize_sort_orchestrator_from_args,
)


def register_spikesort_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("spikesort", aliases=["spikesorting"], help="Run spikesort stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Recompute spikesort outputs for each target")
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	parser.set_defaults(handler=_run_sort_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	from ...runner import run_spikesort_from_runtime
	from .orchestrators.sort import _print_spikesort_aggregate

	return _print_spikesort_aggregate(
		run_spikesort_from_runtime(
			config_path=str(args.config),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_merge_from_args(args: argparse.Namespace) -> int:
	return _run_merge_units_orchestrator_from_args(args)


def _run_merge_slay_from_args(args: argparse.Namespace) -> int:
	return _run_merge_slay_orchestrator_from_args(args)


def _run_merge_si_auto_from_args(args: argparse.Namespace) -> int:
	return _run_merge_si_auto_orchestrator_from_args(args)


def _run_merge_unitmatch_from_args(args: argparse.Namespace) -> int:
	return _run_merge_unitmatch_orchestrator_from_args(args)


def _run_summarize_sort_from_args(args: argparse.Namespace) -> int:
	return _run_summarize_sort_orchestrator_from_args(args)


def _run_bombcell_from_args(args: argparse.Namespace) -> int:
	return _run_bombcell_label_orchestrator_from_args(args)


def _run_bootstrap_concat_binary_from_args(args: argparse.Namespace) -> int:
	return _run_bootstrap_concat_binary_orchestrator_from_args(args)


def _run_cleanup_concat_binary_from_args(args: argparse.Namespace) -> int:
	return _run_cleanup_concat_binary_orchestrator_from_args(args)
