from __future__ import annotations

import argparse

from .orchestrators import (
	_run_compute_metrics_from_args as _run_compute_metrics_orchestrator_from_args,
	_target_datasets_override_from_args,
	_print_analysis_aggregate,
)


def register_analysis_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("analysis", help="Run analysis stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Recompute analysis outputs for each target")
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
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	from ...runner import run_analysis_from_runtime

	target_datasets_override = _target_datasets_override_from_args(args)
	return _print_analysis_aggregate(
		run_analysis_from_runtime(
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


def _run_compute_metrics_from_args(args: argparse.Namespace) -> int:
	return _run_compute_metrics_orchestrator_from_args(args)
