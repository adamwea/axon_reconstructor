from __future__ import annotations

import argparse

from .orchestrators import (
	_run_bombcell_label_from_args as _run_bombcell_label_orchestrator_from_args,
	_run_merge_auto_merge_from_args as _run_merge_auto_merge_orchestrator_from_args,
	_run_merge_slay_from_args as _run_merge_slay_orchestrator_from_args,
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
	return _run_sort_from_args(args)


def _run_merge_from_args(args: argparse.Namespace) -> int:
	return _run_merge_units_orchestrator_from_args(args)


def _run_merge_slay_from_args(args: argparse.Namespace) -> int:
	return _run_merge_slay_orchestrator_from_args(args)


def _run_merge_auto_merge_from_args(args: argparse.Namespace) -> int:
	return _run_merge_auto_merge_orchestrator_from_args(args)


def _run_summarize_sort_from_args(args: argparse.Namespace) -> int:
	return _run_summarize_sort_orchestrator_from_args(args)


def _run_bombcell_from_args(args: argparse.Namespace) -> int:
	return _run_bombcell_label_orchestrator_from_args(args)
