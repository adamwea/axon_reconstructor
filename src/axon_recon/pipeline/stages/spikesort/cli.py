from __future__ import annotations

import argparse

from ...runner import run_spikesort_merge_from_runtime, run_spikesort_summarize_sort_from_runtime
from .orchestrators import _run_sort_from_args


def register_spikesort_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("spikesort", aliases=["spikesorting"], help="Run spikesort stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Recompute spikesort outputs for each target")
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	parser.set_defaults(handler=_run_sort_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	return _run_sort_from_args(args)


def _run_merge_from_args(args: argparse.Namespace) -> int:
	agg = run_spikesort_merge_from_runtime(
		config_path=str(args.config),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
	)
	return _print_merge_aggregate(agg)


def _run_merge_slay_from_args(args: argparse.Namespace) -> int:
	agg = run_spikesort_merge_from_runtime(
		config_path=str(args.config),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		merge_sequence_override=("SLAy",),
		stage_name="spikesort.merge.slay",
	)
	return _print_merge_aggregate(agg)


def _run_merge_auto_merge_from_args(args: argparse.Namespace) -> int:
	agg = run_spikesort_merge_from_runtime(
		config_path=str(args.config),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		merge_sequence_override=("auto_merge",),
		stage_name="spikesort.merge.auto_merge",
	)
	return _print_merge_aggregate(agg)


def _run_summarize_sort_from_args(args: argparse.Namespace) -> int:
	agg = run_spikesort_summarize_sort_from_runtime(
		config_path=str(args.config),
		force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
		force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
	)
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		t = item.target
		if item.status == "ok" and item.result is not None:
			summary_json = getattr(item.result, "summary_json", None)
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"summary={summary_json}"
			)
		else:
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0


def _print_merge_aggregate(agg: object) -> int:
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		t = item.target
		if item.status == "ok" and item.result is not None:
			merge_out_dir = getattr(item.result, "merge_out_dir", None)
			summary_json = getattr(item.result, "summary_json", None)
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"merge_out_dir={merge_out_dir} "
				f"summary={summary_json}"
			)
		else:
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
	return 0
