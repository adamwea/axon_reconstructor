from __future__ import annotations

import argparse

from ...runner import (
	run_preprocess_from_runtime,
	run_preprocess_concatenate_preprocessed_recordings_from_runtime,
	run_preprocess_copy_src_to_scratch_from_runtime,
	run_preprocess_preprocess_segments_from_runtime,
	run_preprocess_save_rec_metadata_from_runtime,
	run_preprocess_save_common_electrodes_from_runtime,
	run_preprocess_wipe_src_scratch_from_runtime,
)


def register_preprocess_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("preprocess", help="Run preprocess stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Recompute preprocess outputs for each target")
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	return _print_preprocess_aggregate(
		run_preprocess_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _print_preprocess_aggregate(agg: object) -> int:
	print(f"stage: {agg.stage}")
	print(f"targets_total: {agg.total_targets}")
	print(f"targets_succeeded: {agg.succeeded_targets}")
	print(f"targets_failed: {agg.failed_targets}")
	for item in agg.target_results:
		t = item.target
		if item.status != "ok" or item.result is None:
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=error "
				f"error={item.error or 'unknown'}"
			)
			continue
		result = item.result
		if isinstance(result, dict):
			print(
				f"target[{t.dataset_index}:{t.stream_id}] status=ok "
				f"phase={result.get('phase', agg.stage)} preprocess_out_dir={result.get('preprocess_out_dir', None)} "
				f"summary={result.get('summary_json', None)}"
			)
			continue
		print(
			f"target[{t.dataset_index}:{t.stream_id}] status=ok "
			f"preprocess_out_dir={result.preprocess_out_dir} "
			f"outputs={len(result.outputs)}"
		)
	return 0


def _run_copy_src_to_scratch_from_args(args: argparse.Namespace) -> int:
	return _print_preprocess_aggregate(
		run_preprocess_copy_src_to_scratch_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_save_rec_metadata_from_args(args: argparse.Namespace) -> int:
	return _print_preprocess_aggregate(
		run_preprocess_save_rec_metadata_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_wipe_src_scratch_from_args(args: argparse.Namespace) -> int:
	return _print_preprocess_aggregate(
		run_preprocess_wipe_src_scratch_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_preprocess_segments_from_args(args: argparse.Namespace) -> int:
	return _print_preprocess_aggregate(
		run_preprocess_preprocess_segments_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_concatenate_preprocessed_recordings_from_args(args: argparse.Namespace) -> int:
	return _print_preprocess_aggregate(
		run_preprocess_concatenate_preprocessed_recordings_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)


def _run_build_preprocessed_recording_from_args(args: argparse.Namespace) -> int:
	return _run_preprocess_segments_from_args(args)


def _run_save_concatenated_recording_from_args(args: argparse.Namespace) -> int:
	return _run_concatenate_preprocessed_recordings_from_args(args)


def _run_save_segment_recordings_from_args(args: argparse.Namespace) -> int:
	return _run_preprocess_segments_from_args(args)


def _run_save_common_electrodes_from_args(args: argparse.Namespace) -> int:
	return _print_preprocess_aggregate(
		run_preprocess_save_common_electrodes_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)
