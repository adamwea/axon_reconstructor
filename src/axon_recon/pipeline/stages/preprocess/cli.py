from __future__ import annotations

import argparse
import logging

from ...runner import run_preprocess_from_runtime
from .orchestrators import (
	_run_concat_segments_from_args,
	_run_copy_src_to_scratch_from_args,
	_run_plot_concat_channel_layout_from_args,
	_run_plot_concat_traces_from_args,
	_run_plot_raster_threshold_from_args,
	_run_plot_segment_channel_layouts_from_args,
	_run_plot_segment_traces_from_args,
	_run_prepare_raw_binaries_from_args,
	_run_preprocess_segments_from_args,
	_run_save_rec_metadata_from_args,
	_run_wipe_src_scratch_from_args,
)
from .orchestrators._shared import (
	_target_datasets_override_from_args,
	print_preprocess_aggregate,
)


LOGGER = logging.getLogger("axon_recon.preprocess.cli")


def _emit_preprocess_aggregate(agg: object) -> int:
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
		if isinstance(result, dict):
			LOGGER.info(
				"target[%s:%s] status=ok phase=%s preprocess_out_dir=%s summary=%s",
				target.dataset_index,
				target.stream_id,
				result.get("phase", agg.stage),
				result.get("preprocess_out_dir", None),
				result.get("summary_json", None),
			)
			continue
		LOGGER.info(
			"target[%s:%s] status=ok preprocess_out_dir=%s outputs=%s",
			target.dataset_index,
			target.stream_id,
			result.preprocess_out_dir,
			len(result.outputs),
		)
	return 0


def register_preprocess_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("preprocess", help="Run preprocess stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Recompute preprocess outputs for each target")
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	parser.add_argument(
		"--target-datasets",
		nargs="+",
		default=None,
		help=(
			"Target specific 0-based dataset indices, for example --target-datasets 0 or "
			"--target-datasets 0,2,8"
		),
	)
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	target_datasets_override = _target_datasets_override_from_args(args)
	return _emit_preprocess_aggregate(
		run_preprocess_from_runtime(
			config_path=str(args.config),
			limit_segments_override=getattr(args, "limit_segments", None),
			limit_datasets_override=getattr(args, "limit_datasets", None),
			target_datasets_override=target_datasets_override,
			limit_wells_per_dataset_override=getattr(args, "limit_wells_per_dataset", None),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)
