from __future__ import annotations

import argparse

from ...runner import run_preprocess_from_runtime
from .orchestrators import (
	_run_concat_segments_from_args,
	_run_copy_src_to_scratch_from_args,
	_run_plot_concat_channel_layout_from_args,
	_run_plot_concat_traces_from_args,
	_run_plot_segment_channel_layouts_from_args,
	_run_plot_segment_traces_from_args,
	_run_prepare_raw_binaries_from_args,
	_run_preprocess_segments_from_args,
	_run_save_rec_metadata_from_args,
	_run_wipe_src_scratch_from_args,
)
from .orchestrators._shared import print_preprocess_aggregate


def register_preprocess_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
	parser = subparsers.add_parser("preprocess", help="Run preprocess stage")
	parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument("--force-restart", action="store_true", help="Recompute preprocess outputs for each target")
	parser.add_argument("--force-replot", action="store_true", help="Alias for force-restart compatibility")
	parser.set_defaults(handler=_run_from_args)


def _run_from_args(args: argparse.Namespace) -> int:
	return print_preprocess_aggregate(
		run_preprocess_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)
