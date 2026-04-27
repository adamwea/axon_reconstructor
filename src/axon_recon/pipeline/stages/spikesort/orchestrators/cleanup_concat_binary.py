from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ....execution.results import MultiTargetStageResult
from ..models.results import SpikesortResult
from ..runner import run_spikesort_cleanup_concat_binary_stage
from .sort import _print_spikesort_aggregate


def run_spikesort_cleanup_concat_binary(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortResult:
	return run_spikesort_cleanup_concat_binary_stage(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=mea_output_root,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)


def run_spikesort_cleanup_concat_binary_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	from ....runner import run_spikesort_cleanup_concat_binary_from_runtime as run_runtime

	return run_runtime(
		config_path=str(config_path),
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def _run_cleanup_concat_binary_from_args(args: argparse.Namespace) -> int:
	return _print_spikesort_aggregate(
		run_spikesort_cleanup_concat_binary_from_runtime(
			config_path=str(args.config),
			force_restart_override=(True if bool(getattr(args, "force_restart", False)) else None),
			force_replot_override=(True if bool(getattr(args, "force_replot", False)) else None),
		)
	)