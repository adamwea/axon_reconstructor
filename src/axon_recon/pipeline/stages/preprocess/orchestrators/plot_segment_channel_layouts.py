from __future__ import annotations

import argparse

from ....execution.results import MultiTargetStageResult
from ..models.inputs import PreprocessInputs
from ._shared import run_preprocess_phase_from_args, run_preprocess_phase_from_runtime, run_selected_preprocess_phase


_PHASE_NAME = "plot_segment_channel_layouts"


def run_preprocess_plot_segment_channel_layouts(inputs: PreprocessInputs) -> dict[str, object]:
	return run_selected_preprocess_phase(inputs=inputs, phase_name=_PHASE_NAME)


def run_preprocess_plot_segment_channel_layouts_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return run_preprocess_phase_from_runtime(
		phase_name=_PHASE_NAME,
		runner_fn=run_preprocess_plot_segment_channel_layouts,
		config_path=config_path,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)


def _run_plot_segment_channel_layouts_from_args(args: argparse.Namespace) -> int:
	return run_preprocess_phase_from_args(args, runtime_runner=run_preprocess_plot_segment_channel_layouts_from_runtime)