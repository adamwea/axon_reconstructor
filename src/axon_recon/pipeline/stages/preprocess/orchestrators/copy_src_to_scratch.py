from __future__ import annotations

import argparse

from ....execution.results import MultiTargetStageResult
from ..models.inputs import PreprocessInputs
from ._shared import run_preprocess_phase_from_args, run_preprocess_phase_from_runtime, run_selected_preprocess_phase


_PHASE_NAME = "copy_src_to_scratch"


def run_preprocess_copy_src_to_scratch(inputs: PreprocessInputs) -> dict[str, object]:
	return run_selected_preprocess_phase(inputs=inputs, phase_name=_PHASE_NAME)


def run_preprocess_copy_src_to_scratch_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return run_preprocess_phase_from_runtime(
		phase_name=_PHASE_NAME,
		runner_fn=run_preprocess_copy_src_to_scratch,
		config_path=config_path,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def _run_copy_src_to_scratch_from_args(args: argparse.Namespace) -> int:
	return run_preprocess_phase_from_args(args, runtime_runner=run_preprocess_copy_src_to_scratch_from_runtime)