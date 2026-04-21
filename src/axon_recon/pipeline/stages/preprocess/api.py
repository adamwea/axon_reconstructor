from __future__ import annotations

from .models.inputs import PreprocessInputs
from .models.results import PreprocessResult
from .runner import (
	run_preprocess_concat_segments_phase,
	run_preprocess_copy_src_to_scratch_phase,
	run_preprocess_plot_concat_traces_phase,
	run_preprocess_plot_segment_traces_phase,
	run_preprocess_preprocess_segments_phase,
	run_preprocess_save_rec_metadata_phase,
	run_preprocess_stage,
	run_preprocess_wipe_src_scratch_phase,
)


def run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
	return run_preprocess_stage(inputs)


def run_preprocess_copy_src_to_scratch(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_copy_src_to_scratch_phase(inputs)


def run_preprocess_save_rec_metadata(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_save_rec_metadata_phase(inputs)


def run_preprocess_wipe_src_scratch(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_wipe_src_scratch_phase(inputs)


def run_preprocess_preprocess_segments(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_preprocess_segments_phase(inputs)


def run_preprocess_plot_segment_traces(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_plot_segment_traces_phase(inputs)


def run_preprocess_concat_segments(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_concat_segments_phase(inputs)


def run_preprocess_plot_concat_traces(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_plot_concat_traces_phase(inputs)
