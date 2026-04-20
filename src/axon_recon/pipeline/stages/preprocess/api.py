from __future__ import annotations

from .models.inputs import PreprocessInputs
from .models.results import PreprocessResult
from .runner import (
	run_preprocess_concatenate_preprocessed_recordings_phase,
	run_preprocess_copy_src_to_scratch_phase,
	run_preprocess_preprocess_segments_phase,
	run_preprocess_save_rec_metadata_phase,
	run_preprocess_save_common_electrodes_phase,
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


def run_preprocess_concatenate_preprocessed_recordings(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_concatenate_preprocessed_recordings_phase(inputs)


def run_preprocess_build_preprocessed_recording(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_preprocess_segments(inputs)


def run_preprocess_save_concatenated_recording(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_concatenate_preprocessed_recordings(inputs)


def run_preprocess_save_segment_recordings(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_preprocess_segments(inputs)


def run_preprocess_save_common_electrodes(inputs: PreprocessInputs) -> dict[str, object]:
	return run_preprocess_save_common_electrodes_phase(inputs)
