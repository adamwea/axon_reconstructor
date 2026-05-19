from __future__ import annotations

from .models.inputs import PreprocessInputs
from .models.results import PreprocessResult
from .orchestrators import (
	run_preprocess_plot_concat_channel_layout,
	run_preprocess_plot_concat_traces,
	run_preprocess_plot_raster_threshold,
	run_preprocess_plot_segment_channel_layouts,
	run_preprocess_plot_segment_traces,
	run_preprocess_preprocess_segments,
	run_preprocess_save_rec_metadata,
)
from .runner import run_preprocess_stage


def run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
	return run_preprocess_stage(inputs)
