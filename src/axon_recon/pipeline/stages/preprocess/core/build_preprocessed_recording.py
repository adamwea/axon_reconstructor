from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from axon_reconstructor.pipeline.stg1_preprocessing.runner import build_concatenated_recording


def run_build_preprocessed_recording_core(
	*,
	h5_path: Path,
	stream_id: str,
	n_jobs: int,
	plot_output_dir: Path | None,
	plot_layouts: bool,
	plot_concat_trace: bool,
	plot_segment_traces: bool,
	epoch_markers_output_dir: Path | None,
	assay_stats_relpath: str,
	channel_layouts_subdir: str,
	segment_traces_subdir: str,
	concat_trace_relpath: str,
	n_representative_channels: int,
	concat_trace_n_reps: int,
	segment_trace_n_reps: int,
	plot_n_jobs: int,
	trace_downsample_hz: float | None,
	trace_max_points: int,
	limit_segments_per_well: int | None,
	temporal_resample_factor: int | None,
	temporal_resample_rate_hz: int | None,
	temporal_resample_margin_ms: float,
	temporal_resample_dtype: str | None,
	saved_assay_stats_path: Path | None,
	require_saved_assay_stats: bool,
	phase_dividers: bool,
	emit_phase_dividers_to_stdout: bool,
	suppress_h5_plugin_messages: bool,
	logger: logging.Logger | None,
) -> tuple[Any, list[int], dict[str, object]]:
	build_result = build_concatenated_recording(
		h5_path=h5_path,
		stream_id=stream_id,
		n_jobs=max(1, int(n_jobs)),
		plot_output_dir=plot_output_dir,
		plot_layouts=bool(plot_layouts),
		plot_concat_trace=bool(plot_concat_trace),
		plot_segment_traces=bool(plot_segment_traces),
		epoch_markers_output_dir=epoch_markers_output_dir,
		assay_stats_relpath=str(assay_stats_relpath),
		channel_layouts_subdir=str(channel_layouts_subdir),
		segment_traces_subdir=str(segment_traces_subdir),
		concat_trace_relpath=str(concat_trace_relpath),
		n_representative_channels=int(n_representative_channels),
		n_representative_channels_concat=int(concat_trace_n_reps),
		n_representative_channels_segment=int(segment_trace_n_reps),
		plot_n_jobs=max(1, int(plot_n_jobs)),
		trace_downsample_hz=trace_downsample_hz,
		trace_max_points=int(trace_max_points),
		limit_segments_per_well=(int(limit_segments_per_well) if limit_segments_per_well is not None else None),
		temporal_resample_factor=(int(temporal_resample_factor) if temporal_resample_factor is not None else None),
		temporal_resample_rate_hz=(int(temporal_resample_rate_hz) if temporal_resample_rate_hz is not None else None),
		temporal_resample_margin_ms=float(temporal_resample_margin_ms),
		temporal_resample_dtype=temporal_resample_dtype,
		saved_assay_stats_path=saved_assay_stats_path,
		require_saved_assay_stats=bool(require_saved_assay_stats),
		logger=logger,
		phase_dividers=bool(phase_dividers),
		emit_phase_dividers_to_stdout=bool(emit_phase_dividers_to_stdout),
		suppress_h5_plugin_messages=bool(suppress_h5_plugin_messages),
		return_artifacts=True,
	)
	if len(build_result) != 3:
		raise RuntimeError("preprocess build core expected artifacts payload from build_concatenated_recording")
	multirecording, common_electrodes, artifacts = build_result
	return multirecording, [int(value) for value in list(common_electrodes)], dict(artifacts)