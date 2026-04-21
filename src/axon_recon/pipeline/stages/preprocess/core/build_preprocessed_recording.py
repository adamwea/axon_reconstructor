from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .plot_segment_traces import _plot_channel_layout, _plot_concat_cluster_traces, _resolve_representative_channels
from .preprocess_segments import (
	_load_centered_segment_with_electrode_channel_ids,
	_select_common_electrode_channels,
	apply_standard_preprocessing,
)
from .save_rec_metadata import find_common_electrodes_from_segments


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
	_ = assay_stats_relpath
	_ = epoch_markers_output_dir
	_ = phase_dividers
	_ = emit_phase_dividers_to_stdout
	_ = suppress_h5_plugin_messages
	t0 = time.perf_counter()
	if bool(require_saved_assay_stats) and saved_assay_stats_path is not None and not Path(saved_assay_stats_path).exists():
		raise RuntimeError(f"Required assay stats artifact is missing: {saved_assay_stats_path}")
	if any(value is not None for value in (temporal_resample_factor, temporal_resample_rate_hz, temporal_resample_dtype)):
		if logger is not None:
			logger.warning(
				"Legacy build_preprocessed_recording compatibility path ignores temporal resample settings; use canonical preprocess phases instead"
			)

	resolved_h5_path = Path(h5_path).expanduser().resolve()
	rec_names, common_electrodes = find_common_electrodes_from_segments(
		h5_path=resolved_h5_path,
		stream_id=str(stream_id),
	)
	if limit_segments_per_well is not None and int(limit_segments_per_well) > 0 and len(rec_names) > int(limit_segments_per_well):
		rec_names = list(rec_names[: int(limit_segments_per_well)])
	if not rec_names:
		raise RuntimeError(f"No recording segments available for build_preprocessed_recording: {resolved_h5_path}")

	try:
		max_workers = min(len(rec_names), max(1, int(n_jobs)))
	except Exception:
		max_workers = 1

	load_preprocess_t0 = time.perf_counter()

	def _process_segment(rec_name: str) -> tuple[Any, Any, dict[str, object]]:
		raw_segment, raw_stats = _load_centered_segment_with_electrode_channel_ids(
			h5_path=resolved_h5_path,
			stream_id=str(stream_id),
			rec_name=str(rec_name),
			center_chunk_size=10_000,
		)
		selected_raw = _select_common_electrode_channels(
			recording=raw_segment,
			common_electrodes=[int(value) for value in common_electrodes],
			rec_name=str(rec_name),
		)
		preprocessed = apply_standard_preprocessing(recording=selected_raw, logger=logger)
		stats_payload = {
			"rec_name": str(rec_name),
			"fs": float(raw_stats.get("fs", 0.0) or 0.0),
			"n_samples": int(raw_stats.get("n_samples", 0) or 0),
			"n_channels": int(preprocessed.get_num_channels()),
		}
		return selected_raw, preprocessed, stats_payload

	if max_workers > 1 and len(rec_names) > 1:
		with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
			results = list(pool.map(_process_segment, rec_names))
	else:
		results = [_process_segment(rec_name) for rec_name in rec_names]

	segment_recordings_raw = [item[0] for item in results]
	segment_recordings_preprocessed = [item[1] for item in results]
	segment_stats = [dict(item[2]) for item in results]

	try:
		import spikeinterface.full as si  # type: ignore[import-not-found]
	except Exception as exc:
		raise RuntimeError(f"SpikeInterface import failed while concatenating segment recordings: {exc}") from exc

	if len(segment_recordings_preprocessed) == 1:
		multirecording = segment_recordings_preprocessed[0]
	else:
		multirecording = si.concatenate_recordings(segment_recordings_preprocessed)

	phase_timing_s: dict[str, float] = {
		"preprocess_segments": float(max(0.0, time.perf_counter() - load_preprocess_t0)),
	}

	representative_channels_segment: list[int] = []
	representative_channels_concat: list[int] = []
	layout_plot_paths: list[str] = []
	segment_trace_paths: list[str] = []
	concat_trace_path: str | None = None
	if plot_output_dir is not None:
		plot_root = Path(plot_output_dir)
		plot_root.mkdir(parents=True, exist_ok=True)
		if segment_recordings_preprocessed:
			reference_recording = segment_recordings_preprocessed[0]
			representative_channels_segment = _resolve_representative_channels(
				recording=reference_recording,
				n_representative_channels=int(segment_trace_n_reps),
				plot_n_jobs=max(1, int(plot_n_jobs)),
				logger=logger,
			)
			if bool(plot_layouts):
				layout_path = plot_root / str(channel_layouts_subdir) / f"common_channel_layout_{stream_id}.png"
				_plot_channel_layout(
					recording=reference_recording,
					out_path=layout_path,
					highlight_channel_ids=representative_channels_segment,
					title=f"Common channel layout ({stream_id})",
				)
				layout_plot_paths.append(str(layout_path))
			if bool(plot_segment_traces):
				segment_dir = plot_root / str(segment_traces_subdir)
				segment_dir.mkdir(parents=True, exist_ok=True)
				for rec_name, recording in zip(rec_names, segment_recordings_preprocessed, strict=False):
					out_path = segment_dir / f"segment_trace_{stream_id}_{rec_name}.png"
					_plot_concat_cluster_traces(
						recording=recording,
						channel_ids=[int(value) for value in representative_channels_segment],
						stitch_frames=[],
						out_path=out_path,
						title=f"Segment trace ({stream_id} / {rec_name})",
						target_hz=trace_downsample_hz,
						max_points=int(trace_max_points),
						logger=logger,
					)
					segment_trace_paths.append(str(out_path))
		representative_channels_concat = _resolve_representative_channels(
			recording=multirecording,
			n_representative_channels=int(concat_trace_n_reps),
			plot_n_jobs=max(1, int(plot_n_jobs)),
			logger=logger,
		)
		if bool(plot_concat_trace):
			trace_plot_path = plot_root / str(concat_trace_relpath)
			stitch_frames: list[int] = []
			frame_cursor = 0
			for stats in segment_stats[:-1]:
				frame_cursor += int(stats.get("n_samples", 0) or 0)
				stitch_frames.append(int(frame_cursor))
			_plot_concat_cluster_traces(
				recording=multirecording,
				channel_ids=[int(value) for value in representative_channels_concat],
				stitch_frames=stitch_frames,
				out_path=trace_plot_path,
				title=f"Concat cluster representatives ({stream_id})",
				target_hz=trace_downsample_hz,
				max_points=int(trace_max_points),
				logger=logger,
			)
			concat_trace_path = str(trace_plot_path)

	phase_timing_s["total"] = float(max(0.0, time.perf_counter() - t0))
	artifacts: dict[str, object] = {
		"rec_names": [str(value) for value in rec_names],
		"segment_stats": [dict(item) for item in segment_stats],
		"segment_recordings_raw": list(segment_recordings_raw),
		"segment_recordings_raw_concat": list(segment_recordings_raw),
		"segment_recordings_preprocessed": list(segment_recordings_preprocessed),
		"segment_recordings_preprocessed_concat": list(segment_recordings_preprocessed),
		"phase_timing_s": phase_timing_s,
		"layout_plot_paths": list(layout_plot_paths),
		"segment_trace_paths": list(segment_trace_paths),
		"concat_trace_path": concat_trace_path,
		"representative_channel_ids_segment": [int(value) for value in representative_channels_segment],
		"representative_channel_ids_concat": [int(value) for value in representative_channels_concat],
	}
	if logger is not None:
		logger.info(
			"Legacy build_preprocessed_recording compatibility build completed stream=%s segments=%d common_electrodes=%d",
			str(stream_id),
			int(len(rec_names)),
			int(len(common_electrodes)),
		)
	return multirecording, [int(value) for value in list(common_electrodes)], artifacts