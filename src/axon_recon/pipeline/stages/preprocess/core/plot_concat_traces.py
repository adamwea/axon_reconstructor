from __future__ import annotations

import logging
from pathlib import Path

from axon_reconstructor.pipeline.stg1_preprocessing.plotting import _plot_concat_cluster_traces

from .artifacts import build_concat_time_vector, load_concat_manifest, load_recording_metadata, load_saved_recording
from .plot_segment_traces import _resolve_representative_channels


def run_plot_concat_traces_core(
	*,
	stream_id: str,
	recording_dir: Path,
	concat_manifest_path: Path,
	segment_epochs_path: Path,
	contiguous_epochs_path: Path,
	sampling_metadata_path: Path,
	plot_output_dir: Path,
	concat_trace_relpath: str,
	plot_concat_trace: bool,
	concat_trace_n_reps: int,
	plot_n_jobs: int,
	trace_downsample_hz: float | None,
	trace_max_points: int,
	logger: logging.Logger | None,
) -> dict[str, object]:
	concat_manifest = load_concat_manifest(concat_manifest_path)
	segment_entries = [
		dict(item)
		for item in concat_manifest.get("segment_entries", [])
		if isinstance(item, dict)
	]
	stitch_frames = [int(value) for value in concat_manifest.get("stitch_frames", []) if value is not None]
	recording = load_saved_recording(recording_dir)
	segment_epochs_payload, contiguous_epochs_payload, sampling_metadata_payload = load_recording_metadata(
		segment_epochs_path=segment_epochs_path,
		contiguous_epochs_path=contiguous_epochs_path,
		sampling_metadata_path=sampling_metadata_path,
	)
	concat_time_vector = build_concat_time_vector(
		segment_entries=segment_entries,
		segment_epochs_payload=segment_epochs_payload,
		contiguous_epochs_payload=contiguous_epochs_payload,
		sampling_metadata_payload=sampling_metadata_payload,
	)
	if concat_time_vector is not None:
		try:
			recording.set_times(concat_time_vector)
		except Exception:
			pass
	representative_channels = _resolve_representative_channels(
		recording=recording,
		n_representative_channels=int(concat_trace_n_reps),
		plot_n_jobs=max(1, int(plot_n_jobs)),
		logger=logger,
	)
	trace_plot_path = Path(plot_output_dir) / str(concat_trace_relpath)
	if bool(plot_concat_trace):
		trace_plot_path.parent.mkdir(parents=True, exist_ok=True)
		_plot_concat_cluster_traces(
			recording=recording,
			channel_ids=[int(value) for value in representative_channels],
			stitch_frames=[int(value) for value in stitch_frames],
			out_path=trace_plot_path,
			title=f"Concat cluster representatives ({stream_id})",
			target_hz=trace_downsample_hz,
			max_points=int(trace_max_points),
			logger=logger,
		)
	return {
		"phase": "plot_concat_traces",
		"segment_count": int(len(segment_entries)),
		"representative_channel_count": int(len(representative_channels)),
		"representative_channel_ids": [int(value) for value in representative_channels],
		"trace_plot_path": str(trace_plot_path),
		"stitch_frame_count": int(len(stitch_frames)),
	}