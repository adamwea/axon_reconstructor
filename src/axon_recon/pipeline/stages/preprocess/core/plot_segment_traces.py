from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from axon_reconstructor.pipeline.stg1_preprocessing.plotting import (
	_activity_score_rms,
	_pick_representative_index_for_cluster,
	_plot_concat_cluster_traces,
	detect_electrode_clusters,
)

from .artifacts import (
	build_segment_time_vector,
	load_recording_metadata,
	load_saved_recording,
	load_segment_manifest,
)


def _plot_channel_layout(
	*,
	recording: Any,
	out_path: Path,
	highlight_channel_ids: list[int],
	title: str,
) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	channel_ids = [int(value) for value in list(recording.get_channel_ids())]
	locations = np.asarray(recording.get_channel_locations(), dtype=float)
	if locations.ndim != 2 or locations.shape[0] != len(channel_ids) or locations.shape[1] < 2:
		return
	xs = np.asarray(locations[:, 0], dtype=float)
	ys = np.asarray(locations[:, 1], dtype=float)
	highlight_mask = np.asarray([int(ch) in set(int(v) for v in highlight_channel_ids) for ch in channel_ids], dtype=bool)
	fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
	ax.scatter(xs[~highlight_mask], ys[~highlight_mask], s=4, c="#888888", alpha=0.75, linewidths=0)
	if highlight_mask.any():
		ax.scatter(xs[highlight_mask], ys[highlight_mask], s=10, c="#c0392b", alpha=0.9, linewidths=0)
	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_aspect("equal", adjustable="box")
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)


def _resolve_representative_channels(
	*,
	recording: Any,
	n_representative_channels: int,
	plot_n_jobs: int,
	logger: logging.Logger | None,
) -> list[int]:
	import numpy as np

	channel_ids = [int(value) for value in list(recording.get_channel_ids())]
	if not channel_ids:
		return []
	locations = np.asarray(recording.get_channel_locations(), dtype=float)
	if locations.ndim != 2 or locations.shape[0] != len(channel_ids) or locations.shape[1] < 2:
		if n_representative_channels <= 0:
			return [int(ch) for ch in channel_ids]
		return [int(ch) for ch in channel_ids[: max(1, int(n_representative_channels))]]
	xs = np.asarray(locations[:, 0], dtype=float)
	ys = np.asarray(locations[:, 1], dtype=float)
	clusters = detect_electrode_clusters(x=xs, y=ys, max_cluster_size_warn=9)
	representatives: list[int] = []
	for cluster in clusters:
		rep_index = _pick_representative_index_for_cluster(x=xs, y=ys, cluster=cluster)
		representatives.append(int(channel_ids[int(rep_index)]))
	if not representatives:
		representatives = [int(ch) for ch in channel_ids]

	def _score_channel(channel_id: int) -> float:
		return float(
			_activity_score_rms(
				recording=recording,
				channel_id=int(channel_id),
				chunk_size=4_000,
				num_chunks=4,
			)
		)

	score_workers = min(max(1, int(plot_n_jobs)), max(1, int(len(representatives))))
	if score_workers > 1 and len(representatives) > 1:
		with ThreadPoolExecutor(max_workers=int(score_workers)) as pool:
			scores = list(pool.map(_score_channel, representatives))
	else:
		scores = [_score_channel(channel_id) for channel_id in representatives]
	sorted_representatives = [
		int(channel_id)
		for channel_id, _score in sorted(
			zip(representatives, scores, strict=False),
			key=lambda item: item[1],
			reverse=True,
		)
	]
	if logger is not None:
		logger.info(
			"Selected %d representative segment-trace channels from %d candidate channels",
			int(len(sorted_representatives)),
			int(len(channel_ids)),
		)
	if n_representative_channels <= 0:
		return list(sorted_representatives)
	return list(sorted_representatives[: max(1, int(n_representative_channels))])


def run_plot_segment_traces_core(
	*,
	stream_id: str,
	segment_manifest_path: Path,
	segment_epochs_path: Path,
	contiguous_epochs_path: Path,
	sampling_metadata_path: Path,
	plot_output_dir: Path,
	channel_layouts_subdir: str,
	segment_traces_subdir: str,
	plot_layouts: bool,
	plot_segment_traces: bool,
	segment_trace_n_reps: int,
	plot_n_jobs: int,
	trace_downsample_hz: float | None,
	trace_max_points: int,
	logger: logging.Logger | None,
) -> dict[str, object]:
	segment_entries = load_segment_manifest(segment_manifest_path)
	if not segment_entries:
		raise RuntimeError(f"No saved segment recordings available for plotting: {segment_manifest_path}")
	segment_epochs_payload, contiguous_epochs_payload, sampling_metadata_payload = load_recording_metadata(
		segment_epochs_path=segment_epochs_path,
		contiguous_epochs_path=contiguous_epochs_path,
		sampling_metadata_path=sampling_metadata_path,
	)
	reference_recording = load_saved_recording(Path(str(segment_entries[0].get("folder"))))
	representative_channels = _resolve_representative_channels(
		recording=reference_recording,
		n_representative_channels=int(segment_trace_n_reps),
		plot_n_jobs=max(1, int(plot_n_jobs)),
		logger=logger,
	)
	layout_paths: list[str] = []
	if bool(plot_layouts):
		layout_path = Path(plot_output_dir) / str(channel_layouts_subdir) / f"common_channel_layout_{stream_id}.png"
		_plot_channel_layout(
			recording=reference_recording,
			out_path=layout_path,
			highlight_channel_ids=representative_channels,
			title=f"Common channel layout ({stream_id})",
		)
		layout_paths.append(str(layout_path))
	segment_trace_paths: list[str] = []
	if bool(plot_segment_traces):
		segment_traces_dir = Path(plot_output_dir) / str(segment_traces_subdir)
		segment_traces_dir.mkdir(parents=True, exist_ok=True)

		def _render_segment(entry: dict[str, Any]) -> str:
			rec_name = str(entry.get("rec_name", "segment"))
			recording = load_saved_recording(Path(str(entry.get("folder"))))
			time_vector = build_segment_time_vector(
				rec_name=str(rec_name),
				segment_epochs_payload=segment_epochs_payload,
				contiguous_epochs_payload=contiguous_epochs_payload,
				sampling_metadata_payload=sampling_metadata_payload,
			)
			if time_vector is not None:
				try:
					recording.set_times(time_vector)
				except Exception:
					pass
			out_path = segment_traces_dir / f"segment_trace_{stream_id}_{rec_name}.png"
			_plot_concat_cluster_traces(
				recording=recording,
				channel_ids=[int(value) for value in representative_channels],
				stitch_frames=[],
				out_path=out_path,
				title=f"Segment trace ({stream_id} / {rec_name})",
				target_hz=trace_downsample_hz,
				max_points=int(trace_max_points),
				logger=logger,
			)
			return str(out_path)

		segment_jobs = min(max(1, int(plot_n_jobs)), max(1, int(len(segment_entries))))
		if segment_jobs > 1 and len(segment_entries) > 1:
			with ThreadPoolExecutor(max_workers=int(segment_jobs)) as pool:
				futures = [pool.submit(_render_segment, dict(entry)) for entry in segment_entries]
				for future in as_completed(futures):
					segment_trace_paths.append(str(future.result()))
		else:
			segment_trace_paths = [_render_segment(dict(entry)) for entry in segment_entries]
	return {
		"phase": "plot_segment_traces",
		"segment_count": int(len(segment_entries)),
		"representative_channel_count": int(len(representative_channels)),
		"representative_channel_ids": [int(value) for value in representative_channels],
		"layout_plot_paths": list(layout_paths),
		"segment_trace_paths": list(segment_trace_paths),
	}