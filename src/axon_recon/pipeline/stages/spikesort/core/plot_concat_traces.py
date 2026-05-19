from __future__ import annotations

import logging
from pathlib import Path

# `plot_concat_traces` moved from the preprocess stage to the spikesort stage in
# phase_roster_cleanup_plan slice 8 (alongside `concat_binary`). The artifact /
# representative-channel helpers still live in the preprocess core package
# (preprocess remains the producer of `segment_epochs.json`, `sampling_metadata.json`,
# etc.), so import them across the stage boundary.
from ...preprocess.core.artifacts import (
	build_concat_time_vector,
	load_concat_manifest,
	load_recording_metadata,
	load_saved_recording,
)
from ...preprocess.core.plot_segment_traces import _resolve_representative_channels


def _plot_concat_cluster_traces(
	*,
	recording: object,
	channel_ids: list[int],
	stitch_frames: list[int],
	out_path: Path,
	title: str | None = None,
	target_hz: float | None = None,
	max_points: int = 150_000,
	logger: logging.Logger | None = None,
) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	total = int(recording.get_num_samples())
	if total <= 0:
		return

	fs = float(recording.get_sampling_frequency())
	has_time_vector = False
	try:
		has_time_vector = bool(recording.has_time_vector())
	except Exception:
		has_time_vector = False

	try:
		parsed_max_points = int(max_points)
	except Exception:
		parsed_max_points = 150_000
	if parsed_max_points <= 0:
		step_by_points = 1
	else:
		points_cap = max(1000, parsed_max_points)
		step_by_points = max(1, total // points_cap)

	step_by_rate = 1
	try:
		if target_hz is not None and float(target_hz) > 0.0 and fs > 0.0:
			step_by_rate = max(1, int(round(fs / float(target_hz))))
	except Exception:
		step_by_rate = 1

	step = max(step_by_points, step_by_rate)
	selected_frames = np.arange(0, total, step, dtype=np.int64)
	expected_points = int(selected_frames.size)
	effective_hz = (float(fs) / float(step)) if step > 0 else float(fs)
	if logger is not None:
		logger.info(
			"plot traces: downsample fs=%.2fHz target_hz=%s step=%d effective_hz=%.2f expected_points_per_channel=%d channels=%d out=%s",
			float(fs),
			(f"{float(target_hz):.2f}" if target_hz is not None else "none"),
			int(step),
			float(effective_hz),
			int(expected_points),
			int(len(channel_ids)),
			out_path,
		)
		if expected_points > 200_000:
			logger.warning(
				"plot traces: high point count after downsampling (%d points/channel); consider lowering trace_downsample_hz or setting trace_max_points",
				int(expected_points),
			)

	if has_time_vector:
		try:
			time_vector = recording.sample_index_to_time(selected_frames)
		except Exception:
			time_vector = selected_frames.astype(float) / fs
	else:
		time_vector = selected_frames.astype(float) / fs

	fig, axes = plt.subplots(len(channel_ids), 1, figsize=(13.33, 7.5), dpi=180, sharex=True)
	if len(channel_ids) == 1:
		axes = [axes]

	block = 200_000
	total_blocks = max(1, int((total + block - 1) // block))
	y_parts_per_channel: list[list[np.ndarray]] = [[] for _ in channel_ids]
	for block_idx, start in enumerate(range(0, total, block), start=1):
		end = min(total, start + block)
		traces_block = recording.get_traces(start_frame=start, end_frame=end, channel_ids=channel_ids)
		offset = (-start) % step
		traces_ds = traces_block[offset::step, :]
		for channel_index in range(len(channel_ids)):
			y_parts_per_channel[channel_index].append(np.asarray(traces_ds[:, channel_index]))

		if logger is not None and (
			block_idx == 1
			or block_idx == total_blocks
			or block_idx % max(1, total_blocks // 10) == 0
		):
			logger.info(
				"plot traces: load progress %d/%d blocks (%.1f%%) out=%s",
				int(block_idx),
				int(total_blocks),
				float((100.0 * block_idx) / max(1, total_blocks)),
				out_path,
			)

	for axis, channel_id, y_parts in zip(axes, channel_ids, y_parts_per_channel, strict=False):
		y = np.concatenate(y_parts).astype(float, copy=False) if y_parts else np.asarray([], dtype=float)
		t_plot = time_vector[: y.size]
		y_plot = y
		if has_time_vector and y_plot.size > 2:
			try:
				dt = np.diff(t_plot.astype(float))
				baseline = float(step) / float(fs)
				jump_idx = np.where(dt > (5.0 * max(baseline, 1e-9)))[0]
				if jump_idx.size:
					y_plot = y_plot.astype(float, copy=True)
					y_plot[jump_idx + 1] = np.nan
			except Exception:
				pass

		axis.plot(t_plot, y_plot, lw=0.2, color="black")
		for stitch_frame in stitch_frames:
			if has_time_vector:
				try:
					xline = float(recording.sample_index_to_time(int(stitch_frame)))
				except Exception:
					xline = float(stitch_frame) / fs
			else:
				xline = float(stitch_frame) / fs
			axis.axvline(xline, color="red", lw=0.6, alpha=0.8)
		axis.set_ylabel(f"ch {channel_id}")
		axis.grid(False)

	axes[-1].set_xlabel("time (s)")
	if title:
		fig.suptitle(title)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)
	if logger is not None:
		logger.info("plot traces: wrote %s", out_path)


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