from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .artifacts import (
	build_segment_time_vector,
	load_recording_metadata,
	load_segment_manifest,
	load_segment_recording_from_entry,
)


_DEFAULT_THRESHOLD_FACTOR = 5.0
_DEFAULT_REFRACTORY_PERIOD_MS = 0.8
_DEFAULT_NOISE_ESTIMATION_WINDOW_FRAMES = 20_000
_DEFAULT_NOISE_ESTIMATION_WINDOWS = 4
_DEFAULT_DETECTION_CHUNK_FRAMES = 50_000


def _resolve_channel_ids(recording: object) -> list[int]:
	channel_ids: list[int] = []
	for value in list(recording.get_channel_ids()):
		try:
			channel_ids.append(int(value))
		except Exception:
			continue
	return channel_ids


def _lookup_segment_payloads(segment_epochs_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
	segments = segment_epochs_payload.get("segments", [])
	if not isinstance(segments, list):
		return {}
	return {
		str(item.get("rec_name", "")): dict(item)
		for item in segments
		if isinstance(item, dict) and str(item.get("rec_name", "")).strip()
	}


def _resolve_absolute_origin_s(
	*,
	segment_entries: list[dict[str, Any]],
	segment_payloads_by_name: dict[str, dict[str, Any]],
) -> float | None:
	for entry in segment_entries:
		rec_name = str(entry.get("rec_name", "")).strip()
		segment_payload = segment_payloads_by_name.get(rec_name)
		if not isinstance(segment_payload, dict):
			continue
		try:
			start_time = segment_payload.get("start_time_seconds_since_epoch", None)
			if start_time is None:
				continue
			return float(start_time)
		except Exception:
			continue
	return None


def _build_segment_boundaries(
	*,
	segment_entries: list[dict[str, Any]],
	segment_payloads_by_name: dict[str, dict[str, Any]],
	absolute_origin_s: float | None,
) -> list[dict[str, Any]]:
	boundaries: list[dict[str, Any]] = []
	if absolute_origin_s is None:
		return boundaries
	for entry in segment_entries:
		rec_name = str(entry.get("rec_name", "")).strip()
		segment_payload = segment_payloads_by_name.get(rec_name)
		if not isinstance(segment_payload, dict):
			continue
		try:
			start_time = float(segment_payload.get("start_time_seconds_since_epoch", 0.0)) - float(absolute_origin_s)
			stop_time = float(segment_payload.get("stop_time_seconds_since_epoch", 0.0)) - float(absolute_origin_s)
		except Exception:
			continue
		if stop_time < start_time:
			continue
		boundaries.append(
			{
				"rec_name": rec_name,
				"start_s": float(start_time),
				"stop_s": float(stop_time),
			}
		)
	return boundaries


def _build_fallback_segment_time_vector(
	*,
	recording: object,
	segment_payload: dict[str, Any] | None,
	absolute_origin_s: float | None,
) -> Any | None:
	import numpy as np

	if not isinstance(segment_payload, dict) or absolute_origin_s is None:
		return None
	try:
		n_samples = int(recording.get_num_samples())
		fs_hz = float(recording.get_sampling_frequency())
		start_time = float(segment_payload.get("start_time_seconds_since_epoch", 0.0)) - float(absolute_origin_s)
	except Exception:
		return None
	if n_samples <= 0 or fs_hz <= 0.0:
		return None
	return float(start_time) + (np.arange(int(n_samples), dtype=float) / float(fs_hz))


def _estimate_channel_thresholds(
	*,
	recording: object,
	channel_ids: list[int],
	threshold_factor: float,
	window_frames: int = _DEFAULT_NOISE_ESTIMATION_WINDOW_FRAMES,
	n_windows: int = _DEFAULT_NOISE_ESTIMATION_WINDOWS,
) -> Any:
	import numpy as np

	total_samples = int(recording.get_num_samples())
	if total_samples <= 0 or not channel_ids:
		return np.asarray([], dtype=float)
	window = max(256, min(int(window_frames), int(total_samples)))
	max_start = max(0, int(total_samples - window))
	if max_start <= 0:
		window_starts = [0]
	else:
		window_starts = np.unique(np.linspace(0, max_start, num=max(1, int(n_windows)), dtype=np.int64)).tolist()
	noise_estimates: list[Any] = []
	for start_frame in window_starts:
		end_frame = min(total_samples, int(start_frame) + int(window))
		traces = recording.get_traces(
			start_frame=int(start_frame),
			end_frame=int(end_frame),
			channel_ids=channel_ids,
		)
		traces_array = np.asarray(traces, dtype=float)
		if traces_array.ndim == 1:
			traces_array = traces_array[:, None]
		if traces_array.size == 0:
			continue
		noise_estimates.append(np.median(np.abs(traces_array), axis=0) / 0.6744897501960817)
	if not noise_estimates:
		return np.full(len(channel_ids), max(1e-6, float(threshold_factor)), dtype=float)
	thresholds = np.median(np.vstack(noise_estimates), axis=0) * float(threshold_factor)
	return np.clip(np.asarray(thresholds, dtype=float), 1e-6, None)


def _collect_threshold_crossings(
	*,
	recording: object,
	channel_ids: list[int],
	time_vector: Any,
	thresholds: Any,
	refractory_period_ms: float,
	detection_chunk_frames: int = _DEFAULT_DETECTION_CHUNK_FRAMES,
) -> tuple[Any, Any]:
	import numpy as np

	total_samples = int(recording.get_num_samples())
	if total_samples <= 0 or not channel_ids:
		return np.asarray([], dtype=float), np.asarray([], dtype=int)
	times = np.asarray(time_vector, dtype=float)
	if times.size != total_samples:
		return np.asarray([], dtype=float), np.asarray([], dtype=int)
	fs_hz = float(recording.get_sampling_frequency())
	refractory_samples = max(1, int(round(float(fs_hz) * (float(refractory_period_ms) / 1000.0))))
	chunk_frames = max(1024, int(detection_chunk_frames))
	last_emitted_sample = np.full(len(channel_ids), -refractory_samples - 1, dtype=np.int64)
	threshold_array = np.asarray(thresholds, dtype=float)
	event_time_parts: list[Any] = []
	event_electrode_parts: list[Any] = []
	for chunk_start in range(0, total_samples, chunk_frames):
		chunk_stop = min(total_samples, int(chunk_start) + int(chunk_frames))
		read_start = max(0, int(chunk_start) - 1)
		read_stop = min(total_samples, int(chunk_stop) + 1)
		traces = recording.get_traces(
			start_frame=int(read_start),
			end_frame=int(read_stop),
			channel_ids=channel_ids,
		)
		traces_array = np.asarray(traces, dtype=float)
		if traces_array.ndim == 1:
			traces_array = traces_array[:, None]
		if traces_array.shape[0] < 3 or traces_array.shape[1] == 0:
			continue
		center = traces_array[1:-1, :]
		left = traces_array[:-2, :]
		right = traces_array[2:, :]
		crossing_mask = (center <= (-threshold_array)[None, :]) & (center <= left) & (center < right)
		if not bool(crossing_mask.any()):
			continue
		for channel_index in range(crossing_mask.shape[1]):
			candidate_samples = np.flatnonzero(crossing_mask[:, channel_index]).astype(np.int64) + 1 + int(read_start)
			if candidate_samples.size == 0:
				continue
			candidate_samples = candidate_samples[
				(candidate_samples >= int(chunk_start)) & (candidate_samples < int(chunk_stop))
			]
			if candidate_samples.size == 0:
				continue
			kept_samples: list[int] = []
			last_sample = int(last_emitted_sample[channel_index])
			for sample_index in candidate_samples.tolist():
				if int(sample_index) - int(last_sample) < int(refractory_samples):
					continue
				kept_samples.append(int(sample_index))
				last_sample = int(sample_index)
			last_emitted_sample[channel_index] = int(last_sample)
			if not kept_samples:
				continue
			selected_samples = np.asarray(kept_samples, dtype=np.int64)
			event_time_parts.append(times[selected_samples])
			event_electrode_parts.append(
				np.full(selected_samples.size, int(channel_ids[channel_index]), dtype=np.int64)
			)
	if not event_time_parts:
		return np.asarray([], dtype=float), np.asarray([], dtype=int)
	return (
		np.concatenate(event_time_parts).astype(float, copy=False),
		np.concatenate(event_electrode_parts).astype(int, copy=False),
	)


def _write_threshold_raster_plot(
	*,
	stream_id: str,
	out_path: Path,
	event_times_s: Any,
	event_electrodes: Any,
	unique_electrodes: list[int],
	segment_boundaries: list[dict[str, Any]],
	title: str,
) -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	fig, ax = plt.subplots(figsize=(16.0, 8.0), dpi=180)
	event_times = np.asarray(event_times_s, dtype=float)
	electrodes = np.asarray(event_electrodes, dtype=int)
	if event_times.size and electrodes.size:
		ax.scatter(
			event_times,
			electrodes,
			s=1.0,
			c="black",
			marker=".",
			linewidths=0,
			alpha=0.6,
			rasterized=True,
		)
	for boundary in segment_boundaries:
		for key in ("start_s", "stop_s"):
			try:
				x_value = float(boundary.get(key, 0.0))
			except Exception:
				continue
			ax.axvline(x_value, color="red", linestyle=":", linewidth=0.8, alpha=0.85)
	if segment_boundaries:
		x_max = max(float(boundary.get("stop_s", 0.0) or 0.0) for boundary in segment_boundaries)
		ax.set_xlim(0.0, max(0.0, x_max))
	elif event_times.size:
		ax.set_xlim(float(event_times.min(initial=0.0)), float(event_times.max(initial=0.0)))
	if unique_electrodes:
		ax.set_ylim(float(min(unique_electrodes) - 1), float(max(unique_electrodes) + 1))
		if len(unique_electrodes) <= 64:
			ax.set_yticks(unique_electrodes)
	ax.set_xlabel("time (s)")
	ax.set_ylabel("electrode id")
	ax.set_title(title or f"Threshold raster ({stream_id})")
	ax.grid(False)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)


def run_plot_raster_threshold_core(
	*,
	stream_id: str,
	segment_manifest_path: Path,
	segment_epochs_path: Path,
	contiguous_epochs_path: Path,
	sampling_metadata_path: Path,
	raster_output_dir: Path,
	report_step_timers: bool = False,
	logger: logging.Logger | None,
) -> dict[str, object]:
	import numpy as np

	step_timers: dict[str, float] = {}
	total_t0 = time.perf_counter()
	load_t0 = time.perf_counter()
	segment_entries = load_segment_manifest(segment_manifest_path)
	segment_epochs_payload, contiguous_epochs_payload, sampling_metadata_payload = load_recording_metadata(
		segment_epochs_path=segment_epochs_path,
		contiguous_epochs_path=contiguous_epochs_path,
		sampling_metadata_path=sampling_metadata_path,
	)
	step_timers["load_manifest_and_metadata"] = float(max(0.0, time.perf_counter() - load_t0))
	segment_payloads_by_name = _lookup_segment_payloads(segment_epochs_payload)
	absolute_origin_s = _resolve_absolute_origin_s(
		segment_entries=segment_entries,
		segment_payloads_by_name=segment_payloads_by_name,
	)
	segment_boundaries = _build_segment_boundaries(
		segment_entries=segment_entries,
		segment_payloads_by_name=segment_payloads_by_name,
		absolute_origin_s=absolute_origin_s,
	)
	all_event_time_parts: list[Any] = []
	all_event_electrode_parts: list[Any] = []
	all_electrode_ids: set[int] = set()
	events_by_segment: list[dict[str, Any]] = []
	scan_t0 = time.perf_counter()
	for entry in segment_entries:
		rec_name = str(entry.get("rec_name", "")).strip()
		if not rec_name:
			continue
		recording = load_segment_recording_from_entry(dict(entry))
		channel_ids = _resolve_channel_ids(recording)
		all_electrode_ids.update(int(value) for value in channel_ids)
		segment_payload = segment_payloads_by_name.get(rec_name)
		time_vector = build_segment_time_vector(
			rec_name=rec_name,
			segment_epochs_payload=segment_epochs_payload,
			contiguous_epochs_payload=contiguous_epochs_payload,
			sampling_metadata_payload=sampling_metadata_payload,
			absolute_origin_s=absolute_origin_s,
		)
		if time_vector is None:
			time_vector = _build_fallback_segment_time_vector(
				recording=recording,
				segment_payload=segment_payload,
				absolute_origin_s=absolute_origin_s,
			)
		if time_vector is None:
			if logger is not None:
				logger.warning(
					"plot_raster_threshold: skipping %s because no segment time vector could be resolved",
					rec_name,
				)
			continue
		thresholds = _estimate_channel_thresholds(
			recording=recording,
			channel_ids=channel_ids,
			threshold_factor=_DEFAULT_THRESHOLD_FACTOR,
		)
		event_times_s, event_electrodes = _collect_threshold_crossings(
			recording=recording,
			channel_ids=channel_ids,
			time_vector=np.asarray(time_vector, dtype=float),
			thresholds=thresholds,
			refractory_period_ms=_DEFAULT_REFRACTORY_PERIOD_MS,
		)
		all_event_time_parts.append(np.asarray(event_times_s, dtype=float))
		all_event_electrode_parts.append(np.asarray(event_electrodes, dtype=int))
		event_count = int(np.asarray(event_times_s).size)
		events_by_segment.append(
			{
				"rec_name": rec_name,
				"electrode_count": int(len(channel_ids)),
				"event_count": int(event_count),
			}
		)
		if logger is not None:
			logger.info(
				"plot_raster_threshold: segment=%s electrodes=%d events=%d",
				rec_name,
				int(len(channel_ids)),
				int(event_count),
			)
	step_timers["scan_segments"] = float(max(0.0, time.perf_counter() - scan_t0))
	all_event_times = (
		np.concatenate(all_event_time_parts).astype(float, copy=False)
		if all_event_time_parts
		else np.asarray([], dtype=float)
	)
	all_event_electrodes = (
		np.concatenate(all_event_electrode_parts).astype(int, copy=False)
		if all_event_electrode_parts
		else np.asarray([], dtype=int)
	)
	unique_electrodes = sorted(int(value) for value in all_electrode_ids)
	raster_plot_path = Path(raster_output_dir) / f"threshold_raster_{stream_id}.png"
	plot_t0 = time.perf_counter()
	_write_threshold_raster_plot(
		stream_id=str(stream_id),
		out_path=raster_plot_path,
		event_times_s=all_event_times,
		event_electrodes=all_event_electrodes,
		unique_electrodes=unique_electrodes,
		segment_boundaries=segment_boundaries,
		title=f"Threshold raster ({stream_id})",
	)
	step_timers["write_raster_plot"] = float(max(0.0, time.perf_counter() - plot_t0))
	total_elapsed_s = float(max(0.0, time.perf_counter() - total_t0))
	if bool(report_step_timers) and logger is not None:
		logger.info(
			"plot_raster_threshold timers well=%s total=%.3fs steps=%s",
			str(stream_id),
			float(total_elapsed_s),
			{str(key): round(float(value), 6) for key, value in step_timers.items()},
		)
	if logger is not None:
		logger.info(
			"plot_raster_threshold: wrote %s segments=%d electrodes=%d events=%d",
			raster_plot_path,
			int(len(segment_entries)),
			int(len(unique_electrodes)),
			int(all_event_times.size),
		)
	return {
		"phase": "plot_raster_threshold",
		"segment_count": int(len(segment_entries)),
		"electrode_count": int(len(unique_electrodes)),
		"electrode_ids": [int(value) for value in unique_electrodes],
		"segment_boundary_count": int(2 * len(segment_boundaries)),
		"segment_boundaries": list(segment_boundaries),
		"raster_plot_path": str(raster_plot_path),
		"raster_output_dir": str(raster_output_dir),
		"total_event_count": int(all_event_times.size),
		"events_by_segment": list(events_by_segment),
		"threshold_factor": float(_DEFAULT_THRESHOLD_FACTOR),
		"refractory_period_ms": float(_DEFAULT_REFRACTORY_PERIOD_MS),
	}
	if bool(report_step_timers):
		payload["step_timers"] = {str(key): float(value) for key, value in step_timers.items()}
		payload["total_elapsed_s"] = float(total_elapsed_s)
	return payload