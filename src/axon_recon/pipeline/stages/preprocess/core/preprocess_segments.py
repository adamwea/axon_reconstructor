from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from axon_reconstructor.pipeline.stg1_preprocessing.concatenation import _load_centered_segment_with_electrode_channel_ids
from axon_reconstructor.pipeline.stg1_preprocessing.preprocessing import apply_standard_preprocessing

from .artifacts import (
	get_segment_names_from_metadata,
	load_common_electrodes,
	load_recording_metadata,
)


def _select_common_electrode_channels(*, recording: Any, common_electrodes: list[int], rec_name: str) -> Any:
	import numpy as np

	if not common_electrodes:
		return recording
	selected = recording.select_channels([int(value) for value in common_electrodes])
	selected_ch = np.asarray(selected.get_channel_ids(), dtype=int)
	expected = np.asarray(common_electrodes, dtype=int)
	if selected_ch.shape != expected.shape or not np.array_equal(selected_ch, expected):
		raise RuntimeError(
			f"Selected common-electrode channel ids mismatch for segment {rec_name}; refusing to continue with misaligned channels"
		)
	return selected


def run_preprocess_segments_core(
	*,
	h5_path: Path,
	stream_id: str,
	n_jobs: int,
	segment_epochs_path: Path,
	contiguous_epochs_path: Path,
	sampling_metadata_path: Path,
	common_electrodes_path: Path,
	output_dir: Path,
	manifest_path: Path,
	overwrite_saved_recording: bool,
	save_n_jobs: int,
	chunk_duration: str,
	progress_bar: bool,
	limit_segments_per_well: int | None,
	logger: logging.Logger | None,
	run_save_segment_recordings_core: Any,
) -> dict[str, object]:
	_ = contiguous_epochs_path
	t0 = time.perf_counter()
	segment_epochs_payload, contiguous_epochs_payload, sampling_metadata_payload = load_recording_metadata(
		segment_epochs_path=segment_epochs_path,
		contiguous_epochs_path=contiguous_epochs_path,
		sampling_metadata_path=sampling_metadata_path,
	)
	_ = contiguous_epochs_payload
	_ = sampling_metadata_payload
	rec_names = get_segment_names_from_metadata(segment_epochs_payload)
	if limit_segments_per_well is not None and int(limit_segments_per_well) > 0 and len(rec_names) > int(limit_segments_per_well):
		rec_names = list(rec_names[: int(limit_segments_per_well)])
	common_electrodes = load_common_electrodes(common_electrodes_path)
	if not rec_names:
		raise RuntimeError(f"No recording segments available in metadata artifact: {segment_epochs_path}")

	try:
		max_workers = min(len(rec_names), max(1, int(n_jobs)))
	except Exception:
		max_workers = 1

	load_preprocess_t0 = time.perf_counter()

	def _process_segment(rec_name: str) -> tuple[Any, dict[str, Any]]:
		segment_recording, stats = _load_centered_segment_with_electrode_channel_ids(
			h5_path=Path(h5_path).expanduser().resolve(),
			stream_id=str(stream_id),
			rec_name=str(rec_name),
			center_chunk_size=10_000,
		)
		segment_recording = _select_common_electrode_channels(
			recording=segment_recording,
			common_electrodes=common_electrodes,
			rec_name=str(rec_name),
		)
		preprocessed = apply_standard_preprocessing(recording=segment_recording, logger=logger)
		stats_payload = {
			"rec_name": str(rec_name),
			"fs": float(stats.get("fs", 0.0) or 0.0),
			"n_samples": int(stats.get("n_samples", 0) or 0),
			"n_channels": int(preprocessed.get_num_channels()),
		}
		return preprocessed, stats_payload

	if max_workers > 1 and len(rec_names) > 1:
		with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
			results = list(pool.map(_process_segment, rec_names))
	else:
		results = [_process_segment(rec_name) for rec_name in rec_names]

	segment_recordings = [item[0] for item in results]
	segment_stats = [dict(item[1]) for item in results]
	phase_timing_s = {
		"preprocess_segments": float(max(0.0, time.perf_counter() - load_preprocess_t0)),
	}
	if logger is not None:
		logger.info(
			"Preprocessed %d segment recording(s) for stream=%s common_electrodes=%d workers=%d",
			int(len(segment_recordings)),
			str(stream_id),
			int(len(common_electrodes)),
			int(max_workers),
		)

	save_result = run_save_segment_recordings_core(
		segment_recordings=list(segment_recordings),
		segment_names=[str(value) for value in rec_names],
		segment_stats=[dict(item) for item in segment_stats],
		output_dir=output_dir,
		manifest_path=manifest_path,
		overwrite_saved_recording=bool(overwrite_saved_recording),
		n_jobs=max(1, int(save_n_jobs)),
		chunk_duration=str(chunk_duration),
		progress_bar=bool(progress_bar),
		logger=logger,
	)
	phase_timing_s["save_preprocessed_segments"] = float(max(0.0, time.perf_counter() - load_preprocess_t0))
	phase_timing_s["total"] = float(max(0.0, time.perf_counter() - t0))

	payload: dict[str, object] = {
		"phase": "preprocess_segments",
		"segment_count": int(len(segment_recordings)),
		"rec_names": [str(value) for value in rec_names],
		"common_electrode_count": int(len(common_electrodes)),
		"phase_timing_s": phase_timing_s,
	}
	payload.update({str(key): value for key, value in dict(save_result).items()})
	return payload