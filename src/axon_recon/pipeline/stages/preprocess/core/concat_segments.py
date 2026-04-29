from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .artifacts import build_stitch_frames_from_segment_manifest, load_segment_manifest, load_segment_recording_from_entry, write_json


def _slice_to_common_electrodes(*, recording: Any, common_electrodes: list[int], rec_name: str) -> Any:
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


def run_concat_segments_core(
	*,
	stream_id: str,
	segment_manifest_path: Path,
	recording_dir: Path,
	concat_manifest_path: Path,
	overwrite_saved_recording: bool,
	output_mode: str,
	n_jobs: int,
	chunk_duration: str,
	progress_bar: bool,
	logger: logging.Logger | None,
	run_save_concatenated_recording_core: Any,
	common_electrodes: list[int] | None = None,
) -> dict[str, object]:
	import spikeinterface.full as si  # type: ignore[import-not-found]

	t0 = time.perf_counter()
	segment_entries = load_segment_manifest(segment_manifest_path)
	if not segment_entries:
		raise RuntimeError(f"No preprocessed segments available to concatenate: {segment_manifest_path}")
	if logger is not None:
		logger.info(
			"Starting concat_segments for well=%s segment_count=%d manifest=%s common_electrodes=%d",
			str(stream_id),
			int(len(segment_entries)),
			segment_manifest_path,
			int(0 if common_electrodes is None else len(common_electrodes)),
		)
	common_electrode_list = list(common_electrodes) if common_electrodes else []
	segment_recordings: list[Any] = []
	for segment_index, item in enumerate(segment_entries, start=1):
		rec_name = str(item.get("rec_name", f"segment_{segment_index - 1:03d}")).strip() or f"segment_{segment_index - 1:03d}"
		loaded = load_segment_recording_from_entry(dict(item))
		# Each persisted segment is saved with its full native channel set so
		# downstream segment-template extraction can use every channel. For
		# concat, all segments must share the same channel set, so slice down
		# to the precomputed common-electrode subset here.
		if common_electrode_list:
			loaded = _slice_to_common_electrodes(
				recording=loaded,
				common_electrodes=common_electrode_list,
				rec_name=rec_name,
			)
		segment_recordings.append(loaded)
		if logger is not None:
			logger.info(
				"concat_segments progress well=%s loaded=%d/%d rec_name=%s",
				str(stream_id),
				int(segment_index),
				int(len(segment_entries)),
				str(rec_name),
			)
	if len(segment_recordings) == 1:
		multirecording = segment_recordings[0]
	else:
		multirecording = si.concatenate_recordings(segment_recordings)
	requested_output_mode = str(output_mode or "binary").strip().lower()
	if requested_output_mode not in {"binary", "lazy"}:
		requested_output_mode = "binary"
	stitch_frames = build_stitch_frames_from_segment_manifest(segment_entries)
	if logger is not None:
		logger.info(
			"Saving concatenated recording for well=%s segment_count=%d out=%s output_mode=%s",
			str(stream_id),
			int(len(segment_entries)),
			recording_dir,
			str(requested_output_mode),
		)
	save_result = run_save_concatenated_recording_core(
		multirecording=multirecording,
		recording_dir=recording_dir,
		overwrite_saved_recording=bool(overwrite_saved_recording),
		output_mode=str(requested_output_mode),
		n_jobs=max(1, int(n_jobs)),
		chunk_duration=str(chunk_duration),
		progress_bar=bool(progress_bar),
		logger=logger,
	)
	concat_manifest_payload = {
		"version": 1,
		"output_mode": str(save_result.get("output_mode", requested_output_mode)),
		"segment_count": int(len(segment_entries)),
		"segment_source": "preprocessed",
		"segment_entries": [dict(item) for item in segment_entries],
		"stitch_frames": [int(value) for value in stitch_frames],
		"recording_dir": str(recording_dir),
	}
	write_json(concat_manifest_path, concat_manifest_payload)
	if logger is not None:
		logger.info(
			"Concatenated %d preprocessed segment recording(s) into %s",
			int(len(segment_entries)),
			recording_dir,
		)
	payload: dict[str, object] = {
		"phase": "concat_segments",
		"output_mode": str(save_result.get("output_mode", requested_output_mode)),
		"segment_count": int(len(segment_entries)),
		"segment_source": "preprocessed",
		"source_segment_count": int(len(segment_entries)),
		"concat_manifest_path": str(concat_manifest_path),
		"stitch_frame_count": int(len(stitch_frames)),
		"phase_timing_s": {
			"concat_segments": float(max(0.0, time.perf_counter() - t0)),
		},
	}
	payload.update({str(key): value for key, value in dict(save_result).items()})
	return payload