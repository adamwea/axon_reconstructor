from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .artifacts import build_stitch_frames_from_segment_manifest, load_saved_recording, load_segment_manifest, write_json


def run_concat_segments_core(
	*,
	segment_manifest_path: Path,
	recording_dir: Path,
	concat_manifest_path: Path,
	overwrite_saved_recording: bool,
	n_jobs: int,
	chunk_duration: str,
	progress_bar: bool,
	logger: logging.Logger | None,
	run_save_concatenated_recording_core: Any,
) -> dict[str, object]:
	import spikeinterface.full as si  # type: ignore[import-not-found]

	t0 = time.perf_counter()
	segment_entries = load_segment_manifest(segment_manifest_path)
	if not segment_entries:
		raise RuntimeError(f"No saved preprocessed segments available to concatenate: {segment_manifest_path}")
	segment_recordings = [load_saved_recording(Path(str(item.get("folder")))) for item in segment_entries]
	if len(segment_recordings) == 1:
		multirecording = segment_recordings[0]
	else:
		multirecording = si.concatenate_recordings(segment_recordings)
	stitch_frames = build_stitch_frames_from_segment_manifest(segment_entries)
	save_result = run_save_concatenated_recording_core(
		multirecording=multirecording,
		recording_dir=recording_dir,
		overwrite_saved_recording=bool(overwrite_saved_recording),
		n_jobs=max(1, int(n_jobs)),
		chunk_duration=str(chunk_duration),
		progress_bar=bool(progress_bar),
		logger=logger,
	)
	concat_manifest_payload = {
		"version": 1,
		"segment_count": int(len(segment_entries)),
		"segment_source": "preprocessed",
		"segment_entries": [dict(item) for item in segment_entries],
		"stitch_frames": [int(value) for value in stitch_frames],
		"recording_dir": str(recording_dir),
	}
	write_json(concat_manifest_path, concat_manifest_payload)
	if logger is not None:
		logger.info(
			"Concatenated %d saved segment recording(s) into %s",
			int(len(segment_entries)),
			recording_dir,
		)
	payload: dict[str, object] = {
		"phase": "concat_segments",
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