from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any


def run_save_segment_recordings_core(
	*,
	segment_recordings: list[Any],
	segment_names: list[str],
	segment_stats: list[dict[str, Any]],
	output_dir: Path,
	manifest_path: Path,
	overwrite_saved_recording: bool,
	n_jobs: int,
	chunk_duration: str,
	progress_bar: bool,
	logger: logging.Logger | None,
) -> dict[str, object]:
	if output_dir.exists() and bool(overwrite_saved_recording):
		shutil.rmtree(output_dir)
	if output_dir.exists() and manifest_path.exists() and not bool(overwrite_saved_recording):
		if logger is not None:
			logger.info("Preprocess segment save skipped; existing directory reused: %s", output_dir)
		return {
			"output_dir": str(output_dir),
			"manifest_path": str(manifest_path),
			"saved": False,
			"reused_existing": True,
			"segment_count": int(len(segment_recordings)),
		}
	output_dir.mkdir(parents=True, exist_ok=True)
	manifest_segments: list[dict[str, object]] = []
	for seg_idx, seg_rec in enumerate(segment_recordings):
		rec_name = segment_names[seg_idx] if seg_idx < len(segment_names) else f"segment_{seg_idx:03d}"
		seg_token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(rec_name)).strip("_")
		if not seg_token:
			seg_token = f"segment_{seg_idx:03d}"
		seg_dir = output_dir / f"{seg_idx:03d}_{seg_token}"
		seg_rec.save(
			folder=seg_dir,
			format="binary",
			overwrite=True,
			n_jobs=max(1, int(n_jobs)),
			chunk_duration=str(chunk_duration),
			progress_bar=bool(progress_bar),
		)
		seg_entry: dict[str, object] = {
			"segment_index": int(seg_idx),
			"rec_name": str(rec_name),
			"folder": str(seg_dir),
		}
		if seg_idx < len(segment_stats) and isinstance(segment_stats[seg_idx], dict):
			seg_entry.update({
				"fs_hz": float(segment_stats[seg_idx].get("fs", 0.0) or 0.0),
				"n_samples": int(segment_stats[seg_idx].get("n_samples", 0) or 0),
				"n_channels": int(segment_stats[seg_idx].get("n_channels", 0) or 0),
			})
		manifest_segments.append(seg_entry)
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(
		json.dumps(
			{
				"version": 1,
				"segments": manifest_segments,
			},
			indent=2,
			sort_keys=True,
		)
		+ "\n",
		encoding="utf-8",
	)
	if logger is not None:
		logger.info("Preprocess segment save wrote %d segment recording(s): %s", len(manifest_segments), manifest_path)
	return {
		"output_dir": str(output_dir),
		"manifest_path": str(manifest_path),
		"saved": True,
		"reused_existing": False,
		"segment_count": int(len(manifest_segments)),
	}