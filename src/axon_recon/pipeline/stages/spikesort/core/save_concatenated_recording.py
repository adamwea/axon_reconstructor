from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any


def _remove_recording_artifact(path: Path) -> None:
	if path.is_dir():
		shutil.rmtree(path)
	elif path.exists():
		path.unlink()


def run_save_concatenated_recording_core(
	*,
	multirecording: Any,
	recording_dir: Path,
	overwrite_saved_recording: bool,
	output_mode: str,
	n_jobs: int,
	chunk_duration: str,
	progress_bar: bool,
	logger: logging.Logger | None,
) -> dict[str, object]:
	requested_output_mode = str(output_mode or "binary").strip().lower()
	if requested_output_mode not in {"binary", "lazy"}:
		requested_output_mode = "binary"

	if recording_dir.exists() and bool(overwrite_saved_recording):
		_remove_recording_artifact(recording_dir)
	if recording_dir.exists() and not bool(overwrite_saved_recording):
		if logger is not None:
			logger.info("Preprocess concat save skipped; existing directory reused: %s", recording_dir)
		return {
			"recording_dir": str(recording_dir),
			"output_mode": str(requested_output_mode),
			"materialized_recording": bool(requested_output_mode == "binary"),
			"saved": False,
			"reused_existing": True,
		}

	if requested_output_mode == "lazy":
		recording_dir.mkdir(parents=True, exist_ok=True)
		cached_json_path = recording_dir / "cached.json"
		multirecording.dump_to_json(cached_json_path, relative_to=recording_dir)
		if logger is not None:
			logger.info("Preprocess concat save wrote lazy recording provenance: %s", cached_json_path)
		return {
			"recording_dir": str(recording_dir),
			"recording_json_path": str(cached_json_path),
			"output_mode": "lazy",
			"materialized_recording": False,
			"saved": True,
			"reused_existing": False,
		}

	multirecording.save(
		folder=recording_dir,
		format="binary",
		overwrite=True,
		n_jobs=max(1, int(n_jobs)),
		chunk_duration=str(chunk_duration),
		progress_bar=bool(progress_bar),
	)
	if logger is not None:
		logger.info("Preprocess concat save wrote recording directory: %s", recording_dir)
	return {
		"recording_dir": str(recording_dir),
		"output_mode": "binary",
		"materialized_recording": True,
		"saved": True,
		"reused_existing": False,
	}