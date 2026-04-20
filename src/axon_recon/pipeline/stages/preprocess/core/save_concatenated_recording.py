from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any


def run_save_concatenated_recording_core(
	*,
	multirecording: Any,
	recording_dir: Path,
	overwrite_saved_recording: bool,
	n_jobs: int,
	chunk_duration: str,
	progress_bar: bool,
	logger: logging.Logger | None,
) -> dict[str, object]:
	if recording_dir.exists() and bool(overwrite_saved_recording):
		shutil.rmtree(recording_dir)
	if recording_dir.exists() and not bool(overwrite_saved_recording):
		if logger is not None:
			logger.info("Preprocess concat save skipped; existing directory reused: %s", recording_dir)
		return {
			"recording_dir": str(recording_dir),
			"saved": False,
			"reused_existing": True,
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
		"saved": True,
		"reused_existing": False,
	}