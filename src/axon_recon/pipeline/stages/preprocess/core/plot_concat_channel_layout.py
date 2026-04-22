from __future__ import annotations

import logging
from pathlib import Path

from .artifacts import load_saved_recording
from .plot_segment_traces import _plot_channel_layout, _resolve_representative_channels


def run_plot_concat_channel_layout_core(
	*,
	stream_id: str,
	recording_dir: Path,
	plot_output_dir: Path,
	channel_layouts_subdir: str,
	n_representative_channels: int,
	plot_n_jobs: int,
	logger: logging.Logger | None,
) -> dict[str, object]:
	recording = load_saved_recording(recording_dir)
	representative_channels = _resolve_representative_channels(
		recording=recording,
		n_representative_channels=int(n_representative_channels),
		plot_n_jobs=max(1, int(plot_n_jobs)),
		logger=logger,
	)
	layout_path = Path(plot_output_dir) / str(channel_layouts_subdir) / f"concat_channel_layout_{stream_id}.png"
	_plot_channel_layout(
		recording=recording,
		out_path=layout_path,
		highlight_channel_ids=[int(value) for value in representative_channels],
		title=f"Concat channel layout ({stream_id})",
	)
	if logger is not None:
		logger.info("plot concat channel layout: wrote %s", layout_path)
	return {
		"phase": "plot_concat_channel_layout",
		"layout_plot_paths": [str(layout_path)],
		"representative_channel_count": int(len(representative_channels)),
		"representative_channel_ids": [int(value) for value in representative_channels],
	}