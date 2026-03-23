from __future__ import annotations

from pathlib import Path

import numpy as np

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.reconstruct.core.unit_plots import write_unit_amplitude_map_png


def test_write_unit_amplitude_map_png_writes_file(tmp_path: Path) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0],
			[-2.0, -4.0, -1.0],
			[-1.0, -6.0, -2.0],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)
	out = tmp_path / "amplitude_map.png"

	write_unit_amplitude_map_png(
		output_png=out,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
	)

	assert out.exists()
	assert out.stat().st_size > 0


def test_write_unit_amplitude_map_png_uses_shared_heatmap_config(tmp_path: Path) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0],
			[-2.0, -4.0, -1.0],
			[-1.0, -6.0, -2.0],
		],
		dtype=float,
	)
	locs_xy = np.array(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)
	out = tmp_path / "amplitude_map_cfg.png"

	cfg = SharedHeatmapConfig(
		background="white",
		show_colorbar=True,
		colorbar_location="bottomleft",
		show_ticks=(1, "dynamic_high"),
		high_color="red",
		mid_color="white",
		low_color="blue",
		scale="linear",
	)

	write_unit_amplitude_map_png(
		output_png=out,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		heatmap_config=cfg,
	)

	assert out.exists()
	assert out.stat().st_size > 0
