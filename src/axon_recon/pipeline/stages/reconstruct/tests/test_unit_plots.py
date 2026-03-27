from __future__ import annotations

from pathlib import Path

import numpy as np

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconOutputConfig
from axon_recon.pipeline.stages.reconstruct.core.unit_plots import write_unit_amplitude_map_png
from axon_recon.pipeline.stages.reconstruct.core.unit_plots import write_unit_circle_recon_plot


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


def test_write_unit_circle_recon_plot_writes_png(tmp_path: Path) -> None:
	template_ch_by_t = np.array(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
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

	class _GraphMock:
		def nodes(self):
			return [0, 1, 2]

	class _GtrMock:
		def __init__(self):
			self._paths_raw = [[0, 1, 2]]
			self.branches = [{"branch_index": 0, "channels": [0, 1, 2]}]
			self.graph = _GraphMock()

	gtr = _GtrMock()
	out_png = tmp_path / "circle_recon.png"
	out_svg = tmp_path / "circle_recon.svg"

	cfg = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base="template_circles",
			channel_scope="nodes_and_branches",
			force_center_soma=True,
			branch_scope="raw",
			unique_color_per_branch=True,
			show_branch_labels=True,
			color_scheme="tab20",
			node_border_linewidth=0.35,
			edge_linewidth=0.8,
		),
		output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="circle_recon", dpi=200.0),
	)

	write_unit_circle_recon_plot(
		output_png=out_png,
		output_svg=out_svg,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=cfg,
		unit_id=1,
	)

	assert out_png.exists()
	assert out_png.stat().st_size > 0


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
