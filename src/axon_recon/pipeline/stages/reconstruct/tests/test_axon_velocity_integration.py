from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.stages.reconstruct.core.branch_styles import ReconstructBranchRecord
from axon_recon.pipeline.stages.reconstruct.core.plot_branch_propagations import write_unit_branch_propagation_plot
from axon_recon.pipeline.stages.reconstruct.core.plot_branch_velocities import write_unit_branch_velocity_plot
from axon_recon.pipeline.stages.reconstruct.core.report_full_chip_layout import FullChipLayoutBranchRecord
from axon_recon.pipeline.stages.reconstruct.core.report_full_chip_layout import FullChipLayoutUnitRecord
from axon_recon.pipeline.stages.reconstruct.core.report_full_chip_layout import write_full_chip_layout_plot
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionBranchPlotOutputConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionBranchPropagationDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionBranchVelocityDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionFullChipLayoutDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionFullChipLayoutOutputConfig
from axon_recon.pipeline.stages.templates.models.inputs import ProbeGeometryConfig

from axon_recon.pipeline.stages.reconstruct.integrations.axon_velocity import _filter_kwargs_for_callable


def test_filter_kwargs_for_callable() -> None:
	def fn(a: int, b: int) -> int:
		return a + b

	filtered = _filter_kwargs_for_callable(fn, {"a": 1, "b": 2, "c": 3})
	assert filtered == {"a": 1, "b": 2}


def test_write_unit_branch_propagation_plot_writes_png(tmp_path: Path, monkeypatch) -> None:
	pytest.importorskip("axon_velocity")
	import matplotlib.axes
	import numpy as np  # type: ignore[import-not-found]

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
			[35.0, 0.0],
		],
		dtype=float,
	)
	branch_record = ReconstructBranchRecord(
		branch_id=0,
		branch_index=0,
		label=0,
		selected_channels=(0, 1, 2),
		color="#e41a1c",
		scope="raw",
	)
	out_png = tmp_path / "branch_prop.png"
	out_svg = tmp_path / "branch_prop.svg"
	seen = {"count": 0}
	orig_invert_yaxis = matplotlib.axes.Axes.invert_yaxis

	def _spy_invert_yaxis(self, *args, **kwargs):
		seen["count"] += 1
		return orig_invert_yaxis(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "invert_yaxis", _spy_invert_yaxis)

	outputs = write_unit_branch_propagation_plot(
		output_png=out_png,
		output_svg=out_svg,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		branch_record=branch_record,
		display_config=ReconstructionBranchPropagationDisplayConfig(figsize=(7.0, 4.0), sort_templates=True, show_title=True, invert_y_axis=True),
		output_config=ReconstructionBranchPlotOutputConfig(write_png=True, write_svg=False, relpath="branch_plots", dpi=180.0),
		unit_id=91,
	)

	assert outputs["png_path"] == str(out_png)
	assert out_png.exists()
	assert out_png.stat().st_size > 0
	assert seen["count"] >= 1


def test_write_unit_branch_velocity_plot_writes_png(tmp_path: Path) -> None:
	pytest.importorskip("axon_velocity")
	branch_record = ReconstructBranchRecord(
		branch_id=3,
		branch_index=3,
		label=3,
		selected_channels=(0, 1, 2),
		color="#377eb8",
		scope="clean",
		velocity=1.8,
		offset=0.2,
		r2=0.92,
		peak_times=(0.8, 1.5, 2.2),
		distances=(12.0, 25.0, 41.0),
	)
	out_png = tmp_path / "branch_vel.png"
	out_svg = tmp_path / "branch_vel.svg"

	outputs = write_unit_branch_velocity_plot(
		output_png=out_png,
		output_svg=out_svg,
		branch_record=branch_record,
		fit_payload={
			"velocity": 1.8,
			"offset": 0.2,
			"r2": 0.92,
			"peak_times": [0.8, 1.5, 2.2],
			"distances": [12.0, 25.0, 41.0],
		},
		display_config=ReconstructionBranchVelocityDisplayConfig(
			figsize=(6.5, 4.5),
			show_title=True,
			show_legend=True,
			legend_fontsize=10.0,
		),
		output_config=ReconstructionBranchPlotOutputConfig(write_png=True, write_svg=False, relpath="branch_plots", dpi=180.0),
		unit_id=91,
	)

	assert outputs["png_path"] == str(out_png)
	assert out_png.exists()
	assert out_png.stat().st_size > 0


def test_write_full_chip_layout_plot_writes_png(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	out_png = tmp_path / "full_chip_layout.png"
	out_svg = tmp_path / "full_chip_layout.svg"
	seen = {"count": 0}
	orig_invert_yaxis = matplotlib.axes.Axes.invert_yaxis

	def _spy_invert_yaxis(self, *args, **kwargs):
		seen["count"] += 1
		return orig_invert_yaxis(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "invert_yaxis", _spy_invert_yaxis)

	outputs = write_full_chip_layout_plot(
		output_png=out_png,
		output_svg=out_svg,
		unit_records=[
			FullChipLayoutUnitRecord(
				unit_id=91,
				color="#e41a1c",
				branches=(
					FullChipLayoutBranchRecord(
						branch_id=0,
						branch_index=0,
						label=0,
						points_xy=((10.0, 10.0), (25.0, 12.0), (42.0, 20.0)),
					),
				),
			),
			FullChipLayoutUnitRecord(
				unit_id=94,
				color="#377eb8",
				branches=(
					FullChipLayoutBranchRecord(
						branch_id=1,
						branch_index=1,
						label=1,
						points_xy=((60.0, 45.0), (74.0, 56.0), (88.0, 68.0)),
					),
				),
			),
		],
		probe_geometry=ProbeGeometryConfig(active_area_um_x=100.0, active_area_um_y=80.0, pitch_um=17.5),
		display_config=ReconstructionFullChipLayoutDisplayConfig(show_legend=True, legend_ncols=2, invert_y_axis=True),
		output_config=ReconstructionFullChipLayoutOutputConfig(write_png=True, write_svg=False, relpath="reports/full_chip_layout", dpi=180.0),
	)

	assert outputs["full_chip_layout_png"] == str(out_png)
	assert out_png.exists()
	assert out_png.stat().st_size > 0
	assert seen["count"] >= 1
