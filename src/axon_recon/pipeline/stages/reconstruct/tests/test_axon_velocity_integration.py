from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.reconstruct.core.branch_styles import ReconstructBranchRecord
from axon_recon.pipeline.stages.reconstruct.core.plot_branch_propagations import write_unit_branch_propagation_plot
from axon_recon.pipeline.stages.reconstruct.core.plot_unit_summary import write_unit_summary_plot
from axon_recon.pipeline.stages.reconstruct.core.plot_branch_velocities import write_unit_branch_velocity_plot
from axon_recon.pipeline.stages.reconstruct.core.report_full_chip_layout import FullChipLayoutBranchRecord
from axon_recon.pipeline.stages.reconstruct.core.report_full_chip_layout import FullChipLayoutUnitRecord
from axon_recon.pipeline.stages.reconstruct.core.report_full_chip_layout import write_full_chip_layout_plot
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionBranchPlotOutputConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionBranchColorsConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionBranchPropagationDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionBranchVelocityDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconOutputConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionFullChipLayoutDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionFullChipLayoutOutputConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionPlotBranchPropagationsPhaseConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionPlotBranchVelocitiesPhaseConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionUnitSummaryDisplayConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionUnitSummaryOutputConfig
from axon_recon.pipeline.stages.templates.models.inputs import ProbeGeometryConfig

from axon_recon.pipeline.stages.reconstruct.integrations.axon_velocity import _filter_kwargs_for_callable


def test_filter_kwargs_for_callable() -> None:
	def fn(a: int, b: int) -> int:
		return a + b

	filtered = _filter_kwargs_for_callable(fn, {"a": 1, "b": 2, "c": 3})
	assert filtered == {"a": 1, "b": 2}


def test_write_unit_branch_propagation_plot_writes_png(tmp_path: Path, monkeypatch) -> None:
	pytest.importorskip("axon_velocity")
	import axon_velocity.plotting as av_plotting
	import matplotlib.axes
	import matplotlib.figure as mpl_figure
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
	branch_records = (
		ReconstructBranchRecord(
			branch_id=0,
			branch_index=0,
			label=0,
			selected_channels=(0, 1, 2),
			color="#e41a1c",
			scope="raw",
		),
		ReconstructBranchRecord(
			branch_id=1,
			branch_index=1,
			label=1,
			selected_channels=(2, 1, 0),
			color="#377eb8",
			scope="raw",
		),
	)
	out_png = tmp_path / "branch_prop.png"
	out_svg = tmp_path / "branch_prop.svg"
	seen = {"count": 0}
	seen_calls: list[dict[str, object]] = []
	seen_savefig_calls: list[dict[str, object]] = []
	orig_invert_yaxis = matplotlib.axes.Axes.invert_yaxis
	orig_savefig = mpl_figure.Figure.savefig

	def _spy_invert_yaxis(self, *args, **kwargs):
		seen["count"] += 1
		return orig_invert_yaxis(self, *args, **kwargs)

	def _spy_plot_template_propagation(template, locations, selected_channels, sort_templates=False, color=None, color_marker=None, ax=None):
		assert isinstance(selected_channels, list)
		seen_calls.append(
			{
				"selected_channels": list(selected_channels),
				"sort_templates": bool(sort_templates),
				"color": color,
				"color_marker": color_marker,
			}
		)
		assert ax is not None
		ax.plot([0.0, 1.0], [0.0, 1.0], color=color)
		ax.plot([0.0], [0.0], marker="o", color=color_marker)
		ax.axis("off")
		return ax

	def _spy_savefig(self, *args, **kwargs):
		seen_savefig_calls.append(
			{
				"figure_facecolor": tuple(float(value) for value in self.get_facecolor()),
				"axes_facecolors": [tuple(float(value) for value in ax.get_facecolor()) for ax in self.axes],
				"fig_size_inches": tuple(float(value) for value in self.get_size_inches()),
				"axes_titles": [ax.get_title() for ax in self.axes],
				"facecolor_kwarg": kwargs.get("facecolor"),
			}
		)
		return orig_savefig(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "invert_yaxis", _spy_invert_yaxis)
	monkeypatch.setattr(av_plotting, "plot_template_propagation", _spy_plot_template_propagation)
	monkeypatch.setattr(mpl_figure.Figure, "savefig", _spy_savefig)

	outputs = write_unit_branch_propagation_plot(
		output_png=out_png,
		output_svg=out_svg,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		branch_records=branch_records,
		display_config=ReconstructionBranchPropagationDisplayConfig(
			figsize=(2.5, 6.0),
			total_width=6.0,
			sort_templates=False,
			show_title=True,
			invert_y_axis=True,
		),
		output_config=ReconstructionBranchPlotOutputConfig(write_png=True, write_svg=False, relpath="branch_plots", dpi=180.0),
		unit_id=91,
	)

	assert outputs["png_path"] == str(out_png)
	assert out_png.exists()
	assert out_png.stat().st_size > 0
	assert seen["count"] >= 2
	assert [call["selected_channels"] for call in seen_calls] == [[0, 1, 2], [2, 1, 0]]
	assert [call["color"] for call in seen_calls] == ["#e41a1c", "#377eb8"]
	assert [call["color_marker"] for call in seen_calls] == ["#e41a1c", "#377eb8"]
	assert len(seen_savefig_calls) >= 1
	assert seen_savefig_calls[0]["figure_facecolor"] == (0.0, 0.0, 0.0, 1.0)
	assert all(facecolor == (0.0, 0.0, 0.0, 1.0) for facecolor in seen_savefig_calls[0]["axes_facecolors"])
	assert seen_savefig_calls[0]["fig_size_inches"] == pytest.approx((6.0, 6.0))
	assert seen_savefig_calls[0]["axes_titles"] == ["b0", "b1"]


def test_write_unit_branch_velocity_plot_writes_png(tmp_path: Path, monkeypatch) -> None:
	pytest.importorskip("axon_velocity")
	import axon_velocity.plotting as av_plotting
	import matplotlib.figure as mpl_figure

	branch_records = (
		ReconstructBranchRecord(
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
		),
		ReconstructBranchRecord(
			branch_id=4,
			branch_index=4,
			label=4,
			selected_channels=(1, 2, 3),
			color="#e41a1c",
			scope="clean",
			velocity=2.3,
			offset=0.1,
			r2=0.81,
			peak_times=(0.7, 1.2, 1.9),
			distances=(10.0, 22.0, 37.0),
		),
	)
	fit_payloads = (
		{
			"velocity": 1.8,
			"offset": 0.2,
			"r2": 0.92,
			"peak_times": [0.8, 1.5, 2.2],
			"distances": [12.0, 25.0, 41.0],
		},
		{
			"velocity": 2.3,
			"offset": 0.1,
			"r2": 0.81,
			"peak_times": [0.7, 1.2, 1.9],
			"distances": [10.0, 22.0, 37.0],
		},
	)
	out_png = tmp_path / "branch_vel.png"
	out_svg = tmp_path / "branch_vel.svg"
	seen_calls: list[dict[str, object]] = []
	seen_savefig_calls: list[dict[str, object]] = []
	orig_plot_velocity = av_plotting.plot_velocity
	orig_savefig = mpl_figure.Figure.savefig

	def _spy_plot_velocity(peak_times, distances, velocity, offset, color=None, r2=None, ax=None, **kwargs):
		seen_calls.append(
			{
				"peak_times": list(peak_times),
				"distances": list(distances),
				"velocity": velocity,
				"offset": offset,
				"color": color,
				"r2": r2,
				"ax_id": id(ax),
			}
		)
		assert ax is not None
		ax.plot(peak_times, distances, marker="o", ls="", color=color)
		ax.plot([peak_times[0], peak_times[-1]], [velocity * peak_times[0] + offset, velocity * peak_times[-1] + offset], ls="--", color=color, label="placeholder")
		ax.legend()
		return ax

	def _spy_savefig(self, *args, **kwargs):
		legend = self.axes[0].get_legend()
		legend_anchor_bounds = None
		if legend is not None:
			legend_anchor_bounds = tuple(
				float(value)
				for value in legend.get_bbox_to_anchor().transformed(self.axes[0].transAxes.inverted()).bounds
			)
		seen_savefig_calls.append(
			{
				"figure_facecolor": tuple(float(value) for value in self.get_facecolor()),
				"axes_facecolors": [tuple(float(value) for value in ax.get_facecolor()) for ax in self.axes],
				"title_fontsize": float(self.axes[0].title.get_fontsize()),
				"xlabel_text": self.axes[0].xaxis.label.get_text(),
				"xlabel_fontsize": float(self.axes[0].xaxis.label.get_fontsize()),
				"ylabel_text": self.axes[0].yaxis.label.get_text(),
				"ylabel_fontsize": float(self.axes[0].yaxis.label.get_fontsize()),
				"tick_fontsizes": sorted(
					{
						float(label.get_fontsize())
						for label in [*self.axes[0].get_xticklabels(), *self.axes[0].get_yticklabels()]
						if label.get_text()
					}
				),
				"legend_labels": ([] if legend is None else [text.get_text() for text in legend.get_texts()]),
				"legend_text_colors": ([] if legend is None else [text.get_color() for text in legend.get_texts()]),
				"legend_anchor_bounds": legend_anchor_bounds,
			}
		)
		return orig_savefig(self, *args, **kwargs)

	monkeypatch.setattr(av_plotting, "plot_velocity", _spy_plot_velocity)
	monkeypatch.setattr(mpl_figure.Figure, "savefig", _spy_savefig)

	outputs = write_unit_branch_velocity_plot(
		output_png=out_png,
		output_svg=out_svg,
		branch_records=branch_records,
		fit_payloads=fit_payloads,
		display_config=ReconstructionBranchVelocityDisplayConfig(
			figsize=(6.5, 4.5),
			show_title=True,
			title_fontsize=16.0,
			axis_label_fontsize=13.0,
			tick_label_fontsize=11.0,
			units_only_axis_labels=True,
			show_legend=True,
			legend_fontsize=10.0,
		),
		output_config=ReconstructionBranchPlotOutputConfig(write_png=True, write_svg=False, relpath="branch_plots", dpi=180.0),
		unit_id=91,
	)

	assert outputs["png_path"] == str(out_png)
	assert out_png.exists()
	assert out_png.stat().st_size > 0
	assert len(seen_calls) == 2
	assert [call["color"] for call in seen_calls] == ["#377eb8", "#e41a1c"]
	assert len({call["ax_id"] for call in seen_calls}) == 1
	assert len(seen_savefig_calls) >= 1
	assert seen_savefig_calls[0]["figure_facecolor"] == (0.0, 0.0, 0.0, 1.0)
	assert all(facecolor == (0.0, 0.0, 0.0, 1.0) for facecolor in seen_savefig_calls[0]["axes_facecolors"])
	assert any("b3:" in label and "r^2=" in label and "r=" not in label for label in seen_savefig_calls[0]["legend_labels"])
	assert any("b4:" in label and "r^2=" in label and "r=" not in label for label in seen_savefig_calls[0]["legend_labels"])
	assert all(color == "white" for color in seen_savefig_calls[0]["legend_text_colors"])
	assert seen_savefig_calls[0]["title_fontsize"] == 16.0
	assert seen_savefig_calls[0]["xlabel_text"] == "ms"
	assert seen_savefig_calls[0]["xlabel_fontsize"] == 13.0
	assert seen_savefig_calls[0]["ylabel_text"] == "$\\mu$m"
	assert seen_savefig_calls[0]["ylabel_fontsize"] == 13.0
	assert seen_savefig_calls[0]["tick_fontsizes"] == [11.0]
	assert seen_savefig_calls[0]["legend_anchor_bounds"] is not None
	assert seen_savefig_calls[0]["legend_anchor_bounds"][0] >= 1.0


def test_write_unit_summary_plot_rerenders_into_shared_figure(tmp_path: Path, monkeypatch) -> None:
	pytest.importorskip("axon_velocity")
	import axon_velocity.plotting as av_plotting
	import matplotlib.figure as mpl_figure
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
	gtr = SimpleNamespace(
		branches=[
			{
				"branch_index": 3,
				"channels": [0, 1, 2],
				"velocity": 1.8,
				"offset": 0.2,
				"r2": 0.92,
				"peak_times": [0.8, 1.5, 2.2],
				"distances": [12.0, 25.0, 41.0],
			}
		],
		graph=SimpleNamespace(nodes=lambda: [0, 1, 2]),
	)
	seen: dict[str, object] = {
		"circle_ax_id": None,
		"circle_unit_label_show": None,
		"circle_branch_legend_show": None,
		"prop_ax_ids": [],
		"velocity_ax_ids": [],
		"velocity_title": None,
		"summary_texts": [],
		"savefig_axes_count": None,
		"fig_size_inches": None,
	}
	orig_savefig = mpl_figure.Figure.savefig

	def _spy_render_template_circles_plot(**kwargs):
		ax = kwargs["ax"]
		fig = kwargs["fig"]
		config = kwargs["config"]
		seen["circle_ax_id"] = id(ax)
		seen["circle_unit_label_show"] = bool(config.unit_id_label.show)
		seen["circle_branch_legend_show"] = bool(config.branch_morphology.show_branch_legend)
		fig.patch.set_facecolor("black")
		ax.set_facecolor("black")
		ax.plot([0.0, 1.0], [0.0, 1.0], color="white")
		ax.set_title("Unit 91 circle recon", color="white")
		return {}

	def _spy_plot_template_propagation(template, locations, selected_channels, sort_templates=False, color=None, color_marker=None, ax=None):
		assert ax is not None
		cast_ids = list(seen["prop_ax_ids"])
		cast_ids.append(id(ax))
		seen["prop_ax_ids"] = cast_ids
		ax.plot([0.0, 1.0], [0.0, 1.0], color=color)
		ax.plot([0.0], [0.0], marker="o", color=color_marker)
		ax.axis("off")
		return ax

	def _spy_plot_velocity(peak_times, distances, velocity, offset, color=None, r2=None, ax=None, **kwargs):
		assert ax is not None
		cast_ids = list(seen["velocity_ax_ids"])
		cast_ids.append(id(ax))
		seen["velocity_ax_ids"] = cast_ids
		ax.plot(peak_times, distances, marker="o", ls="", color=color)
		ax.plot(
			[peak_times[0], peak_times[-1]],
			[velocity * peak_times[0] + offset, velocity * peak_times[-1] + offset],
			ls="--",
			color=color,
		)
		return ax

	def _spy_savefig(self, *args, **kwargs):
		seen["savefig_axes_count"] = len(self.axes)
		seen["fig_size_inches"] = tuple(float(value) for value in self.get_size_inches())
		seen["velocity_title"] = self.axes[1].get_title() if len(self.axes) > 1 else None
		seen["summary_texts"] = [text.get_text() for text in self.texts]
		return orig_savefig(self, *args, **kwargs)

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.templates.core.render.render_template_circles_plot",
		_spy_render_template_circles_plot,
	)
	monkeypatch.setattr(av_plotting, "plot_template_propagation", _spy_plot_template_propagation)
	monkeypatch.setattr(av_plotting, "plot_velocity", _spy_plot_velocity)
	monkeypatch.setattr(mpl_figure.Figure, "savefig", _spy_savefig)

	out_png = tmp_path / "unit_summary.png"
	out_svg = tmp_path / "unit_summary.svg"
	outputs = write_unit_summary_plot(
		output_png=out_png,
		output_svg=out_svg,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=CircleReconConfig(
			display=CircleReconDisplayConfig(base="template_circles", branch_scope="clean", show_branch_legend=True),
			output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="circle_recon", dpi=180.0),
		),
		branch_propagation_phase_config=ReconstructionPlotBranchPropagationsPhaseConfig(
			enabled=True,
			branch_scope="clean",
			display=ReconstructionBranchPropagationDisplayConfig(figsize=(2.5, 5.0), sort_templates=False, show_title=True, invert_y_axis=True),
			output=ReconstructionBranchPlotOutputConfig(write_png=True, write_svg=False, relpath="branch_qc/propagations", dpi=180.0),
		),
		branch_velocity_phase_config=ReconstructionPlotBranchVelocitiesPhaseConfig(
			enabled=True,
			branch_scope="clean",
			display=ReconstructionBranchVelocityDisplayConfig(
				figsize=(4.0, 5.0),
				show_title=True,
				title_fontsize=15.0,
				axis_label_fontsize=12.0,
				tick_label_fontsize=10.0,
				show_legend=True,
				legend_fontsize=8.0,
			),
			output=ReconstructionBranchPlotOutputConfig(write_png=True, write_svg=False, relpath="branch_qc/velocities", dpi=180.0),
		),
		branch_colors=ReconstructionBranchColorsConfig(unique_color_per_branch=True, color_scheme="Set1"),
		display_config=ReconstructionUnitSummaryDisplayConfig(
			show_title=False,
			show_summary_unit_label=True,
			summary_unit_label_fontsize=26.0,
			recon_show_unit_label=False,
			recon_show_branch_legend=False,
			velocity_show_title=False,
			show_velocity_legend=True,
			reserve_velocity_legend_space=True,
			velocity_legend_width=2.0,
			top_row_panel_gap_width=0.75,
			top_row_height=7.0,
			propagation_row_height=4.0,
			circle_panel_width=6.0,
			velocity_panel_width=5.0,
			propagation_panel_width=3.0,
		),
		output_config=ReconstructionUnitSummaryOutputConfig(write_png=True, write_svg=False, relpath="reports/unit_summary", dpi=180.0),
		unit_id=91,
	)

	assert outputs["png_path"] == str(out_png)
	assert out_png.exists()
	assert out_png.stat().st_size > 0
	assert seen["circle_ax_id"] is not None
	assert len(list(seen["prop_ax_ids"])) == 1
	assert len(list(seen["velocity_ax_ids"])) == 1
	assert seen["circle_ax_id"] not in list(seen["prop_ax_ids"])
	assert seen["circle_ax_id"] not in list(seen["velocity_ax_ids"])
	assert seen["circle_unit_label_show"] is False
	assert seen["circle_branch_legend_show"] is False
	assert seen["velocity_title"] == ""
	assert "Unit 91" in list(seen["summary_texts"])
	assert seen["savefig_axes_count"] == 3
	assert seen["fig_size_inches"] == pytest.approx((13.75, 11.0))


def test_write_unit_summary_plot_inherits_velocity_legend_from_standalone_config(tmp_path: Path, monkeypatch) -> None:
	pytest.importorskip("axon_velocity")
	import axon_velocity.plotting as av_plotting
	import matplotlib.figure as mpl_figure
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
	gtr = SimpleNamespace(
		branches=[
			{
				"branch_index": 3,
				"channels": [0, 1, 2],
				"velocity": 1.8,
				"offset": 0.2,
				"r2": 0.92,
				"peak_times": [0.8, 1.5, 2.2],
				"distances": [12.0, 25.0, 41.0],
			}
		],
		graph=SimpleNamespace(nodes=lambda: [0, 1, 2]),
	)
	seen: dict[str, object] = {"legend_labels": [], "fig_size_inches": None}
	orig_savefig = mpl_figure.Figure.savefig

	def _spy_render_template_circles_plot(**kwargs):
		ax = kwargs["ax"]
		fig = kwargs["fig"]
		fig.patch.set_facecolor("black")
		ax.set_facecolor("black")
		ax.plot([0.0, 1.0], [0.0, 1.0], color="white")
		return {}

	def _spy_plot_template_propagation(template, locations, selected_channels, sort_templates=False, color=None, color_marker=None, ax=None):
		assert ax is not None
		ax.plot([0.0, 1.0], [0.0, 1.0], color=color)
		ax.plot([0.0], [0.0], marker="o", color=color_marker)
		ax.axis("off")
		return ax

	def _spy_plot_velocity(peak_times, distances, velocity, offset, color=None, r2=None, ax=None, **kwargs):
		assert ax is not None
		ax.plot(peak_times, distances, marker="o", ls="", color=color)
		ax.plot(
			[peak_times[0], peak_times[-1]],
			[velocity * peak_times[0] + offset, velocity * peak_times[-1] + offset],
			ls="--",
			color=color,
		)
		return ax

	def _spy_savefig(self, *args, **kwargs):
		seen["fig_size_inches"] = tuple(float(value) for value in self.get_size_inches())
		legend = self.axes[1].get_legend() if len(self.axes) > 1 else None
		seen["legend_labels"] = [] if legend is None else [text.get_text() for text in legend.get_texts()]
		return orig_savefig(self, *args, **kwargs)

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.templates.core.render.render_template_circles_plot",
		_spy_render_template_circles_plot,
	)
	monkeypatch.setattr(av_plotting, "plot_template_propagation", _spy_plot_template_propagation)
	monkeypatch.setattr(av_plotting, "plot_velocity", _spy_plot_velocity)
	monkeypatch.setattr(mpl_figure.Figure, "savefig", _spy_savefig)

	out_png = tmp_path / "unit_summary_inherit_legend.png"
	out_svg = tmp_path / "unit_summary_inherit_legend.svg"
	outputs = write_unit_summary_plot(
		output_png=out_png,
		output_svg=out_svg,
		template_ch_by_t=template_ch_by_t,
		locs_xy=locs_xy,
		gtr=gtr,
		circle_config=CircleReconConfig(
			display=CircleReconDisplayConfig(base="template_circles", branch_scope="clean", show_branch_legend=True),
			output=CircleReconOutputConfig(write_png=True, write_svg=False, relpath="circle_recon", dpi=180.0),
		),
		branch_propagation_phase_config=ReconstructionPlotBranchPropagationsPhaseConfig(
			enabled=True,
			branch_scope="clean",
			display=ReconstructionBranchPropagationDisplayConfig(figsize=(2.5, 5.0), sort_templates=False, show_title=True, invert_y_axis=True),
			output=ReconstructionBranchPlotOutputConfig(write_png=True, write_svg=False, relpath="branch_qc/propagations", dpi=180.0),
		),
		branch_velocity_phase_config=ReconstructionPlotBranchVelocitiesPhaseConfig(
			enabled=True,
			branch_scope="clean",
			display=ReconstructionBranchVelocityDisplayConfig(figsize=(4.0, 5.0), show_title=True, show_legend=True, legend_fontsize=8.0),
			output=ReconstructionBranchPlotOutputConfig(write_png=True, write_svg=False, relpath="branch_qc/velocities", dpi=180.0),
		),
		branch_colors=ReconstructionBranchColorsConfig(unique_color_per_branch=True, color_scheme="Set1"),
		display_config=ReconstructionUnitSummaryDisplayConfig(
			show_title=False,
			show_velocity_legend=None,
			reserve_velocity_legend_space=None,
			velocity_legend_width=2.0,
			top_row_panel_gap_width=0.75,
			top_row_height=7.0,
			propagation_row_height=4.0,
			circle_panel_width=6.0,
			velocity_panel_width=5.0,
			propagation_panel_width=3.0,
		),
		output_config=ReconstructionUnitSummaryOutputConfig(write_png=True, write_svg=False, relpath="reports/unit_summary", dpi=180.0),
		unit_id=91,
	)

	assert outputs["png_path"] == str(out_png)
	assert out_png.exists()
	assert any("b3:" in label for label in list(seen["legend_labels"]))
	assert seen["fig_size_inches"] == pytest.approx((13.75, 11.0))


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
