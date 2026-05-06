from __future__ import annotations

from pathlib import Path
import re

import numpy as np  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import matplotlib.collections  # type: ignore[import-not-found]

from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_propagation_plot
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_footprint_amplitude_map
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_footprint_map_grid_from_assets
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_template_wf_overlay
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_wf_overlay_grid_from_assets
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_topographical_amplitude_footprint
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _expand_limits_for_glyph_half_size
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _make_square_limits
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _probe_electrode_dims_um
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _limits_for_template_shape
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _maybe_reversed_colormap
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _add_propagation_scale_bars
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _compute_max_non_overlapping_circle_areas
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _convert_latency_samples_to_units
from axon_recon.pipeline.stages.reconstruct.templates.core.render import _ticks_ending_in_0_or_5_with_max
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_template_circles_plot
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_template_plot
from axon_recon.pipeline.stages.reconstruct.templates.core.render import compute_propagation_channel_order
from axon_recon.pipeline.stages.reconstruct.templates.core.render import render_image_grid
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import (
	CenterMostChannelCoordsConfig,
	FootprintMapConfig,
	ProbeGeometryConfig,
	PropagationLatencyMapConfig,
	PropagationPlotConfig,
	TemplateCirclesOverlapControlsConfig,
	TemplateCirclesPlotConfig,
	TemplateCirclesBranchMorphologyConfig,
	TemplateScaleCircleConfig,
	TemplatePlotConfig,
	TemplateWaveformOverlayConfig,
	TimeUpsampleConfig,
	TopographicalFootprintConfig,
	UnitIdLabelConfig,
	FootprintMapGridReportConfig,
	WfOverlayGridReportConfig,
)


def test_render_propagation_plot_respects_panel_chunk_knobs(tmp_path: Path) -> None:
	n_channels = 18
	n_samples = 80
	x = np.linspace(-1.0, 1.0, n_samples)
	t = np.vstack([
		np.sin((i + 2) * x) * (1.0 - (0.03 * i))
		for i in range(n_channels)
	]).astype(float)
	locs = np.column_stack([
		np.linspace(0.0, 170.0, n_channels),
		np.linspace(0.0, 50.0, n_channels),
	])

	png_path = tmp_path / "propagation.png"
	pdf_path = tmp_path / "propagation.pdf"

	outputs = render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=True,
			pdf_relpath="propagation.pdf",
			write_png=True,
			png_relpath="propagation.png",
			show_title=False,
			top_channels=16,
			channels_per_panel=5,
			channel_overlap=2,
			background="white",
			show_electrode_ids=True,
			electrode_label_fontsize=7.5,
			electrode_label_x_offset_frac=0.02,
			electrode_label_y_offset_frac=0.1,
			show_scale_bar=True,
			scale_bar_anchor_x_frac=0.85,
			scale_bar_anchor_y_frac=0.20,
			scale_bar_time_fraction=0.20,
			scale_bar_amp_fraction=0.25,
			scale_bar_linewidth=2.2,
			scale_bar_fontsize=8.0,
		),
		pdf_path=pdf_path,
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert png_path.exists()
	assert pdf_path.exists()
	assert outputs.get("propagation_plot_png") == str(png_path)
	assert outputs.get("propagation_plot_pdf") == str(pdf_path)


def test_render_propagation_plot_uses_left_panel_png_dpi_for_left_png(tmp_path: Path, monkeypatch) -> None:
	t = np.asarray(
		[
			[0.0, -1.0, -2.0, -0.5, 0.0],
			[0.0, -0.8, -1.6, -0.3, 0.0],
		],
		dtype=float,
	)
	locs = np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float)
	png_path = tmp_path / "propagation_dpi.png"
	pdf_path = tmp_path / "propagation_dpi.pdf"

	observed_dpi: list[float] = []
	orig_savefig = plt.Figure.savefig

	def _spy_savefig(self, fname, *args, **kwargs):
		if str(fname).endswith(".png"):
			observed_dpi.append(float(kwargs.get("dpi", 0.0)))
		return orig_savefig(self, fname, *args, **kwargs)

	monkeypatch.setattr(plt.Figure, "savefig", _spy_savefig)

	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			show_right_panel=True,
			left_panel_png_dpi=550.0,
			right_panel_png_dpi=600.0,
			show_title=False,
			top_channels=2,
			channels_per_panel=2,
			channel_overlap=0,
		),
		pdf_path=pdf_path,
		png_path=png_path,
	)

	assert png_path.exists()
	assert len(observed_dpi) >= 1
	assert observed_dpi[0] == 550.0


def test_compute_propagation_channel_order_emits_signed_relative_numbers() -> None:
	template = np.asarray(
		[
			[0.0, -2.0, -0.2, 0.0],
			[0.0, -0.2, -2.5, 0.0],
			[0.0, -0.1, -0.3, -3.5],
		],
		dtype=float,
	)
	cfg = PropagationPlotConfig(
		top_channels=3,
		force_start_with_max_ptp=True,
		force_start_with_max_negative_peak=False,
	)
	out = compute_propagation_channel_order(template_c_by_t=template, config=cfg)
	rel = dict(out["relative_order_by_channel"])
	assert 0 in set(rel.values())
	assert any(v < 0 for v in rel.values())
	assert len(rel) == 3


def test_compute_propagation_channel_order_supports_negative_peak_latency_mode() -> None:
	template = np.asarray(
		[
			[0.0, -5.0, 0.0, 4.0],
			[0.0, -1.0, 0.0, 8.0],
			[0.0, -2.0, -3.0, 0.0],
		],
		dtype=float,
	)
	base_cfg = dict(top_channels=3, force_start_with_max_ptp=False, force_start_with_max_negative_peak=False)
	out_abs = compute_propagation_channel_order(
		template_c_by_t=template,
		config=PropagationPlotConfig(**base_cfg, ordering_latency_mode="abs_peak"),
	)
	out_neg = compute_propagation_channel_order(
		template_c_by_t=template,
		config=PropagationPlotConfig(**base_cfg, ordering_latency_mode="negative_peak"),
	)
	assert out_abs["ordered_channel_indices"].tolist() != out_neg["ordered_channel_indices"].tolist()


def test_compute_propagation_channel_order_tie_breaker_channel_index_is_deterministic() -> None:
	template = np.asarray(
		[
			[0.0, -1.0, 0.0, 0.0],
			[0.0, -1.0, 0.0, 0.0],
			[0.0, 0.0, -2.0, 0.0],
		],
		dtype=float,
	)
	# Channels 0 and 1 share identical latency and amplitude; tie-breaker should keep lower index first.
	out = compute_propagation_channel_order(
		template_c_by_t=template,
		config=PropagationPlotConfig(
			top_channels=3,
			force_start_with_max_ptp=False,
			force_start_with_max_negative_peak=False,
			ordering_latency_mode="negative_peak",
			latency_tie_breaker="channel_index",
		),
	)
	ordered = out["ordered_channel_indices"].tolist()
	assert ordered.index(0) < ordered.index(1)


def test_compute_propagation_channel_order_tie_breaker_input_order_respects_channel_indices() -> None:
	template = np.asarray(
		[
			[0.0, -1.0, 0.0, 0.0],
			[0.0, -1.0, 0.0, 0.0],
			[0.0, 0.0, -2.0, 0.0],
		],
		dtype=float,
	)
	out = compute_propagation_channel_order(
		template_c_by_t=template,
		config=PropagationPlotConfig(
			top_channels=3,
			force_start_with_max_ptp=False,
			force_start_with_max_negative_peak=False,
			ordering_latency_mode="negative_peak",
			latency_tie_breaker="selected_order",
		),
	)
	ordered = out["ordered_channel_indices"].tolist()
	assert ordered.index(0) < ordered.index(1)


def test_render_propagation_plot_order_index_label_mode_renders_order_numbers(tmp_path: Path, monkeypatch) -> None:
	t = np.asarray(
		[
			[0.0, -2.0, -0.2, 0.0],
			[0.0, -0.5, -3.0, 0.0],
			[0.0, -0.1, -1.5, 0.0],
		],
		dtype=float,
	)
	locs = np.asarray([[0.0, 0.0], [17.5, 0.0], [35.0, 0.0]], dtype=float)
	png_path = tmp_path / "propagation_order_labels.png"
	pdf_path = tmp_path / "propagation_order_labels.pdf"

	orig_text = plt.Axes.text
	seen: list[str] = []
	seen_zero_fontweights: list[str] = []

	def _spy_text(self, x, y, s, *args, **kwargs):
		seen.append(str(s))
		if str(s) == "0":
			seen_zero_fontweights.append(str(kwargs.get("fontweight", "normal")))
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(plt.Axes, "text", _spy_text)

	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			show_title=False,
			top_channels=3,
			force_start_with_max_ptp=False,
			force_start_with_max_negative_peak=False,
			channels_per_panel=3,
			channel_overlap=0,
			trace_label_mode="order_index",
			show_scale_bar=False,
			bold_max_amp_electrode_label=True,
		),
		pdf_path=pdf_path,
		png_path=png_path,
		trace_order_label_by_channel={0: 0, 1: 20, 2: 30},
	)

	assert png_path.exists()
	assert "0" in seen
	assert "20" in seen
	assert "30" in seen
	assert "bold" in seen_zero_fontweights


def test_render_propagation_plot_delay_marker_uses_negative_peak_index_in_negative_peak_mode(tmp_path: Path, monkeypatch) -> None:
	t = np.asarray(
		[
			[0.0, -1.0, 7.0, -2.0, 0.0],
		],
		dtype=float,
	)
	locs = np.asarray([[0.0, 0.0]], dtype=float)
	png_path = tmp_path / "propagation_marker_mode.png"
	pdf_path = tmp_path / "propagation_marker_mode.pdf"

	orig_plot = plt.Axes.plot
	vertical_xs: list[float] = []

	def _spy_plot(self, *args, **kwargs):
		if len(args) >= 2:
			x = np.asarray(args[0], dtype=float)
			y = np.asarray(args[1], dtype=float)
			if x.ndim == 1 and y.ndim == 1 and x.size == 2 and y.size == 2 and float(x[0]) == float(x[1]):
				vertical_xs.append(float(x[0]))
		return orig_plot(self, *args, **kwargs)

	monkeypatch.setattr(plt.Axes, "plot", _spy_plot)

	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			show_title=False,
			top_channels=1,
			channels_per_panel=1,
			channel_overlap=0,
			show_scale_bar=False,
			ordering_latency_mode="negative_peak",
		),
		pdf_path=pdf_path,
		png_path=png_path,
	)

	assert png_path.exists()
	assert len(vertical_xs) >= 1
	# Most negative peak for this waveform is at index 3.
	assert 3.0 in vertical_xs


def test_dynamic_circle_sizing_respects_pairwise_non_overlap_constraint() -> None:
	centers_pt = np.asarray(
		[
			[20.0, 20.0],
			[40.0, 20.0],
		],
		dtype=float,
	)
	base_areas = np.asarray([50.0, 50.0], dtype=float)

	sizes = _compute_max_non_overlapping_circle_areas(
		centers_display_pt=centers_pt,
		base_areas_pt2=base_areas,
		axis_x_limits_pt=(0.0, 200.0),
		axis_y_limits_pt=(0.0, 100.0),
	)

	radii = np.sqrt(np.asarray(sizes, dtype=float) / np.pi)
	dist = float(np.hypot(*(centers_pt[1] - centers_pt[0])))
	assert (radii[0] + radii[1]) <= (dist + 1e-6)


def test_dynamic_circle_sizing_preserves_size_ordering() -> None:
	centers_pt = np.asarray(
		[
			[20.0, 20.0],
			[60.0, 20.0],
		],
		dtype=float,
	)
	base_areas = np.asarray([30.0, 60.0], dtype=float)

	sizes = _compute_max_non_overlapping_circle_areas(
		centers_display_pt=centers_pt,
		base_areas_pt2=base_areas,
		axis_x_limits_pt=(0.0, 200.0),
		axis_y_limits_pt=(0.0, 100.0),
	)

	assert float(sizes[1]) > float(sizes[0])


def test_dynamic_circle_sizing_shrinks_when_display_separation_reduces() -> None:
	far_centers_pt = np.asarray(
		[
			[20.0, 20.0],
			[60.0, 20.0],
		],
		dtype=float,
	)
	close_centers_pt = np.asarray(
		[
			[20.0, 20.0],
			[40.0, 20.0],
		],
		dtype=float,
	)
	base_areas = np.asarray([50.0, 50.0], dtype=float)

	far_sizes = _compute_max_non_overlapping_circle_areas(
		centers_display_pt=far_centers_pt,
		base_areas_pt2=base_areas,
		axis_x_limits_pt=(0.0, 100.0),
		axis_y_limits_pt=(0.0, 100.0),
	)
	close_sizes = _compute_max_non_overlapping_circle_areas(
		centers_display_pt=close_centers_pt,
		base_areas_pt2=base_areas,
		axis_x_limits_pt=(0.0, 100.0),
		axis_y_limits_pt=(0.0, 100.0),
	)

	assert np.all(np.asarray(close_sizes, dtype=float) < np.asarray(far_sizes, dtype=float))


def test_dynamic_circle_sizing_handles_asymmetric_axis_scaling() -> None:
	centers_pt = np.asarray(
		[
			[20.0, 50.0],
			[50.0, 50.0],
			[80.0, 50.0],
		],
		dtype=float,
	)
	base_areas = np.asarray([120.0, 120.0, 120.0], dtype=float)

	sizes = _compute_max_non_overlapping_circle_areas(
		centers_display_pt=centers_pt,
		base_areas_pt2=base_areas,
		axis_x_limits_pt=(0.0, 200.0),
		axis_y_limits_pt=(0.0, 80.0),
	)

	radii = np.sqrt(np.asarray(sizes, dtype=float) / np.pi)
	d01 = float(np.hypot(*(centers_pt[1] - centers_pt[0])))
	d12 = float(np.hypot(*(centers_pt[2] - centers_pt[1])))
	assert (radii[0] + radii[1]) <= (d01 + 1e-6)
	assert (radii[1] + radii[2]) <= (d12 + 1e-6)


def test_dynamic_circle_sizing_ignores_nonpositive_edge_clearance() -> None:
	centers_pt = np.asarray(
		[
			[0.0, 0.0],
			[20.0, 0.0],
		],
		dtype=float,
	)
	base_areas = np.asarray([50.0, 50.0], dtype=float)

	sizes = _compute_max_non_overlapping_circle_areas(
		centers_display_pt=centers_pt,
		base_areas_pt2=base_areas,
		axis_x_limits_pt=(1.0, 100.0),
		axis_y_limits_pt=(-20.0, 20.0),
	)

	radii = np.sqrt(np.asarray(sizes, dtype=float) / np.pi)
	dist = float(np.hypot(*(centers_pt[1] - centers_pt[0])))
	assert np.all(np.asarray(sizes, dtype=float) > 0.0)
	assert (radii[0] + radii[1]) <= (dist + 1e-6)


def test_render_template_circles_plot_sets_non_overlapping_sizes_in_final_layout(tmp_path: Path, monkeypatch) -> None:
	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
			[-0.6, -1.5, -0.3, 0.0, 0.1],
			[-0.7, -1.6, -0.2, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
			[36.0, 0.0],
			[54.0, 0.0],
		],
		dtype=float,
	)

	observed = {"checked": False}
	orig_set_sizes = matplotlib.collections.PathCollection.set_sizes

	def _spy_set_sizes(self, sizes, *args, **kwargs):
		result = orig_set_sizes(self, sizes, *args, **kwargs)
		sz = np.asarray(sizes, dtype=float)
		if sz.ndim != 1 or int(sz.size) != int(locations.shape[0]):
			return result
		ax = self.axes
		if ax is None:
			return result
		fig = ax.figure
		if fig is None:
			return result
		offsets = np.asarray(self.get_offsets(), dtype=float)
		if offsets.ndim != 2 or int(offsets.shape[0]) != int(locations.shape[0]):
			return result
		scale = float(72.0 / float(fig.dpi))
		centers_pt = np.asarray(ax.transData.transform(offsets), dtype=float) * scale
		radii_pt = np.sqrt(np.clip(sz, 0.0, None) / np.pi)
		bbox = ax.get_window_extent()
		xmin_pt = float(bbox.x0) * scale
		xmax_pt = float(bbox.x1) * scale
		ymin_pt = float(bbox.y0) * scale
		ymax_pt = float(bbox.y1) * scale
		for i in range(int(centers_pt.shape[0])):
			xi, yi = float(centers_pt[i, 0]), float(centers_pt[i, 1])
			ri = float(radii_pt[i])
			assert (xi - ri) >= (xmin_pt - 1e-6)
			assert (xi + ri) <= (xmax_pt + 1e-6)
			assert (yi - ri) >= (ymin_pt - 1e-6)
			assert (yi + ri) <= (ymax_pt + 1e-6)
		for i in range(int(centers_pt.shape[0])):
			for j in range(i + 1, int(centers_pt.shape[0])):
				d = float(np.hypot(*(centers_pt[j] - centers_pt[i])))
				assert (radii_pt[i] + radii_pt[j]) <= (d + 1e-6)
		observed["checked"] = True
		return result

	monkeypatch.setattr(matplotlib.collections.PathCollection, "set_sizes", _spy_set_sizes)

	png_path = tmp_path / "circles.png"
	svg_path = tmp_path / "circles.svg"
	outputs = render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			dpi=300,
			background="black",
			size_by="amplitude",
			color_by="latency",
		),
		png_path=png_path,
		svg_path=svg_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert png_path.exists()
	assert outputs.get("template_circles_png") == str(png_path)
	assert observed["checked"] is True


def test_render_template_circles_plot_applies_color_bar_tick_fontsize(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
			[-0.6, -1.5, -0.3, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
			[36.0, 0.0],
		],
		dtype=float,
	)

	seen_labelsizes: list[float] = []
	orig_tick_params = matplotlib.axes.Axes.tick_params

	def _spy_tick_params(self, *args, **kwargs):
		if "labelsize" in kwargs and kwargs["labelsize"] is not None:
			seen_labelsizes.append(float(kwargs["labelsize"]))
		return orig_tick_params(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "tick_params", _spy_tick_params)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			background="black",
			color_bar_tick_fontsize=19,
		),
		png_path=tmp_path / "circles_ticksize.png",
		svg_path=tmp_path / "unused.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert any(np.isclose(v, 19.0) for v in seen_labelsizes)


def test_render_template_circles_plot_force_zero_and_neg_latency_to_first_color_range(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes
	import matplotlib.pyplot as plt

	template = np.asarray(
		[
			[0.0, 0.0, -2.0, 0.0],
			[0.0, -1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, -1.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
			[36.0, 0.0],
		],
		dtype=float,
	)

	seen_norms: list[object] = []
	orig_scatter = matplotlib.axes.Axes.scatter

	def _spy_scatter(self, *args, **kwargs):
		norm = kwargs.get("norm", None)
		if norm is not None and hasattr(norm, "vmin") and hasattr(norm, "vmax"):
			seen_norms.append(norm)
		return orig_scatter(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _spy_scatter)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			background="black",
			color_by="latency",
			color_bar_force_zero_and_neg_values_first_color_range=True,
		),
		png_path=tmp_path / "circles_force_zero_first_range.png",
		svg_path=tmp_path / "unused_force_zero_first_range.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert len(seen_norms) >= 1
	norm = seen_norms[0]
	vmin = float(norm.vmin)
	vmax = float(norm.vmax)
	clip = bool(getattr(norm, "clip", False))
	assert vmin < 0.0
	assert vmax > vmin
	assert clip is True

	# Zero should sit exactly at the first/second color-range boundary.
	n_colors = int(max(2, int(getattr(plt.get_cmap("viridis_r"), "N", 256))))
	expected_zero_frac = 1.0 / float(n_colors)
	assert np.isclose(float(norm(0.0)), expected_zero_frac, atol=1e-6)
	assert np.isclose(float(norm(vmin)), 0.0, atol=1e-9)


def test_render_template_circles_plot_zero_transition_contrast_sharpens_boundary(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes
	import matplotlib.pyplot as plt

	template = np.asarray(
		[
			[0.0, 0.0, -2.0, 0.0],
			[0.0, -1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, -1.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
			[36.0, 0.0],
		],
		dtype=float,
	)

	seen_norms: list[object] = []
	orig_scatter = matplotlib.axes.Axes.scatter

	def _spy_scatter(self, *args, **kwargs):
		norm = kwargs.get("norm", None)
		if norm is not None and hasattr(norm, "vmin") and hasattr(norm, "vmax"):
			seen_norms.append(norm)
		return orig_scatter(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _spy_scatter)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			background="black",
			color_by="latency",
			color_bar_force_zero_and_neg_values_first_color_range=True,
			color_bar_zero_transition_contrast=1.0,
		),
		png_path=tmp_path / "circles_zero_contrast_linear.png",
		svg_path=tmp_path / "unused_zero_contrast_linear.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)
	linear_norm = seen_norms[-1]

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			background="black",
			color_by="latency",
			color_bar_force_zero_and_neg_values_first_color_range=True,
			color_bar_zero_transition_contrast=2.0,
		),
		png_path=tmp_path / "circles_zero_contrast_sharp.png",
		svg_path=tmp_path / "unused_zero_contrast_sharp.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)
	sharp_norm = seen_norms[-1]

	n_colors = int(max(2, int(getattr(plt.get_cmap("viridis_r"), "N", 256))))
	first_range_frac = 1.0 / float(n_colors)
	vmin = float(sharp_norm.vmin)
	vmax = float(sharp_norm.vmax)

	# Zero boundary remains fixed at first/second color transition.
	assert np.isclose(float(sharp_norm(0.0)), first_range_frac, atol=1e-6)

	neg_probe = 0.25 * vmin  # negative value closer to zero than vmin
	pos_probe = 0.25 * vmax
	# Sharper contrast should push both sides away from the zero boundary.
	assert float(sharp_norm(neg_probe)) < float(linear_norm(neg_probe))
	assert float(sharp_norm(pos_probe)) > float(linear_norm(pos_probe))


def test_render_template_circles_plot_applies_scale_bar_x_offset_frac(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	positions: list[float] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if str(s).endswith(" um"):
			positions.append(float(x))
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=True,
			scale_bar_length_um=5.0,
			scale_bar_x_offset_frac=0.05,
			scale_bar_horizontal_alignment="left",
		),
		png_path=tmp_path / "circles_bar_left.png",
		svg_path=tmp_path / "unused_left.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=True,
			scale_bar_length_um=5.0,
			scale_bar_x_offset_frac=0.35,
			scale_bar_horizontal_alignment="left",
		),
		png_path=tmp_path / "circles_bar_right.png",
		svg_path=tmp_path / "unused_right.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert len(positions) >= 2
	assert positions[1] > positions[0]


def test_render_template_circles_plot_applies_scale_bar_alignment(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	x_midpoints: list[float] = []
	y_positions: list[float] = []
	orig_plot = matplotlib.axes.Axes.plot

	def _spy_plot(self, *args, **kwargs):
		if len(args) >= 2:
			x = np.asarray(args[0], dtype=float)
			y = np.asarray(args[1], dtype=float)
			if x.ndim == 1 and y.ndim == 1 and x.size == 2 and y.size == 2 and np.isclose(y[0], y[1]):
				x_midpoints.append(float((x[0] + x[1]) / 2.0))
				y_positions.append(float(y[0]))
		return orig_plot(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=True,
			scale_bar_length_um=5.0,
			scale_bar_x_offset_frac=0.05,
			scale_bar_horizontal_alignment="left",
			scale_bar_vertical_alignment="bottom",
		),
		png_path=tmp_path / "circles_bar_align_left_bottom.png",
		svg_path=tmp_path / "unused_align_lb.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)
	x_left_bottom = x_midpoints[-1]
	y_left_bottom = y_positions[-1]

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=True,
			scale_bar_length_um=5.0,
			scale_bar_x_offset_frac=0.05,
			scale_bar_horizontal_alignment="right",
			scale_bar_vertical_alignment="top",
		),
		png_path=tmp_path / "circles_bar_align_right_top.png",
		svg_path=tmp_path / "unused_align_rt.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)
	x_right_top = x_midpoints[-1]
	y_right_top = y_positions[-1]

	assert x_right_top > x_left_bottom
	assert y_right_top > y_left_bottom


def test_render_template_circles_plot_scale_bar_x_offset_can_consider_fontsize(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	x_midpoints: list[float] = []
	orig_plot = matplotlib.axes.Axes.plot

	def _spy_plot(self, *args, **kwargs):
		if len(args) >= 2:
			x = np.asarray(args[0], dtype=float)
			y = np.asarray(args[1], dtype=float)
			if x.ndim == 1 and y.ndim == 1 and x.size == 2 and y.size == 2 and np.isclose(y[0], y[1]):
				x_midpoints.append(float((x[0] + x[1]) / 2.0))
		return orig_plot(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=True,
			scale_bar_length_um=1.0,
			scale_bar_x_offset_frac=0.02,
			scale_bar_horizontal_alignment="right",
			scale_bar_fontsize=40.0,
			scale_bar_x_offset_considers_fontsize=False,
		),
		png_path=tmp_path / "circles_bar_fontsize_off.png",
		svg_path=tmp_path / "unused_fontsize_off.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)
	x_without_font_pad = x_midpoints[-1]

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=True,
			scale_bar_length_um=1.0,
			scale_bar_x_offset_frac=0.02,
			scale_bar_horizontal_alignment="right",
			scale_bar_fontsize=40.0,
			scale_bar_x_offset_considers_fontsize=True,
		),
		png_path=tmp_path / "circles_bar_fontsize_on.png",
		svg_path=tmp_path / "unused_fontsize_on.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)
	x_with_font_pad = x_midpoints[-1]

	# Font-aware mode should push a right-anchored bar inward (toward smaller x) to reserve label width.
	assert x_with_font_pad < x_without_font_pad


def test_render_template_circles_plot_draws_scale_circle(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	seen_scale_circle_patch = {"count": 0}
	seen_scale_circle_labels: list[str] = []
	orig_add_patch = matplotlib.axes.Axes.add_patch
	orig_text = matplotlib.axes.Axes.text

	def _spy_add_patch(self, patch, *args, **kwargs):
		if str(getattr(patch, "get_gid", lambda: "")() or "") == "template_scale_circle_patch":
			seen_scale_circle_patch["count"] += 1
		return orig_add_patch(self, patch, *args, **kwargs)

	def _spy_text(self, x, y, s, *args, **kwargs):
		text_artist = orig_text(self, x, y, s, *args, **kwargs)
		if "uV" in str(s):
			seen_scale_circle_labels.append(str(s))
		return text_artist

	monkeypatch.setattr(matplotlib.axes.Axes, "add_patch", _spy_add_patch)
	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_circle=True,
			scale_circle_color="white",
		),
		png_path=tmp_path / "circles_scale_circle.png",
		svg_path=tmp_path / "unused_scale_circle.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert seen_scale_circle_patch["count"] >= 1
	assert any("uV" in label for label in seen_scale_circle_labels)


def test_render_template_circles_plot_fast_render_clamps_dpi_and_skips_scale_circle(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes
	import matplotlib.figure

	import axon_recon.pipeline.stages.reconstruct.templates.core.render as render_mod

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	seen_scale_circle_patch = {"count": 0}
	seen_savefig_kwargs: list[dict[str, object]] = []
	seen_colorbar_kwargs: list[dict[str, object]] = []
	orig_add_patch = matplotlib.axes.Axes.add_patch
	orig_savefig = matplotlib.figure.Figure.savefig
	orig_colorbar = matplotlib.figure.Figure.colorbar

	def _spy_add_patch(self, patch, *args, **kwargs):
		if str(getattr(patch, "get_gid", lambda: "")() or "") == "template_scale_circle_patch":
			seen_scale_circle_patch["count"] += 1
		return orig_add_patch(self, patch, *args, **kwargs)

	def _spy_savefig(self, *args, **kwargs):
		seen_savefig_kwargs.append(dict(kwargs))
		return orig_savefig(self, *args, **kwargs)

	def _spy_colorbar(self, *args, **kwargs):
		seen_colorbar_kwargs.append(dict(kwargs))
		return orig_colorbar(self, *args, **kwargs)

	def _unexpected_non_overlap_sizing(**kwargs):
		raise AssertionError("fast render should not compute final non-overlapping circle sizes")

	monkeypatch.setattr(matplotlib.axes.Axes, "add_patch", _spy_add_patch)
	monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _spy_savefig)
	monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", _spy_colorbar)
	monkeypatch.setattr(render_mod, "_compute_max_non_overlapping_circle_areas", _unexpected_non_overlap_sizing)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			dpi=420,
			fast_render=True,
			show_scale_circle=True,
			scale_circle_color="white",
			overlap_controls=TemplateCirclesOverlapControlsConfig(
				scalebar_coords_overlap_detect=True,
				scalecircle_channel_overlap_detect=True,
				max_overlap_check_iterations=5,
			),
		),
		png_path=tmp_path / "circles_fast_render.png",
		svg_path=tmp_path / "unused_fast_render.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert float(seen_savefig_kwargs[-1]["dpi"]) == 220.0
	assert "bbox_inches" not in seen_savefig_kwargs[-1]
	assert "boundaries" not in seen_colorbar_kwargs[-1]
	assert "spacing" not in seen_colorbar_kwargs[-1]
	assert seen_scale_circle_patch["count"] == 0


def test_render_template_circles_plot_scale_circle_label_precision_knob(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.25],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	labels: list[str] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		text = str(s)
		if "uV" in text:
			labels.append(text)
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_circle=True,
			scale_circle_color="white",
			scale_circle=TemplateScaleCircleConfig(digits_after_decimal=1),
		),
		png_path=tmp_path / "circles_scale_circle_precision.png",
		svg_path=tmp_path / "unused_scale_circle_precision.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert any(re.match(r"^-?\d+\.\d uV$", label) for label in labels)


def test_render_template_circles_plot_scale_circle_left_top_corner_alignment(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes
	import matplotlib.patches

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	patches_seen: list[tuple[float, float, float, float]] = []
	orig_add_patch = matplotlib.axes.Axes.add_patch

	def _spy_add_patch(self, patch, *args, **kwargs):
		if isinstance(patch, matplotlib.patches.Ellipse) and str(getattr(patch, "get_gid", lambda: "")() or "") == "template_scale_circle_patch":
			center = patch.get_center()
			patches_seen.append((float(center[0]), float(center[1]), float(patch.width), float(patch.height)))
		return orig_add_patch(self, patch, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "add_patch", _spy_add_patch)

	x_off = 0.02
	y_off = 0.00
	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_circle=True,
			scale_circle_color="white",
			scale_circle=TemplateScaleCircleConfig(
				x_offset_frac=x_off,
				y_offset_frac=y_off,
				horizontal_alignment="left",
				vertical_alignment="top",
			),
		),
		png_path=tmp_path / "circles_scale_circle_corner_align.png",
		svg_path=tmp_path / "unused_scale_circle_corner_align.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert len(patches_seen) >= 1
	cx, cy, w, h = patches_seen[-1]
	left_edge = float(cx - (w / 2.0))
	top_edge = float(cy + (h / 2.0))
	assert np.isclose(left_edge, x_off, atol=1e-6)
	assert np.isclose(top_edge, 1.0 - y_off, atol=1e-6)


def test_render_template_circles_plot_scale_circle_right_text_is_outside(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes
	import matplotlib.patches

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	patch_edges: list[float] = []
	text_xs: list[float] = []
	text_has: list[str] = []
	orig_add_patch = matplotlib.axes.Axes.add_patch
	orig_text = matplotlib.axes.Axes.text

	def _spy_add_patch(self, patch, *args, **kwargs):
		if isinstance(patch, matplotlib.patches.Ellipse) and str(getattr(patch, "get_gid", lambda: "")() or "") == "template_scale_circle_patch":
			cx, cy = patch.get_center()
			patch_edges.append(float(cx + (patch.width / 2.0)))
		return orig_add_patch(self, patch, *args, **kwargs)

	def _spy_text(self, x, y, s, *args, **kwargs):
		if "uV" in str(s):
			text_xs.append(float(x))
			text_has.append(str(kwargs.get("horizontalalignment", "")))
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "add_patch", _spy_add_patch)
	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_circle=True,
			scale_circle_color="white",
			scale_circle=TemplateScaleCircleConfig(font_location="right"),
		),
		png_path=tmp_path / "circles_scale_circle_text_right_outside.png",
		svg_path=tmp_path / "unused_scale_circle_text_right_outside.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert len(patch_edges) >= 1
	assert len(text_xs) >= 1
	assert text_has[-1] == "left"
	assert text_xs[-1] > patch_edges[-1]


def test_render_template_circles_plot_amplitude_size_uses_abs_negative_peak(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	# Channel 0 has larger PTP but smaller abs-negative peak than channel 1.
	template = np.asarray(
		[
			[-5.0, 5.0, 0.0],
			[-6.0, 0.0, 0.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	seen_sizes: list[np.ndarray] = []
	orig_scatter = matplotlib.axes.Axes.scatter

	def _spy_scatter(self, *args, **kwargs):
		if "s" in kwargs:
			seen_sizes.append(np.asarray(kwargs["s"], dtype=float))
		return orig_scatter(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _spy_scatter)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			size_by="amplitude",
		),
		png_path=tmp_path / "circles_abs_neg_size_metric.png",
		svg_path=tmp_path / "unused_abs_neg_size_metric.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert len(seen_sizes) >= 1
	# With abs-negative-peak sizing: channel 1 (abs(-6)=6) should be larger than channel 0 (abs(-5)=5).
	assert float(seen_sizes[0][1]) > float(seen_sizes[0][0])


def test_render_template_circles_plot_scale_circle_label_uses_abs_negative_peak(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-5.0, 5.0, 0.0],
			[-6.0, 0.0, 0.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	labels: list[str] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		text = str(s)
		if text.endswith(" uV"):
			labels.append(text)
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_circle=True,
			scale_circle=TemplateScaleCircleConfig(digits_after_decimal=0),
		),
		png_path=tmp_path / "circles_abs_neg_scale_circle_label.png",
		svg_path=tmp_path / "unused_abs_neg_scale_circle_label.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	# abs-negative-peak max is 6 uV (not 10 uV PTP).
	assert any(label == "6 uV" for label in labels)


def test_render_template_circles_plot_scale_circle_style_knobs_apply_to_patch(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes
	import matplotlib.colors as mcolors
	import matplotlib.patches

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
		],
		dtype=float,
	)

	patch_styles: list[dict[str, Any]] = []
	orig_add_patch = matplotlib.axes.Axes.add_patch

	def _spy_add_patch(self, patch, *args, **kwargs):
		if isinstance(patch, matplotlib.patches.Ellipse) and str(getattr(patch, "get_gid", lambda: "")() or "") == "template_scale_circle_patch":
			patch_styles.append(
				{
					"fill": bool(patch.get_fill()),
					"facecolor": patch.get_facecolor(),
					"edgecolor": patch.get_edgecolor(),
					"linewidth": float(patch.get_linewidth()),
				}
			)
		return orig_add_patch(self, patch, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "add_patch", _spy_add_patch)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_circle=True,
			scale_circle_color="yellow",
			scale_circle=TemplateScaleCircleConfig(
				linestyle=None,
				fill=True,
				fill_color="white",
			),
		),
		png_path=tmp_path / "circles_scale_circle_style_knobs.png",
		svg_path=tmp_path / "unused_scale_circle_style_knobs.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert len(patch_styles) >= 1
	style = patch_styles[-1]
	assert style["fill"] is True
	assert np.allclose(style["facecolor"], mcolors.to_rgba("white"))
	assert np.isclose(float(style["edgecolor"][3]), 0.0)
	assert np.isclose(style["linewidth"], 0.0)


def test_render_template_circles_plot_branch_morphology_draws_node_borders_and_clipped_edges(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-2.0, 0.0, 0.0],
			[-1.5, 0.0, 0.0],
			[-1.0, 0.0, 0.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[20.0, 0.0],
			[40.0, 0.0],
		],
		dtype=float,
	)

	border_scatter_calls = {"count": 0}
	edge_plot_calls = {"count": 0}
	orig_scatter = matplotlib.axes.Axes.scatter
	orig_plot = matplotlib.axes.Axes.plot

	def _spy_scatter(self, *args, **kwargs):
		if kwargs.get("facecolors", None) == "none" and "edgecolors" in kwargs:
			border_scatter_calls["count"] += 1
		return orig_scatter(self, *args, **kwargs)

	def _spy_plot(self, *args, **kwargs):
		if np.isclose(float(kwargs.get("zorder", 0.0)), 7.1):
			edge_plot_calls["count"] += 1
		return orig_plot(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _spy_scatter)
	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=False,
			show_scale_circle=False,
			branch_morphology=TemplateCirclesBranchMorphologyConfig(
				enabled=True,
				node_border_linewidth=0.3,
				edge_linewidth=0.9,
			),
		),
		png_path=tmp_path / "circles_branch_overlay.png",
		svg_path=tmp_path / "unused_branch_overlay.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		branch_morphology={
			"branches": [
				{
					"branch_index": 0,
					"channels": [0, 1, 2],
				}
			]
		},
	)

	assert border_scatter_calls["count"] >= 1
	assert edge_plot_calls["count"] >= 1


def test_render_template_circles_plot_branch_morphology_uses_gtr_payload_when_enabled(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-2.0, 0.0, 0.0],
			[-1.5, 0.0, 0.0],
			[-1.0, 0.0, 0.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[20.0, 0.0],
			[40.0, 0.0],
		],
		dtype=float,
	)

	edge_plot_calls = {"count": 0}
	orig_plot = matplotlib.axes.Axes.plot

	def _spy_plot(self, *args, **kwargs):
		if np.isclose(float(kwargs.get("zorder", 0.0)), 7.1):
			edge_plot_calls["count"] += 1
		return orig_plot(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

	class _DummyGtr:
		def __init__(self) -> None:
			self.branches = [{"branch_index": 0, "electrode_ids": [0, 1, 2]}]

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=False,
			show_scale_circle=False,
			branch_morphology=TemplateCirclesBranchMorphologyConfig(
				enabled=True,
				node_border_linewidth=0.3,
				edge_linewidth=0.9,
			),
		),
		png_path=tmp_path / "circles_branch_overlay_from_gtr.png",
		svg_path=tmp_path / "unused_branch_overlay_from_gtr.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		gtr=_DummyGtr(),
	)

	assert edge_plot_calls["count"] >= 1


def test_render_template_circles_plot_branch_outline_preserves_color_scheme_stroke(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-2.0, 0.0, 0.0],
			[-1.5, 0.0, 0.0],
			[-1.0, 0.0, 0.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[20.0, 0.0],
			[40.0, 0.0],
		],
		dtype=float,
	)

	outline_plot_calls: list[dict[str, object]] = []
	main_plot_calls: list[dict[str, object]] = []
	orig_plot = matplotlib.axes.Axes.plot

	def _spy_plot(self, *args, **kwargs):
		zorder = float(kwargs.get("zorder", 0.0))
		if np.isclose(zorder, 7.05):
			outline_plot_calls.append({"color": kwargs.get("color"), "linewidth": kwargs.get("linewidth")})
		elif np.isclose(zorder, 7.1):
			main_plot_calls.append({"color": kwargs.get("color"), "linewidth": kwargs.get("linewidth")})
		return orig_plot(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=False,
			show_scale_circle=False,
			branch_morphology=TemplateCirclesBranchMorphologyConfig(
				enabled=True,
				edge_linewidth=0.9,
				color_scheme="Set1",
				branch_outline_color="white",
				branch_outline_linewidth=1.5,
			),
		),
		png_path=tmp_path / "circles_branch_outline_overlay.png",
		svg_path=tmp_path / "unused_branch_outline_overlay.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		branch_morphology={
			"branches": [
				{
					"branch_index": 0,
					"channels": [0, 1, 2],
				}
			]
		},
	)

	assert len(outline_plot_calls) >= 1
	assert len(main_plot_calls) >= 1
	assert all(call["color"] == "white" for call in outline_plot_calls)
	assert all(np.isclose(float(call["linewidth"]), 3.9) for call in outline_plot_calls)
	assert all(call["color"] != "white" for call in main_plot_calls)
	assert all(np.isclose(float(call["linewidth"]), 0.9) for call in main_plot_calls)


def test_render_template_circles_plot_branch_legend_uses_branch_colors_and_labels(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-2.0, 0.0, 0.0],
			[-1.5, 0.0, 0.0],
			[-1.0, 0.0, 0.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[20.0, 0.0],
			[40.0, 0.0],
		],
		dtype=float,
	)

	legend_calls: list[dict[str, Any]] = []
	orig_legend = matplotlib.axes.Axes.legend

	def _spy_legend(self, *args, **kwargs):
		handles = list(kwargs.get("handles", []))
		legend_calls.append(
			{
				"labels": [str(handle.get_label()) for handle in handles],
				"colors": [handle.get_color() for handle in handles],
				"title": kwargs.get("title"),
				"loc": kwargs.get("loc"),
			}
		)
		return orig_legend(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "legend", _spy_legend)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_scale_bar=False,
			show_scale_circle=False,
			branch_morphology=TemplateCirclesBranchMorphologyConfig(
				enabled=True,
				edge_linewidth=0.9,
				show_branch_legend=True,
				color_scheme="Set1",
			),
		),
		png_path=tmp_path / "circles_branch_legend.png",
		svg_path=tmp_path / "unused_branch_legend.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		branch_morphology={
			"branches": [
				{
					"branch_index": 0,
					"channels": [0, 1],
					"label": "A",
				},
				{
					"branch_index": 1,
					"channels": [1, 2],
					"label": "B",
				},
			]
		},
	)

	assert len(legend_calls) == 1
	assert legend_calls[0]["title"] == "Branches"
	assert legend_calls[0]["labels"] == ["A", "B"]
	assert legend_calls[0]["colors"][0] != legend_calls[0]["colors"][1]
	assert legend_calls[0]["loc"] == "center right"


def test_render_template_circles_plot_keeps_scale_bar_in_bottom_right_when_y_inverted(tmp_path: Path) -> None:
	template = np.asarray(
		[
			[-2.0, 0.0, 0.0],
			[-1.5, 0.0, 0.0],
			[-1.0, 0.0, 0.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[20.0, 0.0],
			[40.0, 0.0],
		],
		dtype=float,
	)

	fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
	try:
		render_template_circles_plot(
			template=template,
			locations_xy=locations,
			config=TemplateCirclesPlotConfig(
				write_png=False,
				write_svg=False,
				invert_y_axis=True,
				show_scale_bar=True,
				show_scale_circle=False,
				background="black",
				unit_id_label=UnitIdLabelConfig(show=True),
			),
			png_path=tmp_path / "unused_scale_bar_inverted.png",
			svg_path=tmp_path / "unused_scale_bar_inverted.svg",
			probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
			unit_id=91,
			fig=fig,
			ax=ax,
			close_figure=False,
		)
		fig.canvas.draw()
		renderer = fig.canvas.get_renderer()
		scale_bar_text = next(
			text for text in ax.texts if str(getattr(text, "get_gid", lambda: "")() or "") == "template_scale_bar_text"
		)
		unit_id_text = next(
			text for text in ax.texts if str(getattr(text, "get_gid", lambda: "")() or "") == "template_unit_id_label"
		)
		scale_bbox = scale_bar_text.get_window_extent(renderer=renderer)
		unit_bbox = unit_id_text.get_window_extent(renderer=renderer)
		axes_bbox = ax.get_window_extent(renderer=renderer)
		scale_center_x = 0.5 * (float(scale_bbox.x0) + float(scale_bbox.x1))
		scale_center_y = 0.5 * (float(scale_bbox.y0) + float(scale_bbox.y1))
		axes_mid_x = float(axes_bbox.x0) + (0.5 * float(axes_bbox.width))
		axes_mid_y = float(axes_bbox.y0) + (0.5 * float(axes_bbox.height))
		unit_center_y = 0.5 * (float(unit_bbox.y0) + float(unit_bbox.y1))

		assert scale_center_x > axes_mid_x
		assert scale_center_y < axes_mid_y
		assert scale_center_y < unit_center_y
	finally:
		plt.close(fig)


def test_render_template_circles_plot_overlap_controls_can_trigger_zoom_out(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray([[0.0, -1.0, 0.0]], dtype=float)
	locations = np.asarray([[0.0, 0.0]], dtype=float)

	xlim_call_count = {"count": 0}
	scalebar_text_xs: list[float] = []
	coords_text_xs: list[float] = []
	orig_set_xlim = matplotlib.axes.Axes.set_xlim
	orig_text = matplotlib.axes.Axes.text

	def _spy_set_xlim(self, *args, **kwargs):
		xlim_call_count["count"] += 1
		return orig_set_xlim(self, *args, **kwargs)

	def _spy_text(self, x, y, s, *args, **kwargs):
		text = str(s)
		if text.endswith(" um"):
			scalebar_text_xs.append(float(x))
		if text.startswith("(") and text.endswith(")") and "," in text:
			coords_text_xs.append(float(x))
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "set_xlim", _spy_set_xlim)
	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			background="black",
			center_most_channel_coords=CenterMostChannelCoordsConfig(
				show=True,
				horizontal_alignment="left",
				vertical_alignment="bottom",
				x_offset_frac=0.02,
				y_offset_frac=0.02,
			),
			unit_id_label=UnitIdLabelConfig(
				show=True,
				fontsize=48,
				horizontal_alignment="center",
				vertical_alignment="center",
				x_offset_frac=0.0,
				y_offset_frac=0.0,
			),
			overlap_controls=TemplateCirclesOverlapControlsConfig(
				unitid_label_channel_overlap_detect=True,
				max_overlap_check_iterations=2,
			),
		),
		png_path=tmp_path / "circles_overlap_controls.png",
		svg_path=tmp_path / "unused.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		unit_id=42,
	)

	# Initial limit set plus at least one overlap-driven expansion call.
	assert xlim_call_count["count"] >= 2
	# Overlays should be re-laid out after zoom-out so their data-space anchor positions update.
	assert len(scalebar_text_xs) >= 2
	assert not np.isclose(scalebar_text_xs[0], scalebar_text_xs[-1])
	assert len(coords_text_xs) >= 2
	assert not np.isclose(coords_text_xs[0], coords_text_xs[-1])


def test_render_template_circles_plot_scale_circle_overlap_controls_can_trigger_zoom_out(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray([[0.0, -1.0, 0.0]], dtype=float)
	locations = np.asarray([[0.0, 0.0]], dtype=float)

	xlim_call_count = {"count": 0}
	orig_set_xlim = matplotlib.axes.Axes.set_xlim

	def _spy_set_xlim(self, *args, **kwargs):
		xlim_call_count["count"] += 1
		return orig_set_xlim(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "set_xlim", _spy_set_xlim)

	render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			background="black",
			show_scale_circle=True,
			scale_circle_color="white",
			overlap_controls=TemplateCirclesOverlapControlsConfig(
				scalecircle_channel_overlap_detect=True,
				max_overlap_check_iterations=2,
			),
		),
		png_path=tmp_path / "circles_scale_circle_overlap_controls.png",
		svg_path=tmp_path / "unused_scale_circle_overlap.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
	)

	assert xlim_call_count["count"] >= 2


def test_render_template_plot_applies_unit_id_center_coords_and_hides_axes(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
			[-0.6, -1.5, -0.3, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
			[36.0, 0.0],
		],
		dtype=float,
	)

	seen_texts: list[str] = []
	seen_axis_off = {"count": 0}
	orig_text = matplotlib.axes.Axes.text
	orig_set_axis_off = matplotlib.axes.Axes.set_axis_off

	def _spy_text(self, x, y, s, *args, **kwargs):
		seen_texts.append(str(s))
		return orig_text(self, x, y, s, *args, **kwargs)

	def _spy_set_axis_off(self, *args, **kwargs):
		seen_axis_off["count"] += 1
		return orig_set_axis_off(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)
	monkeypatch.setattr(matplotlib.axes.Axes, "set_axis_off", _spy_set_axis_off)

	png_path = tmp_path / "template_waveforms.png"
	outputs = render_template_plot(
		template=template,
		locations_xy=locations,
		config=TemplatePlotConfig(
			write_png=True,
			write_svg=False,
			show_axes=False,
			unit_id_label=UnitIdLabelConfig(show=True),
			center_most_channel_coords=CenterMostChannelCoordsConfig(show=True),
		),
		png_path=png_path,
		svg_path=tmp_path / "unused.svg",
		unit_id=42,
	)

	assert outputs.get("template_png") == str(png_path)
	assert png_path.exists()
	assert any(t == "unit 42" for t in seen_texts)
	assert any(t.startswith("(") and t.endswith(")") and "," in t for t in seen_texts)
	assert seen_axis_off["count"] >= 1


def test_render_template_circles_plot_applies_unit_id_center_coords_and_hides_axes(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
			[-0.6, -1.5, -0.3, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
			[36.0, 0.0],
		],
		dtype=float,
	)

	seen_texts: list[str] = []
	seen_axis_off = {"count": 0}
	orig_text = matplotlib.axes.Axes.text
	orig_set_axis_off = matplotlib.axes.Axes.set_axis_off

	def _spy_text(self, x, y, s, *args, **kwargs):
		seen_texts.append(str(s))
		return orig_text(self, x, y, s, *args, **kwargs)

	def _spy_set_axis_off(self, *args, **kwargs):
		seen_axis_off["count"] += 1
		return orig_set_axis_off(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)
	monkeypatch.setattr(matplotlib.axes.Axes, "set_axis_off", _spy_set_axis_off)

	png_path = tmp_path / "template_circles.png"
	outputs = render_template_circles_plot(
		template=template,
		locations_xy=locations,
		config=TemplateCirclesPlotConfig(
			write_png=True,
			write_svg=False,
			show_axes=False,
			unit_id_label=UnitIdLabelConfig(show=True),
			center_most_channel_coords=CenterMostChannelCoordsConfig(show=True),
		),
		png_path=png_path,
		svg_path=tmp_path / "unused.svg",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		unit_id=99,
	)

	assert outputs.get("template_circles_png") == str(png_path)
	assert png_path.exists()
	assert any(t == "unit 99" for t in seen_texts)
	assert any(t.startswith("(") and t.endswith(")") and "," in t for t in seen_texts)
	assert seen_axis_off["count"] >= 1


def test_render_template_plot_center_coords_placed_at_bottom_left_corner(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
			[-0.6, -1.5, -0.3, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
			[36.0, 0.0],
		],
		dtype=float,
	)

	seen_coords_label: dict[str, float | str] = {}
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str) and s.startswith("(") and s.endswith(")") and "," in s:
			seen_coords_label["x"] = float(x)
			seen_coords_label["y"] = float(y)
			seen_coords_label["ha"] = str(kwargs.get("ha", ""))
			seen_coords_label["va"] = str(kwargs.get("va", ""))
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "template_waveforms_direction.png"
	render_template_plot(
		template=template,
		locations_xy=locations,
		config=TemplatePlotConfig(
			write_png=True,
			write_svg=False,
			invert_y_axis=False,
			center_most_channel_coords=CenterMostChannelCoordsConfig(
				show=True,
				horizontal_alignment="left",
				vertical_alignment="bottom",
				x_offset_frac=0.05,
				y_offset_frac=0.05,
			),
		),
		png_path=png_path,
		svg_path=tmp_path / "unused.svg",
	)

	assert png_path.exists()
	assert float(seen_coords_label["x"]) < 5.0
	assert float(seen_coords_label["y"]) < -0.1
	assert seen_coords_label["ha"] == "left"
	assert seen_coords_label["va"] == "bottom"


def test_render_template_plot_inverts_y_axis_when_enabled(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-1.0, -2.0, -0.5, 0.0, 0.2],
			[-0.8, -1.8, -0.4, 0.0, 0.1],
			[-0.6, -1.5, -0.3, 0.0, 0.1],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[18.0, 0.0],
			[36.0, 0.0],
		],
		dtype=float,
	)
	seen = {"count": 0}
	orig_invert_yaxis = matplotlib.axes.Axes.invert_yaxis

	def _spy_invert_yaxis(self, *args, **kwargs):
		seen["count"] += 1
		return orig_invert_yaxis(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "invert_yaxis", _spy_invert_yaxis)

	render_template_plot(
		template=template,
		locations_xy=locations,
		config=TemplatePlotConfig(write_png=True, write_svg=False, invert_y_axis=True),
		png_path=tmp_path / "template_waveforms_inverted.png",
		svg_path=tmp_path / "unused.svg",
	)

	assert seen["count"] >= 1


def test_render_footprint_amplitude_map_inverts_y_axis_when_enabled(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	template = np.asarray(
		[
			[-5.0, -10.0, -3.0],
			[-2.0, -4.0, -1.0],
			[-1.0, -6.0, -2.0],
		],
		dtype=float,
	)
	locations = np.asarray(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)
	seen = {"count": 0}
	orig_invert_yaxis = matplotlib.axes.Axes.invert_yaxis

	def _spy_invert_yaxis(self, *args, **kwargs):
		seen["count"] += 1
		return orig_invert_yaxis(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "invert_yaxis", _spy_invert_yaxis)

	render_footprint_amplitude_map(
		template=template,
		locations_xy=locations,
		config=FootprintMapConfig(write_png=True, write_svg=False, invert_y_axis=True),
		png_path=tmp_path / "footprint_amp_inverted.png",
		svg_path=tmp_path / "unused.svg",
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert seen["count"] >= 1


def test_render_footprint_map_grid_hides_title_when_disabled(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.figure as mpl_figure

	img_path = tmp_path / "unit_0094.png"
	img = np.zeros((8, 8, 3), dtype=np.uint8)
	img[:, :, 1] = 255
	plt.imsave(img_path, img)

	seen_suptitles: list[str] = []
	orig_suptitle = mpl_figure.Figure.suptitle

	def _spy_suptitle(self, t, *args, **kwargs):
		seen_suptitles.append(str(t))
		return orig_suptitle(self, t, *args, **kwargs)

	monkeypatch.setattr(mpl_figure.Figure, "suptitle", _spy_suptitle)

	png_path = tmp_path / "grid.png"
	pdf_path = tmp_path / "grid.pdf"
	outputs = render_footprint_map_grid_from_assets(
		image_paths=[img_path],
		config=FootprintMapGridReportConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="grid.png",
			show_title=False,
		),
		pdf_path=pdf_path,
		png_path=png_path,
		pdf_output_key="grid_pdf",
		png_output_key="grid_png",
		title="Amplitude map grid",
	)

	assert png_path.exists()
	assert outputs.get("grid_png") == str(png_path)
	assert seen_suptitles == []


def test_render_footprint_map_grid_from_assets_returns_empty_without_assets(tmp_path: Path) -> None:
	outputs = render_footprint_map_grid_from_assets(
		image_paths=[],
		config=FootprintMapGridReportConfig(write_pdf=False, write_png=True, show_title=False),
		pdf_path=tmp_path / "unused.pdf",
		png_path=tmp_path / "grid.png",
		pdf_output_key="grid_pdf",
		png_output_key="grid_png",
		title="Amplitude map grid",
	)

	assert outputs == {}


def test_render_wf_overlay_grid_from_assets_returns_empty_without_assets(tmp_path: Path) -> None:
	outputs = render_wf_overlay_grid_from_assets(
		overlay_png_paths=[],
		config=WfOverlayGridReportConfig(write_pdf=False, write_png=True),
		pdf_path=tmp_path / "unused.pdf",
		png_path=tmp_path / "wf_overlay_grid.png",
	)

	assert outputs == {}


def test_render_wf_overlay_grid_from_assets_composes_svg_with_assets(tmp_path: Path) -> None:
	png_path = tmp_path / "unit_94.png"
	svg_panel_path = tmp_path / "unit_94.svg"
	svg_grid_path = tmp_path / "wf_overlay_grid.svg"

	img = np.zeros((10, 10, 3), dtype=np.float32)
	img[:, :, 2] = 1.0
	plt.imsave(png_path, img)
	svg_panel_path.write_text(
		'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10" viewBox="0 0 20 10"><rect x="0" y="0" width="20" height="10" fill="blue"/></svg>',
		encoding="utf-8",
	)

	outputs = render_wf_overlay_grid_from_assets(
		overlay_png_paths=[png_path],
		config=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
		pdf_path=tmp_path / "unused.pdf",
		png_path=tmp_path / "unused.png",
		write_svg=True,
		svg_path=svg_grid_path,
		svg_output_key="wf_grid_svg",
	)

	assert svg_grid_path.exists()
	assert outputs.get("wf_grid_svg") == str(svg_grid_path)


def test_render_footprint_map_grid_from_assets_composes_from_assets(tmp_path: Path) -> None:
	png_path = tmp_path / "unit_94.png"
	svg_path = tmp_path / "unit_94.svg"

	img = np.zeros((10, 10, 3), dtype=np.float32)
	img[:, :, 1] = 1.0
	plt.imsave(png_path, img)
	svg_path.write_text(
		'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10" viewBox="0 0 20 10"><rect x="0" y="0" width="20" height="10" fill="green"/></svg>',
		encoding="utf-8",
	)

	outputs = render_footprint_map_grid_from_assets(
		image_paths=[png_path],
		config=FootprintMapGridReportConfig(
			write_pdf=False,
			write_png=True,
			show_title=False,
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=tmp_path / "circles_grid.png",
		write_svg=True,
		svg_path=tmp_path / "circles_grid.svg",
		svg_output_key="grid_svg",
		pdf_output_key="grid_pdf",
		png_output_key="grid_png",
		title="Circles grid",
	)

	assert outputs.get("grid_png") == str(tmp_path / "circles_grid.png")
	assert outputs.get("grid_svg") == str(tmp_path / "circles_grid.svg")


def test_render_image_grid_composes_svg_from_panel_svgs(tmp_path: Path, monkeypatch) -> None:
	png_a = tmp_path / "unit_a.png"
	png_b = tmp_path / "unit_b.png"
	svg_a = tmp_path / "unit_a.svg"
	svg_b = tmp_path / "unit_b.svg"
	out_svg = tmp_path / "grid.svg"

	img = np.zeros((12, 12, 3), dtype=np.float32)
	img[:, :, 0] = 1.0
	plt.imsave(png_a, img)
	plt.imsave(png_b, img)
	svg_a.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10" viewBox="0 0 20 10"><rect x="0" y="0" width="20" height="10" fill="red"/></svg>', encoding="utf-8")
	svg_b.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="20" viewBox="0 0 10 20"><rect x="0" y="0" width="10" height="20" fill="blue"/></svg>', encoding="utf-8")

	orig_savefig = plt.Figure.savefig

	def _spy_savefig(self, fname, *args, **kwargs):
		fmt = str(kwargs.get("format", "")).lower()
		if str(fname).endswith(".svg") or fmt == "svg":
			raise AssertionError("SVG should be composed from panel SVGs, not saved from matplotlib figure")
		return orig_savefig(self, fname, *args, **kwargs)

	monkeypatch.setattr(plt.Figure, "savefig", _spy_savefig)

	outputs = render_image_grid(
		image_paths=[png_a, png_b],
		write_pdf=False,
		pdf_path=tmp_path / "unused.pdf",
		pdf_output_key="grid_pdf",
		write_png=False,
		png_path=tmp_path / "unused.png",
		png_output_key="grid_png",
		write_svg=True,
		svg_path=out_svg,
		svg_output_key="grid_svg",
		title="Composed Grid",
		show_title=True,
		dpi=180,
	)

	assert out_svg.exists()
	assert outputs.get("grid_svg") == str(out_svg)
	text = out_svg.read_text(encoding="utf-8")
	assert "Composed Grid" in text




def test_render_propagation_plot_without_latency_map_uses_single_column_layout(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.figure as mpl_figure

	n_channels = 10
	n_samples = 50
	x = np.linspace(-1.0, 1.0, n_samples)
	t = np.vstack([np.sin((i + 1) * x) for i in range(n_channels)]).astype(float)
	locs = np.column_stack([
		np.linspace(0.0, 90.0, n_channels),
		np.linspace(0.0, 20.0, n_channels),
	])

	seen: dict[str, int] = {}
	orig_add_gridspec = mpl_figure.Figure.add_gridspec

	def _spy_add_gridspec(self, *args, **kwargs):
		ncols = kwargs.get("ncols")
		if ncols is None and len(args) >= 2:
			ncols = int(args[1])
		if ncols is not None:
			seen["ncols"] = int(ncols)
		return orig_add_gridspec(self, *args, **kwargs)

	monkeypatch.setattr(mpl_figure.Figure, "add_gridspec", _spy_add_gridspec)

	png_path = tmp_path / "propagation_no_map.png"
	outputs = render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_no_map.png",
			top_channels=8,
			channels_per_panel=4,
			channel_overlap=1,
			latency_map=PropagationLatencyMapConfig(show=False),
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert seen.get("ncols") == 1
	assert png_path.exists()
	assert outputs.get("propagation_plot_png") == str(png_path)


def test_render_propagation_plot_accepts_negative_electrode_label_x_offset(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	n_channels = 6
	n_samples = 40
	t = np.vstack([np.sin((i + 1) * np.linspace(-1.0, 1.0, n_samples)) for i in range(n_channels)]).astype(float)
	locs = np.column_stack([
		np.linspace(0.0, 50.0, n_channels),
		np.linspace(0.0, 10.0, n_channels),
	])

	seen_text_x: list[float] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str) and s.startswith("eid "):
			seen_text_x.append(float(x))
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "propagation_negative_label_offset.png"
	outputs = render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_negative_label_offset.png",
			top_channels=6,
			channels_per_panel=6,
			channel_overlap=0,
			show_electrode_ids=True,
			electrode_label_x_offset_frac=-0.15,
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert png_path.exists()
	assert outputs.get("propagation_plot_png") == str(png_path)
	assert len(seen_text_x) > 0
	assert min(seen_text_x) < 0.0


def test_render_propagation_plot_honors_electrode_label_alignment(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	n_channels = 4
	n_samples = 30
	t = np.vstack([np.sin((i + 1) * np.linspace(-1.0, 1.0, n_samples)) for i in range(n_channels)]).astype(float)
	locs = np.column_stack([
		np.linspace(0.0, 30.0, n_channels),
		np.linspace(0.0, 8.0, n_channels),
	])

	seen_alignments: list[str] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str) and s.startswith("eid "):
			seen_alignments.append(str(kwargs.get("horizontalalignment", "")))
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "propagation_label_alignment.png"
	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_label_alignment.png",
			top_channels=4,
			channels_per_panel=4,
			channel_overlap=0,
			show_electrode_ids=True,
			electrode_label_alignment="right",
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert png_path.exists()
	assert len(seen_alignments) > 0
	assert all(a == "right" for a in seen_alignments)


def test_render_propagation_plot_abbreviates_post_ap_signal_with_marker(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	n_channels = 3
	n_samples = 120
	t = np.zeros((n_channels, n_samples), dtype=float)
	for i in range(n_channels):
		t[i, :] = 0.15 * np.sin(np.linspace(0.0, 6.0, n_samples))
		t[i, 10] = -3.0 - float(i)
	locs = np.column_stack([
		np.linspace(0.0, 20.0, n_channels),
		np.linspace(0.0, 10.0, n_channels),
	])

	seen_trace_x: list[np.ndarray] = []
	seen_texts: list[str] = []
	orig_plot = matplotlib.axes.Axes.plot
	orig_text = matplotlib.axes.Axes.text

	def _spy_plot(self, xdata, ydata, *args, **kwargs):
		x_arr = np.asarray(xdata, dtype=float)
		y_arr = np.asarray(ydata, dtype=float)
		if x_arr.ndim == 1 and y_arr.ndim == 1 and x_arr.size > 3:
			seen_trace_x.append(x_arr)
		return orig_plot(self, xdata, ydata, *args, **kwargs)

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str):
			seen_texts.append(s)
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)
	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "propagation_abbrev.png"
	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_abbrev.png",
			top_channels=3,
			channels_per_panel=3,
			channel_overlap=0,
			show_scale_bar=False,
			abbreviate_post_ap_signal=True,
			post_ap_abbrev_start_ms=1.0,
			post_ap_abbrev_cut_fraction=0.5,
			post_ap_abbrev_gap_samples=8,
			post_ap_abbrev_marker_text="/.../",
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10000.0, pitch_um=17.5),
	)

	assert png_path.exists()
	assert len(seen_trace_x) > 0
	# Original span would reach sample 119; with abbreviation this is compressed.
	assert max(float(np.max(arr)) for arr in seen_trace_x) < 110.0
	marker_count = sum(1 for s in seen_texts if str(s).strip() == "/.../")
	assert marker_count >= 3


def test_render_propagation_plot_abbrev_uses_sample_fallback_when_ms_start_out_of_range(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	n_channels = 3
	n_samples = 95
	t = np.zeros((n_channels, n_samples), dtype=float)
	for i in range(n_channels):
		t[i, :] = 0.1 * np.sin(np.linspace(0.0, 8.0, n_samples))
		t[i, 15] = -2.0 - float(i)
	locs = np.column_stack([
		np.linspace(0.0, 20.0, n_channels),
		np.linspace(0.0, 10.0, n_channels),
	])

	seen_trace_x: list[np.ndarray] = []
	orig_plot = matplotlib.axes.Axes.plot

	def _spy_plot(self, xdata, ydata, *args, **kwargs):
		x_arr = np.asarray(xdata, dtype=float)
		y_arr = np.asarray(ydata, dtype=float)
		if x_arr.ndim == 1 and y_arr.ndim == 1 and x_arr.size > 3:
			seen_trace_x.append(x_arr)
		return orig_plot(self, xdata, ydata, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

	png_path = tmp_path / "propagation_abbrev_fallback.png"
	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_abbrev_fallback.png",
			top_channels=3,
			channels_per_panel=3,
			channel_overlap=0,
			show_scale_bar=False,
			abbreviate_post_ap_signal=True,
			post_ap_abbrev_start_ms=2.5,
			post_ap_abbrev_start_samples=10,
			post_ap_abbrev_cut_fraction=0.8,
			post_ap_abbrev_gap_samples=8,
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0, pitch_um=17.5),
	)

	assert png_path.exists()
	assert len(seen_trace_x) > 0
	# Without fallback the trace would stay near full width; with fallback it compresses noticeably.
	assert max(float(np.max(arr)) for arr in seen_trace_x) < 90.0


def test_render_propagation_plot_shows_duration_info_text(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	n_channels = 3
	n_samples = 100
	t = np.zeros((n_channels, n_samples), dtype=float)
	for i in range(n_channels):
		t[i, :] = 0.05 * np.sin(np.linspace(0.0, 6.0, n_samples))
		t[i, 20] = -2.0 - float(i)
	locs = np.column_stack([
		np.linspace(0.0, 20.0, n_channels),
		np.linspace(0.0, 10.0, n_channels),
	])

	seen_texts: list[str] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str):
			seen_texts.append(s)
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "propagation_duration_info.png"
	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_duration_info.png",
			top_channels=3,
			channels_per_panel=3,
			channel_overlap=0,
			show_scale_bar=False,
			show_duration_info=True,
			duration_info_x_frac=0.7,
			duration_info_y_frac=0.95,
			duration_info_horizontal_alignment="right",
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10000.0, pitch_um=17.5),
	)

	assert png_path.exists()
	joined = "\n".join(seen_texts).lower()
	assert "before:" in joined
	assert "after:" in joined
	assert "total:" in joined
	assert "ms" in joined


def test_render_propagation_plot_layout_accepts_area_aspect_ratio_knob(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.pyplot as plt

	n_channels = 6
	n_samples = 60
	t = np.vstack([np.sin((i + 1) * np.linspace(-1.0, 1.0, n_samples)) for i in range(n_channels)]).astype(float)
	locs = np.column_stack([
		np.linspace(0.0, 50.0, n_channels),
		np.linspace(0.0, 10.0, n_channels),
	])

	seen_sizes: list[tuple[float, float]] = []
	orig_figure = plt.figure

	def _spy_figure(*args, **kwargs):
		size = kwargs.get("figsize", None)
		if isinstance(size, tuple) and len(size) == 2:
			seen_sizes.append((float(size[0]), float(size[1])))
		return orig_figure(*args, **kwargs)

	monkeypatch.setattr(plt, "figure", _spy_figure)

	png_path = tmp_path / "propagation_layout_ratio.png"
	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_layout_ratio.png",
			top_channels=6,
			channels_per_panel=3,
			channel_overlap=0,
			show_scale_bar=False,
			plot_width_in=12.0,
			plot_area_aspect_ratio=6.0,
			plot_extra_height_in=0.8,
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert png_path.exists()
	assert len(seen_sizes) >= 1
	# n_panels=2, panel_height=12/6=2.0, total=2*2.0 + 0.8 = 4.8
	w, h = seen_sizes[-1]
	assert np.isclose(w, 12.0)
	assert np.isclose(h, 4.8)


def test_render_template_wf_overlay_defaults_hide_title_axes_and_channel_labels(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	t = np.asarray(
		[
			[0.0, -2.0, 0.5, 0.0, 0.0, 0.0],
			[0.0, -1.0, 0.4, 0.0, 0.0, 0.0],
		],
		dtype=float,
	)

	seen_channel_labels: list[str] = []
	title_calls = 0
	xlabel_calls = 0
	ylabel_calls = 0

	orig_text = matplotlib.axes.Axes.text
	orig_set_title = matplotlib.axes.Axes.set_title
	orig_set_xlabel = matplotlib.axes.Axes.set_xlabel
	orig_set_ylabel = matplotlib.axes.Axes.set_ylabel

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str) and s.startswith("eid "):
			seen_channel_labels.append(s)
		return orig_text(self, x, y, s, *args, **kwargs)

	def _spy_set_title(self, *args, **kwargs):
		nonlocal title_calls
		title_calls += 1
		return orig_set_title(self, *args, **kwargs)

	def _spy_set_xlabel(self, *args, **kwargs):
		nonlocal xlabel_calls
		xlabel_calls += 1
		return orig_set_xlabel(self, *args, **kwargs)

	def _spy_set_ylabel(self, *args, **kwargs):
		nonlocal ylabel_calls
		ylabel_calls += 1
		return orig_set_ylabel(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)
	monkeypatch.setattr(matplotlib.axes.Axes, "set_title", _spy_set_title)
	monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", _spy_set_xlabel)
	monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", _spy_set_ylabel)

	png_path = tmp_path / "overlay_defaults.png"
	outputs = render_template_wf_overlay(
		template=t,
		config=TemplateWaveformOverlayConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="overlay_defaults.png",
			include_mean=False,
			include_scale_bar=False,
		),
		time_upsample=TimeUpsampleConfig(enabled=False, factor=1, method="sinc"),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10000.0),
	)

	assert png_path.exists()
	assert outputs.get("template_wf_overlay_png") == str(png_path)
	assert len(seen_channel_labels) == 0
	assert title_calls == 0
	assert xlabel_calls == 0
	assert ylabel_calls == 0


def test_render_template_wf_overlay_scale_bar_labels_use_ms_and_uv_with_sampling_rate(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	t = np.asarray(
		[
			[0.0, -2.0, 0.5, 0.0, 0.0, 0.0],
			[0.0, -1.0, 0.4, 0.0, 0.0, 0.0],
		],
		dtype=float,
	)

	seen_texts: list[str] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str):
			seen_texts.append(s)
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "overlay_scalebar_units.png"
	render_template_wf_overlay(
		template=t,
		config=TemplateWaveformOverlayConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="overlay_scalebar_units.png",
			include_mean=False,
			include_scale_bar=True,
			show_channel_labels=False,
		),
		time_upsample=TimeUpsampleConfig(enabled=True, factor=10, method="sinc"),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10000.0),
	)

	assert png_path.exists()
	assert any("ms" in s.lower() for s in seen_texts)
	assert any("uv" in s.lower() for s in seen_texts)
	assert all("samples" not in s.lower() for s in seen_texts)


def test_render_template_wf_overlay_hides_mean_label_when_channel_labels_hidden(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	t = np.asarray(
		[
			[0.0, -2.0, 0.5, 0.0, 0.0, 0.0],
			[0.0, -1.0, 0.4, 0.0, 0.0, 0.0],
		],
		dtype=float,
	)

	seen_texts: list[str] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str):
			seen_texts.append(s)
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "overlay_no_mean_label.png"
	render_template_wf_overlay(
		template=t,
		config=TemplateWaveformOverlayConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="overlay_no_mean_label.png",
			include_mean=True,
			include_scale_bar=False,
			show_channel_labels=False,
		),
		time_upsample=TimeUpsampleConfig(enabled=False, factor=1, method="sinc"),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10000.0),
	)

	assert png_path.exists()
	assert all(str(s).strip().lower() != "mean" for s in seen_texts)


def test_render_template_wf_overlay_waveform_info_text_uses_top_channel_and_counts(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	wf = np.tile(np.linspace(-2.0, 1.0, 30, dtype=float), (200, 1))
	template = np.vstack([np.linspace(-1.0, 0.0, 30), np.linspace(-0.5, 0.0, 30)]).astype(float)

	seen_texts: list[str] = []
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str):
			seen_texts.append(s)
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "overlay_waveform_info.png"
	render_template_wf_overlay(
		template=template,
		config=TemplateWaveformOverlayConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="overlay_waveform_info.png",
			include_mean=False,
			max_waveforms_to_show=50,
			show_top_channel_info=True,
			show_waveform_count_info=True,
		),
		time_upsample=TimeUpsampleConfig(enabled=False, factor=1, method="sinc"),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10000.0),
		waveform_traces=wf,
		top_electrode_id=11,
		total_waveforms_at_channel=200,
	)

	assert png_path.exists()
	joined = "\n".join(seen_texts).lower()
	assert "extremum eid: 11" in joined
	assert "wfs at eid: 200" in joined
	assert "wfs shown: 50" in joined


def test_render_template_wf_overlay_mean_uses_all_waveforms_not_display_subset(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	# Mean of all waveforms at sample 0 is 5.0; subset mean would be 0.0.
	wf = np.asarray(
		[
			[0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0],
			[10.0, 10.0, 10.0, 10.0],
			[10.0, 10.0, 10.0, 10.0],
		],
		dtype=float,
	)
	template = np.vstack([np.linspace(-1.0, 0.0, 4), np.linspace(-0.5, 0.0, 4)]).astype(float)

	mean_line_first_y: list[float] = []
	orig_plot = matplotlib.axes.Axes.plot

	def _spy_plot(self, xdata, ydata, *args, **kwargs):
		if str(kwargs.get("color", "")) == "red" and float(kwargs.get("linewidth", 0.0)) >= 1.4:
			y_arr = np.asarray(ydata, dtype=float)
			if y_arr.ndim == 1 and y_arr.size > 0:
				mean_line_first_y.append(float(y_arr[0]))
		return orig_plot(self, xdata, ydata, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

	png_path = tmp_path / "overlay_mean_all.png"
	render_template_wf_overlay(
		template=template,
		config=TemplateWaveformOverlayConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="overlay_mean_all.png",
			include_mean=True,
			max_waveforms_to_show=2,
			waveform_sampling_mode="first",
			show_channel_labels=False,
			include_scale_bar=False,
		),
		time_upsample=TimeUpsampleConfig(enabled=False, factor=1, method="sinc"),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10000.0),
		waveform_traces=wf,
		top_electrode_id=2,
		total_waveforms_at_channel=4,
	)

	assert png_path.exists()
	assert len(mean_line_first_y) >= 1
	assert np.isclose(mean_line_first_y[-1], 5.0)


def test_render_propagation_plot_bolds_max_amplitude_electrode_label(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	n_channels = 4
	n_samples = 40
	t = np.asarray(
		[
			[0.0, -1.0, 0.5, 0.0] * 10,
			[0.0, -8.0, 1.0, 0.0] * 10,
			[0.0, -3.0, 1.0, 0.0] * 10,
			[0.0, -2.0, 1.0, 0.0] * 10,
		],
		dtype=float,
	)
	locs = np.column_stack([
		np.linspace(0.0, 30.0, n_channels),
		np.linspace(0.0, 10.0, n_channels),
	])

	seen_label_weights: dict[int, str] = {}
	orig_text = matplotlib.axes.Axes.text

	def _spy_text(self, x, y, s, *args, **kwargs):
		if isinstance(s, str) and s.startswith("eid "):
			try:
				ch = int(s.split(" ")[1])
			except Exception:
				ch = -1
			seen_label_weights[ch] = str(kwargs.get("fontweight", "normal"))
		return orig_text(self, x, y, s, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

	png_path = tmp_path / "propagation_bold_max_channel.png"
	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_bold_max_channel.png",
			top_channels=4,
			channels_per_panel=4,
			channel_overlap=0,
			show_electrode_ids=True,
			bold_max_amp_electrode_label=True,
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert png_path.exists()
	assert seen_label_weights.get(1) == "bold"
	assert seen_label_weights.get(0) == "normal"
	assert seen_label_weights.get(2) == "normal"
	assert seen_label_weights.get(3) == "normal"


def test_render_propagation_plot_logs_pre_gain_channel_max_amplitudes(tmp_path: Path, caplog) -> None:
	import logging

	n_channels = 3
	n_samples = 6
	t = np.asarray(
		[
			[0.0, -5.0, 1.0, 2.0, -1.0, 0.0],
			[0.0, -2.0, 1.0, -1.0, 0.0, 0.0],
			[0.0, 3.0, -1.0, 0.0, 0.0, 0.0],
		],
		dtype=float,
	)
	locs = np.column_stack([
		np.linspace(0.0, 20.0, n_channels),
		np.linspace(0.0, 10.0, n_channels),
	])

	png_path = tmp_path / "propagation_debug_log.png"

	with caplog.at_level(logging.INFO, logger="axon_recon.templates.render"):
		render_propagation_plot(
			template=t,
			locations_xy=locs,
			config=PropagationPlotConfig(
				write_pdf=False,
				write_png=True,
				png_relpath="propagation_debug_log.png",
				top_channels=3,
				channels_per_panel=3,
				channel_overlap=0,
				trace_gain=2.0,
				debug_max_amps_at_each_channel=True,
			),
			pdf_path=tmp_path / "unused.pdf",
			png_path=png_path,
			probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
		)

	assert png_path.exists()
	message = "\n".join(record.getMessage() for record in caplog.records)
	assert "before gain" in message
	assert "0: 5.0" in message
	assert "1: 2.0" in message
	assert "2: 3.0" in message


def test_render_propagation_plot_peak_marker_knobs_control_height_and_thickness(tmp_path: Path, monkeypatch) -> None:
	import matplotlib.axes

	n_channels = 3
	n_samples = 60
	x = np.linspace(-1.0, 1.0, n_samples)
	t = np.vstack([
		np.sin((i + 1) * x) * (2.0 + i)
		for i in range(n_channels)
	]).astype(float)
	locs = np.column_stack([
		np.linspace(0.0, 20.0, n_channels),
		np.linspace(0.0, 10.0, n_channels),
	])

	orig_plot = matplotlib.axes.Axes.plot
	marker_heights_small: list[float] = []
	marker_widths_small: list[float] = []
	marker_heights_large: list[float] = []
	marker_widths_large: list[float] = []

	active_bucket = "small"

	def _spy_plot(self, xdata, ydata, *args, **kwargs):
		x_arr = np.asarray(xdata)
		y_arr = np.asarray(ydata)
		if x_arr.shape == (2,) and y_arr.shape == (2,) and np.isclose(float(x_arr[0]), float(x_arr[1])) and str(kwargs.get("color", "")) == "black":
			h = float(abs(float(y_arr[1]) - float(y_arr[0])))
			lw = float(kwargs.get("linewidth", 0.0))
			if active_bucket == "small":
				marker_heights_small.append(h)
				marker_widths_small.append(lw)
			else:
				marker_heights_large.append(h)
				marker_widths_large.append(lw)
		return orig_plot(self, xdata, ydata, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_peak_markers_small.png",
			top_channels=3,
			channels_per_panel=3,
			channel_overlap=0,
			show_scale_bar=False,
			peak_marker_height_frac=0.10,
			peak_marker_linewidth=0.8,
		),
		pdf_path=tmp_path / "unused_small.pdf",
		png_path=tmp_path / "propagation_peak_markers_small.png",
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	active_bucket = "large"
	render_propagation_plot(
		template=t,
		locations_xy=locs,
		config=PropagationPlotConfig(
			write_pdf=False,
			write_png=True,
			png_relpath="propagation_peak_markers_large.png",
			top_channels=3,
			channels_per_panel=3,
			channel_overlap=0,
			show_scale_bar=False,
			peak_marker_height_frac=0.30,
			peak_marker_linewidth=2.4,
		),
		pdf_path=tmp_path / "unused_large.pdf",
		png_path=tmp_path / "propagation_peak_markers_large.png",
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert len(marker_heights_small) > 0
	assert len(marker_heights_large) > 0
	assert len(marker_widths_small) > 0
	assert len(marker_widths_large) > 0
	assert max(marker_heights_large) > max(marker_heights_small)
	assert max(marker_widths_large) > max(marker_widths_small)


def test_render_topographical_amplitude_footprint_linear_mode_writes_png(tmp_path: Path) -> None:
	template = np.asarray(
		[
			[-5.0, -10.0, -3.0],
			[-2.0, -4.0, -1.0],
			[-1.0, -6.0, -2.0],
		],
		dtype=float,
	)
	locs = np.asarray(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)
	png_path = tmp_path / "topo_amp.png"

	outputs = render_topographical_amplitude_footprint(
		template=template,
		locations_xy=locs,
		config=TopographicalFootprintConfig(
			write_png=True,
			write_svg=False,
			relpath="topo_amp",
			scale="linear",
			show_color_bar=True,
		),
		png_path=png_path,
		svg_path=tmp_path / "unused.svg",
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert png_path.exists()
	assert outputs.get("topographical_amplitude_footprint_png") == str(png_path)


def test_render_topographical_amplitude_footprint_inverts_y_axis_when_enabled(tmp_path: Path, monkeypatch) -> None:
	from mpl_toolkits.mplot3d.axes3d import Axes3D

	template = np.asarray(
		[
			[-5.0, -10.0, -3.0],
			[-2.0, -4.0, -1.0],
			[-1.0, -6.0, -2.0],
		],
		dtype=float,
	)
	locs = np.asarray(
		[
			[0.0, 0.0],
			[17.5, 0.0],
			[0.0, 17.5],
		],
		dtype=float,
	)
	seen = {"count": 0}
	orig_invert_yaxis = Axes3D.invert_yaxis

	def _spy_invert_yaxis(self, *args, **kwargs):
		seen["count"] += 1
		return orig_invert_yaxis(self, *args, **kwargs)

	monkeypatch.setattr(Axes3D, "invert_yaxis", _spy_invert_yaxis)

	render_topographical_amplitude_footprint(
		template=template,
		locations_xy=locs,
		config=TopographicalFootprintConfig(write_png=True, write_svg=False, invert_y_axis=True),
		png_path=tmp_path / "topo_amp_inverted.png",
		svg_path=tmp_path / "unused.svg",
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert seen["count"] >= 1


def test_limits_for_template_shape_square_enforces_equal_span() -> None:
	locs = np.asarray(
		[
			[0.0, 0.0],
			[30.0, 0.0],
			[30.0, 10.0],
			[5.0, 10.0],
		],
		dtype=float,
	)
	xmin, xmax, ymin, ymax = _limits_for_template_shape(locs, template_shape="square")
	assert np.isclose(float(xmax - xmin), float(ymax - ymin))


def test_limits_for_template_shape_full_keeps_rectangular_span() -> None:
	locs = np.asarray(
		[
			[0.0, 0.0],
			[30.0, 0.0],
			[30.0, 10.0],
			[5.0, 10.0],
		],
		dtype=float,
	)
	xmin, xmax, ymin, ymax = _limits_for_template_shape(locs, template_shape="full")
	assert not np.isclose(float(xmax - xmin), float(ymax - ymin))


def test_probe_electrode_dims_um_returns_square_side() -> None:
	dims = _probe_electrode_dims_um(
		ProbeGeometryConfig(
			pitch_um=17.5,
			electrode_size_um_x=10.0,
			electrode_size_um_y=40.0,
		)
	)
	assert dims is not None
	assert np.isclose(dims[0], dims[1])
	assert np.isclose(float(dims[0]), 17.5)


def test_expand_limits_for_glyph_half_size() -> None:
	xmin, xmax, ymin, ymax = _expand_limits_for_glyph_half_size(
		xmin=0.0,
		xmax=10.0,
		ymin=20.0,
		ymax=40.0,
		half_dx=2.0,
		half_dy=3.0,
	)
	assert np.isclose(xmin, -2.0)
	assert np.isclose(xmax, 12.0)
	assert np.isclose(ymin, 17.0)
	assert np.isclose(ymax, 43.0)


def test_make_square_limits_with_center_keeps_original_bounds_visible() -> None:
	xmin, xmax, ymin, ymax = _make_square_limits(
		xmin=0.0,
		xmax=100.0,
		ymin=0.0,
		ymax=10.0,
		center_xy=(0.0, 0.0),
	)
	assert xmin <= 0.0
	assert xmax >= 100.0
	assert ymin <= 0.0
	assert ymax >= 10.0
	assert np.isclose(float(xmax - xmin), float(ymax - ymin))


def test_render_template_circles_plot_respects_explicit_scope_points_for_zoom(monkeypatch, tmp_path: Path) -> None:
	import matplotlib.axes  # type: ignore[import-not-found]

	template = np.asarray(
		[
			[-5.0, -10.0, -3.0, -1.0],
			[-2.0, -4.0, -1.0, -0.5],
			[-1.0, -6.0, -2.0, -0.5],
		],
		dtype=float,
	)
	locs = np.asarray(
		[
			[0.0, 0.0],
			[10.0, 10.0],
			[100.0, 100.0],
		],
		dtype=float,
	)
	scope_locs = np.asarray(
		[
			[0.0, 0.0],
			[10.0, 10.0],
		],
		dtype=float,
	)

	seen_xlims: list[tuple[float, float]] = []
	seen_ylims: list[tuple[float, float]] = []
	orig_set_xlim = matplotlib.axes.Axes.set_xlim
	orig_set_ylim = matplotlib.axes.Axes.set_ylim

	def _spy_set_xlim(self, *args, **kwargs):
		if len(args) >= 2 and str(self.get_xlabel()) == "x (um)" and str(self.get_ylabel()) == "y (um)":
			seen_xlims.append((float(args[0]), float(args[1])))
		return orig_set_xlim(self, *args, **kwargs)

	def _spy_set_ylim(self, *args, **kwargs):
		if len(args) >= 2 and str(self.get_xlabel()) == "x (um)" and str(self.get_ylabel()) == "y (um)":
			seen_ylims.append((float(args[0]), float(args[1])))
		return orig_set_ylim(self, *args, **kwargs)

	monkeypatch.setattr(matplotlib.axes.Axes, "set_xlim", _spy_set_xlim)
	monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", _spy_set_ylim)

	render_template_circles_plot(
		template=template,
		locations_xy=locs,
		config=TemplateCirclesPlotConfig(
			write_png=False,
			write_svg=False,
			force_square_aspect=False,
			force_center_soma=False,
			show_scale_bar=False,
			show_scale_circle=False,
		),
		png_path=tmp_path / "unused.png",
		svg_path=tmp_path / "unused.svg",
		plot_scope_points_xy=scope_locs,
		zoom_padding_percent=20.0,
		allow_scope_expansion=False,
	)

	assert seen_xlims
	assert seen_ylims
	assert np.isclose(float(seen_xlims[-1][0]), -2.0)
	assert np.isclose(float(seen_xlims[-1][1]), 12.0)
	assert np.isclose(float(seen_ylims[-1][0]), -2.0)
	assert np.isclose(float(seen_ylims[-1][1]), 12.0)


def test_convert_latency_samples_to_units_ms() -> None:
	lat = np.asarray([0.0, 10.0, -5.0], dtype=float)
	converted, label = _convert_latency_samples_to_units(
		lat,
		units="ms",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10000.0),
	)
	assert label == "ms"
	np.testing.assert_allclose(converted, np.asarray([0.0, 1.0, -0.5], dtype=float))


def test_convert_latency_samples_to_units_without_sampling_rate_falls_back() -> None:
	lat = np.asarray([0.0, 10.0], dtype=float)
	converted, label = _convert_latency_samples_to_units(
		lat,
		units="ms",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=None),
	)
	assert label == "samples"
	np.testing.assert_allclose(converted, lat)


def test_maybe_reversed_colormap_appends_reverse_suffix() -> None:
	assert _maybe_reversed_colormap("viridis", reverse=True) == "viridis_r"


def test_maybe_reversed_colormap_preserves_existing_reverse_suffix() -> None:
	assert _maybe_reversed_colormap("viridis_r", reverse=True) == "viridis_r"
	assert _maybe_reversed_colormap("viridis", reverse=False) == "viridis"


def test_ticks_ending_in_0_or_5_with_max_and_decimal_places() -> None:
	ticks = _ticks_ending_in_0_or_5_with_max(vmin=0.001, vmax=0.023, decimal_places=3)
	assert ticks.size >= 3
	assert np.isclose(float(ticks[-1]), 0.023)

	# Thousandths place should end in 0 or 5 for generated grid ticks.
	for t in ticks[:-1]:
		thousandths_digit = int(round(abs(float(t)) * 1000.0)) % 10
		assert thousandths_digit in (0, 5)


def test_ticks_ending_in_0_or_5_respects_target_count() -> None:
	ticks_sparse = _ticks_ending_in_0_or_5_with_max(vmin=0.001, vmax=0.023, decimal_places=3, target_count=4)
	ticks_dense = _ticks_ending_in_0_or_5_with_max(vmin=0.001, vmax=0.023, decimal_places=3, target_count=12)
	assert ticks_dense.size >= ticks_sparse.size
	assert np.isclose(float(ticks_sparse[-1]), 0.023)
	assert np.isclose(float(ticks_dense[-1]), 0.023)


def test_add_propagation_scale_bars_amp_fraction_controls_vertical_bar() -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_subplot(111)
	ax.set_xlim(0.0, 100.0)
	ax.set_ylim(0.0, 100.0)

	_add_propagation_scale_bars(
		ax=ax,
		n_samples=200,
		trace_offset_step=10.0,
		trace_gain=2.0,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0),
		text_color="black",
		anchor_x_frac=0.9,
		anchor_y_frac=0.1,
		time_fraction=0.15,
		amp_fraction=0.2,
		linewidth=1.0,
		fontsize=7.0,
		time_label_offset_frac=0.04,
		amp_label_offset_frac=0.02,
	)
	vertical_small = float(abs(ax.lines[1].get_ydata()[1] - ax.lines[1].get_ydata()[0]))

	_add_propagation_scale_bars(
		ax=ax,
		n_samples=200,
		trace_offset_step=10.0,
		trace_gain=2.0,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0),
		text_color="black",
		anchor_x_frac=0.9,
		anchor_y_frac=0.1,
		time_fraction=0.15,
		amp_fraction=1.0,
		linewidth=1.0,
		fontsize=7.0,
		time_label_offset_frac=0.04,
		amp_label_offset_frac=0.02,
	)
	vertical_large = float(abs(ax.lines[3].get_ydata()[1] - ax.lines[3].get_ydata()[0]))

	assert vertical_large > vertical_small
	plt.close(fig)


def test_add_propagation_scale_bars_labels_are_plain_ms_and_mv() -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_subplot(111)
	ax.set_xlim(0.0, 100.0)
	ax.set_ylim(0.0, 100.0)

	_add_propagation_scale_bars(
		ax=ax,
		n_samples=200,
		trace_offset_step=10.0,
		trace_gain=2.0,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0),
		text_color="black",
		anchor_x_frac=0.9,
		anchor_y_frac=0.1,
		time_fraction=0.15,
		amp_fraction=1.0,
		linewidth=1.0,
		fontsize=7.0,
		time_label_offset_frac=0.04,
		amp_label_offset_frac=0.02,
	)

	labels = [txt.get_text() for txt in ax.texts]
	assert len(labels) >= 2
	time_label = labels[0]
	amp_label = labels[1]
	assert "ms" in time_label
	assert "uv" in amp_label.lower()
	assert "e" not in time_label.lower()
	assert "e" not in amp_label.lower()
	plt.close(fig)


def test_add_propagation_scale_bars_force_amp_frac_to_max_amp_overrides_fraction() -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_subplot(111)
	ax.set_xlim(0.0, 100.0)
	ax.set_ylim(0.0, 100.0)

	_add_propagation_scale_bars(
		ax=ax,
		n_samples=200,
		trace_offset_step=10.0,
		trace_gain=1.0,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0),
		text_color="black",
		anchor_x_frac=0.9,
		anchor_y_frac=0.1,
		time_fraction=0.15,
		amp_fraction=0.05,
		linewidth=1.0,
		fontsize=7.0,
		time_label_offset_frac=0.04,
		amp_label_offset_frac=0.02,
		max_trace_amplitude_units=30.0,
		force_amp_frac_to_max_amp=False,
	)
	vertical_fractional = float(abs(ax.lines[1].get_ydata()[1] - ax.lines[1].get_ydata()[0]))

	_add_propagation_scale_bars(
		ax=ax,
		n_samples=200,
		trace_offset_step=10.0,
		trace_gain=1.0,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0),
		text_color="black",
		anchor_x_frac=0.9,
		anchor_y_frac=0.1,
		time_fraction=0.15,
		amp_fraction=0.05,
		linewidth=1.0,
		fontsize=7.0,
		time_label_offset_frac=0.04,
		amp_label_offset_frac=0.02,
		max_trace_amplitude_units=30.0,
		force_amp_frac_to_max_amp=True,
	)
	vertical_forced = float(abs(ax.lines[3].get_ydata()[1] - ax.lines[3].get_ydata()[0]))

	assert vertical_forced > vertical_fractional
	plt.close(fig)


def test_add_propagation_scale_bars_force_mode_label_not_inflated_by_trace_gain() -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_subplot(111)
	ax.set_xlim(0.0, 100.0)
	ax.set_ylim(0.0, 400.0)

	_add_propagation_scale_bars(
		ax=ax,
		n_samples=200,
		trace_offset_step=10.0,
		trace_gain=1.0,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0),
		text_color="black",
		anchor_x_frac=0.9,
		anchor_y_frac=0.1,
		time_fraction=0.15,
		amp_fraction=0.05,
		linewidth=1.0,
		fontsize=7.0,
		time_label_offset_frac=0.04,
		amp_label_offset_frac=0.02,
		max_trace_amplitude_units=30.0,
		force_amp_frac_to_max_amp=True,
	)
	amp_label_gain_1 = str(ax.texts[1].get_text())

	_add_propagation_scale_bars(
		ax=ax,
		n_samples=200,
		trace_offset_step=10.0,
		trace_gain=2.0,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0),
		text_color="black",
		anchor_x_frac=0.9,
		anchor_y_frac=0.1,
		time_fraction=0.15,
		amp_fraction=0.05,
		linewidth=1.0,
		fontsize=7.0,
		time_label_offset_frac=0.04,
		amp_label_offset_frac=0.02,
		max_trace_amplitude_units=30.0,
		force_amp_frac_to_max_amp=True,
	)
	amp_label_gain_2 = str(ax.texts[3].get_text())

	assert amp_label_gain_1 == amp_label_gain_2
	assert "uv" in amp_label_gain_1.lower()
	plt.close(fig)


def test_add_propagation_scale_bars_force_mode_uses_exact_max_units_not_rounded_up() -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_subplot(111)
	ax.set_xlim(0.0, 100.0)
	ax.set_ylim(0.0, 4000.0)

	_add_propagation_scale_bars(
		ax=ax,
		n_samples=200,
		trace_offset_step=10.0,
		trace_gain=1.0,
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=100000.0),
		text_color="black",
		anchor_x_frac=0.9,
		anchor_y_frac=0.1,
		time_fraction=0.15,
		amp_fraction=0.05,
		linewidth=1.0,
		fontsize=7.0,
		time_label_offset_frac=0.04,
		amp_label_offset_frac=0.02,
		max_trace_amplitude_units=600.0,
		force_amp_frac_to_max_amp=True,
	)

	amp_label = str(ax.texts[1].get_text())
	assert amp_label.startswith("600")
	assert "1000" not in amp_label
	plt.close(fig)
