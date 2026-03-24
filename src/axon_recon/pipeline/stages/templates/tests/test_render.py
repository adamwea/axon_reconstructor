from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.stages.templates.core.render import render_propagation_plot
from axon_recon.pipeline.stages.templates.core.render import render_topographical_amplitude_footprint
from axon_recon.pipeline.stages.templates.core.render import _expand_limits_for_glyph_half_size
from axon_recon.pipeline.stages.templates.core.render import _probe_electrode_dims_um
from axon_recon.pipeline.stages.templates.core.render import _limits_for_template_shape
from axon_recon.pipeline.stages.templates.core.render import _maybe_reversed_colormap
from axon_recon.pipeline.stages.templates.core.render import _convert_latency_samples_to_units
from axon_recon.pipeline.stages.templates.core.render import _ticks_ending_in_0_or_5_with_max
from axon_recon.pipeline.stages.templates.models.inputs import (
	ProbeGeometryConfig,
	PropagationLatencyMapConfig,
	PropagationPlotConfig,
	TopographicalFootprintConfig,
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
			channel_label_fontsize=7.5,
			channel_label_x_offset_frac=0.02,
			channel_label_y_offset_frac=0.1,
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


def test_render_propagation_plot_accepts_negative_channel_label_x_offset(tmp_path: Path, monkeypatch) -> None:
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
		if isinstance(s, str) and s.startswith("ch "):
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
			channel_label_x_offset_frac=-0.15,
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert png_path.exists()
	assert outputs.get("propagation_plot_png") == str(png_path)
	assert len(seen_text_x) > 0
	assert min(seen_text_x) < 0.0


def test_render_propagation_plot_honors_channel_label_alignment(tmp_path: Path, monkeypatch) -> None:
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
		if isinstance(s, str) and s.startswith("ch "):
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
			channel_label_alignment="right",
		),
		pdf_path=tmp_path / "unused.pdf",
		png_path=png_path,
		probe_geometry=ProbeGeometryConfig(pitch_um=17.5),
	)

	assert png_path.exists()
	assert len(seen_alignments) > 0
	assert all(a == "right" for a in seen_alignments)


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
