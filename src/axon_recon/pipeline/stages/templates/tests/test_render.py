from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.stages.templates.core.render import render_propagation_plot
from axon_recon.pipeline.stages.templates.core.render import render_template_wf_overlay
from axon_recon.pipeline.stages.templates.core.render import render_topographical_amplitude_footprint
from axon_recon.pipeline.stages.templates.core.render import _expand_limits_for_glyph_half_size
from axon_recon.pipeline.stages.templates.core.render import _probe_electrode_dims_um
from axon_recon.pipeline.stages.templates.core.render import _limits_for_template_shape
from axon_recon.pipeline.stages.templates.core.render import _maybe_reversed_colormap
from axon_recon.pipeline.stages.templates.core.render import _add_propagation_scale_bars
from axon_recon.pipeline.stages.templates.core.render import _convert_latency_samples_to_units
from axon_recon.pipeline.stages.templates.core.render import _ticks_ending_in_0_or_5_with_max
from axon_recon.pipeline.stages.templates.models.inputs import (
	ProbeGeometryConfig,
	PropagationLatencyMapConfig,
	PropagationPlotConfig,
	TemplateWaveformOverlayConfig,
	TimeUpsampleConfig,
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
