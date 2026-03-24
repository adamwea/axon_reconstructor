from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TemplatePlotConfig:
	write_png: bool = True
	write_svg: bool = False
	relpath: str = "template"
	channel_scope: str = "contributing_channels"
	background: str = "black"
	signal_color: str = "white"
	force_center_soma: bool = False
	force_square_aspect: bool = True
	show_scale_bar: bool = True
	scale_bar_color: str = "white"
	scale_bar_text_offset_frac: float = 0.02
	scale_bar_y_offset_frac: float = 0.06
	scale_bar_fontsize: float = 6.0
	scale_bar_linewidth: float = 1.8
	scale_bar_length_um: float | None = None


@dataclass(frozen=True)
class TemplateCirclesPlotConfig(TemplatePlotConfig):
	write_png: bool = False
	write_svg: bool = False
	relpath: str = "template_circles"
	circle_size_scale_factor: float = 1.0
	size_by: str = "amplitude"
	color_by: str = "latency"
	color_bar_units: str = ""
	color_bar_title: str = ""
	color_bar_show_axes_title: bool = True
	color_bar_show_unit_labels: bool = True
	color_bar_tick_decimal_places: int = 3
	color_bar_tick_target_count: int | None = None


@dataclass(frozen=True)
class TemplatePlotsConfig:
	waveforms: TemplatePlotConfig = field(default_factory=TemplatePlotConfig)
	circles: TemplateCirclesPlotConfig = field(default_factory=TemplateCirclesPlotConfig)


@dataclass(frozen=True)
class TemplateWaveformOverlayConfig:
	write_pdf: bool = False
	pdf_relpath: str = "template_wf_overlay.pdf"
	write_png: bool = True
	png_relpath: str = "template_wf_overlay.png"
	top_channels_per_template: int = 10
	style: str = "overlay"
	include_mean: bool = True
	include_scale_bar: bool = True
	scale_bar_color: str = "black"
	scale_bar_fontsize: float = 6.0
	scale_bar_linewidth: float = 1.8
	background: str = "white"


@dataclass(frozen=True)
class TimeUpsampleConfig:
	enabled: bool = False
	factor: int = 1
	method: str = "sinc"
	mismatch_tolerance_hz: float = 0.5
	raw_rate_fallback_hz: float | None = None


@dataclass(frozen=True)
class FootprintMapConfig:
	write_png: bool = True
	write_svg: bool = False
	relpath: str = "footprint_map"
	background: str = "black"
	color_map: str = "viridis"
	template_shape: str = "square"
	template_padding_value: str = "zero"
	show_color_bar: bool = True
	color_bar_location: str = "topright"
	color_bar_fontsize: float = 6.0
	color_bar_length_fraction: float = 0.3
	color_bar_pad_fraction: float = 0.02
	force_low_value: float | None = 0.0
	force_high_value: float | None = None
	scale: str = "linear"
	percentile_low: float = 5.0
	percentile_high_linear: float = 99.0
	percentile_high_log: float = 99.5
	knot_anchor_values: tuple[float, float] = (1.0, 10.0)
	knot_y1_min: float = 0.02
	knot_y1_max: float = 0.90
	knot_y2_min: float = 0.07
	knot_y2_max: float = 0.98
	knot_min_gap: float = 0.05
	linear_cap_rounding_mode: str = "ceil_step"
	linear_cap_rounding_step: float = 10.0
	linear_cap_min_vmax: float = 11.0
	show_ticks: tuple[Any, ...] = (1, 10, "dynamic_high")


@dataclass(frozen=True)
class FootprintPlotsConfig:
	amplitude_map: FootprintMapConfig = field(
		default_factory=lambda: FootprintMapConfig(relpath="footprint_amplitude_map")
	)
	latency_map: FootprintMapConfig = field(
		default_factory=lambda: FootprintMapConfig(relpath="footprint_latency_map")
	)


@dataclass(frozen=True)
class TopographicalFootprintConfig:
	write_png: bool = True
	write_svg: bool = False
	relpath: str = "topographical_footprint"
	background: str = "black"
	color_map: str = "viridis"
	template_shape: str = "square"
	template_padding_value: str = "zero"
	show_color_bar: bool = True
	color_bar_location: str = "topright"
	color_bar_fontsize: float = 6.0
	color_bar_length_fraction: float = 0.3
	color_bar_pad_fraction: float = 0.02
	force_low_value: float | None = 0.0
	force_high_value: float | None = None
	scale: str = "linear"
	percentile_low: float = 5.0
	percentile_high_linear: float = 99.0
	percentile_high_log: float = 99.5
	knot_anchor_values: tuple[float, float] = (1.0, 10.0)
	knot_y1_min: float = 0.02
	knot_y1_max: float = 0.90
	knot_y2_min: float = 0.07
	knot_y2_max: float = 0.98
	knot_min_gap: float = 0.05
	linear_cap_rounding_mode: str = "ceil_step"
	linear_cap_rounding_step: float = 10.0
	linear_cap_min_vmax: float = 11.0
	show_ticks: tuple[Any, ...] = (1, 10, "dynamic_high")
	elevation_deg: float = 35.0
	azimuth_deg: float = -60.0
	marker_size: float = 14.0


@dataclass(frozen=True)
class TopographicalFootprintsConfig:
	amplitude: TopographicalFootprintConfig = field(
		default_factory=lambda: TopographicalFootprintConfig(relpath="topographical_amplitude_footprint")
	)
	latency: TopographicalFootprintConfig = field(
		default_factory=lambda: TopographicalFootprintConfig(relpath="topographical_latency_footprint")
	)


@dataclass(frozen=True)
class PropagationAxesConfig:
	show: bool = True
	xlabel: str = "x (um)"
	ylabel: str = "y (um)"
	label_fontsize: float = 6.0
	tick_fontsize: float = 5.0


@dataclass(frozen=True)
class PropagationLatencyMapConfig:
	show: bool = True
	color_map: str = "viridis"
	force_square_aspect: bool = True
	title: str = "Latency Map"
	fontsize: float = 6.0
	template_shape: str = "top_channels_only"
	show_color_bar: bool = True
	color_bar_location: str = "topright"
	color_bar_fontsize: float = 6.0
	color_bar_length_fraction: float = 0.3
	color_bar_pad_fraction: float = 0.02
	force_low_value: float | None = 0.0
	force_high_value: float | None = None
	scale: str = "linear"
	percentile_low: float = 5.0
	percentile_high_linear: float = 99.0
	percentile_high_log: float = 99.5
	knot_anchor_values: tuple[float, float] = (1.0, 10.0)
	knot_y1_min: float = 0.02
	knot_y1_max: float = 0.90
	knot_y2_min: float = 0.07
	knot_y2_max: float = 0.98
	knot_min_gap: float = 0.05
	linear_cap_rounding_mode: str = "ceil_step"
	linear_cap_rounding_step: float = 10.0
	linear_cap_min_vmax: float = 11.0
	show_ticks: tuple[Any, ...] = (1, 10, "dynamic_high")
	axes: PropagationAxesConfig = field(default_factory=PropagationAxesConfig)


@dataclass(frozen=True)
class PropagationPlotConfig:
	write_pdf: bool = False
	pdf_relpath: str = "propagation_plot.pdf"
	write_png: bool = True
	png_relpath: str = "propagation_plot.png"
	show_title: bool = True
	title_template: str = "Propagation traces {start}-{end} / {total}"
	title_fontsize: float = 9.0
	top_channels: int = 25
	channels_per_panel: int = 25
	channel_overlap: int = 5
	background: str = "white"
	show_electrode_ids: bool = False
	channel_label_fontsize: float = 6.0
	channel_label_x_offset_frac: float = 0.01
	channel_label_y_offset_frac: float = 0.0
	channel_label_alignment: str = "left"
	trace_gain: float = 1.0
	trace_spacing: float = 1.0
	peak_marker_height_frac: float = 0.24
	peak_marker_linewidth: float = 1.4
	show_scale_bar: bool = True
	scale_bar_anchor_x_frac: float = 0.92
	scale_bar_anchor_y_frac: float = 0.12
	scale_bar_time_fraction: float = 0.15
	scale_bar_amp_fraction: float = 0.20
	force_amp_frac_to_max_amp: bool = False
	debug_max_amps_at_each_channel: bool = False
	bold_max_amp_channel_label: bool = False
	scale_bar_linewidth: float = 1.8
	scale_bar_fontsize: float = 7.0
	scale_bar_time_label_offset_frac: float = 0.04
	scale_bar_amp_label_offset_frac: float = 0.02
	latency_map: PropagationLatencyMapConfig = field(default_factory=PropagationLatencyMapConfig)


@dataclass(frozen=True)
class TemplateArtifactConfig:
	write_npy: bool = False
	npy_relpath: str = "template.npy"
	channel_locations_npy_relpath: str | None = None
	padding_value: str = "zero"


@dataclass(frozen=True)
class MergeConfig:
	enable: bool = True
	method: str = "mean_all_waveforms"
	centering_method: str = "pre_peak_robust_baseline"
	max_waveforms_per_source_channel: int | None = 500
	overlap_match_priority: tuple[str, ...] = ("electrode_id", "channel_id", "location")
	location_tolerance_um: float = 1.0


@dataclass(frozen=True)
class WfOverlayGridReportConfig:
	write_pdf: bool = False
	pdf_relpath: str = "wf_overlay_grid.pdf"
	write_png: bool = True
	png_relpath: str = "wf_overlay_grid.png"
	top_channels_per_template: int = 10


@dataclass(frozen=True)
class FootprintMapGridReportConfig:
	write_pdf: bool = False
	pdf_relpath: str = "footprint_map_grid.pdf"
	write_png: bool = True
	png_relpath: str = "footprint_map_grid.png"
	template_shape: str = "square"
	global_color_scale: bool = True


@dataclass(frozen=True)
class FootprintGridsReportConfig:
	amplitude_map_grid: FootprintMapGridReportConfig = field(
		default_factory=lambda: FootprintMapGridReportConfig(
			pdf_relpath="amplitude_map_grid.pdf",
			png_relpath="amplitude_map_grid.png",
		)
	)
	latency_map_grid: FootprintMapGridReportConfig = field(
		default_factory=lambda: FootprintMapGridReportConfig(
			pdf_relpath="latency_map_grid.pdf",
			png_relpath="latency_map_grid.png",
		)
	)


@dataclass(frozen=True)
class MultiSourcePdfReportConfig:
	enabled: bool = False
	pdf_relpath: str = "reports/template_multi_source.pdf"


@dataclass(frozen=True)
class ReportsConfig:
	plot_multi_source_pdf: MultiSourcePdfReportConfig = field(default_factory=MultiSourcePdfReportConfig)
	replot_from_disk: bool = False
	time_upsample: TimeUpsampleConfig = field(default_factory=TimeUpsampleConfig)
	wf_overlay_grid: WfOverlayGridReportConfig = field(default_factory=WfOverlayGridReportConfig)
	footprint_grids: FootprintGridsReportConfig = field(default_factory=FootprintGridsReportConfig)

	@property
	def foot_print_grids(self) -> FootprintGridsReportConfig:
		# Compatibility alias for older schema spelling.
		return self.footprint_grids


@dataclass(frozen=True)
class PerUnitTemplatesOutputsConfig:
	unit_reldir: str = "units/{unit_id:04d}/"
	merged_template: TemplateArtifactConfig = field(
		default_factory=lambda: TemplateArtifactConfig(
			write_npy=True,
			npy_relpath="merged_template.npy",
			channel_locations_npy_relpath="merged_channel_locations.npy",
		)
	)
	square_template: TemplateArtifactConfig = field(
		default_factory=lambda: TemplateArtifactConfig(
			write_npy=False,
			npy_relpath="square_template.npy",
			channel_locations_npy_relpath="square_channel_locations.npy",
			padding_value="zero",
		)
	)
	scan_template: TemplateArtifactConfig = field(
		default_factory=lambda: TemplateArtifactConfig(
			write_npy=False,
			npy_relpath="scan_template.npy",
			channel_locations_npy_relpath="scan_channel_locations.npy",
			padding_value="zero",
		)
	)
	full_template: TemplateArtifactConfig = field(
		default_factory=lambda: TemplateArtifactConfig(
			write_npy=False,
			npy_relpath="full_template.npy",
			channel_locations_npy_relpath="full_channel_locations_xy.npy",
			padding_value="zero",
		)
	)
	template: TemplatePlotConfig = field(default_factory=TemplatePlotConfig)
	template_circles: TemplateCirclesPlotConfig = field(default_factory=TemplateCirclesPlotConfig)
	template_wf_overlay: TemplateWaveformOverlayConfig = field(default_factory=TemplateWaveformOverlayConfig)
	footprint_plots: FootprintPlotsConfig = field(default_factory=FootprintPlotsConfig)
	topographical_footprints: TopographicalFootprintsConfig = field(default_factory=TopographicalFootprintsConfig)
	propagation_plots: PropagationPlotConfig = field(default_factory=PropagationPlotConfig)

	@property
	def template_plots(self) -> TemplatePlotsConfig:
		# Compatibility convenience for new nested runtime schema.
		return TemplatePlotsConfig(waveforms=self.template, circles=self.template_circles)


@dataclass(frozen=True)
class ProbeGeometryConfig:
	pitch_um: float | None = None
	electrode_size_um_x: float | None = None
	electrode_size_um_y: float | None = None
	active_area_um_x: float | None = None
	active_area_um_y: float | None = None
	sampling_rate_hz: float | None = None


@dataclass(frozen=True)
class TemplatesInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path

	output_rel_root: str = "templates_outputs"
	per_unit_outputs: PerUnitTemplatesOutputsConfig = field(default_factory=PerUnitTemplatesOutputsConfig)
	reports: ReportsConfig = field(default_factory=ReportsConfig)

	unit_ids: list[Any] | None = None
	unit_limit: int | None = None

	force_restart: bool = False
	force_replot: bool = False
	force_replot_per_unit: bool = False
	require_curated_units: bool = True
	include_concat: bool = True
	include_segments: bool = True
	execution_upsampling: TimeUpsampleConfig = field(default_factory=TimeUpsampleConfig)
	merge: MergeConfig = field(default_factory=MergeConfig)
	probe_geometry: ProbeGeometryConfig | None = None
	n_jobs: int = 1
