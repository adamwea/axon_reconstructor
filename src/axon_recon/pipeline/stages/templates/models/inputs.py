from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UnitIdLabelConfig:
	show: bool = False
	fontsize: float = 12.0
	color: str = "white"
	x_offset_frac: float = 0.02
	y_offset_frac: float = 0.02
	horizontal_alignment: str = "right"
	vertical_alignment: str = "top"


@dataclass(frozen=True)
class CenterMostChannelCoordsConfig:
	show: bool = False
	fontsize: float = 10.0
	color: str = "white"
	x_offset_frac: float = 0.02
	y_offset_frac: float = 0.01
	horizontal_alignment: str = "left"
	vertical_alignment: str = "top"


@dataclass(frozen=True)
class TemplatePlotConfig:
	write_png: bool = True
	write_svg: bool = False
	dpi: float = 300.0
	relpath: str = "template"
	channel_scope: str = "contributing_channels"
	background: str = "black"
	signal_color: str = "white"
	force_center_soma: bool = False
	force_square_aspect: bool = True
	show_scale_bar: bool = True
	scale_bar_color: str = "white"
	scale_bar_text_offset_frac: float = 0.02
	scale_bar_x_offset_frac: float | None = None
	scale_bar_x_offset_considers_fontsize: bool = False
	scale_bar_horizontal_alignment: str = "right"
	scale_bar_vertical_alignment: str = "bottom"
	scale_bar_y_offset_frac: float = 0.06
	scale_bar_fontsize: float = 6.0
	scale_bar_linewidth: float = 1.8
	scale_bar_length_um: float | None = None
	show_axes: bool = True
	unit_id_label: UnitIdLabelConfig = field(default_factory=UnitIdLabelConfig)
	center_most_channel_coords: CenterMostChannelCoordsConfig = field(default_factory=CenterMostChannelCoordsConfig)


@dataclass(frozen=True)
class TemplateCirclesOverlapControlsConfig:
	scalebar_coords_overlap_detect: bool = False
	scalebar_colorbar_overlap_detect: bool = False
	unitid_label_channel_overlap_detect: bool = False
	coords_channel_overlap_detect: bool = False
	scalebar_channel_overlap_detect: bool = False
	scalecircle_channel_overlap_detect: bool = False
	max_overlap_check_iterations: int = 0


@dataclass(frozen=True)
class TemplateScaleCircleConfig:
	diameter: str | float = "equal_to_max_amplitude"
	linewidth: float = 1.8
	linestyle: str = "solid"
	fontsize: float = 6.0
	digits_after_decimal: int = 0
	horizontal_alignment: str = "left"
	vertical_alignment: str = "top"
	x_offset_frac: float = 0.02
	y_offset_frac: float = 0.02
	font_location: str = "inside"
	font_location_circle_too_small: str = "below"
	units: str = "uV"


@dataclass(frozen=True)
class TemplateCirclesBranchMorphologyConfig:
	enabled: bool = False
	node_border_linewidth: float = 0.35
	edge_linewidth: float = 0.8
	show_branch_labels: bool = False
	unique_color_per_branch: bool = True
	color_scheme: str = "tab20"
	node_outline_color: str | None = None
	node_outline_linewidth: float = 0.0
	branch_outline_color: str | None = None
	branch_outline_linewidth: float = 0.0


@dataclass(frozen=True)
class TemplateCirclesPlotConfig(TemplatePlotConfig):
	write_png: bool = False
	write_svg: bool = False
	relpath: str = "template_circles"
	size_by: str = "amplitude"
	color_by: str = "latency"
	show_propagation_order_labels: bool = False
	propagation_order_label_fontsize: float = 6.0
	propagation_order_label_color: str = "white"
	propagation_order_label_bbox_alpha: float = 0.35
	color_bar_units: str = ""
	color_bar_title: str = ""
	color_bar_show_axes_title: bool = True
	color_bar_show_unit_labels: bool = True
	color_bar_tick_fontsize: float = 6.0
	color_bar_tick_decimal_places: int = 3
	color_bar_tick_target_count: int | None = None
	color_bar_force_zero_and_neg_values_first_color_range: bool = False
	color_bar_zero_transition_contrast: float = 1.0
	show_scale_circle: bool = False
	scale_circle_color: str = "white"
	scale_circle: TemplateScaleCircleConfig = field(default_factory=TemplateScaleCircleConfig)
	branch_morphology: TemplateCirclesBranchMorphologyConfig = field(default_factory=TemplateCirclesBranchMorphologyConfig)
	overlap_controls: TemplateCirclesOverlapControlsConfig = field(default_factory=TemplateCirclesOverlapControlsConfig)


@dataclass(frozen=True)
class TemplatePlotsConfig:
	waveforms: TemplatePlotConfig = field(default_factory=TemplatePlotConfig)
	circles: TemplateCirclesPlotConfig = field(default_factory=TemplateCirclesPlotConfig)


@dataclass(frozen=True)
class TemplateWaveformOverlayConfig:
	debug_mode: bool = False
	write_pdf: bool = False
	pdf_relpath: str = "extremum_ch_wf_overlay.pdf"
	write_png: bool = True
	png_relpath: str = "extremum_ch_wf_overlay.png"
	top_channels_per_template: int = 10
	style: str = "overlay"
	show_title: bool = False
	show_axes: bool = False
	show_channel_labels: bool = False
	show_top_channel_info: bool = True
	show_waveform_count_info: bool = True
	include_mean: bool = True
	max_waveforms_to_show: int = 100
	waveform_sampling_mode: str = "uniform"
	random_seed: int | None = 0
	include_scale_bar: bool = True
	scale_bar_color: str = "black"
	scale_bar_fontsize: float = 6.0
	scale_bar_linewidth: float = 1.8
	scale_bar_time_fraction: float = 0.10
	scale_bar_amp_fraction: float = 0.10
	scale_bar_time_label_offset_frac: float = 0.03
	scale_bar_amp_label_offset_frac: float = 0.02
	background: str = "white"


@dataclass(frozen=True)
class TimeUpsampleConfig:
	enabled: bool = False
	factor: int = 1
	method: str = "sinc"
	mismatch_tolerance_hz: float = 0.5
	raw_rate_fallback_hz: float | None = None


@dataclass(frozen=True)
class WaveformExtractionConfig:
	ms_before: float | None = None
	ms_after: float | None = None
	max_spikes_per_unit: int | None = None


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
	write_svg: bool = True
	show_right_panel: bool = False
	write_circles_template_numbered_png: bool = True
	write_circles_template_numbered_svg: bool = True
	circles_template_numbered_relpath: str = "circles_template_numbered"
	write_propagation_2panel_png: bool = True
	write_propagation_2panel_svg: bool = True
	propagation_2panel_relpath: str = "propagation_2panel"
	show_title: bool = True
	title_template: str = "Propagation traces {start}-{end} / {total}"
	title_fontsize: float = 9.0
	top_channels: int = 25
	window_strategy: str = "max_ptp_sum"
	channels_per_panel: int = 25
	channel_overlap: int = 5
	force_start_with_max_ptp: bool = True
	force_start_with_max_negative_peak: bool = False
	force_min_neg_peak_index_zero: bool = False
	ordering_latency_mode: str = "abs_peak"
	latency_tie_breaker: str = "channel_index"
	debug_ordering: bool = False
	trace_label_mode: str = "electrode_id"
	relative_signed_order_numbers: bool = True
	right_panel_gap_fraction: float = 0.04
	right_panel_width_scale: float = 1.0
	right_panel_keep_temp_svg: bool = False
	right_panel_svg_relpath: str = "propagation_plot__right_temp.svg"
	right_panel_png_relpath: str = "propagation_plot__right_temp.png"
	left_panel_png_dpi: float | None = None
	right_panel_png_dpi: float = 300.0
	composed_png_dpi: float | None = None
	background: str = "white"
	show_electrode_ids: bool = False
	electrode_label_fontsize: float = 6.0
	electrode_label_x_offset_frac: float = 0.01
	electrode_label_y_offset_frac: float = 0.0
	electrode_label_alignment: str = "left"
	trace_gain: float = 1.0
	trace_spacing: float = 1.0
	peak_marker_height_frac: float = 0.24
	peak_marker_linewidth: float = 1.4
	show_multiple_peak_markers: bool = False
	delay_peak_marker_color: str = "black"
	show_scale_bar: bool = True
	scale_bar_anchor_x_frac: float = 0.92
	scale_bar_anchor_y_frac: float = 0.12
	scale_bar_time_fraction: float = 0.15
	scale_bar_amp_fraction: float = 0.20
	force_amp_frac_to_max_amp: bool = False
	debug_max_amps_at_each_channel: bool = False
	bold_max_amp_electrode_label: bool = False
	scale_bar_linewidth: float = 1.8
	scale_bar_fontsize: float = 7.0
	scale_bar_time_label_offset_frac: float = 0.04
	scale_bar_amp_label_offset_frac: float = 0.02
	abbreviate_post_ap_signal: bool = False
	post_ap_abbrev_start_ms: float = 1.0
	post_ap_abbrev_start_samples: int = 10
	post_ap_abbrev_cut_fraction: float = 0.5
	post_ap_abbrev_min_samples_to_cut: int = 5
	post_ap_abbrev_gap_samples: int = 8
	post_ap_abbrev_marker_text: str = "/.../"
	post_ap_abbrev_marker_fontsize: float = 7.0
	post_ap_abbrev_marker_y_offset_frac: float = 0.0
	show_duration_info: bool = False
	duration_info_x_frac: float = 0.01
	duration_info_y_frac: float = 0.99
	duration_info_fontsize: float = 6.0
	duration_info_horizontal_alignment: str = "left"
	duration_info_vertical_alignment: str = "top"
	plot_width_in: float = 13.0
	plot_panel_height_in: float = 2.8
	plot_extra_height_in: float = 1.0
	plot_hspace: float = 0.35
	plot_area_aspect_ratio: float | None = None
	latency_map: PropagationLatencyMapConfig = field(default_factory=PropagationLatencyMapConfig)

	@property
	def channel_label_fontsize(self) -> float:
		return self.electrode_label_fontsize

	@property
	def channel_label_x_offset_frac(self) -> float:
		return self.electrode_label_x_offset_frac

	@property
	def channel_label_y_offset_frac(self) -> float:
		return self.electrode_label_y_offset_frac

	@property
	def channel_label_alignment(self) -> str:
		return self.electrode_label_alignment

	@property
	def bold_max_amp_channel_label(self) -> bool:
		return self.bold_max_amp_electrode_label


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
class MultipleNegativePeaksCheckConfig:
	enable: bool = False
	prominence_fraction: float = 0.30
	min_separation_samples: int = 8
	max_peaks_per_channel: int = 2


@dataclass(frozen=True)
class QualityCheckJsonOutputConfig:
	write_json: bool = True
	json_relpath: str = "quality_checks_multiple_negative_peaks.json"


@dataclass(frozen=True)
class QualityCheckPlotOutputConfig:
	write_png: bool = True
	write_svg: bool = False
	relpath: str = "multiple_peaks_at_channel_templates"
	show_multiple_peak_markers: bool = False
	delay_peak_marker_color: str = "black"


@dataclass(frozen=True)
class MultipleNegativePeaksOutputsConfig:
	write_json: bool = True
	json_relpath: str = "quality_checks_multiple_negative_peaks.json"
	plot: QualityCheckPlotOutputConfig = field(default_factory=QualityCheckPlotOutputConfig)


@dataclass(frozen=True)
class PerUnitQualityChecksOutputsConfig:
	check_for_multiple_peaks_at_channel_templates: MultipleNegativePeaksOutputsConfig = field(
		default_factory=MultipleNegativePeaksOutputsConfig
	)


@dataclass(frozen=True)
class DataQualityChecksOutputsConfig:
	check_for_multiple_peaks_at_channel_templates: QualityCheckJsonOutputConfig = field(
		default_factory=QualityCheckJsonOutputConfig
	)


@dataclass(frozen=True)
class QualityChecksConfig:
	enable: bool = False
	suppress_warnings: bool = False
	check_for_multiple_peaks_at_channel_templates: MultipleNegativePeaksCheckConfig = field(
		default_factory=MultipleNegativePeaksCheckConfig
	)


@dataclass(frozen=True)
class WfOverlayGridReportConfig:
	write_pdf: bool = False
	pdf_relpath: str = "wf_overlay_grid.pdf"
	write_png: bool = True
	png_relpath: str = "wf_overlay_grid.png"
	write_svg: bool = False
	svg_relpath: str = "wf_overlay_grid.svg"
	keep_temp_svg: bool = False
	temp_svg_relpath: str = "wf_overlay_grid__temp.svg"
	top_channels_per_template: int = 10
	subplot_background_color: str = "white"
	figure_background_color: str = "white"
	render_mode: str = "direct_replot"
	dpi: float = 300.0


@dataclass(frozen=True)
class FootprintMapGridReportConfig:
	write_pdf: bool = False
	pdf_relpath: str = "footprint_map_grid.pdf"
	write_png: bool = True
	png_relpath: str = "footprint_map_grid.png"
	write_svg: bool = False
	svg_relpath: str = "footprint_map_grid.svg"
	keep_temp_svg: bool = False
	temp_svg_relpath: str = "footprint_map_grid__temp.svg"
	show_title: bool = True
	template_shape: str = "square"
	global_color_scale: bool = True
	subplot_background_color: str = "white"
	figure_background_color: str = "white"
	render_mode: str = "direct_replot"
	dpi: float = 300.0


@dataclass(frozen=True)
class FootprintGridsReportConfig:
	circles_map_grid: FootprintMapGridReportConfig = field(
		default_factory=lambda: FootprintMapGridReportConfig(
			pdf_relpath="circles_map_grid.pdf",
			png_relpath="circles_map_grid.png",
		)
	)
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
class UnitLocationsReportConfig:
	write_json: bool = True
	json_relpath: str = "unit_locations.json"
	write_png: bool = False
	png_relpath: str = "unit_locations.png"
	write_svg: bool = False
	svg_relpath: str = "unit_locations.svg"
	background: str = "black"
	chip_scatter_color: str = "white"
	chip_scatter_size: float = 14.0
	chip_scatter_alpha: float = 0.8
	invert_y_axis: bool = True
	use_probe_active_area: bool = True
	underlay_concat_channels: bool = True
	concat_channel_scatter_color: str = "#808080"
	concat_channel_scatter_size: float = 2.5
	concat_channel_scatter_alpha: float = 0.35
	underlay_template_channels: bool = False
	template_channel_scatter_size: float = 2.0
	template_channel_scatter_alpha: float = 0.30
	template_channel_colormap: str = "tab20"
	show_original_to_current_redlines: bool = False
	redline_color: str = "red"
	redline_alpha: float = 0.9
	redline_linewidth: float = 0.7
	show_unit_id_labels: bool = True
	unit_id_label_fontsize: float = 6.0
	unit_id_label_color: str = "white"
	unit_id_label_x_offset_frac: float = 0.02
	unit_id_label_y_offset_frac: float = 0.02
	unit_id_label_horizontal_alignment: str = "right"
	unit_id_label_vertical_alignment: str = "top"


@dataclass(frozen=True)
class MultiSourcePdfReportConfig:
	enabled: bool = False
	pdf_relpath: str = "reports/template_multi_source.pdf"


@dataclass(frozen=True)
class AnalyzerCacheConfig:
	enabled: bool = True
	relpath: str = "analyzers"
	concat_analyzer_subdir: str = "concat"
	segment_analyzers_subdir: str = ""
	cleanup_on_success: bool = False
	reuse_on_force_restart: bool = False

	@property
	def relpath_root(self) -> str:
		# Compatibility alias for clearer schema naming.
		return self.relpath


@dataclass(frozen=True)
class ReportsConfig:
	plot_multi_source_pdf: MultiSourcePdfReportConfig = field(default_factory=MultiSourcePdfReportConfig)
	replot_from_disk: bool = False
	overwrite_on_unit_rerun: bool = False
	grid_sort_by: str = "unit_id"
	locations: UnitLocationsReportConfig = field(default_factory=UnitLocationsReportConfig)
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
	quality_checks: PerUnitQualityChecksOutputsConfig = field(default_factory=PerUnitQualityChecksOutputsConfig)
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

	@property
	def extremum_ch_wf_overlay(self) -> TemplateWaveformOverlayConfig:
		# Alias for clearer external naming while preserving internal field compatibility.
		return self.template_wf_overlay


@dataclass(frozen=True)
class ProbeGeometryConfig:
	pitch_um: float | None = None
	electrode_size_um_x: float | None = None
	electrode_size_um_y: float | None = None
	active_area_um_x: float | None = None
	active_area_um_y: float | None = None
	sampling_rate_hz: float | None = None


@dataclass(frozen=True)
class ResolveSourcesPhaseConfig:
	enabled: bool = True
	show_header: bool = True
	log_candidates: bool = True
	check_path_exists: bool = True
	include_alternate_well_dirs: bool = True
	probe_curated_units: bool = True
	max_candidates_per_source: int = 12
	fail_if_required_sources_missing: bool = False
	write_json: bool = False
	json_relpath: str = "context/resolve_sources_summary.json"


@dataclass(frozen=True)
class TemplatesInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path
	final_output_root: Path | None = None
	artifact_lookup_roots: tuple[Path, ...] = field(default_factory=tuple)
	concat_analyzer_relpath: str | None = None
	concat_sorting_relpath: str | None = None
	preprocessed_concat_reldir: str | None = None
	preprocessed_segments_reldir: str | None = None
	preproc_seg_sources_reldir: str | None = None

	output_rel_root: str = "templates_outputs"
	analyzer_cache: AnalyzerCacheConfig = field(default_factory=AnalyzerCacheConfig)
	per_unit_outputs: PerUnitTemplatesOutputsConfig = field(default_factory=PerUnitTemplatesOutputsConfig)
	reports: ReportsConfig = field(default_factory=ReportsConfig)

	unit_ids: list[Any] | None = None
	unit_limit: int | None = None

	force_restart: bool = False
	force_replot: bool = False
	force_replot_per_unit: bool = False
	force_rereport: bool = False
	require_curated_units: bool = True
	include_concat: bool = True
	include_segments: bool = True
	require_concat_analyzer: bool = False
	require_segment_analyzers: bool = False
	waveform_extraction: WaveformExtractionConfig = field(default_factory=WaveformExtractionConfig)
	execution_upsampling: TimeUpsampleConfig = field(default_factory=TimeUpsampleConfig)
	merge: MergeConfig = field(default_factory=MergeConfig)
	quality_checks: QualityChecksConfig = field(default_factory=QualityChecksConfig)
	quality_checks_outputs: DataQualityChecksOutputsConfig = field(default_factory=DataQualityChecksOutputsConfig)
	resolve_sources_phase: ResolveSourcesPhaseConfig = field(default_factory=ResolveSourcesPhaseConfig)
	probe_geometry: ProbeGeometryConfig | None = None
	n_jobs: int = 1
