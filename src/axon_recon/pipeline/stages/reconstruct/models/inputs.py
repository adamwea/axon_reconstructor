from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.templates.models.inputs import FootprintMapGridReportConfig, TemplatesInputs


@dataclass(frozen=True)
class CircleReconDisplayConfig:
	base: str = "template_circles"
	channel_scope: str = "nodes_and_branches"
	zoom_padding_percent: float = 20.0
	invert_y_axis: bool = True
	force_center_soma: bool = True
	branch_scope: str = "raw"
	unique_color_per_branch: bool = True
	show_branch_labels: bool = False
	show_branch_legend: bool = False
	color_scheme: str = "tab20"
	node_outline_color: str | None = None
	node_outline_linewidth: float = 0.0
	branch_outline_color: str | None = None
	branch_outline_linewidth: float = 0.0
	node_border_linewidth: float = 0.35
	edge_linewidth: float = 0.8


@dataclass(frozen=True)
class CircleReconOutputConfig:
	write_png: bool = False
	write_svg: bool = False
	relpath: str = "circle_recon"
	dpi: float = 300.0


@dataclass(frozen=True)
class CircleReconConfig:
	display: CircleReconDisplayConfig = field(default_factory=CircleReconDisplayConfig)
	output: CircleReconOutputConfig = field(default_factory=CircleReconOutputConfig)
	base_template_circles: Any | None = None
	base_footprint_amplitude: Any | None = None
	base_footprint_latency: Any | None = None


@dataclass(frozen=True)
class ReconstructionDiagnosticFigureConfig:
	write_png: bool = False
	write_svg: bool = False
	relpath: str = "diagnostic_figure"
	dpi: float = 300.0
	invert_y_axis: bool = True


def _default_channel_selection_figure_config() -> ReconstructionDiagnosticFigureConfig:
	return ReconstructionDiagnosticFigureConfig(relpath="diagnostic_figs/channel_selection")


def _default_axon_reconstruction_figure_config() -> ReconstructionDiagnosticFigureConfig:
	return ReconstructionDiagnosticFigureConfig(relpath="diagnostic_figs/axon_reconstruction")


@dataclass(frozen=True)
class PerUnitOutputsConfig:
	unit_reldir: str = "units/{unit_id:04d}/"

	write_branches_raw_json: bool = True
	branches_raw_relpath: str = "branches_raw.json"

	write_branches_json: bool = True
	branches_relpath: str = "branches.json"

	write_detection_filter_json: bool = False
	detection_filter_relpath: str = "detection_filter.json"

	write_kurtosis_filter_json: bool = False
	kurtosis_filter_relpath: str = "kurtosis_filter.json"

	write_peak_std_filter_json: bool = False
	peak_std_filter_relpath: str = "peak_std_filter.json"

	write_delay_filter_json: bool = False
	delay_filter_relpath: str = "delay_filter.json"

	write_all_filters_json: bool = False
	all_filters_relpath: str = "all_filters.json"

	write_heuristics_json: bool = True
	heuristics_relpath: str = "heuristics.json"

	write_gtr_pkl: bool = True
	gtr_pkl_relpath: str = "gtr.pkl"
	template_source: str = "square"

	write_gtr_json: bool = False
	gtr_json_relpath: str = "gtr.json"
	channel_selection_figure: ReconstructionDiagnosticFigureConfig = field(
		default_factory=_default_channel_selection_figure_config
	)
	axon_reconstruction_figure: ReconstructionDiagnosticFigureConfig = field(
		default_factory=_default_axon_reconstruction_figure_config
	)

	write_amplitude_map_png: bool = False
	amplitude_map_png_relpath: str = "amplitude_map.png"
	amplitude_map_heatmap: SharedHeatmapConfig = field(default_factory=SharedHeatmapConfig)
	circle_recon: CircleReconConfig = field(default_factory=CircleReconConfig)


@dataclass(frozen=True)
class ReconstructionGridReportsConfig:
	sort_by: str = "unit_id"
	circle_recon_grid: FootprintMapGridReportConfig = field(
		default_factory=lambda: FootprintMapGridReportConfig(
			write_png=False,
			pdf_relpath="reports/circle_recon_grid.pdf",
			png_relpath="reports/circle_recon_grid.png",
			svg_relpath="reports/circle_recon_grid.svg",
			temp_svg_relpath="reports/circle_recon_grid__temp.svg",
		)
	)


@dataclass(frozen=True)
class ReconstructionReportsConfig:
	grids: ReconstructionGridReportsConfig = field(default_factory=ReconstructionGridReportsConfig)
	overwrite_on_unit_rerun: bool = False


@dataclass(frozen=True)
class ReconstructionBranchColorsConfig:
	unique_color_per_branch: bool = True
	color_scheme: str = "tab20"


@dataclass(frozen=True)
class ReconstructionAxonVelocityPhaseConfig:
	enabled: bool = True
	params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionGenerateGtrsOutputsConfig:
	write_branches_raw_json: bool = True
	branches_raw_relpath: str = "branches_raw.json"

	write_branches_json: bool = True
	branches_relpath: str = "branches.json"

	write_detection_filter_json: bool = False
	detection_filter_relpath: str = "detection_filter.json"

	write_kurtosis_filter_json: bool = False
	kurtosis_filter_relpath: str = "kurtosis_filter.json"

	write_peak_std_filter_json: bool = False
	peak_std_filter_relpath: str = "peak_std_filter.json"

	write_delay_filter_json: bool = False
	delay_filter_relpath: str = "delay_filter.json"

	write_all_filters_json: bool = False
	all_filters_relpath: str = "all_filters.json"

	write_heuristics_json: bool = True
	heuristics_relpath: str = "heuristics.json"

	write_gtr_pkl: bool = True
	gtr_pkl_relpath: str = "gtr.pkl"
	template_source: str = "square"

	write_gtr_json: bool = False
	gtr_json_relpath: str = "gtr.json"
	channel_selection_figure: ReconstructionDiagnosticFigureConfig = field(
		default_factory=_default_channel_selection_figure_config
	)
	axon_reconstruction_figure: ReconstructionDiagnosticFigureConfig = field(
		default_factory=_default_axon_reconstruction_figure_config
	)


@dataclass(frozen=True)
class ReconstructionGenerateGtrsPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/generate_gtrs_summary.json"
	unit_procs: int | None = None
	unit_batch_size: int | None = None
	outputs: ReconstructionGenerateGtrsOutputsConfig = field(default_factory=ReconstructionGenerateGtrsOutputsConfig)
	axon_velocity: ReconstructionAxonVelocityPhaseConfig = field(default_factory=ReconstructionAxonVelocityPhaseConfig)


@dataclass(frozen=True)
class ReconstructionBranchPlotOutputConfig:
	write_png: bool = False
	write_svg: bool = False
	relpath: str = "branch_plots"
	manifest_relpath: str = "branch_plots_manifest.json"
	dpi: float = 300.0


@dataclass(frozen=True)
class ReconstructionBranchPropagationDisplayConfig:
	figsize: tuple[float, float] = (2.75, 6.0)
	total_width: float | None = None
	sort_templates: bool = False
	show_title: bool = True
	invert_y_axis: bool = True


@dataclass(frozen=True)
class ReconstructionBranchVelocityDisplayConfig:
	figsize: tuple[float, float] = (6.0, 4.0)
	show_title: bool = True
	title_fontsize: float = 12.0
	axis_label_fontsize: float = 10.0
	tick_label_fontsize: float = 10.0
	units_only_axis_labels: bool = True
	show_legend: bool = True
	legend_fontsize: float = 8.0


@dataclass(frozen=True)
class ReconstructionFullChipLayoutColorConfig:
	strategy: str = "distinct_hsv"
	color_scheme: str = "nipy_spectral"


@dataclass(frozen=True)
class ReconstructionFullChipLayoutDisplayConfig:
	figsize: tuple[float, float] = (11.0, 6.0)
	show_title: bool = True
	title: str = "Full-chip reconstructed branch layout"
	invert_y_axis: bool = True
	alpha: float = 0.8
	linewidth: float = 1.25
	show_legend: bool = False
	legend_fontsize: float = 6.0
	legend_ncols: int = 1
	draw_chip_outline: bool = True
	chip_outline_color: str = "#666666"
	chip_outline_linewidth: float = 1.0
	background_color: str = "white"


@dataclass(frozen=True)
class ReconstructionFullChipLayoutOutputConfig:
	write_png: bool = True
	write_svg: bool = False
	relpath: str = "reports/full_chip_layout"
	manifest_relpath: str = "reports/full_chip_layout_manifest.json"
	dpi: float = 300.0


def _default_branch_propagations_output_config() -> ReconstructionBranchPlotOutputConfig:
	return ReconstructionBranchPlotOutputConfig(
		relpath="branch_plots/propagations",
		manifest_relpath="branch_propagations_manifest.json",
	)


def _default_branch_velocities_output_config() -> ReconstructionBranchPlotOutputConfig:
	return ReconstructionBranchPlotOutputConfig(
		relpath="branch_plots/velocities",
		manifest_relpath="branch_velocities_manifest.json",
	)


@dataclass(frozen=True)
class ReconstructionPlotReconsPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/plot_recons_summary.json"


@dataclass(frozen=True)
class ReconstructionPlotBranchPropagationsPhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "context/plot_branch_propagations_summary.json"
	branch_scope: str = "raw"
	display: ReconstructionBranchPropagationDisplayConfig = field(
		default_factory=ReconstructionBranchPropagationDisplayConfig
	)
	output: ReconstructionBranchPlotOutputConfig = field(default_factory=_default_branch_propagations_output_config)


@dataclass(frozen=True)
class ReconstructionPlotBranchVelocitiesPhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "context/plot_branch_velocities_summary.json"
	branch_scope: str = "raw"
	display: ReconstructionBranchVelocityDisplayConfig = field(
		default_factory=ReconstructionBranchVelocityDisplayConfig
	)
	output: ReconstructionBranchPlotOutputConfig = field(default_factory=_default_branch_velocities_output_config)


@dataclass(frozen=True)
class ReconstructionUnitSummaryDisplayConfig:
	show_title: bool = False
	show_summary_unit_label: bool = False
	summary_unit_label_fontsize: float = 24.0
	summary_unit_label_x_frac: float = 0.015
	summary_unit_label_y_frac: float = 0.985
	recon_show_unit_label: bool | None = None
	recon_show_branch_legend: bool | None = None
	velocity_show_title: bool | None = None
	show_velocity_legend: bool | None = None
	reserve_velocity_legend_space: bool | None = None
	velocity_legend_width: float = 2.25
	top_row_panel_gap_width: float | None = None
	top_row_height: float | None = None
	propagation_row_height: float | None = None
	circle_panel_width: float | None = None
	velocity_panel_width: float | None = None
	propagation_panel_width: float | None = None
	recon_x_offset_frac: float = 0.0
	recon_y_offset_frac: float = 0.0
	velocity_x_offset_frac: float = 0.0
	velocity_y_offset_frac: float = 0.0
	propagation_x_offset_frac: float = 0.0
	propagation_y_offset_frac: float = 0.0


@dataclass(frozen=True)
class ReconstructionUnitSummaryOutputConfig:
	write_png: bool = True
	write_svg: bool = False
	relpath: str = "reports/unit_summary"
	dpi: float = 300.0


@dataclass(frozen=True)
class ReconstructionPlotUnitSummaryPhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "context/plot_unit_summary_summary.json"
	display: ReconstructionUnitSummaryDisplayConfig = field(
		default_factory=ReconstructionUnitSummaryDisplayConfig
	)
	output: ReconstructionUnitSummaryOutputConfig = field(
		default_factory=ReconstructionUnitSummaryOutputConfig
	)


@dataclass(frozen=True)
class ReconstructionAvReconsConfig:
	write_pdf: bool = False
	pdf_relpath: str = "av_recons.pdf"


@dataclass(frozen=True)
class ReconstructionReportReconsPhaseConfig:
	enabled: bool = True
	summary_json_relpath: str = "context/report_recons_summary.json"
	av_recons: ReconstructionAvReconsConfig = field(default_factory=ReconstructionAvReconsConfig)


@dataclass(frozen=True)
class ReconstructionReportFullChipLayoutPhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "context/report_full_chip_layout_summary.json"
	branch_scope: str = "raw"
	unit_colors: ReconstructionFullChipLayoutColorConfig = field(
		default_factory=ReconstructionFullChipLayoutColorConfig
	)
	display: ReconstructionFullChipLayoutDisplayConfig = field(
		default_factory=ReconstructionFullChipLayoutDisplayConfig
	)
	output: ReconstructionFullChipLayoutOutputConfig = field(
		default_factory=ReconstructionFullChipLayoutOutputConfig
	)


@dataclass(frozen=True)
class ReconstructionReportSummariesPhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "context/report_summaries_summary.json"
	write_pdf: bool = True
	pdf_relpath: str = "reports/reconstruct_summary_deck.pdf"


@dataclass(frozen=True)
class ReconstructionClearTemplatesCachePhaseConfig:
	enabled: bool = False
	summary_json_relpath: str = "reports/clear_templates_cache_summary.json"
	keep_merged_per_unit_outputs: bool = True
	keep_full_channels_templates: bool = False


@dataclass(frozen=True)
class ReconstructionPhasesConfig:
	clear_templates_cache: ReconstructionClearTemplatesCachePhaseConfig = field(
		default_factory=ReconstructionClearTemplatesCachePhaseConfig
	)
	generate_gtrs: ReconstructionGenerateGtrsPhaseConfig = field(default_factory=ReconstructionGenerateGtrsPhaseConfig)
	plot_recons: ReconstructionPlotReconsPhaseConfig = field(default_factory=ReconstructionPlotReconsPhaseConfig)
	plot_branch_propagations: ReconstructionPlotBranchPropagationsPhaseConfig = field(
		default_factory=ReconstructionPlotBranchPropagationsPhaseConfig
	)
	plot_branch_velocities: ReconstructionPlotBranchVelocitiesPhaseConfig = field(
		default_factory=ReconstructionPlotBranchVelocitiesPhaseConfig
	)
	plot_unit_summary: ReconstructionPlotUnitSummaryPhaseConfig = field(
		default_factory=ReconstructionPlotUnitSummaryPhaseConfig
	)
	report_recons: ReconstructionReportReconsPhaseConfig = field(default_factory=ReconstructionReportReconsPhaseConfig)
	report_full_chip_layout: ReconstructionReportFullChipLayoutPhaseConfig = field(
		default_factory=ReconstructionReportFullChipLayoutPhaseConfig
	)
	report_summaries: ReconstructionReportSummariesPhaseConfig = field(
		default_factory=ReconstructionReportSummariesPhaseConfig
	)


@dataclass(frozen=True)
class ReconstructionInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path
	final_output_root: Path | None = None
	templates_inputs: TemplatesInputs | None = None
	debug_prints: bool = False

	output_rel_root: str = "recon_outputs"
	reports: ReconstructionReportsConfig = field(default_factory=ReconstructionReportsConfig)
	branch_colors: ReconstructionBranchColorsConfig = field(default_factory=ReconstructionBranchColorsConfig)
	write_summary_png: bool = False
	summary_png_relpath: str = "summary.png"
	summary_grid_ncols: int = 5
	write_report_md: bool = False
	report_md_relpath: str = "report.md"
	cleanup_failed_unit_outputs: bool = False
	failed_units_summary_relpath: str = "failed_units_summary.json"
	per_unit_outputs: PerUnitOutputsConfig = field(default_factory=PerUnitOutputsConfig)

	unit_ids: list[Any] | None = None
	unit_limit: int | None = None
	limit_segments: int | None = None
	phase_sequence: tuple[str, ...] | None = None
	phases: ReconstructionPhasesConfig = field(default_factory=ReconstructionPhasesConfig)

	use_full_channels_templates: bool = True
	require_full_channels_templates: bool = True

	force_restart: bool = False
	force_replot: bool = False
	n_jobs: int = 1

	axon_velocity_params: dict[str, Any] = field(default_factory=dict)
	axon_velocity_repo_root: Path | None = None
	probe_geometry: Any | None = None
