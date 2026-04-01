from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.templates.models.inputs import FootprintMapGridReportConfig


@dataclass(frozen=True)
class CircleReconDisplayConfig:
	base: str = "template_circles"
	channel_scope: str = "nodes_and_branches"
	zoom_padding_percent: float = 20.0
	force_center_soma: bool = True
	branch_scope: str = "raw"
	unique_color_per_branch: bool = True
	show_branch_labels: bool = False
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
class PerUnitOutputsConfig:
	unit_reldir: str = "units/{unit_id:04d}/"

	write_branches_raw_json: bool = True
	branches_raw_relpath: str = "branches_raw.json"

	write_branches_json: bool = True
	branches_relpath: str = "branches.json"

	write_heuristics_json: bool = True
	heuristics_relpath: str = "heuristics.json"

	write_gtr_pkl: bool = True
	gtr_pkl_relpath: str = "gtr.pkl"
	template_source: str = "square"

	write_gtr_json: bool = False
	gtr_json_relpath: str = "gtr.json"

	write_amplitude_map_png: bool = False
	amplitude_map_png_relpath: str = "amplitude_map.png"
	amplitude_map_heatmap: SharedHeatmapConfig = field(default_factory=SharedHeatmapConfig)
	circle_recon: CircleReconConfig = field(default_factory=CircleReconConfig)


@dataclass(frozen=True)
class ReconstructionGridReportsConfig:
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
class ReconstructionInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path
	final_output_root: Path | None = None

	output_rel_root: str = "recon_outputs"
	reports: ReconstructionReportsConfig = field(default_factory=ReconstructionReportsConfig)
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
	load_assets_from_v2pipeline_templates_stage: bool = False

	use_full_channels_templates: bool = True
	require_full_channels_templates: bool = True

	force_restart: bool = False
	force_replot: bool = False
	n_jobs: int = 1

	axon_velocity_params: dict[str, Any] = field(default_factory=dict)
	axon_velocity_repo_root: Path | None = None
	probe_geometry: Any | None = None
