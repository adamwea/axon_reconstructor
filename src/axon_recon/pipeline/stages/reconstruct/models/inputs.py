from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig


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
class ReconstructionInputs:
	h5_path: Path
	stream_id: str
	mea_output_root: Path

	output_rel_root: str = "recon_outputs"
	write_summary_png: bool = False
	summary_png_relpath: str = "summary.png"
	summary_grid_ncols: int = 5
	write_report_md: bool = False
	report_md_relpath: str = "report.md"
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
