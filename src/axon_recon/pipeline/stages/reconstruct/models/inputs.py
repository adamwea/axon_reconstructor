from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig


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

	write_gtr_json: bool = False
	gtr_json_relpath: str = "gtr.json"

	write_amplitude_map_png: bool = False
	amplitude_map_png_relpath: str = "amplitude_map.png"
	amplitude_map_heatmap: SharedHeatmapConfig = field(default_factory=SharedHeatmapConfig)


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

	use_full_channels_templates: bool = True
	require_full_channels_templates: bool = True

	force_restart: bool = False
	force_replot: bool = False
	n_jobs: int = 1

	axon_velocity_params: dict[str, Any] = field(default_factory=dict)
	axon_velocity_repo_root: Path | None = None

