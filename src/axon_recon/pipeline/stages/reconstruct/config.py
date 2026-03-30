from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig
from axon_recon.pipeline.shared.plotting import build_stage_plot_block
from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.templates.config import _build_footprint_grid_report_config
from axon_recon.pipeline.stages.templates.config import parse_probe_geometry_from_data_config

from ...execution.context import ExecutionTarget
from .models.inputs import (
	CircleReconConfig,
	CircleReconDisplayConfig,
	CircleReconOutputConfig,
	PerUnitOutputsConfig,
	ReconstructionGridReportsConfig,
	ReconstructionInputs,
	ReconstructionReportsConfig,
)


def _as_bool(value: Any, default: bool) -> bool:
	if value is None:
		return bool(default)
	if isinstance(value, bool):
		return value
	token = str(value).strip().lower()
	if token in {"1", "true", "yes", "on"}:
		return True
	if token in {"0", "false", "no", "off"}:
		return False
	return bool(default)


def _resolve_data_config_path(runtime_config_path: Path, data_ref: str | None) -> Path:
	if not data_ref:
		raise ValueError("Runtime config must define data: <path-to-data-config>")
	p = Path(str(data_ref)).expanduser()
	if not p.is_absolute():
		p = (runtime_config_path.parent / p).resolve()
	return p


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
	merged = dict(base)
	for key, value in override.items():
		existing = merged.get(key)
		if isinstance(existing, dict) and isinstance(value, dict):
			merged[key] = _deep_merge_dict(existing, value)
		else:
			merged[key] = value
	return merged


def _normalize_png_relpath(raw: Any, default: str) -> str:
	text = str(raw).strip() if raw is not None else str(default)
	if not text:
		text = str(default)
	if "." not in Path(text).name:
		return f"{text}.png"
	return text


def _get_reconstruct_amplitude_map_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
			"stages.reconstruct.outputs.amplitude_map",
		),
		global_paths=(
			"default",
			"reconstruct",
			"reconstruct.amplitude_map",
			"footprint",
			"footprint_plots.default",
			"footprint_plots.amplitude_map",
		),
	)
	per_unit_block = runtime_config.get("stages.reconstruct.outputs.per_unit_outputs.amplitude_map", {})
	if isinstance(per_unit_block, dict) and per_unit_block:
		return _deep_merge_dict(stage_block, dict(per_unit_block))
	return stage_block


@dataclass(frozen=True)
class ReconstructionStageConfig:
	output_rel_root: str
	reports: ReconstructionReportsConfig
	write_summary_png: bool
	summary_png_relpath: str
	summary_grid_ncols: int
	write_report_md: bool
	report_md_relpath: str
	cleanup_failed_unit_outputs: bool
	failed_units_summary_relpath: str
	per_unit_outputs: PerUnitOutputsConfig
	unit_ids: list[int] | None
	unit_limit: int | None
	load_assets_from_v2pipeline_templates_stage: bool
	use_full_channels_templates: bool
	require_full_channels_templates: bool
	force_restart: bool
	force_replot: bool
	axon_velocity_params: dict[str, Any]


def parse_reconstruction_stage_config(
	*,
	runtime_config: RuntimeConfig,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> ReconstructionStageConfig:
	stage_cfg = runtime_config.get("stages.reconstruct", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	try:
		from axon_recon.pipeline.stages.templates.config import parse_templates_stage_config

		tpl_stage_cfg = parse_templates_stage_config(runtime_config=runtime_config)
		tpl_circles_defaults = tpl_stage_cfg.per_unit_outputs.template_circles
		tpl_footprint_plots_defaults = getattr(tpl_stage_cfg.per_unit_outputs, "footprint_plots", None)
		tpl_footprint_amplitude_defaults = getattr(tpl_footprint_plots_defaults, "amplitude_map", None)
		tpl_footprint_latency_defaults = getattr(tpl_footprint_plots_defaults, "latency_map", None)
	except Exception:
		tpl_circles_defaults = None
		tpl_footprint_amplitude_defaults = None
		tpl_footprint_latency_defaults = None
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	inputs_cfg = stage_cfg.get("inputs", {}) if isinstance(stage_cfg.get("inputs", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}
	reports_cfg = outputs_cfg.get("reports", {}) if isinstance(outputs_cfg.get("reports", {}), dict) else {}
	grids_cfg = reports_cfg.get("grids", {}) if isinstance(reports_cfg.get("grids", {}), dict) else {}
	circle_recon_grid_cfg = grids_cfg.get("circle_recon_grid", {}) if isinstance(grids_cfg.get("circle_recon_grid", {}), dict) else {}
	per_unit_cfg = outputs_cfg.get("per_unit_outputs", {}) if isinstance(outputs_cfg.get("per_unit_outputs", {}), dict) else {}
	av_cfg = stage_cfg.get("av", {}) if isinstance(stage_cfg.get("av", {}), dict) else {}
	amplitude_map_cfg = _get_reconstruct_amplitude_map_block(runtime_config)

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if force_replot_override is not None:
		force_replot = bool(force_replot_override)

	unit_limit_raw = stage_cfg.get("unit_limit", None)
	unit_limit: int | None
	if unit_limit_raw is None:
		unit_limit = None
	else:
		try:
			parsed = int(unit_limit_raw)
			unit_limit = parsed if parsed > 0 else None
		except Exception:
			unit_limit = None

	unit_ids = [int(unit_id_override)] if unit_id_override is not None else None
	load_assets_from_v2pipeline_templates_stage = _as_bool(
		inputs_cfg.get(
			"load_assets_from_v2pipeline_templates_stage",
			inputs_cfg.get("load_assets_from_v2pipeline_tempaltes_stage", False),
		),
		False,
	)

	write_summary_png = _as_bool(outputs_cfg.get("write_summary", False), False)
	summary_png_relpath = _normalize_png_relpath(outputs_cfg.get("summary_relpath", "summary.png"), "summary.png")
	write_report_md = _as_bool(outputs_cfg.get("write_report_md", False), False)
	report_md_relpath = str(outputs_cfg.get("report_md_relpath", "report.md"))
	cleanup_failed_unit_outputs = _as_bool(outputs_cfg.get("cleanup_failed_unit_outputs", False), False)
	failed_units_summary_relpath = str(outputs_cfg.get("failed_units_summary_relpath", "failed_units_summary.json"))
	try:
		summary_grid_ncols = max(1, int(outputs_cfg.get("summary_grid_ncols", 5)))
	except Exception:
		summary_grid_ncols = 5

	if "write_amplitude_map_png" in per_unit_cfg:
		write_amplitude_map_png = _as_bool(per_unit_cfg.get("write_amplitude_map_png", False), False)
	else:
		write_amplitude_map_png = _as_bool(amplitude_map_cfg.get("write_png", False), False)

	if "amplitude_map_png_relpath" in per_unit_cfg:
		amplitude_map_png_relpath = str(per_unit_cfg.get("amplitude_map_png_relpath", "amplitude_map.png"))
	else:
		amplitude_map_png_relpath = _normalize_png_relpath(
			amplitude_map_cfg.get("relpath", "amplitude_map"),
			"amplitude_map.png",
		)

	recon_plots_cfg = per_unit_cfg.get("recon_plots", {}) if isinstance(per_unit_cfg.get("recon_plots", {}), dict) else {}
	circle_recon_cfg = recon_plots_cfg.get("circle_recon", {}) if isinstance(recon_plots_cfg.get("circle_recon", {}), dict) else {}
	circle_display_cfg = (
		circle_recon_cfg.get("display", {}) if isinstance(circle_recon_cfg.get("display", {}), dict) else {}
	)
	circle_output_cfg = (
		circle_recon_cfg.get("output", {}) if isinstance(circle_recon_cfg.get("output", {}), dict) else {}
	)

	default_circle_base = "template_circles"
	default_circle_force_center = bool(getattr(tpl_circles_defaults, "force_center_soma", True))
	default_circle_unique_color = bool(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "unique_color_per_branch", True)
	)
	default_circle_show_labels = bool(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "show_branch_labels", False)
	)
	default_circle_color_scheme = str(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "color_scheme", "tab20") or "tab20"
	)
	default_circle_node_lw = float(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "node_border_linewidth", 0.35)
	)
	default_circle_node_outline_lw = float(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "node_outline_linewidth", 0.0)
	)
	default_circle_edge_lw = float(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "edge_linewidth", 0.8)
	)
	default_circle_branch_outline_lw = float(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "branch_outline_linewidth", 0.0)
	)
	default_circle_dpi = float(getattr(tpl_circles_defaults, "dpi", 300.0))

	circle_base = str(circle_display_cfg.get("base", default_circle_base) or default_circle_base).strip().lower()
	if circle_base not in {"template_circles", "amplitude_map", "latency_map"}:
		circle_base = default_circle_base
	circle_channel_scope = str(circle_display_cfg.get("channel_scope", "nodes_and_branches") or "nodes_and_branches").strip().lower()
	if circle_channel_scope not in {"nodes_and_branches", "branches_only", "nodes_only"}:
		circle_channel_scope = "nodes_and_branches"
	try:
		circle_zoom_padding_percent = float(circle_display_cfg.get("zoom_padding_percent", 20.0))
	except Exception:
		circle_zoom_padding_percent = 20.0
	circle_zoom_padding_percent = float(max(0.0, circle_zoom_padding_percent))
	circle_branch_scope = str(circle_display_cfg.get("branch_scope", "raw") or "raw").strip().lower()
	if circle_branch_scope not in {"raw", "clean"}:
		circle_branch_scope = "raw"
	try:
		circle_node_border_lw = float(
			circle_display_cfg.get("node_border_linewidth", circle_display_cfg.get("node_inline_linewidth_pt", default_circle_node_lw))
		)
	except Exception:
		circle_node_border_lw = default_circle_node_lw
	try:
		circle_node_outline_lw = float(circle_display_cfg.get("node_outline_linewidth", default_circle_node_outline_lw))
	except Exception:
		circle_node_outline_lw = default_circle_node_outline_lw
	try:
		circle_edge_lw = float(circle_display_cfg.get("edge_linewidth", circle_display_cfg.get("branch_linewidth_pt", default_circle_edge_lw)))
	except Exception:
		circle_edge_lw = default_circle_edge_lw
	try:
		circle_branch_outline_lw = float(circle_display_cfg.get("branch_outline_linewidth", default_circle_branch_outline_lw))
	except Exception:
		circle_branch_outline_lw = default_circle_branch_outline_lw

	raw_node_outline_color = circle_display_cfg.get("node_outline_color", None)
	if raw_node_outline_color is None:
		circle_node_outline_color = None
	else:
		node_outline_text = str(raw_node_outline_color).strip()
		circle_node_outline_color = (node_outline_text if node_outline_text else None)

	raw_branch_outline_color = circle_display_cfg.get("branch_outline_color", None)
	if raw_branch_outline_color is None:
		circle_branch_outline_color = None
	else:
		branch_outline_text = str(raw_branch_outline_color).strip()
		circle_branch_outline_color = (branch_outline_text if branch_outline_text else None)

	circle_relpath = str(circle_output_cfg.get("relpath", "circle_recon") or "circle_recon").strip()
	if not circle_relpath:
		circle_relpath = "circle_recon"
	try:
		circle_dpi = float(circle_output_cfg.get("dpi", default_circle_dpi))
	except Exception:
		circle_dpi = default_circle_dpi

	circle_recon = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base=circle_base,
			channel_scope=circle_channel_scope,
			zoom_padding_percent=circle_zoom_padding_percent,
			force_center_soma=_as_bool(circle_display_cfg.get("force_center_soma", default_circle_force_center), default_circle_force_center),
			branch_scope=circle_branch_scope,
			unique_color_per_branch=_as_bool(circle_display_cfg.get("unique_color_per_branch", default_circle_unique_color), default_circle_unique_color),
			show_branch_labels=_as_bool(circle_display_cfg.get("show_branch_labels", default_circle_show_labels), default_circle_show_labels),
			color_scheme=str(circle_display_cfg.get("color_scheme", default_circle_color_scheme) or default_circle_color_scheme),
			node_outline_color=circle_node_outline_color,
				node_outline_linewidth=float(max(0.0, circle_node_outline_lw)),
			branch_outline_color=circle_branch_outline_color,
				branch_outline_linewidth=float(max(0.0, circle_branch_outline_lw)),
			node_border_linewidth=float(max(0.0, circle_node_border_lw)),
			edge_linewidth=float(max(0.0, circle_edge_lw)),
		),
		output=CircleReconOutputConfig(
			write_png=_as_bool(circle_output_cfg.get("write_png", False), False),
			write_svg=_as_bool(circle_output_cfg.get("write_svg", False), False),
			relpath=circle_relpath,
			dpi=float(max(72.0, circle_dpi)),
		),
		base_template_circles=tpl_circles_defaults,
		base_footprint_amplitude=tpl_footprint_amplitude_defaults,
		base_footprint_latency=tpl_footprint_latency_defaults,
	)

	per_unit = PerUnitOutputsConfig(
		unit_reldir=str(per_unit_cfg.get("unit_reldir", "units/{unit_id:04d}/")),
		write_branches_raw_json=_as_bool(per_unit_cfg.get("write_branches_raw_json", True), True),
		branches_raw_relpath=str(per_unit_cfg.get("branches_raw_relpath", "branches_raw.json")),
		write_branches_json=_as_bool(per_unit_cfg.get("write_branches_json", True), True),
		branches_relpath=str(per_unit_cfg.get("branches_relpath", "branches.json")),
		write_heuristics_json=_as_bool(per_unit_cfg.get("write_heuristics_json", True), True),
		heuristics_relpath=str(per_unit_cfg.get("heuristics_relpath", "heuristics.json")),
		write_gtr_pkl=_as_bool(per_unit_cfg.get("write_gtr_pkl", True), True),
		gtr_pkl_relpath=str(per_unit_cfg.get("gtr_pkl_relpath", "gtr.pkl")),
		template_source=(
			str(per_unit_cfg.get("template_source", "square") or "square").strip().lower()
			if str(per_unit_cfg.get("template_source", "square") or "square").strip().lower() in {"square", "merged", "full", "full_from_merged"}
			else "square"
		),
		write_gtr_json=_as_bool(per_unit_cfg.get("write_gtr_json", False), False),
		gtr_json_relpath=str(per_unit_cfg.get("gtr_json_relpath", "gtr.json")),
		write_amplitude_map_png=write_amplitude_map_png,
		amplitude_map_png_relpath=amplitude_map_png_relpath,
		amplitude_map_heatmap=SharedHeatmapConfig.from_block(amplitude_map_cfg),
		circle_recon=circle_recon,
	)
	reports = ReconstructionReportsConfig(
		grids=ReconstructionGridReportsConfig(
			circle_recon_grid=_build_footprint_grid_report_config(
				circle_recon_grid_cfg,
				pdf_relpath_default="reports/circle_recon_grid.pdf",
				png_relpath_default="reports/circle_recon_grid.png",
				svg_relpath_default="reports/circle_recon_grid.svg",
				temp_svg_relpath_default="reports/circle_recon_grid__temp.svg",
			),
		),
		overwrite_on_unit_rerun=_as_bool(reports_cfg.get("overwrite_on_unit_rerun", False), False),
	)

	return ReconstructionStageConfig(
		output_rel_root=str(outputs_cfg.get("output_rel_root", "recon_outputs")),
		reports=reports,
		write_summary_png=write_summary_png,
		summary_png_relpath=summary_png_relpath,
		summary_grid_ncols=summary_grid_ncols,
		write_report_md=write_report_md,
		report_md_relpath=report_md_relpath,
		cleanup_failed_unit_outputs=cleanup_failed_unit_outputs,
		failed_units_summary_relpath=failed_units_summary_relpath,
		per_unit_outputs=per_unit,
		unit_ids=unit_ids,
		unit_limit=unit_limit,
		load_assets_from_v2pipeline_templates_stage=load_assets_from_v2pipeline_templates_stage,
		use_full_channels_templates=True,
		require_full_channels_templates=True,
		force_restart=force_restart,
		force_replot=force_replot,
		axon_velocity_params=dict(av_cfg),
	)


def build_reconstruction_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: ReconstructionStageConfig,
	unit_workers: int,
	probe_geometry: Any | None = None,
) -> ReconstructionInputs:
	return ReconstructionInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=stage_config.output_rel_root,
		reports=stage_config.reports,
		write_summary_png=stage_config.write_summary_png,
		summary_png_relpath=stage_config.summary_png_relpath,
		summary_grid_ncols=stage_config.summary_grid_ncols,
		write_report_md=stage_config.write_report_md,
		report_md_relpath=stage_config.report_md_relpath,
		cleanup_failed_unit_outputs=stage_config.cleanup_failed_unit_outputs,
		failed_units_summary_relpath=stage_config.failed_units_summary_relpath,
		per_unit_outputs=stage_config.per_unit_outputs,
		unit_ids=stage_config.unit_ids,
		unit_limit=stage_config.unit_limit,
		load_assets_from_v2pipeline_templates_stage=stage_config.load_assets_from_v2pipeline_templates_stage,
		use_full_channels_templates=stage_config.use_full_channels_templates,
		require_full_channels_templates=stage_config.require_full_channels_templates,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		n_jobs=max(1, int(unit_workers)),
		axon_velocity_params=dict(stage_config.axon_velocity_params),
		probe_geometry=probe_geometry,
	)


def load_reconstruction_inputs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> ReconstructionInputs:
	runtime_config_path = Path(config_path).expanduser().resolve()
	runtime_cfg = RuntimeConfig.load(runtime_config_path)
	data_cfg_path = _resolve_data_config_path(runtime_config_path, runtime_cfg.get("data", None))
	data_cfg = RuntimeConfig.load(data_cfg_path)

	datasets = data_cfg.get("datasets", [])
	if not isinstance(datasets, list) or not datasets:
		raise ValueError("Data config must define a non-empty datasets list")
	selected = next((ds for ds in datasets if isinstance(ds, dict) and _as_bool(ds.get("include_in_runtime", False), False)), None)
	if selected is None:
		selected = next((ds for ds in datasets if isinstance(ds, dict)), None)
	if selected is None:
		raise ValueError("No valid dataset object found in data config")

	h5_raw = selected.get("raw_data_h5_path")
	if not h5_raw:
		raise ValueError("Selected dataset missing raw_data_h5_path")
	h5_path = Path(str(h5_raw)).expanduser().resolve()
	output_root = Path(str(data_cfg.get("output_root", ""))).expanduser().resolve()
	if str(output_root).strip() == "":
		raise ValueError("Data config missing output_root")

	wells = selected.get("wells", [])
	stream_id = "well000"
	if isinstance(wells, list) and wells and isinstance(wells[0], dict) and wells[0].get("well_id"):
		stream_id = str(wells[0].get("well_id"))

	stage_cfg = parse_reconstruction_stage_config(
		runtime_config=runtime_cfg,
		unit_id_override=unit_id_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=data_cfg)

	return ReconstructionInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		output_rel_root=stage_cfg.output_rel_root,
		reports=stage_cfg.reports,
		write_summary_png=stage_cfg.write_summary_png,
		summary_png_relpath=stage_cfg.summary_png_relpath,
		summary_grid_ncols=stage_cfg.summary_grid_ncols,
		write_report_md=stage_cfg.write_report_md,
		report_md_relpath=stage_cfg.report_md_relpath,
		cleanup_failed_unit_outputs=stage_cfg.cleanup_failed_unit_outputs,
		failed_units_summary_relpath=stage_cfg.failed_units_summary_relpath,
		per_unit_outputs=stage_cfg.per_unit_outputs,
		unit_ids=stage_cfg.unit_ids,
		unit_limit=stage_cfg.unit_limit,
		load_assets_from_v2pipeline_templates_stage=stage_cfg.load_assets_from_v2pipeline_templates_stage,
		use_full_channels_templates=stage_cfg.use_full_channels_templates,
		require_full_channels_templates=stage_cfg.require_full_channels_templates,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		n_jobs=1,
		axon_velocity_params=stage_cfg.axon_velocity_params,
		probe_geometry=probe_geometry,
	)
