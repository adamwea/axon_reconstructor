from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig
from axon_recon.pipeline.shared.grid_sorting import normalize_grid_sort_by
from axon_recon.pipeline.shared.plotting import build_stage_plot_block
from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.templates.config import _build_footprint_grid_report_config
from axon_recon.pipeline.stages.templates.config import parse_probe_geometry_from_data_config

from ...execution.context import ExecutionTarget
from .models.inputs import (
	ReconstructionAvReconsConfig,
	ReconstructionAxonVelocityPhaseConfig,
	CircleReconConfig,
	CircleReconDisplayConfig,
	CircleReconOutputConfig,
	PerUnitOutputsConfig,
	ReconstructionDiagnosticFigureConfig,
	ReconstructionGenerateGtrsOutputsConfig,
	ReconstructionGenerateGtrsPhaseConfig,
	ReconstructionGridReportsConfig,
	ReconstructionInputs,
	ReconstructionPhasesConfig,
	ReconstructionPlotReconsPhaseConfig,
	ReconstructionReportReconsPhaseConfig,
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


def _normalize_unit_ids(raw: Any) -> list[int] | None:
	if raw is None:
		return None
	if isinstance(raw, str):
		tokens = [token.strip() for token in raw.split(",")]
	elif isinstance(raw, (list, tuple, set)):
		tokens = list(raw)
	else:
		tokens = [raw]

	normalized: list[int] = []
	seen: set[int] = set()
	for token in tokens:
		if token is None:
			continue
		try:
			value = int(token)
		except Exception:
			continue
		if value < 0:
			continue
		if value in seen:
			continue
		seen.add(value)
		normalized.append(value)

	return normalized or None


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


def _phase_enabled(block: Any, default: bool = True) -> bool:
	if not isinstance(block, dict):
		return bool(default)
	if "enabled" in block:
		return _as_bool(block.get("enabled"), default)
	if "enable" in block:
		return _as_bool(block.get("enable"), default)
	return bool(default)


def _parse_optional_positive_int(value: Any) -> int | None:
	try:
		parsed = int(value)
	except Exception:
		return None
	if parsed <= 0:
		return None
	return int(parsed)


def _normalize_reconstruct_template_source(raw: Any, default: str = "square") -> str:
	text = str(raw if raw is not None else default).strip().lower()
	if text in {"square", "merged", "full", "full_from_merged"}:
		return text
	return str(default)


def _build_reconstruct_figure_output_config(
	block: Any,
	*,
	default_relpath: str,
) -> ReconstructionDiagnosticFigureConfig:
	data = block if isinstance(block, dict) else {}
	return ReconstructionDiagnosticFigureConfig(
		write_png=_as_bool(data.get("write_png", False), False),
		write_svg=_as_bool(data.get("write_svg", False), False),
		relpath=str(data.get("relpath", default_relpath) or default_relpath).strip() or str(default_relpath),
		dpi=float(data.get("dpi", 300.0) or 300.0),
	)


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
	phases: ReconstructionPhasesConfig
	use_full_channels_templates: bool
	require_full_channels_templates: bool
	force_restart: bool
	force_replot: bool
	axon_velocity_params: dict[str, Any]


def parse_reconstruction_stage_config(
	*,
	runtime_config: RuntimeConfig,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
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
	phases_cfg = stage_cfg.get("phases", {}) if isinstance(stage_cfg.get("phases", {}), dict) else {}
	generate_gtrs_cfg = phases_cfg.get("generate_gtrs", {}) if isinstance(phases_cfg.get("generate_gtrs", {}), dict) else {}
	plot_recons_cfg = phases_cfg.get("plot_recons", {}) if isinstance(phases_cfg.get("plot_recons", {}), dict) else {}
	report_recons_cfg = phases_cfg.get("report_recons", {}) if isinstance(phases_cfg.get("report_recons", {}), dict) else {}
	generate_gtrs_resources_cfg = (
		generate_gtrs_cfg.get("resources", {}) if isinstance(generate_gtrs_cfg.get("resources", {}), dict) else {}
	)
	phase_generate_outputs_cfg = (
		generate_gtrs_cfg.get("outputs", {}) if isinstance(generate_gtrs_cfg.get("outputs", {}), dict) else {}
	)
	phase_generate_diagnostic_figs_cfg = (
		phase_generate_outputs_cfg.get("diagnostic_figs", {})
		if isinstance(phase_generate_outputs_cfg.get("diagnostic_figs", {}), dict)
		else {}
	)
	phase_plot_outputs_cfg = (
		plot_recons_cfg.get("outputs", {}) if isinstance(plot_recons_cfg.get("outputs", {}), dict) else {}
	)
	reports_cfg = outputs_cfg.get("reports", {}) if isinstance(outputs_cfg.get("reports", {}), dict) else {}
	grids_cfg = reports_cfg.get("grids", {}) if isinstance(reports_cfg.get("grids", {}), dict) else {}
	circle_recon_grid_cfg = grids_cfg.get("circle_recon_grid", {}) if isinstance(grids_cfg.get("circle_recon_grid", {}), dict) else {}
	per_unit_cfg = outputs_cfg.get("per_unit_outputs", {}) if isinstance(outputs_cfg.get("per_unit_outputs", {}), dict) else {}
	legacy_diagnostic_figs_cfg = (
		per_unit_cfg.get("diagnostic_figs", {}) if isinstance(per_unit_cfg.get("diagnostic_figs", {}), dict) else {}
	)
	legacy_channel_selection_fig_cfg = (
		legacy_diagnostic_figs_cfg.get("channel_selection", {})
		if isinstance(legacy_diagnostic_figs_cfg.get("channel_selection", {}), dict)
		else {}
	)
	legacy_axon_reconstruction_fig_cfg = (
		legacy_diagnostic_figs_cfg.get("axon_reconstruction", {})
		if isinstance(legacy_diagnostic_figs_cfg.get("axon_reconstruction", {}), dict)
		else {}
	)
	phase_channel_selection_fig_cfg = (
		phase_generate_diagnostic_figs_cfg.get("channel_selection", {})
		if isinstance(phase_generate_diagnostic_figs_cfg.get("channel_selection", {}), dict)
		else {}
	)
	phase_axon_reconstruction_fig_cfg = (
		phase_generate_diagnostic_figs_cfg.get("axon_reconstruction", {})
		if isinstance(phase_generate_diagnostic_figs_cfg.get("axon_reconstruction", {}), dict)
		else {}
	)
	av_cfg: dict[str, Any] = {}
	legacy_av_cfg = stage_cfg.get("av", {})
	if isinstance(legacy_av_cfg, dict):
		av_cfg.update(dict(legacy_av_cfg))
	canonical_av_cfg = stage_cfg.get("axon_velocity", {})
	if isinstance(canonical_av_cfg, dict):
		av_cfg.update(dict(canonical_av_cfg))
	phase_axon_velocity_cfg = (
		generate_gtrs_cfg.get("axon_velocity", {})
		if isinstance(generate_gtrs_cfg.get("axon_velocity", {}), dict)
		else {}
	)
	phase_axon_velocity_params = (
		phase_axon_velocity_cfg.get("params", {})
		if isinstance(phase_axon_velocity_cfg.get("params", {}), dict)
		else {}
	)
	if phase_axon_velocity_params:
		av_cfg.update(dict(phase_axon_velocity_params))
	generate_gtrs_unit_procs = _parse_optional_positive_int(
		generate_gtrs_resources_cfg.get("unit_procs", generate_gtrs_cfg.get("unit_procs", None))
	)
	generate_gtrs_unit_batch_size = _parse_optional_positive_int(
		generate_gtrs_resources_cfg.get("unit_batch_size", generate_gtrs_cfg.get("unit_batch_size", None))
	)
	amplitude_map_cfg = _get_reconstruct_amplitude_map_block(runtime_config)
	phase_amplitude_map_cfg = (
		phase_plot_outputs_cfg.get("amplitude_map", {})
		if isinstance(phase_plot_outputs_cfg.get("amplitude_map", {}), dict)
		else {}
	)
	if phase_amplitude_map_cfg:
		amplitude_map_cfg = _deep_merge_dict(amplitude_map_cfg, dict(phase_amplitude_map_cfg))
	channel_selection_figure_cfg = _build_reconstruct_figure_output_config(
		_deep_merge_dict(legacy_channel_selection_fig_cfg, phase_channel_selection_fig_cfg),
		default_relpath="diagnostic_figs/channel_selection",
	)
	axon_reconstruction_figure_cfg = _build_reconstruct_figure_output_config(
		_deep_merge_dict(legacy_axon_reconstruction_fig_cfg, phase_axon_reconstruction_fig_cfg),
		default_relpath="diagnostic_figs/axon_reconstruction",
	)
	generate_gtrs_outputs = ReconstructionGenerateGtrsOutputsConfig(
		write_branches_raw_json=_as_bool(
			phase_generate_outputs_cfg.get("write_branches_raw_json", per_unit_cfg.get("write_branches_raw_json", True)),
			True,
		),
		branches_raw_relpath=str(
			phase_generate_outputs_cfg.get("branches_raw_relpath", per_unit_cfg.get("branches_raw_relpath", "branches_raw.json"))
		),
		write_branches_json=_as_bool(
			phase_generate_outputs_cfg.get("write_branches_json", per_unit_cfg.get("write_branches_json", True)),
			True,
		),
		branches_relpath=str(
			phase_generate_outputs_cfg.get("branches_relpath", per_unit_cfg.get("branches_relpath", "branches.json"))
		),
		write_detection_filter_json=_as_bool(
			phase_generate_outputs_cfg.get(
				"write_detection_filter_json",
				per_unit_cfg.get("write_detection_filter_json", False),
			),
			False,
		),
		detection_filter_relpath=str(
			phase_generate_outputs_cfg.get(
				"detection_filter_relpath",
				per_unit_cfg.get("detection_filter_relpath", "detection_filter.json"),
			)
		),
		write_kurtosis_filter_json=_as_bool(
			phase_generate_outputs_cfg.get(
				"write_kurtosis_filter_json",
				per_unit_cfg.get("write_kurtosis_filter_json", False),
			),
			False,
		),
		kurtosis_filter_relpath=str(
			phase_generate_outputs_cfg.get(
				"kurtosis_filter_relpath",
				per_unit_cfg.get("kurtosis_filter_relpath", "kurtosis_filter.json"),
			)
		),
		write_peak_std_filter_json=_as_bool(
			phase_generate_outputs_cfg.get(
				"write_peak_std_filter_json",
				per_unit_cfg.get("write_peak_std_filter_json", False),
			),
			False,
		),
		peak_std_filter_relpath=str(
			phase_generate_outputs_cfg.get(
				"peak_std_filter_relpath",
				per_unit_cfg.get("peak_std_filter_relpath", "peak_std_filter.json"),
			)
		),
		write_delay_filter_json=_as_bool(
			phase_generate_outputs_cfg.get(
				"write_delay_filter_json",
				per_unit_cfg.get("write_delay_filter_json", False),
			),
			False,
		),
		delay_filter_relpath=str(
			phase_generate_outputs_cfg.get(
				"delay_filter_relpath",
				per_unit_cfg.get("delay_filter_relpath", "delay_filter.json"),
			)
		),
		write_all_filters_json=_as_bool(
			phase_generate_outputs_cfg.get(
				"write_all_filters_json",
				per_unit_cfg.get("write_all_filters_json", False),
			),
			False,
		),
		all_filters_relpath=str(
			phase_generate_outputs_cfg.get(
				"all_filters_relpath",
				per_unit_cfg.get("all_filters_relpath", "all_filters.json"),
			)
		),
		write_heuristics_json=_as_bool(
			phase_generate_outputs_cfg.get("write_heuristics_json", per_unit_cfg.get("write_heuristics_json", True)),
			True,
		),
		heuristics_relpath=str(
			phase_generate_outputs_cfg.get("heuristics_relpath", per_unit_cfg.get("heuristics_relpath", "heuristics.json"))
		),
		write_gtr_pkl=_as_bool(
			phase_generate_outputs_cfg.get("write_gtr_pkl", per_unit_cfg.get("write_gtr_pkl", True)),
			True,
		),
		gtr_pkl_relpath=str(
			phase_generate_outputs_cfg.get("gtr_pkl_relpath", per_unit_cfg.get("gtr_pkl_relpath", "gtr.pkl"))
		),
		template_source=_normalize_reconstruct_template_source(
			phase_generate_outputs_cfg.get("template_source", per_unit_cfg.get("template_source", "square")),
			default="square",
		),
		write_gtr_json=_as_bool(
			phase_generate_outputs_cfg.get("write_gtr_json", per_unit_cfg.get("write_gtr_json", False)),
			False,
		),
		gtr_json_relpath=str(
			phase_generate_outputs_cfg.get("gtr_json_relpath", per_unit_cfg.get("gtr_json_relpath", "gtr.json"))
		),
		channel_selection_figure=channel_selection_figure_cfg,
		axon_reconstruction_figure=axon_reconstruction_figure_cfg,
	)

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if force_replot_override is not None:
		force_replot = bool(force_replot_override)

	unit_limit_raw = execution_cfg.get("unit_limit", stage_cfg.get("unit_limit", None))
	unit_limit: int | None
	if unit_limit_raw is None:
		unit_limit = None
	else:
		try:
			parsed = int(unit_limit_raw)
			unit_limit = parsed if parsed > 0 else None
		except Exception:
			unit_limit = None

	runtime_unit_ids = _normalize_unit_ids(execution_cfg.get("unit_ids", stage_cfg.get("unit_ids", None)))
	if unit_ids_override is not None:
		unit_ids = _normalize_unit_ids(unit_ids_override)
	elif unit_id_override is not None:
		unit_ids = [int(unit_id_override)]
	else:
		unit_ids = runtime_unit_ids
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

	if "write_png" in phase_amplitude_map_cfg:
		write_amplitude_map_png = _as_bool(phase_amplitude_map_cfg.get("write_png", False), False)
	elif "write_amplitude_map_png" in per_unit_cfg:
		write_amplitude_map_png = _as_bool(per_unit_cfg.get("write_amplitude_map_png", False), False)
	else:
		write_amplitude_map_png = _as_bool(amplitude_map_cfg.get("write_png", False), False)

	if "relpath" in phase_amplitude_map_cfg:
		amplitude_map_png_relpath = _normalize_png_relpath(
			phase_amplitude_map_cfg.get("relpath", "amplitude_map"),
			"amplitude_map.png",
		)
	elif "amplitude_map_png_relpath" in per_unit_cfg:
		amplitude_map_png_relpath = str(per_unit_cfg.get("amplitude_map_png_relpath", "amplitude_map.png"))
	else:
		amplitude_map_png_relpath = _normalize_png_relpath(
			amplitude_map_cfg.get("relpath", "amplitude_map"),
			"amplitude_map.png",
		)

	recon_plots_cfg = per_unit_cfg.get("recon_plots", {}) if isinstance(per_unit_cfg.get("recon_plots", {}), dict) else {}
	legacy_circle_recon_cfg = (
		recon_plots_cfg.get("circle_recon", {}) if isinstance(recon_plots_cfg.get("circle_recon", {}), dict) else {}
	)
	phase_circle_recon_cfg = (
		phase_plot_outputs_cfg.get("circle_recon", {})
		if isinstance(phase_plot_outputs_cfg.get("circle_recon", {}), dict)
		else {}
	)
	circle_recon_cfg = _deep_merge_dict(legacy_circle_recon_cfg, phase_circle_recon_cfg)
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
	report_av_recons_cfg = (
		report_recons_cfg.get("av_recons", {}) if isinstance(report_recons_cfg.get("av_recons", {}), dict) else {}
	)
	phases = ReconstructionPhasesConfig(
		generate_gtrs=ReconstructionGenerateGtrsPhaseConfig(
			enabled=_phase_enabled(generate_gtrs_cfg, True),
			summary_json_relpath=str(
				generate_gtrs_cfg.get("summary_json_relpath", "context/generate_gtrs_summary.json")
			),
			unit_procs=generate_gtrs_unit_procs,
			unit_batch_size=generate_gtrs_unit_batch_size,
			outputs=generate_gtrs_outputs,
			axon_velocity=ReconstructionAxonVelocityPhaseConfig(
				enabled=_phase_enabled(phase_axon_velocity_cfg, True),
				params=dict(av_cfg),
			),
		),
		plot_recons=ReconstructionPlotReconsPhaseConfig(
			enabled=_phase_enabled(plot_recons_cfg, True),
			summary_json_relpath=str(plot_recons_cfg.get("summary_json_relpath", "context/plot_recons_summary.json")),
		),
		report_recons=ReconstructionReportReconsPhaseConfig(
			enabled=_phase_enabled(report_recons_cfg, True),
			summary_json_relpath=str(
				report_recons_cfg.get("summary_json_relpath", "context/report_recons_summary.json")
			),
			av_recons=ReconstructionAvReconsConfig(
				write_pdf=_as_bool(report_av_recons_cfg.get("write_pdf", False), False),
				pdf_relpath=str(report_av_recons_cfg.get("pdf_relpath", "av_recons.pdf")),
			),
		),
	)

	per_unit = PerUnitOutputsConfig(
		unit_reldir=str(phase_plot_outputs_cfg.get("unit_reldir", per_unit_cfg.get("unit_reldir", "units/{unit_id:04d}/"))),
		write_branches_raw_json=bool(generate_gtrs_outputs.write_branches_raw_json),
		branches_raw_relpath=str(generate_gtrs_outputs.branches_raw_relpath),
		write_branches_json=bool(generate_gtrs_outputs.write_branches_json),
		branches_relpath=str(generate_gtrs_outputs.branches_relpath),
		write_detection_filter_json=bool(generate_gtrs_outputs.write_detection_filter_json),
		detection_filter_relpath=str(generate_gtrs_outputs.detection_filter_relpath),
		write_kurtosis_filter_json=bool(generate_gtrs_outputs.write_kurtosis_filter_json),
		kurtosis_filter_relpath=str(generate_gtrs_outputs.kurtosis_filter_relpath),
		write_peak_std_filter_json=bool(generate_gtrs_outputs.write_peak_std_filter_json),
		peak_std_filter_relpath=str(generate_gtrs_outputs.peak_std_filter_relpath),
		write_delay_filter_json=bool(generate_gtrs_outputs.write_delay_filter_json),
		delay_filter_relpath=str(generate_gtrs_outputs.delay_filter_relpath),
		write_all_filters_json=bool(generate_gtrs_outputs.write_all_filters_json),
		all_filters_relpath=str(generate_gtrs_outputs.all_filters_relpath),
		write_heuristics_json=bool(generate_gtrs_outputs.write_heuristics_json),
		heuristics_relpath=str(generate_gtrs_outputs.heuristics_relpath),
		write_gtr_pkl=bool(generate_gtrs_outputs.write_gtr_pkl),
		gtr_pkl_relpath=str(generate_gtrs_outputs.gtr_pkl_relpath),
		template_source=str(generate_gtrs_outputs.template_source),
		write_gtr_json=bool(generate_gtrs_outputs.write_gtr_json),
		gtr_json_relpath=str(generate_gtrs_outputs.gtr_json_relpath),
		channel_selection_figure=generate_gtrs_outputs.channel_selection_figure,
		axon_reconstruction_figure=generate_gtrs_outputs.axon_reconstruction_figure,
		write_amplitude_map_png=write_amplitude_map_png,
		amplitude_map_png_relpath=amplitude_map_png_relpath,
		amplitude_map_heatmap=SharedHeatmapConfig.from_block(amplitude_map_cfg),
		circle_recon=circle_recon,
	)
	reports = ReconstructionReportsConfig(
		grids=ReconstructionGridReportsConfig(
			sort_by=normalize_grid_sort_by(
				grids_cfg.get("sort_by", reports_cfg.get("sort_by", "unit_id")),
				default="unit_id",
			),
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
		phases=phases,
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
		final_output_root=(target.final_output_root or target.mea_output_root),
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
		phases=stage_config.phases,
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
	unit_ids_override: list[int] | None = None,
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
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=data_cfg)

	return ReconstructionInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		final_output_root=output_root,
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
		phases=stage_cfg.phases,
		use_full_channels_templates=stage_cfg.use_full_channels_templates,
		require_full_channels_templates=stage_cfg.require_full_channels_templates,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		n_jobs=1,
		axon_velocity_params=stage_cfg.axon_velocity_params,
		probe_geometry=probe_geometry,
	)
