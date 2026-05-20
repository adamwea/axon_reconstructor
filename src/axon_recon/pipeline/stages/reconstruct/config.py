from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any

from axon_recon.runtime_config import RuntimeConfig
from axon_recon.pipeline.shared.grid_sorting import normalize_grid_sort_by
from axon_recon.pipeline.shared.plotting import build_stage_plot_block
from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.reconstruct.templates.config import build_templates_inputs_for_target
from axon_recon.pipeline.stages.reconstruct.templates.config import parse_probe_geometry_from_data_config
from axon_recon.pipeline.stages.reconstruct.templates.config import parse_reconstruct_templates_config

from ...execution.context import ExecutionTarget
from ...resources import parse_resources_config, validate_phase_resource_class
from .models.inputs import (
	ReconstructionBranchColorsConfig,
	ReconstructionBranchPlotOutputConfig,
	ReconstructionBranchPropagationDisplayConfig,
	ReconstructionBranchVelocityDisplayConfig,
	ReconstructionAmplitudeMapOutputConfig,
	ReconstructionFullChipLayoutColorConfig,
	ReconstructionFullChipLayoutDisplayConfig,
	ReconstructionFullChipLayoutOutputConfig,
	ReconstructionAvReconsConfig,
	ReconstructionAxonVelocityPhaseConfig,
	CircleReconConfig,
	CircleReconDisplayConfig,
	CircleReconOutputConfig,
	PerUnitOutputsConfig,
	ReconstructionDiagnosticFigureConfig,
	ReconstructionGenerateGtrsOutputsConfig,
	ReconstructionGenerateGtrsPhaseConfig,
	ReconstructionInputs,
	ReconstructionPhasesConfig,
	ReconstructionPlotBranchPropagationsPhaseConfig,
	ReconstructionPlotBranchVelocitiesPhaseConfig,
	ReconstructionPlotReconsOutputsConfig,
	ReconstructionPlotUnitSummaryPhaseConfig,
	ReconstructionPlotReconsPhaseConfig,
	ReconstructionClearTemplatesCachePhaseConfig,
	ReconstructionKssynthPhaseConfig,
	ReconstructionReconGridDisplayConfig,
	ReconstructionReconGridOutputConfig,
	ReconstructionReconGridRenderConfig,
	ReconstructionReportFullChipLayoutPhaseConfig,
	ReconstructionReportMarkdownConfig,
	ReconstructionReportReconGridPhaseConfig,
	ReconstructionReportReconsPhaseConfig,
	ReconstructionReportSummariesPhaseConfig,
	ReconstructionSummaryPngConfig,
	ReconstructionUnitSummaryDisplayConfig,
	ReconstructionUnitSummaryOutputConfig,
)


LOGGER = logging.getLogger("axon_recon.reconstruct.config")


DEFAULT_RECONSTRUCTION_PHASE_SEQUENCE: tuple[str, ...] = (
	"axon_velocity_gtrs",
	"plot_recons",
	"plot_branch_propagations",
	"plot_branch_velocities",
	"plot_unit_summary",
	"report_recons",
	"report_recon_grid",
	"report_full_chip_layout",
	"report_summaries",
)


_RECONSTRUCTION_PHASE_ALIASES: dict[str, str] = {
	"resolve_sources": "templates_resolve_sources",
	"templates.resolve_sources": "templates_resolve_sources",
	"templates_resolve_sources": "templates_resolve_sources",
	"analyzers": "templates_analyzers",
	"templates.analyzers": "templates_analyzers",
	"templates_analyzers": "templates_analyzers",
	"extract_partial_templates": "templates_extract_partial_templates",
	"templates.extract_partial_templates": "templates_extract_partial_templates",
	"templates_extract_partial_templates": "templates_extract_partial_templates",
	"build_templates": "templates_build_templates",
	"templates.build_templates": "templates_build_templates",
	"templates_build_templates": "templates_build_templates",
	"compute_template_similarity": "templates_compute_template_similarity",
	"templates.compute_template_similarity": "templates_compute_template_similarity",
	"templates_compute_template_similarity": "templates_compute_template_similarity",
	"plot_templates_v2": "templates_plot_templates_v2",
	"templates.plot_templates_v2": "templates_plot_templates_v2",
	"templates_plot_templates_v2": "templates_plot_templates_v2",
	"template_plots_v2": "templates_plot_templates_v2",
	"report_templates": "templates_report_templates",
	"templates.report_templates": "templates_report_templates",
	"templates_report_templates": "templates_report_templates",
	"axon_velocity_gtrs": "axon_velocity_gtrs",
	"generate": "axon_velocity_gtrs",
	"gtrs": "axon_velocity_gtrs",
	"plot_recons": "plot_recons",
	"plot_reconstructions": "plot_recons",
	"plot_branch_propagations": "plot_branch_propagations",
	"plot_branch_velocities": "plot_branch_velocities",
	"plot_unit_summary": "plot_unit_summary",
	"report_recons": "report_recons",
	"report_reconstructions": "report_recons",
	"report_recon_grid": "report_recon_grid",
	"report_full_chip_layout": "report_full_chip_layout",
	"report_summaries": "report_summaries",
	"clear_cache": "clear_templates_cache",
	"clear_templates_cache": "clear_templates_cache",
}


def normalize_reconstruction_phase_name(raw: Any) -> str:
	token = str(raw or "").strip().replace("-", "_").replace(" ", "_")
	return _RECONSTRUCTION_PHASE_ALIASES.get(token, token)


def build_reconstruct_templates_runtime_config(runtime_config: RuntimeConfig) -> RuntimeConfig:
	return runtime_config


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


def _normalize_phase_sequence(raw: Any, default: tuple[str, ...]) -> tuple[str, ...]:
	if raw is None:
		return default
	if isinstance(raw, str):
		items = [part.strip() for part in raw.split(",")]
	elif isinstance(raw, (list, tuple)):
		items = [str(item).strip() for item in raw]
	else:
		return default
	sequence: list[str] = []
	for item in items:
		phase = normalize_reconstruction_phase_name(item)
		if not phase:
			continue
		if phase not in _RECONSTRUCTION_PHASE_ALIASES.values():
			raise ValueError(f"Unknown reconstruct phase in phase_sequence: {item!r}")
		if phase not in sequence:
			sequence.append(phase)
	return tuple(sequence) or default


def _as_optional_positive_int(raw: Any) -> int | None:
	if raw is None or str(raw).strip() == "":
		return None
	try:
		value = int(raw)
	except (TypeError, ValueError):
		return None
	return value if value > 0 else None


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
	display_raw = data.get("display", {})
	display = display_raw if isinstance(display_raw, dict) else {}
	return ReconstructionDiagnosticFigureConfig(
		write_png=_as_bool(data.get("write_png", False), False),
		write_svg=_as_bool(data.get("write_svg", False), False),
		relpath=str(data.get("relpath", default_relpath) or default_relpath).strip() or str(default_relpath),
		dpi=float(data.get("dpi", 300.0) or 300.0),
		invert_y_axis=_as_bool(display.get("invert_y_axis", data.get("invert_y_axis", True)), True),
	)


def _normalize_branch_scope(raw: Any, default: str = "raw") -> str:
	text = str(raw if raw is not None else default).strip().lower()
	if text not in {"raw", "clean"}:
		return str(default)
	return text


def _parse_figsize(raw: Any, default: tuple[float, float]) -> tuple[float, float]:
	if isinstance(raw, dict):
		try:
			width = float(raw.get("width", default[0]))
			height = float(raw.get("height", default[1]))
			return (max(1.0, width), max(1.0, height))
		except Exception:
			return default
	if isinstance(raw, (list, tuple)) and len(raw) >= 2:
		try:
			return (max(1.0, float(raw[0])), max(1.0, float(raw[1])))
		except Exception:
			return default
	if isinstance(raw, str) and "," in raw:
		parts = [part.strip() for part in raw.split(",", 1)]
		if len(parts) == 2:
			try:
				return (max(1.0, float(parts[0])), max(1.0, float(parts[1])))
			except Exception:
				return default
	return default


def _build_branch_plot_output_config(
	block: Any,
	*,
	default_relpath: str,
	default_manifest_relpath: str,
) -> ReconstructionBranchPlotOutputConfig:
	data = block if isinstance(block, dict) else {}
	relpath = str(data.get("relpath", default_relpath) or default_relpath).strip() or str(default_relpath)
	manifest_relpath = str(data.get("manifest_relpath", default_manifest_relpath) or default_manifest_relpath).strip()
	if not manifest_relpath:
		manifest_relpath = str(default_manifest_relpath)
	try:
		dpi = float(data.get("dpi", 300.0) or 300.0)
	except Exception:
		dpi = 300.0
	return ReconstructionBranchPlotOutputConfig(
		write_png=_as_bool(data.get("write_png", False), False),
		write_svg=_as_bool(data.get("write_svg", False), False),
		relpath=relpath,
		manifest_relpath=manifest_relpath,
		dpi=float(max(72.0, dpi)),
	)


def _build_branch_propagation_display_config(block: Any) -> ReconstructionBranchPropagationDisplayConfig:
	data = block if isinstance(block, dict) else {}
	default_figsize = ReconstructionBranchPropagationDisplayConfig().figsize
	total_width_raw = data.get("total_width", None)
	try:
		total_width = None if total_width_raw is None else float(total_width_raw)
	except Exception:
		total_width = None
	if total_width is not None and total_width <= 0.0:
		total_width = None
	return ReconstructionBranchPropagationDisplayConfig(
		figsize=_parse_figsize(data.get("figsize", default_figsize), default_figsize),
		total_width=total_width,
		sort_templates=_as_bool(data.get("sort_templates", False), False),
		show_title=_as_bool(data.get("show_title", True), True),
		invert_y_axis=_as_bool(data.get("invert_y_axis", True), True),
	)


def _build_branch_velocity_display_config(block: Any) -> ReconstructionBranchVelocityDisplayConfig:
	data = block if isinstance(block, dict) else {}
	default_cfg = ReconstructionBranchVelocityDisplayConfig()
	try:
		legend_fontsize = float(data.get("legend_fontsize", default_cfg.legend_fontsize) or default_cfg.legend_fontsize)
	except Exception:
		legend_fontsize = default_cfg.legend_fontsize
	try:
		title_fontsize = float(data.get("title_fontsize", default_cfg.title_fontsize) or default_cfg.title_fontsize)
	except Exception:
		title_fontsize = default_cfg.title_fontsize
	try:
		axis_label_fontsize = float(
			data.get("axis_label_fontsize", default_cfg.axis_label_fontsize) or default_cfg.axis_label_fontsize
		)
	except Exception:
		axis_label_fontsize = default_cfg.axis_label_fontsize
	try:
		tick_label_fontsize = float(
			data.get("tick_label_fontsize", default_cfg.tick_label_fontsize) or default_cfg.tick_label_fontsize
		)
	except Exception:
		tick_label_fontsize = default_cfg.tick_label_fontsize
	return ReconstructionBranchVelocityDisplayConfig(
		figsize=_parse_figsize(data.get("figsize", default_cfg.figsize), default_cfg.figsize),
		show_title=_as_bool(data.get("show_title", default_cfg.show_title), default_cfg.show_title),
		title_fontsize=float(max(1.0, title_fontsize)),
		axis_label_fontsize=float(max(1.0, axis_label_fontsize)),
		tick_label_fontsize=float(max(1.0, tick_label_fontsize)),
		units_only_axis_labels=_as_bool(
			data.get("units_only_axis_labels", default_cfg.units_only_axis_labels),
			default_cfg.units_only_axis_labels,
		),
		show_legend=_as_bool(data.get("show_legend", default_cfg.show_legend), default_cfg.show_legend),
		legend_fontsize=float(max(1.0, legend_fontsize)),
	)


def _build_unit_summary_display_config(block: Any) -> ReconstructionUnitSummaryDisplayConfig:
	data = block if isinstance(block, dict) else {}
	default_cfg = ReconstructionUnitSummaryDisplayConfig()

	def _optional_bool(*, key: str, default: bool | None) -> bool | None:
		if key not in data:
			return default
		return _as_bool(data.get(key), False)

	def _optional_positive_float(raw: Any, default: float | None) -> float | None:
		if raw is None:
			return default
		try:
			parsed = float(raw)
		except Exception:
			return default
		if parsed <= 0.0:
			return default
		return parsed

	def _optional_float(raw: Any, default: float) -> float:
		if raw is None:
			return default
		try:
			return float(raw)
		except Exception:
			return default

	return ReconstructionUnitSummaryDisplayConfig(
		show_title=_as_bool(data.get("show_title", default_cfg.show_title), default_cfg.show_title),
		show_summary_unit_label=_as_bool(
			data.get("show_summary_unit_label", default_cfg.show_summary_unit_label),
			default_cfg.show_summary_unit_label,
		),
		summary_unit_label_fontsize=float(
			max(
				1.0,
				float(
					_optional_positive_float(
						data.get("summary_unit_label_fontsize", default_cfg.summary_unit_label_fontsize),
						default_cfg.summary_unit_label_fontsize,
					)
					or default_cfg.summary_unit_label_fontsize
				)
			)
		),
		summary_unit_label_x_frac=_optional_float(
			data.get("summary_unit_label_x_frac", default_cfg.summary_unit_label_x_frac),
			default_cfg.summary_unit_label_x_frac,
		),
		summary_unit_label_y_frac=_optional_float(
			data.get("summary_unit_label_y_frac", default_cfg.summary_unit_label_y_frac),
			default_cfg.summary_unit_label_y_frac,
		),
		recon_show_unit_label=_optional_bool(
			key="recon_show_unit_label",
			default=default_cfg.recon_show_unit_label,
		),
		recon_show_branch_legend=_optional_bool(
			key="recon_show_branch_legend",
			default=default_cfg.recon_show_branch_legend,
		),
		velocity_show_title=_optional_bool(
			key="velocity_show_title",
			default=default_cfg.velocity_show_title,
		),
		show_velocity_legend=_optional_bool(
			key="show_velocity_legend",
			default=default_cfg.show_velocity_legend,
		),
		reserve_velocity_legend_space=_optional_bool(
			key="reserve_velocity_legend_space",
			default=default_cfg.reserve_velocity_legend_space,
		),
		velocity_legend_width=float(
			max(
				0.5,
				float(
					_optional_positive_float(
						data.get("velocity_legend_width", default_cfg.velocity_legend_width),
						default_cfg.velocity_legend_width,
					)
					or default_cfg.velocity_legend_width
				)
			)
		),
		top_row_panel_gap_width=_optional_positive_float(
			data.get("top_row_panel_gap_width", default_cfg.top_row_panel_gap_width),
			default_cfg.top_row_panel_gap_width,
		),
		top_row_height=_optional_positive_float(data.get("top_row_height", default_cfg.top_row_height), default_cfg.top_row_height),
		propagation_row_height=_optional_positive_float(
			data.get("propagation_row_height", default_cfg.propagation_row_height),
			default_cfg.propagation_row_height,
		),
		circle_panel_width=_optional_positive_float(
			data.get("circle_panel_width", default_cfg.circle_panel_width),
			default_cfg.circle_panel_width,
		),
		velocity_panel_width=_optional_positive_float(
			data.get("velocity_panel_width", default_cfg.velocity_panel_width),
			default_cfg.velocity_panel_width,
		),
		propagation_panel_width=_optional_positive_float(
			data.get("propagation_panel_width", default_cfg.propagation_panel_width),
			default_cfg.propagation_panel_width,
		),
		recon_x_offset_frac=_optional_float(
			data.get("recon_x_offset_frac", default_cfg.recon_x_offset_frac),
			default_cfg.recon_x_offset_frac,
		),
		recon_y_offset_frac=_optional_float(
			data.get("recon_y_offset_frac", default_cfg.recon_y_offset_frac),
			default_cfg.recon_y_offset_frac,
		),
		velocity_x_offset_frac=_optional_float(
			data.get("velocity_x_offset_frac", default_cfg.velocity_x_offset_frac),
			default_cfg.velocity_x_offset_frac,
		),
		velocity_y_offset_frac=_optional_float(
			data.get("velocity_y_offset_frac", default_cfg.velocity_y_offset_frac),
			default_cfg.velocity_y_offset_frac,
		),
		propagation_x_offset_frac=_optional_float(
			data.get("propagation_x_offset_frac", default_cfg.propagation_x_offset_frac),
			default_cfg.propagation_x_offset_frac,
		),
		propagation_y_offset_frac=_optional_float(
			data.get("propagation_y_offset_frac", default_cfg.propagation_y_offset_frac),
			default_cfg.propagation_y_offset_frac,
		),
	)


def _build_unit_summary_output_config(
	block: Any,
	*,
	default_relpath: str,
) -> ReconstructionUnitSummaryOutputConfig:
	data = block if isinstance(block, dict) else {}
	relpath = str(data.get("relpath", default_relpath) or default_relpath).strip() or str(default_relpath)
	try:
		dpi = float(data.get("dpi", 300.0) or 300.0)
	except Exception:
		dpi = 300.0
	return ReconstructionUnitSummaryOutputConfig(
		write_png=_as_bool(data.get("write_png", True), True),
		write_svg=_as_bool(data.get("write_svg", False), False),
		relpath=relpath,
		dpi=float(max(72.0, dpi)),
	)


def _normalize_color_strategy(raw: Any, default: str = "distinct_hsv") -> str:
	text = str(raw if raw is not None else default).strip().lower().replace("-", "_").replace(" ", "_")
	if text in {"colormap", "cmap", "sampled_colormap", "sample_colormap"}:
		return "colormap"
	if text in {"distinct_hsv", "hsv", "golden_hsv"}:
		return "distinct_hsv"
	return str(default)


def _build_full_chip_layout_output_config(
	block: Any,
	*,
	default_relpath: str,
	default_manifest_relpath: str,
) -> ReconstructionFullChipLayoutOutputConfig:
	data = block if isinstance(block, dict) else {}
	relpath = str(data.get("relpath", default_relpath) or default_relpath).strip() or str(default_relpath)
	manifest_relpath = str(data.get("manifest_relpath", default_manifest_relpath) or default_manifest_relpath).strip()
	if not manifest_relpath:
		manifest_relpath = str(default_manifest_relpath)
	try:
		dpi = float(data.get("dpi", 300.0) or 300.0)
	except Exception:
		dpi = 300.0
	return ReconstructionFullChipLayoutOutputConfig(
		write_png=_as_bool(data.get("write_png", True), True),
		write_svg=_as_bool(data.get("write_svg", False), False),
		relpath=relpath,
		manifest_relpath=manifest_relpath,
		dpi=float(max(72.0, dpi)),
	)


def _build_full_chip_layout_color_config(block: Any) -> ReconstructionFullChipLayoutColorConfig:
	data = block if isinstance(block, dict) else {}
	default_cfg = ReconstructionFullChipLayoutColorConfig()
	return ReconstructionFullChipLayoutColorConfig(
		strategy=_normalize_color_strategy(data.get("strategy", default_cfg.strategy), default=default_cfg.strategy),
		color_scheme=str(data.get("color_scheme", default_cfg.color_scheme) or default_cfg.color_scheme),
	)


def _build_full_chip_layout_display_config(block: Any) -> ReconstructionFullChipLayoutDisplayConfig:
	data = block if isinstance(block, dict) else {}
	default_cfg = ReconstructionFullChipLayoutDisplayConfig()
	try:
		alpha = float(data.get("alpha", default_cfg.alpha) or default_cfg.alpha)
	except Exception:
		alpha = default_cfg.alpha
	try:
		linewidth = float(data.get("linewidth", default_cfg.linewidth) or default_cfg.linewidth)
	except Exception:
		linewidth = default_cfg.linewidth
	try:
		legend_fontsize = float(data.get("legend_fontsize", default_cfg.legend_fontsize) or default_cfg.legend_fontsize)
	except Exception:
		legend_fontsize = default_cfg.legend_fontsize
	try:
		legend_ncols = int(data.get("legend_ncols", default_cfg.legend_ncols) or default_cfg.legend_ncols)
	except Exception:
		legend_ncols = default_cfg.legend_ncols
	try:
		chip_outline_linewidth = float(
			data.get("chip_outline_linewidth", default_cfg.chip_outline_linewidth) or default_cfg.chip_outline_linewidth
		)
	except Exception:
		chip_outline_linewidth = default_cfg.chip_outline_linewidth
	title = str(data.get("title", default_cfg.title) or default_cfg.title)
	chip_outline_color = str(data.get("chip_outline_color", default_cfg.chip_outline_color) or default_cfg.chip_outline_color)
	background_color = str(data.get("background_color", default_cfg.background_color) or default_cfg.background_color)
	return ReconstructionFullChipLayoutDisplayConfig(
		figsize=_parse_figsize(data.get("figsize", default_cfg.figsize), default_cfg.figsize),
		show_title=_as_bool(data.get("show_title", default_cfg.show_title), default_cfg.show_title),
		title=title,
		invert_y_axis=_as_bool(data.get("invert_y_axis", default_cfg.invert_y_axis), default_cfg.invert_y_axis),
		alpha=float(min(1.0, max(0.0, alpha))),
		linewidth=float(max(0.1, linewidth)),
		show_legend=_as_bool(data.get("show_legend", default_cfg.show_legend), default_cfg.show_legend),
		legend_fontsize=float(max(1.0, legend_fontsize)),
		legend_ncols=max(1, int(legend_ncols)),
		draw_chip_outline=_as_bool(data.get("draw_chip_outline", default_cfg.draw_chip_outline), default_cfg.draw_chip_outline),
		chip_outline_color=(chip_outline_color if chip_outline_color.strip() else default_cfg.chip_outline_color),
		chip_outline_linewidth=float(max(0.1, chip_outline_linewidth)),
		background_color=(background_color if background_color.strip() else default_cfg.background_color),
	)


def _get_reconstruct_amplitude_map_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
			"stages.reconstruct.outputs.amplitude_map",
		),
	)
	per_unit_block = runtime_config.get("stages.reconstruct.outputs.per_unit_outputs.amplitude_map", {})
	if isinstance(per_unit_block, dict) and per_unit_block:
		return _deep_merge_dict(stage_block, dict(per_unit_block))
	return stage_block


@dataclass(frozen=True)
class ReconstructionStageConfig:
	output_rel_root: str
	unit_reldir: str
	report_sort_by: str
	overwrite_report_outputs_on_unit_rerun: bool
	phase_sequence: tuple[str, ...]
	debug_prints: bool
	debug_mode_enabled: bool
	debug_limit_datasets: int | None
	debug_limit_wells: int | None
	debug_limit_wells_per_dataset: int | None
	branch_colors: ReconstructionBranchColorsConfig
	cleanup_failed_unit_outputs: bool
	failed_units_summary_relpath: str
	per_unit_outputs: PerUnitOutputsConfig
	unit_ids: list[int] | None
	unit_limit: int | None
	limit_segments: int | None
	phases: ReconstructionPhasesConfig
	use_full_channels_templates: bool
	require_full_channels_templates: bool
	force_restart: bool
	replot: bool
	axon_velocity_params: dict[str, Any]


def parse_reconstruction_stage_config(
	*,
	runtime_config: RuntimeConfig,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
) -> ReconstructionStageConfig:
	stage_cfg = runtime_config.get("stages.reconstruct", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	resources_config = parse_resources_config(runtime_config=runtime_config, logger=LOGGER)

	def _phase_resource_class(raw_cfg: dict[str, Any] | None, phase_name: str) -> str | None:
		phase_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
		return validate_phase_resource_class(
			resource_class=phase_cfg.get("resource_class", None),
			resources=resources_config,
			phase_name=f"reconstruct.{phase_name}",
		)

	templates_runtime_config = build_reconstruct_templates_runtime_config(runtime_config)
	try:
		template_defaults_cfg = parse_reconstruct_templates_config(runtime_config=templates_runtime_config)
		tpl_circles_defaults = template_defaults_cfg.per_unit_outputs.template_circles
		tpl_footprint_plots_defaults = getattr(template_defaults_cfg.per_unit_outputs, "footprint_plots", None)
		tpl_footprint_amplitude_defaults = getattr(tpl_footprint_plots_defaults, "amplitude_map", None)
		tpl_footprint_latency_defaults = getattr(tpl_footprint_plots_defaults, "latency_map", None)
		tpl_plot_templates_v2_defaults = getattr(template_defaults_cfg.phases, "plot_templates_v2", None)
	except Exception:
		tpl_circles_defaults = None
		tpl_footprint_amplitude_defaults = None
		tpl_footprint_latency_defaults = None
		tpl_plot_templates_v2_defaults = None
	default_circle_unique_color = bool(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "unique_color_per_branch", True)
	)
	default_circle_color_scheme = str(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "color_scheme", "tab20") or "tab20"
	)
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	debug_mode_cfg = stage_cfg.get("debug_mode", {}) if isinstance(stage_cfg.get("debug_mode", {}), dict) else {}
	debug_prints = _as_bool(stage_cfg.get("debug_prints", stage_cfg.get("debug_plotting_prints", False)), False)
	inputs_cfg = stage_cfg.get("inputs", {}) if isinstance(stage_cfg.get("inputs", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}
	stage_resources_cfg = stage_cfg.get("resources", {}) if isinstance(stage_cfg.get("resources", {}), dict) else {}
	branch_colors_cfg = stage_cfg.get("branch_colors", {}) if isinstance(stage_cfg.get("branch_colors", {}), dict) else {}
	phases_cfg = stage_cfg.get("phases", {}) if isinstance(stage_cfg.get("phases", {}), dict) else {}
	phase_sequence = _normalize_phase_sequence(stage_cfg.get("phase_sequence", None), DEFAULT_RECONSTRUCTION_PHASE_SEQUENCE)
	debug_mode_enabled = _as_bool(debug_mode_cfg.get("enabled", False), False)
	debug_limit_datasets = _as_optional_positive_int(debug_mode_cfg.get("limit_datasets", None))
	debug_limit_wells = _as_optional_positive_int(debug_mode_cfg.get("limit_wells", None))
	debug_limit_wells_per_dataset = _as_optional_positive_int(debug_mode_cfg.get("limit_wells_per_dataset", None))
	axon_velocity_gtrs_cfg = phases_cfg.get("axon_velocity_gtrs", {}) if isinstance(phases_cfg.get("axon_velocity_gtrs", {}), dict) else {}
	plot_recons_cfg = phases_cfg.get("plot_recons", {}) if isinstance(phases_cfg.get("plot_recons", {}), dict) else {}
	plot_branch_propagations_cfg = (
		phases_cfg.get("plot_branch_propagations", {})
		if isinstance(phases_cfg.get("plot_branch_propagations", {}), dict)
		else {}
	)
	plot_branch_velocities_cfg = (
		phases_cfg.get("plot_branch_velocities", {})
		if isinstance(phases_cfg.get("plot_branch_velocities", {}), dict)
		else {}
	)
	plot_unit_summary_cfg = (
		phases_cfg.get("plot_unit_summary", {})
		if isinstance(phases_cfg.get("plot_unit_summary", {}), dict)
		else {}
	)
	report_recons_cfg = phases_cfg.get("report_recons", {}) if isinstance(phases_cfg.get("report_recons", {}), dict) else {}
	report_recon_grid_cfg = (
		phases_cfg.get("report_recon_grid", {})
		if isinstance(phases_cfg.get("report_recon_grid", {}), dict)
		else {}
	)
	report_summaries_cfg = (
		phases_cfg.get("report_summaries", {})
		if isinstance(phases_cfg.get("report_summaries", {}), dict)
		else {}
	)
	clear_templates_cache_cfg = (
		phases_cfg.get("clear_templates_cache", {})
		if isinstance(phases_cfg.get("clear_templates_cache", {}), dict)
		else {}
	)
	kssynth_cfg = (
		phases_cfg.get("kssynth", {})
		if isinstance(phases_cfg.get("kssynth", {}), dict)
		else {}
	)
	report_full_chip_layout_cfg = (
		phases_cfg.get("report_full_chip_layout", {})
		if isinstance(phases_cfg.get("report_full_chip_layout", {}), dict)
		else {}
	)
	axon_velocity_gtrs_resources_cfg = (
		axon_velocity_gtrs_cfg.get("resources", {}) if isinstance(axon_velocity_gtrs_cfg.get("resources", {}), dict) else {}
	)
	phase_axon_velocity_outputs_cfg = (
		axon_velocity_gtrs_cfg.get("outputs", {}) if isinstance(axon_velocity_gtrs_cfg.get("outputs", {}), dict) else {}
	)
	phase_generate_diagnostic_figs_cfg = (
		phase_axon_velocity_outputs_cfg.get("diagnostic_figs", {})
		if isinstance(phase_axon_velocity_outputs_cfg.get("diagnostic_figs", {}), dict)
		else {}
	)
	phase_plot_outputs_cfg = (
		plot_recons_cfg.get("outputs", {}) if isinstance(plot_recons_cfg.get("outputs", {}), dict) else {}
	)
	plot_branch_propagations_output_cfg = (
		plot_branch_propagations_cfg.get("output", {})
		if isinstance(plot_branch_propagations_cfg.get("output", {}), dict)
		else {}
	)
	plot_branch_velocities_output_cfg = (
		plot_branch_velocities_cfg.get("output", {})
		if isinstance(plot_branch_velocities_cfg.get("output", {}), dict)
		else {}
	)
	plot_branch_propagations_display_cfg = (
		plot_branch_propagations_cfg.get("display", {})
		if isinstance(plot_branch_propagations_cfg.get("display", {}), dict)
		else {}
	)
	plot_branch_velocities_display_cfg = (
		plot_branch_velocities_cfg.get("display", {})
		if isinstance(plot_branch_velocities_cfg.get("display", {}), dict)
		else {}
	)
	plot_unit_summary_output_cfg = (
		plot_unit_summary_cfg.get("output", {})
		if isinstance(plot_unit_summary_cfg.get("output", {}), dict)
		else {}
	)
	plot_unit_summary_display_cfg = (
		plot_unit_summary_cfg.get("display", {})
		if isinstance(plot_unit_summary_cfg.get("display", {}), dict)
		else {}
	)
	report_recon_grid_output_cfg = (
		report_recon_grid_cfg.get("output", {})
		if isinstance(report_recon_grid_cfg.get("output", {}), dict)
		else {}
	)
	report_recon_grid_display_cfg = (
		report_recon_grid_cfg.get("display", {})
		if isinstance(report_recon_grid_cfg.get("display", {}), dict)
		else {}
	)
	report_recon_grid_render_cfg = (
		report_recon_grid_cfg.get("render", {})
		if isinstance(report_recon_grid_cfg.get("render", {}), dict)
		else {}
	)
	report_full_chip_layout_output_cfg = (
		report_full_chip_layout_cfg.get("output", {})
		if isinstance(report_full_chip_layout_cfg.get("output", {}), dict)
		else {}
	)
	report_full_chip_layout_display_cfg = (
		report_full_chip_layout_cfg.get("display", {})
		if isinstance(report_full_chip_layout_cfg.get("display", {}), dict)
		else {}
	)
	report_full_chip_layout_colors_cfg = (
		report_full_chip_layout_cfg.get("unit_colors", {})
		if isinstance(report_full_chip_layout_cfg.get("unit_colors", {}), dict)
		else {}
	)
	reports_cfg = outputs_cfg.get("reports", {}) if isinstance(outputs_cfg.get("reports", {}), dict) else {}
	grids_cfg = reports_cfg.get("grids", {}) if isinstance(reports_cfg.get("grids", {}), dict) else {}
	legacy_circle_recon_grid_cfg = (
		grids_cfg.get("circle_recon_grid", {}) if isinstance(grids_cfg.get("circle_recon_grid", {}), dict) else {}
	)
	per_unit_cfg = outputs_cfg.get("per_unit_outputs", {}) if isinstance(outputs_cfg.get("per_unit_outputs", {}), dict) else {}
	report_sort_by = normalize_grid_sort_by(
		stage_cfg.get("report_sort_by", grids_cfg.get("sort_by", reports_cfg.get("sort_by", "unit_id"))),
		default="unit_id",
	)
	overwrite_report_outputs_on_unit_rerun = _as_bool(
		stage_cfg.get(
			"overwrite_report_outputs_on_unit_rerun",
			reports_cfg.get("overwrite_on_unit_rerun", False),
		),
		False,
	)
	unit_reldir = str(
		stage_cfg.get(
			"unit_reldir",
			phase_plot_outputs_cfg.get("unit_reldir", per_unit_cfg.get("unit_reldir", "units/{unit_id:04d}/")),
		)
	)
	if not unit_reldir.strip():
		unit_reldir = "units/{unit_id:04d}/"
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
	canonical_av_cfg = stage_cfg.get("axon_velocity", {})
	av_cfg: dict[str, Any] = dict(canonical_av_cfg) if isinstance(canonical_av_cfg, dict) else {}
	phase_axon_velocity_cfg = (
		axon_velocity_gtrs_cfg.get("axon_velocity", {})
		if isinstance(axon_velocity_gtrs_cfg.get("axon_velocity", {}), dict)
		else {}
	)
	phase_axon_velocity_params = (
		phase_axon_velocity_cfg.get("params", {})
		if isinstance(phase_axon_velocity_cfg.get("params", {}), dict)
		else {}
	)
	if phase_axon_velocity_params:
		av_cfg.update(dict(phase_axon_velocity_params))
	axon_velocity_gtrs_unit_procs = _parse_optional_positive_int(
		axon_velocity_gtrs_resources_cfg.get("unit_procs", axon_velocity_gtrs_cfg.get("unit_procs", None))
	)
	axon_velocity_gtrs_unit_batch_size = _parse_optional_positive_int(
		axon_velocity_gtrs_resources_cfg.get("unit_batch_size", axon_velocity_gtrs_cfg.get("unit_batch_size", None))
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
		phase_channel_selection_fig_cfg,
		default_relpath="diagnostic_figs/channel_selection",
	)
	axon_reconstruction_figure_cfg = _build_reconstruct_figure_output_config(
		phase_axon_reconstruction_fig_cfg,
		default_relpath="diagnostic_figs/axon_reconstruction",
	)
	axon_velocity_gtrs_outputs = ReconstructionGenerateGtrsOutputsConfig(
		write_branches_raw_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get("write_branches_raw_json", per_unit_cfg.get("write_branches_raw_json", True)),
			True,
		),
		branches_raw_relpath=str(
			phase_axon_velocity_outputs_cfg.get("branches_raw_relpath", per_unit_cfg.get("branches_raw_relpath", "branches_raw.json"))
		),
		write_branches_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get("write_branches_json", per_unit_cfg.get("write_branches_json", True)),
			True,
		),
		branches_relpath=str(
			phase_axon_velocity_outputs_cfg.get("branches_relpath", per_unit_cfg.get("branches_relpath", "branches.json"))
		),
		write_detection_filter_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get(
				"write_detection_filter_json",
				per_unit_cfg.get("write_detection_filter_json", False),
			),
			False,
		),
		detection_filter_relpath=str(
			phase_axon_velocity_outputs_cfg.get(
				"detection_filter_relpath",
				per_unit_cfg.get("detection_filter_relpath", "detection_filter.json"),
			)
		),
		write_kurtosis_filter_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get(
				"write_kurtosis_filter_json",
				per_unit_cfg.get("write_kurtosis_filter_json", False),
			),
			False,
		),
		kurtosis_filter_relpath=str(
			phase_axon_velocity_outputs_cfg.get(
				"kurtosis_filter_relpath",
				per_unit_cfg.get("kurtosis_filter_relpath", "kurtosis_filter.json"),
			)
		),
		write_peak_std_filter_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get(
				"write_peak_std_filter_json",
				per_unit_cfg.get("write_peak_std_filter_json", False),
			),
			False,
		),
		peak_std_filter_relpath=str(
			phase_axon_velocity_outputs_cfg.get(
				"peak_std_filter_relpath",
				per_unit_cfg.get("peak_std_filter_relpath", "peak_std_filter.json"),
			)
		),
		write_delay_filter_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get(
				"write_delay_filter_json",
				per_unit_cfg.get("write_delay_filter_json", False),
			),
			False,
		),
		delay_filter_relpath=str(
			phase_axon_velocity_outputs_cfg.get(
				"delay_filter_relpath",
				per_unit_cfg.get("delay_filter_relpath", "delay_filter.json"),
			)
		),
		write_all_filters_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get(
				"write_all_filters_json",
				per_unit_cfg.get("write_all_filters_json", False),
			),
			False,
		),
		all_filters_relpath=str(
			phase_axon_velocity_outputs_cfg.get(
				"all_filters_relpath",
				per_unit_cfg.get("all_filters_relpath", "all_filters.json"),
			)
		),
		write_heuristics_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get("write_heuristics_json", per_unit_cfg.get("write_heuristics_json", True)),
			True,
		),
		heuristics_relpath=str(
			phase_axon_velocity_outputs_cfg.get("heuristics_relpath", per_unit_cfg.get("heuristics_relpath", "heuristics.json"))
		),
		write_gtr_pkl=_as_bool(
			phase_axon_velocity_outputs_cfg.get("write_gtr_pkl", per_unit_cfg.get("write_gtr_pkl", True)),
			True,
		),
		gtr_pkl_relpath=str(
			phase_axon_velocity_outputs_cfg.get("gtr_pkl_relpath", per_unit_cfg.get("gtr_pkl_relpath", "gtr.pkl"))
		),
		template_source=_normalize_reconstruct_template_source(
			phase_axon_velocity_outputs_cfg.get("template_source", per_unit_cfg.get("template_source", "square")),
			default="square",
		),
		write_gtr_json=_as_bool(
			phase_axon_velocity_outputs_cfg.get("write_gtr_json", per_unit_cfg.get("write_gtr_json", False)),
			False,
		),
		gtr_json_relpath=str(
			phase_axon_velocity_outputs_cfg.get("gtr_json_relpath", per_unit_cfg.get("gtr_json_relpath", "gtr.json"))
		),
		channel_selection_figure=channel_selection_figure_cfg,
		axon_reconstruction_figure=axon_reconstruction_figure_cfg,
	)

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	replot = _as_bool(execution_cfg.get("replot", False), False)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if replot_override is not None:
		replot = bool(replot_override)

	unit_limit_raw = (
		debug_mode_cfg.get("unit_limit", None)
		if "unit_limit" in debug_mode_cfg
		else execution_cfg.get("unit_limit", stage_cfg.get("unit_limit", None))
	)
	unit_limit = _as_optional_positive_int(unit_limit_raw)
	if unit_limit_override is not None:
		unit_limit = _as_optional_positive_int(unit_limit_override)
	limit_segments_raw = (
		debug_mode_cfg.get("limit_segments", None)
		if "limit_segments" in debug_mode_cfg
		else execution_cfg.get("limit_segments", stage_cfg.get("limit_segments", None))
	)
	limit_segments = _as_optional_positive_int(limit_segments_raw)
	if limit_segments_override is not None:
		limit_segments = _as_optional_positive_int(limit_segments_override)

	runtime_unit_ids = _normalize_unit_ids(execution_cfg.get("unit_ids", stage_cfg.get("unit_ids", None)))
	if unit_ids_override is not None:
		unit_ids = _normalize_unit_ids(unit_ids_override)
	elif unit_id_override is not None:
		unit_ids = [int(unit_id_override)]
	else:
		unit_ids = runtime_unit_ids

	output_rel_root = str(stage_cfg.get("output_rel_root", outputs_cfg.get("output_rel_root", "recon_outputs"))).strip()
	if not output_rel_root:
		output_rel_root = "recon_outputs"
	cleanup_failed_unit_outputs = _as_bool(
		stage_cfg.get("cleanup_failed_unit_outputs", outputs_cfg.get("cleanup_failed_unit_outputs", False)),
		False,
	)
	failed_units_summary_relpath = str(
		stage_cfg.get(
			"failed_units_summary_relpath",
			outputs_cfg.get("failed_units_summary_relpath", "failed_units_summary.json"),
		)
	)
	if not failed_units_summary_relpath.strip():
		failed_units_summary_relpath = "failed_units_summary.json"

	report_recons_summary_png_cfg = (
		report_recons_cfg.get("summary_png", {})
		if isinstance(report_recons_cfg.get("summary_png", {}), dict)
		else {}
	)
	report_recons_report_md_cfg = (
		report_recons_cfg.get("report_md", {})
		if isinstance(report_recons_cfg.get("report_md", {}), dict)
		else {}
	)
	write_summary_png = _as_bool(
		report_recons_summary_png_cfg.get("write", outputs_cfg.get("write_summary", False)),
		False,
	)
	summary_png_relpath = _normalize_png_relpath(
		report_recons_summary_png_cfg.get("relpath", outputs_cfg.get("summary_relpath", "summary.png")),
		"summary.png",
	)
	try:
		summary_grid_ncols = max(
			1,
			int(report_recons_summary_png_cfg.get("grid_ncols", outputs_cfg.get("summary_grid_ncols", 5))),
		)
	except Exception:
		summary_grid_ncols = 5
	write_report_md = _as_bool(
		report_recons_report_md_cfg.get("write", outputs_cfg.get("write_report_md", False)),
		False,
	)
	report_md_relpath = str(
		report_recons_report_md_cfg.get("relpath", outputs_cfg.get("report_md_relpath", "report.md"))
	)
	if not report_md_relpath.strip():
		report_md_relpath = "report.md"

	if "write_png" in phase_amplitude_map_cfg:
		amplitude_map_write_png = _as_bool(phase_amplitude_map_cfg.get("write_png", False), False)
	elif "write_amplitude_map_png" in per_unit_cfg:
		amplitude_map_write_png = _as_bool(per_unit_cfg.get("write_amplitude_map_png", False), False)
	else:
		amplitude_map_write_png = _as_bool(amplitude_map_cfg.get("write_png", False), False)

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

	phase_circle_recon_cfg = (
		phase_plot_outputs_cfg.get("circle_recon", {})
		if isinstance(phase_plot_outputs_cfg.get("circle_recon", {}), dict)
		else {}
	)
	circle_recon_cfg = phase_circle_recon_cfg
	circle_display_cfg = (
		circle_recon_cfg.get("display", {}) if isinstance(circle_recon_cfg.get("display", {}), dict) else {}
	)
	circle_output_cfg = (
		circle_recon_cfg.get("output", {}) if isinstance(circle_recon_cfg.get("output", {}), dict) else {}
	)

	default_circle_base = "template_circles"
	default_circle_force_center = bool(getattr(tpl_circles_defaults, "force_center_soma", True))
	default_circle_show_labels = bool(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "show_branch_labels", False)
	)
	default_circle_show_legend = bool(
		getattr(getattr(tpl_circles_defaults, "branch_morphology", None), "show_branch_legend", False)
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
	branch_colors = ReconstructionBranchColorsConfig(
		unique_color_per_branch=_as_bool(
			branch_colors_cfg.get("unique_color_per_branch", default_circle_unique_color),
			default_circle_unique_color,
		),
		color_scheme=str(branch_colors_cfg.get("color_scheme", default_circle_color_scheme) or default_circle_color_scheme),
	)

	circle_base = str(circle_display_cfg.get("base", default_circle_base) or default_circle_base).strip().lower()
	if circle_base not in {"template_circles", "amplitude_map", "latency_map"}:
		circle_base = default_circle_base
	circle_channel_scope = str(circle_display_cfg.get("channel_scope", "nodes_and_branches") or "nodes_and_branches").strip().lower()
	if circle_channel_scope in {"filtered", "filtered_channels", "selected", "selected_channel"}:
		circle_channel_scope = "selected_channels"
	if circle_channel_scope not in {"nodes_and_branches", "branches_only", "nodes_only", "selected_channels"}:
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

	circle_zoom_to_branches = _as_bool(circle_display_cfg.get("zoom_to_branches", False), False)
	circle_recon = CircleReconConfig(
		display=CircleReconDisplayConfig(
			base=circle_base,
			channel_scope=circle_channel_scope,
			zoom_padding_percent=circle_zoom_padding_percent,
			zoom_to_branches=circle_zoom_to_branches,
			invert_y_axis=_as_bool(
				circle_display_cfg.get("invert_y_axis", getattr(tpl_circles_defaults, "invert_y_axis", True)),
				getattr(tpl_circles_defaults, "invert_y_axis", True),
			),
			force_center_soma=_as_bool(circle_display_cfg.get("force_center_soma", default_circle_force_center), default_circle_force_center),
			branch_scope=circle_branch_scope,
			unique_color_per_branch=_as_bool(
				circle_display_cfg.get("unique_color_per_branch", branch_colors.unique_color_per_branch),
				branch_colors.unique_color_per_branch,
			),
			show_branch_labels=_as_bool(circle_display_cfg.get("show_branch_labels", default_circle_show_labels), default_circle_show_labels),
				show_branch_legend=_as_bool(circle_display_cfg.get("show_branch_legend", default_circle_show_legend), default_circle_show_legend),
			color_scheme=str(circle_display_cfg.get("color_scheme", branch_colors.color_scheme) or branch_colors.color_scheme),
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
		base_template_circles_v2=tpl_plot_templates_v2_defaults,
		base_footprint_amplitude=tpl_footprint_amplitude_defaults,
		base_footprint_latency=tpl_footprint_latency_defaults,
	)
	plot_recons_outputs = ReconstructionPlotReconsOutputsConfig(
		amplitude_map=ReconstructionAmplitudeMapOutputConfig(
			write_png=amplitude_map_write_png,
			png_relpath=amplitude_map_png_relpath,
			heatmap=SharedHeatmapConfig.from_block(amplitude_map_cfg),
		),
		circle_recon=circle_recon,
	)
	report_av_recons_cfg = (
		report_recons_cfg.get("av_recons", {}) if isinstance(report_recons_cfg.get("av_recons", {}), dict) else {}
	)
	legacy_report_recon_grid_output_cfg = (
		legacy_circle_recon_grid_cfg.get("output", {})
		if isinstance(legacy_circle_recon_grid_cfg.get("output", {}), dict)
		else {}
	)
	legacy_report_recon_grid_display_cfg = (
		legacy_circle_recon_grid_cfg.get("display", {})
		if isinstance(legacy_circle_recon_grid_cfg.get("display", {}), dict)
		else {}
	)
	legacy_report_recon_grid_render_cfg = (
		legacy_circle_recon_grid_cfg.get("render", {})
		if isinstance(legacy_circle_recon_grid_cfg.get("render", {}), dict)
		else {}
	)
	report_recon_grid_output_source_cfg = (
		report_recon_grid_output_cfg if report_recon_grid_output_cfg else legacy_report_recon_grid_output_cfg
	)
	report_recon_grid_display_source_cfg = (
		report_recon_grid_display_cfg if report_recon_grid_display_cfg else legacy_report_recon_grid_display_cfg
	)
	report_recon_grid_render_source_cfg = (
		report_recon_grid_render_cfg if report_recon_grid_render_cfg else legacy_report_recon_grid_render_cfg
	)
	try:
		report_recon_grid_dpi = float(report_recon_grid_render_source_cfg.get("dpi", 300.0) or 300.0)
	except Exception:
		report_recon_grid_dpi = 300.0
	phases = ReconstructionPhasesConfig(
		clear_templates_cache=ReconstructionClearTemplatesCachePhaseConfig(
			enabled=_phase_enabled(clear_templates_cache_cfg, False),
			summary_json_relpath=str(
				clear_templates_cache_cfg.get(
					"summary_json_relpath",
					"reports/clear_templates_cache_summary.json",
				)
				or "reports/clear_templates_cache_summary.json"
			),
			resource_class=_phase_resource_class(clear_templates_cache_cfg, "clear_templates_cache"),
			keep_merged_per_unit_outputs=_as_bool(
				clear_templates_cache_cfg.get("keep_merged_per_unit_outputs", True),
				True,
			),
			keep_full_channels_templates=_as_bool(
				clear_templates_cache_cfg.get("keep_full_channels_templates", False),
				False,
			),
		),
		kssynth=ReconstructionKssynthPhaseConfig(
			enabled=_phase_enabled(kssynth_cfg, False),
			summary_json_relpath=str(
				kssynth_cfg.get(
					"summary_json_relpath",
					"synth_sorter_output/kssynth_summary.json",
				)
				or "synth_sorter_output/kssynth_summary.json"
			),
			resource_class=_phase_resource_class(kssynth_cfg, "kssynth"),
			channel_grid=str(kssynth_cfg.get("channel_grid", "union") or "union"),
			aggregation=str(
				kssynth_cfg.get("aggregation", "spike_count_weighted_mean")
				or "spike_count_weighted_mean"
			),
			tolerance_um=float(kssynth_cfg.get("tolerance_um", 1.0) or 1.0),
			dtype=str(kssynth_cfg.get("dtype", "int16") or "int16"),
			treat_zero_as_missing=_as_bool(
				kssynth_cfg.get("treat_zero_as_missing", True), True
			),
			clobber=_as_bool(kssynth_cfg.get("clobber", True), True),
		),
		axon_velocity_gtrs=ReconstructionGenerateGtrsPhaseConfig(
			enabled=_phase_enabled(axon_velocity_gtrs_cfg, True),
			summary_json_relpath=str(
				axon_velocity_gtrs_cfg.get("summary_json_relpath", "context/axon_velocity_gtrs_summary.json")
			),
			resource_class=_phase_resource_class(axon_velocity_gtrs_cfg, "axon_velocity_gtrs"),
			unit_procs=axon_velocity_gtrs_unit_procs,
			unit_batch_size=axon_velocity_gtrs_unit_batch_size,
			outputs=axon_velocity_gtrs_outputs,
			axon_velocity=ReconstructionAxonVelocityPhaseConfig(
				enabled=_phase_enabled(phase_axon_velocity_cfg, True),
				params=dict(av_cfg),
			),
		),
		plot_recons=ReconstructionPlotReconsPhaseConfig(
			enabled=_phase_enabled(plot_recons_cfg, True),
			summary_json_relpath=str(plot_recons_cfg.get("summary_json_relpath", "context/plot_recons_summary.json")),
			resource_class=_phase_resource_class(plot_recons_cfg, "plot_recons"),
			outputs=plot_recons_outputs,
		),
		plot_branch_propagations=ReconstructionPlotBranchPropagationsPhaseConfig(
			enabled=_phase_enabled(plot_branch_propagations_cfg, False),
			summary_json_relpath=str(
				plot_branch_propagations_cfg.get(
					"summary_json_relpath",
					"context/plot_branch_propagations_summary.json",
				)
			),
			resource_class=_phase_resource_class(
				plot_branch_propagations_cfg,
				"plot_branch_propagations",
			),
			branch_scope=_normalize_branch_scope(
				plot_branch_propagations_cfg.get("branch_scope", "raw"),
				default="raw",
			),
			display=_build_branch_propagation_display_config(plot_branch_propagations_display_cfg),
			output=_build_branch_plot_output_config(
				plot_branch_propagations_output_cfg,
				default_relpath="branch_plots/propagations",
				default_manifest_relpath="branch_propagations_manifest.json",
			),
		),
		plot_branch_velocities=ReconstructionPlotBranchVelocitiesPhaseConfig(
			enabled=_phase_enabled(plot_branch_velocities_cfg, False),
			summary_json_relpath=str(
				plot_branch_velocities_cfg.get(
					"summary_json_relpath",
					"context/plot_branch_velocities_summary.json",
				)
			),
			resource_class=_phase_resource_class(
				plot_branch_velocities_cfg,
				"plot_branch_velocities",
			),
			branch_scope=_normalize_branch_scope(
				plot_branch_velocities_cfg.get("branch_scope", "raw"),
				default="raw",
			),
			display=_build_branch_velocity_display_config(plot_branch_velocities_display_cfg),
			output=_build_branch_plot_output_config(
				plot_branch_velocities_output_cfg,
				default_relpath="branch_plots/velocities",
				default_manifest_relpath="branch_velocities_manifest.json",
			),
		),
		plot_unit_summary=ReconstructionPlotUnitSummaryPhaseConfig(
			enabled=_phase_enabled(plot_unit_summary_cfg, False),
			summary_json_relpath=str(
				plot_unit_summary_cfg.get(
					"summary_json_relpath",
					"context/plot_unit_summary_summary.json",
				)
			),
			resource_class=_phase_resource_class(plot_unit_summary_cfg, "plot_unit_summary"),
			display=_build_unit_summary_display_config(plot_unit_summary_display_cfg),
			output=_build_unit_summary_output_config(
				plot_unit_summary_output_cfg,
				default_relpath="reports/unit_summary",
			),
		),
		report_recons=ReconstructionReportReconsPhaseConfig(
			enabled=_phase_enabled(report_recons_cfg, True),
			summary_json_relpath=str(
				report_recons_cfg.get("summary_json_relpath", "context/report_recons_summary.json")
			),
			resource_class=_phase_resource_class(report_recons_cfg, "report_recons"),
			av_recons=ReconstructionAvReconsConfig(
				write_pdf=_as_bool(report_av_recons_cfg.get("write_pdf", False), False),
				pdf_relpath=str(report_av_recons_cfg.get("pdf_relpath", "av_recons.pdf")),
			),
			summary_png=ReconstructionSummaryPngConfig(
				write=write_summary_png,
				relpath=summary_png_relpath,
				grid_ncols=int(summary_grid_ncols),
			),
			report_md=ReconstructionReportMarkdownConfig(
				write=write_report_md,
				relpath=report_md_relpath,
			),
		),
		report_recon_grid=ReconstructionReportReconGridPhaseConfig(
			enabled=_phase_enabled(report_recon_grid_cfg, True),
			summary_json_relpath=str(
				report_recon_grid_cfg.get("summary_json_relpath", "context/report_recon_grid_summary.json")
			),
			resource_class=_phase_resource_class(report_recon_grid_cfg, "report_recon_grid"),
			output=ReconstructionReconGridOutputConfig(
				write_pdf=_as_bool(report_recon_grid_output_source_cfg.get("write_pdf", False), False),
				pdf_relpath=str(
					report_recon_grid_output_source_cfg.get("pdf_relpath", "reports/circle_recon_grid.pdf")
				),
				write_png=_as_bool(report_recon_grid_output_source_cfg.get("write_png", False), False),
				png_relpath=str(
					report_recon_grid_output_source_cfg.get("png_relpath", "reports/circle_recon_grid.png")
				),
				write_svg=_as_bool(report_recon_grid_output_source_cfg.get("write_svg", False), False),
				svg_relpath=str(
					report_recon_grid_output_source_cfg.get("svg_relpath", "reports/circle_recon_grid.svg")
				),
				keep_temp_svg=_as_bool(report_recon_grid_output_source_cfg.get("keep_temp_svg", False), False),
				temp_svg_relpath=str(
					report_recon_grid_output_source_cfg.get(
						"temp_svg_relpath",
						"reports/circle_recon_grid__temp.svg",
					)
				),
			),
			display=ReconstructionReconGridDisplayConfig(
				show_title=_as_bool(report_recon_grid_display_source_cfg.get("show_title", True), True),
			),
			render=ReconstructionReconGridRenderConfig(
				dpi=float(max(72.0, report_recon_grid_dpi)),
			),
		),
		report_full_chip_layout=ReconstructionReportFullChipLayoutPhaseConfig(
			enabled=_phase_enabled(report_full_chip_layout_cfg, False),
			summary_json_relpath=str(
				report_full_chip_layout_cfg.get(
					"summary_json_relpath",
					"context/report_full_chip_layout_summary.json",
				)
			),
			resource_class=_phase_resource_class(
				report_full_chip_layout_cfg,
				"report_full_chip_layout",
			),
			branch_scope=_normalize_branch_scope(
				report_full_chip_layout_cfg.get("branch_scope", "raw"),
				default="raw",
			),
			unit_colors=_build_full_chip_layout_color_config(report_full_chip_layout_colors_cfg),
			display=_build_full_chip_layout_display_config(report_full_chip_layout_display_cfg),
			output=_build_full_chip_layout_output_config(
				report_full_chip_layout_output_cfg,
				default_relpath="reports/full_chip_layout",
				default_manifest_relpath="reports/full_chip_layout_manifest.json",
			),
		),
		report_summaries=ReconstructionReportSummariesPhaseConfig(
			enabled=_phase_enabled(report_summaries_cfg, False),
			summary_json_relpath=str(
				report_summaries_cfg.get(
					"summary_json_relpath",
					"context/report_summaries_summary.json",
				)
			),
			resource_class=_phase_resource_class(report_summaries_cfg, "report_summaries"),
			write_pdf=_as_bool(report_summaries_cfg.get("write_pdf", True), True),
			pdf_relpath=str(
				report_summaries_cfg.get("pdf_relpath", "reports/reconstruct_summary_deck.pdf")
			),
		),
	)

	per_unit = PerUnitOutputsConfig(
		unit_reldir=unit_reldir,
		write_branches_raw_json=bool(axon_velocity_gtrs_outputs.write_branches_raw_json),
		branches_raw_relpath=str(axon_velocity_gtrs_outputs.branches_raw_relpath),
		write_branches_json=bool(axon_velocity_gtrs_outputs.write_branches_json),
		branches_relpath=str(axon_velocity_gtrs_outputs.branches_relpath),
		write_detection_filter_json=bool(axon_velocity_gtrs_outputs.write_detection_filter_json),
		detection_filter_relpath=str(axon_velocity_gtrs_outputs.detection_filter_relpath),
		write_kurtosis_filter_json=bool(axon_velocity_gtrs_outputs.write_kurtosis_filter_json),
		kurtosis_filter_relpath=str(axon_velocity_gtrs_outputs.kurtosis_filter_relpath),
		write_peak_std_filter_json=bool(axon_velocity_gtrs_outputs.write_peak_std_filter_json),
		peak_std_filter_relpath=str(axon_velocity_gtrs_outputs.peak_std_filter_relpath),
		write_delay_filter_json=bool(axon_velocity_gtrs_outputs.write_delay_filter_json),
		delay_filter_relpath=str(axon_velocity_gtrs_outputs.delay_filter_relpath),
		write_all_filters_json=bool(axon_velocity_gtrs_outputs.write_all_filters_json),
		all_filters_relpath=str(axon_velocity_gtrs_outputs.all_filters_relpath),
		write_heuristics_json=bool(axon_velocity_gtrs_outputs.write_heuristics_json),
		heuristics_relpath=str(axon_velocity_gtrs_outputs.heuristics_relpath),
		write_gtr_pkl=bool(axon_velocity_gtrs_outputs.write_gtr_pkl),
		gtr_pkl_relpath=str(axon_velocity_gtrs_outputs.gtr_pkl_relpath),
		template_source=str(axon_velocity_gtrs_outputs.template_source),
		write_gtr_json=bool(axon_velocity_gtrs_outputs.write_gtr_json),
		gtr_json_relpath=str(axon_velocity_gtrs_outputs.gtr_json_relpath),
		channel_selection_figure=axon_velocity_gtrs_outputs.channel_selection_figure,
		axon_reconstruction_figure=axon_velocity_gtrs_outputs.axon_reconstruction_figure,
		amplitude_map_png_relpath=plot_recons_outputs.amplitude_map.png_relpath,
		circle_recon=plot_recons_outputs.circle_recon,
	)

	return ReconstructionStageConfig(
		output_rel_root=output_rel_root,
		unit_reldir=unit_reldir,
		report_sort_by=report_sort_by,
		overwrite_report_outputs_on_unit_rerun=overwrite_report_outputs_on_unit_rerun,
		phase_sequence=phase_sequence,
		debug_prints=debug_prints,
		debug_mode_enabled=debug_mode_enabled,
		debug_limit_datasets=debug_limit_datasets,
		debug_limit_wells=debug_limit_wells,
		debug_limit_wells_per_dataset=debug_limit_wells_per_dataset,
		branch_colors=branch_colors,
		cleanup_failed_unit_outputs=cleanup_failed_unit_outputs,
		failed_units_summary_relpath=failed_units_summary_relpath,
		per_unit_outputs=per_unit,
		unit_ids=unit_ids,
		unit_limit=unit_limit,
		limit_segments=limit_segments,
		phases=phases,
		use_full_channels_templates=True,
		require_full_channels_templates=True,
		force_restart=force_restart,
		replot=replot,
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
		debug_prints=stage_config.debug_prints,
		output_rel_root=stage_config.output_rel_root,
		unit_reldir=stage_config.unit_reldir,
		report_sort_by=stage_config.report_sort_by,
		debug_mode_enabled=stage_config.debug_mode_enabled,
		debug_limit_datasets=stage_config.debug_limit_datasets,
		debug_limit_wells=stage_config.debug_limit_wells,
		debug_limit_wells_per_dataset=stage_config.debug_limit_wells_per_dataset,
		overwrite_report_outputs_on_unit_rerun=stage_config.overwrite_report_outputs_on_unit_rerun,
		phase_sequence=stage_config.phase_sequence,
		branch_colors=stage_config.branch_colors,
		cleanup_failed_unit_outputs=stage_config.cleanup_failed_unit_outputs,
		failed_units_summary_relpath=stage_config.failed_units_summary_relpath,
		per_unit_outputs=stage_config.per_unit_outputs,
		unit_ids=stage_config.unit_ids,
		unit_limit=stage_config.unit_limit,
		limit_segments=stage_config.limit_segments,
		phases=stage_config.phases,
		use_full_channels_templates=stage_config.use_full_channels_templates,
		require_full_channels_templates=stage_config.require_full_channels_templates,
		force_restart=stage_config.force_restart,
		replot=stage_config.replot,
		n_jobs=max(1, int(unit_workers)),
		axon_velocity_params=dict(stage_config.axon_velocity_params),
		probe_geometry=probe_geometry,
	)


def load_reconstruction_inputs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
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
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=data_cfg)

	reconstruct_inputs = ReconstructionInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		final_output_root=output_root,
		debug_prints=stage_cfg.debug_prints,
		output_rel_root=stage_cfg.output_rel_root,
		unit_reldir=stage_cfg.unit_reldir,
		report_sort_by=stage_cfg.report_sort_by,
		debug_mode_enabled=stage_cfg.debug_mode_enabled,
		debug_limit_datasets=stage_cfg.debug_limit_datasets,
		debug_limit_wells=stage_cfg.debug_limit_wells,
		debug_limit_wells_per_dataset=stage_cfg.debug_limit_wells_per_dataset,
		overwrite_report_outputs_on_unit_rerun=stage_cfg.overwrite_report_outputs_on_unit_rerun,
		phase_sequence=stage_cfg.phase_sequence,
		branch_colors=stage_cfg.branch_colors,
		cleanup_failed_unit_outputs=stage_cfg.cleanup_failed_unit_outputs,
		failed_units_summary_relpath=stage_cfg.failed_units_summary_relpath,
		per_unit_outputs=stage_cfg.per_unit_outputs,
		unit_ids=stage_cfg.unit_ids,
		unit_limit=stage_cfg.unit_limit,
		limit_segments=stage_cfg.limit_segments,
		phases=stage_cfg.phases,
		use_full_channels_templates=stage_cfg.use_full_channels_templates,
		require_full_channels_templates=stage_cfg.require_full_channels_templates,
		force_restart=stage_cfg.force_restart,
		replot=stage_cfg.replot,
		n_jobs=1,
		axon_velocity_params=stage_cfg.axon_velocity_params,
		probe_geometry=probe_geometry,
	)
	templates_runtime_config = build_reconstruct_templates_runtime_config(runtime_cfg)
	reconstruct_templates_cfg = parse_reconstruct_templates_config(
		runtime_config=templates_runtime_config,
		probe_geometry=probe_geometry,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=stage_cfg.unit_limit,
		limit_segments_override=stage_cfg.limit_segments,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	target = ExecutionTarget(
		dataset_index=0,
		dataset_id="dataset_000",
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		final_output_root=output_root,
	)
	templates_inputs = build_templates_inputs_for_target(
		target=target,
		stage_config=reconstruct_templates_cfg,
		unit_workers=1,
		probe_geometry=probe_geometry,
	)
	return replace(reconstruct_inputs, templates_inputs=templates_inputs)
