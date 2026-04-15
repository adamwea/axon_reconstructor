from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig
from axon_recon.pipeline.shared.grid_sorting import normalize_grid_sort_by
from axon_recon.pipeline.shared.plotting import build_stage_plot_block

from ...execution.context import ExecutionTarget
from .models.inputs import (
	AnalyzerPreparationPolicyConfig,
	AnalyzerCacheConfig,
	AnalyzerSourcePhaseConfig,
	CenterMostChannelCoordsConfig,
	DataQualityChecksOutputsConfig,
	FootprintGridsReportConfig,
	FootprintMapGridReportConfig,
	FootprintMapConfig,
	FootprintPlotsConfig,
	MergeConfig,
	MultipleNegativePeaksOutputsConfig,
	MultipleNegativePeaksCheckConfig,
	MultiSourcePdfReportConfig,
	PerUnitQualityChecksOutputsConfig,
	PerUnitTemplatesOutputsConfig,
	ProbeGeometryConfig,
	TemplateAnalysisPhaseConfig,
	TemplateBuildTemplatesPhaseConfig,
	TemplateExtractTemplateSegmentsPhaseConfig,
	TemplateLeafPhaseConfig,
	TemplatePerUnitProcessingPhaseConfig,
	TemplatePlotsPhaseConfig,
	TemplateReportTemplatesPhaseConfig,
	TemplatePropagationOrderingPhaseConfig,
	TemplateQualityChecksPhaseConfig,
	TemplateReportsPhaseConfig,
	PropagationAxesConfig,
	PropagationLatencyMapConfig,
	PropagationPlotConfig,
	QualityCheckJsonOutputConfig,
	QualityCheckPlotOutputConfig,
	QualityChecksConfig,
	ResolveSourcesPhaseConfig,
	ReportsConfig,
	TemplateArtifactConfig,
	TemplatesAnalyzersPhaseConfig,
	TemplateCirclesBranchMorphologyConfig,
	TemplateCirclesOverlapControlsConfig,
	TemplateCirclesPlotConfig,
	TemplateScaleCircleConfig,
	TemplatePlotConfig,
	TemplateWaveformOverlayConfig,
	TemplatesPhasesConfig,
	UnitLocationsReportConfig,
	UnitIdLabelConfig,
	TopographicalFootprintConfig,
	TopographicalFootprintsConfig,
	TimeUpsampleConfig,
	TemplatesInputs,
	WaveformExtractionConfig,
	WfOverlayGridReportConfig,
)


LOGGER = logging.getLogger("axon_recon.templates.config")


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


def _as_float(value: Any, default: float) -> float:
	if value is None:
		return float(default)
	try:
		return float(value)
	except Exception:
		return float(default)


def _as_int(value: Any, default: int) -> int:
	if value is None:
		return int(default)
	try:
		return int(value)
	except Exception:
		return int(default)


def _normalize_channel_scope(raw: Any) -> str:
	v = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
	if v in {"contributing", "contributing_channel", "contributing_channels", "branches"}:
		return "contributing_channels"
	if v in {"recorded", "recorded_channel", "recorded_channels"}:
		return "recorded_channels"
	if v in {"all", "all_channel", "all_channels"}:
		return "all_channels"
	return "contributing_channels"


def _normalize_template_metric(raw: Any, default: str) -> str:
	v = str(raw or default).strip().lower().replace("-", "_").replace(" ", "_")
	if v in {"amplitude", "amp", "ptp"}:
		return "amplitude"
	if v in {"latency", "lat"}:
		return "latency"
	return str(default)


def _normalize_horizontal_alignment(raw: Any, default: str = "left") -> str:
	v = str(raw or default).strip().lower()
	if v in {"left", "center", "right"}:
		return v
	return str(default)


def _normalize_vertical_alignment(raw: Any, default: str = "top") -> str:
	v = str(raw or default).strip().lower()
	if v in {"top", "center", "bottom"}:
		return v
	return str(default)


def _normalize_optional_linestyle(raw: Any, default: str | None = "solid") -> str | None:
	if raw is None:
		return None
	v = str(raw).strip()
	if v.lower() in {"", "none", "null", "off", "false"}:
		return None
	return v

def _normalize_grid_render_mode(raw: Any, default: str = "image_composite") -> str:
	v = str(raw or default).strip().lower().replace("-", "_").replace(" ", "_")
	if v in {"direct_replot", "image_composite"}:
		return v
	return str(default)


def _first_dict_block(runtime_config: RuntimeConfig, paths: tuple[str, ...]) -> dict[str, Any]:
	for path in paths:
		block = runtime_config.get(path, {})
		if isinstance(block, dict) and block:
			return dict(block)
	return {}


def _output_paths(*suffixes: str) -> tuple[str, ...]:
	paths: list[str] = []
	for suffix in suffixes:
		s = suffix.strip(".")
		paths.append(f"stages.templates.outputs.{s}")
		paths.append(f"stages.outputs.{s}")
	return tuple(paths)


def _phase_plot_output_paths(*suffixes: str) -> tuple[str, ...]:
	paths: list[str] = []
	for suffix in suffixes:
		s = suffix.strip(".")
		if s:
			paths.append(f"stages.templates.phases.plot_templates.outputs.{s}")
			paths.append(f"stages.templates.phases.plot_templates.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.plots.outputs.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.plots.{s}")
		else:
			paths.append("stages.templates.phases.plot_templates.outputs")
			paths.append("stages.templates.phases.plot_templates")
			paths.append("stages.templates.phases.per_unit_processing.plots.outputs")
			paths.append("stages.templates.phases.per_unit_processing.plots")
	return tuple(paths)


def _phase_build_output_paths(*suffixes: str) -> tuple[str, ...]:
	paths: list[str] = []
	for suffix in suffixes:
		s = suffix.strip(".")
		if s:
			paths.append(f"stages.templates.phases.build_templates.outputs.{s}")
			paths.append(f"stages.templates.phases.build_templates.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.build_templates.outputs.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.build_templates.{s}")
		else:
			paths.append("stages.templates.phases.build_templates.outputs")
			paths.append("stages.templates.phases.build_templates")
			paths.append("stages.templates.phases.per_unit_processing.build_templates.outputs")
			paths.append("stages.templates.phases.per_unit_processing.build_templates")
	return tuple(paths)


def _phase_reports_paths(*suffixes: str) -> tuple[str, ...]:
	paths: list[str] = []
	for suffix in suffixes:
		s = suffix.strip(".")
		if s:
			paths.append(f"stages.templates.phases.reports.config.{s}")
			paths.append(f"stages.templates.phases.reports.{s}")
		else:
			paths.append("stages.templates.phases.reports.config")
			paths.append("stages.templates.phases.reports")
	return tuple(paths)


def _phase_quality_paths(*suffixes: str) -> tuple[str, ...]:
	paths: list[str] = []
	for suffix in suffixes:
		s = suffix.strip(".")
		if s:
			paths.append(f"stages.templates.phases.per_unit_processing.quality_checks.outputs.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.quality_checks.config.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.quality_checks.{s}")
		else:
			paths.append("stages.templates.phases.per_unit_processing.quality_checks.outputs")
			paths.append("stages.templates.phases.per_unit_processing.quality_checks.config")
			paths.append("stages.templates.phases.per_unit_processing.quality_checks")
	return tuple(paths)


def _phase_analyzer_output_paths(*suffixes: str) -> tuple[str, ...]:
	paths: list[str] = []
	for suffix in suffixes:
		s = suffix.strip(".")
		if s:
			paths.append(f"stages.templates.phases.analyzers.outputs.{s}")
			paths.append(f"stages.templates.phases.analyzers.{s}")
		else:
			paths.append("stages.templates.phases.analyzers.outputs")
			paths.append("stages.templates.phases.analyzers")
	return tuple(paths)


def _phase_per_unit_output_paths(*suffixes: str) -> tuple[str, ...]:
	paths: list[str] = []
	for suffix in suffixes:
		s = suffix.strip(".")
		if s:
			paths.append(f"stages.templates.phases.build_templates.outputs.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.outputs.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.plots.outputs.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.build_templates.outputs.{s}")
			paths.append(f"stages.templates.phases.per_unit_processing.quality_checks.outputs.{s}")
		else:
			paths.append("stages.templates.phases.build_templates.outputs")
			paths.append("stages.templates.phases.per_unit_processing.outputs")
			paths.append("stages.templates.phases.per_unit_processing.plots.outputs")
			paths.append("stages.templates.phases.per_unit_processing.build_templates.outputs")
			paths.append("stages.templates.phases.per_unit_processing.quality_checks.outputs")
	return tuple(paths)


def _get_analyzer_cache_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_phase_analyzer_output_paths("analyzer_cache"),
			*_output_paths("analyzer_cache"),
		),
	)


def _get_template_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
			*_phase_plot_output_paths("template_plots.waveforms"),
			*_phase_plot_output_paths("full_template.template_plots.waveforms"),
			*_phase_plot_output_paths("template"),
			*_phase_plot_output_paths("template_plot"),
			*_output_paths("per_unit_outputs.template_plots.waveforms"),
			*_output_paths("per_unit_outputs.full_template.template_plots.waveforms"),
			*_output_paths("per_unit_outputs.template"),
			*_output_paths("per_unit_outputs.template_plot"),
			"stages.reconstruct.outputs.per_unit_outputs.template",
			"stages.reconstruct.outputs.per_unit_outputs.template_plot",
		),
		global_paths=(
			"default",
			"template",
			"template_plots.default",
			"template_plots.waveforms",
		),
	)


def _get_template_circles_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
			*_phase_plot_output_paths("circles"),
			*_phase_plot_output_paths("template_plots.circles"),
			*_phase_plot_output_paths("full_template.template_plots.circles"),
			*_phase_plot_output_paths("template_circles"),
			*_output_paths("per_unit_outputs.template_plots.circles"),
			*_output_paths("per_unit_outputs.full_template.template_plots.circles"),
			*_output_paths("per_unit_outputs.template_circles"),
		),
		global_paths=(
			"default",
			"template",
			"template_plots.default",
			"template_plots.circles",
			"circles",
		),
	)


def _get_template_wf_overlay_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_phase_plot_output_paths("extremum_ch_wf_overlay"),
			*_phase_plot_output_paths("full_template.extremum_ch_wf_overlay"),
			*_phase_plot_output_paths("template_wf_overlay"),
			*_phase_plot_output_paths("full_template.template_wf_overlay"),
			*_output_paths("per_unit_outputs.extremum_ch_wf_overlay"),
			*_output_paths("per_unit_outputs.full_template.extremum_ch_wf_overlay"),
			*_output_paths("per_unit_outputs.template_wf_overlay"),
			*_output_paths("per_unit_outputs.full_template.template_wf_overlay"),
		),
	)


def _get_per_unit_quality_checks_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_phase_quality_paths("quality_checks"),
			*_output_paths("per_unit_outputs.quality_checks"),
		),
	)


def _get_data_quality_checks_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_phase_quality_paths("data_outputs.quality_checks"),
			*_output_paths("data_outputs.quality_checks"),
		),
	)


def _get_reports_wf_overlay_grid_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_phase_reports_paths("grids.wf_overlay_grid"),
			*_phase_reports_paths("wf_overlay_grid"),
			*_output_paths("reports.grids.wf_overlay_grid"),
			*_output_paths("reports.wf_overlay_grid"),
			*_output_paths("per_unit_outputs.reports.wf_overlay_grid"),
		),
	)


def _get_reports_locations_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_phase_reports_paths("locations"),
			*_phase_reports_paths("unit_locations"),
			*_output_paths("reports.locations"),
			*_output_paths("reports.unit_locations"),
			*_output_paths("per_unit_outputs.reports.locations"),
			*_output_paths("per_unit_outputs.reports.unit_locations"),
		),
	)


def _get_reports_footprint_grids_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = _first_dict_block(
		runtime_config,
		(
			*_phase_reports_paths("grids.footprint_grids"),
			*_phase_reports_paths("footprint_grids"),
			*_output_paths("reports.grids.footprint_grids"),
			*_output_paths("reports.footprint_grids"),
			*_output_paths("per_unit_outputs.reports.footprint_grids"),
		),
	)
	if stage_block:
		return stage_block

	# Compatibility alias for older split spelling.
	return _first_dict_block(
		runtime_config,
		(
			*_phase_reports_paths("grids.foot_print_grids"),
			*_phase_reports_paths("foot_print_grids"),
			*_output_paths("reports.grids.foot_print_grids"),
			*_output_paths("reports.foot_print_grids"),
			*_output_paths("per_unit_outputs.reports.foot_print_grids"),
		),
	)


def _get_reports_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_phase_reports_paths("config"),
			*_phase_reports_paths(""),
			*_output_paths("reports"),
			*_output_paths("per_unit_outputs.reports"),
		),
	)


def _get_footprint_map_block(runtime_config: RuntimeConfig, map_name: str) -> dict[str, Any]:
	return build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
			*_phase_plot_output_paths(f"footprint_plots.{map_name}"),
			*_phase_plot_output_paths(f"full_template.footprint_plots.{map_name}"),
			*_output_paths(f"per_unit_outputs.footprint_plots.{map_name}"),
			*_output_paths(f"per_unit_outputs.full_template.footprint_plots.{map_name}"),
		),
		global_paths=(
			"default",
			"footprint_plots.default",
			f"footprint_plots.{map_name}",
			"maps.default",
			f"maps.{map_name}",
		),
	)


def _get_topographical_footprint_block(runtime_config: RuntimeConfig, map_name: str) -> dict[str, Any]:
	return build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
			*_phase_plot_output_paths(f"topographical_footprints.{map_name}"),
			*_phase_plot_output_paths(f"full_template.topographical_footprints.{map_name}"),
			*_output_paths(f"per_unit_outputs.topographical_footprints.{map_name}"),
			*_output_paths(f"per_unit_outputs.full_template.topographical_footprints.{map_name}"),
		),
		global_paths=(
			"default",
			"topographical_footprints.default",
			f"topographical_footprints.{map_name}",
		),
	)


def _get_propagation_plots_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
			*_phase_plot_output_paths("propagation_plots"),
			*_phase_plot_output_paths("full_template.propagation_plots"),
			*_output_paths("per_unit_outputs.propagation_plots"),
			*_output_paths("per_unit_outputs.full_template.propagation_plots"),
		),
		global_paths=(
			"default",
			"propagation_plots.default",
			"propagation_plots",
		),
	)


def _get_template_artifact_block(runtime_config: RuntimeConfig, block_name: str) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_phase_build_output_paths(block_name),
			*_output_paths(f"per_unit_outputs.{block_name}"),
		),
	)


def _get_merge_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			"stages.templates.phases.build_templates.merge",
			"stages.templates.phases.per_unit_processing.build_templates.merge",
			"stages.templates.merge",
			"stages.templates.execution.merge",
		),
	)


def _get_resolve_sources_phase_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = runtime_config.get("stages.templates.phases.resolve_sources", {})
	if isinstance(stage_block, dict) and stage_block:
		return dict(stage_block)
	return {}


def _as_float_or_none(value: Any, default: float | None) -> float | None:
	if value is None:
		return default
	try:
		return float(value)
	except Exception:
		return default


def _as_tuple(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
	if value is None:
		return tuple(default)
	if isinstance(value, (list, tuple)):
		return tuple(value)
	return tuple(default)


def _normalize_padding_value(raw: Any) -> str:
	v = str(raw or "zero").strip().lower()
	if v in {"zero", "zeros", "0"}:
		return "zero"
	if v in {"one", "ones", "1"}:
		return "one"
	if v in {"nan", "na"}:
		return "nan"
	return "zero"


def _normalize_template_shape(raw: Any, default: str = "square") -> str:
	v = str(raw or default).strip().lower().replace("-", "_").replace(" ", "_")
	if v in {"square"}:
		return "square"
	if v in {"scan", "scanline", "scan_line"}:
		return "scan"
	if v in {"full", "all", "full_grid"}:
		return "full"
	if v in {"top_channels_only", "top_channels", "topchannels", "top"}:
		return "top_channels_only"
	return default


def _parse_max_waveforms_per_source_channel(raw: Any, default: int = 500) -> int | None:
	# Values <= 0 are treated as unlimited.
	parsed = _as_int(raw, default)
	if parsed <= 0:
		return None
	return parsed


def _get_nested_block(raw_cfg: dict[str, Any], key: str) -> dict[str, Any]:
	block = raw_cfg.get(key, {})
	if isinstance(block, dict) and block:
		return dict(block)
	return {}


def _nested_or_flat(
	raw_cfg: dict[str, Any],
	*,
	block: str,
	key: str,
	flat_keys: tuple[str, ...],
	default: Any,
) -> Any:
	nested = _get_nested_block(raw_cfg, block)
	if key in nested:
		return nested.get(key)
	for flat_key in flat_keys:
		if flat_key in raw_cfg:
			return raw_cfg.get(flat_key)
	return default


def _nested_path_or_flat(
	raw_cfg: dict[str, Any],
	*,
	path: tuple[str, ...],
	key: str,
	flat_keys: tuple[str, ...],
	default: Any,
) -> Any:
	block: Any = raw_cfg
	for path_key in path:
		if not isinstance(block, dict):
			block = {}
			break
		next_block = block.get(path_key, {})
		if not isinstance(next_block, dict):
			next_block = {}
		block = next_block
	if isinstance(block, dict) and key in block:
		return block.get(key)
	for flat_key in flat_keys:
		if flat_key in raw_cfg:
			return raw_cfg.get(flat_key)
	return default


def _build_footprint_grid_report_config(
	raw_cfg: dict[str, Any],
	*,
	pdf_relpath_default: str,
	png_relpath_default: str,
	svg_relpath_default: str,
	temp_svg_relpath_default: str,
) -> FootprintMapGridReportConfig:
	return FootprintMapGridReportConfig(
		write_pdf=_as_bool(
			_nested_or_flat(raw_cfg, block="output", key="write_pdf", flat_keys=("write_pdf",), default=False),
			False,
		),
		pdf_relpath=str(
			_nested_or_flat(raw_cfg, block="output", key="pdf_relpath", flat_keys=("pdf_relpath",), default=pdf_relpath_default)
		),
		write_png=_as_bool(
			_nested_or_flat(raw_cfg, block="output", key="write_png", flat_keys=("write_png",), default=True),
			True,
		),
		png_relpath=str(
			_nested_or_flat(raw_cfg, block="output", key="png_relpath", flat_keys=("png_relpath",), default=png_relpath_default)
		),
		write_svg=_as_bool(
			_nested_or_flat(raw_cfg, block="output", key="write_svg", flat_keys=("write_svg",), default=False),
			False,
		),
		svg_relpath=str(
			_nested_or_flat(raw_cfg, block="output", key="svg_relpath", flat_keys=("svg_relpath",), default=svg_relpath_default)
		),
		keep_temp_svg=_as_bool(
			_nested_or_flat(raw_cfg, block="output", key="keep_temp_svg", flat_keys=("keep_temp_svg",), default=False),
			False,
		),
		temp_svg_relpath=str(
			_nested_or_flat(
				raw_cfg,
				block="output",
				key="temp_svg_relpath",
				flat_keys=("temp_svg_relpath",),
				default=temp_svg_relpath_default,
			)
		),
		show_title=_as_bool(
			_nested_or_flat(raw_cfg, block="display", key="show_title", flat_keys=("show_title",), default=True),
			True,
		),
		template_shape=_normalize_template_shape(
			_nested_or_flat(
				raw_cfg,
				block="render",
				key="template",
				flat_keys=("template_shape", "template"),
				default="square",
			),
			"square",
		),
		global_color_scale=_as_bool(
			_nested_or_flat(raw_cfg, block="render", key="global_color_scale", flat_keys=("global_color_scale",), default=True),
			True,
		),
		subplot_background_color=str(
			_nested_or_flat(
				raw_cfg,
				block="render",
				key="subplot_background_color",
				flat_keys=("subplot_background_color",),
				default="white",
			)
		),
		figure_background_color=str(
			_nested_or_flat(
				raw_cfg,
				block="render",
				key="figure_background_color",
				flat_keys=("figure_background_color",),
				default="white",
			)
		),
		render_mode=_normalize_grid_render_mode(
			_nested_or_flat(raw_cfg, block="render", key="mode", flat_keys=("render_mode",), default="direct_replot"),
			"direct_replot",
		),
		dpi=max(
			72.0,
			_as_float(
				_nested_or_flat(raw_cfg, block="render", key="dpi", flat_keys=("dpi",), default=300.0),
				300.0,
			),
		),
	)


def _normalize_overlap_priority(raw: Any) -> tuple[str, ...]:
	items = _as_tuple(raw, ("electrode_id", "channel_id", "location"))
	norm: list[str] = []
	for item in items:
		t = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
		if t in {"electrode", "electrode_id", "eid"}:
			norm.append("electrode_id")
		elif t in {"channel", "channel_id", "chid"}:
			norm.append("channel_id")
		elif t in {"location", "xy", "loc"}:
			norm.append("location")
	seen: list[str] = []
	for n in norm:
		if n not in seen:
			seen.append(n)
	if not seen:
		seen = ["electrode_id", "channel_id", "location"]
	return tuple(seen)


def _build_footprint_map_config(raw_cfg: dict[str, Any], *, relpath_default: str) -> FootprintMapConfig:
	template_cfg = _get_nested_block(raw_cfg, "template")
	color_bar_cfg = _get_nested_block(raw_cfg, "color_bar")
	return FootprintMapConfig(
		write_png=_as_bool(_nested_or_flat(raw_cfg, block="output", key="write_png", flat_keys=("write_png",), default=True), True),
		write_svg=_as_bool(_nested_or_flat(raw_cfg, block="output", key="write_svg", flat_keys=("write_svg",), default=False), False),
		relpath=str(_nested_or_flat(raw_cfg, block="output", key="relpath", flat_keys=("relpath",), default=relpath_default)),
		background=str(_nested_or_flat(raw_cfg, block="render", key="background", flat_keys=("background",), default="black")),
		color_map=str(_nested_or_flat(raw_cfg, block="render", key="color_map", flat_keys=("color_map",), default="viridis")),
		template_shape=_normalize_template_shape(template_cfg.get("shape", raw_cfg.get("template_shape", "square")), "square"),
		template_padding_value=_normalize_padding_value(template_cfg.get("padding_value", raw_cfg.get("template_padding_value", "zero"))),
		show_color_bar=_as_bool(color_bar_cfg.get("show", color_bar_cfg.get("show_color_bar", raw_cfg.get("show_color_bar", True))), True),
		color_bar_location=str(color_bar_cfg.get("location", color_bar_cfg.get("color_bar_location", raw_cfg.get("color_bar_location", "topright")))),
		color_bar_fontsize=_as_float(color_bar_cfg.get("fontsize", color_bar_cfg.get("color_bar_fontsize", raw_cfg.get("color_bar_fontsize", 6.0))), 6.0),
		color_bar_length_fraction=_as_float(color_bar_cfg.get("length_fraction", color_bar_cfg.get("color_bar_length_fraction", raw_cfg.get("color_bar_length_fraction", 0.3))), 0.3),
		color_bar_pad_fraction=_as_float(color_bar_cfg.get("pad_fraction", color_bar_cfg.get("color_bar_pad_fraction", raw_cfg.get("color_bar_pad_fraction", 0.02))), 0.02),
		force_low_value=_as_float_or_none(color_bar_cfg.get("force_low_value", raw_cfg.get("force_low_value", 0.0)), 0.0),
		force_high_value=_as_float_or_none(color_bar_cfg.get("force_high_value", raw_cfg.get("force_high_value", None)), None),
		scale=str(color_bar_cfg.get("scale", raw_cfg.get("scale", "linear"))),
		percentile_low=_as_float(color_bar_cfg.get("percentile_low", raw_cfg.get("percentile_low", 5.0)), 5.0),
		percentile_high_linear=_as_float(color_bar_cfg.get("percentile_high_linear", raw_cfg.get("percentile_high_linear", 99.0)), 99.0),
		percentile_high_log=_as_float(color_bar_cfg.get("percentile_high_log", raw_cfg.get("percentile_high_log", 99.5)), 99.5),
		knot_anchor_values=tuple(_as_tuple(color_bar_cfg.get("knot_anchor_values", raw_cfg.get("knot_anchor_values", (1.0, 10.0))), (1.0, 10.0))),
		knot_y1_min=_as_float(color_bar_cfg.get("knot_y1_min", raw_cfg.get("knot_y1_min", 0.02)), 0.02),
		knot_y1_max=_as_float(color_bar_cfg.get("knot_y1_max", raw_cfg.get("knot_y1_max", 0.90)), 0.90),
		knot_y2_min=_as_float(color_bar_cfg.get("knot_y2_min", raw_cfg.get("knot_y2_min", 0.07)), 0.07),
		knot_y2_max=_as_float(color_bar_cfg.get("knot_y2_max", raw_cfg.get("knot_y2_max", 0.98)), 0.98),
		knot_min_gap=_as_float(color_bar_cfg.get("knot_min_gap", raw_cfg.get("knot_min_gap", 0.05)), 0.05),
		linear_cap_rounding_mode=str(color_bar_cfg.get("linear_cap_rounding_mode", raw_cfg.get("linear_cap_rounding_mode", "ceil_step"))),
		linear_cap_rounding_step=_as_float(color_bar_cfg.get("linear_cap_rounding_step", raw_cfg.get("linear_cap_rounding_step", 10.0)), 10.0),
		linear_cap_min_vmax=_as_float(color_bar_cfg.get("linear_cap_min_vmax", raw_cfg.get("linear_cap_min_vmax", 11.0)), 11.0),
		show_ticks=tuple(_as_tuple(color_bar_cfg.get("show_ticks", raw_cfg.get("show_ticks", (1, 10, "dynamic_high"))), (1, 10, "dynamic_high"))),
	)


def _build_time_upsample_config(raw_cfg: dict[str, Any]) -> TimeUpsampleConfig:
	factor = max(1, _as_int(raw_cfg.get("factor", 1), 1))
	enabled_raw = raw_cfg.get("enabled", None)
	enabled = _as_bool(enabled_raw, factor > 1) if enabled_raw is not None else bool(factor > 1)
	raw_fallback_raw = raw_cfg.get("raw_rate_fallback_hz", None)
	raw_rate_fallback_hz = None if raw_fallback_raw in {None, ""} else _as_float(raw_fallback_raw, 0.0)
	if raw_rate_fallback_hz is not None and raw_rate_fallback_hz <= 0.0:
		raw_rate_fallback_hz = None
	return TimeUpsampleConfig(
		enabled=bool(enabled),
		factor=int(factor),
		method=str(raw_cfg.get("method", "sinc")),
		mismatch_tolerance_hz=max(0.0, _as_float(raw_cfg.get("mismatch_tolerance_hz", 0.5), 0.5)),
		raw_rate_fallback_hz=raw_rate_fallback_hz,
	)


def _parse_max_spikes_per_unit(raw: Any) -> int | None:
	if raw in {None, ""}:
		return None
	parsed = _as_int(raw, -1)
	if parsed <= 0:
		return None
	return int(parsed)


def _parse_optional_positive_int(raw: Any) -> int | None:
	if raw in {None, ""}:
		return None
	parsed = _as_int(raw, -1)
	if parsed <= 0:
		return None
	return int(parsed)


def _parse_optional_text(raw: Any) -> str | None:
	if raw is None:
		return None
	text = str(raw).strip()
	return (text or None)


def _build_waveform_extraction_config(
	*,
	spikeinterface_cfg: dict[str, Any],
	runtime_config: RuntimeConfig,
) -> WaveformExtractionConfig:
	wf_extract_cfg = spikeinterface_cfg.get("waveform_extraction", {}) if isinstance(spikeinterface_cfg.get("waveform_extraction", {}), dict) else {}
	wf_extract_window_cfg = _get_nested_block(wf_extract_cfg, "window")
	legacy_waveforms_cfg = runtime_config.get("stages.waveforms", {}) if isinstance(runtime_config.get("stages.waveforms", {}), dict) else {}

	ms_before = _as_float_or_none(
		wf_extract_window_cfg.get(
			"ms_before",
			wf_extract_cfg.get("ms_before", legacy_waveforms_cfg.get("ms_before", None)),
		),
		None,
	)
	ms_after = _as_float_or_none(
		wf_extract_window_cfg.get(
			"ms_after",
			wf_extract_cfg.get("ms_after", legacy_waveforms_cfg.get("ms_after", None)),
		),
		None,
	)
	max_spikes_per_unit = _parse_max_spikes_per_unit(
		wf_extract_cfg.get("max_spikes_per_unit", legacy_waveforms_cfg.get("max_spikes_per_unit", None))
	)

	return WaveformExtractionConfig(
		ms_before=ms_before,
		ms_after=ms_after,
		max_spikes_per_unit=max_spikes_per_unit,
	)


def _build_quality_checks_config(raw_cfg: dict[str, Any]) -> QualityChecksConfig:
	multiple_peaks_cfg = _get_multiple_peaks_block(raw_cfg)
	enabled_raw = raw_cfg.get("enable", raw_cfg.get("enabled", False))
	suppress_warnings_raw = raw_cfg.get("suppress_warnings", raw_cfg.get("surpress_warnings", False))
	multiple_enabled_raw = multiple_peaks_cfg.get("enable", multiple_peaks_cfg.get("enabled", None))
	if multiple_enabled_raw is None:
		multiple_enabled = bool(_as_bool(enabled_raw, False))
	else:
		multiple_enabled = bool(_as_bool(multiple_enabled_raw, False))
	return QualityChecksConfig(
		enable=_as_bool(enabled_raw, False),
		suppress_warnings=_as_bool(suppress_warnings_raw, False),
		check_for_multiple_peaks_at_channel_templates=MultipleNegativePeaksCheckConfig(
			enable=multiple_enabled,
			prominence_fraction=max(0.0, _as_float(multiple_peaks_cfg.get("prominence_fraction", 0.30), 0.30)),
			min_separation_samples=max(1, _as_int(multiple_peaks_cfg.get("min_separation_samples", 8), 8)),
			max_peaks_per_channel=max(2, _as_int(multiple_peaks_cfg.get("max_peaks_per_channel", 2), 2)),
		),
	)


def _get_multiple_peaks_block(raw_cfg: dict[str, Any]) -> dict[str, Any]:
	check_keys = (
		"check_for_multiple_peaks_at_channel_templates",
		"multiple_peaks_at_channel_templates",
		"multiple_negative_peaks",
	)
	for key in check_keys:
		block = _get_nested_block(raw_cfg, key)
		if block:
			return block
	return {}


def _build_per_unit_quality_checks_outputs_config(raw_cfg: dict[str, Any]) -> PerUnitQualityChecksOutputsConfig:
	multiple_cfg = _get_multiple_peaks_block(raw_cfg)
	plot_cfg = _get_nested_block(multiple_cfg, "plot")
	return PerUnitQualityChecksOutputsConfig(
		check_for_multiple_peaks_at_channel_templates=MultipleNegativePeaksOutputsConfig(
			write_json=_as_bool(multiple_cfg.get("write_json", True), True),
			json_relpath=str(multiple_cfg.get("json_relpath", "quality_checks_multiple_negative_peaks.json")),
			plot=QualityCheckPlotOutputConfig(
				write_png=_as_bool(plot_cfg.get("write_png", True), True),
				write_svg=_as_bool(plot_cfg.get("write_svg", False), False),
				relpath=str(plot_cfg.get("relpath", "multiple_peaks_at_channel_templates")),
				show_multiple_peak_markers=_as_bool(plot_cfg.get("show_multiple_peak_markers", False), False),
				delay_peak_marker_color=str(plot_cfg.get("delay_peak_marker_color", "black")),
			),
		),
	)


def _build_data_quality_checks_outputs_config(raw_cfg: dict[str, Any]) -> DataQualityChecksOutputsConfig:
	multiple_cfg = _get_multiple_peaks_block(raw_cfg)
	return DataQualityChecksOutputsConfig(
		check_for_multiple_peaks_at_channel_templates=QualityCheckJsonOutputConfig(
			write_json=_as_bool(multiple_cfg.get("write_json", True), True),
			json_relpath=str(multiple_cfg.get("json_relpath", "quality_checks_multiple_negative_peaks.json")),
		),
	)


def _build_topographical_footprint_config(raw_cfg: dict[str, Any], *, relpath_default: str) -> TopographicalFootprintConfig:
	template_cfg = _get_nested_block(raw_cfg, "template")
	color_bar_cfg = _get_nested_block(raw_cfg, "color_bar")
	return TopographicalFootprintConfig(
		write_png=_as_bool(_nested_or_flat(raw_cfg, block="output", key="write_png", flat_keys=("write_png",), default=True), True),
		write_svg=_as_bool(_nested_or_flat(raw_cfg, block="output", key="write_svg", flat_keys=("write_svg",), default=False), False),
		relpath=str(_nested_or_flat(raw_cfg, block="output", key="relpath", flat_keys=("relpath",), default=relpath_default)),
		background=str(_nested_or_flat(raw_cfg, block="render", key="background", flat_keys=("background",), default="black")),
		color_map=str(_nested_or_flat(raw_cfg, block="render", key="color_map", flat_keys=("color_map",), default="viridis")),
		template_shape=_normalize_template_shape(template_cfg.get("shape", raw_cfg.get("template_shape", "square")), "square"),
		template_padding_value=_normalize_padding_value(template_cfg.get("padding_value", raw_cfg.get("template_padding_value", "zero"))),
		show_color_bar=_as_bool(color_bar_cfg.get("show", color_bar_cfg.get("show_color_bar", raw_cfg.get("show_color_bar", True))), True),
		color_bar_location=str(color_bar_cfg.get("location", color_bar_cfg.get("color_bar_location", raw_cfg.get("color_bar_location", "topright")))),
		color_bar_fontsize=_as_float(color_bar_cfg.get("fontsize", color_bar_cfg.get("color_bar_fontsize", raw_cfg.get("color_bar_fontsize", 6.0))), 6.0),
		color_bar_length_fraction=_as_float(color_bar_cfg.get("length_fraction", color_bar_cfg.get("color_bar_length_fraction", raw_cfg.get("color_bar_length_fraction", 0.3))), 0.3),
		color_bar_pad_fraction=_as_float(color_bar_cfg.get("pad_fraction", color_bar_cfg.get("color_bar_pad_fraction", raw_cfg.get("color_bar_pad_fraction", 0.02))), 0.02),
		force_low_value=_as_float_or_none(color_bar_cfg.get("force_low_value", raw_cfg.get("force_low_value", 0.0)), 0.0),
		force_high_value=_as_float_or_none(color_bar_cfg.get("force_high_value", raw_cfg.get("force_high_value", None)), None),
		scale=str(color_bar_cfg.get("scale", raw_cfg.get("scale", "linear"))),
		percentile_low=_as_float(color_bar_cfg.get("percentile_low", raw_cfg.get("percentile_low", 5.0)), 5.0),
		percentile_high_linear=_as_float(color_bar_cfg.get("percentile_high_linear", raw_cfg.get("percentile_high_linear", 99.0)), 99.0),
		percentile_high_log=_as_float(color_bar_cfg.get("percentile_high_log", raw_cfg.get("percentile_high_log", 99.5)), 99.5),
		knot_anchor_values=tuple(_as_tuple(color_bar_cfg.get("knot_anchor_values", raw_cfg.get("knot_anchor_values", (1.0, 10.0))), (1.0, 10.0))),
		knot_y1_min=_as_float(color_bar_cfg.get("knot_y1_min", raw_cfg.get("knot_y1_min", 0.02)), 0.02),
		knot_y1_max=_as_float(color_bar_cfg.get("knot_y1_max", raw_cfg.get("knot_y1_max", 0.90)), 0.90),
		knot_y2_min=_as_float(color_bar_cfg.get("knot_y2_min", raw_cfg.get("knot_y2_min", 0.07)), 0.07),
		knot_y2_max=_as_float(color_bar_cfg.get("knot_y2_max", raw_cfg.get("knot_y2_max", 0.98)), 0.98),
		knot_min_gap=_as_float(color_bar_cfg.get("knot_min_gap", raw_cfg.get("knot_min_gap", 0.05)), 0.05),
		linear_cap_rounding_mode=str(color_bar_cfg.get("linear_cap_rounding_mode", raw_cfg.get("linear_cap_rounding_mode", "ceil_step"))),
		linear_cap_rounding_step=_as_float(color_bar_cfg.get("linear_cap_rounding_step", raw_cfg.get("linear_cap_rounding_step", 10.0)), 10.0),
		linear_cap_min_vmax=_as_float(color_bar_cfg.get("linear_cap_min_vmax", raw_cfg.get("linear_cap_min_vmax", 11.0)), 11.0),
		show_ticks=tuple(_as_tuple(color_bar_cfg.get("show_ticks", raw_cfg.get("show_ticks", (1, 10, "dynamic_high"))), (1, 10, "dynamic_high"))),
		elevation_deg=_as_float(_nested_or_flat(raw_cfg, block="display", key="elevation_deg", flat_keys=("elevation_deg",), default=35.0), 35.0),
		azimuth_deg=_as_float(_nested_or_flat(raw_cfg, block="display", key="azimuth_deg", flat_keys=("azimuth_deg",), default=-60.0), -60.0),
		marker_size=_as_float(_nested_or_flat(raw_cfg, block="display", key="marker_size", flat_keys=("marker_size",), default=14.0), 14.0),
	)


def _build_propagation_axes_config(raw_cfg: dict[str, Any]) -> PropagationAxesConfig:
	return PropagationAxesConfig(
		show=_as_bool(raw_cfg.get("show", True), True),
		xlabel=str(raw_cfg.get("xlabel", "x (um)")),
		ylabel=str(raw_cfg.get("ylabel", "y (um)")),
		label_fontsize=_as_float(raw_cfg.get("label_fontsize", 6.0), 6.0),
		tick_fontsize=_as_float(raw_cfg.get("tick_fontsize", 5.0), 5.0),
	)


def _build_propagation_latency_map_config(raw_cfg: dict[str, Any]) -> PropagationLatencyMapConfig:
	axes_cfg = _get_nested_block(raw_cfg, "axes")
	color_bar_cfg = _get_nested_block(raw_cfg, "color_bar")
	return PropagationLatencyMapConfig(
		show=_as_bool(raw_cfg.get("show", True), True),
		color_map=str(raw_cfg.get("color_map", "viridis")),
		force_square_aspect=_as_bool(raw_cfg.get("force_square_aspect", True), True),
		title=str(raw_cfg.get("title", "Latency Map")),
		fontsize=_as_float(raw_cfg.get("fontsize", 6.0), 6.0),
		template_shape=_normalize_template_shape(
			raw_cfg.get("template_shape", raw_cfg.get("template", "top_channels_only")),
			"top_channels_only",
		),
		show_color_bar=_as_bool(color_bar_cfg.get("show", color_bar_cfg.get("show_color_bar", raw_cfg.get("show_color_bar", True))), True),
		color_bar_location=str(color_bar_cfg.get("location", color_bar_cfg.get("color_bar_location", raw_cfg.get("color_bar_location", "topright")))),
		color_bar_fontsize=_as_float(color_bar_cfg.get("fontsize", color_bar_cfg.get("color_bar_fontsize", raw_cfg.get("color_bar_fontsize", 6.0))), 6.0),
		color_bar_length_fraction=_as_float(color_bar_cfg.get("length_fraction", color_bar_cfg.get("color_bar_length_fraction", raw_cfg.get("color_bar_length_fraction", 0.3))), 0.3),
		color_bar_pad_fraction=_as_float(color_bar_cfg.get("pad_fraction", color_bar_cfg.get("color_bar_pad_fraction", raw_cfg.get("color_bar_pad_fraction", 0.02))), 0.02),
		force_low_value=_as_float_or_none(color_bar_cfg.get("force_low_value", raw_cfg.get("force_low_value", 0.0)), 0.0),
		force_high_value=_as_float_or_none(color_bar_cfg.get("force_high_value", raw_cfg.get("force_high_value", None)), None),
		scale=str(color_bar_cfg.get("scale", raw_cfg.get("scale", "linear"))),
		percentile_low=_as_float(color_bar_cfg.get("percentile_low", raw_cfg.get("percentile_low", 5.0)), 5.0),
		percentile_high_linear=_as_float(color_bar_cfg.get("percentile_high_linear", raw_cfg.get("percentile_high_linear", 99.0)), 99.0),
		percentile_high_log=_as_float(color_bar_cfg.get("percentile_high_log", raw_cfg.get("percentile_high_log", 99.5)), 99.5),
		knot_anchor_values=tuple(_as_tuple(color_bar_cfg.get("knot_anchor_values", raw_cfg.get("knot_anchor_values", (1.0, 10.0))), (1.0, 10.0))),
		knot_y1_min=_as_float(color_bar_cfg.get("knot_y1_min", raw_cfg.get("knot_y1_min", 0.02)), 0.02),
		knot_y1_max=_as_float(color_bar_cfg.get("knot_y1_max", raw_cfg.get("knot_y1_max", 0.90)), 0.90),
		knot_y2_min=_as_float(color_bar_cfg.get("knot_y2_min", raw_cfg.get("knot_y2_min", 0.07)), 0.07),
		knot_y2_max=_as_float(color_bar_cfg.get("knot_y2_max", raw_cfg.get("knot_y2_max", 0.98)), 0.98),
		knot_min_gap=_as_float(color_bar_cfg.get("knot_min_gap", raw_cfg.get("knot_min_gap", 0.05)), 0.05),
		linear_cap_rounding_mode=str(color_bar_cfg.get("linear_cap_rounding_mode", raw_cfg.get("linear_cap_rounding_mode", "ceil_step"))),
		linear_cap_rounding_step=_as_float(color_bar_cfg.get("linear_cap_rounding_step", raw_cfg.get("linear_cap_rounding_step", 10.0)), 10.0),
		linear_cap_min_vmax=_as_float(color_bar_cfg.get("linear_cap_min_vmax", raw_cfg.get("linear_cap_min_vmax", 11.0)), 11.0),
		show_ticks=tuple(_as_tuple(color_bar_cfg.get("show_ticks", raw_cfg.get("show_ticks", (1, 10, "dynamic_high"))), (1, 10, "dynamic_high"))),
		axes=_build_propagation_axes_config(axes_cfg),
	)


def _build_template_artifact_config(
	raw_cfg: dict[str, Any],
	*,
	relpath_default: str,
	channel_locations_relpath_default: str | None,
) -> TemplateArtifactConfig:
	if "channel_locations_npy_relpath" in raw_cfg:
		loc_relpath_raw = raw_cfg.get("channel_locations_npy_relpath", None)
	else:
		loc_relpath_raw = channel_locations_relpath_default
	return TemplateArtifactConfig(
		write_npy=_as_bool(raw_cfg.get("write_npy", False), False),
		npy_relpath=str(raw_cfg.get("npy_relpath", relpath_default)),
		channel_locations_npy_relpath=(None if loc_relpath_raw in {None, ""} else str(loc_relpath_raw)),
		padding_value=_normalize_padding_value(raw_cfg.get("padding_value", "zero")),
	)


def _get_unit_reldir(runtime_config: RuntimeConfig) -> str:
	raw = None
	for path in (
		*_phase_per_unit_output_paths("unit_reldir"),
		*_output_paths("per_unit_outputs.unit_reldir"),
	):
		raw = runtime_config.get(path, None)
		if raw is not None and str(raw).strip() != "":
			break
	if raw is not None and str(raw).strip() != "":
		return str(raw)

	legacy = runtime_config.get("stages.reconstruct.outputs.per_unit_outputs.unit_reldir", None)
	if legacy is not None and str(legacy).strip() != "":
		return str(legacy)

	return "units/{unit_id:04d}/"


@dataclass(frozen=True)
class TemplatesStageConfig:
	output_rel_root: str
	analyzer_cache: AnalyzerCacheConfig
	per_unit_outputs: PerUnitTemplatesOutputsConfig
	reports: ReportsConfig
	quality_checks_outputs: DataQualityChecksOutputsConfig
	resolve_sources_phase: ResolveSourcesPhaseConfig
	phases: TemplatesPhasesConfig
	concat_analyzer_relpath: str | None
	concat_sorting_relpath: str | None
	preprocessed_concat_reldir: str | None
	preprocessed_segments_reldir: str | None
	preproc_seg_sources_reldir: str | None
	unit_ids: list[int] | None
	unit_limit: int | None
	force_restart: bool
	force_replot: bool
	force_replot_per_unit: bool
	force_rereport: bool
	require_curated_units: bool
	include_concat: bool
	include_segments: bool
	require_concat_analyzer: bool
	require_segment_analyzers: bool
	waveform_extraction: WaveformExtractionConfig
	execution_upsampling: TimeUpsampleConfig
	merge: MergeConfig
	quality_checks: QualityChecksConfig
	probe_geometry: ProbeGeometryConfig | None = None


def _normalize_optional_path_token(raw: Any) -> str | None:
	if raw is None:
		return None
	token = str(raw).strip()
	if not token:
		return None
	return token


def parse_probe_geometry_from_data_config(*, data_config: RuntimeConfig) -> ProbeGeometryConfig | None:
	default_pitch_um = 17.5
	default_electrode_size_um_x = 12.0
	default_electrode_size_um_y = 8.8
	default_active_area_um_x = 3850.0
	default_active_area_um_y = 2100.0

	probe_raw = data_config.get("Probe", {})
	probe_cfg = probe_raw if isinstance(probe_raw, dict) else {}

	pitch_um = _as_float_or_none(probe_cfg.get("pitch_um", None), default_pitch_um)

	elec_cfg = probe_cfg.get("electrode_size_um", {}) if isinstance(probe_cfg.get("electrode_size_um", {}), dict) else {}
	electrode_size_um_x = _as_float_or_none(elec_cfg.get("x", None), None)
	electrode_size_um_y = _as_float_or_none(elec_cfg.get("y", None), None)
	if electrode_size_um_x is None:
		electrode_size_um_x = default_electrode_size_um_x
	if electrode_size_um_y is None:
		electrode_size_um_y = default_electrode_size_um_y

	chip_dims_um_cfg = probe_cfg.get("chip_dimensions_um", {}) if isinstance(probe_cfg.get("chip_dimensions_um", {}), dict) else {}
	chip_dims_mm_cfg = probe_cfg.get("chip_dimensions_mm", {}) if isinstance(probe_cfg.get("chip_dimensions_mm", {}), dict) else {}
	active_cfg = probe_cfg.get("active_sensing_area_mm", {}) if isinstance(probe_cfg.get("active_sensing_area_mm", {}), dict) else {}

	active_area_um_x = _as_float_or_none(chip_dims_um_cfg.get("x", None), None)
	active_area_um_y = _as_float_or_none(chip_dims_um_cfg.get("y", None), None)

	if active_area_um_x is None:
		active_area_um_x = _as_float_or_none(chip_dims_mm_cfg.get("x", None), None)
		if active_area_um_x is not None:
			active_area_um_x *= 1000.0
	if active_area_um_y is None:
		active_area_um_y = _as_float_or_none(chip_dims_mm_cfg.get("y", None), None)
		if active_area_um_y is not None:
			active_area_um_y *= 1000.0

	if active_area_um_x is None:
		active_area_um_x = _as_float_or_none(active_cfg.get("x", None), None)
		if active_area_um_x is not None:
			active_area_um_x *= 1000.0
	if active_area_um_y is None:
		active_area_um_y = _as_float_or_none(active_cfg.get("y", None), None)
		if active_area_um_y is not None:
			active_area_um_y *= 1000.0

	if active_area_um_x is None:
		active_area_um_x = default_active_area_um_x
	if active_area_um_y is None:
		active_area_um_y = default_active_area_um_y

	sampling_rate_hz = _as_float_or_none(probe_cfg.get("sampling_rate_hz", None), None)

	return ProbeGeometryConfig(
		pitch_um=pitch_um,
		electrode_size_um_x=electrode_size_um_x,
		electrode_size_um_y=electrode_size_um_y,
		active_area_um_x=active_area_um_x,
		active_area_um_y=active_area_um_y,
		sampling_rate_hz=sampling_rate_hz,
	)


def _phase_block(raw_cfg: dict[str, Any], *path: str) -> dict[str, Any]:
	block: Any = raw_cfg
	for key in path:
		if not isinstance(block, dict):
			return {}
		next_block = block.get(key, {})
		if not isinstance(next_block, dict):
			return {}
		block = next_block
	return (dict(block) if isinstance(block, dict) else {})


def _normalize_sparsity_mode(raw: Any, default: str = "inherit") -> str:
	value = str(raw or default).strip().lower().replace("-", "_").replace(" ", "_")
	if value in {"dense", "dense_no_mask", "no_mask", "full"}:
		return "dense"
	if value in {"inherit", "default", "existing", "cached"}:
		return "inherit"
	return str(default)


def _normalize_random_spikes_method(raw: Any, default: str = "uniform") -> str:
	value = str(raw or default).strip().lower().replace("-", "_").replace(" ", "_")
	if value in {"percentage", "percent", "fraction", "proportion"}:
		return "percentage"
	if value in {"all", "full", "all_spikes"}:
		return "all"
	if value in {"uniform", "default", "sample"}:
		return "uniform"
	return str(default)


def _parse_random_spikes_percentage(raw: Any, *, field_name: str) -> float | None:
	if raw is None:
		return None
	if isinstance(raw, bool):
		raise ValueError(f"{field_name} must be a number in the interval (0, 100]")

	is_percent_token = False
	if isinstance(raw, str):
		text = str(raw).strip()
		if not text:
			return None
		if text.endswith("%"):
			is_percent_token = True
			text = text[:-1].strip()
		try:
			value = float(text)
		except Exception as exc:
			raise ValueError(f"{field_name} must be numeric") from exc
	else:
		try:
			value = float(raw)
		except Exception as exc:
			raise ValueError(f"{field_name} must be numeric") from exc

	if (not math.isfinite(value)) or value <= 0.0:
		raise ValueError(f"{field_name} must be in the interval (0, 100]")
	if is_percent_token or value > 1.0:
		if value > 100.0:
			raise ValueError(f"{field_name} must be in the interval (0, 100]")
		return float(value) / 100.0
	return float(value)


def _normalize_sparsity_method(raw: Any, default: str = "radius") -> str:
	value = str(raw or default).strip().lower().replace("-", "_").replace(" ", "_")
	if value in {"best", "best_channel", "best_channels", "num_channels"}:
		return "best_channels"
	if value in {"threshold", "snr"}:
		return "threshold"
	if value in {"by_property", "property", "group"}:
		return "by_property"
	return str(default)


def _normalize_peak_sign(raw: Any, default: str = "neg") -> str:
	value = str(raw or default).strip().lower()
	if value in {"pos", "positive"}:
		return "pos"
	if value in {"both", "all"}:
		return "both"
	return str(default)


def _build_analyzer_preparation_policy_config(
	*,
	raw_cfg: dict[str, Any],
	default_waveform_extraction: WaveformExtractionConfig,
	default_policy: AnalyzerPreparationPolicyConfig | None = None,
) -> AnalyzerPreparationPolicyConfig:
	policy_defaults = default_policy or AnalyzerPreparationPolicyConfig(
		ms_before=default_waveform_extraction.ms_before,
		ms_after=default_waveform_extraction.ms_after,
		max_spikes_per_unit=default_waveform_extraction.max_spikes_per_unit,
	)
	policy_cfg = _get_nested_block(raw_cfg, "policy")
	waveform_cfg = _get_nested_block(raw_cfg, "waveform_extraction")
	waveforms_cfg = _get_nested_block(raw_cfg, "waveforms")
	template_extraction_cfg = _get_nested_block(raw_cfg, "template_extraction")
	sparsity_cfg = _get_nested_block(raw_cfg, "sparsity")
	waveform_window_cfg = _get_nested_block(waveform_cfg, "window")
	waveforms_window_cfg = _get_nested_block(waveforms_cfg, "window")
	ms_before = _as_float_or_none(
		waveforms_window_cfg.get(
			"ms_before",
			waveforms_cfg.get(
				"ms_before",
				waveform_window_cfg.get(
					"ms_before",
					waveform_cfg.get("ms_before", policy_cfg.get("ms_before", policy_defaults.ms_before)),
				),
			),
		),
		policy_defaults.ms_before,
	)
	ms_after = _as_float_or_none(
		waveforms_window_cfg.get(
			"ms_after",
			waveforms_cfg.get(
				"ms_after",
				waveform_window_cfg.get(
					"ms_after",
					waveform_cfg.get("ms_after", policy_cfg.get("ms_after", policy_defaults.ms_after)),
				),
			),
		),
		policy_defaults.ms_after,
	)
	max_spikes_per_unit = _parse_max_spikes_per_unit(
		template_extraction_cfg.get(
			"max_spikes_per_unit",
			waveform_cfg.get(
				"max_spikes_per_unit",
				policy_cfg.get("max_spikes_per_unit", policy_defaults.max_spikes_per_unit),
			),
		)
	)
	min_spikes_per_unit = _parse_optional_positive_int(
		template_extraction_cfg.get(
			"min_spikes_per_unit",
			policy_cfg.get("min_spikes_per_unit", policy_defaults.min_spikes_per_unit),
		)
	)
	random_spikes_percentage_raw = template_extraction_cfg.get(
		"random_spikes_percentage",
		template_extraction_cfg.get(
			"min_perc_spikes_per_unit",
			policy_cfg.get("random_spikes_percentage", policy_defaults.random_spikes_percentage),
		),
	)
	random_spikes_percentage = _parse_random_spikes_percentage(
		random_spikes_percentage_raw,
		field_name="random_spikes_percentage",
	)
	random_seed_raw = template_extraction_cfg.get(
		"random_seed",
		policy_cfg.get("random_seed", raw_cfg.get("random_seed", policy_defaults.random_seed)),
	)
	if random_seed_raw in {None, ""}:
		random_seed = None
	else:
		random_seed = _as_int(random_seed_raw, 0)
	compute_sparsity_raw = sparsity_cfg.get(
		"compute_sparsity",
		policy_cfg.get("compute_sparsity", raw_cfg.get("compute_sparsity", None)),
	)
	legacy_density_mode = template_extraction_cfg.get("density_mode", None)
	has_explicit_sparsity_mode = (
		policy_cfg.get("sparsity_mode") not in {None, ""}
		or raw_cfg.get("sparsity_mode") not in {None, ""}
		or legacy_density_mode not in {None, ""}
	)
	normalized_sparsity_mode = _normalize_sparsity_mode(
		policy_cfg.get("sparsity_mode", raw_cfg.get("sparsity_mode", policy_defaults.sparsity_mode)),
		default=policy_defaults.sparsity_mode,
	)
	if compute_sparsity_raw is None:
		compute_sparsity = (False if _normalize_sparsity_mode(legacy_density_mode, normalized_sparsity_mode) == "dense" else normalized_sparsity_mode != "dense")
	else:
		compute_sparsity = _as_bool(compute_sparsity_raw, policy_defaults.compute_sparsity)
		if compute_sparsity and (not has_explicit_sparsity_mode) and normalized_sparsity_mode == "dense":
			normalized_sparsity_mode = "inherit"
	sparsity_mode = ("dense" if not compute_sparsity else normalized_sparsity_mode)
	dtype = _parse_optional_text(
		waveforms_cfg.get("dtype", policy_cfg.get("dtype", policy_defaults.dtype))
	)
	return AnalyzerPreparationPolicyConfig(
		ms_before=ms_before,
		ms_after=ms_after,
		dtype=dtype,
		max_spikes_per_unit=max_spikes_per_unit,
		min_spikes_per_unit=min_spikes_per_unit,
		random_spikes_percentage=random_spikes_percentage,
		log_before_after_spike_counts=_as_bool(
			template_extraction_cfg.get(
				"log_before_after_spike_counts",
				policy_cfg.get(
					"log_before_after_spike_counts",
					policy_defaults.log_before_after_spike_counts,
				),
			),
			policy_defaults.log_before_after_spike_counts,
		),
		margin_size=_parse_optional_positive_int(
			template_extraction_cfg.get(
				"margin_size",
				policy_cfg.get("margin_size", policy_defaults.margin_size),
			),
		),
		sparsity_mode=sparsity_mode,
		compute_sparsity=bool(compute_sparsity),
		sparsity_method=_normalize_sparsity_method(
			sparsity_cfg.get(
				"method",
				policy_cfg.get("sparsity_method", policy_defaults.sparsity_method),
			),
			default=policy_defaults.sparsity_method,
		),
		sparsity_radius_um=_as_float_or_none(
			sparsity_cfg.get("radius_um", policy_cfg.get("sparsity_radius_um", policy_defaults.sparsity_radius_um)),
			policy_defaults.sparsity_radius_um,
		),
		sparsity_num_channels=_parse_optional_positive_int(
			sparsity_cfg.get(
				"num_channels",
				policy_cfg.get("sparsity_num_channels", policy_defaults.sparsity_num_channels),
			),
		),
		sparsity_threshold=_as_float_or_none(
			sparsity_cfg.get("threshold", policy_cfg.get("sparsity_threshold", policy_defaults.sparsity_threshold)),
			policy_defaults.sparsity_threshold,
		),
		sparsity_peak_sign=_normalize_peak_sign(
			sparsity_cfg.get(
				"peak_sign",
				policy_cfg.get("sparsity_peak_sign", policy_defaults.sparsity_peak_sign),
			),
			default=policy_defaults.sparsity_peak_sign,
		),
		sparsity_num_spikes_for_sparsity=_parse_optional_positive_int(
			sparsity_cfg.get(
				"num_spikes_for_sparsity",
				policy_cfg.get(
					"sparsity_num_spikes_for_sparsity",
					policy_defaults.sparsity_num_spikes_for_sparsity,
				),
			),
		),
		sparsity_by_property=_parse_optional_text(
			sparsity_cfg.get(
				"by_property",
				policy_cfg.get("sparsity_by_property", policy_defaults.sparsity_by_property),
			),
		),
		random_spikes_method=_normalize_random_spikes_method(
			template_extraction_cfg.get(
				"random_spikes_method",
				policy_cfg.get(
				"random_spikes_method",
				raw_cfg.get("random_spikes_method", policy_defaults.random_spikes_method),
				),
			),
			default=policy_defaults.random_spikes_method,
		),
		random_seed=random_seed,
		n_jobs=_parse_optional_positive_int(raw_cfg.get("n_jobs", policy_defaults.n_jobs)),
		chunk_duration=_parse_optional_text(raw_cfg.get("chunk_duration", policy_defaults.chunk_duration)),
	)


def _build_analyzer_source_phase_config(
	*,
	source_cfg: dict[str, Any],
	defaults_cfg: dict[str, Any],
	default_enabled: bool,
	default_required: bool,
	default_analyzer_relpath: str | None,
	default_sorting_relpath: str | None,
	default_preprocessed_recording_reldir: str | None,
	default_preprocessed_sources_reldir: str | None,
	default_waveform_extraction: WaveformExtractionConfig,
) -> AnalyzerSourcePhaseConfig:
	default_policy = _build_analyzer_preparation_policy_config(
		raw_cfg=defaults_cfg,
		default_waveform_extraction=default_waveform_extraction,
	)
	policy = _build_analyzer_preparation_policy_config(
		raw_cfg=source_cfg,
		default_waveform_extraction=default_waveform_extraction,
		default_policy=default_policy,
	)
	return AnalyzerSourcePhaseConfig(
		enabled=_as_bool(source_cfg.get("enabled", defaults_cfg.get("enabled", default_enabled)), default_enabled),
		required=_as_bool(source_cfg.get("required", defaults_cfg.get("required", default_required)), default_required),
		use_existing_analyzer=_as_bool(
			source_cfg.get("use_existing_analyzer", defaults_cfg.get("use_existing_analyzer", True)),
			True,
		),
		build_if_missing=_as_bool(
			source_cfg.get("build_if_missing", defaults_cfg.get("build_if_missing", True)),
			True,
		),
		analyzer_relpath=_normalize_optional_path_token(
			source_cfg.get("analyzer_relpath", defaults_cfg.get("analyzer_relpath", default_analyzer_relpath))
		),
		sorting_relpath=_normalize_optional_path_token(
			source_cfg.get("sorting_relpath", defaults_cfg.get("sorting_relpath", default_sorting_relpath))
		),
		preprocessed_recording_reldir=_normalize_optional_path_token(
			source_cfg.get(
				"preprocessed_recording_reldir",
				defaults_cfg.get("preprocessed_recording_reldir", default_preprocessed_recording_reldir),
			)
		),
		preprocessed_sources_reldir=_normalize_optional_path_token(
			source_cfg.get(
				"preprocessed_sources_reldir",
				defaults_cfg.get("preprocessed_sources_reldir", default_preprocessed_sources_reldir),
			)
		),
		policy=policy,
	)


def parse_templates_stage_config(
	*,
	runtime_config: RuntimeConfig,
	probe_geometry: ProbeGeometryConfig | None = None,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> TemplatesStageConfig:
	stage_cfg = runtime_config.get("stages.templates", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	phases_cfg = stage_cfg.get("phases", {}) if isinstance(stage_cfg.get("phases", {}), dict) else {}
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}
	if not outputs_cfg:
		outputs_cfg = runtime_config.get("stages.outputs", {}) if isinstance(runtime_config.get("stages.outputs", {}), dict) else {}
	analyzer_cache_cfg = _get_analyzer_cache_block(runtime_config)

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	force_replot_per_unit = _as_bool(execution_cfg.get("force_replot_per_unit", False), False)
	force_rereport = _as_bool(execution_cfg.get("force_rereport", False), False)
	require_curated_units = _as_bool(execution_cfg.get("require_curated_units", True), True)
	stage_inputs_cfg = stage_cfg.get("inputs", {}) if isinstance(stage_cfg.get("inputs", {}), dict) else {}
	execution_inputs_cfg = execution_cfg.get("inputs", {}) if isinstance(execution_cfg.get("inputs", {}), dict) else {}
	inputs_cfg = dict(execution_inputs_cfg)
	inputs_cfg.update(stage_inputs_cfg)
	concat_analyzer_relpath = _normalize_optional_path_token(inputs_cfg.get("concat_analyzer_relpath", None))
	concat_sorting_relpath = _normalize_optional_path_token(inputs_cfg.get("concat_sorting_relpath", None))
	preprocessed_concat_reldir = _normalize_optional_path_token(inputs_cfg.get("preprocessed_concat_reldir", None))
	preprocessed_segments_reldir = _normalize_optional_path_token(inputs_cfg.get("preprocessed_segments_reldir", None))
	if preprocessed_segments_reldir is None:
		preprocessed_segments_reldir = _normalize_optional_path_token(inputs_cfg.get("preproc_seg_sources_reldir", None))
	preproc_seg_sources_reldir = preprocessed_segments_reldir
	stage_spikeinterface_cfg = stage_cfg.get("spikeinterface", {}) if isinstance(stage_cfg.get("spikeinterface", {}), dict) else {}
	execution_spikeinterface_cfg = execution_cfg.get("spikeinterface", {}) if isinstance(execution_cfg.get("spikeinterface", {}), dict) else {}
	spk_tpl_sources = dict(execution_spikeinterface_cfg)
	spk_tpl_sources.update(stage_spikeinterface_cfg)
	spk_tpl_extract = spk_tpl_sources.get("template_extraction", {}) if isinstance(spk_tpl_sources.get("template_extraction", {}), dict) else {}
	spk_tpl_extract_sources = spk_tpl_extract.get("sources", {}) if isinstance(spk_tpl_extract.get("sources", {}), dict) else {}
	legacy_include_concat = _as_bool(spk_tpl_extract_sources.get("include_concat", True), True)
	legacy_include_segments = _as_bool(spk_tpl_extract_sources.get("include_segments", True), True)
	legacy_require_concat_analyzer = _as_bool(
		spk_tpl_extract_sources.get(
			"require_concat",
			spk_tpl_extract_sources.get("require_concat_analyzer", False),
		),
		False,
	)
	legacy_require_segment_analyzers = _as_bool(
		spk_tpl_extract_sources.get(
			"require_segments",
			spk_tpl_extract_sources.get("require_segment_analyzers", False),
		),
		False,
	)
	waveform_extraction = _build_waveform_extraction_config(
		spikeinterface_cfg=spk_tpl_sources,
		runtime_config=runtime_config,
	)
	analyzers_phase_cfg_raw = _phase_block(phases_cfg, "analyzers")
	analyzers_defaults_cfg = _phase_block(analyzers_phase_cfg_raw, "defaults")
	concat_phase_cfg = _build_analyzer_source_phase_config(
		source_cfg=_phase_block(analyzers_phase_cfg_raw, "concat"),
		defaults_cfg=analyzers_defaults_cfg,
		default_enabled=legacy_include_concat,
		default_required=legacy_require_concat_analyzer,
		default_analyzer_relpath=concat_analyzer_relpath,
		default_sorting_relpath=concat_sorting_relpath,
		default_preprocessed_recording_reldir=preprocessed_concat_reldir,
		default_preprocessed_sources_reldir=None,
		default_waveform_extraction=waveform_extraction,
	)
	segments_phase_cfg = _build_analyzer_source_phase_config(
		source_cfg=_phase_block(analyzers_phase_cfg_raw, "segments"),
		defaults_cfg=analyzers_defaults_cfg,
		default_enabled=legacy_include_segments,
		default_required=legacy_require_segment_analyzers,
		default_analyzer_relpath=None,
		default_sorting_relpath=concat_sorting_relpath,
		default_preprocessed_recording_reldir=None,
		default_preprocessed_sources_reldir=preprocessed_segments_reldir,
		default_waveform_extraction=waveform_extraction,
	)
	include_concat = bool(concat_phase_cfg.enabled)
	include_segments = bool(segments_phase_cfg.enabled)
	require_concat_analyzer = bool(concat_phase_cfg.required) if include_concat else False
	require_segment_analyzers = bool(segments_phase_cfg.required) if include_segments else False
	concat_analyzer_relpath = concat_phase_cfg.analyzer_relpath
	concat_sorting_relpath = concat_phase_cfg.sorting_relpath or concat_sorting_relpath
	preprocessed_concat_reldir = concat_phase_cfg.preprocessed_recording_reldir or preprocessed_concat_reldir
	preprocessed_segments_reldir = segments_phase_cfg.preprocessed_sources_reldir or preprocessed_segments_reldir
	preproc_seg_sources_reldir = preprocessed_segments_reldir
	stage_upsampling_cfg = stage_cfg.get("upsampling", {}) if isinstance(stage_cfg.get("upsampling", {}), dict) else {}
	execution_upsampling_cfg = execution_cfg.get("upsampling", {}) if isinstance(execution_cfg.get("upsampling", {}), dict) else {}
	phase_build_cfg = _phase_block(phases_cfg, "build_templates")
	if not phase_build_cfg:
		phase_build_cfg = _phase_block(phases_cfg, "per_unit_processing", "build_templates")
	phase_build_upsampling_cfg = _phase_block(phase_build_cfg, "execution_upsampling")
	if not phase_build_upsampling_cfg:
		phase_build_upsampling_cfg = _phase_block(phase_build_cfg, "upsampling")
	upsampling_cfg = dict(execution_upsampling_cfg)
	upsampling_cfg.update(stage_upsampling_cfg)
	upsampling_cfg.update(phase_build_upsampling_cfg)
	execution_upsampling = _build_time_upsample_config(upsampling_cfg)
	quality_checks_cfg_raw = execution_cfg.get("quality_checks", {}) if isinstance(execution_cfg.get("quality_checks", {}), dict) else {}
	stage_quality_checks_cfg = stage_cfg.get("quality_checks", {}) if isinstance(stage_cfg.get("quality_checks", {}), dict) else {}
	quality_checks_cfg_raw = dict(quality_checks_cfg_raw)
	quality_checks_cfg_raw.update(stage_quality_checks_cfg)
	phase_quality_checks_cfg = _phase_block(phases_cfg, "per_unit_processing", "quality_checks", "config")
	if not phase_quality_checks_cfg:
		phase_quality_checks_cfg = _phase_block(phases_cfg, "per_unit_processing", "quality_checks")
	quality_checks_cfg_raw.update(phase_quality_checks_cfg)
	quality_checks = _build_quality_checks_config(quality_checks_cfg_raw)
	analysis_cfg = execution_cfg.get("analysis", {}) if isinstance(execution_cfg.get("analysis", {}), dict) else {}
	stage_analysis_cfg = stage_cfg.get("analysis", {}) if isinstance(stage_cfg.get("analysis", {}), dict) else {}
	analysis_cfg = dict(analysis_cfg)
	analysis_cfg.update(stage_analysis_cfg)
	phase_analysis_cfg = _phase_block(phases_cfg, "per_unit_processing", "analysis")
	phase_prop_order_cfg = _phase_block(phase_analysis_cfg, "propagation_ordering")
	if phase_prop_order_cfg:
		analysis_cfg["propagation_ordering"] = dict(phase_prop_order_cfg)
	prop_order_analysis_cfg = analysis_cfg.get("propagation_ordering", {}) if isinstance(analysis_cfg.get("propagation_ordering", {}), dict) else {}
	prop_order_analysis_enabled = _as_bool(prop_order_analysis_cfg.get("enable", False), False)
	analysis_ordering_latency_mode = (
		str(prop_order_analysis_cfg.get("latency_mode", "abs_peak"))
		if prop_order_analysis_enabled
		else "abs_peak"
	)
	analysis_debug_ordering = (
		_as_bool(prop_order_analysis_cfg.get("debug", False), False)
		if prop_order_analysis_enabled
		else False
	)
	merge_cfg = _get_merge_block(runtime_config)
	resolve_sources_phase_cfg = _get_resolve_sources_phase_block(runtime_config)
	merge = MergeConfig(
		enable=_as_bool(merge_cfg.get("enable", True), True),
		method=str(merge_cfg.get("method", "mean_all_waveforms")),
		centering_method=str(merge_cfg.get("centering_method", "pre_peak_robust_baseline")),
		max_waveforms_per_source_channel=_parse_max_waveforms_per_source_channel(
			merge_cfg.get("max_waveforms_per_source_channel", 500),
			500,
		),
		overlap_match_priority=_normalize_overlap_priority(
			merge_cfg.get("overlap_match_priority", ("electrode_id", "channel_id", "location"))
		),
		location_tolerance_um=max(1e-6, _as_float(merge_cfg.get("location_tolerance_um", 1.0), 1.0)),
	)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if force_replot_override is not None:
		force_replot = bool(force_replot_override)
	if force_rereport:
		# Report-only reruns should not trigger per-unit regeneration.
		force_restart = False
		force_replot = False
		force_replot_per_unit = False

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

	tpl_cfg = _get_template_block(runtime_config)
	tpl_circles_cfg = _get_template_circles_block(runtime_config)
	tpl_wf_overlay_cfg = _get_template_wf_overlay_block(runtime_config)
	report_overlay_grid_cfg = _get_reports_wf_overlay_grid_block(runtime_config)
	report_locations_cfg = _get_reports_locations_block(runtime_config)
	reports_cfg = _get_reports_block(runtime_config)
	reports_grids_cfg = reports_cfg.get("grids", {}) if isinstance(reports_cfg.get("grids", {}), dict) else {}
	per_unit_quality_checks_cfg = _get_per_unit_quality_checks_block(runtime_config)
	data_quality_checks_cfg = _get_data_quality_checks_block(runtime_config)
	footprint_grids_cfg = _get_reports_footprint_grids_block(runtime_config)
	amp_map_cfg = _get_footprint_map_block(runtime_config, "amplitude_map")
	lat_map_cfg = _get_footprint_map_block(runtime_config, "latency_map")
	topo_amp_cfg = _get_topographical_footprint_block(runtime_config, "amplitude")
	topo_lat_cfg = _get_topographical_footprint_block(runtime_config, "latency")
	propagation_cfg = _get_propagation_plots_block(runtime_config)
	merged_template_cfg = _get_template_artifact_block(runtime_config, "merged_template")
	square_template_cfg = _get_template_artifact_block(runtime_config, "square_template")
	scan_template_cfg = _get_template_artifact_block(runtime_config, "scan_template")
	full_template_cfg = _get_template_artifact_block(runtime_config, "full_template")
	time_upsample_cfg_raw = reports_cfg.get("time_upsample", {}) if isinstance(reports_cfg.get("time_upsample", {}), dict) else {}
	if not time_upsample_cfg_raw:
		time_upsample_cfg_raw = report_overlay_grid_cfg.get("time_upsample", {}) if isinstance(report_overlay_grid_cfg.get("time_upsample", {}), dict) else {}
	tpl = TemplatePlotConfig(
		write_png=_as_bool(_nested_or_flat(tpl_cfg, block="output", key="write_png", flat_keys=("write_png",), default=True), True),
		write_svg=_as_bool(_nested_or_flat(tpl_cfg, block="output", key="write_svg", flat_keys=("write_svg",), default=False), False),
		dpi=max(72.0, _as_float(_nested_or_flat(tpl_cfg, block="output", key="dpi", flat_keys=("dpi",), default=300.0), 300.0)),
		relpath=str(_nested_or_flat(tpl_cfg, block="output", key="relpath", flat_keys=("relpath",), default="template")),
		channel_scope=_normalize_channel_scope(_nested_or_flat(tpl_cfg, block="display", key="channel_scope", flat_keys=("channel_scope",), default="contributing_channels")),
		background=str(_nested_or_flat(tpl_cfg, block="render", key="background", flat_keys=("background",), default="black")),
		signal_color=str(_nested_or_flat(tpl_cfg, block="render", key="signal_color", flat_keys=("signal_color",), default="white")),
		force_center_soma=_as_bool(_nested_or_flat(tpl_cfg, block="display", key="force_center_soma", flat_keys=("force_center_soma",), default=False), False),
		force_square_aspect=_as_bool(_nested_or_flat(tpl_cfg, block="display", key="force_square_aspect", flat_keys=("force_square_aspect",), default=True), True),
		show_scale_bar=_as_bool(_nested_or_flat(tpl_cfg, block="display", key="show_scale_bar", flat_keys=("show_scale_bar",), default=True), True),
		scale_bar_color=str(_nested_or_flat(tpl_cfg, block="render", key="scale_bar_color", flat_keys=("scale_bar_color",), default="white")),
		scale_bar_text_offset_frac=_as_float(
			_nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="text_offset_frac",
				flat_keys=("scale_bar_text_offset_frac",),
				default=0.02,
			),
			0.02,
		),
		scale_bar_x_offset_frac=(
			None
			if _nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="x_offset_frac",
				flat_keys=("scale_bar_x_offset_frac",),
				default=None,
			)
			is None
			else _as_float(
				_nested_path_or_flat(
					tpl_cfg,
					path=("display", "scale_bar"),
					key="x_offset_frac",
					flat_keys=("scale_bar_x_offset_frac",),
					default=None,
				),
				0.02,
			)
		),
		scale_bar_x_offset_considers_fontsize=_as_bool(
			_nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="x_offset_considers_fontsize",
				flat_keys=("scale_bar_x_offset_considers_fontsize",),
				default=False,
			),
			False,
		),
		scale_bar_horizontal_alignment=_normalize_horizontal_alignment(
			_nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="horizontal_alignment",
				flat_keys=("scale_bar_horizontal_alignment",),
				default="right",
			),
			default="right",
		),
		scale_bar_vertical_alignment=_normalize_vertical_alignment(
			_nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="vertical_alignment",
				flat_keys=("scale_bar_vertical_alignment",),
				default="bottom",
			),
			default="bottom",
		),
		scale_bar_y_offset_frac=_as_float(
			_nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="y_offset_frac",
				flat_keys=("scale_bar_y_offset_frac",),
				default=0.06,
			),
			0.06,
		),
		scale_bar_fontsize=_as_float(
			_nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="fontsize",
				flat_keys=("scale_bar_fontsize",),
				default=6.0,
			),
			6.0,
		),
		scale_bar_linewidth=_as_float(
			_nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="linewidth",
				flat_keys=("scale_bar_linewidth",),
				default=1.8,
			),
			1.8,
		),
		scale_bar_length_um=(
			None
			if _nested_path_or_flat(
				tpl_cfg,
				path=("display", "scale_bar"),
				key="length_um",
				flat_keys=("scale_bar_length_um",),
				default=None,
			)
			is None
			else _as_float(
				_nested_path_or_flat(
					tpl_cfg,
					path=("display", "scale_bar"),
					key="length_um",
					flat_keys=("scale_bar_length_um",),
					default=None,
				),
				0.0,
			)
		),
		show_axes=_as_bool(_nested_or_flat(tpl_cfg, block="display", key="show_axes", flat_keys=("show_axes",), default=True), True),
		unit_id_label=UnitIdLabelConfig(
			show=_as_bool(_get_nested_block(tpl_cfg, "unit_id_label").get("show", False), False),
			fontsize=_as_float(_get_nested_block(tpl_cfg, "unit_id_label").get("fontsize", 12.0), 12.0),
			color=str(_get_nested_block(tpl_cfg, "unit_id_label").get("color", "white")),
			x_offset_frac=_as_float(_get_nested_block(tpl_cfg, "unit_id_label").get("x_offset_frac", 0.02), 0.02),
			y_offset_frac=_as_float(_get_nested_block(tpl_cfg, "unit_id_label").get("y_offset_frac", 0.02), 0.02),
			horizontal_alignment=_normalize_horizontal_alignment(
				_get_nested_block(tpl_cfg, "unit_id_label").get("horizontal_alignment", "right"),
				default="right",
			),
			vertical_alignment=_normalize_vertical_alignment(
				_get_nested_block(tpl_cfg, "unit_id_label").get("vertical_alignment", "top"),
				default="top",
			),
		),
		center_most_channel_coords=CenterMostChannelCoordsConfig(
			show=_as_bool(_get_nested_block(tpl_cfg, "center_most_channel_coords").get("show", False), False),
			fontsize=_as_float(_get_nested_block(tpl_cfg, "center_most_channel_coords").get("fontsize", 10.0), 10.0),
			color=str(_get_nested_block(tpl_cfg, "center_most_channel_coords").get("color", "white")),
			x_offset_frac=_as_float(_get_nested_block(tpl_cfg, "center_most_channel_coords").get("x_offset_frac", 0.02), 0.02),
			y_offset_frac=_as_float(_get_nested_block(tpl_cfg, "center_most_channel_coords").get("y_offset_frac", 0.01), 0.01),
			horizontal_alignment=_normalize_horizontal_alignment(
				_get_nested_block(tpl_cfg, "center_most_channel_coords").get("horizontal_alignment", "left"),
				default="left",
			),
			vertical_alignment=_normalize_vertical_alignment(
				_get_nested_block(tpl_cfg, "center_most_channel_coords").get("vertical_alignment", "top"),
				default="top",
			),
		),
	)
	tpl_circles = TemplateCirclesPlotConfig(
		# Keep circles-specific nested config under color_bar for runtime ergonomics.
		# Example:
		# circles:
		#   color_bar:
		#     units: ms
		write_png=_as_bool(_nested_or_flat(tpl_circles_cfg, block="output", key="write_png", flat_keys=("write_png",), default=False), False),
		write_svg=_as_bool(_nested_or_flat(tpl_circles_cfg, block="output", key="write_svg", flat_keys=("write_svg",), default=False), False),
		dpi=max(72.0, _as_float(_nested_or_flat(tpl_circles_cfg, block="output", key="dpi", flat_keys=("dpi",), default=tpl.dpi), 300.0)),
		relpath=str(_nested_or_flat(tpl_circles_cfg, block="output", key="relpath", flat_keys=("relpath",), default="template_circles")),
		channel_scope=_normalize_channel_scope(_nested_or_flat(tpl_circles_cfg, block="display", key="channel_scope", flat_keys=("channel_scope",), default="contributing_channels")),
		background=str(_nested_or_flat(tpl_circles_cfg, block="render", key="background", flat_keys=("background",), default="black")),
		signal_color=str(_nested_or_flat(tpl_circles_cfg, block="render", key="signal_color", flat_keys=("signal_color",), default="white")),
		fast_render=_as_bool(_nested_or_flat(tpl_circles_cfg, block="render", key="fast_render", flat_keys=("fast_render",), default=False), False),
		force_center_soma=_as_bool(_nested_or_flat(tpl_circles_cfg, block="display", key="force_center_soma", flat_keys=("force_center_soma",), default=False), False),
		force_square_aspect=_as_bool(_nested_or_flat(tpl_circles_cfg, block="display", key="force_square_aspect", flat_keys=("force_square_aspect",), default=True), True),
		show_scale_bar=_as_bool(_nested_or_flat(tpl_circles_cfg, block="display", key="show_scale_bar", flat_keys=("show_scale_bar",), default=True), True),
		show_scale_circle=_as_bool(_nested_or_flat(tpl_circles_cfg, block="display", key="show_scale_circle", flat_keys=("show_scale_circle",), default=False), False),
		scale_bar_color=str(_nested_or_flat(tpl_circles_cfg, block="render", key="scale_bar_color", flat_keys=("scale_bar_color",), default="white")),
		scale_circle_color=str(_nested_or_flat(tpl_circles_cfg, block="render", key="scale_circle_color", flat_keys=("scale_circle_color",), default="white")),
		scale_bar_text_offset_frac=_as_float(
			_nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="text_offset_frac",
				flat_keys=("scale_bar_text_offset_frac",),
				default=0.02,
			),
			0.02,
		),
		scale_bar_x_offset_frac=(
			None
			if _nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="x_offset_frac",
				flat_keys=("scale_bar_x_offset_frac",),
				default=None,
			)
			is None
			else _as_float(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "scale_bar"),
					key="x_offset_frac",
					flat_keys=("scale_bar_x_offset_frac",),
					default=None,
				),
				0.02,
			)
		),
		scale_bar_x_offset_considers_fontsize=_as_bool(
			_nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="x_offset_considers_fontsize",
				flat_keys=("scale_bar_x_offset_considers_fontsize",),
				default=False,
			),
			False,
		),
		scale_bar_horizontal_alignment=_normalize_horizontal_alignment(
			_nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="horizontal_alignment",
				flat_keys=("scale_bar_horizontal_alignment",),
				default="right",
			),
			default="right",
		),
		scale_bar_vertical_alignment=_normalize_vertical_alignment(
			_nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="vertical_alignment",
				flat_keys=("scale_bar_vertical_alignment",),
				default="bottom",
			),
			default="bottom",
		),
		scale_bar_y_offset_frac=_as_float(
			_nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="y_offset_frac",
				flat_keys=("scale_bar_y_offset_frac",),
				default=0.06,
			),
			0.06,
		),
		scale_bar_fontsize=_as_float(
			_nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="fontsize",
				flat_keys=("scale_bar_fontsize",),
				default=6.0,
			),
			6.0,
		),
		scale_bar_linewidth=_as_float(
			_nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="linewidth",
				flat_keys=("scale_bar_linewidth",),
				default=1.8,
			),
			1.8,
		),
		scale_bar_length_um=(
			None
			if _nested_path_or_flat(
				tpl_circles_cfg,
				path=("display", "scale_bar"),
				key="length_um",
				flat_keys=("scale_bar_length_um",),
				default=None,
			)
			is None
			else _as_float(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "scale_bar"),
					key="length_um",
					flat_keys=("scale_bar_length_um",),
					default=None,
				),
				0.0,
			)
		),
		show_axes=_as_bool(_nested_or_flat(tpl_circles_cfg, block="display", key="show_axes", flat_keys=("show_axes",), default=True), True),
		unit_id_label=UnitIdLabelConfig(
			show=_as_bool(_get_nested_block(tpl_circles_cfg, "unit_id_label").get("show", False), False),
			fontsize=_as_float(_get_nested_block(tpl_circles_cfg, "unit_id_label").get("fontsize", 12.0), 12.0),
			color=str(_get_nested_block(tpl_circles_cfg, "unit_id_label").get("color", "white")),
			x_offset_frac=_as_float(_get_nested_block(tpl_circles_cfg, "unit_id_label").get("x_offset_frac", 0.02), 0.02),
			y_offset_frac=_as_float(_get_nested_block(tpl_circles_cfg, "unit_id_label").get("y_offset_frac", 0.02), 0.02),
			horizontal_alignment=_normalize_horizontal_alignment(
				_get_nested_block(tpl_circles_cfg, "unit_id_label").get("horizontal_alignment", "right"),
				default="right",
			),
			vertical_alignment=_normalize_vertical_alignment(
				_get_nested_block(tpl_circles_cfg, "unit_id_label").get("vertical_alignment", "top"),
				default="top",
			),
		),
		center_most_channel_coords=CenterMostChannelCoordsConfig(
			show=_as_bool(_get_nested_block(tpl_circles_cfg, "center_most_channel_coords").get("show", False), False),
			fontsize=_as_float(_get_nested_block(tpl_circles_cfg, "center_most_channel_coords").get("fontsize", 10.0), 10.0),
			color=str(_get_nested_block(tpl_circles_cfg, "center_most_channel_coords").get("color", "white")),
			x_offset_frac=_as_float(_get_nested_block(tpl_circles_cfg, "center_most_channel_coords").get("x_offset_frac", 0.02), 0.02),
			y_offset_frac=_as_float(_get_nested_block(tpl_circles_cfg, "center_most_channel_coords").get("y_offset_frac", 0.01), 0.01),
			horizontal_alignment=_normalize_horizontal_alignment(
				_get_nested_block(tpl_circles_cfg, "center_most_channel_coords").get("horizontal_alignment", "left"),
				default="left",
			),
			vertical_alignment=_normalize_vertical_alignment(
				_get_nested_block(tpl_circles_cfg, "center_most_channel_coords").get("vertical_alignment", "top"),
				default="top",
			),
		),
		size_by=_normalize_template_metric(_nested_or_flat(tpl_circles_cfg, block="display", key="size_by", flat_keys=("size_by",), default="amplitude"), "amplitude"),
		color_by=_normalize_template_metric(_nested_or_flat(tpl_circles_cfg, block="display", key="color_by", flat_keys=("color_by",), default="latency"), "latency"),
		show_propagation_order_labels=_as_bool(
			_get_nested_block(tpl_circles_cfg, "propagation_order_labels").get("show", tpl_circles_cfg.get("show_propagation_order_labels", False)),
			False,
		),
		propagation_order_label_fontsize=_as_float(
			_get_nested_block(tpl_circles_cfg, "propagation_order_labels").get("fontsize", tpl_circles_cfg.get("propagation_order_label_fontsize", 6.0)),
			6.0,
		),
		propagation_order_label_color=str(
			_get_nested_block(tpl_circles_cfg, "propagation_order_labels").get("color", tpl_circles_cfg.get("propagation_order_label_color", "white"))
		),
		propagation_order_label_bbox_alpha=_as_float(
			_get_nested_block(tpl_circles_cfg, "propagation_order_labels").get("bbox_alpha", tpl_circles_cfg.get("propagation_order_label_bbox_alpha", 0.35)),
			0.35,
		),
		color_bar_units=str(
			(_get_nested_block(tpl_circles_cfg, "color_bar").get("units", tpl_circles_cfg.get("color_bar_units", "")) or "")
		).strip(),
		color_bar_title=str(
			(_get_nested_block(tpl_circles_cfg, "color_bar").get("title", tpl_circles_cfg.get("color_bar_title", "")) or "")
		).strip(),
		color_bar_show_axes_title=_as_bool(
			_get_nested_block(tpl_circles_cfg, "color_bar").get(
				"show_axes_title",
				tpl_circles_cfg.get("color_bar_show_axes_title", True),
			),
			True,
		),
		color_bar_show_unit_labels=_as_bool(
			_get_nested_block(tpl_circles_cfg, "color_bar").get(
				"show_unit_labels",
				_get_nested_block(tpl_circles_cfg, "color_bar").get(
					"show_unit_label",
					tpl_circles_cfg.get("color_bar_show_unit_labels", True),
				),
			),
			True,
		),
		color_bar_tick_fontsize=max(
			1.0,
			_as_float(
				_get_nested_block(tpl_circles_cfg, "color_bar").get(
					"tick_fontsize",
					tpl_circles_cfg.get("color_bar_tick_fontsize", 6.0),
				),
				6.0,
			),
		),
		color_bar_tick_decimal_places=max(
			0,
			_as_int(
				_get_nested_block(tpl_circles_cfg, "color_bar").get(
					"tick_decimal_places",
					_get_nested_block(tpl_circles_cfg, "color_bar").get(
						"decimal_places",
						tpl_circles_cfg.get("color_bar_tick_decimal_places", 3),
					),
				),
				3,
			),
		),
		color_bar_tick_target_count=(
			None
			if _as_int(
				_get_nested_block(tpl_circles_cfg, "color_bar").get(
					"tick_target_count",
					_get_nested_block(tpl_circles_cfg, "color_bar").get(
						"target_tick_count",
						tpl_circles_cfg.get("color_bar_tick_target_count", 0),
					),
				),
				0,
			)
			<= 0
			else _as_int(
				_get_nested_block(tpl_circles_cfg, "color_bar").get(
					"tick_target_count",
					_get_nested_block(tpl_circles_cfg, "color_bar").get(
						"target_tick_count",
						tpl_circles_cfg.get("color_bar_tick_target_count", 0),
					),
				),
				0,
			)
		),
		color_bar_force_zero_and_neg_values_first_color_range=_as_bool(
			_get_nested_block(tpl_circles_cfg, "color_bar").get(
				"force_zero_and_neg_values_first_color_range",
				tpl_circles_cfg.get("color_bar_force_zero_and_neg_values_first_color_range", False),
			),
			False,
		),
		color_bar_zero_transition_contrast=max(
			1.0,
			_as_float(
				_get_nested_block(tpl_circles_cfg, "color_bar").get(
					"zero_transition_contrast",
					tpl_circles_cfg.get("color_bar_zero_transition_contrast", 1.0),
				),
				1.0,
			),
		),
		scale_circle=TemplateScaleCircleConfig(
			diameter=(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "scale_circle"),
					key="diameter",
					flat_keys=("scale_circle_diameter",),
					default="equal_to_max_amplitude",
				)
				if isinstance(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="diameter",
						flat_keys=("scale_circle_diameter",),
						default="equal_to_max_amplitude",
					),
					str,
				)
				else _as_float(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="diameter",
						flat_keys=("scale_circle_diameter",),
						default="equal_to_max_amplitude",
					),
					0.0,
				)
			),
			linewidth=max(
				0.1,
				_as_float(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="linewidth",
						flat_keys=("scale_circle_linewidth",),
						default=1.8,
					),
					1.8,
				),
			),
				linestyle=_normalize_optional_linestyle(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="linestyle",
						flat_keys=("scale_circle_linestyle",),
						default="solid",
					),
					"solid",
				),
				fill=_as_bool(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="fill",
						flat_keys=("scale_circle_fill",),
						default=False,
					),
					False,
				),
				fill_color=(
					lambda raw: None if raw is None else (str(raw).strip() or None)
				)(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="fill_color",
						flat_keys=("scale_circle_fill_color",),
						default=None,
					)
				),
			fontsize=max(
				1.0,
				_as_float(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="fontsize",
						flat_keys=("scale_circle_fontsize",),
						default=6.0,
					),
					6.0,
				),
			),
			digits_after_decimal=max(
				0,
				_as_int(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="digits_after_decimal",
						flat_keys=("scale_circle_digits_after_decimal",),
						default=0,
					),
					0,
				),
			),
			horizontal_alignment=_normalize_horizontal_alignment(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "scale_circle"),
					key="horizontal_alignment",
					flat_keys=("scale_circle_horizontal_alignment",),
					default="left",
				),
				default="left",
			),
			vertical_alignment=_normalize_vertical_alignment(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "scale_circle"),
					key="vertical_alignment",
					flat_keys=("scale_circle_vertical_alignment",),
					default="top",
				),
				default="top",
			),
			x_offset_frac=max(
				0.0,
				_as_float(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="x_offset_frac",
						flat_keys=("scale_circle_x_offset_frac",),
						default=0.02,
					),
					0.02,
				),
			),
			y_offset_frac=max(
				0.0,
				_as_float(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "scale_circle"),
						key="y_offset_frac",
						flat_keys=("scale_circle_y_offset_frac",),
						default=0.02,
					),
					0.02,
				),
			),
			font_location=str(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "scale_circle"),
					key="font_location",
					flat_keys=("scale_circle_font_location",),
					default="inside",
				)
			),
			font_location_circle_too_small=str(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "scale_circle"),
					key="font_location_circle_too_small",
					flat_keys=("scale_circle_font_location_circle_too_small",),
					default="below",
				)
			),
			units=str(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "scale_circle"),
					key="units",
					flat_keys=("scale_circle_units",),
					default="uV",
				)
			),
		),
		branch_morphology=TemplateCirclesBranchMorphologyConfig(
			enabled=_as_bool(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "branch_morphology"),
					key="enabled",
					flat_keys=("branch_morphology_enabled",),
					default=False,
				),
				False,
			),
			node_border_linewidth=max(
				0.0,
				_as_float(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "branch_morphology"),
						key="node_border_linewidth",
						flat_keys=("branch_morphology_node_border_linewidth",),
						default=0.35,
					),
					0.35,
				),
			),
			edge_linewidth=max(
				0.0,
				_as_float(
					_nested_path_or_flat(
						tpl_circles_cfg,
						path=("display", "branch_morphology"),
						key="edge_linewidth",
						flat_keys=("branch_morphology_edge_linewidth",),
						default=0.8,
					),
					0.8,
				),
			),
			show_branch_labels=_as_bool(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "branch_morphology"),
					key="show_branch_labels",
					flat_keys=("branch_morphology_show_branch_labels",),
					default=False,
				),
				False,
			),
			show_branch_legend=_as_bool(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "branch_morphology"),
					key="show_branch_legend",
					flat_keys=("branch_morphology_show_branch_legend",),
					default=False,
				),
				False,
			),
			unique_color_per_branch=_as_bool(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "branch_morphology"),
					key="unique_color_per_branch",
					flat_keys=("branch_morphology_unique_color_per_branch",),
					default=True,
				),
				True,
			),
			color_scheme=str(
				_nested_path_or_flat(
					tpl_circles_cfg,
					path=("display", "branch_morphology"),
					key="color_scheme",
					flat_keys=("branch_morphology_color_scheme",),
					default="tab20",
				)
			),
		),
		overlap_controls=TemplateCirclesOverlapControlsConfig(
			scalebar_coords_overlap_detect=_as_bool(
				_get_nested_block(tpl_circles_cfg, "overlap_controls").get("scalebar_coords_overlap_detect", False),
				False,
			),
			scalebar_colorbar_overlap_detect=_as_bool(
				_get_nested_block(tpl_circles_cfg, "overlap_controls").get("scalebar_colorbar_overlap_detect", False),
				False,
			),
			unitid_label_channel_overlap_detect=_as_bool(
				_get_nested_block(tpl_circles_cfg, "overlap_controls").get(
					"unitid_label_channel_overlap_detect",
					_get_nested_block(tpl_circles_cfg, "overlap_controls").get("unit_id_label_channel_overlap_detect", False),
				),
				False,
			),
			coords_channel_overlap_detect=_as_bool(
				_get_nested_block(tpl_circles_cfg, "overlap_controls").get("coords_channel_overlap_detect", False),
				False,
			),
			scalebar_channel_overlap_detect=_as_bool(
				_get_nested_block(tpl_circles_cfg, "overlap_controls").get("scalebar_channel_overlap_detect", False),
				False,
			),
			scalecircle_channel_overlap_detect=_as_bool(
				_get_nested_block(tpl_circles_cfg, "overlap_controls").get("scalecircle_channel_overlap_detect", False),
				False,
			),
			max_overlap_check_iterations=max(
				0,
				_as_int(
					_get_nested_block(tpl_circles_cfg, "overlap_controls").get("max_overlap_check_iterations", 0),
					0,
				),
			),
		),
	)
	tpl_wf_overlay = TemplateWaveformOverlayConfig(
		debug_mode=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="debug_mode", flat_keys=("debug_mode",), default=False),
			False,
		),
		write_pdf=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="output", key="write_pdf", flat_keys=("write_pdf",), default=False),
			False,
		),
		pdf_relpath=str(
			_nested_or_flat(
				tpl_wf_overlay_cfg,
				block="output",
				key="pdf_relpath",
				flat_keys=("pdf_relpath",),
				default="extremum_ch_wf_overlay.pdf",
			)
		),
		write_png=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="output", key="write_png", flat_keys=("write_png",), default=True),
			True,
		),
		png_relpath=str(
			_nested_or_flat(
				tpl_wf_overlay_cfg,
				block="output",
				key="png_relpath",
				flat_keys=("png_relpath",),
				default="extremum_ch_wf_overlay.png",
			)
		),
		top_channels_per_template=max(
			1,
			_as_int(
				_nested_or_flat(
					tpl_wf_overlay_cfg,
					block="display",
					key="top_channels_per_template",
					flat_keys=("top_channels_per_template",),
					default=10,
				),
				10,
			),
		),
		style=str(_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="style", flat_keys=("style",), default="overlay")),
		show_title=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="show_title", flat_keys=("show_title",), default=False),
			False,
		),
		show_axes=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="show_axes", flat_keys=("show_axes",), default=False),
			False,
		),
		show_channel_labels=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="show_channel_labels", flat_keys=("show_channel_labels",), default=False),
			False,
		),
		show_top_channel_info=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="show_top_channel_info", flat_keys=("show_top_channel_info",), default=True),
			True,
		),
		show_waveform_count_info=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="show_waveform_count_info", flat_keys=("show_waveform_count_info",), default=True),
			True,
		),
		include_mean=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="include_mean", flat_keys=("include_mean",), default=True),
			True,
		),
		max_waveforms_to_show=max(
			1,
			_as_int(
				_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="max_waveforms_to_show", flat_keys=("max_waveforms_to_show",), default=100),
				100,
			),
		),
		waveform_sampling_mode=str(
			_nested_or_flat(
				tpl_wf_overlay_cfg,
				block="display",
				key="waveform_sampling_mode",
				flat_keys=("waveform_sampling_mode",),
				default="uniform",
			)
		),
		random_seed=(
			None
			if _nested_or_flat(tpl_wf_overlay_cfg, block="display", key="random_seed", flat_keys=("random_seed",), default=0) is None
			else _as_int(_nested_or_flat(tpl_wf_overlay_cfg, block="display", key="random_seed", flat_keys=("random_seed",), default=0), 0)
		),
		include_scale_bar=_as_bool(
			_nested_or_flat(tpl_wf_overlay_cfg, block="scale_bar", key="include", flat_keys=("include_scale_bar",), default=True),
			True,
		),
		scale_bar_color=str(
			_nested_or_flat(tpl_wf_overlay_cfg, block="scale_bar", key="color", flat_keys=("scale_bar_color",), default="black")
		),
		scale_bar_fontsize=_as_float(
			_nested_or_flat(tpl_wf_overlay_cfg, block="scale_bar", key="fontsize", flat_keys=("scale_bar_fontsize",), default=6.0),
			6.0,
		),
		scale_bar_linewidth=_as_float(
			_nested_or_flat(tpl_wf_overlay_cfg, block="scale_bar", key="linewidth", flat_keys=("scale_bar_linewidth",), default=1.8),
			1.8,
		),
		scale_bar_time_fraction=_as_float(
			_nested_or_flat(tpl_wf_overlay_cfg, block="scale_bar", key="time_fraction", flat_keys=("scale_bar_time_fraction",), default=0.10),
			0.10,
		),
		scale_bar_amp_fraction=_as_float(
			_nested_or_flat(tpl_wf_overlay_cfg, block="scale_bar", key="amp_fraction", flat_keys=("scale_bar_amp_fraction",), default=0.10),
			0.10,
		),
		scale_bar_time_label_offset_frac=_as_float(
			_nested_or_flat(
				tpl_wf_overlay_cfg,
				block="scale_bar",
				key="time_label_offset_frac",
				flat_keys=("scale_bar_time_label_offset_frac",),
				default=0.03,
			),
			0.03,
		),
		scale_bar_amp_label_offset_frac=_as_float(
			_nested_or_flat(
				tpl_wf_overlay_cfg,
				block="scale_bar",
				key="amp_label_offset_frac",
				flat_keys=("scale_bar_amp_label_offset_frac",),
				default=0.02,
			),
			0.02,
		),
		background=str(
			_nested_or_flat(tpl_wf_overlay_cfg, block="render", key="background", flat_keys=("background",), default="white")
		),
	)
	reports_replot_from_disk = _as_bool(reports_cfg.get("replot_from_disk", False), False)
	if force_rereport:
		reports_replot_from_disk = True

	reports = ReportsConfig(
		plot_multi_source_pdf=MultiSourcePdfReportConfig(
			enabled=_as_bool(reports_cfg.get("plot_multi_source_pdf", False), False),
			pdf_relpath=str(reports_cfg.get("multi_source_pdf_relpath", "reports/template_multi_source.pdf")),
		),
		replot_from_disk=reports_replot_from_disk,
		overwrite_on_unit_rerun=_as_bool(reports_cfg.get("overwrite_on_unit_rerun", False), False),
		grid_sort_by=normalize_grid_sort_by(
			reports_grids_cfg.get("sort_by", reports_cfg.get("sort_by", "unit_id")),
			default="unit_id",
		),
		locations=UnitLocationsReportConfig(
			write_json=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="output",
					key="write_json",
					flat_keys=("write_json",),
					default=True,
				),
				True,
			),
			json_relpath=str(
				_nested_or_flat(
					report_locations_cfg,
					block="output",
					key="json_relpath",
					flat_keys=("json_relpath",),
					default="unit_locations.json",
				)
			),
			write_png=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="output",
					key="write_png",
					flat_keys=("write_png",),
					default=False,
				),
				False,
			),
			png_relpath=str(
				_nested_or_flat(
					report_locations_cfg,
					block="output",
					key="png_relpath",
					flat_keys=("png_relpath",),
					default="unit_locations.png",
				)
			),
			write_svg=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="output",
					key="write_svg",
					flat_keys=("write_svg",),
					default=False,
				),
				False,
			),
			svg_relpath=str(
				_nested_or_flat(
					report_locations_cfg,
					block="output",
					key="svg_relpath",
					flat_keys=("svg_relpath",),
					default="unit_locations.svg",
				)
			),
			background=str(
				_nested_or_flat(
					report_locations_cfg,
					block="render",
					key="background",
					flat_keys=("background",),
					default="black",
				)
			),
			chip_scatter_color=str(
				_nested_or_flat(
					report_locations_cfg,
					block="render",
					key="chip_scatter_color",
					flat_keys=("chip_scatter_color",),
					default="white",
				)
			),
			chip_scatter_size=max(
				0.0,
				_as_float(
					_nested_or_flat(
						report_locations_cfg,
						block="render",
						key="chip_scatter_size",
						flat_keys=("chip_scatter_size", "unit_scatter_size"),
						default=14.0,
					),
					14.0,
				),
			),
			chip_scatter_alpha=min(
				1.0,
				max(
					0.0,
					_as_float(
						_nested_or_flat(
							report_locations_cfg,
							block="render",
							key="chip_scatter_alpha",
							flat_keys=("chip_scatter_alpha",),
							default=0.8,
						),
						0.8,
					),
				),
			),
			invert_y_axis=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="render",
					key="invert_y_axis",
					flat_keys=("invert_y_axis",),
					default=True,
				),
				True,
			),
			use_probe_active_area=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="render",
					key="use_probe_active_area",
					flat_keys=("use_probe_active_area",),
					default=True,
				),
				True,
			),
			underlay_concat_channels=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="render",
					key="underlay_concat_channels",
					flat_keys=("underlay_concat_channels",),
					default=True,
				),
				True,
			),
			concat_channel_scatter_color=str(
				_nested_or_flat(
					report_locations_cfg,
					block="render",
					key="concat_channel_scatter_color",
					flat_keys=("concat_channel_scatter_color",),
					default="#808080",
				)
			),
			concat_channel_scatter_size=max(
				0.0,
				_as_float(
					_nested_or_flat(
						report_locations_cfg,
						block="render",
						key="concat_channel_scatter_size",
						flat_keys=("concat_channel_scatter_size",),
						default=2.5,
					),
					2.5,
				),
			),
			concat_channel_scatter_alpha=min(
				1.0,
				max(
					0.0,
					_as_float(
						_nested_or_flat(
							report_locations_cfg,
							block="render",
							key="concat_channel_scatter_alpha",
							flat_keys=("concat_channel_scatter_alpha",),
							default=0.35,
						),
						0.35,
					),
				),
			),
			underlay_template_channels=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="render",
					key="underlay_template_channels",
					flat_keys=("underlay_template_channels",),
					default=False,
				),
				False,
			),
			template_channel_scatter_size=max(
				0.0,
				_as_float(
					_nested_or_flat(
						report_locations_cfg,
						block="render",
						key="template_channel_scatter_size",
						flat_keys=("template_channel_scatter_size",),
						default=2.0,
					),
					2.0,
				),
			),
			template_channel_scatter_alpha=min(
				1.0,
				max(
					0.0,
					_as_float(
						_nested_or_flat(
							report_locations_cfg,
							block="render",
							key="template_channel_scatter_alpha",
							flat_keys=("template_channel_scatter_alpha",),
							default=0.30,
						),
						0.30,
					),
				),
			),
			template_channel_colormap=str(
				_nested_or_flat(
					report_locations_cfg,
					block="render",
					key="template_channel_colormap",
					flat_keys=("template_channel_colormap",),
					default="tab20",
				)
			),
			show_original_to_current_redlines=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="display",
					key="show_original_to_current_redlines",
					flat_keys=("show_original_to_current_redlines",),
					default=False,
				),
				False,
			),
			redline_color=str(
				_nested_or_flat(
					report_locations_cfg,
					block="display",
					key="redline_color",
					flat_keys=("redline_color",),
					default="red",
				)
			),
			redline_alpha=min(
				1.0,
				max(
					0.0,
					_as_float(
						_nested_or_flat(
							report_locations_cfg,
							block="display",
							key="redline_alpha",
							flat_keys=("redline_alpha",),
							default=0.9,
						),
						0.9,
					),
				),
			),
			redline_linewidth=max(
				0.1,
				_as_float(
					_nested_or_flat(
						report_locations_cfg,
						block="display",
						key="redline_linewidth",
						flat_keys=("redline_linewidth",),
						default=0.7,
					),
					0.7,
				),
			),
			show_unit_id_labels=_as_bool(
				_nested_or_flat(
					report_locations_cfg,
					block="display",
					key="show_unit_id_labels",
					flat_keys=("show_unit_id_labels",),
					default=True,
				),
				True,
			),
			unit_id_label_fontsize=max(
				1.0,
				_as_float(
					_nested_or_flat(
						report_locations_cfg,
						block="display",
						key="unit_id_label_fontsize",
						flat_keys=("unit_id_label_fontsize",),
						default=6.0,
					),
					6.0,
				),
			),
			unit_id_label_color=str(
				_nested_or_flat(
					report_locations_cfg,
					block="display",
					key="unit_id_label_color",
					flat_keys=("unit_id_label_color",),
					default="white",
				)
			),
			unit_id_label_x_offset_frac=_as_float(
				_nested_or_flat(
					report_locations_cfg,
					block="display",
					key="unit_id_label_x_offset_frac",
					flat_keys=("unit_id_label_x_offset_frac",),
					default=0.02,
				),
				0.02,
			),
			unit_id_label_y_offset_frac=_as_float(
				_nested_or_flat(
					report_locations_cfg,
					block="display",
					key="unit_id_label_y_offset_frac",
					flat_keys=("unit_id_label_y_offset_frac",),
					default=0.02,
				),
				0.02,
			),
			unit_id_label_horizontal_alignment=_normalize_horizontal_alignment(
				_nested_or_flat(
					report_locations_cfg,
					block="display",
					key="unit_id_label_horizontal_alignment",
					flat_keys=("unit_id_label_horizontal_alignment",),
					default="right",
				),
				default="right",
			),
			unit_id_label_vertical_alignment=_normalize_vertical_alignment(
				_nested_or_flat(
					report_locations_cfg,
					block="display",
					key="unit_id_label_vertical_alignment",
					flat_keys=("unit_id_label_vertical_alignment",),
					default="top",
				),
				default="top",
			),
		),
		time_upsample=_build_time_upsample_config(time_upsample_cfg_raw),
		wf_overlay_grid=WfOverlayGridReportConfig(
			write_pdf=_as_bool(
				_nested_or_flat(report_overlay_grid_cfg, block="output", key="write_pdf", flat_keys=("write_pdf",), default=False),
				False,
			),
			pdf_relpath=str(
				_nested_or_flat(report_overlay_grid_cfg, block="output", key="pdf_relpath", flat_keys=("pdf_relpath",), default="wf_overlay_grid.pdf")
			),
			write_png=_as_bool(
				_nested_or_flat(report_overlay_grid_cfg, block="output", key="write_png", flat_keys=("write_png",), default=True),
				True,
			),
			png_relpath=str(
				_nested_or_flat(report_overlay_grid_cfg, block="output", key="png_relpath", flat_keys=("png_relpath",), default="wf_overlay_grid.png")
			),
			write_svg=_as_bool(
				_nested_or_flat(report_overlay_grid_cfg, block="output", key="write_svg", flat_keys=("write_svg",), default=False),
				False,
			),
			svg_relpath=str(
				_nested_or_flat(report_overlay_grid_cfg, block="output", key="svg_relpath", flat_keys=("svg_relpath",), default="wf_overlay_grid.svg")
			),
			keep_temp_svg=_as_bool(
				_nested_or_flat(report_overlay_grid_cfg, block="output", key="keep_temp_svg", flat_keys=("keep_temp_svg",), default=False),
				False,
			),
			temp_svg_relpath=str(
				_nested_or_flat(
					report_overlay_grid_cfg,
					block="output",
					key="temp_svg_relpath",
					flat_keys=("temp_svg_relpath",),
					default="wf_overlay_grid__temp.svg",
				)
			),
			top_channels_per_template=max(
				1,
				_as_int(
					_nested_or_flat(
						report_overlay_grid_cfg,
						block="display",
						key="top_channels_per_template",
						flat_keys=("top_channels_per_template",),
						default=10,
					),
					10,
				),
			),
			subplot_background_color=str(
				_nested_or_flat(
					report_overlay_grid_cfg,
					block="render",
					key="subplot_background_color",
					flat_keys=("subplot_background_color",),
					default="white",
				)
			),
			figure_background_color=str(
				_nested_or_flat(
					report_overlay_grid_cfg,
					block="render",
					key="figure_background_color",
					flat_keys=("figure_background_color",),
					default="white",
				)
			),
			render_mode=_normalize_grid_render_mode(
				_nested_or_flat(report_overlay_grid_cfg, block="render", key="mode", flat_keys=("render_mode",), default="direct_replot"),
				"direct_replot",
			),
			dpi=max(
				72.0,
				_as_float(
					_nested_or_flat(report_overlay_grid_cfg, block="render", key="dpi", flat_keys=("dpi",), default=300.0),
					300.0,
				),
			),
		),
		footprint_grids=FootprintGridsReportConfig(
			circles_map_grid=_build_footprint_grid_report_config(
				(footprint_grids_cfg.get("circles_map_grid", {}) if isinstance(footprint_grids_cfg.get("circles_map_grid", {}), dict) else {}),
				pdf_relpath_default="circles_map_grid.pdf",
				png_relpath_default="circles_map_grid.png",
				svg_relpath_default="circles_map_grid.svg",
				temp_svg_relpath_default="circles_map_grid__temp.svg",
			),
			amplitude_map_grid=_build_footprint_grid_report_config(
				(footprint_grids_cfg.get("amplitude_map_grid", {}) if isinstance(footprint_grids_cfg.get("amplitude_map_grid", {}), dict) else {}),
				pdf_relpath_default="amplitude_map_grid.pdf",
				png_relpath_default="amplitude_map_grid.png",
				svg_relpath_default="amplitude_map_grid.svg",
				temp_svg_relpath_default="amplitude_map_grid__temp.svg",
			),
			latency_map_grid=_build_footprint_grid_report_config(
				(footprint_grids_cfg.get("latency_map_grid", {}) if isinstance(footprint_grids_cfg.get("latency_map_grid", {}), dict) else {}),
				pdf_relpath_default="latency_map_grid.pdf",
				png_relpath_default="latency_map_grid.png",
				svg_relpath_default="latency_map_grid.svg",
				temp_svg_relpath_default="latency_map_grid__temp.svg",
			),
		),
	)
	analyzer_cache = AnalyzerCacheConfig(
		enabled=_as_bool(analyzer_cache_cfg.get("enabled", True), True),
		relpath=str(analyzer_cache_cfg.get("relpath", analyzer_cache_cfg.get("relpath_root", "analyzers")) or "analyzers"),
		concat_analyzer_subdir=str(analyzer_cache_cfg.get("concat_analyzer_subdir", "concat") or "concat").strip().strip("/"),
		segment_analyzers_subdir=str(analyzer_cache_cfg.get("segment_analyzers_subdir", "") or "").strip().strip("/"),
		cleanup_on_success=_as_bool(analyzer_cache_cfg.get("cleanup_on_success", False), False),
		reuse_on_force_restart=_as_bool(analyzer_cache_cfg.get("reuse_on_force_restart", False), False),
	)
	footprint_plots = FootprintPlotsConfig(
		amplitude_map=_build_footprint_map_config(amp_map_cfg, relpath_default="footprint_amplitude_map"),
		latency_map=_build_footprint_map_config(lat_map_cfg, relpath_default="footprint_latency_map"),
	)
	topographical_footprints = TopographicalFootprintsConfig(
		amplitude=_build_topographical_footprint_config(topo_amp_cfg, relpath_default="topographical_amplitude_footprint"),
		latency=_build_topographical_footprint_config(topo_lat_cfg, relpath_default="topographical_latency_footprint"),
	)
	propagation_output_cfg = _get_nested_block(propagation_cfg, "output")
	propagation_plot_output_cfg = _get_nested_block(propagation_output_cfg, "propagation_plot")
	circles_numbered_output_cfg = _get_nested_block(propagation_output_cfg, "circles_template_numbered")
	propagation_2panel_output_cfg = _get_nested_block(propagation_output_cfg, "propagation_2panel")
	propagation_2panel_layout_cfg = _get_nested_block(propagation_2panel_output_cfg, "layout")
	propagation_plots = PropagationPlotConfig(
		write_pdf=_as_bool(
			propagation_plot_output_cfg.get(
				"write_pdf",
				_nested_or_flat(propagation_cfg, block="output", key="write_pdf", flat_keys=("write_pdf",), default=False),
			),
			False,
		),
		pdf_relpath=str(
			propagation_plot_output_cfg.get(
				"pdf_relpath",
				_nested_or_flat(propagation_cfg, block="output", key="pdf_relpath", flat_keys=("pdf_relpath",), default="propagation_plot.pdf"),
			)
		),
		write_png=_as_bool(
			propagation_plot_output_cfg.get(
				"write_png",
				_nested_or_flat(propagation_cfg, block="output", key="write_png", flat_keys=("write_png",), default=True),
			),
			True,
		),
		png_relpath=str(
			propagation_plot_output_cfg.get(
				"png_relpath",
				_nested_or_flat(propagation_cfg, block="output", key="png_relpath", flat_keys=("png_relpath",), default="propagation_plot.png"),
			)
		),
		write_svg=_as_bool(
			propagation_plot_output_cfg.get(
				"write_svg",
				_nested_or_flat(propagation_cfg, block="output", key="write_svg", flat_keys=("write_svg",), default=True),
			),
			True,
		),
		write_circles_template_numbered_png=_as_bool(
			circles_numbered_output_cfg.get(
				"write_png",
				_nested_or_flat(
					propagation_cfg,
					block="output",
					key="write_circles_template_numbered_png",
					flat_keys=("write_circles_template_numbered_png",),
					default=True,
				),
			),
			True,
		),
		write_circles_template_numbered_svg=_as_bool(
			circles_numbered_output_cfg.get(
				"write_svg",
				_nested_or_flat(
					propagation_cfg,
					block="output",
					key="write_circles_template_numbered_svg",
					flat_keys=("write_circles_template_numbered_svg",),
					default=True,
				),
			),
			True,
		),
		circles_template_numbered_relpath=str(
			circles_numbered_output_cfg.get(
				"relpath",
				_nested_or_flat(
					propagation_cfg,
					block="output",
					key="circles_template_numbered_relpath",
					flat_keys=("circles_template_numbered_relpath",),
					default=_nested_or_flat(
						propagation_cfg,
						block="display",
						key="right_panel_png_relpath",
						flat_keys=("right_panel_png_relpath",),
						default="circles_template_numbered",
					),
				),
			)
		),
		write_propagation_2panel_png=_as_bool(
			propagation_2panel_output_cfg.get(
				"write_png",
				_nested_or_flat(
					propagation_cfg,
					block="output",
					key="write_propagation_2panel_png",
					flat_keys=("write_propagation_2panel_png",),
					default=True,
				),
			),
			True,
		),
		write_propagation_2panel_svg=_as_bool(
			propagation_2panel_output_cfg.get(
				"write_svg",
				_nested_or_flat(
					propagation_cfg,
					block="output",
					key="write_propagation_2panel_svg",
					flat_keys=("write_propagation_2panel_svg",),
					default=True,
				),
			),
			True,
		),
		propagation_2panel_relpath=str(
			propagation_2panel_output_cfg.get(
				"relpath",
				_nested_or_flat(
					propagation_cfg,
					block="output",
					key="propagation_2panel_relpath",
					flat_keys=("propagation_2panel_relpath",),
					default="propagation_2panel",
				),
			)
		),
		show_title=_as_bool(_nested_or_flat(propagation_cfg, block="display", key="show_title", flat_keys=("show_title",), default=True), True),
		title_template=str(_nested_or_flat(propagation_cfg, block="display", key="title_template", flat_keys=("title_template",), default="Propagation traces {start}-{end} / {total}")),
		title_fontsize=_as_float(_nested_or_flat(propagation_cfg, block="display", key="title_fontsize", flat_keys=("title_fontsize",), default=9.0), 9.0),
		top_channels=max(1, _as_int(_nested_or_flat(propagation_cfg, block="display", key="top_channels", flat_keys=("top_channels",), default=25), 25)),
		window_strategy=str(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="window_strategy",
				flat_keys=("window_strategy",),
				default="max_ptp_sum",
			)
		),
		channels_per_panel=max(1, _as_int(_nested_or_flat(propagation_cfg, block="display", key="channels_per_panel", flat_keys=("channels_per_panel",), default=25), 25)),
		channel_overlap=max(0, _as_int(_nested_or_flat(propagation_cfg, block="display", key="channel_overlap", flat_keys=("channel_overlap",), default=5), 5)),
		force_start_with_max_ptp=_as_bool(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="force_start_with_max_ptp",
				flat_keys=("force_start_with_max_ptp", "force_start_with_largest_peak_to_peak"),
				default=True,
			),
			True,
		),
		force_start_with_max_negative_peak=_as_bool(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="force_start_with_max_negative_peak",
				flat_keys=("force_start_with_max_negative_peak", "force_start_with_most_negative_peak"),
				default=False,
			),
			False,
		),
		force_min_neg_peak_index_zero=_as_bool(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="force_min_neg_peak_index_zero",
				flat_keys=("force_min_neg_peak_index_zero", "force_max_ptp_index_zero"),
				default=False,
			),
			False,
		),
		ordering_latency_mode=str(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="ordering_latency_mode",
				flat_keys=("ordering_latency_mode", "latency_mode"),
				default=analysis_ordering_latency_mode,
			)
		),
		latency_tie_breaker=str(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="latency_tie_breaker",
				flat_keys=("latency_tie_breaker",),
				default="channel_index",
			)
		),
		debug_ordering=_as_bool(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="debug_ordering",
				flat_keys=("debug_ordering", "debug"),
				default=analysis_debug_ordering,
			),
			False,
		),
		trace_label_mode=str(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="trace_label_mode",
				flat_keys=("trace_label_mode",),
				default="electrode_id",
			)
		),
		relative_signed_order_numbers=_as_bool(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="relative_signed_order_numbers",
				flat_keys=("relative_signed_order_numbers",),
				default=True,
			),
			True,
		),
		show_right_panel=_as_bool(
			propagation_2panel_output_cfg.get(
				"enabled",
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="show_right_panel",
				flat_keys=("show_right_panel",),
				default=False,
			),
			),
			False,
		),
		right_panel_gap_fraction=_as_float(
			propagation_2panel_layout_cfg.get(
				"gap_fraction",
				_nested_or_flat(
					propagation_cfg,
					block="display",
					key="right_panel_gap_fraction",
					flat_keys=("right_panel_gap_fraction",),
					default=0.04,
				),
			),
			0.04,
		),
		right_panel_width_scale=_as_float(
			propagation_2panel_layout_cfg.get(
				"width_scale",
				_nested_or_flat(
					propagation_cfg,
					block="display",
					key="right_panel_width_scale",
					flat_keys=("right_panel_width_scale",),
					default=1.0,
				),
			),
			1.0,
		),
		right_panel_keep_temp_svg=_as_bool(
			propagation_2panel_layout_cfg.get(
				"keep_temp_svg",
				_nested_or_flat(
					propagation_cfg,
					block="display",
					key="right_panel_keep_temp_svg",
					flat_keys=("right_panel_keep_temp_svg",),
					default=False,
				),
			),
			False,
		),
		right_panel_svg_relpath=str(
			propagation_2panel_layout_cfg.get(
				"right_panel_svg_relpath",
				_nested_or_flat(
					propagation_cfg,
					block="display",
					key="right_panel_svg_relpath",
					flat_keys=("right_panel_svg_relpath",),
					default="propagation_plot__right_temp.svg",
				),
			)
		),
		right_panel_png_relpath=str(
			propagation_2panel_layout_cfg.get(
				"right_panel_png_relpath",
				_nested_or_flat(
					propagation_cfg,
					block="display",
					key="right_panel_png_relpath",
					flat_keys=("right_panel_png_relpath",),
					default="propagation_plot__right_temp.png",
				),
			)
		),
		left_panel_png_dpi=_as_float_or_none(
			propagation_plot_output_cfg.get(
				"png_dpi",
				_nested_or_flat(
					propagation_cfg,
					block="display",
					key="left_panel_png_dpi",
					flat_keys=("left_panel_png_dpi",),
					default=None,
				),
			),
			None,
		),
		right_panel_png_dpi=_as_float(
			circles_numbered_output_cfg.get(
				"png_dpi",
				_nested_or_flat(
					propagation_cfg,
					block="display",
					key="right_panel_png_dpi",
					flat_keys=("right_panel_png_dpi",),
					default=300.0,
				),
			),
			300.0,
		),
		composed_png_dpi=_as_float_or_none(
			propagation_2panel_output_cfg.get(
				"png_dpi",
				_nested_or_flat(
					propagation_cfg,
					block="display",
					key="composed_png_dpi",
					flat_keys=("composed_png_dpi",),
					default=None,
				),
			),
			None,
		),
		background=str(_nested_or_flat(propagation_cfg, block="render", key="background", flat_keys=("background",), default="white")),
		show_electrode_ids=_as_bool(_nested_or_flat(propagation_cfg, block="labels", key="show_electrode_ids", flat_keys=("show_electrode_ids",), default=False), False),
		electrode_label_fontsize=_as_float(
			_nested_or_flat(
				propagation_cfg,
				block="labels",
				key="electrode_label_fontsize",
				flat_keys=("electrode_label_fontsize", "channel_label_fontsize"),
				default=6.0,
			),
			6.0,
		),
		electrode_label_x_offset_frac=_as_float(
			_nested_or_flat(
				propagation_cfg,
				block="labels",
				key="electrode_label_x_offset_frac",
				flat_keys=("electrode_label_x_offset_frac", "channel_label_x_offset_frac"),
				default=0.01,
			),
			0.01,
		),
		electrode_label_y_offset_frac=_as_float(
			_nested_or_flat(
				propagation_cfg,
				block="labels",
				key="electrode_label_y_offset_frac",
				flat_keys=("electrode_label_y_offset_frac", "channel_label_y_offset_frac"),
				default=0.0,
			),
			0.0,
		),
		electrode_label_alignment=str(
			_nested_or_flat(
				propagation_cfg,
				block="labels",
				key="electrode_label_alignment",
				flat_keys=("electrode_label_alignment", "channel_label_alignment"),
				default="left",
			)
		),
		trace_gain=_as_float(_nested_or_flat(propagation_cfg, block="render", key="trace_gain", flat_keys=("trace_gain",), default=1.0), 1.0),
		trace_spacing=_as_float(_nested_or_flat(propagation_cfg, block="render", key="trace_spacing", flat_keys=("trace_spacing",), default=1.0), 1.0),
		peak_marker_height_frac=_as_float(_nested_or_flat(propagation_cfg, block="render", key="peak_marker_height_frac", flat_keys=("peak_marker_height_frac",), default=0.24), 0.24),
		peak_marker_linewidth=_as_float(_nested_or_flat(propagation_cfg, block="render", key="peak_marker_linewidth", flat_keys=("peak_marker_linewidth",), default=1.4), 1.4),
		show_multiple_peak_markers=_as_bool(_nested_or_flat(propagation_cfg, block="render", key="show_multiple_peak_markers", flat_keys=("show_multiple_peak_markers",), default=False), False),
		delay_peak_marker_color=str(_nested_or_flat(propagation_cfg, block="render", key="delay_peak_marker_color", flat_keys=("delay_peak_marker_color",), default="black")),
		show_scale_bar=_as_bool(_nested_or_flat(propagation_cfg, block="scale_bar", key="show", flat_keys=("show_scale_bar",), default=True), True),
		scale_bar_anchor_x_frac=_as_float(_nested_or_flat(propagation_cfg, block="scale_bar", key="anchor_x_frac", flat_keys=("scale_bar_anchor_x_frac",), default=0.92), 0.92),
		scale_bar_anchor_y_frac=_as_float(_nested_or_flat(propagation_cfg, block="scale_bar", key="anchor_y_frac", flat_keys=("scale_bar_anchor_y_frac",), default=0.12), 0.12),
		scale_bar_time_fraction=_as_float(_nested_or_flat(propagation_cfg, block="scale_bar", key="time_fraction", flat_keys=("scale_bar_time_fraction",), default=0.15), 0.15),
		scale_bar_amp_fraction=_as_float(_nested_or_flat(propagation_cfg, block="scale_bar", key="amp_fraction", flat_keys=("scale_bar_amp_fraction",), default=0.20), 0.20),
		force_amp_frac_to_max_amp=_as_bool(_nested_or_flat(propagation_cfg, block="scale_bar", key="force_amp_frac_to_max_amp", flat_keys=("force_amp_frac_to_max_amp",), default=False), False),
		debug_max_amps_at_each_channel=_as_bool(_nested_or_flat(propagation_cfg, block="scale_bar", key="debug_max_amps_at_each_channel", flat_keys=("debug_max_amps_at_each_channel",), default=False), False),
		bold_max_amp_electrode_label=_as_bool(
			_nested_or_flat(
				propagation_cfg,
				block="labels",
				key="bold_max_amp_electrode_label",
				flat_keys=("bold_max_amp_electrode_label", "bold_max_amp_channel_label"),
				default=False,
			),
			False,
		),
		scale_bar_linewidth=_as_float(_nested_or_flat(propagation_cfg, block="scale_bar", key="linewidth", flat_keys=("scale_bar_linewidth",), default=1.8), 1.8),
		scale_bar_fontsize=_as_float(_nested_or_flat(propagation_cfg, block="scale_bar", key="fontsize", flat_keys=("scale_bar_fontsize",), default=7.0), 7.0),
		scale_bar_time_label_offset_frac=_as_float(_nested_or_flat(propagation_cfg, block="scale_bar", key="time_label_offset_frac", flat_keys=("scale_bar_time_label_offset_frac",), default=0.04), 0.04),
		scale_bar_amp_label_offset_frac=_as_float(_nested_or_flat(propagation_cfg, block="scale_bar", key="amp_label_offset_frac", flat_keys=("scale_bar_amp_label_offset_frac",), default=0.02), 0.02),
		abbreviate_post_ap_signal=_as_bool(
			_get_nested_block(propagation_cfg, "post_ap_abbrev").get("enabled", propagation_cfg.get("abbreviate_post_ap_signal", False)),
			False,
		),
		post_ap_abbrev_start_ms=_as_float(
			_get_nested_block(propagation_cfg, "post_ap_abbrev").get("start_ms", propagation_cfg.get("post_ap_abbrev_start_ms", 1.0)),
			1.0,
		),
		post_ap_abbrev_start_samples=max(
			0,
			_as_int(
				_get_nested_block(propagation_cfg, "post_ap_abbrev").get("start_samples", propagation_cfg.get("post_ap_abbrev_start_samples", 10)),
				10,
			),
		),
		post_ap_abbrev_cut_fraction=_as_float(
			_get_nested_block(propagation_cfg, "post_ap_abbrev").get("cut_fraction", propagation_cfg.get("post_ap_abbrev_cut_fraction", 0.5)),
			0.5,
		),
		post_ap_abbrev_min_samples_to_cut=max(
			1,
			_as_int(
				_get_nested_block(propagation_cfg, "post_ap_abbrev").get("min_samples_to_cut", propagation_cfg.get("post_ap_abbrev_min_samples_to_cut", 5)),
				5,
			),
		),
		post_ap_abbrev_gap_samples=max(
			0,
			_as_int(
				_get_nested_block(propagation_cfg, "post_ap_abbrev").get("gap_samples", propagation_cfg.get("post_ap_abbrev_gap_samples", 8)),
				8,
			),
		),
		post_ap_abbrev_marker_text=str(
			_get_nested_block(propagation_cfg, "post_ap_abbrev").get("marker_text", propagation_cfg.get("post_ap_abbrev_marker_text", "/.../"))
		),
		post_ap_abbrev_marker_fontsize=_as_float(
			_get_nested_block(propagation_cfg, "post_ap_abbrev").get("marker_fontsize", propagation_cfg.get("post_ap_abbrev_marker_fontsize", 7.0)),
			7.0,
		),
		post_ap_abbrev_marker_y_offset_frac=_as_float(
			_get_nested_block(propagation_cfg, "post_ap_abbrev").get("marker_y_offset_frac", propagation_cfg.get("post_ap_abbrev_marker_y_offset_frac", 0.0)),
			0.0,
		),
		show_duration_info=_as_bool(
			_get_nested_block(propagation_cfg, "duration_info").get("show", propagation_cfg.get("show_duration_info", False)),
			False,
		),
		duration_info_x_frac=_as_float(
			_get_nested_block(propagation_cfg, "duration_info").get("x_frac", propagation_cfg.get("duration_info_x_frac", 0.01)),
			0.01,
		),
		duration_info_y_frac=_as_float(
			_get_nested_block(propagation_cfg, "duration_info").get("y_frac", propagation_cfg.get("duration_info_y_frac", 0.99)),
			0.99,
		),
		duration_info_fontsize=_as_float(
			_get_nested_block(propagation_cfg, "duration_info").get("fontsize", propagation_cfg.get("duration_info_fontsize", 6.0)),
			6.0,
		),
		duration_info_horizontal_alignment=str(
			_get_nested_block(propagation_cfg, "duration_info").get("horizontal_alignment", propagation_cfg.get("duration_info_horizontal_alignment", "left"))
		),
		duration_info_vertical_alignment=str(
			_get_nested_block(propagation_cfg, "duration_info").get("vertical_alignment", propagation_cfg.get("duration_info_vertical_alignment", "top"))
		),
		plot_width_in=_as_float(
			_get_nested_block(propagation_cfg, "plot_layout").get("width_in", propagation_cfg.get("plot_width_in", 13.0)),
			13.0,
		),
		plot_panel_height_in=_as_float(
			_get_nested_block(propagation_cfg, "plot_layout").get("panel_height_in", propagation_cfg.get("plot_panel_height_in", 2.8)),
			2.8,
		),
		plot_extra_height_in=_as_float(
			_get_nested_block(propagation_cfg, "plot_layout").get("extra_height_in", propagation_cfg.get("plot_extra_height_in", 1.0)),
			1.0,
		),
		plot_hspace=_as_float(
			_get_nested_block(propagation_cfg, "plot_layout").get("hspace", propagation_cfg.get("plot_hspace", 0.35)),
			0.35,
		),
		plot_area_aspect_ratio=_as_float_or_none(
			_get_nested_block(propagation_cfg, "plot_layout").get("area_aspect_ratio", propagation_cfg.get("plot_area_aspect_ratio", None)),
			None,
		),
		latency_map=_build_propagation_latency_map_config(
			_get_nested_block(propagation_cfg, "latency_map")
		),
	)

	prop_display_cfg = _get_nested_block(propagation_cfg, "display")
	legacy_prop_keys = (
		"show_right_panel",
		"right_panel_gap_fraction",
		"right_panel_width_scale",
		"right_panel_keep_temp_svg",
		"right_panel_svg_relpath",
		"right_panel_png_relpath",
	)
	if any(k in prop_display_cfg for k in legacy_prop_keys):
		LOGGER.warning(
			"templates.propagation_plots legacy right_panel keys are deprecated; prefer output.circles_template_numbered and output.propagation_2panel blocks"
		)

	per_unit = PerUnitTemplatesOutputsConfig(
		unit_reldir=_get_unit_reldir(runtime_config),
		quality_checks=_build_per_unit_quality_checks_outputs_config(per_unit_quality_checks_cfg),
		merged_template=_build_template_artifact_config(
			merged_template_cfg,
			relpath_default="merged_template.npy",
			channel_locations_relpath_default="merged_channel_locations.npy",
		),
		square_template=_build_template_artifact_config(
			square_template_cfg,
			relpath_default="square_template.npy",
			channel_locations_relpath_default="square_channel_locations.npy",
		),
		scan_template=_build_template_artifact_config(
			scan_template_cfg,
			relpath_default="scan_template.npy",
			channel_locations_relpath_default="scan_channel_locations.npy",
		),
		full_template=_build_template_artifact_config(
			full_template_cfg,
			relpath_default="full_template.npy",
			channel_locations_relpath_default="full_channel_locations_xy.npy",
		),
		template=tpl,
		template_circles=tpl_circles,
		template_wf_overlay=tpl_wf_overlay,
		footprint_plots=footprint_plots,
		topographical_footprints=topographical_footprints,
		propagation_plots=propagation_plots,
	)

	max_candidates_per_source_raw = _as_int(resolve_sources_phase_cfg.get("max_candidates_per_source", 12), 12)
	max_candidates_per_source = int(max(1, max_candidates_per_source_raw))

	resolve_sources_phase = ResolveSourcesPhaseConfig(
		enabled=_as_bool(resolve_sources_phase_cfg.get("enabled", True), True),
		show_header=_as_bool(resolve_sources_phase_cfg.get("show_header", True), True),
		log_candidates=_as_bool(resolve_sources_phase_cfg.get("log_candidates", True), True),
		check_path_exists=_as_bool(resolve_sources_phase_cfg.get("check_path_exists", True), True),
		include_alternate_well_dirs=_as_bool(resolve_sources_phase_cfg.get("include_alternate_well_dirs", True), True),
		probe_curated_units=_as_bool(resolve_sources_phase_cfg.get("probe_curated_units", True), True),
		max_candidates_per_source=max_candidates_per_source,
		fail_if_required_sources_missing=_as_bool(
			resolve_sources_phase_cfg.get("fail_if_required_sources_missing", False),
			False,
		),
		write_json=_as_bool(resolve_sources_phase_cfg.get("write_json", False), False),
		json_relpath=str(resolve_sources_phase_cfg.get("json_relpath", "context/resolve_sources_summary.json")),
	)

	analyzers_phase = TemplatesAnalyzersPhaseConfig(
		enabled=_as_bool(analyzers_phase_cfg_raw.get("enabled", True), True),
		summary_json_relpath=str(analyzers_phase_cfg_raw.get("summary_json_relpath", "context/analyzers_summary.json")),
		concat=concat_phase_cfg,
		segments=segments_phase_cfg,
	)
	phase_extract_cfg = _phase_block(phases_cfg, "per_unit_processing", "extract_template_segments")
	phase_quality_cfg_raw = _phase_block(phases_cfg, "per_unit_processing", "quality_checks")
	phase_analysis_cfg = _phase_block(phases_cfg, "per_unit_processing", "analysis")
	phase_plot_templates_cfg = _phase_block(phases_cfg, "plot_templates")
	phase_report_templates_cfg = _phase_block(phases_cfg, "report_templates")
	phase_plots_cfg = _phase_block(phases_cfg, "per_unit_processing", "plots")
	phase_reports_cfg = _phase_block(phases_cfg, "reports")
	effective_plot_phase_cfg = (phase_plot_templates_cfg if phase_plot_templates_cfg else phase_plots_cfg)
	plot_phase_resources_cfg = _phase_block(effective_plot_phase_cfg, "resources")
	plot_phase_unit_workers = _parse_optional_positive_int(
		plot_phase_resources_cfg.get("unit_workers", effective_plot_phase_cfg.get("unit_workers", None))
	)
	plot_phase_unit_procs = _parse_optional_positive_int(
		plot_phase_resources_cfg.get("unit_procs", effective_plot_phase_cfg.get("unit_procs", None))
	)
	plot_phase_unit_batch_size = _parse_optional_positive_int(
		plot_phase_resources_cfg.get("unit_batch_size", effective_plot_phase_cfg.get("unit_batch_size", None))
	)
	build_templates_phase = TemplateBuildTemplatesPhaseConfig(
		enabled=_as_bool(phase_build_cfg.get("enabled", True), True),
		summary_json_relpath=str(phase_build_cfg.get("summary_json_relpath", "context/build_templates_summary.json")),
		merge=merge,
		execution_upsampling=execution_upsampling,
	)
	plot_templates_phase = TemplatePlotsPhaseConfig(
		enabled=_as_bool(effective_plot_phase_cfg.get("enabled", True), True),
		summary_json_relpath=str(
			effective_plot_phase_cfg.get("summary_json_relpath", "context/plot_templates_summary.json")
		),
		unit_workers=plot_phase_unit_workers,
		unit_procs=plot_phase_unit_procs,
		unit_batch_size=plot_phase_unit_batch_size,
		outputs=per_unit,
	)
	report_templates_phase = TemplateReportTemplatesPhaseConfig(
		enabled=_as_bool(phase_report_templates_cfg.get("enabled", True), True),
		summary_json_relpath=str(
			phase_report_templates_cfg.get("summary_json_relpath", "context/report_templates_summary.json")
		),
		relpath=str(phase_report_templates_cfg.get("relpath", "template_report.pdf")),
		write_pdf=_as_bool(phase_report_templates_cfg.get("write_pdf", True), True),
	)
	per_unit_processing_phase = TemplatePerUnitProcessingPhaseConfig(
		enabled=_as_bool(_phase_block(phases_cfg, "per_unit_processing").get("enabled", True), True),
		extract_template_segments=TemplateExtractTemplateSegmentsPhaseConfig(
			enabled=_as_bool(phase_extract_cfg.get("enabled", True), True),
			output_rel_root=str(phase_extract_cfg.get("output_rel_root", phase_extract_cfg.get("relpath_root", "templates/source_payloads"))),
			summary_json_relpath=str(phase_extract_cfg.get("summary_json_relpath", "context/extract_template_segments_summary.json")),
		),
		build_templates=build_templates_phase,
		quality_checks=TemplateQualityChecksPhaseConfig(
			enabled=_as_bool(phase_quality_cfg_raw.get("enabled", quality_checks.enable), quality_checks.enable),
			config=quality_checks,
		),
		analysis=TemplateAnalysisPhaseConfig(
			enabled=_as_bool(phase_analysis_cfg.get("enabled", prop_order_analysis_enabled), prop_order_analysis_enabled),
			propagation_ordering=TemplatePropagationOrderingPhaseConfig(
				enabled=_as_bool(phase_prop_order_cfg.get("enabled", prop_order_analysis_enabled), prop_order_analysis_enabled),
				latency_mode=str(phase_prop_order_cfg.get("latency_mode", analysis_ordering_latency_mode)),
				latency_tie_breaker=str(phase_prop_order_cfg.get("latency_tie_breaker", prop_order_analysis_cfg.get("latency_tie_breaker", "channel_index"))),
				debug=_as_bool(phase_prop_order_cfg.get("debug", analysis_debug_ordering), analysis_debug_ordering),
			),
		),
		plots=plot_templates_phase,
	)
	reports_phase = TemplateReportsPhaseConfig(
		enabled=_as_bool(phase_reports_cfg.get("enabled", True), True),
		summary_json_relpath=str(phase_reports_cfg.get("summary_json_relpath", "context/reports_summary.json")),
		config=reports,
		locations=TemplateLeafPhaseConfig(
			enabled=_as_bool(
				_phase_block(phase_reports_cfg, "locations").get(
					"enabled",
					bool(reports.locations.write_json or reports.locations.write_png or reports.locations.write_svg),
				),
				bool(reports.locations.write_json or reports.locations.write_png or reports.locations.write_svg),
			),
		),
		wf_overlay_grid=TemplateLeafPhaseConfig(
			enabled=_as_bool(
				_phase_block(phase_reports_cfg, "wf_overlay_grid").get(
					"enabled",
					bool(reports.wf_overlay_grid.write_pdf or reports.wf_overlay_grid.write_png or reports.wf_overlay_grid.write_svg),
				),
				bool(reports.wf_overlay_grid.write_pdf or reports.wf_overlay_grid.write_png or reports.wf_overlay_grid.write_svg),
			),
		),
		footprint_grids=TemplateLeafPhaseConfig(
			enabled=_as_bool(
				_phase_block(phase_reports_cfg, "footprint_grids").get("enabled", True),
				True,
			),
		),
		multi_source_pdf=TemplateLeafPhaseConfig(
			enabled=_as_bool(
				_phase_block(phase_reports_cfg, "multi_source_pdf").get("enabled", reports.plot_multi_source_pdf.enabled),
				reports.plot_multi_source_pdf.enabled,
			),
		),
	)
	phases = TemplatesPhasesConfig(
		resolve_sources=resolve_sources_phase,
		analyzers=analyzers_phase,
		build_templates=build_templates_phase,
		plot_templates=plot_templates_phase,
		report_templates=report_templates_phase,
		per_unit_processing=per_unit_processing_phase,
		reports=reports_phase,
	)

	return TemplatesStageConfig(
		output_rel_root=str(stage_cfg.get("output_rel_root", outputs_cfg.get("output_rel_root", "templates_outputs"))),
		analyzer_cache=analyzer_cache,
		per_unit_outputs=per_unit,
		reports=reports,
		quality_checks_outputs=_build_data_quality_checks_outputs_config(data_quality_checks_cfg),
		resolve_sources_phase=resolve_sources_phase,
		phases=phases,
		concat_analyzer_relpath=concat_analyzer_relpath,
		concat_sorting_relpath=concat_sorting_relpath,
		preprocessed_concat_reldir=preprocessed_concat_reldir,
		preprocessed_segments_reldir=preprocessed_segments_reldir,
		preproc_seg_sources_reldir=preproc_seg_sources_reldir,
		unit_ids=unit_ids,
		unit_limit=unit_limit,
		force_restart=force_restart,
		force_replot=force_replot,
		force_replot_per_unit=force_replot_per_unit,
		force_rereport=force_rereport,
		require_curated_units=require_curated_units,
		include_concat=include_concat,
		include_segments=include_segments,
		require_concat_analyzer=require_concat_analyzer,
		require_segment_analyzers=require_segment_analyzers,
		waveform_extraction=waveform_extraction,
		execution_upsampling=execution_upsampling,
		merge=merge,
		quality_checks=quality_checks,
		probe_geometry=probe_geometry,
	)


def build_templates_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: TemplatesStageConfig,
	unit_workers: int,
	probe_geometry: ProbeGeometryConfig | None = None,
) -> TemplatesInputs:
	resolved_probe_geometry = probe_geometry if probe_geometry is not None else stage_config.probe_geometry
	return TemplatesInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		final_output_root=(target.final_output_root or target.mea_output_root),
		artifact_lookup_roots=tuple(target.artifact_lookup_roots or ()),
		concat_analyzer_relpath=stage_config.concat_analyzer_relpath,
		concat_sorting_relpath=stage_config.concat_sorting_relpath,
		preprocessed_concat_reldir=stage_config.preprocessed_concat_reldir,
		preprocessed_segments_reldir=stage_config.preprocessed_segments_reldir,
		preproc_seg_sources_reldir=stage_config.preproc_seg_sources_reldir,
		output_rel_root=stage_config.output_rel_root,
		analyzer_cache=stage_config.analyzer_cache,
		per_unit_outputs=stage_config.per_unit_outputs,
		reports=stage_config.reports,
		quality_checks_outputs=stage_config.quality_checks_outputs,
		resolve_sources_phase=stage_config.resolve_sources_phase,
		phases=stage_config.phases,
		unit_ids=stage_config.unit_ids,
		unit_limit=stage_config.unit_limit,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		force_replot_per_unit=stage_config.force_replot_per_unit,
		force_rereport=stage_config.force_rereport,
		require_curated_units=stage_config.require_curated_units,
		include_concat=stage_config.include_concat,
		include_segments=stage_config.include_segments,
		require_concat_analyzer=stage_config.require_concat_analyzer,
		require_segment_analyzers=stage_config.require_segment_analyzers,
		waveform_extraction=stage_config.waveform_extraction,
		execution_upsampling=stage_config.execution_upsampling,
		merge=stage_config.merge,
		quality_checks=stage_config.quality_checks,
		probe_geometry=resolved_probe_geometry,
		n_jobs=max(1, int(unit_workers)),
	)


def load_templates_inputs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> TemplatesInputs:
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
	artifact_lookup_roots: list[Path] = []
	for lookup_roots_raw in (selected.get("output_root_2", None), data_cfg.get("output_root_2", None)):
		if lookup_roots_raw is None:
			continue
		lookup_tokens = lookup_roots_raw if isinstance(lookup_roots_raw, (list, tuple, set)) else [lookup_roots_raw]
		for token in lookup_tokens:
			if token is None:
				continue
			text = str(token).strip()
			if text == "":
				continue
			candidate = Path(text).expanduser().resolve()
			if candidate == output_root:
				continue
			if candidate in artifact_lookup_roots:
				continue
			artifact_lookup_roots.append(candidate)

	wells = selected.get("wells", [])
	stream_id = "well000"
	if isinstance(wells, list) and wells and isinstance(wells[0], dict) and wells[0].get("well_id"):
		stream_id = str(wells[0].get("well_id"))

	stage_cfg = parse_templates_stage_config(
		runtime_config=runtime_cfg,
		probe_geometry=parse_probe_geometry_from_data_config(data_config=data_cfg),
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	return TemplatesInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		final_output_root=output_root,
		artifact_lookup_roots=tuple(artifact_lookup_roots),
		concat_analyzer_relpath=stage_cfg.concat_analyzer_relpath,
		concat_sorting_relpath=stage_cfg.concat_sorting_relpath,
		preprocessed_concat_reldir=stage_cfg.preprocessed_concat_reldir,
		preprocessed_segments_reldir=stage_cfg.preprocessed_segments_reldir,
		preproc_seg_sources_reldir=stage_cfg.preproc_seg_sources_reldir,
		output_rel_root=stage_cfg.output_rel_root,
		analyzer_cache=stage_cfg.analyzer_cache,
		per_unit_outputs=stage_cfg.per_unit_outputs,
		reports=stage_cfg.reports,
		quality_checks_outputs=stage_cfg.quality_checks_outputs,
		resolve_sources_phase=stage_cfg.resolve_sources_phase,
		phases=stage_cfg.phases,
		unit_ids=stage_cfg.unit_ids,
		unit_limit=stage_cfg.unit_limit,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		force_replot_per_unit=stage_cfg.force_replot_per_unit,
		force_rereport=stage_cfg.force_rereport,
		require_curated_units=stage_cfg.require_curated_units,
		include_concat=stage_cfg.include_concat,
		include_segments=stage_cfg.include_segments,
		require_concat_analyzer=stage_cfg.require_concat_analyzer,
		require_segment_analyzers=stage_cfg.require_segment_analyzers,
		waveform_extraction=stage_cfg.waveform_extraction,
		execution_upsampling=stage_cfg.execution_upsampling,
		merge=stage_cfg.merge,
		quality_checks=stage_cfg.quality_checks,
		probe_geometry=stage_cfg.probe_geometry,
		n_jobs=1,
	)
