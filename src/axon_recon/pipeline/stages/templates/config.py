from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from ...execution.context import ExecutionTarget
from .models.inputs import (
	FootprintGridsReportConfig,
	FootprintMapGridReportConfig,
	FootprintMapConfig,
	FootprintPlotsConfig,
	MergeConfig,
	MultiSourcePdfReportConfig,
	PerUnitTemplatesOutputsConfig,
	ProbeGeometryConfig,
	PropagationAxesConfig,
	PropagationLatencyMapConfig,
	PropagationPlotConfig,
	ReportsConfig,
	TemplateArtifactConfig,
	TemplateCirclesPlotConfig,
	TemplatePlotConfig,
	TemplateWaveformOverlayConfig,
	TopographicalFootprintConfig,
	TopographicalFootprintsConfig,
	TimeUpsampleConfig,
	TemplatesInputs,
	WfOverlayGridReportConfig,
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


def _get_template_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths("per_unit_outputs.template_plots.waveforms"),
			*_output_paths("per_unit_outputs.full_template.template_plots.waveforms"),
			*_output_paths("per_unit_outputs.template"),
			*_output_paths("per_unit_outputs.template_plot"),
			"stages.reconstruct.outputs.per_unit_outputs.template",
			"stages.reconstruct.outputs.per_unit_outputs.template_plot",
		),
	)


def _get_template_circles_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths("per_unit_outputs.template_plots.circles"),
			*_output_paths("per_unit_outputs.full_template.template_plots.circles"),
			*_output_paths("per_unit_outputs.template_circles"),
		),
	)


def _get_template_wf_overlay_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths("per_unit_outputs.template_wf_overlay"),
			*_output_paths("per_unit_outputs.full_template.template_wf_overlay"),
		),
	)


def _get_reports_wf_overlay_grid_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths("reports.wf_overlay_grid"),
			*_output_paths("per_unit_outputs.reports.wf_overlay_grid"),
		),
	)


def _get_reports_footprint_grids_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = _first_dict_block(
		runtime_config,
		(
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
			*_output_paths("reports.foot_print_grids"),
			*_output_paths("per_unit_outputs.reports.foot_print_grids"),
		),
	)


def _get_reports_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths("reports"),
			*_output_paths("per_unit_outputs.reports"),
		),
	)


def _get_footprint_map_block(runtime_config: RuntimeConfig, map_name: str) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths(f"per_unit_outputs.footprint_plots.{map_name}"),
			*_output_paths(f"per_unit_outputs.full_template.footprint_plots.{map_name}"),
		),
	)


def _get_topographical_footprint_block(runtime_config: RuntimeConfig, map_name: str) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths(f"per_unit_outputs.topographical_footprints.{map_name}"),
			*_output_paths(f"per_unit_outputs.full_template.topographical_footprints.{map_name}"),
		),
	)


def _get_propagation_plots_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths("per_unit_outputs.propagation_plots"),
			*_output_paths("per_unit_outputs.full_template.propagation_plots"),
		),
	)


def _get_template_artifact_block(runtime_config: RuntimeConfig, block_name: str) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		_output_paths(f"per_unit_outputs.{block_name}"),
	)


def _get_merge_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = runtime_config.get("stages.templates.execution.merge", {})
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
		write_png=_as_bool(raw_cfg.get("write_png", True), True),
		write_svg=_as_bool(raw_cfg.get("write_svg", False), False),
		relpath=str(raw_cfg.get("relpath", relpath_default)),
		background=str(raw_cfg.get("background", "black")),
		color_map=str(raw_cfg.get("color_map", "viridis")),
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
	return TimeUpsampleConfig(
		enabled=bool(enabled),
		factor=int(factor),
		method=str(raw_cfg.get("method", "sinc")),
	)


def _build_topographical_footprint_config(raw_cfg: dict[str, Any], *, relpath_default: str) -> TopographicalFootprintConfig:
	template_cfg = _get_nested_block(raw_cfg, "template")
	color_bar_cfg = _get_nested_block(raw_cfg, "color_bar")
	return TopographicalFootprintConfig(
		write_png=_as_bool(raw_cfg.get("write_png", True), True),
		write_svg=_as_bool(raw_cfg.get("write_svg", False), False),
		relpath=str(raw_cfg.get("relpath", relpath_default)),
		background=str(raw_cfg.get("background", "black")),
		color_map=str(raw_cfg.get("color_map", "viridis")),
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
		elevation_deg=_as_float(raw_cfg.get("elevation_deg", 35.0), 35.0),
		azimuth_deg=_as_float(raw_cfg.get("azimuth_deg", -60.0), -60.0),
		marker_size=_as_float(raw_cfg.get("marker_size", 14.0), 14.0),
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


def _build_template_artifact_config(raw_cfg: dict[str, Any], *, relpath_default: str) -> TemplateArtifactConfig:
	return TemplateArtifactConfig(
		write_npy=_as_bool(raw_cfg.get("write_npy", False), False),
		npy_relpath=str(raw_cfg.get("npy_relpath", relpath_default)),
		padding_value=_normalize_padding_value(raw_cfg.get("padding_value", "zero")),
	)


def _get_unit_reldir(runtime_config: RuntimeConfig) -> str:
	raw = runtime_config.get("stages.templates.outputs.per_unit_outputs.unit_reldir", None)
	if raw is None:
		raw = runtime_config.get("stages.outputs.per_unit_outputs.unit_reldir", None)
	if raw is not None and str(raw).strip() != "":
		return str(raw)

	legacy = runtime_config.get("stages.reconstruct.outputs.per_unit_outputs.unit_reldir", None)
	if legacy is not None and str(legacy).strip() != "":
		return str(legacy)

	return "units/{unit_id:04d}/"


@dataclass(frozen=True)
class TemplatesStageConfig:
	output_rel_root: str
	per_unit_outputs: PerUnitTemplatesOutputsConfig
	reports: ReportsConfig
	unit_ids: list[int] | None
	unit_limit: int | None
	force_restart: bool
	force_replot: bool
	force_replot_per_unit: bool
	require_curated_units: bool
	include_concat: bool
	include_segments: bool
	merge: MergeConfig
	probe_geometry: ProbeGeometryConfig | None = None


def parse_probe_geometry_from_data_config(*, data_config: RuntimeConfig) -> ProbeGeometryConfig | None:
	probe_cfg = data_config.get("Probe", {})
	if not isinstance(probe_cfg, dict) or not probe_cfg:
		return None
	pitch_um = _as_float_or_none(probe_cfg.get("pitch_um", None), None)
	elec_cfg = probe_cfg.get("electrode_size_um", {}) if isinstance(probe_cfg.get("electrode_size_um", {}), dict) else {}
	electrode_size_um_x = _as_float_or_none(elec_cfg.get("x", None), None)
	electrode_size_um_y = _as_float_or_none(elec_cfg.get("y", None), None)
	if electrode_size_um_x is None and pitch_um is not None:
		electrode_size_um_x = float(pitch_um * 0.7)
	if electrode_size_um_y is None and pitch_um is not None:
		electrode_size_um_y = float(pitch_um * 0.7)
	active_cfg = probe_cfg.get("active_sensing_area_mm", {}) if isinstance(probe_cfg.get("active_sensing_area_mm", {}), dict) else {}
	active_area_um_x = _as_float_or_none(active_cfg.get("x", None), None)
	active_area_um_y = _as_float_or_none(active_cfg.get("y", None), None)
	sampling_rate_hz = _as_float_or_none(probe_cfg.get("sampling_rate_hz", None), None)
	if active_area_um_x is not None:
		active_area_um_x *= 1000.0
	if active_area_um_y is not None:
		active_area_um_y *= 1000.0
	if (
		electrode_size_um_x is None
		and electrode_size_um_y is None
		and pitch_um is None
		and active_area_um_x is None
		and active_area_um_y is None
		and sampling_rate_hz is None
	):
		return None
	return ProbeGeometryConfig(
		pitch_um=pitch_um,
		electrode_size_um_x=electrode_size_um_x,
		electrode_size_um_y=electrode_size_um_y,
		active_area_um_x=active_area_um_x,
		active_area_um_y=active_area_um_y,
		sampling_rate_hz=sampling_rate_hz,
	)


def parse_templates_stage_config(
	*,
	runtime_config: RuntimeConfig,
	probe_geometry: ProbeGeometryConfig | None = None,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> TemplatesStageConfig:
	stage_cfg = runtime_config.get("stages.templates", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}
	if not outputs_cfg:
		outputs_cfg = runtime_config.get("stages.outputs", {}) if isinstance(runtime_config.get("stages.outputs", {}), dict) else {}

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	force_replot_per_unit = _as_bool(execution_cfg.get("force_replot_per_unit", False), False)
	require_curated_units = _as_bool(execution_cfg.get("require_curated_units", True), True)
	spk_tpl_sources = execution_cfg.get("spikeinterface", {}) if isinstance(execution_cfg.get("spikeinterface", {}), dict) else {}
	spk_tpl_extract = spk_tpl_sources.get("template_extraction", {}) if isinstance(spk_tpl_sources.get("template_extraction", {}), dict) else {}
	spk_tpl_extract_sources = spk_tpl_extract.get("sources", {}) if isinstance(spk_tpl_extract.get("sources", {}), dict) else {}
	include_concat = _as_bool(spk_tpl_extract_sources.get("include_concat", True), True)
	include_segments = _as_bool(spk_tpl_extract_sources.get("include_segments", True), True)
	merge_cfg = _get_merge_block(runtime_config)
	merge = MergeConfig(
		enable=_as_bool(merge_cfg.get("enable", True), True),
		method=str(merge_cfg.get("method", "mean_all_waveforms")),
		centering_method=str(merge_cfg.get("centering_method", "pre_peak_robust_baseline")),
		weighting_mode=str(merge_cfg.get("weighting_mode", "per_channel_waveform_count")),
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

	unit_ids = [int(unit_id_override)] if unit_id_override is not None else None

	tpl_cfg = _get_template_block(runtime_config)
	tpl_circles_cfg = _get_template_circles_block(runtime_config)
	tpl_wf_overlay_cfg = _get_template_wf_overlay_block(runtime_config)
	report_overlay_grid_cfg = _get_reports_wf_overlay_grid_block(runtime_config)
	reports_cfg = _get_reports_block(runtime_config)
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
		write_png=_as_bool(tpl_cfg.get("write_png", True), True),
		write_svg=_as_bool(tpl_cfg.get("write_svg", False), False),
		relpath=str(tpl_cfg.get("relpath", "template")),
		channel_scope=_normalize_channel_scope(tpl_cfg.get("channel_scope", "contributing_channels")),
		background=str(tpl_cfg.get("background", "black")),
		signal_color=str(tpl_cfg.get("signal_color", "white")),
		force_center_soma=_as_bool(tpl_cfg.get("force_center_soma", False), False),
		force_square_aspect=_as_bool(tpl_cfg.get("force_square_aspect", True), True),
		show_scale_bar=_as_bool(tpl_cfg.get("show_scale_bar", True), True),
		scale_bar_color=str(tpl_cfg.get("scale_bar_color", "white")),
		scale_bar_text_offset_frac=_as_float(tpl_cfg.get("scale_bar_text_offset_frac", 0.02), 0.02),
		scale_bar_y_offset_frac=_as_float(tpl_cfg.get("scale_bar_y_offset_frac", 0.06), 0.06),
		scale_bar_fontsize=_as_float(tpl_cfg.get("scale_bar_fontsize", 6.0), 6.0),
		scale_bar_linewidth=_as_float(tpl_cfg.get("scale_bar_linewidth", 1.8), 1.8),
		scale_bar_length_um=(
			None
			if tpl_cfg.get("scale_bar_length_um", None) is None
			else _as_float(tpl_cfg.get("scale_bar_length_um", None), 0.0)
		),
	)
	tpl_circles = TemplateCirclesPlotConfig(
		# Keep circles-specific nested config under color_bar for runtime ergonomics.
		# Example:
		# circles:
		#   color_bar:
		#     units: ms
		write_png=_as_bool(tpl_circles_cfg.get("write_png", False), False),
		write_svg=_as_bool(tpl_circles_cfg.get("write_svg", False), False),
		relpath=str(tpl_circles_cfg.get("relpath", "template_circles")),
		channel_scope=_normalize_channel_scope(tpl_circles_cfg.get("channel_scope", "contributing_channels")),
		background=str(tpl_circles_cfg.get("background", "black")),
		signal_color=str(tpl_circles_cfg.get("signal_color", "white")),
		force_center_soma=_as_bool(tpl_circles_cfg.get("force_center_soma", False), False),
		force_square_aspect=_as_bool(tpl_circles_cfg.get("force_square_aspect", True), True),
		show_scale_bar=_as_bool(tpl_circles_cfg.get("show_scale_bar", True), True),
		scale_bar_color=str(tpl_circles_cfg.get("scale_bar_color", "white")),
		scale_bar_text_offset_frac=_as_float(tpl_circles_cfg.get("scale_bar_text_offset_frac", 0.02), 0.02),
		scale_bar_y_offset_frac=_as_float(tpl_circles_cfg.get("scale_bar_y_offset_frac", 0.06), 0.06),
		scale_bar_fontsize=_as_float(tpl_circles_cfg.get("scale_bar_fontsize", 6.0), 6.0),
		scale_bar_linewidth=_as_float(tpl_circles_cfg.get("scale_bar_linewidth", 1.8), 1.8),
		scale_bar_length_um=(
			None
			if tpl_circles_cfg.get("scale_bar_length_um", None) is None
			else _as_float(tpl_circles_cfg.get("scale_bar_length_um", None), 0.0)
		),
		circle_size_scale_factor=max(0.0, _as_float(tpl_circles_cfg.get("circle_size_scale_factor", 1.0), 1.0)),
		size_by=_normalize_template_metric(tpl_circles_cfg.get("size_by", "amplitude"), "amplitude"),
		color_by=_normalize_template_metric(tpl_circles_cfg.get("color_by", "latency"), "latency"),
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
	)
	tpl_wf_overlay = TemplateWaveformOverlayConfig(
		write_pdf=_as_bool(tpl_wf_overlay_cfg.get("write_pdf", False), False),
		pdf_relpath=str(tpl_wf_overlay_cfg.get("pdf_relpath", "template_wf_overlay.pdf")),
		write_png=_as_bool(tpl_wf_overlay_cfg.get("write_png", True), True),
		png_relpath=str(tpl_wf_overlay_cfg.get("png_relpath", "template_wf_overlay.png")),
		top_channels_per_template=max(1, _as_int(tpl_wf_overlay_cfg.get("top_channels_per_template", 10), 10)),
		style=str(tpl_wf_overlay_cfg.get("style", "overlay")),
		include_mean=_as_bool(tpl_wf_overlay_cfg.get("include_mean", True), True),
		include_scale_bar=_as_bool(tpl_wf_overlay_cfg.get("include_scale_bar", True), True),
		scale_bar_color=str(tpl_wf_overlay_cfg.get("scale_bar_color", "black")),
		scale_bar_fontsize=_as_float(tpl_wf_overlay_cfg.get("scale_bar_fontsize", 6.0), 6.0),
		scale_bar_linewidth=_as_float(tpl_wf_overlay_cfg.get("scale_bar_linewidth", 1.8), 1.8),
		background=str(tpl_wf_overlay_cfg.get("background", "white")),
	)
	reports = ReportsConfig(
		plot_multi_source_pdf=MultiSourcePdfReportConfig(
			enabled=_as_bool(reports_cfg.get("plot_multi_source_pdf", False), False),
			pdf_relpath=str(reports_cfg.get("multi_source_pdf_relpath", "reports/template_multi_source.pdf")),
		),
		replot_from_disk=_as_bool(reports_cfg.get("replot_from_disk", False), False),
		time_upsample=_build_time_upsample_config(time_upsample_cfg_raw),
		wf_overlay_grid=WfOverlayGridReportConfig(
			write_pdf=_as_bool(report_overlay_grid_cfg.get("write_pdf", False), False),
			pdf_relpath=str(report_overlay_grid_cfg.get("pdf_relpath", "wf_overlay_grid.pdf")),
			write_png=_as_bool(report_overlay_grid_cfg.get("write_png", True), True),
			png_relpath=str(report_overlay_grid_cfg.get("png_relpath", "wf_overlay_grid.png")),
			top_channels_per_template=max(1, _as_int(report_overlay_grid_cfg.get("top_channels_per_template", 10), 10)),
		),
		footprint_grids=FootprintGridsReportConfig(
			amplitude_map_grid=FootprintMapGridReportConfig(
				write_pdf=_as_bool((footprint_grids_cfg.get("amplitude_map_grid", {}) if isinstance(footprint_grids_cfg.get("amplitude_map_grid", {}), dict) else {}).get("write_pdf", False), False),
				pdf_relpath=str((footprint_grids_cfg.get("amplitude_map_grid", {}) if isinstance(footprint_grids_cfg.get("amplitude_map_grid", {}), dict) else {}).get("pdf_relpath", "amplitude_map_grid.pdf")),
				write_png=_as_bool((footprint_grids_cfg.get("amplitude_map_grid", {}) if isinstance(footprint_grids_cfg.get("amplitude_map_grid", {}), dict) else {}).get("write_png", True), True),
				png_relpath=str((footprint_grids_cfg.get("amplitude_map_grid", {}) if isinstance(footprint_grids_cfg.get("amplitude_map_grid", {}), dict) else {}).get("png_relpath", "amplitude_map_grid.png")),
				template_shape=_normalize_template_shape((footprint_grids_cfg.get("amplitude_map_grid", {}) if isinstance(footprint_grids_cfg.get("amplitude_map_grid", {}), dict) else {}).get("template_shape", (footprint_grids_cfg.get("amplitude_map_grid", {}) if isinstance(footprint_grids_cfg.get("amplitude_map_grid", {}), dict) else {}).get("template", "square")), "square"),
				global_color_scale=_as_bool((footprint_grids_cfg.get("amplitude_map_grid", {}) if isinstance(footprint_grids_cfg.get("amplitude_map_grid", {}), dict) else {}).get("global_color_scale", True), True),
			),
			latency_map_grid=FootprintMapGridReportConfig(
				write_pdf=_as_bool((footprint_grids_cfg.get("latency_map_grid", {}) if isinstance(footprint_grids_cfg.get("latency_map_grid", {}), dict) else {}).get("write_pdf", False), False),
				pdf_relpath=str((footprint_grids_cfg.get("latency_map_grid", {}) if isinstance(footprint_grids_cfg.get("latency_map_grid", {}), dict) else {}).get("pdf_relpath", "latency_map_grid.pdf")),
				write_png=_as_bool((footprint_grids_cfg.get("latency_map_grid", {}) if isinstance(footprint_grids_cfg.get("latency_map_grid", {}), dict) else {}).get("write_png", True), True),
				png_relpath=str((footprint_grids_cfg.get("latency_map_grid", {}) if isinstance(footprint_grids_cfg.get("latency_map_grid", {}), dict) else {}).get("png_relpath", "latency_map_grid.png")),
				template_shape=_normalize_template_shape((footprint_grids_cfg.get("latency_map_grid", {}) if isinstance(footprint_grids_cfg.get("latency_map_grid", {}), dict) else {}).get("template_shape", (footprint_grids_cfg.get("latency_map_grid", {}) if isinstance(footprint_grids_cfg.get("latency_map_grid", {}), dict) else {}).get("template", "square")), "square"),
				global_color_scale=_as_bool((footprint_grids_cfg.get("latency_map_grid", {}) if isinstance(footprint_grids_cfg.get("latency_map_grid", {}), dict) else {}).get("global_color_scale", True), True),
			),
		),
	)
	footprint_plots = FootprintPlotsConfig(
		amplitude_map=_build_footprint_map_config(amp_map_cfg, relpath_default="footprint_amplitude_map"),
		latency_map=_build_footprint_map_config(lat_map_cfg, relpath_default="footprint_latency_map"),
	)
	topographical_footprints = TopographicalFootprintsConfig(
		amplitude=_build_topographical_footprint_config(topo_amp_cfg, relpath_default="topographical_amplitude_footprint"),
		latency=_build_topographical_footprint_config(topo_lat_cfg, relpath_default="topographical_latency_footprint"),
	)
	propagation_plots = PropagationPlotConfig(
		write_pdf=_as_bool(propagation_cfg.get("write_pdf", False), False),
		pdf_relpath=str(propagation_cfg.get("pdf_relpath", "propagation_plot.pdf")),
		write_png=_as_bool(propagation_cfg.get("write_png", True), True),
		png_relpath=str(propagation_cfg.get("png_relpath", "propagation_plot.png")),
		top_channels=max(1, _as_int(propagation_cfg.get("top_channels", 25), 25)),
		channels_per_panel=max(1, _as_int(propagation_cfg.get("channels_per_panel", 25), 25)),
		channel_overlap=max(0, _as_int(propagation_cfg.get("channel_overlap", 5), 5)),
		background=str(propagation_cfg.get("background", "white")),
		show_electrode_ids=_as_bool(propagation_cfg.get("show_electrode_ids", False), False),
		trace_gain=_as_float(propagation_cfg.get("trace_gain", 1.0), 1.0),
		trace_spacing=_as_float(propagation_cfg.get("trace_spacing", 1.0), 1.0),
		latency_map=_build_propagation_latency_map_config(
			_get_nested_block(propagation_cfg, "latency_map")
		),
	)

	per_unit = PerUnitTemplatesOutputsConfig(
		unit_reldir=_get_unit_reldir(runtime_config),
		merged_template=_build_template_artifact_config(merged_template_cfg, relpath_default="merged_template.npy"),
		square_template=_build_template_artifact_config(square_template_cfg, relpath_default="square_template.npy"),
		scan_template=_build_template_artifact_config(scan_template_cfg, relpath_default="scan_template.npy"),
		full_template=_build_template_artifact_config(full_template_cfg, relpath_default="full_template.npy"),
		template=tpl,
		template_circles=tpl_circles,
		template_wf_overlay=tpl_wf_overlay,
		footprint_plots=footprint_plots,
		topographical_footprints=topographical_footprints,
		propagation_plots=propagation_plots,
	)

	return TemplatesStageConfig(
		output_rel_root=str(outputs_cfg.get("output_rel_root", "templates_outputs")),
		per_unit_outputs=per_unit,
		reports=reports,
		unit_ids=unit_ids,
		unit_limit=unit_limit,
		force_restart=force_restart,
		force_replot=force_replot,
		force_replot_per_unit=force_replot_per_unit,
		require_curated_units=require_curated_units,
		include_concat=include_concat,
		include_segments=include_segments,
		merge=merge,
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
		output_rel_root=stage_config.output_rel_root,
		per_unit_outputs=stage_config.per_unit_outputs,
		reports=stage_config.reports,
		unit_ids=stage_config.unit_ids,
		unit_limit=stage_config.unit_limit,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		force_replot_per_unit=stage_config.force_replot_per_unit,
		require_curated_units=stage_config.require_curated_units,
		include_concat=stage_config.include_concat,
		include_segments=stage_config.include_segments,
		merge=stage_config.merge,
		probe_geometry=resolved_probe_geometry,
		n_jobs=max(1, int(unit_workers)),
	)


def load_templates_inputs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
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

	wells = selected.get("wells", [])
	stream_id = "well000"
	if isinstance(wells, list) and wells and isinstance(wells[0], dict) and wells[0].get("well_id"):
		stream_id = str(wells[0].get("well_id"))

	stage_cfg = parse_templates_stage_config(
		runtime_config=runtime_cfg,
		probe_geometry=parse_probe_geometry_from_data_config(data_config=data_cfg),
		unit_id_override=unit_id_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	return TemplatesInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		output_rel_root=stage_cfg.output_rel_root,
		per_unit_outputs=stage_cfg.per_unit_outputs,
		reports=stage_cfg.reports,
		unit_ids=stage_cfg.unit_ids,
		unit_limit=stage_cfg.unit_limit,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		force_replot_per_unit=stage_cfg.force_replot_per_unit,
		require_curated_units=stage_cfg.require_curated_units,
		include_concat=stage_cfg.include_concat,
		include_segments=stage_cfg.include_segments,
		merge=stage_cfg.merge,
		probe_geometry=stage_cfg.probe_geometry,
		n_jobs=1,
	)
