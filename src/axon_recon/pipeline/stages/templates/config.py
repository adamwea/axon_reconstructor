from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig
from axon_recon.pipeline.shared.plotting import build_stage_plot_block

from ...execution.context import ExecutionTarget
from .models.inputs import (
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
	PropagationAxesConfig,
	PropagationLatencyMapConfig,
	PropagationPlotConfig,
	QualityCheckJsonOutputConfig,
	QualityCheckPlotOutputConfig,
	QualityChecksConfig,
	ReportsConfig,
	TemplateArtifactConfig,
	TemplateCirclesPlotConfig,
	TemplatePlotConfig,
	TemplateWaveformOverlayConfig,
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


def _get_template_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
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
			*_output_paths("per_unit_outputs.quality_checks"),
		),
	)


def _get_data_quality_checks_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths("data_outputs.quality_checks"),
		),
	)


def _get_reports_wf_overlay_grid_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	return _first_dict_block(
		runtime_config,
		(
			*_output_paths("reports.grids.wf_overlay_grid"),
			*_output_paths("reports.wf_overlay_grid"),
			*_output_paths("per_unit_outputs.reports.wf_overlay_grid"),
		),
	)


def _get_reports_footprint_grids_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = _first_dict_block(
		runtime_config,
		(
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
			*_output_paths("reports.grids.foot_print_grids"),
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
	return build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
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


def _build_waveform_extraction_config(
	*,
	execution_cfg: dict[str, Any],
	runtime_config: RuntimeConfig,
) -> WaveformExtractionConfig:
	spikeinterface_cfg = execution_cfg.get("spikeinterface", {}) if isinstance(execution_cfg.get("spikeinterface", {}), dict) else {}
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
	multiple_enabled_raw = multiple_peaks_cfg.get("enable", multiple_peaks_cfg.get("enabled", None))
	if multiple_enabled_raw is None:
		multiple_enabled = bool(_as_bool(enabled_raw, False))
	else:
		multiple_enabled = bool(_as_bool(multiple_enabled_raw, False))
	return QualityChecksConfig(
		enable=_as_bool(enabled_raw, False),
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
	quality_checks_outputs: DataQualityChecksOutputsConfig
	unit_ids: list[int] | None
	unit_limit: int | None
	force_restart: bool
	force_replot: bool
	force_replot_per_unit: bool
	require_curated_units: bool
	include_concat: bool
	include_segments: bool
	waveform_extraction: WaveformExtractionConfig
	execution_upsampling: TimeUpsampleConfig
	merge: MergeConfig
	quality_checks: QualityChecksConfig
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
	waveform_extraction = _build_waveform_extraction_config(
		execution_cfg=execution_cfg,
		runtime_config=runtime_config,
	)
	execution_upsampling_cfg = execution_cfg.get("upsampling", {}) if isinstance(execution_cfg.get("upsampling", {}), dict) else {}
	execution_upsampling = _build_time_upsample_config(execution_upsampling_cfg)
	quality_checks_cfg_raw = execution_cfg.get("quality_checks", {}) if isinstance(execution_cfg.get("quality_checks", {}), dict) else {}
	quality_checks = _build_quality_checks_config(quality_checks_cfg_raw)
	merge_cfg = _get_merge_block(runtime_config)
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
		force_center_soma=_as_bool(_nested_or_flat(tpl_circles_cfg, block="display", key="force_center_soma", flat_keys=("force_center_soma",), default=False), False),
		force_square_aspect=_as_bool(_nested_or_flat(tpl_circles_cfg, block="display", key="force_square_aspect", flat_keys=("force_square_aspect",), default=True), True),
		show_scale_bar=_as_bool(_nested_or_flat(tpl_circles_cfg, block="display", key="show_scale_bar", flat_keys=("show_scale_bar",), default=True), True),
		scale_bar_color=str(_nested_or_flat(tpl_circles_cfg, block="render", key="scale_bar_color", flat_keys=("scale_bar_color",), default="white")),
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
	reports = ReportsConfig(
		plot_multi_source_pdf=MultiSourcePdfReportConfig(
			enabled=_as_bool(reports_cfg.get("plot_multi_source_pdf", False), False),
			pdf_relpath=str(reports_cfg.get("multi_source_pdf_relpath", "reports/template_multi_source.pdf")),
		),
		replot_from_disk=_as_bool(reports_cfg.get("replot_from_disk", False), False),
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
				flat_keys=("ordering_latency_mode",),
				default="abs_peak",
			)
		),
		debug_ordering=_as_bool(
			_nested_or_flat(
				propagation_cfg,
				block="display",
				key="debug_ordering",
				flat_keys=("debug_ordering",),
				default=False,
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

	return TemplatesStageConfig(
		output_rel_root=str(outputs_cfg.get("output_rel_root", "templates_outputs")),
		per_unit_outputs=per_unit,
		reports=reports,
		quality_checks_outputs=_build_data_quality_checks_outputs_config(data_quality_checks_cfg),
		unit_ids=unit_ids,
		unit_limit=unit_limit,
		force_restart=force_restart,
		force_replot=force_replot,
		force_replot_per_unit=force_replot_per_unit,
		require_curated_units=require_curated_units,
		include_concat=include_concat,
		include_segments=include_segments,
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
		output_rel_root=stage_config.output_rel_root,
		per_unit_outputs=stage_config.per_unit_outputs,
		reports=stage_config.reports,
		quality_checks_outputs=stage_config.quality_checks_outputs,
		unit_ids=stage_config.unit_ids,
		unit_limit=stage_config.unit_limit,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		force_replot_per_unit=stage_config.force_replot_per_unit,
		require_curated_units=stage_config.require_curated_units,
		include_concat=stage_config.include_concat,
		include_segments=stage_config.include_segments,
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
		quality_checks_outputs=stage_cfg.quality_checks_outputs,
		unit_ids=stage_cfg.unit_ids,
		unit_limit=stage_cfg.unit_limit,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		force_replot_per_unit=stage_cfg.force_replot_per_unit,
		require_curated_units=stage_cfg.require_curated_units,
		include_concat=stage_cfg.include_concat,
		include_segments=stage_cfg.include_segments,
		waveform_extraction=stage_cfg.waveform_extraction,
		execution_upsampling=stage_cfg.execution_upsampling,
		merge=stage_cfg.merge,
		quality_checks=stage_cfg.quality_checks,
		probe_geometry=stage_cfg.probe_geometry,
		n_jobs=1,
	)
