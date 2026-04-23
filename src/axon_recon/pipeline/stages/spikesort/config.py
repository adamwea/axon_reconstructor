from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from ...execution.context import ExecutionTarget
from .models.inputs import SpikesortInputs


_DEFAULT_OUTPUT_REL_ROOT = "spikesort_outputs"
_LEGACY_OUTPUT_REL_ROOT = "stg2_spikesorting_outputs"


LOGGER = logging.getLogger("axon_recon.spikesort.config")


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


def _as_int(value: Any, default: int) -> int:
	if value is None:
		return int(default)
	try:
		return int(value)
	except Exception:
		return int(default)


def _as_optional_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except Exception:
		return None


def _as_optional_positive_int(value: Any) -> int | None:
	parsed = _as_optional_int(value)
	if parsed is None:
		return None
	return (int(parsed) if int(parsed) > 0 else None)


def _as_optional_float(value: Any) -> float | None:
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _as_optional_str(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text if text else None


def _as_optional_dict(value: Any) -> dict[str, Any] | None:
	if value is None:
		return None
	if isinstance(value, dict):
		return dict(value)
	return None


def _as_section(value: Any) -> dict[str, Any]:
	if isinstance(value, dict) and value:
		return dict(value)
	return {}


def _coalesce(*values: Any) -> Any:
	for value in values:
		if value is not None:
			return value
	return None


def _get_nested_value(raw: dict[str, Any], path: tuple[str, ...]) -> Any:
	cursor: Any = raw
	for key in path:
		if not isinstance(cursor, dict):
			return None
		if key not in cursor:
			return None
		cursor = cursor.get(key)
	return cursor


def _normalize_optional_relpath(raw: Any) -> str | None:
	text = _as_optional_str(raw)
	if text is None:
		return None
	normalized = text.lstrip("/")
	return normalized or None


def _normalize_merge_analyzer_density_mode(raw: Any) -> str:
	if isinstance(raw, bool):
		return ("dense" if raw else "auto")
	token = str(raw or "auto").strip().lower()
	if token in {"dense", "full"}:
		return "dense"
	return "auto"


def _normalize_merge_template_random_spikes_method(raw: Any) -> str:
	if isinstance(raw, bool):
		return ("all" if raw else "default")
	token = str(raw or "default").strip().lower()
	if token in {"percentage", "percent", "fraction", "proportion"}:
		return "percentage"
	if token in {"all", "full", "every"}:
		return "all"
	return "default"


def _parse_merge_template_random_spikes_percentage(raw: Any, *, field_name: str) -> float | None:
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


def _warn_legacy_merge_template_random_spikes_percentage_alias() -> None:
	LOGGER.warning(
		"stages.spikesort.phases.merge_units.analyzer.template_extraction.min_perc_spikes_per_unit is deprecated; "
		"prefer random_spikes_percentage instead."
	)


def _warn_legacy_merge_analyzer_density_mode_alias() -> None:
	LOGGER.warning(
		"stages.spikesort.phases.merge_units.analyzer.template_extraction.density_mode is deprecated; "
		"prefer sparsity.compute_sparsity and template_extraction.random_spikes_method instead."
	)


def _normalize_merge_analyzer_sparsity_method(raw: Any) -> str:
	token = str(raw or "radius").strip().lower()
	if token in {"best", "best_channel", "best_channels", "num_channels"}:
		return "best_channels"
	if token in {"threshold", "snr"}:
		return "threshold"
	if token in {"by_property", "property", "group"}:
		return "by_property"
	return "radius"


def _normalize_merge_analyzer_peak_sign(raw: Any) -> str:
	token = str(raw or "neg").strip().lower()
	if token in {"pos", "positive"}:
		return "pos"
	if token in {"both", "all"}:
		return "both"
	return "neg"


def _normalize_merge_template_heatmap_magnitude_mode(raw: Any) -> str:
	token = str(raw or "ptp").strip().lower()
	if token in {"peak_to_peak", "ptp"}:
		return "ptp"
	if token in {"abs_peak", "absolute_peak", "extremum"}:
		return "abs_peak"
	if token in {"peak", "positive_peak", "max"}:
		return "peak"
	if token in {"trough", "negative_peak", "neg_peak", "min"}:
		return "trough"
	return "ptp"


def _as_list_of_strings(value: Any) -> list[str]:
	if value is None:
		return []
	if isinstance(value, str):
		return [token.strip() for token in value.split(",") if token.strip()]
	if isinstance(value, (list, tuple, set)):
		out: list[str] = []
		for item in value:
			text = _as_optional_str(item)
			if text is not None:
				out.append(text)
		return out
	text = _as_optional_str(value)
	return ([text] if text is not None else [])


def _as_optional_float_tuple(value: Any) -> tuple[float, ...] | None:
	if value is None:
		return None
	items: list[Any]
	if isinstance(value, str):
		items = [token.strip() for token in value.split(",") if token.strip()]
	elif isinstance(value, (list, tuple, set)):
		items = list(value)
	else:
		items = [value]
	parsed: list[float] = []
	for item in items:
		val = _as_optional_float(item)
		if val is not None:
			parsed.append(float(val))
	return (tuple(parsed) if parsed else None)


def _get_with_fallback(primary: dict[str, Any], fallback: dict[str, Any], key: str, default: Any) -> Any:
	if key in primary:
		return primary.get(key)
	if key in fallback:
		return fallback.get(key)
	return default


def _normalize_output_rel_root(raw: Any) -> str:
	text = str(raw or _DEFAULT_OUTPUT_REL_ROOT).strip()
	if not text:
		return _DEFAULT_OUTPUT_REL_ROOT
	text = text.lstrip("/")
	if text == _LEGACY_OUTPUT_REL_ROOT:
		return _DEFAULT_OUTPUT_REL_ROOT
	return text or _DEFAULT_OUTPUT_REL_ROOT


def _resolve_data_config_path(runtime_config_path: Path, data_ref: str | None) -> Path:
	if not data_ref:
		raise ValueError("Runtime config must define data: <path-to-data-config>")
	p = Path(str(data_ref)).expanduser()
	if not p.is_absolute():
		p = (runtime_config_path.parent / p).resolve()
	return p


@dataclass(frozen=True)
class SpikesortStageConfig:
	output_rel_root: str
	preprocess_concat_recording_relpath: str | None
	merge_sequence: tuple[str, ...]
	logging_enabled: bool
	logging_verbose: bool
	logging_file_relpath: str | None
	debug_limit_wells: int | None
	sort_debug_mode_enabled: bool
	sort_debug_limit_datasets: int | None
	sort_debug_limit_wells: int | None
	summarize_sort_debug_mode_enabled: bool
	summarize_sort_debug_limit_datasets: int | None
	summarize_sort_debug_limit_wells: int | None
	sorter: str
	docker_image: str | None
	recording_num: str
	verbose: bool

	ks_batch_duration_s: float | None
	ks_batch_size: int | None
	ks_th_universal: float | None
	ks_th_learned: float | None
	ks_th_single_ch: float | None
	ks_cluster_downsampling: int | None
	ks_nearest_chans: int | None
	ks_max_channel_distance: float | None

	n_jobs: int | None
	chunk_duration: str | None
	cuda_visible_devices: str | None

	run_analyzer: bool
	run_reports: bool
	sort_enabled: bool
	sort_delete_outputs_on_force_restart: bool
	plot_enabled: bool
	plot_mode: str
	plot_debug: bool
	raster_sort: str | None
	fixed_y: bool
	no_curation: bool
	export_to_phy: bool
	force_rerun_analyzer: bool
	summarize_sort_enabled: bool
	summarize_sort_emit_logs: bool
	summarize_sort_generate_artifacts: bool
	bombcell_label_enabled: bool
	bombcell_label_relpath: str
	bombcell_label_delete_outputs_on_force_restart: bool
	bombcell_label_thresholds: dict[str, Any] | None
	bombcell_label_thresholds_path: str | None
	bombcell_label_label_non_somatic: bool
	bombcell_label_split_non_somatic_good_mua: bool
	bombcell_label_apply_to_sorter_output: bool
	bombcell_label_write_cluster_group: bool
	bombcell_label_fail_on_error: bool
	bombcell_label_reports_enabled: bool
	bombcell_label_reports_summary_json_enabled: bool
	bombcell_label_reports_summary_json_relpath: str
	um_kwargs: dict[str, Any] | None
	am_kwargs: dict[str, Any] | None
	option_kwargs: dict[str, Any] | None
	slay_enabled: bool
	slay_relpath: str
	slay_package_root: str | None
	slay_sorter_output_relpath: str | None
	slay_output_json_relpath: str
	slay_candidate_pairs_relpath: str
	slay_merge_groups_relpath: str
	slay_allow_numpy_fallback: bool
	slay_plot_merges: bool
	slay_auto_accept_merges: bool
	slay_copy_automerge_artifacts: bool
	slay_delete_outputs_on_force_restart: bool
	slay_recompute_analyzer: bool
	slay_model_cache_relpath: str | None
	slay_model_cache_use_cached_model: bool
	slay_model_cache_write_model: bool
	slay_force_restart_retrain_model: bool
	slay_params: dict[str, Any] | None
	auto_merge_enabled: bool
	auto_merge_relpath: str
	auto_merge_delete_outputs_on_force_restart: bool
	auto_merge_candidate_pairs_reldir: str
	auto_merge_merged_units_reldir: str
	auto_merge_auto_accept_merges: bool
	auto_merge_template_diff_thresholds: tuple[float, ...]
	merge_units_enabled: bool
	merge_rel_output_root: str | None
	merge_delete_outputs_on_force_restart: bool
	merge_force_restart: bool
	merge_force_replot: bool
	merge_analyzer_regenerate_on_replot: bool
	merge_analyzer_check_if_regen_is_needed: bool
	merge_analyzer_compute_sparsity: bool
	# Deprecated compatibility alias; no longer used as the canonical control surface.
	merge_analyzer_density_mode: str
	merge_template_random_spikes_method: str
	merge_template_random_spikes_percentage: float | None
	merge_template_random_spikes_max_spikes_per_unit: int | None
	merge_template_random_spikes_min_spikes_per_unit: int | None
	merge_template_random_spikes_log_before_after_spike_counts: bool
	merge_template_random_spikes_margin_size: int | None
	merge_template_random_spikes_seed: int | None
	merge_analyzer_n_jobs: int | None
	merge_analyzer_chunk_duration: str | None
	merge_analyzer_sparsity_method: str
	merge_analyzer_sparsity_radius_um: float | None
	merge_analyzer_sparsity_num_channels: int | None
	merge_analyzer_sparsity_threshold: float | None
	merge_analyzer_sparsity_peak_sign: str
	merge_analyzer_sparsity_num_spikes_for_sparsity: int | None
	merge_analyzer_sparsity_by_property: str | None
	merge_analyzer_waveforms_ms_before: float | None
	merge_analyzer_waveforms_ms_after: float | None
	merge_analyzer_waveforms_dtype: str | None
	cache_sorting_outputs_before_merge: bool
	cache_sorting_outputs_before_merge_relpath: str
	cache_sorting_outputs_before_merge_cleanup_on_success: bool
	cache_sorting_outputs_before_merge_use_cache_on_force_restart: bool
	cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart: bool
	cache_sorting_outputs_before_merge_refresh_on_run: bool
	cache_sorting_outputs_before_merge_strict_restore_on_force_restart: bool
	cache_sorting_outputs_before_merge_use_canonical_workspace: bool
	cache_sorting_outputs_before_merge_canonical_workspace_relpath: str
	cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run: bool
	cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer: bool
	cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success: bool
	cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure: bool
	cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace: bool
	cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace: bool
	merge_reports_enabled: bool
	merge_reports_unit_diff_json_enabled: bool
	merge_reports_unit_diff_json_relpath: str
	merge_reports_unit_diff_map_enabled: bool
	merge_reports_unit_diff_map_relpath: str
	merge_reports_unit_diff_map_flat_enabled: bool
	merge_reports_unit_diff_map_flat_relpath: str
	merge_reports_post_merge_unit_locations_enabled: bool
	merge_reports_post_merge_unit_locations_relpath: str
	merge_reports_2panel_enabled: bool
	merge_reports_2panel_point_size: float
	merge_reports_2panel_relpath: str
	merge_reports_2panel_label_pre_and_post_units: bool
	merge_reports_2panel_write_png: bool
	merge_reports_2panel_write_svg: bool
	merge_reports_2panel_before_relpath: str
	merge_reports_2panel_before_write_png: bool
	merge_reports_2panel_before_write_svg: bool
	merge_reports_2panel_before_point_color: str
	merge_reports_2panel_after_relpath: str
	merge_reports_2panel_after_write_png: bool
	merge_reports_2panel_after_write_svg: bool
	merge_reports_2panel_after_point_color: str
	merge_reports_2panel_highlight_merges_enabled: bool
	merge_reports_2panel_highlight_merges_linked: bool
	merge_reports_2panel_highlight_plot_after_other_units: bool
	merge_reports_2panel_highlight_label_affected_units: bool
	merge_reports_2panel_highlight_show_legend: bool
	merge_reports_2panel_highlight_legend_position: str
	merge_reports_2panel_highlight_legend_x: float
	merge_reports_2panel_highlight_legend_y: float
	merge_reports_2panel_highlight_sort_pre_legend_by_groups: bool
	merge_reports_2panel_highlight_debug_json_enabled: bool
	merge_reports_2panel_highlight_debug_json_relpath: str
	merge_reports_2panel_highlight_before_color: str
	merge_reports_2panel_highlight_after_color: str
	merge_reports_2panel_highlight_palette: str
	merge_reports_2panel_inherit_probe_dimensions: bool
	merge_reports_2panel_zoom_to_affected_units: bool
	merge_reports_2panel_probe_dim_x_um: float | None
	merge_reports_2panel_probe_dim_y_um: float | None
	merge_reports_template_heatmaps_enabled: bool
	merge_reports_template_heatmaps_relpath: str
	merge_reports_template_heatmaps_assets_reldir: str
	merge_reports_template_heatmaps_write_png: bool
	merge_reports_template_heatmaps_write_svg: bool
	merge_reports_template_heatmaps_write_assets_png: bool
	merge_reports_template_heatmaps_write_assets_svg: bool
	merge_reports_template_heatmaps_panel_width_in: float
	merge_reports_template_heatmaps_panel_height_in: float
	merge_reports_template_heatmaps_marker_size: float
	merge_reports_template_heatmaps_cmap: str
	merge_reports_template_heatmaps_show_colorbar: bool
	merge_reports_template_heatmaps_relative_color_bar_height: float
	merge_reports_template_heatmaps_color_scale: str
	merge_reports_template_heatmaps_log_epsilon: float
	merge_reports_template_heatmaps_magnitude_mode: str
	merge_reports_template_heatmaps_max_merges: int | None
	merge_reports_template_heatmaps_debug_json_relpath: str
	merge_reports_template_heatmaps_inherit_probe_dimensions: bool
	merge_reports_template_heatmaps_probe_dim_x_um: float | None
	merge_reports_template_heatmaps_probe_dim_y_um: float | None
	merge_reports_template_heatmaps_probe_pitch_um: float | None
	merge_reports_template_heatmaps_probe_electrode_size_um_x: float | None
	merge_reports_template_heatmaps_probe_electrode_size_um_y: float | None
	merge_metadata_enabled: bool
	merge_metadata_write_json: bool
	merge_metadata_json_relpath: str
	merge_metadata_include_unit_locations: bool
	merge_metadata_log_summary_details: bool
	pre_merge_metadata_enabled: bool
	pre_merge_metadata_write_json: bool
	pre_merge_metadata_json_relpath: str
	pre_merge_metadata_include_unit_locations: bool
	pre_merge_metadata_log_summary_details: bool
	post_merge_metadata_enabled: bool
	post_merge_metadata_write_json: bool
	post_merge_metadata_json_relpath: str
	post_merge_metadata_include_unit_locations: bool
	post_merge_metadata_log_summary_details: bool

	force_restart: bool
	force_replot: bool
	resume_from: str | None


def parse_spikesort_stage_config(
	*,
	runtime_config: RuntimeConfig,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> SpikesortStageConfig:
	stage_cfg = runtime_config.get("stages.spikesort", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	execution_cfg = _as_section(stage_cfg.get("execution", {}))
	phases_cfg = _as_section(stage_cfg.get("phases", {}))
	sort_phase_cfg = _as_section(phases_cfg.get("sort", {}))
	merge_units_phase_cfg = _as_section(phases_cfg.get("merge_units", {}))
	merge_analyzer_cfg = _as_section(merge_units_phase_cfg.get("analyzer", {}))
	merge_analyzer_waveforms_cfg = _as_section(merge_analyzer_cfg.get("waveforms", {}))
	merge_analyzer_sparsity_cfg = _as_section(merge_analyzer_cfg.get("sparsity", {}))
	merge_analyzer_template_extraction_cfg = _as_section(
		merge_analyzer_cfg.get("template_extraction", {})
	)
	summarize_sort_phase_cfg = _as_section(phases_cfg.get("summarize_sort", {}))
	bombcell_phase_cfg_raw = phases_cfg.get("bombcell_label", None)
	bombcell_phase_cfg = _as_section(bombcell_phase_cfg_raw)
	bombcell_params_cfg = _as_section(bombcell_phase_cfg.get("params", {}))
	bombcell_reports_cfg = _as_section(bombcell_phase_cfg.get("reports", {}))
	bombcell_reports_summary_json_cfg = _as_section(bombcell_reports_cfg.get("summary_json", {}))
	resources_cfg = _as_section(stage_cfg.get("resources", {}))
	logging_cfg = _as_section(stage_cfg.get("logging", {}))
	debug_cfg = _as_section(stage_cfg.get("debug", {}))
	inputs_cfg = _as_section(stage_cfg.get("inputs", {}))
	execution_inputs_cfg = _as_section(execution_cfg.get("inputs", {}))
	resolved_inputs_cfg = dict(execution_inputs_cfg)
	resolved_inputs_cfg.update(inputs_cfg)
	plot_cfg = _as_section(stage_cfg.get("plot", {}))
	execution_plot_cfg = _as_section(execution_cfg.get("plot", {}))
	sort_phase_plot_cfg = _as_section(sort_phase_cfg.get("plot", {}))
	if execution_plot_cfg:
		plot_cfg = dict(plot_cfg)
		plot_cfg.update(execution_plot_cfg)
	if sort_phase_plot_cfg:
		plot_cfg = dict(plot_cfg)
		plot_cfg.update(sort_phase_plot_cfg)
	report_cfg = _as_section(stage_cfg.get("report", {}))
	execution_report_cfg = _as_section(execution_cfg.get("report", {}))
	sort_phase_report_cfg = _as_section(sort_phase_cfg.get("report", {}))
	if execution_report_cfg:
		report_cfg = dict(report_cfg)
		report_cfg.update(execution_report_cfg)
	if sort_phase_report_cfg:
		report_cfg = dict(report_cfg)
		report_cfg.update(sort_phase_report_cfg)
	outputs_cfg = _as_section(stage_cfg.get("outputs", {}))

	stage_kilosort_cfg = _as_section(stage_cfg.get("kilosort", {}))
	execution_kilosort_cfg = _as_section(execution_cfg.get("kilosort", {}))
	sort_phase_kilosort_cfg = _as_section(sort_phase_cfg.get("kilosort", {}))

	stage_unitmatch_cfg = _as_section(stage_cfg.get("unitmatch", {}))
	execution_unitmatch_cfg = _as_section(execution_cfg.get("unitmatch", {}))
	merge_phase_unitmatch_cfg = _as_section(merge_units_phase_cfg.get("unitmatch", {}))
	unitmatch_cfg = dict(stage_unitmatch_cfg)
	unitmatch_cfg.update(execution_unitmatch_cfg)
	unitmatch_cfg.update(merge_phase_unitmatch_cfg)
	unitmatch_limits_cfg = _as_section(unitmatch_cfg.get("limits", {}))
	unitmatch_iterations_cfg = _as_section(unitmatch_cfg.get("iterations", {}))

	stage_auto_merge_cfg = _as_section(stage_cfg.get("auto_merge", {}))
	execution_auto_merge_cfg = _as_section(execution_cfg.get("auto_merge", {}))
	merge_phase_auto_merge_cfg = _as_section(merge_units_phase_cfg.get("auto_merge", {}))
	cache_sorting_outputs_cfg_raw = merge_units_phase_cfg.get("cache_sorting_outputs_before_merge", None)
	cache_sorting_outputs_cfg = _as_section(cache_sorting_outputs_cfg_raw)
	canonical_workspace_cfg_raw = merge_units_phase_cfg.get("use_cache_as_canonical_workspace", None)
	canonical_workspace_cfg = _as_section(canonical_workspace_cfg_raw)
	merge_reports_cfg = _as_section(merge_units_phase_cfg.get("reports", {}))
	merge_reports_unit_diff_json_cfg = _as_section(merge_reports_cfg.get("unit_diff_json", {}))
	merge_reports_unit_diff_map_cfg = _as_section(merge_reports_cfg.get("unit_diff_map", {}))
	merge_reports_unit_diff_map_flat_cfg = _as_section(merge_reports_cfg.get("unit_diff_map_flat", {}))
	merge_reports_post_merge_unit_locations_cfg = _as_section(merge_reports_cfg.get("post_merge_unit_locations", {}))
	merge_reports_2panel_cfg = _as_section(merge_reports_cfg.get("2panels_unit_locations_before_after_merge", {}))
	merge_reports_2panel_assets_cfg = _as_section(merge_reports_2panel_cfg.get("assets", {}))
	merge_reports_2panel_before_cfg = _as_section(merge_reports_2panel_assets_cfg.get("before", {}))
	merge_reports_2panel_after_cfg = _as_section(merge_reports_2panel_assets_cfg.get("after", {}))
	merge_reports_2panel_highlight_cfg = _as_section(merge_reports_2panel_cfg.get("highlight_merges", {}))
	merge_reports_template_heatmaps_cfg = _as_section(merge_reports_cfg.get("template_heatmaps_per_merge", {}))
	merge_reports_template_heatmaps_assets_cfg = _as_section(
		merge_reports_template_heatmaps_cfg.get("assets", {})
	)
	merge_reports_template_heatmaps_electrode_size_cfg = _as_section(
		merge_reports_template_heatmaps_cfg.get("electrode_size_um", {})
	)
	merge_metadata_cfg = _as_section(merge_units_phase_cfg.get("merge_metadata", {}))
	pre_merge_metadata_cfg = _as_section(merge_units_phase_cfg.get("pre_merge_metadata", {}))
	post_merge_metadata_cfg = _as_section(merge_units_phase_cfg.get("post_merge_metadata", {}))
	auto_merge_cfg = dict(stage_auto_merge_cfg)
	auto_merge_cfg.update(execution_auto_merge_cfg)
	auto_merge_cfg.update(merge_phase_auto_merge_cfg)

	stage_slay_cfg = _as_section(
		_coalesce(
			stage_cfg.get("SLAy", None),
			stage_cfg.get("slay", None),
			{},
		)
	)
	execution_slay_cfg = _as_section(
		_coalesce(
			execution_cfg.get("SLAy", None),
			execution_cfg.get("slay", None),
			{},
		)
	)
	merge_phase_slay_cfg = _as_section(
		_coalesce(
			merge_units_phase_cfg.get("SLAy", None),
			merge_units_phase_cfg.get("slay", None),
			{},
		)
	)
	slay_cfg = dict(stage_slay_cfg)
	slay_cfg.update(execution_slay_cfg)
	slay_cfg.update(merge_phase_slay_cfg)
	slay_model_cache_cfg = _as_section(slay_cfg.get("model_cache", {}))

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if force_replot_override is not None:
		force_replot = bool(force_replot_override)

	legacy_debug_default = _as_bool(execution_cfg.get("debug", False), False)
	logging_enabled = _as_bool(logging_cfg.get("enabled", True), True)
	logging_verbose = _as_bool(logging_cfg.get("verbose", legacy_debug_default), legacy_debug_default)
	logging_file_relpath = _as_optional_str(logging_cfg.get("file_relpath", None))
	preprocess_concat_recording_relpath = _normalize_optional_relpath(
		_coalesce(
			resolved_inputs_cfg.get("preprocess_concat_recording_relpath", None),
			resolved_inputs_cfg.get("preprocess_concat_recording_reldir", None),
			resolved_inputs_cfg.get("preprocessed_concat_recording_relpath", None),
		)
	)

	debug_limit_wells = _as_optional_positive_int(
		_coalesce(
			execution_cfg.get("limit_wells", None),
			debug_cfg.get("limit_wells", None),
		)
	)
	sort_debug_cfg = _as_section(sort_phase_cfg.get("debug_mode", {}))
	sort_debug_mode_enabled = _as_bool(sort_debug_cfg.get("enabled", False), False)
	sort_debug_limit_datasets = _as_optional_positive_int(sort_debug_cfg.get("limit_datasets", None))
	sort_debug_limit_wells = _as_optional_positive_int(sort_debug_cfg.get("limit_wells", None))
	summarize_sort_debug_cfg = _as_section(summarize_sort_phase_cfg.get("debug_mode", {}))
	summarize_sort_debug_mode_enabled = _as_bool(summarize_sort_debug_cfg.get("enabled", False), False)
	summarize_sort_debug_limit_datasets = _as_optional_positive_int(
		summarize_sort_debug_cfg.get("limit_datasets", None)
	)
	summarize_sort_debug_limit_wells = _as_optional_positive_int(
		summarize_sort_debug_cfg.get("limit_wells", None)
	)

	plot_enabled = _as_bool(plot_cfg.get("enabled", True), True)
	sort_enabled = _as_bool(
		_coalesce(
			sort_phase_cfg.get("enabled", None),
			stage_cfg.get("sort_enabled", None),
			True,
		),
		True,
	)
	sort_delete_outputs_on_force_restart = _as_bool(
		_coalesce(
			sort_phase_cfg.get("delete_outputs_on_force_restart", None),
			sort_phase_cfg.get("delete_on_force_restart", None),
			execution_cfg.get("delete_outputs_on_force_restart", None),
			execution_cfg.get("sort_delete_outputs_on_force_restart", None),
			stage_cfg.get("delete_outputs_on_force_restart", None),
			stage_cfg.get("sort_delete_outputs_on_force_restart", None),
			False,
		),
		False,
	)
	plot_mode = _as_optional_str(plot_cfg.get("mode", plot_cfg.get("plot_mode", "separate"))) or "separate"
	plot_debug = _as_bool(plot_cfg.get("debug", plot_cfg.get("plot_debug", False)), False)
	raster_sort = _as_optional_str(plot_cfg.get("raster_sort", None))
	fixed_y = _as_bool(plot_cfg.get("fixed_y", False), False)

	run_reports = _as_bool(_get_with_fallback(execution_cfg, stage_cfg, "run_reports", True), True)
	run_reports = _as_bool(
		_coalesce(
			sort_phase_cfg.get("run_reports", None),
			execution_cfg.get("run_reports", None),
			stage_cfg.get("run_reports", None),
			run_reports,
		),
		True,
	)
	if "enabled" in report_cfg:
		run_reports = _as_bool(report_cfg.get("enabled", run_reports), run_reports)
	if not bool(plot_enabled):
		run_reports = False

	no_curation_default = _as_bool(_get_with_fallback(execution_cfg, stage_cfg, "no_curation", False), False)
	no_curation = no_curation_default
	if "no_curation" in report_cfg:
		no_curation = _as_bool(report_cfg.get("no_curation", no_curation_default), no_curation_default)

	export_to_phy = _as_bool(_get_with_fallback(execution_cfg, stage_cfg, "export_to_phy", False), False)
	if "export_to_phy" in report_cfg:
		export_to_phy = _as_bool(report_cfg.get("export_to_phy", export_to_phy), export_to_phy)

	force_rerun_analyzer = _as_bool(
		_coalesce(
			merge_units_phase_cfg.get("force_rerun_analyzer", None),
			merge_units_phase_cfg.get("rerun_analyzer", None),
			execution_cfg.get("force_rerun_analyzer", None),
			execution_cfg.get("rerun_analyzer", None),
			stage_cfg.get("force_rerun_analyzer", None),
			stage_cfg.get("rerun_analyzer", None),
			False,
		),
		False,
	)
	summarize_sort_enabled = _as_bool(
		_coalesce(
			summarize_sort_phase_cfg.get("enabled", None),
			False,
		),
		False,
	)
	summarize_sort_emit_logs = _as_bool(
		_coalesce(
			summarize_sort_phase_cfg.get("emit_logs", None),
			True,
		),
		True,
	)
	summarize_sort_generate_artifacts = _as_bool(
		_coalesce(
			summarize_sort_phase_cfg.get("generate_artifacts", None),
			False,
		),
		False,
	)
	unitmatch_enabled = _as_bool(
		_coalesce(
			unitmatch_cfg.get("enabled", None),
			merge_units_phase_cfg.get("unitmatch_enabled", None),
			execution_cfg.get("unitmatch_enabled", None),
			stage_cfg.get("unitmatch_enabled", None),
			True,
		),
		True,
	)

	um_kwargs = (
		_as_optional_dict(
			_coalesce(
				merge_units_phase_cfg.get("um_kwargs", None),
				execution_cfg.get("um_kwargs", None),
				stage_cfg.get("um_kwargs", None),
			)
		)
		or {}
	)
	um_kwargs.setdefault("enabled", bool(unitmatch_enabled))
	requested_merge_units = _as_bool(
		_coalesce(
			unitmatch_cfg.get("merge_units", None),
			merge_units_phase_cfg.get("unitmatch_merge_units", None),
			stage_cfg.get("unitmatch_merge_units", None),
			False,
		),
		False,
	)
	if not bool(unitmatch_enabled):
		um_kwargs["merge_units"] = False
	else:
		um_kwargs.setdefault("merge_units", requested_merge_units)
	um_kwargs.setdefault(
		"dry_run",
		_as_bool(
			_coalesce(
				unitmatch_cfg.get("dry_run", None),
				merge_units_phase_cfg.get("unitmatch_dry_run", None),
				stage_cfg.get("unitmatch_dry_run", None),
				True,
			),
			True,
		),
	)
	um_kwargs.setdefault(
		"scored_dry_run",
		_as_bool(
			_coalesce(
				unitmatch_cfg.get("scored_dry_run", None),
				merge_units_phase_cfg.get("unitmatch_scored_dry_run", None),
				stage_cfg.get("unitmatch_scored_dry_run", None),
				True,
			),
			True,
		),
	)
	um_output_subdir = _as_optional_str(
		_coalesce(
			unitmatch_cfg.get("output_subdir_name", None),
			merge_units_phase_cfg.get("unitmatch_output_subdir_name", None),
			stage_cfg.get("unitmatch_output_subdir_name", None),
			"unitmatch_outputs",
		)
	)
	if um_output_subdir is not None:
		um_kwargs.setdefault("output_subdir_name", um_output_subdir)
	um_throughput_subdir = _as_optional_str(
		_coalesce(
			unitmatch_cfg.get("throughput_subdir_name", None),
			merge_units_phase_cfg.get("unitmatch_throughput_subdir_name", None),
			stage_cfg.get("unitmatch_throughput_subdir_name", None),
			"unitmatch_throughput",
		)
	)
	if um_throughput_subdir is not None:
		um_kwargs.setdefault("throughput_subdir_name", um_throughput_subdir)
	um_oversplit_prob = _as_optional_float(
		_coalesce(
			unitmatch_cfg.get("oversplit_min_probability", None),
			merge_units_phase_cfg.get("unitmatch_oversplit_min_probability", None),
			stage_cfg.get("unitmatch_oversplit_min_probability", None),
		)
	)
	if um_oversplit_prob is not None:
		um_kwargs.setdefault("oversplit_min_probability", float(um_oversplit_prob))
	um_apply_merges = _as_bool(
		_coalesce(
			unitmatch_cfg.get("apply_merges", None),
			merge_units_phase_cfg.get("unitmatch_apply_merges", None),
			stage_cfg.get("unitmatch_apply_merges", None),
			False,
		),
		False,
	)
	um_kwargs.setdefault("apply_merges", um_apply_merges)
	um_recursive = _as_bool(
		_coalesce(
			unitmatch_cfg.get("recursive", None),
			merge_units_phase_cfg.get("unitmatch_recursive", None),
			stage_cfg.get("unitmatch_recursive", None),
			False,
		),
		False,
	)
	um_kwargs.setdefault("recursive", um_recursive)
	um_keep_all_iterations = _as_bool(
		_coalesce(
			unitmatch_cfg.get("keep_all_iterations", None),
			_get_nested_value(unitmatch_iterations_cfg, ("keep_all",)),
			merge_units_phase_cfg.get("unitmatch_keep_all_iterations", None),
			stage_cfg.get("unitmatch_keep_all_iterations", None),
			True,
		),
		True,
	)
	um_kwargs.setdefault("keep_all_iterations", um_keep_all_iterations)
	um_max_candidate_pairs = _as_optional_int(
		_coalesce(
			unitmatch_cfg.get("max_candidate_pairs", None),
			unitmatch_limits_cfg.get("max_candidate_pairs", None),
			merge_units_phase_cfg.get("unitmatch_max_candidate_pairs", None),
			stage_cfg.get("unitmatch_max_candidate_pairs", None),
		)
	)
	if um_max_candidate_pairs is not None:
		um_kwargs.setdefault("max_candidate_pairs", int(um_max_candidate_pairs))
	um_oversplit_max_suggestions = _as_optional_int(
		_coalesce(
			unitmatch_cfg.get("oversplit_max_suggestions", None),
			unitmatch_limits_cfg.get("oversplit_max_suggestions", None),
			merge_units_phase_cfg.get("unitmatch_oversplit_max_suggestions", None),
			stage_cfg.get("unitmatch_oversplit_max_suggestions", None),
		)
	)
	if um_oversplit_max_suggestions is not None:
		um_kwargs.setdefault("oversplit_max_suggestions", int(um_oversplit_max_suggestions))
	um_max_iterations = _as_optional_int(
		_coalesce(
			unitmatch_cfg.get("max_iterations", None),
			unitmatch_iterations_cfg.get("max", None),
			merge_units_phase_cfg.get("unitmatch_max_iterations", None),
			stage_cfg.get("unitmatch_max_iterations", None),
		)
	)
	if um_max_iterations is not None:
		um_kwargs.setdefault("max_iterations", int(um_max_iterations))
	um_max_spikes = _as_optional_int(
		_coalesce(
			unitmatch_cfg.get("max_spikes_per_unit", None),
			merge_units_phase_cfg.get("unitmatch_max_spikes_per_unit", None),
			stage_cfg.get("unitmatch_max_spikes_per_unit", None),
		)
	)
	if um_max_spikes is not None:
		um_kwargs.setdefault("max_spikes_per_unit", int(um_max_spikes))

	auto_merge_enabled = _as_bool(
		_coalesce(
			auto_merge_cfg.get("enabled", None),
			merge_units_phase_cfg.get("auto_merge_units", None),
			execution_cfg.get("auto_merge_units", None),
			stage_cfg.get("auto_merge_units", None),
			False,
		),
		False,
	)
	auto_merge_template_diff_thresh = _as_optional_str(
		_coalesce(
			auto_merge_cfg.get("template_diff_thresh", None),
			merge_units_phase_cfg.get("auto_merge_template_diff_thresh", None),
			execution_cfg.get("auto_merge_template_diff_thresh", None),
			stage_cfg.get("auto_merge_template_diff_thresh", None),
		)
	)
	auto_merge_template_diff_thresholds = (
		_as_optional_float_tuple(auto_merge_template_diff_thresh)
		or _as_optional_float_tuple(auto_merge_cfg.get("template_diff_thresh", None))
		or (0.25,)
	)
	auto_merge_relpath = _normalize_optional_relpath(
		_coalesce(
			auto_merge_cfg.get("relpath", None),
			auto_merge_cfg.get("output_relpath", None),
			"automerge_outputs",
		)
	) or "automerge_outputs"
	auto_merge_delete_outputs_on_force_restart = _as_bool(
		_coalesce(
			auto_merge_cfg.get("delete_outputs_on_force_restart", None),
			auto_merge_cfg.get("delete_on_force_restart", None),
			True,
		),
		True,
	)
	auto_merge_candidate_pairs_reldir = _normalize_optional_relpath(
		_coalesce(
			auto_merge_cfg.get("candidate_pairs_reldir", None),
			auto_merge_cfg.get("candidate_pairs_relpath", None),
			"recommended_merge_candidates",
		)
	) or "recommended_merge_candidates"
	auto_merge_merged_units_reldir = _normalize_optional_relpath(
		_coalesce(
			auto_merge_cfg.get("merged_units_reldir", None),
			auto_merge_cfg.get("merged_units_relpath", None),
			"merged_units",
		)
	) or "merged_units"
	auto_merge_auto_accept_merges = _as_bool(
		_coalesce(
			auto_merge_cfg.get("auto_accept_merges", None),
			auto_merge_cfg.get("apply_merges", None),
			False,
		),
		False,
	)
	merge_units_enabled = _as_bool(
		_coalesce(
			merge_units_phase_cfg.get("enabled", None),
			execution_cfg.get("merge_units_enabled", None),
			stage_cfg.get("merge_units_enabled", None),
			True,
		),
		True,
	)
	merge_rel_output_root = _normalize_optional_relpath(
		_coalesce(
			merge_units_phase_cfg.get("rel_output_root", None),
			merge_units_phase_cfg.get("output_rel_root", None),
			merge_units_phase_cfg.get("merge_rel_output_root", None),
			merge_units_phase_cfg.get("merge_output_rel_root", None),
			execution_cfg.get("merge_rel_output_root", None),
			execution_cfg.get("merge_output_rel_root", None),
			stage_cfg.get("merge_rel_output_root", None),
			stage_cfg.get("merge_output_rel_root", None),
			None,
		)
	)
	merge_delete_outputs_on_force_restart = _as_bool(
		_coalesce(
			merge_units_phase_cfg.get("delete_outputs_on_force_restart", None),
			merge_units_phase_cfg.get("delete_on_force_restart", None),
			execution_cfg.get("merge_delete_outputs_on_force_restart", None),
			stage_cfg.get("merge_delete_outputs_on_force_restart", None),
			False,
		),
		False,
	)
	merge_force_restart = _as_bool(
		_coalesce(
			merge_units_phase_cfg.get("force_restart", None),
			merge_units_phase_cfg.get("merge_force_restart", None),
			execution_cfg.get("merge_force_restart", None),
			stage_cfg.get("merge_force_restart", None),
			False,
		),
		False,
	)
	merge_force_replot = _as_bool(
		_coalesce(
			merge_units_phase_cfg.get("force_replot", None),
			merge_units_phase_cfg.get("merge_force_replot", None),
			execution_cfg.get("merge_force_replot", None),
			stage_cfg.get("merge_force_replot", None),
			False,
		),
		False,
	)
	# Merge-phase force flags inherit global spikesort execution toggles.
	merge_force_restart = bool(force_restart or merge_force_restart)
	merge_force_replot = bool(force_replot or merge_force_replot)
	cache_sorting_outputs_before_merge = _as_bool(
		_coalesce(
			cache_sorting_outputs_cfg.get("enabled", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_enabled", None),
			(
				cache_sorting_outputs_cfg_raw
				if not isinstance(cache_sorting_outputs_cfg_raw, dict)
				else None
			),
			execution_cfg.get("cache_sorting_outputs_before_merge", None),
			stage_cfg.get("cache_sorting_outputs_before_merge", None),
			False,
		),
		False,
	)
	cache_sorting_outputs_before_merge_relpath = _normalize_optional_relpath(
		_coalesce(
			cache_sorting_outputs_cfg.get("relpath", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_relpath", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_relpath", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_relpath", None),
			"pre_merge_cache",
		)
	) or "pre_merge_cache"
	cache_sorting_outputs_before_merge_cleanup_on_success = _as_bool(
		_coalesce(
			cache_sorting_outputs_cfg.get("cleanup_on_success", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_cleanup_on_success", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_cleanup_on_success", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_cleanup_on_success", None),
			False,
		),
		False,
	)
	cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart = _as_bool(
		_coalesce(
			cache_sorting_outputs_cfg.get("replace_sorting_with_cache_before_force_restart", None),
			cache_sorting_outputs_cfg.get("use_cache_on_force_restart", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_use_cache_on_force_restart", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_use_cache_on_force_restart", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_use_cache_on_force_restart", None),
			False,
		),
		False,
	)
	cache_sorting_outputs_before_merge_use_cache_on_force_restart = bool(
		cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart
	)
	cache_sorting_outputs_before_merge_refresh_on_run = _as_bool(
		_coalesce(
			cache_sorting_outputs_cfg.get("refresh_on_run", None),
			cache_sorting_outputs_cfg.get("refresh_cache_on_run", None),
			cache_sorting_outputs_cfg.get("overwrite_existing_cache", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_refresh_on_run", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_refresh_on_run", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_refresh_on_run", None),
			False,
		),
		False,
	)
	cache_sorting_outputs_before_merge_strict_restore_on_force_restart = _as_bool(
		_coalesce(
			cache_sorting_outputs_cfg.get("strict_restore_on_force_restart", None),
			cache_sorting_outputs_cfg.get("require_cache_on_force_restart", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_strict_restore_on_force_restart", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_strict_restore_on_force_restart", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_strict_restore_on_force_restart", None),
			True,
		),
		True,
	)
	cache_sorting_outputs_before_merge_use_canonical_workspace = _as_bool(
		_coalesce(
			canonical_workspace_cfg.get("enabled", None),
			(
				canonical_workspace_cfg_raw
				if not isinstance(canonical_workspace_cfg_raw, dict)
				else None
			),
			cache_sorting_outputs_cfg.get("use_as_canonical_workspace", None),
			cache_sorting_outputs_cfg.get("run_merge_in_cached_workspace", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_use_canonical_workspace", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_use_canonical_workspace", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_use_canonical_workspace", None),
			False,
		),
		False,
	)
	cache_sorting_outputs_before_merge_canonical_workspace_relpath = _normalize_optional_relpath(
		_coalesce(
			canonical_workspace_cfg.get("canonical_workspace_relpath", None),
			canonical_workspace_cfg.get("workspace_relpath", None),
			canonical_workspace_cfg.get("relpath", None),
			cache_sorting_outputs_cfg.get("canonical_workspace_relpath", None),
			cache_sorting_outputs_cfg.get("workspace_relpath", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_relpath", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_relpath", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_relpath", None),
			"cache/merge_canonical_workspace",
		)
	) or "cache/merge_canonical_workspace"
	cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run = _as_bool(
		_coalesce(
			canonical_workspace_cfg.get("canonical_workspace_refresh_on_run", None),
			canonical_workspace_cfg.get("refresh_on_run", None),
			cache_sorting_outputs_cfg.get("canonical_workspace_refresh_on_run", None),
			cache_sorting_outputs_cfg.get("canonical_workspace_always_refresh_on_run", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run", None),
			True,
		),
		True,
	)
	cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer = _as_bool(
		_coalesce(
			canonical_workspace_cfg.get("canonical_workspace_rebuild_analyzer", None),
			canonical_workspace_cfg.get("rebuild_analyzer", None),
			cache_sorting_outputs_cfg.get("canonical_workspace_rebuild_analyzer", None),
			cache_sorting_outputs_cfg.get("canonical_workspace_recompute_analyzer", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer", None),
			True,
		),
		True,
	)
	cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success = _as_bool(
		_coalesce(
			canonical_workspace_cfg.get("publish_to_stage_outputs_on_success", None),
			cache_sorting_outputs_cfg.get("publish_to_stage_outputs_on_success", None),
			cache_sorting_outputs_cfg.get("publish_canonical_to_stage_outputs_on_success", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success", None),
			False,
		),
		False,
	)
	cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure = _as_bool(
		_coalesce(
			canonical_workspace_cfg.get("publish_to_stage_outputs_on_failure", None),
			cache_sorting_outputs_cfg.get("publish_to_stage_outputs_on_failure", None),
			cache_sorting_outputs_cfg.get("publish_canonical_to_stage_outputs_on_failure", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure", None),
			False,
		),
		False,
	)
	cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace = _as_bool(
		_coalesce(
			canonical_workspace_cfg.get("assert_slay_uses_canonical_workspace", None),
			canonical_workspace_cfg.get("assert_slay_uses_workspace", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace", None),
			True,
		),
		True,
	)
	cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace = _as_bool(
		_coalesce(
			canonical_workspace_cfg.get("assert_auto_merge_uses_canonical_workspace", None),
			canonical_workspace_cfg.get("assert_auto_merge_uses_workspace", None),
			merge_units_phase_cfg.get("cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace", None),
			execution_cfg.get("cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace", None),
			stage_cfg.get("cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace", None),
			True,
		),
		True,
	)
	merge_reports_enabled = _as_bool(
		_coalesce(
			merge_reports_cfg.get("enabled", None),
			False,
		),
		False,
	)
	merge_reports_unit_diff_json_enabled = _as_bool(
		_coalesce(
			merge_reports_unit_diff_json_cfg.get("enabled", None),
			False,
		),
		False,
	)
	merge_reports_unit_diff_json_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_unit_diff_json_cfg.get("relpath", None),
			"unit_diffs_after_merge.json",
		)
	) or "unit_diffs_after_merge.json"
	merge_reports_unit_diff_map_enabled = _as_bool(
		_coalesce(
			merge_reports_unit_diff_map_cfg.get("enabled", None),
			False,
		),
		False,
	)
	merge_reports_unit_diff_map_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_unit_diff_map_cfg.get("relpath", None),
			"unit_diff_map.json",
		)
	) or "unit_diff_map.json"
	merge_reports_unit_diff_map_flat_enabled = _as_bool(
		_coalesce(
			merge_reports_unit_diff_map_flat_cfg.get("enabled", None),
			False,
		),
		False,
	)
	merge_reports_unit_diff_map_flat_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_unit_diff_map_flat_cfg.get("relpath", None),
			"unit_diff_map_flat.json",
		)
	) or "unit_diff_map_flat.json"
	merge_reports_post_merge_unit_locations_enabled = _as_bool(
		_coalesce(
			merge_reports_post_merge_unit_locations_cfg.get("enabled", None),
			False,
		),
		False,
	)
	merge_reports_post_merge_unit_locations_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_post_merge_unit_locations_cfg.get("relpath", None),
			"post_merge_unit_locations.json",
		)
	) or "post_merge_unit_locations.json"
	merge_reports_2panel_enabled = _as_bool(
		_coalesce(
			merge_reports_2panel_cfg.get("enabled", None),
			False,
		),
		False,
	)
	merge_reports_2panel_point_size = _as_optional_float(
		_coalesce(
			merge_reports_2panel_cfg.get("point_size", None),
			9.0,
		)
	)
	if merge_reports_2panel_point_size is None or float(merge_reports_2panel_point_size) <= 0.0:
		merge_reports_2panel_point_size = 9.0
	merge_reports_2panel_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_2panel_cfg.get("relpath", None),
			"unit_locations_before_after_merge.png",
		)
	) or "unit_locations_before_after_merge.png"
	merge_reports_2panel_label_pre_and_post_units = _as_bool(
		_coalesce(
			merge_reports_2panel_cfg.get("label_pre_and_post_units", None),
			False,
		),
		False,
	)
	merge_reports_2panel_write_png = _as_bool(
		_coalesce(
			merge_reports_2panel_cfg.get("write_png", None),
			True,
		),
		True,
	)
	merge_reports_2panel_write_svg = _as_bool(
		_coalesce(
			merge_reports_2panel_cfg.get("write_svg", None),
			False,
		),
		False,
	)
	merge_reports_2panel_before_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_2panel_before_cfg.get("relpath", None),
			"unit_locations_before_merge.png",
		)
	) or "unit_locations_before_merge.png"
	merge_reports_2panel_before_write_png = _as_bool(
		_coalesce(
			merge_reports_2panel_before_cfg.get("write_png", None),
			True,
		),
		True,
	)
	merge_reports_2panel_before_write_svg = _as_bool(
		_coalesce(
			merge_reports_2panel_before_cfg.get("write_svg", None),
			False,
		),
		False,
	)
	merge_reports_2panel_before_point_color = _as_optional_str(
		_coalesce(
			merge_reports_2panel_before_cfg.get("point_color", None),
			merge_reports_2panel_cfg.get("before_point_color", None),
			"#7a7a7a",
		)
	) or "#7a7a7a"
	merge_reports_2panel_after_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_2panel_after_cfg.get("relpath", None),
			"unit_locations_after_merge.png",
		)
	) or "unit_locations_after_merge.png"
	merge_reports_2panel_after_write_png = _as_bool(
		_coalesce(
			merge_reports_2panel_after_cfg.get("write_png", None),
			True,
		),
		True,
	)
	merge_reports_2panel_after_write_svg = _as_bool(
		_coalesce(
			merge_reports_2panel_after_cfg.get("write_svg", None),
			False,
		),
		False,
	)
	merge_reports_2panel_after_point_color = _as_optional_str(
		_coalesce(
			merge_reports_2panel_after_cfg.get("point_color", None),
			merge_reports_2panel_cfg.get("after_point_color", None),
			"#7a7a7a",
		)
	) or "#7a7a7a"
	merge_reports_2panel_highlight_merges_enabled = _as_bool(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("enabled", None),
			False,
		),
		False,
	)
	merge_reports_2panel_highlight_merges_linked = _as_bool(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("linked_highlight", None),
			True,
		),
		True,
	)
	merge_reports_2panel_highlight_plot_after_other_units = _as_bool(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("plot_after_other_units", None),
			False,
		),
		False,
	)
	merge_reports_2panel_highlight_label_affected_units = _as_bool(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("label_affected_units", None),
			False,
		),
		False,
	)
	merge_reports_2panel_highlight_show_legend = _as_bool(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("show_legend", None),
			False,
		),
		False,
	)
	merge_reports_2panel_highlight_legend_position = _as_optional_str(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("legend_position", None),
			"center left",
		)
	) or "center left"
	merge_reports_2panel_highlight_legend_x = _as_optional_float(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("legend_x", None),
			-0.2,
		)
	)
	if merge_reports_2panel_highlight_legend_x is None:
		merge_reports_2panel_highlight_legend_x = -0.2
	merge_reports_2panel_highlight_legend_y = _as_optional_float(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("legend_y", None),
			0.5,
		)
	)
	if merge_reports_2panel_highlight_legend_y is None:
		merge_reports_2panel_highlight_legend_y = 0.5
	merge_reports_2panel_highlight_sort_pre_legend_by_groups = _as_bool(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("sort_pre_legend_by_groups", None),
			False,
		),
		False,
	)
	merge_reports_2panel_highlight_debug_json_enabled = _as_bool(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("debug_json", None),
			merge_reports_2panel_highlight_cfg.get("debug_json_enabled", None),
			True,
		),
		True,
	)
	merge_reports_2panel_highlight_debug_json_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("debug_json_relpath", None),
			"unit_locations_highlight_linkage.json",
		)
	) or "unit_locations_highlight_linkage.json"
	merge_reports_2panel_highlight_before_color = _as_optional_str(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("before_color", None),
			"#ff7f0e",
		)
	) or "#ff7f0e"
	merge_reports_2panel_highlight_after_color = _as_optional_str(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("after_color", None),
			"#2ca02c",
		)
	) or "#2ca02c"
	merge_reports_2panel_highlight_palette = _as_optional_str(
		_coalesce(
			merge_reports_2panel_highlight_cfg.get("palette", None),
			"tab20",
		)
	) or "tab20"
	merge_reports_2panel_inherit_probe_dimensions = _as_bool(
		_coalesce(
			merge_reports_2panel_cfg.get("inherit_probe_dimensions", None),
			False,
		),
		False,
	)
	merge_reports_2panel_zoom_to_affected_units = _as_bool(
		_coalesce(
			merge_reports_2panel_cfg.get("zoom_to_affected_units", None),
			False,
		),
		False,
	)
	merge_reports_2panel_probe_dim_x_um = _as_optional_float(
		_coalesce(
			merge_reports_2panel_cfg.get("probe_dim_x_um", None),
			merge_reports_2panel_cfg.get("active_area_um_x", None),
		)
	)
	merge_reports_2panel_probe_dim_y_um = _as_optional_float(
		_coalesce(
			merge_reports_2panel_cfg.get("probe_dim_y_um", None),
			merge_reports_2panel_cfg.get("active_area_um_y", None),
		)
	)
	merge_reports_template_heatmaps_enabled = _as_bool(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("enabled", None),
			False,
		),
		False,
	)
	merge_reports_template_heatmaps_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("relpath", None),
			"template_heatmaps_per_merge",
		)
	) or "template_heatmaps_per_merge"
	merge_reports_template_heatmaps_assets_reldir = _normalize_optional_relpath(
		_coalesce(
			merge_reports_template_heatmaps_assets_cfg.get("relpath", None),
			merge_reports_template_heatmaps_cfg.get("assets_reldir", None),
			"assets",
		)
	) or "assets"
	merge_reports_template_heatmaps_write_png = _as_bool(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("write_png", None),
			True,
		),
		True,
	)
	merge_reports_template_heatmaps_write_svg = _as_bool(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("write_svg", None),
			False,
		),
		False,
	)
	merge_reports_template_heatmaps_write_assets_png = _as_bool(
		_coalesce(
			merge_reports_template_heatmaps_assets_cfg.get("write_png", None),
			True,
		),
		True,
	)
	merge_reports_template_heatmaps_write_assets_svg = _as_bool(
		_coalesce(
			merge_reports_template_heatmaps_assets_cfg.get("write_svg", None),
			False,
		),
		False,
	)
	merge_reports_template_heatmaps_panel_width_in = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("panel_width_in", None),
			11.0,
		)
	)
	if merge_reports_template_heatmaps_panel_width_in is None or float(merge_reports_template_heatmaps_panel_width_in) <= 0.0:
		merge_reports_template_heatmaps_panel_width_in = 11.0
	merge_reports_template_heatmaps_panel_height_in = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("panel_height_in", None),
			6.0,
		)
	)
	if merge_reports_template_heatmaps_panel_height_in is None or float(merge_reports_template_heatmaps_panel_height_in) <= 0.0:
		merge_reports_template_heatmaps_panel_height_in = 6.0
	merge_reports_template_heatmaps_marker_size = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("marker_size", None),
			10.0,
		)
	)
	if merge_reports_template_heatmaps_marker_size is None or float(merge_reports_template_heatmaps_marker_size) <= 0.0:
		merge_reports_template_heatmaps_marker_size = 10.0
	merge_reports_template_heatmaps_cmap = _as_optional_str(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("cmap", None),
			"viridis",
		)
	) or "viridis"
	merge_reports_template_heatmaps_show_colorbar = _as_bool(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("show_colorbar", None),
			True,
		),
		True,
	)
	merge_reports_template_heatmaps_relative_color_bar_height = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("relative_color_bar_height", None),
			1.0,
		)
	)
	if (
		merge_reports_template_heatmaps_relative_color_bar_height is None
		or (not math.isfinite(float(merge_reports_template_heatmaps_relative_color_bar_height)))
		or float(merge_reports_template_heatmaps_relative_color_bar_height) <= 0.0
	):
		merge_reports_template_heatmaps_relative_color_bar_height = 1.0
	merge_reports_template_heatmaps_relative_color_bar_height = max(
		0.05,
		min(1.0, float(merge_reports_template_heatmaps_relative_color_bar_height)),
	)
	merge_reports_template_heatmaps_color_scale_raw = _as_optional_str(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("color_scale", None),
			"linear",
		)
	)
	merge_reports_template_heatmaps_color_scale = str(
		merge_reports_template_heatmaps_color_scale_raw or "linear"
	).strip().lower()
	if merge_reports_template_heatmaps_color_scale in {"log10", "logarithmic"}:
		merge_reports_template_heatmaps_color_scale = "log"
	if merge_reports_template_heatmaps_color_scale not in {"linear", "log"}:
		merge_reports_template_heatmaps_color_scale = "linear"
	merge_reports_template_heatmaps_log_epsilon = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("log_epsilon", None),
			1e-3,
		)
	)
	if (
		merge_reports_template_heatmaps_log_epsilon is None
		or float(merge_reports_template_heatmaps_log_epsilon) <= 0.0
	):
		merge_reports_template_heatmaps_log_epsilon = 1e-3
	merge_reports_template_heatmaps_magnitude_mode = _normalize_merge_template_heatmap_magnitude_mode(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("magnitude_mode", None),
			"ptp",
		)
	)
	merge_reports_template_heatmaps_max_merges = _as_optional_int(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("max_merges", None),
			None,
		)
	)
	merge_reports_template_heatmaps_debug_json_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("debug_json_relpath", None),
			"template_heatmaps_per_merge_report.json",
		)
	) or "template_heatmaps_per_merge_report.json"
	merge_reports_template_heatmaps_inherit_probe_dimensions = _as_bool(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("inherit_probe_dimensions", None),
			False,
		),
		False,
	)
	merge_reports_template_heatmaps_probe_dim_x_um = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("probe_dim_x_um", None),
			merge_reports_template_heatmaps_cfg.get("active_area_um_x", None),
		)
	)
	merge_reports_template_heatmaps_probe_dim_y_um = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("probe_dim_y_um", None),
			merge_reports_template_heatmaps_cfg.get("active_area_um_y", None),
		)
	)
	merge_reports_template_heatmaps_probe_pitch_um = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_cfg.get("probe_pitch_um", None),
			merge_reports_template_heatmaps_cfg.get("pitch_um", None),
			None,
		)
	)
	merge_reports_template_heatmaps_probe_electrode_size_um_x = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_electrode_size_cfg.get("x", None),
			merge_reports_template_heatmaps_cfg.get("electrode_size_um_x", None),
			None,
		)
	)
	merge_reports_template_heatmaps_probe_electrode_size_um_y = _as_optional_float(
		_coalesce(
			merge_reports_template_heatmaps_electrode_size_cfg.get("y", None),
			merge_reports_template_heatmaps_cfg.get("electrode_size_um_y", None),
			None,
		)
	)

	merge_metadata_enabled = _as_bool(
		_coalesce(
			merge_metadata_cfg.get("enabled", None),
			merge_units_phase_cfg.get("merge_metadata_enabled", None),
			False,
		),
		False,
	)
	merge_metadata_write_json = _as_bool(
		_coalesce(
			merge_metadata_cfg.get("write_json", None),
			merge_units_phase_cfg.get("merge_metadata_write_json", None),
			True,
		),
		True,
	)
	merge_metadata_json_relpath = _normalize_optional_relpath(
		_coalesce(
			merge_metadata_cfg.get("json_relpath", None),
			merge_metadata_cfg.get("summary_json_relpath", None),
			merge_units_phase_cfg.get("merge_metadata_json_relpath", None),
			"merge_metadata_summary.json",
		)
	) or "merge_metadata_summary.json"
	merge_metadata_include_unit_locations = _as_bool(
		_coalesce(
			merge_metadata_cfg.get("include_unit_locations", None),
			merge_units_phase_cfg.get("merge_metadata_include_unit_locations", None),
			True,
		),
		True,
	)
	merge_metadata_log_summary_details = _as_bool(
		_coalesce(
			merge_metadata_cfg.get("log_summary_details", None),
			merge_units_phase_cfg.get("merge_metadata_log_summary_details", None),
			False,
		),
		False,
	)
	pre_merge_metadata_enabled = _as_bool(
		_coalesce(
			pre_merge_metadata_cfg.get("enabled", None),
			False,
		),
		False,
	)
	pre_merge_metadata_write_json = _as_bool(
		_coalesce(
			pre_merge_metadata_cfg.get("write_json", None),
			merge_metadata_write_json,
			True,
		),
		True,
	)
	pre_merge_metadata_json_relpath = _normalize_optional_relpath(
		_coalesce(
			pre_merge_metadata_cfg.get("json_relpath", None),
			"pre_merge_metadata_summary.json",
		)
	) or "pre_merge_metadata_summary.json"
	pre_merge_metadata_include_unit_locations = _as_bool(
		_coalesce(
			pre_merge_metadata_cfg.get("include_unit_locations", None),
			merge_metadata_include_unit_locations,
			True,
		),
		True,
	)
	pre_merge_metadata_log_summary_details = _as_bool(
		_coalesce(
			pre_merge_metadata_cfg.get("log_summary_details", None),
			merge_metadata_log_summary_details,
			False,
		),
		False,
	)
	post_merge_metadata_enabled = _as_bool(
		_coalesce(
			post_merge_metadata_cfg.get("enabled", None),
			False,
		),
		False,
	)
	post_merge_metadata_write_json = _as_bool(
		_coalesce(
			post_merge_metadata_cfg.get("write_json", None),
			merge_metadata_write_json,
			True,
		),
		True,
	)
	post_merge_metadata_json_relpath = _normalize_optional_relpath(
		_coalesce(
			post_merge_metadata_cfg.get("json_relpath", None),
			"post_merge_metadata_summary.json",
		)
	) or "post_merge_metadata_summary.json"
	post_merge_metadata_include_unit_locations = _as_bool(
		_coalesce(
			post_merge_metadata_cfg.get("include_unit_locations", None),
			merge_metadata_include_unit_locations,
			True,
		),
		True,
	)
	post_merge_metadata_log_summary_details = _as_bool(
		_coalesce(
			post_merge_metadata_cfg.get("log_summary_details", None),
			merge_metadata_log_summary_details,
			False,
		),
		False,
	)

	merge_sequence = tuple(
		_as_list_of_strings(
			_coalesce(
				merge_units_phase_cfg.get("sequence", None),
				execution_cfg.get("merge_sequence", None),
				stage_cfg.get("merge_sequence", None),
			)
		)
	)
	if not merge_sequence:
		merge_sequence = ("SLAy", "auto_merge", "unitmatch")

	am_kwargs = (
		_as_optional_dict(
			_coalesce(
				merge_units_phase_cfg.get("am_kwargs", None),
				execution_cfg.get("am_kwargs", None),
				stage_cfg.get("am_kwargs", None),
			)
		)
		or {}
	)
	am_kwargs.setdefault("enabled", bool(auto_merge_enabled))
	if auto_merge_template_diff_thresh is not None:
		am_kwargs.setdefault("template_diff_thresh", str(auto_merge_template_diff_thresh))
	am_kwargs.setdefault("relpath", str(auto_merge_relpath))
	am_kwargs.setdefault("delete_outputs_on_force_restart", bool(auto_merge_delete_outputs_on_force_restart))
	am_kwargs.setdefault("candidate_pairs_reldir", str(auto_merge_candidate_pairs_reldir))
	am_kwargs.setdefault("merged_units_reldir", str(auto_merge_merged_units_reldir))
	am_kwargs.setdefault("auto_accept_merges", bool(auto_merge_auto_accept_merges))
	am_kwargs.setdefault("template_diff_thresh_values", list(auto_merge_template_diff_thresholds))

	option_kwargs = (
		_as_optional_dict(
			_coalesce(
				merge_units_phase_cfg.get("option_kwargs", None),
				execution_cfg.get("option_kwargs", None),
				stage_cfg.get("option_kwargs", None),
			)
		)
		or {}
	)
	option_kwargs.setdefault("force_rerun_analyzer", bool(force_rerun_analyzer))

	bombcell_label_enabled = _as_bool(
		_coalesce(
			bombcell_phase_cfg.get("enabled", None),
			bool(bombcell_phase_cfg_raw is not None),
		),
		False,
	)
	bombcell_label_relpath = _normalize_optional_relpath(
		_coalesce(
			bombcell_phase_cfg.get("relpath", None),
			bombcell_phase_cfg.get("output_relpath", None),
			"bombcell_label_outputs",
		)
	) or "bombcell_label_outputs"
	bombcell_label_delete_outputs_on_force_restart = _as_bool(
		_coalesce(
			bombcell_phase_cfg.get("delete_outputs_on_force_restart", None),
			bombcell_phase_cfg.get("delete_on_force_restart", None),
			True,
		),
		True,
	)
	bombcell_label_thresholds = _as_optional_dict(
		_coalesce(
			bombcell_params_cfg.get("thresholds", None),
			bombcell_params_cfg.get("threshold_dict", None),
			bombcell_phase_cfg.get("thresholds", None),
			bombcell_phase_cfg.get("threshold_dict", None),
		)
	)
	bombcell_label_thresholds_path = _as_optional_str(
		_coalesce(
			bombcell_params_cfg.get("thresholds_path", None),
			bombcell_params_cfg.get("thresholds_json", None),
			bombcell_params_cfg.get("thresholds_json_path", None),
			bombcell_phase_cfg.get("thresholds_path", None),
			bombcell_phase_cfg.get("thresholds_json", None),
			bombcell_phase_cfg.get("thresholds_json_path", None),
		)
	)
	bombcell_label_label_non_somatic = _as_bool(
		_coalesce(
			bombcell_params_cfg.get("label_non_somatic", None),
			bombcell_phase_cfg.get("label_non_somatic", None),
			True,
		),
		True,
	)
	bombcell_label_split_non_somatic_good_mua = _as_bool(
		_coalesce(
			bombcell_params_cfg.get("split_non_somatic_good_mua", None),
			bombcell_phase_cfg.get("split_non_somatic_good_mua", None),
			True,
		),
		True,
	)
	bombcell_label_apply_to_sorter_output = _as_bool(
		_coalesce(
			bombcell_phase_cfg.get("apply_to_sorter_output", None),
			bombcell_phase_cfg.get("apply_labels_to_sorter_output", None),
			True,
		),
		True,
	)
	bombcell_label_write_cluster_group = _as_bool(
		_coalesce(
			bombcell_phase_cfg.get("write_cluster_group", None),
			True,
		),
		True,
	)
	bombcell_label_fail_on_error = _as_bool(
		_coalesce(
			bombcell_phase_cfg.get("fail_on_error", None),
			False,
		),
		False,
	)
	bombcell_label_reports_enabled = _as_bool(
		_coalesce(
			bombcell_reports_cfg.get("enabled", None),
			True,
		),
		True,
	)
	bombcell_label_reports_summary_json_enabled = _as_bool(
		_coalesce(
			bombcell_reports_summary_json_cfg.get("enabled", None),
			bombcell_label_reports_enabled,
			True,
		),
		True,
	)
	bombcell_label_reports_summary_json_relpath = _normalize_optional_relpath(
		_coalesce(
			bombcell_reports_summary_json_cfg.get("relpath", None),
			"bombcell_label_summary.json",
		)
	) or "bombcell_label_summary.json"

	slay_enabled = _as_bool(_coalesce(slay_cfg.get("enabled", None), False), False)
	slay_relpath = _normalize_optional_relpath(
		_coalesce(
			slay_cfg.get("relpath", None),
			slay_cfg.get("output_relpath", None),
			"SLAy_outputs",
		)
	) or "SLAy_outputs"
	slay_package_root = _as_optional_str(
		_coalesce(
			slay_cfg.get("package_root", None),
			slay_cfg.get("package_path", None),
			slay_cfg.get("repo_root", None),
		)
	)
	slay_sorter_output_relpath = _normalize_optional_relpath(
		_coalesce(
			slay_cfg.get("sorter_output_relpath", None),
			slay_cfg.get("ks_dir_relpath", None),
			slay_cfg.get("ks_folder_relpath", None),
		)
	)
	slay_output_json_relpath = _normalize_optional_relpath(
		_coalesce(
			slay_cfg.get("output_json_relpath", None),
			slay_cfg.get("run_output_relpath", None),
			"run-output.json",
		)
	) or "run-output.json"
	slay_candidate_pairs_relpath = _normalize_optional_relpath(
		_coalesce(
			slay_cfg.get("candidate_pairs_relpath", None),
			slay_cfg.get("candidates_relpath", None),
			"recommended_merge_candidates.tsv",
		)
	) or "recommended_merge_candidates.tsv"
	slay_merge_groups_relpath = _normalize_optional_relpath(
		_coalesce(
			slay_cfg.get("merge_groups_relpath", None),
			slay_cfg.get("groups_relpath", None),
			"recommended_merge_groups.json",
		)
	) or "recommended_merge_groups.json"
	slay_allow_numpy_fallback = _as_bool(_coalesce(slay_cfg.get("allow_numpy_fallback", None), True), True)
	slay_plot_merges = _as_bool(_coalesce(slay_cfg.get("plot_merges", None), False), False)
	slay_auto_accept_merges = _as_bool(_coalesce(slay_cfg.get("auto_accept_merges", None), False), False)
	slay_copy_automerge_artifacts = _as_bool(
		_coalesce(
			slay_cfg.get("copy_automerge_artifacts", None),
			True,
		),
		True,
	)
	slay_delete_outputs_on_force_restart = _as_bool(
		_coalesce(
			slay_cfg.get("delete_outputs_on_force_restart", None),
			slay_cfg.get("delete_on_force_restart", None),
			True,
		),
		True,
	)
	slay_recompute_analyzer = _as_bool(
		_coalesce(
			slay_cfg.get("recompute_analyzer", None),
			slay_cfg.get("rerun_analyzer", None),
			False,
		),
		False,
	)
	slay_model_cache_enabled_raw = _coalesce(
		slay_model_cache_cfg.get("enabled", None),
		slay_cfg.get("model_cache_enabled", None),
	)
	slay_model_cache_enabled_default = (
		_as_bool(slay_model_cache_enabled_raw, True)
		if slay_model_cache_enabled_raw is not None
		else True
	)
	slay_model_cache_relpath = _normalize_optional_relpath(
		_coalesce(
			slay_model_cache_cfg.get("relpath", None),
			slay_model_cache_cfg.get("model_path", None),
			slay_model_cache_cfg.get("path", None),
			slay_cfg.get("model_cache_relpath", None),
			slay_cfg.get("model_relpath", None),
			"cache/slay_model/ae.pt",
		)
	)
	slay_model_cache_use_cached_model = _as_bool(
		_coalesce(
			slay_model_cache_cfg.get("use_cached_model", None),
			slay_model_cache_cfg.get("use_cache", None),
			slay_model_cache_cfg.get("read_from_cache", None),
			slay_cfg.get("model_cache_use_cached_model", None),
			slay_cfg.get("use_cached_model", None),
			slay_model_cache_enabled_raw,
			True,
		),
		slay_model_cache_enabled_default,
	)
	slay_model_cache_write_model = _as_bool(
		_coalesce(
			slay_model_cache_cfg.get("write_model_cache", None),
			slay_model_cache_cfg.get("write_cache", None),
			slay_model_cache_cfg.get("save_model", None),
			slay_cfg.get("model_cache_write_model", None),
			slay_cfg.get("write_model_cache", None),
			slay_model_cache_enabled_raw,
			True,
		),
		slay_model_cache_enabled_default,
	)
	slay_force_restart_retrain_model = _as_bool(
		_coalesce(
			slay_cfg.get("force_restart_retrain_model", None),
			slay_cfg.get("retrain_model_on_force_restart", None),
			False,
		),
		False,
	)
	slay_params = _as_optional_dict(slay_cfg.get("params", None))
	legacy_merge_analyzer_density_mode_raw = _coalesce(
		merge_analyzer_template_extraction_cfg.get("density_mode", None),
		merge_analyzer_cfg.get("density_mode", None),
		merge_units_phase_cfg.get("analyzer_density_mode", None),
		merge_units_phase_cfg.get("density_mode", None),
		None,
	)
	legacy_merge_analyzer_density_mode = None
	if legacy_merge_analyzer_density_mode_raw is not None:
		_warn_legacy_merge_analyzer_density_mode_alias()
		legacy_merge_analyzer_density_mode = _normalize_merge_analyzer_density_mode(
			legacy_merge_analyzer_density_mode_raw
		)
	merge_analyzer_compute_sparsity_raw = _coalesce(
		merge_analyzer_sparsity_cfg.get("compute_sparsity", None),
		merge_analyzer_cfg.get("compute_sparsity", None),
		merge_units_phase_cfg.get("analyzer_compute_sparsity", None),
		merge_units_phase_cfg.get("compute_sparsity", None),
		None,
	)
	if merge_analyzer_compute_sparsity_raw is None:
		merge_analyzer_compute_sparsity = bool(legacy_merge_analyzer_density_mode != "dense")
	else:
		merge_analyzer_compute_sparsity = _as_bool(merge_analyzer_compute_sparsity_raw, True)
	merge_template_random_spikes_method_explicit_raw = _coalesce(
			merge_analyzer_template_extraction_cfg.get("random_spikes_method", None),
			merge_analyzer_template_extraction_cfg.get("template_random_spikes_method", None),
			merge_analyzer_template_extraction_cfg.get("method", None),
			merge_analyzer_cfg.get("template_random_spikes_method", None),
			merge_analyzer_cfg.get("random_spikes_method", None),
			merge_units_phase_cfg.get("template_random_spikes_method", None),
			merge_units_phase_cfg.get("random_spikes_method", None),
			None,
		)
	merge_template_random_spikes_method_raw = _coalesce(
			merge_template_random_spikes_method_explicit_raw,
			("all" if legacy_merge_analyzer_density_mode == "dense" else None),
			"default",
		)
	merge_template_random_spikes_method = _normalize_merge_template_random_spikes_method(
		merge_template_random_spikes_method_raw
	)
	merge_template_random_spikes_percentage_raw = _coalesce(
		merge_analyzer_template_extraction_cfg.get("random_spikes_percentage", None),
		merge_analyzer_template_extraction_cfg.get("template_random_spikes_percentage", None),
		merge_analyzer_cfg.get("template_random_spikes_percentage", None),
		merge_analyzer_cfg.get("random_spikes_percentage", None),
		merge_units_phase_cfg.get("template_random_spikes_percentage", None),
		merge_units_phase_cfg.get("random_spikes_percentage", None),
		None,
	)
	legacy_merge_template_random_spikes_percentage_raw = _coalesce(
		merge_analyzer_template_extraction_cfg.get("min_perc_spikes_per_unit", None),
		merge_analyzer_cfg.get("min_perc_spikes_per_unit", None),
		merge_units_phase_cfg.get("min_perc_spikes_per_unit", None),
		None,
	)
	merge_template_random_spikes_percentage = None
	if merge_template_random_spikes_percentage_raw is not None:
		merge_template_random_spikes_percentage = _parse_merge_template_random_spikes_percentage(
			merge_template_random_spikes_percentage_raw,
			field_name="random_spikes_percentage",
		)
	elif legacy_merge_template_random_spikes_percentage_raw is not None:
		_warn_legacy_merge_template_random_spikes_percentage_alias()
		merge_template_random_spikes_percentage = _parse_merge_template_random_spikes_percentage(
			legacy_merge_template_random_spikes_percentage_raw,
			field_name="min_perc_spikes_per_unit",
		)
	if merge_template_random_spikes_method == "all" and merge_template_random_spikes_percentage is not None:
		LOGGER.warning(
			"Merge analyzer random_spikes_percentage is ignored when random_spikes_method=all."
		)
		merge_template_random_spikes_percentage = None
	elif merge_template_random_spikes_method == "default" and merge_template_random_spikes_percentage is not None:
		merge_template_random_spikes_method = "percentage"
	elif merge_template_random_spikes_method == "percentage" and merge_template_random_spikes_percentage is None:
		raise ValueError(
			"Merge analyzer random_spikes_method=percentage requires random_spikes_percentage."
		)
	merge_template_random_spikes_max_spikes_per_unit_raw = _as_optional_int(
		_coalesce(
			merge_analyzer_template_extraction_cfg.get("random_spikes_max_spikes_per_unit", None),
			merge_analyzer_template_extraction_cfg.get("template_random_spikes_max_spikes_per_unit", None),
			merge_analyzer_template_extraction_cfg.get("max_spikes_per_unit", None),
			merge_analyzer_cfg.get("template_random_spikes_max_spikes_per_unit", None),
			merge_analyzer_cfg.get("random_spikes_max_spikes_per_unit", None),
			merge_analyzer_cfg.get("max_spikes_per_unit", None),
			merge_units_phase_cfg.get("template_random_spikes_max_spikes_per_unit", None),
			merge_units_phase_cfg.get("random_spikes_max_spikes_per_unit", None),
			merge_units_phase_cfg.get("max_spikes_per_unit", None),
			None,
		)
	)
	merge_template_random_spikes_max_spikes_per_unit = (
		int(merge_template_random_spikes_max_spikes_per_unit_raw)
		if (
			merge_template_random_spikes_max_spikes_per_unit_raw is not None
			and int(merge_template_random_spikes_max_spikes_per_unit_raw) > 0
		)
		else None
	)
	if (
		merge_template_random_spikes_method != "percentage"
		and merge_template_random_spikes_max_spikes_per_unit is None
	):
		merge_template_random_spikes_max_spikes_per_unit = 500
	merge_template_random_spikes_min_spikes_per_unit = _as_optional_positive_int(
		_coalesce(
			merge_analyzer_template_extraction_cfg.get("min_spikes_per_unit", None),
			merge_analyzer_template_extraction_cfg.get("template_random_spikes_min_spikes_per_unit", None),
			merge_analyzer_template_extraction_cfg.get("random_spikes_min_spikes_per_unit", None),
			merge_analyzer_cfg.get("template_random_spikes_min_spikes_per_unit", None),
			merge_analyzer_cfg.get("random_spikes_min_spikes_per_unit", None),
			merge_analyzer_cfg.get("min_spikes_per_unit", None),
			merge_units_phase_cfg.get("template_random_spikes_min_spikes_per_unit", None),
			merge_units_phase_cfg.get("random_spikes_min_spikes_per_unit", None),
			merge_units_phase_cfg.get("min_spikes_per_unit", None),
			None,
		)
	)
	merge_template_random_spikes_log_before_after_spike_counts = _as_bool(
		_coalesce(
			merge_analyzer_template_extraction_cfg.get("log_before_after_spike_counts", None),
			merge_analyzer_template_extraction_cfg.get(
				"template_random_spikes_log_before_after_spike_counts",
				None,
			),
			merge_analyzer_template_extraction_cfg.get(
				"random_spikes_log_before_after_spike_counts",
				None,
			),
			merge_analyzer_cfg.get("template_random_spikes_log_before_after_spike_counts", None),
			merge_analyzer_cfg.get("random_spikes_log_before_after_spike_counts", None),
			merge_analyzer_cfg.get("log_before_after_spike_counts", None),
			merge_units_phase_cfg.get("template_random_spikes_log_before_after_spike_counts", None),
			merge_units_phase_cfg.get("random_spikes_log_before_after_spike_counts", None),
			merge_units_phase_cfg.get("log_before_after_spike_counts", None),
			False,
		),
		False,
	)
	merge_template_random_spikes_margin_size = _as_optional_int(
		_coalesce(
			merge_analyzer_template_extraction_cfg.get("random_spikes_margin_size", None),
			merge_analyzer_template_extraction_cfg.get("template_random_spikes_margin_size", None),
			merge_analyzer_template_extraction_cfg.get("margin_size", None),
			merge_analyzer_cfg.get("template_random_spikes_margin_size", None),
			merge_analyzer_cfg.get("random_spikes_margin_size", None),
			merge_analyzer_cfg.get("margin_size", None),
			merge_units_phase_cfg.get("template_random_spikes_margin_size", None),
			merge_units_phase_cfg.get("random_spikes_margin_size", None),
			merge_units_phase_cfg.get("margin_size", None),
			None,
		)
	)
	if (
		merge_template_random_spikes_margin_size is not None
		and int(merge_template_random_spikes_margin_size) < 0
	):
		merge_template_random_spikes_margin_size = None
	merge_template_random_spikes_seed = _as_optional_int(
		_coalesce(
			merge_analyzer_template_extraction_cfg.get("random_spikes_seed", None),
			merge_analyzer_template_extraction_cfg.get("template_random_spikes_seed", None),
			merge_analyzer_template_extraction_cfg.get("seed", None),
			merge_analyzer_cfg.get("template_random_spikes_seed", None),
			merge_analyzer_cfg.get("random_spikes_seed", None),
			merge_analyzer_cfg.get("seed", None),
			merge_units_phase_cfg.get("template_random_spikes_seed", None),
			merge_units_phase_cfg.get("random_spikes_seed", None),
			merge_units_phase_cfg.get("seed", None),
			None,
		)
	)
	merge_analyzer_regenerate_on_replot = _as_bool(
		_coalesce(
			merge_analyzer_cfg.get("regenerate_on_replot", None),
			merge_analyzer_cfg.get("regenereate_on_replot", None),
			merge_units_phase_cfg.get("analyzer_regenerate_on_replot", None),
			merge_units_phase_cfg.get("regenerate_on_replot", None),
			True,
		),
		True,
	)
	merge_analyzer_check_if_regen_is_needed = _as_bool(
		_coalesce(
			merge_analyzer_cfg.get("check_if_regen_is_needed", None),
			merge_units_phase_cfg.get("analyzer_check_if_regen_is_needed", None),
			merge_units_phase_cfg.get("check_if_regen_is_needed", None),
			True,
		),
		True,
	)
	merge_analyzer_n_jobs = _as_optional_int(
		_coalesce(
			merge_analyzer_cfg.get("n_jobs", None),
			merge_units_phase_cfg.get("analyzer_n_jobs", None),
			None,
		)
	)
	merge_analyzer_chunk_duration = _as_optional_str(
		_coalesce(
			merge_analyzer_cfg.get("chunk_duration", None),
			merge_units_phase_cfg.get("analyzer_chunk_duration", None),
			None,
		)
	)
	merge_analyzer_sparsity_method = _normalize_merge_analyzer_sparsity_method(
		_coalesce(
			merge_analyzer_sparsity_cfg.get("sparsity_method", None),
			merge_analyzer_sparsity_cfg.get("method", None),
			merge_analyzer_cfg.get("sparsity_method", None),
			merge_analyzer_cfg.get("method", None),
			merge_units_phase_cfg.get("analyzer_sparsity_method", None),
			merge_units_phase_cfg.get("sparsity_method", None),
			"radius",
		)
	)
	merge_analyzer_sparsity_radius_um = _as_optional_float(
		_coalesce(
			merge_analyzer_sparsity_cfg.get("radius_um", None),
			merge_analyzer_sparsity_cfg.get("sparsity_radius_um", None),
			merge_analyzer_cfg.get("sparsity_radius_um", None),
			merge_analyzer_cfg.get("radius_um", None),
			merge_units_phase_cfg.get("analyzer_sparsity_radius_um", None),
			merge_units_phase_cfg.get("sparsity_radius_um", None),
			100.0,
		)
	)
	if (
		merge_analyzer_sparsity_radius_um is None
		or float(merge_analyzer_sparsity_radius_um) <= 0.0
	):
		merge_analyzer_sparsity_radius_um = 100.0
	merge_analyzer_sparsity_num_channels = _as_optional_int(
		_coalesce(
			merge_analyzer_sparsity_cfg.get("num_channels", None),
			merge_analyzer_sparsity_cfg.get("sparsity_num_channels", None),
			merge_analyzer_cfg.get("sparsity_num_channels", None),
			merge_analyzer_cfg.get("num_channels", None),
			merge_units_phase_cfg.get("analyzer_sparsity_num_channels", None),
			merge_units_phase_cfg.get("sparsity_num_channels", None),
			5,
		)
	)
	if (
		merge_analyzer_sparsity_num_channels is None
		or int(merge_analyzer_sparsity_num_channels) <= 0
	):
		merge_analyzer_sparsity_num_channels = 5
	merge_analyzer_sparsity_threshold = _as_optional_float(
		_coalesce(
			merge_analyzer_sparsity_cfg.get("threshold", None),
			merge_analyzer_sparsity_cfg.get("sparsity_threshold", None),
			merge_analyzer_cfg.get("sparsity_threshold", None),
			merge_analyzer_cfg.get("threshold", None),
			merge_units_phase_cfg.get("analyzer_sparsity_threshold", None),
			merge_units_phase_cfg.get("sparsity_threshold", None),
			5.0,
		)
	)
	if (
		merge_analyzer_sparsity_threshold is None
		or float(merge_analyzer_sparsity_threshold) <= 0.0
	):
		merge_analyzer_sparsity_threshold = 5.0
	merge_analyzer_sparsity_peak_sign = _normalize_merge_analyzer_peak_sign(
		_coalesce(
			merge_analyzer_sparsity_cfg.get("peak_sign", None),
			merge_analyzer_sparsity_cfg.get("sparsity_peak_sign", None),
			merge_analyzer_cfg.get("sparsity_peak_sign", None),
			merge_analyzer_cfg.get("peak_sign", None),
			merge_units_phase_cfg.get("analyzer_sparsity_peak_sign", None),
			merge_units_phase_cfg.get("sparsity_peak_sign", None),
			merge_units_phase_cfg.get("peak_sign", None),
			"neg",
		)
	)
	merge_analyzer_sparsity_num_spikes_for_sparsity = _as_optional_int(
		_coalesce(
			merge_analyzer_sparsity_cfg.get("num_spikes_for_sparsity", None),
			merge_analyzer_sparsity_cfg.get("sparsity_num_spikes_for_sparsity", None),
			merge_analyzer_cfg.get("sparsity_num_spikes_for_sparsity", None),
			merge_analyzer_cfg.get("num_spikes_for_sparsity", None),
			merge_units_phase_cfg.get("analyzer_sparsity_num_spikes_for_sparsity", None),
			merge_units_phase_cfg.get("sparsity_num_spikes_for_sparsity", None),
			merge_units_phase_cfg.get("num_spikes_for_sparsity", None),
			100,
		)
	)
	if (
		merge_analyzer_sparsity_num_spikes_for_sparsity is None
		or int(merge_analyzer_sparsity_num_spikes_for_sparsity) <= 0
	):
		merge_analyzer_sparsity_num_spikes_for_sparsity = 100
	merge_analyzer_sparsity_by_property = _as_optional_str(
		_coalesce(
			merge_analyzer_sparsity_cfg.get("by_property", None),
			merge_analyzer_sparsity_cfg.get("sparsity_by_property", None),
			merge_analyzer_cfg.get("sparsity_by_property", None),
			merge_analyzer_cfg.get("by_property", None),
			merge_units_phase_cfg.get("analyzer_sparsity_by_property", None),
			merge_units_phase_cfg.get("sparsity_by_property", None),
			None,
		)
	)
	merge_analyzer_waveforms_ms_before = _as_optional_float(
		_coalesce(
			merge_analyzer_waveforms_cfg.get("ms_before", None),
			merge_analyzer_waveforms_cfg.get("waveforms_ms_before", None),
			merge_analyzer_waveforms_cfg.get("template_ms_before", None),
			merge_analyzer_cfg.get("waveforms_ms_before", None),
			merge_analyzer_cfg.get("template_ms_before", None),
			merge_units_phase_cfg.get("analyzer_waveforms_ms_before", None),
			merge_units_phase_cfg.get("waveforms_ms_before", None),
			merge_units_phase_cfg.get("template_ms_before", None),
			1.0,
		)
	)
	if (
		merge_analyzer_waveforms_ms_before is None
		or float(merge_analyzer_waveforms_ms_before) < 0.0
	):
		merge_analyzer_waveforms_ms_before = 1.0
	merge_analyzer_waveforms_ms_after = _as_optional_float(
		_coalesce(
			merge_analyzer_waveforms_cfg.get("ms_after", None),
			merge_analyzer_waveforms_cfg.get("waveforms_ms_after", None),
			merge_analyzer_waveforms_cfg.get("template_ms_after", None),
			merge_analyzer_cfg.get("waveforms_ms_after", None),
			merge_analyzer_cfg.get("template_ms_after", None),
			merge_units_phase_cfg.get("analyzer_waveforms_ms_after", None),
			merge_units_phase_cfg.get("waveforms_ms_after", None),
			merge_units_phase_cfg.get("template_ms_after", None),
			2.0,
		)
	)
	if (
		merge_analyzer_waveforms_ms_after is None
		or float(merge_analyzer_waveforms_ms_after) < 0.0
	):
		merge_analyzer_waveforms_ms_after = 2.0
	merge_analyzer_waveforms_dtype = _as_optional_str(
		_coalesce(
			merge_analyzer_waveforms_cfg.get("dtype", None),
			merge_analyzer_waveforms_cfg.get("waveforms_dtype", None),
			merge_analyzer_waveforms_cfg.get("template_waveforms_dtype", None),
			merge_analyzer_cfg.get("waveforms_dtype", None),
			merge_analyzer_cfg.get("template_waveforms_dtype", None),
			merge_units_phase_cfg.get("analyzer_waveforms_dtype", None),
			merge_units_phase_cfg.get("waveforms_dtype", None),
			merge_units_phase_cfg.get("template_waveforms_dtype", None),
			None,
		)
	)

	resolved_um_kwargs = (um_kwargs if um_kwargs else None)
	resolved_am_kwargs = (am_kwargs if am_kwargs else None)
	resolved_option_kwargs = (option_kwargs if option_kwargs else None)

	return SpikesortStageConfig(
		output_rel_root=_normalize_output_rel_root(
			_coalesce(
				execution_cfg.get("output_root", None),
				execution_cfg.get("output_rel_root", None),
				stage_cfg.get("output_root", None),
				stage_cfg.get("output_rel_root", None),
				outputs_cfg.get("output_root", None),
				outputs_cfg.get("output_rel_root", None),
				_DEFAULT_OUTPUT_REL_ROOT,
			)
		),
		preprocess_concat_recording_relpath=preprocess_concat_recording_relpath,
		merge_sequence=merge_sequence,
		logging_enabled=logging_enabled,
		logging_verbose=logging_verbose,
		logging_file_relpath=logging_file_relpath,
		debug_limit_wells=debug_limit_wells,
		sort_debug_mode_enabled=bool(sort_debug_mode_enabled),
		sort_debug_limit_datasets=sort_debug_limit_datasets,
		sort_debug_limit_wells=sort_debug_limit_wells,
		summarize_sort_debug_mode_enabled=bool(summarize_sort_debug_mode_enabled),
		summarize_sort_debug_limit_datasets=summarize_sort_debug_limit_datasets,
		summarize_sort_debug_limit_wells=summarize_sort_debug_limit_wells,
		sorter=str(
			_coalesce(
				sort_phase_cfg.get("sorter", None),
				execution_cfg.get("sorter", None),
				stage_cfg.get("sorter", None),
				"kilosort4",
			)
			or "kilosort4"
		),
		docker_image=_as_optional_str(
			_coalesce(
				sort_phase_cfg.get("docker_image", None),
				execution_cfg.get("docker_image", None),
				stage_cfg.get("docker_image", None),
			)
		),
		recording_num=str(_get_with_fallback(execution_cfg, stage_cfg, "recording_num", "rec0000") or "rec0000"),
		verbose=_as_bool(_get_with_fallback(execution_cfg, stage_cfg, "verbose", False), False),
		ks_batch_duration_s=_as_optional_float(
			_coalesce(
				sort_phase_kilosort_cfg.get("batch_duration_s", None),
				execution_kilosort_cfg.get("batch_duration_s", None),
				stage_kilosort_cfg.get("batch_duration_s", None),
				_get_with_fallback(execution_cfg, stage_cfg, "ks_batch_duration_s", None),
			)
		),
		ks_batch_size=_as_optional_int(
			_coalesce(
				sort_phase_kilosort_cfg.get("batch_size", None),
				execution_kilosort_cfg.get("batch_size", None),
				stage_kilosort_cfg.get("batch_size", None),
				_get_with_fallback(execution_cfg, stage_cfg, "ks_batch_size", None),
			)
		),
		ks_th_universal=_as_optional_float(
			_coalesce(
				_get_nested_value(sort_phase_kilosort_cfg, ("thresholds", "universal")),
				_get_nested_value(execution_kilosort_cfg, ("thresholds", "universal")),
				_get_nested_value(stage_kilosort_cfg, ("thresholds", "universal")),
				_get_with_fallback(execution_cfg, stage_cfg, "ks_th_universal", None),
			)
		),
		ks_th_learned=_as_optional_float(
			_coalesce(
				_get_nested_value(sort_phase_kilosort_cfg, ("thresholds", "learned")),
				_get_nested_value(execution_kilosort_cfg, ("thresholds", "learned")),
				_get_nested_value(stage_kilosort_cfg, ("thresholds", "learned")),
				_get_with_fallback(execution_cfg, stage_cfg, "ks_th_learned", None),
			)
		),
		ks_th_single_ch=_as_optional_float(
			_coalesce(
				_get_nested_value(sort_phase_kilosort_cfg, ("thresholds", "single_ch")),
				_get_nested_value(execution_kilosort_cfg, ("thresholds", "single_ch")),
				_get_nested_value(stage_kilosort_cfg, ("thresholds", "single_ch")),
				_get_with_fallback(execution_cfg, stage_cfg, "ks_th_single_ch", None),
			)
		),
		ks_cluster_downsampling=_as_optional_int(
			_coalesce(
				_get_nested_value(sort_phase_kilosort_cfg, ("clustering", "downsampling")),
				_get_nested_value(execution_kilosort_cfg, ("clustering", "downsampling")),
				_get_nested_value(stage_kilosort_cfg, ("clustering", "downsampling")),
				_get_with_fallback(execution_cfg, stage_cfg, "ks_cluster_downsampling", None),
			)
		),
		ks_nearest_chans=_as_optional_int(
			_coalesce(
				_get_nested_value(sort_phase_kilosort_cfg, ("channels", "nearest")),
				_get_nested_value(execution_kilosort_cfg, ("channels", "nearest")),
				_get_nested_value(stage_kilosort_cfg, ("channels", "nearest")),
				_get_with_fallback(execution_cfg, stage_cfg, "ks_nearest_chans", None),
			)
		),
		ks_max_channel_distance=_as_optional_float(
			_coalesce(
				_get_nested_value(sort_phase_kilosort_cfg, ("channels", "max_distance")),
				_get_nested_value(execution_kilosort_cfg, ("channels", "max_distance")),
				_get_nested_value(stage_kilosort_cfg, ("channels", "max_distance")),
				_get_with_fallback(execution_cfg, stage_cfg, "ks_max_channel_distance", None),
			)
		),
		n_jobs=_as_optional_int(
			_coalesce(
				resources_cfg.get("n_jobs", None),
				_get_with_fallback(execution_cfg, stage_cfg, "n_jobs", None),
			)
		),
		chunk_duration=_as_optional_str(
			_coalesce(
				resources_cfg.get("chunk_duration", None),
				_get_with_fallback(execution_cfg, stage_cfg, "chunk_duration", None),
			)
		),
		cuda_visible_devices=_as_optional_str(_get_with_fallback(execution_cfg, stage_cfg, "cuda_visible_devices", None)),
		run_analyzer=_as_bool(_get_with_fallback(execution_cfg, stage_cfg, "run_analyzer", True), True),
		run_reports=run_reports,
		sort_enabled=bool(sort_enabled),
		sort_delete_outputs_on_force_restart=bool(sort_delete_outputs_on_force_restart),
		plot_enabled=plot_enabled,
		plot_mode=plot_mode,
		plot_debug=plot_debug,
		raster_sort=raster_sort,
		fixed_y=fixed_y,
		no_curation=no_curation,
		export_to_phy=export_to_phy,
		force_rerun_analyzer=force_rerun_analyzer,
		summarize_sort_enabled=bool(summarize_sort_enabled),
		summarize_sort_emit_logs=bool(summarize_sort_emit_logs),
		summarize_sort_generate_artifacts=bool(summarize_sort_generate_artifacts),
		bombcell_label_enabled=bool(bombcell_label_enabled),
		bombcell_label_relpath=str(bombcell_label_relpath),
		bombcell_label_delete_outputs_on_force_restart=bool(bombcell_label_delete_outputs_on_force_restart),
		bombcell_label_thresholds=(dict(bombcell_label_thresholds) if isinstance(bombcell_label_thresholds, dict) else None),
		bombcell_label_thresholds_path=(str(bombcell_label_thresholds_path) if bombcell_label_thresholds_path is not None else None),
		bombcell_label_label_non_somatic=bool(bombcell_label_label_non_somatic),
		bombcell_label_split_non_somatic_good_mua=bool(bombcell_label_split_non_somatic_good_mua),
		bombcell_label_apply_to_sorter_output=bool(bombcell_label_apply_to_sorter_output),
		bombcell_label_write_cluster_group=bool(bombcell_label_write_cluster_group),
		bombcell_label_fail_on_error=bool(bombcell_label_fail_on_error),
		bombcell_label_reports_enabled=bool(bombcell_label_reports_enabled),
		bombcell_label_reports_summary_json_enabled=bool(bombcell_label_reports_summary_json_enabled),
		bombcell_label_reports_summary_json_relpath=str(bombcell_label_reports_summary_json_relpath),
		um_kwargs=resolved_um_kwargs,
		am_kwargs=resolved_am_kwargs,
		option_kwargs=resolved_option_kwargs,
		slay_enabled=bool(slay_enabled),
		slay_relpath=slay_relpath,
		slay_package_root=slay_package_root,
		slay_sorter_output_relpath=slay_sorter_output_relpath,
		slay_output_json_relpath=slay_output_json_relpath,
		slay_candidate_pairs_relpath=slay_candidate_pairs_relpath,
		slay_merge_groups_relpath=slay_merge_groups_relpath,
		slay_allow_numpy_fallback=bool(slay_allow_numpy_fallback),
		slay_plot_merges=bool(slay_plot_merges),
		slay_auto_accept_merges=bool(slay_auto_accept_merges),
		slay_copy_automerge_artifacts=bool(slay_copy_automerge_artifacts),
		slay_delete_outputs_on_force_restart=bool(slay_delete_outputs_on_force_restart),
		slay_recompute_analyzer=bool(slay_recompute_analyzer),
		slay_model_cache_relpath=(str(slay_model_cache_relpath) if slay_model_cache_relpath is not None else None),
		slay_model_cache_use_cached_model=bool(slay_model_cache_use_cached_model),
		slay_model_cache_write_model=bool(slay_model_cache_write_model),
		slay_force_restart_retrain_model=bool(slay_force_restart_retrain_model),
		slay_params=(dict(slay_params) if isinstance(slay_params, dict) else None),
		auto_merge_enabled=bool(auto_merge_enabled),
		auto_merge_relpath=str(auto_merge_relpath),
		auto_merge_delete_outputs_on_force_restart=bool(auto_merge_delete_outputs_on_force_restart),
		auto_merge_candidate_pairs_reldir=str(auto_merge_candidate_pairs_reldir),
		auto_merge_merged_units_reldir=str(auto_merge_merged_units_reldir),
		auto_merge_auto_accept_merges=bool(auto_merge_auto_accept_merges),
		auto_merge_template_diff_thresholds=tuple(auto_merge_template_diff_thresholds),
		merge_units_enabled=bool(merge_units_enabled),
		merge_rel_output_root=(str(merge_rel_output_root) if merge_rel_output_root is not None else None),
		merge_delete_outputs_on_force_restart=bool(merge_delete_outputs_on_force_restart),
		merge_force_restart=bool(merge_force_restart),
		merge_force_replot=bool(merge_force_replot),
		merge_analyzer_regenerate_on_replot=bool(merge_analyzer_regenerate_on_replot),
		merge_analyzer_check_if_regen_is_needed=bool(merge_analyzer_check_if_regen_is_needed),
		merge_analyzer_compute_sparsity=bool(merge_analyzer_compute_sparsity),
		merge_analyzer_density_mode=str(legacy_merge_analyzer_density_mode or "auto"),
		merge_template_random_spikes_method=str(merge_template_random_spikes_method),
		merge_template_random_spikes_percentage=(
			float(merge_template_random_spikes_percentage)
			if merge_template_random_spikes_percentage is not None
			else None
		),
		merge_template_random_spikes_max_spikes_per_unit=(
			int(merge_template_random_spikes_max_spikes_per_unit)
			if merge_template_random_spikes_max_spikes_per_unit is not None
			else None
		),
		merge_template_random_spikes_min_spikes_per_unit=(
			int(merge_template_random_spikes_min_spikes_per_unit)
			if merge_template_random_spikes_min_spikes_per_unit is not None
			else None
		),
		merge_template_random_spikes_log_before_after_spike_counts=bool(
			merge_template_random_spikes_log_before_after_spike_counts
		),
		merge_template_random_spikes_margin_size=(
			int(merge_template_random_spikes_margin_size)
			if merge_template_random_spikes_margin_size is not None
			else None
		),
		merge_template_random_spikes_seed=(
			int(merge_template_random_spikes_seed)
			if merge_template_random_spikes_seed is not None
			else None
		),
		merge_analyzer_n_jobs=(int(merge_analyzer_n_jobs) if merge_analyzer_n_jobs is not None else None),
		merge_analyzer_chunk_duration=(
			str(merge_analyzer_chunk_duration) if merge_analyzer_chunk_duration is not None else None
		),
		merge_analyzer_sparsity_method=str(merge_analyzer_sparsity_method),
		merge_analyzer_sparsity_radius_um=(
			float(merge_analyzer_sparsity_radius_um)
			if merge_analyzer_sparsity_radius_um is not None
			else None
		),
		merge_analyzer_sparsity_num_channels=(
			int(merge_analyzer_sparsity_num_channels)
			if merge_analyzer_sparsity_num_channels is not None
			else None
		),
		merge_analyzer_sparsity_threshold=(
			float(merge_analyzer_sparsity_threshold)
			if merge_analyzer_sparsity_threshold is not None
			else None
		),
		merge_analyzer_sparsity_peak_sign=str(merge_analyzer_sparsity_peak_sign),
		merge_analyzer_sparsity_num_spikes_for_sparsity=(
			int(merge_analyzer_sparsity_num_spikes_for_sparsity)
			if merge_analyzer_sparsity_num_spikes_for_sparsity is not None
			else None
		),
		merge_analyzer_sparsity_by_property=(
			str(merge_analyzer_sparsity_by_property)
			if merge_analyzer_sparsity_by_property is not None
			else None
		),
		merge_analyzer_waveforms_ms_before=(
			float(merge_analyzer_waveforms_ms_before)
			if merge_analyzer_waveforms_ms_before is not None
			else None
		),
		merge_analyzer_waveforms_ms_after=(
			float(merge_analyzer_waveforms_ms_after)
			if merge_analyzer_waveforms_ms_after is not None
			else None
		),
		merge_analyzer_waveforms_dtype=(
			str(merge_analyzer_waveforms_dtype)
			if merge_analyzer_waveforms_dtype is not None
			else None
		),
		cache_sorting_outputs_before_merge=bool(cache_sorting_outputs_before_merge),
		cache_sorting_outputs_before_merge_relpath=str(cache_sorting_outputs_before_merge_relpath),
		cache_sorting_outputs_before_merge_cleanup_on_success=bool(cache_sorting_outputs_before_merge_cleanup_on_success),
		cache_sorting_outputs_before_merge_use_cache_on_force_restart=bool(
			cache_sorting_outputs_before_merge_use_cache_on_force_restart
		),
		cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart=bool(
			cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart
		),
		cache_sorting_outputs_before_merge_refresh_on_run=bool(
			cache_sorting_outputs_before_merge_refresh_on_run
		),
		cache_sorting_outputs_before_merge_strict_restore_on_force_restart=bool(
			cache_sorting_outputs_before_merge_strict_restore_on_force_restart
		),
		cache_sorting_outputs_before_merge_use_canonical_workspace=bool(
			cache_sorting_outputs_before_merge_use_canonical_workspace
		),
		cache_sorting_outputs_before_merge_canonical_workspace_relpath=str(
			cache_sorting_outputs_before_merge_canonical_workspace_relpath
		),
		cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run=bool(
			cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run
		),
		cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer=bool(
			cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer
		),
		cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success=bool(
			cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success
		),
		cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure=bool(
			cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure
		),
		cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace=bool(
			cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace
		),
		cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace=bool(
			cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace
		),
		merge_reports_enabled=bool(merge_reports_enabled),
		merge_reports_unit_diff_json_enabled=bool(merge_reports_unit_diff_json_enabled),
		merge_reports_unit_diff_json_relpath=str(merge_reports_unit_diff_json_relpath),
		merge_reports_unit_diff_map_enabled=bool(merge_reports_unit_diff_map_enabled),
		merge_reports_unit_diff_map_relpath=str(merge_reports_unit_diff_map_relpath),
		merge_reports_unit_diff_map_flat_enabled=bool(merge_reports_unit_diff_map_flat_enabled),
		merge_reports_unit_diff_map_flat_relpath=str(merge_reports_unit_diff_map_flat_relpath),
		merge_reports_post_merge_unit_locations_enabled=bool(merge_reports_post_merge_unit_locations_enabled),
		merge_reports_post_merge_unit_locations_relpath=str(merge_reports_post_merge_unit_locations_relpath),
		merge_reports_2panel_enabled=bool(merge_reports_2panel_enabled),
		merge_reports_2panel_point_size=float(merge_reports_2panel_point_size),
		merge_reports_2panel_relpath=str(merge_reports_2panel_relpath),
		merge_reports_2panel_label_pre_and_post_units=bool(merge_reports_2panel_label_pre_and_post_units),
		merge_reports_2panel_write_png=bool(merge_reports_2panel_write_png),
		merge_reports_2panel_write_svg=bool(merge_reports_2panel_write_svg),
		merge_reports_2panel_before_relpath=str(merge_reports_2panel_before_relpath),
		merge_reports_2panel_before_write_png=bool(merge_reports_2panel_before_write_png),
		merge_reports_2panel_before_write_svg=bool(merge_reports_2panel_before_write_svg),
		merge_reports_2panel_before_point_color=str(merge_reports_2panel_before_point_color),
		merge_reports_2panel_after_relpath=str(merge_reports_2panel_after_relpath),
		merge_reports_2panel_after_write_png=bool(merge_reports_2panel_after_write_png),
		merge_reports_2panel_after_write_svg=bool(merge_reports_2panel_after_write_svg),
		merge_reports_2panel_after_point_color=str(merge_reports_2panel_after_point_color),
		merge_reports_2panel_highlight_merges_enabled=bool(merge_reports_2panel_highlight_merges_enabled),
		merge_reports_2panel_highlight_merges_linked=bool(merge_reports_2panel_highlight_merges_linked),
		merge_reports_2panel_highlight_plot_after_other_units=bool(
			merge_reports_2panel_highlight_plot_after_other_units
		),
		merge_reports_2panel_highlight_label_affected_units=bool(
			merge_reports_2panel_highlight_label_affected_units
		),
		merge_reports_2panel_highlight_show_legend=bool(merge_reports_2panel_highlight_show_legend),
		merge_reports_2panel_highlight_legend_position=str(merge_reports_2panel_highlight_legend_position),
		merge_reports_2panel_highlight_legend_x=float(merge_reports_2panel_highlight_legend_x),
		merge_reports_2panel_highlight_legend_y=float(merge_reports_2panel_highlight_legend_y),
		merge_reports_2panel_highlight_sort_pre_legend_by_groups=bool(
			merge_reports_2panel_highlight_sort_pre_legend_by_groups
		),
		merge_reports_2panel_highlight_debug_json_enabled=bool(
			merge_reports_2panel_highlight_debug_json_enabled
		),
		merge_reports_2panel_highlight_debug_json_relpath=str(
			merge_reports_2panel_highlight_debug_json_relpath
		),
		merge_reports_2panel_highlight_before_color=str(merge_reports_2panel_highlight_before_color),
		merge_reports_2panel_highlight_after_color=str(merge_reports_2panel_highlight_after_color),
		merge_reports_2panel_highlight_palette=str(merge_reports_2panel_highlight_palette),
		merge_reports_2panel_inherit_probe_dimensions=bool(merge_reports_2panel_inherit_probe_dimensions),
		merge_reports_2panel_zoom_to_affected_units=bool(merge_reports_2panel_zoom_to_affected_units),
		merge_reports_2panel_probe_dim_x_um=(
			float(merge_reports_2panel_probe_dim_x_um)
			if merge_reports_2panel_probe_dim_x_um is not None
			else None
		),
		merge_reports_2panel_probe_dim_y_um=(
			float(merge_reports_2panel_probe_dim_y_um)
			if merge_reports_2panel_probe_dim_y_um is not None
			else None
		),
		merge_reports_template_heatmaps_enabled=bool(merge_reports_template_heatmaps_enabled),
		merge_reports_template_heatmaps_relpath=str(merge_reports_template_heatmaps_relpath),
		merge_reports_template_heatmaps_assets_reldir=str(merge_reports_template_heatmaps_assets_reldir),
		merge_reports_template_heatmaps_write_png=bool(merge_reports_template_heatmaps_write_png),
		merge_reports_template_heatmaps_write_svg=bool(merge_reports_template_heatmaps_write_svg),
		merge_reports_template_heatmaps_write_assets_png=bool(merge_reports_template_heatmaps_write_assets_png),
		merge_reports_template_heatmaps_write_assets_svg=bool(merge_reports_template_heatmaps_write_assets_svg),
		merge_reports_template_heatmaps_panel_width_in=float(merge_reports_template_heatmaps_panel_width_in),
		merge_reports_template_heatmaps_panel_height_in=float(merge_reports_template_heatmaps_panel_height_in),
		merge_reports_template_heatmaps_marker_size=float(merge_reports_template_heatmaps_marker_size),
		merge_reports_template_heatmaps_cmap=str(merge_reports_template_heatmaps_cmap),
		merge_reports_template_heatmaps_show_colorbar=bool(merge_reports_template_heatmaps_show_colorbar),
		merge_reports_template_heatmaps_relative_color_bar_height=float(
			merge_reports_template_heatmaps_relative_color_bar_height
		),
		merge_reports_template_heatmaps_color_scale=str(merge_reports_template_heatmaps_color_scale),
		merge_reports_template_heatmaps_log_epsilon=float(merge_reports_template_heatmaps_log_epsilon),
		merge_reports_template_heatmaps_magnitude_mode=str(merge_reports_template_heatmaps_magnitude_mode),
		merge_reports_template_heatmaps_max_merges=(
			int(merge_reports_template_heatmaps_max_merges)
			if merge_reports_template_heatmaps_max_merges is not None
			else None
		),
		merge_reports_template_heatmaps_debug_json_relpath=str(
			merge_reports_template_heatmaps_debug_json_relpath
		),
		merge_reports_template_heatmaps_inherit_probe_dimensions=bool(
			merge_reports_template_heatmaps_inherit_probe_dimensions
		),
		merge_reports_template_heatmaps_probe_dim_x_um=(
			float(merge_reports_template_heatmaps_probe_dim_x_um)
			if merge_reports_template_heatmaps_probe_dim_x_um is not None
			else None
		),
		merge_reports_template_heatmaps_probe_dim_y_um=(
			float(merge_reports_template_heatmaps_probe_dim_y_um)
			if merge_reports_template_heatmaps_probe_dim_y_um is not None
			else None
		),
		merge_reports_template_heatmaps_probe_pitch_um=(
			float(merge_reports_template_heatmaps_probe_pitch_um)
			if merge_reports_template_heatmaps_probe_pitch_um is not None
			else None
		),
		merge_reports_template_heatmaps_probe_electrode_size_um_x=(
			float(merge_reports_template_heatmaps_probe_electrode_size_um_x)
			if merge_reports_template_heatmaps_probe_electrode_size_um_x is not None
			else None
		),
		merge_reports_template_heatmaps_probe_electrode_size_um_y=(
			float(merge_reports_template_heatmaps_probe_electrode_size_um_y)
			if merge_reports_template_heatmaps_probe_electrode_size_um_y is not None
			else None
		),
		merge_metadata_enabled=bool(merge_metadata_enabled),
		merge_metadata_write_json=bool(merge_metadata_write_json),
		merge_metadata_json_relpath=str(merge_metadata_json_relpath),
		merge_metadata_include_unit_locations=bool(merge_metadata_include_unit_locations),
		merge_metadata_log_summary_details=bool(merge_metadata_log_summary_details),
		pre_merge_metadata_enabled=bool(pre_merge_metadata_enabled),
		pre_merge_metadata_write_json=bool(pre_merge_metadata_write_json),
		pre_merge_metadata_json_relpath=str(pre_merge_metadata_json_relpath),
		pre_merge_metadata_include_unit_locations=bool(pre_merge_metadata_include_unit_locations),
		pre_merge_metadata_log_summary_details=bool(pre_merge_metadata_log_summary_details),
		post_merge_metadata_enabled=bool(post_merge_metadata_enabled),
		post_merge_metadata_write_json=bool(post_merge_metadata_write_json),
		post_merge_metadata_json_relpath=str(post_merge_metadata_json_relpath),
		post_merge_metadata_include_unit_locations=bool(post_merge_metadata_include_unit_locations),
		post_merge_metadata_log_summary_details=bool(post_merge_metadata_log_summary_details),
		force_restart=force_restart,
		force_replot=force_replot,
		resume_from=_as_optional_str(_get_with_fallback(execution_cfg, stage_cfg, "resume_from", None)),
	)


def build_spikesort_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: SpikesortStageConfig,
	unit_workers: int,
) -> SpikesortInputs:
	n_jobs = stage_config.n_jobs
	if n_jobs is None:
		n_jobs = max(1, int(unit_workers))
	return SpikesortInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		final_output_root=(target.final_output_root or target.mea_output_root),
		output_rel_root=stage_config.output_rel_root,
		preprocess_concat_recording_relpath=stage_config.preprocess_concat_recording_relpath,
		logging_enabled=stage_config.logging_enabled,
		logging_verbose=stage_config.logging_verbose,
		logging_file_relpath=stage_config.logging_file_relpath,
		sorter=stage_config.sorter,
		docker_image=stage_config.docker_image,
		recording_num=stage_config.recording_num,
		verbose=stage_config.verbose,
		ks_batch_duration_s=stage_config.ks_batch_duration_s,
		ks_batch_size=stage_config.ks_batch_size,
		ks_th_universal=stage_config.ks_th_universal,
		ks_th_learned=stage_config.ks_th_learned,
		ks_th_single_ch=stage_config.ks_th_single_ch,
		ks_cluster_downsampling=stage_config.ks_cluster_downsampling,
		ks_nearest_chans=stage_config.ks_nearest_chans,
		ks_max_channel_distance=stage_config.ks_max_channel_distance,
		n_jobs=n_jobs,
		chunk_duration=stage_config.chunk_duration,
		cuda_visible_devices=stage_config.cuda_visible_devices,
		run_analyzer=stage_config.run_analyzer,
		run_reports=stage_config.run_reports,
		sort_enabled=stage_config.sort_enabled,
		sort_delete_outputs_on_force_restart=stage_config.sort_delete_outputs_on_force_restart,
		plot_enabled=stage_config.plot_enabled,
		plot_mode=stage_config.plot_mode,
		plot_debug=stage_config.plot_debug,
		raster_sort=stage_config.raster_sort,
		fixed_y=stage_config.fixed_y,
		no_curation=stage_config.no_curation,
		export_to_phy=stage_config.export_to_phy,
		force_rerun_analyzer=stage_config.force_rerun_analyzer,
		summarize_sort_enabled=stage_config.summarize_sort_enabled,
		summarize_sort_emit_logs=stage_config.summarize_sort_emit_logs,
		summarize_sort_generate_artifacts=stage_config.summarize_sort_generate_artifacts,
		um_kwargs=stage_config.um_kwargs,
		am_kwargs=stage_config.am_kwargs,
		option_kwargs=stage_config.option_kwargs,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		resume_from=stage_config.resume_from,
		merge_analyzer_compute_sparsity=stage_config.merge_analyzer_compute_sparsity,
		merge_analyzer_density_mode=stage_config.merge_analyzer_density_mode,
		merge_template_random_spikes_method=stage_config.merge_template_random_spikes_method,
		merge_template_random_spikes_percentage=stage_config.merge_template_random_spikes_percentage,
		merge_template_random_spikes_max_spikes_per_unit=stage_config.merge_template_random_spikes_max_spikes_per_unit,
		merge_template_random_spikes_min_spikes_per_unit=stage_config.merge_template_random_spikes_min_spikes_per_unit,
		merge_template_random_spikes_log_before_after_spike_counts=stage_config.merge_template_random_spikes_log_before_after_spike_counts,
		merge_template_random_spikes_margin_size=stage_config.merge_template_random_spikes_margin_size,
		merge_template_random_spikes_seed=stage_config.merge_template_random_spikes_seed,
		merge_analyzer_n_jobs=stage_config.merge_analyzer_n_jobs,
		merge_analyzer_chunk_duration=stage_config.merge_analyzer_chunk_duration,
		merge_analyzer_sparsity_method=stage_config.merge_analyzer_sparsity_method,
		merge_analyzer_sparsity_radius_um=stage_config.merge_analyzer_sparsity_radius_um,
		merge_analyzer_sparsity_num_channels=stage_config.merge_analyzer_sparsity_num_channels,
		merge_analyzer_sparsity_threshold=stage_config.merge_analyzer_sparsity_threshold,
		merge_analyzer_sparsity_peak_sign=stage_config.merge_analyzer_sparsity_peak_sign,
		merge_analyzer_sparsity_num_spikes_for_sparsity=stage_config.merge_analyzer_sparsity_num_spikes_for_sparsity,
		merge_analyzer_sparsity_by_property=stage_config.merge_analyzer_sparsity_by_property,
		merge_analyzer_waveforms_ms_before=stage_config.merge_analyzer_waveforms_ms_before,
		merge_analyzer_waveforms_ms_after=stage_config.merge_analyzer_waveforms_ms_after,
		merge_analyzer_waveforms_dtype=stage_config.merge_analyzer_waveforms_dtype,
	)


def load_spikesort_inputs_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> SpikesortInputs:
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

	stage_cfg = parse_spikesort_stage_config(
		runtime_config=runtime_cfg,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	n_jobs = stage_cfg.n_jobs if stage_cfg.n_jobs is not None else 1
	return SpikesortInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		final_output_root=output_root,
		output_rel_root=stage_cfg.output_rel_root,
		preprocess_concat_recording_relpath=stage_cfg.preprocess_concat_recording_relpath,
		logging_enabled=stage_cfg.logging_enabled,
		logging_verbose=stage_cfg.logging_verbose,
		logging_file_relpath=stage_cfg.logging_file_relpath,
		sorter=stage_cfg.sorter,
		docker_image=stage_cfg.docker_image,
		recording_num=stage_cfg.recording_num,
		verbose=stage_cfg.verbose,
		ks_batch_duration_s=stage_cfg.ks_batch_duration_s,
		ks_batch_size=stage_cfg.ks_batch_size,
		ks_th_universal=stage_cfg.ks_th_universal,
		ks_th_learned=stage_cfg.ks_th_learned,
		ks_th_single_ch=stage_cfg.ks_th_single_ch,
		ks_cluster_downsampling=stage_cfg.ks_cluster_downsampling,
		ks_nearest_chans=stage_cfg.ks_nearest_chans,
		ks_max_channel_distance=stage_cfg.ks_max_channel_distance,
		n_jobs=n_jobs,
		chunk_duration=stage_cfg.chunk_duration,
		cuda_visible_devices=stage_cfg.cuda_visible_devices,
		run_analyzer=stage_cfg.run_analyzer,
		run_reports=stage_cfg.run_reports,
		sort_enabled=stage_cfg.sort_enabled,
		sort_delete_outputs_on_force_restart=stage_cfg.sort_delete_outputs_on_force_restart,
		plot_enabled=stage_cfg.plot_enabled,
		plot_mode=stage_cfg.plot_mode,
		plot_debug=stage_cfg.plot_debug,
		raster_sort=stage_cfg.raster_sort,
		fixed_y=stage_cfg.fixed_y,
		no_curation=stage_cfg.no_curation,
		export_to_phy=stage_cfg.export_to_phy,
		force_rerun_analyzer=stage_cfg.force_rerun_analyzer,
		summarize_sort_enabled=stage_cfg.summarize_sort_enabled,
		summarize_sort_emit_logs=stage_cfg.summarize_sort_emit_logs,
		summarize_sort_generate_artifacts=stage_cfg.summarize_sort_generate_artifacts,
		um_kwargs=stage_cfg.um_kwargs,
		am_kwargs=stage_cfg.am_kwargs,
		option_kwargs=stage_cfg.option_kwargs,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		resume_from=stage_cfg.resume_from,
		merge_analyzer_compute_sparsity=stage_cfg.merge_analyzer_compute_sparsity,
		merge_analyzer_density_mode=stage_cfg.merge_analyzer_density_mode,
		merge_template_random_spikes_method=stage_cfg.merge_template_random_spikes_method,
		merge_template_random_spikes_percentage=stage_cfg.merge_template_random_spikes_percentage,
		merge_template_random_spikes_max_spikes_per_unit=stage_cfg.merge_template_random_spikes_max_spikes_per_unit,
		merge_template_random_spikes_min_spikes_per_unit=stage_cfg.merge_template_random_spikes_min_spikes_per_unit,
		merge_template_random_spikes_log_before_after_spike_counts=stage_cfg.merge_template_random_spikes_log_before_after_spike_counts,
		merge_template_random_spikes_margin_size=stage_cfg.merge_template_random_spikes_margin_size,
		merge_template_random_spikes_seed=stage_cfg.merge_template_random_spikes_seed,
		merge_analyzer_n_jobs=stage_cfg.merge_analyzer_n_jobs,
		merge_analyzer_chunk_duration=stage_cfg.merge_analyzer_chunk_duration,
		merge_analyzer_sparsity_method=stage_cfg.merge_analyzer_sparsity_method,
		merge_analyzer_sparsity_radius_um=stage_cfg.merge_analyzer_sparsity_radius_um,
		merge_analyzer_sparsity_num_channels=stage_cfg.merge_analyzer_sparsity_num_channels,
		merge_analyzer_sparsity_threshold=stage_cfg.merge_analyzer_sparsity_threshold,
		merge_analyzer_sparsity_peak_sign=stage_cfg.merge_analyzer_sparsity_peak_sign,
		merge_analyzer_sparsity_num_spikes_for_sparsity=stage_cfg.merge_analyzer_sparsity_num_spikes_for_sparsity,
		merge_analyzer_sparsity_by_property=stage_cfg.merge_analyzer_sparsity_by_property,
		merge_analyzer_waveforms_ms_before=stage_cfg.merge_analyzer_waveforms_ms_before,
		merge_analyzer_waveforms_ms_after=stage_cfg.merge_analyzer_waveforms_ms_after,
		merge_analyzer_waveforms_dtype=stage_cfg.merge_analyzer_waveforms_dtype,
	)
