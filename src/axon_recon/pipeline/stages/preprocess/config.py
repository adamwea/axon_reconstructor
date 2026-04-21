from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig
from axon_reconstructor.pipeline.stg1_preprocessing.constants import (
	LEGACY_PREPROCESS_OUTPUTS_DIRNAME,
	PREPROCESS_OUTPUTS_DIRNAME,
)

from ...execution.context import ExecutionTarget
from .models.inputs import (
	PreprocessConcatSegmentsPhaseConfig,
	PreprocessConcatenatePreprocessedRecordingsPhaseConfig,
	PreprocessCopySrcToScratchPhaseConfig,
	PreprocessInputs,
	PreprocessPhaseConfig,
	PreprocessPhaseOutputsConfig,
	PreprocessPhasesConfig,
	PreprocessPlotConfig,
	PreprocessPlotConcatTracesPhaseConfig,
	PreprocessPlotSegmentTracesPhaseConfig,
	PreprocessSaveRecMetadataPhaseConfig,
	PreprocessSegmentsPhaseConfig,
	PreprocessWipeSrcScratchPhaseConfig,
)


_DEFAULT_OUTPUT_REL_ROOT = PREPROCESS_OUTPUTS_DIRNAME


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
		parsed = int(value)
	except Exception:
		return None
	return parsed if parsed > 0 else None


def _as_optional_float(value: Any) -> float | None:
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _normalize_positive_int_or_all(value: Any, default: int) -> int:
	parsed = _as_int(value, int(default))
	return (-1 if int(parsed) <= 0 else max(1, int(parsed)))


def _as_optional_str(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text if text else None


def _as_mapping_or_none(value: Any) -> dict[str, Any] | None:
	if isinstance(value, dict):
		return dict(value)
	return None


def _plot_toggle_enabled(*, raw: Any, default: bool) -> bool:
	block = _as_mapping_or_none(raw)
	if block is None:
		return _as_bool(raw, default)
	if "enabled" in block:
		return _as_bool(block.get("enabled"), default)
	return bool(default)


def _plot_toggle_n_reps(*, raw: Any, default: int) -> int:
	block = _as_mapping_or_none(raw)
	if block is None:
		return int(default)
	return _normalize_positive_int_or_all(
		block.get("n_reps", block.get("n_representative_channels", default)),
		int(default),
	)


def _normalize_observability_mode(value: Any) -> str:
	token = str(value or "off").strip().lower()
	if token in {"off", "none", "disabled", "false", "0"}:
		return "off"
	if token in {"basic", "standard", "on", "enabled", "true", "1"}:
		return "basic"
	if token in {"detailed", "verbose", "debug", "full", "meta"}:
		return "detailed"
	return "off"


def _normalize_output_rel_root(raw: Any) -> str:
	text = str(raw or _DEFAULT_OUTPUT_REL_ROOT).strip()
	if not text:
		return _DEFAULT_OUTPUT_REL_ROOT
	text = text.lstrip("/")
	if text == LEGACY_PREPROCESS_OUTPUTS_DIRNAME:
		return PREPROCESS_OUTPUTS_DIRNAME
	return text or _DEFAULT_OUTPUT_REL_ROOT


def _normalize_plot_output_dir(raw: Any) -> str | None:
	text = _as_optional_str(raw)
	if text is None:
		return None
	path = Path(text)
	if path.is_absolute():
		return text
	parts = list(path.parts)
	if parts and parts[0] == LEGACY_PREPROCESS_OUTPUTS_DIRNAME:
		parts[0] = PREPROCESS_OUTPUTS_DIRNAME
		return str(Path(*parts))
	return text


def _resolve_data_config_path(runtime_config_path: Path, data_ref: str | None) -> Path:
	if not data_ref:
		raise ValueError("Runtime config must define data: <path-to-data-config>")
	p = Path(str(data_ref)).expanduser()
	if not p.is_absolute():
		p = (runtime_config_path.parent / p).resolve()
	return p


@dataclass(frozen=True)
class PreprocessStageConfig:
	output_rel_root: str
	force_restart: bool
	force_replot: bool
	debug_limit_wells: int | None
	debug_limit_segments_per_well: int | None
	logging_enabled: bool
	logging_verbose: bool
	logging_file_relpath: str | None
	logging_suppress_h5_plugin_messages: bool
	logging_phase_dividers: bool
	enable_checkpointing: bool
	n_jobs: int | None
	plot_layouts: bool
	plot_concat_trace: bool
	plot_segment_traces: bool
	plot_output_dir: str | None
	epoch_markers_output_dir: str | None
	assay_stats_relpath: str
	channel_layouts_subdir: str
	segment_traces_subdir: str
	concat_trace_relpath: str
	n_representative_channels: int
	concat_trace_n_reps: int
	segment_trace_n_reps: int
	plot_n_jobs: int | None
	trace_downsample_hz: float | None
	trace_max_points: int
	observability_mode: str
	observability_output_subdir: str
	observability_save_run_manifest: bool
	observability_save_event_timeline: bool
	observability_save_environment: bool
	observability_save_artifact_inventory: bool
	observability_save_stage_log: bool
	observability_stage_log_relpath: str
	temporal_resample_factor: int | None
	temporal_resample_rate_hz: int | None
	temporal_resample_margin_ms: float
	temporal_resample_dtype: str | None
	save_recording: bool
	overwrite_saved_recording: bool
	save_concat_recording: bool
	save_segment_recordings: bool
	save_chunk_duration: str
	save_progress_bar: bool
	concat_save_n_jobs: int | None
	segment_save_n_jobs: int | None
	print_n_jobs_used: bool
	phases: PreprocessPhasesConfig


def _parse_simple_phase_config(
	*,
	raw_cfg: dict[str, Any] | None,
	default_enabled: bool,
	default_summary_json_relpath: str,
) -> PreprocessPhaseConfig:
	phase_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
	return PreprocessPhaseConfig(
		enabled=_as_bool(phase_cfg.get("enabled", phase_cfg.get("enable", default_enabled)), default_enabled),
		summary_json_relpath=str(
			phase_cfg.get("summary_json_relpath", default_summary_json_relpath)
			or default_summary_json_relpath
		),
	)


def _parse_save_rec_metadata_phase_config(
	*,
	raw_cfg: dict[str, Any] | None,
) -> PreprocessSaveRecMetadataPhaseConfig:
	phase_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
	return PreprocessSaveRecMetadataPhaseConfig(
		enabled=_as_bool(phase_cfg.get("enabled", phase_cfg.get("enable", False)), False),
		verbose=_as_bool(phase_cfg.get("verbose", False), False),
		summary_json_relpath=str(
			phase_cfg.get("summary_json_relpath", "context/recording_metadata_summary.json")
			or "context/recording_metadata_summary.json"
		),
		segment_epochs_relpath=str(
			phase_cfg.get("segment_epochs_relpath", "segment_epochs.json")
			or "segment_epochs.json"
		),
		contiguous_epochs_relpath=str(
			phase_cfg.get("contiguous_epochs_relpath", "continuous_epochs.json")
			or "continuous_epochs.json"
		),
		sampling_metadata_relpath=str(
			phase_cfg.get("sampling_metadata_relpath", "sampling_rate_metadata.json")
			or "sampling_rate_metadata.json"
		),
		common_electrodes_relpath=str(
			phase_cfg.get("common_electrodes_relpath", "common_electrodes.npy")
			or "common_electrodes.npy"
		),
		common_electrodes_summary_json_relpath=str(
			phase_cfg.get("common_electrodes_summary_json_relpath", "context/save_common_electrodes_summary.json")
			or "context/save_common_electrodes_summary.json"
		),
	)


def _parse_plot_phase_config(*, raw_cfg: dict[str, Any] | None, defaults: PreprocessPlotConfig) -> PreprocessPlotConfig:
	plot_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
	concat_trace_raw = plot_cfg.get("concat_trace", defaults.concat_trace)
	segment_traces_raw = plot_cfg.get("segment_traces", defaults.segment_traces)
	per_segment_traces_raw = plot_cfg.get("per_segment_traces", None)
	segment_trace_block_raw = per_segment_traces_raw if per_segment_traces_raw is not None else segment_traces_raw

	plot_concat_trace = _plot_toggle_enabled(raw=concat_trace_raw, default=bool(defaults.concat_trace))
	segment_default_enabled = _as_bool(segment_traces_raw, bool(defaults.segment_traces))
	plot_segment_traces = _plot_toggle_enabled(raw=segment_trace_block_raw, default=segment_default_enabled)
	plot_layouts = _as_bool(plot_cfg.get("layouts", defaults.layouts), bool(defaults.layouts))

	raw_disable_all_png_diagnostics = plot_cfg.get(
		"disable_all_png_diagnostics",
		defaults.disable_all_png_diagnostics,
	)
	disable_all_png_diagnostics = (
		None
		if raw_disable_all_png_diagnostics is None
		else _as_bool(raw_disable_all_png_diagnostics, False)
	)
	if raw_disable_all_png_diagnostics is not None:
		force_all_plots_enabled = not bool(disable_all_png_diagnostics)
		plot_layouts = bool(force_all_plots_enabled)
		plot_concat_trace = bool(force_all_plots_enabled)
		plot_segment_traces = bool(force_all_plots_enabled)

	raw_n_representative_channels = plot_cfg.get(
		"n_representative_channels",
		plot_cfg.get("n_reps_per_segment", defaults.n_representative_channels),
	)
	raw_segment_trace_n_reps = plot_cfg.get("n_reps_per_segment", None)
	n_representative_channels = _normalize_positive_int_or_all(
		raw_n_representative_channels,
		int(defaults.n_representative_channels),
	)
	concat_trace_n_reps = _plot_toggle_n_reps(raw=concat_trace_raw, default=int(defaults.concat_trace_n_reps))
	segment_trace_n_reps = _plot_toggle_n_reps(
		raw=segment_trace_block_raw,
		default=int(defaults.segment_trace_n_reps),
	)
	if raw_segment_trace_n_reps is not None:
		segment_trace_n_reps = _normalize_positive_int_or_all(
			raw_segment_trace_n_reps,
			int(defaults.segment_trace_n_reps),
		)
	plot_n_jobs = _as_optional_int(plot_cfg.get("n_jobs", defaults.n_jobs))
	trace_downsample_hz = _as_optional_float(plot_cfg.get("trace_downsample_hz", defaults.trace_downsample_hz))
	if trace_downsample_hz is not None and trace_downsample_hz <= 0.0:
		trace_downsample_hz = None

	raw_trace_max_points = _as_int(plot_cfg.get("trace_max_points", defaults.trace_max_points), defaults.trace_max_points)
	trace_max_points = (-1 if int(raw_trace_max_points) <= 0 else max(1000, int(raw_trace_max_points)))

	return PreprocessPlotConfig(
		disable_all_png_diagnostics=disable_all_png_diagnostics,
		layouts=bool(plot_layouts),
		concat_trace=bool(plot_concat_trace),
		segment_traces=bool(plot_segment_traces),
		output_dir=_normalize_plot_output_dir(plot_cfg.get("output_dir", defaults.output_dir)),
		epoch_markers_output_dir=_normalize_plot_output_dir(
			plot_cfg.get("epoch_markers_output_dir", defaults.epoch_markers_output_dir)
		),
		assay_stats_relpath=str(plot_cfg.get("assay_stats_relpath", defaults.assay_stats_relpath) or defaults.assay_stats_relpath),
		channel_layouts_subdir=str(
			plot_cfg.get("channel_layouts_subdir", defaults.channel_layouts_subdir)
			or defaults.channel_layouts_subdir
		),
		segment_traces_subdir=str(
			plot_cfg.get("segment_traces_subdir", defaults.segment_traces_subdir)
			or defaults.segment_traces_subdir
		),
		concat_trace_relpath=str(
			plot_cfg.get("concat_trace_relpath", defaults.concat_trace_relpath)
			or defaults.concat_trace_relpath
		),
		n_representative_channels=int(n_representative_channels),
		concat_trace_n_reps=int(concat_trace_n_reps),
		segment_trace_n_reps=int(segment_trace_n_reps),
		n_jobs=plot_n_jobs,
		trace_downsample_hz=trace_downsample_hz,
		trace_max_points=int(trace_max_points),
	)


def _parse_phase_outputs_config(
	*,
	raw_cfg: dict[str, Any] | None,
	defaults: PreprocessPhaseOutputsConfig,
) -> PreprocessPhaseOutputsConfig:
	outputs_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
	return PreprocessPhaseOutputsConfig(
		save_chunk_duration=_as_optional_str(
			outputs_cfg.get("save_chunk_duration", defaults.save_chunk_duration)
		)
		or defaults.save_chunk_duration,
		save_progress_bar=_as_bool(
			outputs_cfg.get("save_progress_bar", defaults.save_progress_bar),
			bool(defaults.save_progress_bar),
		),
		concat_save_n_jobs=_as_optional_int(outputs_cfg.get("concat_save_n_jobs", defaults.concat_save_n_jobs)),
		segment_save_n_jobs=_as_optional_int(outputs_cfg.get("segment_save_n_jobs", defaults.segment_save_n_jobs)),
		print_n_jobs_used=_as_bool(
			outputs_cfg.get("print_n_jobs_used", defaults.print_n_jobs_used),
			bool(defaults.print_n_jobs_used),
		),
	)


def parse_preprocess_stage_config(
	*,
	runtime_config: RuntimeConfig,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> PreprocessStageConfig:
	stage_cfg = runtime_config.get("stages.preprocess", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	debug_cfg = stage_cfg.get("debug", {}) if isinstance(stage_cfg.get("debug", {}), dict) else {}
	logging_cfg = stage_cfg.get("logging", {}) if isinstance(stage_cfg.get("logging", {}), dict) else {}
	plot_cfg = stage_cfg.get("plot", {}) if isinstance(stage_cfg.get("plot", {}), dict) else {}
	observability_cfg = stage_cfg.get("observability", {}) if isinstance(stage_cfg.get("observability", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}
	phases_cfg = stage_cfg.get("phases", {}) if isinstance(stage_cfg.get("phases", {}), dict) else {}
	using_new_phase_schema = any(
		key in phases_cfg
		for key in (
			"copy_src_to_scratch",
			"save_rec_metadata",
			"wipe_src_scratch",
			"preprocess_segments",
			"plot_segment_traces",
			"concat_segments",
			"plot_concat_traces",
			"concatenate_recordings",
			"concatenate_preprocessed_recordings",
		)
	)

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if force_replot_override is not None:
		force_replot = bool(force_replot_override)

	legacy_debug_default = _as_bool(execution_cfg.get("debug", True), True)
	logging_enabled = _as_bool(logging_cfg.get("enabled", True), True)
	logging_verbose = _as_bool(logging_cfg.get("verbose", legacy_debug_default), legacy_debug_default)
	logging_file_relpath = _as_optional_str(logging_cfg.get("file_relpath", None))
	logging_suppress_h5_plugin_messages = _as_bool(logging_cfg.get("suppress_h5_plugin_messages", False), False)
	logging_phase_dividers = _as_bool(logging_cfg.get("phase_dividers", True), True)

	concat_trace_raw = plot_cfg.get("concat_trace", True)
	segment_traces_raw = plot_cfg.get("segment_traces", True)
	per_segment_traces_raw = plot_cfg.get("per_segment_traces", None)
	segment_trace_block_raw = per_segment_traces_raw if per_segment_traces_raw is not None else segment_traces_raw

	plot_concat_trace_cfg = _plot_toggle_enabled(raw=concat_trace_raw, default=True)
	segment_default_enabled = _as_bool(segment_traces_raw, True)
	plot_segment_traces_cfg = _plot_toggle_enabled(raw=segment_trace_block_raw, default=segment_default_enabled)

	# Stage-level master override for preprocess PNG diagnostics.
	raw_disable_all_png_diagnostics = plot_cfg.get("disable_all_png_diagnostics", None)
	if raw_disable_all_png_diagnostics is not None:
		disable_all_png_diagnostics = _as_bool(raw_disable_all_png_diagnostics, False)
		force_all_plots_enabled = not bool(disable_all_png_diagnostics)
		plot_layouts_effective = bool(force_all_plots_enabled)
		plot_concat_trace_effective = bool(force_all_plots_enabled)
		plot_segment_traces_effective = bool(force_all_plots_enabled)
	else:
		plot_layouts_effective = _as_bool(execution_cfg.get("plot_layouts", plot_cfg.get("layouts", True)), True)
		plot_concat_trace_effective = _as_bool(
			execution_cfg.get("plot_concat_trace", plot_concat_trace_cfg),
			True,
		)
		plot_segment_traces_effective = _as_bool(
			execution_cfg.get("plot_segment_traces", plot_segment_traces_cfg),
			True,
		)

	trace_downsample_hz = _as_optional_float(plot_cfg.get("trace_downsample_hz", None))
	if trace_downsample_hz is not None and trace_downsample_hz <= 0.0:
		trace_downsample_hz = None

	raw_n_representative_channels = plot_cfg.get(
		"n_representative_channels",
		plot_cfg.get("n_reps_per_segment", 4),
	)
	n_representative_channels = _normalize_positive_int_or_all(raw_n_representative_channels, 4)
	concat_trace_n_reps = _plot_toggle_n_reps(raw=concat_trace_raw, default=n_representative_channels)
	segment_trace_n_reps = _plot_toggle_n_reps(raw=segment_trace_block_raw, default=n_representative_channels)
	plot_n_jobs = _as_optional_int(plot_cfg.get("n_jobs", None))

	raw_trace_max_points = _as_int(plot_cfg.get("trace_max_points", 150000), 150000)
	trace_max_points = (-1 if int(raw_trace_max_points) <= 0 else max(1000, int(raw_trace_max_points)))

	observability_mode = _normalize_observability_mode(observability_cfg.get("mode", "off"))
	obs_enabled_default = observability_mode in {"basic", "detailed"}
	obs_enabled = _as_bool(observability_cfg.get("enabled", obs_enabled_default), obs_enabled_default)

	obs_output_subdir = str(observability_cfg.get("output_subdir", "run_metadata") or "run_metadata").strip()
	if not obs_output_subdir:
		obs_output_subdir = "run_metadata"

	obs_stage_log_relpath = str(
		observability_cfg.get("stage_log_relpath", "logs/preprocess_pipeline.log")
		or "logs/preprocess_pipeline.log"
	).strip()
	if not obs_stage_log_relpath:
		obs_stage_log_relpath = "logs/preprocess_pipeline.log"

	obs_save_run_manifest_default = obs_enabled
	obs_save_event_timeline_default = observability_mode == "detailed"
	obs_save_environment_default = observability_mode == "detailed"
	obs_save_artifact_inventory_default = obs_enabled
	obs_save_stage_log_default = observability_mode == "detailed"

	observability_save_run_manifest = obs_enabled and _as_bool(
		observability_cfg.get("save_run_manifest", obs_save_run_manifest_default),
		obs_save_run_manifest_default,
	)
	observability_save_event_timeline = obs_enabled and _as_bool(
		observability_cfg.get("save_event_timeline", obs_save_event_timeline_default),
		obs_save_event_timeline_default,
	)
	observability_save_environment = obs_enabled and _as_bool(
		observability_cfg.get("save_environment", obs_save_environment_default),
		obs_save_environment_default,
	)
	observability_save_artifact_inventory = obs_enabled and _as_bool(
		observability_cfg.get("save_artifact_inventory", obs_save_artifact_inventory_default),
		obs_save_artifact_inventory_default,
	)
	observability_save_stage_log = obs_enabled and _as_bool(
		observability_cfg.get("save_stage_log", obs_save_stage_log_default),
		obs_save_stage_log_default,
	)

	if not obs_enabled:
		observability_mode = "off"

	assay_stats_relpath = str(plot_cfg.get("assay_stats_relpath", "assay_stats_{stream_id}.txt") or "assay_stats_{stream_id}.txt")
	channel_layouts_subdir = str(plot_cfg.get("channel_layouts_subdir", "channel_layouts") or "channel_layouts")
	segment_traces_subdir = str(plot_cfg.get("segment_traces_subdir", "segment_traces") or "segment_traces")
	concat_trace_relpath = str(
		plot_cfg.get("concat_trace_relpath", "concat_cluster_reps_{stream_id}.png")
		or "concat_cluster_reps_{stream_id}.png"
	)

	save_recording = _as_bool(execution_cfg.get("save_recording", True), True)
	save_concat_recording = _as_bool(outputs_cfg.get("save_concat_recording", save_recording), save_recording)
	save_segment_recordings = _as_bool(outputs_cfg.get("save_segment_recordings", save_recording), save_recording)
	save_chunk_duration = _as_optional_str(outputs_cfg.get("save_chunk_duration", "1s")) or "1s"
	save_progress_bar = _as_bool(outputs_cfg.get("save_progress_bar", False), False)
	concat_save_n_jobs = _as_optional_int(outputs_cfg.get("concat_save_n_jobs", None))
	segment_save_n_jobs = _as_optional_int(outputs_cfg.get("segment_save_n_jobs", None))
	print_n_jobs_used = _as_bool(outputs_cfg.get("print_n_jobs_used", False), False)
	legacy_segment_phase_cfg = phases_cfg.get("save_segment_recordings", {}) if isinstance(phases_cfg.get("save_segment_recordings", {}), dict) else {}
	legacy_concat_phase_cfg = phases_cfg.get("save_concatenated_recording", {}) if isinstance(phases_cfg.get("save_concatenated_recording", {}), dict) else {}
	legacy_common_phase_cfg = phases_cfg.get("save_common_electrodes", {}) if isinstance(phases_cfg.get("save_common_electrodes", {}), dict) else {}

	legacy_segment_phase = _parse_simple_phase_config(
		raw_cfg=legacy_segment_phase_cfg,
		default_enabled=save_segment_recordings,
		default_summary_json_relpath="context/segment_recordings_summary.json",
	)
	legacy_concat_phase = _parse_simple_phase_config(
		raw_cfg=legacy_concat_phase_cfg,
		default_enabled=save_concat_recording,
		default_summary_json_relpath="context/concat_segments_summary.json",
	)
	legacy_common_phase = _parse_simple_phase_config(
		raw_cfg=legacy_common_phase_cfg,
		default_enabled=save_recording or save_concat_recording or save_segment_recordings,
		default_summary_json_relpath="context/save_common_electrodes_summary.json",
	)

	copy_phase_cfg = phases_cfg.get("copy_src_to_scratch", {}) if isinstance(phases_cfg.get("copy_src_to_scratch", {}), dict) else {}
	save_rec_metadata_phase_cfg = phases_cfg.get("save_rec_metadata", {}) if isinstance(phases_cfg.get("save_rec_metadata", {}), dict) else {}
	wipe_src_scratch_phase_cfg = phases_cfg.get("wipe_src_scratch", {}) if isinstance(phases_cfg.get("wipe_src_scratch", {}), dict) else {}
	preprocess_segments_phase_cfg = phases_cfg.get("preprocess_segments", {}) if isinstance(phases_cfg.get("preprocess_segments", {}), dict) else {}
	plot_segment_traces_phase_cfg = phases_cfg.get("plot_segment_traces", {}) if isinstance(phases_cfg.get("plot_segment_traces", {}), dict) else {}
	concatenate_phase_cfg_raw = phases_cfg.get("concat_segments", None)
	if not isinstance(concatenate_phase_cfg_raw, dict):
		concatenate_phase_cfg_raw = phases_cfg.get("concatenate_recordings", None)
	if not isinstance(concatenate_phase_cfg_raw, dict):
		concatenate_phase_cfg_raw = phases_cfg.get("concatenate_preprocessed_recordings", {})
	concatenate_phase_cfg = concatenate_phase_cfg_raw if isinstance(concatenate_phase_cfg_raw, dict) else {}
	plot_concat_traces_phase_cfg = phases_cfg.get("plot_concat_traces", {}) if isinstance(phases_cfg.get("plot_concat_traces", {}), dict) else {}
	concatenate_save_common_cfg = legacy_common_phase_cfg
	if isinstance(concatenate_phase_cfg.get("save_common_electrodes", {}), dict):
		concatenate_save_common_cfg = dict(concatenate_phase_cfg.get("save_common_electrodes", {}))
	if concatenate_save_common_cfg:
		save_rec_metadata_phase_cfg = dict(save_rec_metadata_phase_cfg)
		save_rec_metadata_phase_cfg.setdefault(
			"common_electrodes_summary_json_relpath",
			concatenate_save_common_cfg.get("summary_json_relpath", "context/save_common_electrodes_summary.json"),
		)

	segment_plot_defaults = PreprocessPlotConfig(
		disable_all_png_diagnostics=raw_disable_all_png_diagnostics,
		layouts=bool(plot_layouts_effective),
		concat_trace=(bool(plot_concat_trace_effective) if not using_new_phase_schema else False),
		segment_traces=bool(plot_segment_traces_effective),
		output_dir=_normalize_plot_output_dir(plot_cfg.get("output_dir", None)),
		epoch_markers_output_dir=_normalize_plot_output_dir(plot_cfg.get("epoch_markers_output_dir", None)),
		assay_stats_relpath=assay_stats_relpath,
		channel_layouts_subdir=channel_layouts_subdir,
		segment_traces_subdir=segment_traces_subdir,
		concat_trace_relpath=concat_trace_relpath,
		n_representative_channels=n_representative_channels,
		concat_trace_n_reps=concat_trace_n_reps,
		segment_trace_n_reps=segment_trace_n_reps,
		n_jobs=plot_n_jobs,
		trace_downsample_hz=trace_downsample_hz,
		trace_max_points=trace_max_points,
	)
	segment_outputs_defaults = PreprocessPhaseOutputsConfig(
		save_chunk_duration=save_chunk_duration,
		save_progress_bar=save_progress_bar,
		segment_save_n_jobs=segment_save_n_jobs,
		print_n_jobs_used=print_n_jobs_used,
	)

	preprocess_segments_phase = PreprocessSegmentsPhaseConfig(
		enabled=_as_bool(
			preprocess_segments_phase_cfg.get(
				"enabled",
				preprocess_segments_phase_cfg.get(
					"enable",
					legacy_segment_phase.enabled,
				),
			),
			legacy_segment_phase.enabled,
		),
		summary_json_relpath=str(
			preprocess_segments_phase_cfg.get(
				"summary_json_relpath",
				legacy_segment_phase.summary_json_relpath,
			)
			or legacy_segment_phase.summary_json_relpath
		),
		rel_output_root=str(
			preprocess_segments_phase_cfg.get("rel_output_root", "preprocessed_segments")
			or "preprocessed_segments"
		),
		outputs=_parse_phase_outputs_config(
			raw_cfg=preprocess_segments_phase_cfg.get("outputs", {}),
			defaults=segment_outputs_defaults,
		),
	)
	plot_segment_traces_phase_raw = plot_segment_traces_phase_cfg
	if not plot_segment_traces_phase_raw and isinstance(preprocess_segments_phase_cfg.get("plot", {}), dict):
		plot_segment_traces_phase_raw = {
			"enabled": True,
			"plot": dict(preprocess_segments_phase_cfg.get("plot", {})),
		}
	plot_segment_traces_phase = PreprocessPlotSegmentTracesPhaseConfig(
		enabled=_as_bool(
			plot_segment_traces_phase_raw.get("enabled", preprocess_segments_phase.enabled),
			preprocess_segments_phase.enabled,
		),
		summary_json_relpath=str(
			plot_segment_traces_phase_raw.get("summary_json_relpath", "context/plot_segment_traces_summary.json")
			or "context/plot_segment_traces_summary.json"
		),
		plot=_parse_plot_phase_config(
			raw_cfg=plot_segment_traces_phase_raw.get("plot", {}),
			defaults=segment_plot_defaults,
		),
	)
	concat_plot_defaults = PreprocessPlotConfig(
		disable_all_png_diagnostics=plot_segment_traces_phase.plot.disable_all_png_diagnostics,
		layouts=(plot_segment_traces_phase.plot.layouts if not using_new_phase_schema else False),
		concat_trace=bool(plot_concat_trace_effective),
		segment_traces=(plot_segment_traces_phase.plot.segment_traces if not using_new_phase_schema else False),
		output_dir=plot_segment_traces_phase.plot.output_dir,
		epoch_markers_output_dir=plot_segment_traces_phase.plot.epoch_markers_output_dir,
		assay_stats_relpath=plot_segment_traces_phase.plot.assay_stats_relpath,
		channel_layouts_subdir=plot_segment_traces_phase.plot.channel_layouts_subdir,
		segment_traces_subdir=plot_segment_traces_phase.plot.segment_traces_subdir,
		concat_trace_relpath=plot_segment_traces_phase.plot.concat_trace_relpath,
		n_representative_channels=plot_segment_traces_phase.plot.n_representative_channels,
		concat_trace_n_reps=concat_trace_n_reps,
		segment_trace_n_reps=plot_segment_traces_phase.plot.segment_trace_n_reps,
		n_jobs=plot_segment_traces_phase.plot.n_jobs,
		trace_downsample_hz=plot_segment_traces_phase.plot.trace_downsample_hz,
		trace_max_points=plot_segment_traces_phase.plot.trace_max_points,
	)
	concat_outputs_defaults = PreprocessPhaseOutputsConfig(
		save_chunk_duration=preprocess_segments_phase.outputs.save_chunk_duration,
		save_progress_bar=preprocess_segments_phase.outputs.save_progress_bar,
		concat_save_n_jobs=concat_save_n_jobs,
		print_n_jobs_used=preprocess_segments_phase.outputs.print_n_jobs_used,
	)
	concat_segments_phase = PreprocessConcatSegmentsPhaseConfig(
		enabled=_as_bool(
			concatenate_phase_cfg.get(
				"enabled",
				concatenate_phase_cfg.get("enable", legacy_concat_phase.enabled),
			),
			legacy_concat_phase.enabled,
		),
		concatenate_preprocessed_recordings=_as_bool(
			concatenate_phase_cfg.get("concatenate_preprocessed_recordings", True),
			True,
		),
		summary_json_relpath=str(
			concatenate_phase_cfg.get(
				"summary_json_relpath",
				legacy_concat_phase.summary_json_relpath,
			)
			or legacy_concat_phase.summary_json_relpath
		),
		rel_output_root=str(
			concatenate_phase_cfg.get("rel_output_root", "concatenated_recording")
			or "concatenated_recording"
		),
		manifest_relpath=str(
			concatenate_phase_cfg.get("manifest_relpath", "context/concat_segments_manifest.json")
			or "context/concat_segments_manifest.json"
		),
		outputs=_parse_phase_outputs_config(
			raw_cfg=concatenate_phase_cfg.get("outputs", {}),
			defaults=concat_outputs_defaults,
		),
	)
	plot_concat_traces_phase_raw = plot_concat_traces_phase_cfg
	if not plot_concat_traces_phase_raw and isinstance(concatenate_phase_cfg.get("plot", {}), dict):
		plot_concat_traces_phase_raw = {
			"enabled": True,
			"plot": dict(concatenate_phase_cfg.get("plot", {})),
		}
	plot_concat_traces_phase = PreprocessPlotConcatTracesPhaseConfig(
		enabled=_as_bool(
			plot_concat_traces_phase_raw.get("enabled", concat_segments_phase.enabled),
			concat_segments_phase.enabled,
		),
		summary_json_relpath=str(
			plot_concat_traces_phase_raw.get("summary_json_relpath", "context/plot_concat_traces_summary.json")
			or "context/plot_concat_traces_summary.json"
		),
		plot=_parse_plot_phase_config(
			raw_cfg=plot_concat_traces_phase_raw.get("plot", {}),
			defaults=concat_plot_defaults,
		),
	)
	copy_src_to_scratch_phase = PreprocessCopySrcToScratchPhaseConfig(
		enabled=_as_bool(copy_phase_cfg.get("enabled", copy_phase_cfg.get("enable", False)), False),
		requires_use_scratch_root=_as_bool(
			copy_phase_cfg.get("requires_use_scratch_root", False),
			False,
		),
		summary_json_relpath=str(
			copy_phase_cfg.get("summary_json_relpath", "context/copy_src_to_scratch_summary.json")
			or "context/copy_src_to_scratch_summary.json"
		),
	)
	save_rec_metadata_phase = _parse_save_rec_metadata_phase_config(
		raw_cfg=save_rec_metadata_phase_cfg,
	)
	wipe_src_scratch_phase = PreprocessWipeSrcScratchPhaseConfig(
		enabled=_as_bool(wipe_src_scratch_phase_cfg.get("enabled", wipe_src_scratch_phase_cfg.get("enable", False)), False),
		dry_run=_as_bool(wipe_src_scratch_phase_cfg.get("dry_run", False), False),
		requires_use_scratch_root=_as_bool(
			wipe_src_scratch_phase_cfg.get("requires_use_scratch_root", False),
			False,
		),
		summary_json_relpath=str(
			wipe_src_scratch_phase_cfg.get("summary_json_relpath", "context/wipe_src_scratch_summary.json")
			or "context/wipe_src_scratch_summary.json"
		),
	)

	return PreprocessStageConfig(
		output_rel_root=_normalize_output_rel_root(outputs_cfg.get("output_rel_root", _DEFAULT_OUTPUT_REL_ROOT)),
		force_restart=force_restart,
		force_replot=force_replot,
		debug_limit_wells=_as_optional_int(debug_cfg.get("limit_wells", None)),
		debug_limit_segments_per_well=_as_optional_int(debug_cfg.get("limit_segments_per_well", None)),
		logging_enabled=logging_enabled,
		logging_verbose=logging_verbose,
		logging_file_relpath=logging_file_relpath,
		logging_suppress_h5_plugin_messages=logging_suppress_h5_plugin_messages,
		logging_phase_dividers=logging_phase_dividers,
		enable_checkpointing=_as_bool(execution_cfg.get("enable_checkpointing", True), True),
		n_jobs=_as_optional_int(execution_cfg.get("n_jobs", None)),
		plot_layouts=plot_layouts_effective,
		plot_concat_trace=plot_concat_trace_effective,
		plot_segment_traces=plot_segment_traces_effective,
		plot_output_dir=_normalize_plot_output_dir(plot_cfg.get("output_dir", None)),
		epoch_markers_output_dir=_normalize_plot_output_dir(plot_cfg.get("epoch_markers_output_dir", None)),
		assay_stats_relpath=assay_stats_relpath,
		channel_layouts_subdir=channel_layouts_subdir,
		segment_traces_subdir=segment_traces_subdir,
		concat_trace_relpath=concat_trace_relpath,
		n_representative_channels=n_representative_channels,
		concat_trace_n_reps=concat_trace_n_reps,
		segment_trace_n_reps=segment_trace_n_reps,
		plot_n_jobs=plot_n_jobs,
		trace_downsample_hz=trace_downsample_hz,
		trace_max_points=trace_max_points,
		observability_mode=observability_mode,
		observability_output_subdir=obs_output_subdir,
		observability_save_run_manifest=observability_save_run_manifest,
		observability_save_event_timeline=observability_save_event_timeline,
		observability_save_environment=observability_save_environment,
		observability_save_artifact_inventory=observability_save_artifact_inventory,
		observability_save_stage_log=observability_save_stage_log,
		observability_stage_log_relpath=obs_stage_log_relpath,
		temporal_resample_factor=_as_optional_int(execution_cfg.get("temporal_resample_factor", None)),
		temporal_resample_rate_hz=_as_optional_int(execution_cfg.get("temporal_resample_rate_hz", None)),
		temporal_resample_margin_ms=float(_as_optional_float(execution_cfg.get("temporal_resample_margin_ms", None)) or 100.0),
		temporal_resample_dtype=_as_optional_str(execution_cfg.get("temporal_resample_dtype", None)),
		save_recording=save_recording,
		overwrite_saved_recording=_as_bool(execution_cfg.get("overwrite_saved_recording", True), True),
		save_concat_recording=save_concat_recording,
		save_segment_recordings=save_segment_recordings,
		save_chunk_duration=save_chunk_duration,
		save_progress_bar=save_progress_bar,
		concat_save_n_jobs=concat_save_n_jobs,
		segment_save_n_jobs=segment_save_n_jobs,
		print_n_jobs_used=print_n_jobs_used,
		phases=PreprocessPhasesConfig(
			copy_src_to_scratch=copy_src_to_scratch_phase,
			save_rec_metadata=save_rec_metadata_phase,
			wipe_src_scratch=wipe_src_scratch_phase,
			preprocess_segments=preprocess_segments_phase,
			plot_segment_traces=plot_segment_traces_phase,
			concat_segments=concat_segments_phase,
			plot_concat_traces=plot_concat_traces_phase,
		),
	)


def build_preprocess_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: PreprocessStageConfig,
	unit_workers: int,
) -> PreprocessInputs:
	n_jobs = stage_config.n_jobs if stage_config.n_jobs is not None else max(1, int(unit_workers))
	plot_n_jobs = stage_config.plot_n_jobs if stage_config.plot_n_jobs is not None else int(n_jobs)
	source_h5_path = target.source_h5_path or target.h5_path
	try:
		copied_to_scratch = Path(source_h5_path).expanduser().resolve() != Path(target.h5_path).expanduser().resolve()
	except Exception:
		copied_to_scratch = Path(source_h5_path) != Path(target.h5_path)
	return PreprocessInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		final_output_root=(target.final_output_root or target.mea_output_root),
		source_h5_path=source_h5_path,
		copied_to_scratch=bool(copied_to_scratch),
		output_rel_root=stage_config.output_rel_root,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		debug_limit_segments_per_well=stage_config.debug_limit_segments_per_well,
		logging_enabled=stage_config.logging_enabled,
		logging_verbose=stage_config.logging_verbose,
		logging_file_relpath=stage_config.logging_file_relpath,
		logging_suppress_h5_plugin_messages=stage_config.logging_suppress_h5_plugin_messages,
		logging_phase_dividers=stage_config.logging_phase_dividers,
		enable_checkpointing=stage_config.enable_checkpointing,
		n_jobs=max(1, int(n_jobs)),
		plot_layouts=stage_config.plot_layouts,
		plot_concat_trace=stage_config.plot_concat_trace,
		plot_segment_traces=stage_config.plot_segment_traces,
		plot_output_dir=stage_config.plot_output_dir,
		epoch_markers_output_dir=stage_config.epoch_markers_output_dir,
		assay_stats_relpath=stage_config.assay_stats_relpath,
		channel_layouts_subdir=stage_config.channel_layouts_subdir,
		segment_traces_subdir=stage_config.segment_traces_subdir,
		concat_trace_relpath=stage_config.concat_trace_relpath,
		n_representative_channels=stage_config.n_representative_channels,
		concat_trace_n_reps=stage_config.concat_trace_n_reps,
		segment_trace_n_reps=stage_config.segment_trace_n_reps,
		plot_n_jobs=max(1, int(plot_n_jobs)),
		trace_downsample_hz=stage_config.trace_downsample_hz,
		trace_max_points=stage_config.trace_max_points,
		observability_mode=stage_config.observability_mode,
		observability_output_subdir=stage_config.observability_output_subdir,
		observability_save_run_manifest=stage_config.observability_save_run_manifest,
		observability_save_event_timeline=stage_config.observability_save_event_timeline,
		observability_save_environment=stage_config.observability_save_environment,
		observability_save_artifact_inventory=stage_config.observability_save_artifact_inventory,
		observability_save_stage_log=stage_config.observability_save_stage_log,
		observability_stage_log_relpath=stage_config.observability_stage_log_relpath,
		temporal_resample_factor=stage_config.temporal_resample_factor,
		temporal_resample_rate_hz=stage_config.temporal_resample_rate_hz,
		temporal_resample_margin_ms=stage_config.temporal_resample_margin_ms,
		temporal_resample_dtype=stage_config.temporal_resample_dtype,
		save_recording=stage_config.save_recording,
		overwrite_saved_recording=stage_config.overwrite_saved_recording,
		save_concat_recording=stage_config.save_concat_recording,
		save_segment_recordings=stage_config.save_segment_recordings,
		save_chunk_duration=stage_config.save_chunk_duration,
		save_progress_bar=stage_config.save_progress_bar,
		concat_save_n_jobs=stage_config.concat_save_n_jobs,
		segment_save_n_jobs=stage_config.segment_save_n_jobs,
		print_n_jobs_used=stage_config.print_n_jobs_used,
		phases=stage_config.phases,
	)


def load_preprocess_inputs_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> PreprocessInputs:
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

	stage_cfg = parse_preprocess_stage_config(
		runtime_config=runtime_cfg,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	n_jobs = stage_cfg.n_jobs if stage_cfg.n_jobs is not None else 1
	plot_n_jobs = stage_cfg.plot_n_jobs if stage_cfg.plot_n_jobs is not None else int(n_jobs)
	return PreprocessInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		final_output_root=output_root,
		source_h5_path=h5_path,
		copied_to_scratch=False,
		output_rel_root=stage_cfg.output_rel_root,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		debug_limit_segments_per_well=stage_cfg.debug_limit_segments_per_well,
		logging_enabled=stage_cfg.logging_enabled,
		logging_verbose=stage_cfg.logging_verbose,
		logging_file_relpath=stage_cfg.logging_file_relpath,
		logging_suppress_h5_plugin_messages=stage_cfg.logging_suppress_h5_plugin_messages,
		logging_phase_dividers=stage_cfg.logging_phase_dividers,
		enable_checkpointing=stage_cfg.enable_checkpointing,
		n_jobs=max(1, int(n_jobs)),
		plot_layouts=stage_cfg.plot_layouts,
		plot_concat_trace=stage_cfg.plot_concat_trace,
		plot_segment_traces=stage_cfg.plot_segment_traces,
		plot_output_dir=stage_cfg.plot_output_dir,
		epoch_markers_output_dir=stage_cfg.epoch_markers_output_dir,
		assay_stats_relpath=stage_cfg.assay_stats_relpath,
		channel_layouts_subdir=stage_cfg.channel_layouts_subdir,
		segment_traces_subdir=stage_cfg.segment_traces_subdir,
		concat_trace_relpath=stage_cfg.concat_trace_relpath,
		n_representative_channels=stage_cfg.n_representative_channels,
		concat_trace_n_reps=stage_cfg.concat_trace_n_reps,
		segment_trace_n_reps=stage_cfg.segment_trace_n_reps,
		plot_n_jobs=max(1, int(plot_n_jobs)),
		trace_downsample_hz=stage_cfg.trace_downsample_hz,
		trace_max_points=stage_cfg.trace_max_points,
		observability_mode=stage_cfg.observability_mode,
		observability_output_subdir=stage_cfg.observability_output_subdir,
		observability_save_run_manifest=stage_cfg.observability_save_run_manifest,
		observability_save_event_timeline=stage_cfg.observability_save_event_timeline,
		observability_save_environment=stage_cfg.observability_save_environment,
		observability_save_artifact_inventory=stage_cfg.observability_save_artifact_inventory,
		observability_save_stage_log=stage_cfg.observability_save_stage_log,
		observability_stage_log_relpath=stage_cfg.observability_stage_log_relpath,
		temporal_resample_factor=stage_cfg.temporal_resample_factor,
		temporal_resample_rate_hz=stage_cfg.temporal_resample_rate_hz,
		temporal_resample_margin_ms=stage_cfg.temporal_resample_margin_ms,
		temporal_resample_dtype=stage_cfg.temporal_resample_dtype,
		save_recording=stage_cfg.save_recording,
		overwrite_saved_recording=stage_cfg.overwrite_saved_recording,
		save_concat_recording=stage_cfg.save_concat_recording,
		save_segment_recordings=stage_cfg.save_segment_recordings,
		save_chunk_duration=stage_cfg.save_chunk_duration,
		save_progress_bar=stage_cfg.save_progress_bar,
		concat_save_n_jobs=stage_cfg.concat_save_n_jobs,
		segment_save_n_jobs=stage_cfg.segment_save_n_jobs,
		print_n_jobs_used=stage_cfg.print_n_jobs_used,
		phases=stage_cfg.phases,
	)
