from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from ...execution.context import ExecutionTarget
from .models.inputs import SpikesortInputs


_DEFAULT_OUTPUT_REL_ROOT = "spikesort_outputs"
_LEGACY_OUTPUT_REL_ROOT = "stg2_spikesorting_outputs"


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
	logging_enabled: bool
	logging_verbose: bool
	logging_file_relpath: str | None
	debug_limit_wells: int | None
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
	slay_params: dict[str, Any] | None

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
			execution_cfg.get("sort_delete_outputs_on_force_restart", None),
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
	slay_params = _as_optional_dict(slay_cfg.get("params", None))

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
		logging_enabled=logging_enabled,
		logging_verbose=logging_verbose,
		logging_file_relpath=logging_file_relpath,
		debug_limit_wells=debug_limit_wells,
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
		slay_params=(dict(slay_params) if isinstance(slay_params, dict) else None),
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
		um_kwargs=stage_config.um_kwargs,
		am_kwargs=stage_config.am_kwargs,
		option_kwargs=stage_config.option_kwargs,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		resume_from=stage_config.resume_from,
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
		um_kwargs=stage_cfg.um_kwargs,
		am_kwargs=stage_cfg.am_kwargs,
		option_kwargs=stage_cfg.option_kwargs,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		resume_from=stage_cfg.resume_from,
	)
