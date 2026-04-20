from __future__ import annotations

import contextlib
import datetime as dt
import getpass
import io
import json
import logging
import os
import platform
import shutil
import socket
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_reconstructor.pipeline.pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from axon_reconstructor.pipeline.stg1_preprocessing.constants import PREPROCESS_OUTPUTS_DIRNAME
from axon_reconstructor.pipeline.stg1_preprocessing.planning import build_preprocess_plan

from .core import (
	run_build_preprocessed_recording_core,
	run_save_common_electrodes_core,
	run_save_concatenated_recording_core,
	run_save_segment_recordings_core,
)
from .models.inputs import (
	PreprocessConcatenatePreprocessedRecordingsPhaseConfig,
	PreprocessInputs,
	PreprocessPlotConfig,
	PreprocessSegmentsPhaseConfig,
)
from .models.results import PreprocessResult


LOGGER = logging.getLogger("axon_recon.preprocess")


_SCRATCH_INPUT_USAGE_LOCK = threading.Lock()
_SCRATCH_INPUT_ACTIVE_COUNTS: dict[str, int] = {}


@dataclass(frozen=True)
class _PreprocessPathSet:
	well_out_dir: Path
	preprocess_out_dir: Path
	legacy_out_dir: Path
	recording_dir: Path
	common_electrodes_path: Path
	per_segment_preprocessed_dir: Path
	per_segment_manifest_path: Path
	preprocess_config_path: Path
	stage_summary_json: Path
	stage_log_source: Path
	plot_output_dir: Path | None
	epoch_markers_output_dir: Path | None


def _utc_now_iso() -> str:
	return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalize_observability_mode(raw: str | None) -> str:
	token = str(raw or "off").strip().lower()
	if token in {"off", "none", "disabled", "false", "0"}:
		return "off"
	if token in {"basic", "standard", "on", "enabled", "true", "1"}:
		return "basic"
	if token in {"detailed", "verbose", "debug", "full", "meta"}:
		return "detailed"
	return "off"


def _is_observability_enabled(inputs: PreprocessInputs) -> bool:
	mode = _normalize_observability_mode(inputs.observability_mode)
	return bool(
		mode != "off"
		or inputs.observability_save_run_manifest
		or inputs.observability_save_event_timeline
		or inputs.observability_save_environment
		or inputs.observability_save_artifact_inventory
		or inputs.observability_save_stage_log
	)


def _json_ready(value: Any) -> Any:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(k): _json_ready(v) for k, v in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_json_ready(v) for v in value]
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	return str(value)


def _normalize_trace_max_points(raw: int | float | str | None) -> int:
	try:
		parsed = int(raw) if raw is not None else 150000
	except Exception:
		parsed = 150000
	return (-1 if parsed <= 0 else max(1000, parsed))


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path_from_root(*, root_dir: Path, raw: str | None, default: str) -> Path:
	token = str(raw).strip() if raw is not None else ""
	if not token:
		token = str(default)
	path = Path(token).expanduser()
	if path.is_absolute():
		return Path(os.path.abspath(str(path)))
	return Path(os.path.abspath(str(root_dir / path)))


def _safe_path_exists(path: Path) -> bool:
	try:
		return bool(path.exists())
	except Exception:
		return False


def _paths_equal_no_resolve(*, left: Path, right: Path) -> bool:
	try:
		left_abs = Path(os.path.abspath(str(left.expanduser())))
		right_abs = Path(os.path.abspath(str(right.expanduser())))
		return left_abs == right_abs
	except Exception:
		return str(left) == str(right)


def _is_self_referential_symlink(path: Path) -> bool:
	try:
		if not path.is_symlink():
			return False
		target_raw = os.readlink(path)
	except Exception:
		return False

	target_path = Path(target_raw).expanduser()
	if not target_path.is_absolute():
		target_path = path.parent / target_path
	return _paths_equal_no_resolve(left=path, right=target_path)


def _clear_self_referential_symlink(path: Path) -> bool:
	if not _is_self_referential_symlink(path):
		return False
	try:
		path.unlink()
		LOGGER.warning("Removed self-referential symlink: %s", path)
		return True
	except Exception:
		LOGGER.warning("Failed removing self-referential symlink: %s", path, exc_info=True)
		return False


def _resolve_stage_log_source(*, inputs: PreprocessInputs, well_out_dir: Path) -> Path:
	if inputs.logging_file_relpath is not None and str(inputs.logging_file_relpath).strip():
		return _resolve_path_from_root(
			root_dir=well_out_dir,
			raw=inputs.logging_file_relpath,
			default="logs/preprocess_pipeline.log",
		)
	return compute_pipeline_log_file(
		well_out_dir=well_out_dir,
		data_file=inputs.h5_path,
		stream_id=inputs.stream_id,
	)


def _collect_path_details(path: Path) -> dict[str, Any]:
	info: dict[str, Any] = {
		"path": str(path),
		"exists": bool(path.exists()),
	}
	if not path.exists():
		return info

	try:
		info["is_file"] = bool(path.is_file())
		info["is_dir"] = bool(path.is_dir())
		stat = path.stat()
		info["size_bytes"] = int(stat.st_size)
		info["mtime_utc"] = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).isoformat()
		if path.is_dir():
			try:
				info["n_entries"] = int(sum(1 for _ in path.iterdir()))
			except Exception:
				pass
	except Exception:
		pass
	return info


def _build_artifact_inventory(*, outputs: dict[str, str]) -> dict[str, dict[str, Any]]:
	inventory: dict[str, dict[str, Any]] = {}
	for key, raw_path in outputs.items():
		try:
			p = Path(str(raw_path)).expanduser().resolve()
		except Exception:
			p = Path(str(raw_path))
		inventory[str(key)] = _collect_path_details(p)
	return inventory


def _capture_stage_log(*, source: Path, destination: Path) -> str | None:
	_clear_self_referential_symlink(source)
	_clear_self_referential_symlink(destination)

	if _paths_equal_no_resolve(left=source, right=destination):
		LOGGER.info("Skipping stage log capture because source and destination are identical: %s", source)
		return None

	if not _safe_path_exists(source):
		return None
	destination.parent.mkdir(parents=True, exist_ok=True)
	if _safe_path_exists(destination) or destination.is_symlink():
		if destination.is_dir() and not destination.is_symlink():
			shutil.rmtree(destination)
		else:
			destination.unlink()
	try:
		destination.symlink_to(source)
		return "symlink"
	except Exception:
		shutil.copy2(source, destination)
		return "copy"


def _write_observability_artifacts(
	*,
	inputs: PreprocessInputs,
	well_out_dir: Path,
	preprocess_out_dir: Path,
	legacy_out_dir: Path,
	outputs: dict[str, str],
	summary_json: Path | None,
	event_records: list[dict[str, Any]],
	common_electrodes: list[int],
	stage_status: str,
	stage_started_utc: str,
	stage_elapsed_s: float,
	stage_error: Exception | None,
	stage_log_source: Path | None,
) -> dict[str, str]:
	if not _is_observability_enabled(inputs):
		return {}

	mode = _normalize_observability_mode(inputs.observability_mode)
	obs_root = _resolve_path_from_root(
		root_dir=preprocess_out_dir,
		raw=inputs.observability_output_subdir,
		default="run_metadata",
	)
	obs_root.mkdir(parents=True, exist_ok=True)

	extra_outputs: dict[str, str] = {
		"observability.dir": str(obs_root),
	}

	if stage_log_source is not None and _safe_path_exists(stage_log_source):
		extra_outputs["pipeline_log"] = str(stage_log_source)
		if bool(inputs.observability_save_stage_log):
			captured_log_path = _resolve_path_from_root(
				root_dir=obs_root,
				raw=inputs.observability_stage_log_relpath,
				default="logs/preprocess_pipeline.log",
			)
			capture_mode = _capture_stage_log(source=stage_log_source, destination=captured_log_path)
			if capture_mode is not None:
				extra_outputs["observability.stage_log"] = str(captured_log_path)
				extra_outputs["observability.stage_log_capture_mode"] = str(capture_mode)

	environment_payload: dict[str, Any] | None = None
	if bool(inputs.observability_save_environment):
		environment_payload = {
			"generated_utc": _utc_now_iso(),
			"hostname": socket.gethostname(),
			"user": getpass.getuser(),
			"pid": int(os.getpid()),
			"cwd": os.getcwd(),
			"python_version": platform.python_version(),
			"python_executable": sys.executable,
			"platform": platform.platform(),
			"conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
			"env": {
				"AXON_RECON_PYTHON": os.environ.get("AXON_RECON_PYTHON"),
				"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
				"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
			},
		}
		environment_path = obs_root / "environment.json"
		_write_json(environment_path, _json_ready(environment_payload))
		extra_outputs["observability.environment_json"] = str(environment_path)

	if bool(inputs.observability_save_artifact_inventory):
		inventory_payload_outputs = dict(outputs)
		inventory_payload_outputs.update(extra_outputs)
		if summary_json is not None:
			inventory_payload_outputs["preprocess_summary_json"] = str(summary_json)
		artifact_inventory_path = obs_root / "artifact_inventory.json"
		_write_json(
			artifact_inventory_path,
			{
				"generated_utc": _utc_now_iso(),
				"items": _build_artifact_inventory(outputs=inventory_payload_outputs),
			},
		)
		extra_outputs["observability.artifact_inventory_json"] = str(artifact_inventory_path)

	if bool(inputs.observability_save_event_timeline):
		event_timeline_path = obs_root / "event_timeline.jsonl"
		event_lines = [json.dumps(_json_ready(ev), sort_keys=True) for ev in event_records]
		event_timeline_path.write_text(
			"\n".join(event_lines) + ("\n" if event_lines else ""),
			encoding="utf-8",
		)
		extra_outputs["observability.event_timeline_jsonl"] = str(event_timeline_path)

	if bool(inputs.observability_save_run_manifest):
		run_manifest_path = obs_root / "run_manifest.json"
		extra_outputs["observability.run_manifest_json"] = str(run_manifest_path)

		common_preview_limit = 256 if mode == "detailed" else 64
		common_preview = [int(x) for x in list(common_electrodes)[:common_preview_limit]]

		error_payload: dict[str, Any] | None = None
		if stage_error is not None:
			error_payload = {
				"type": type(stage_error).__name__,
				"message": str(stage_error),
				"traceback": traceback.format_exception(type(stage_error), stage_error, stage_error.__traceback__),
			}

		manifest_outputs = dict(outputs)
		manifest_outputs.update(extra_outputs)
		if summary_json is not None:
			manifest_outputs["preprocess_summary_json"] = str(summary_json)

		manifest_payload = {
			"schema_version": 1,
			"generated_utc": _utc_now_iso(),
			"stage": "preprocess",
			"status": str(stage_status),
			"mode": mode,
			"timestamps": {
				"started_utc": str(stage_started_utc),
				"ended_utc": _utc_now_iso(),
				"duration_seconds": float(max(0.0, stage_elapsed_s)),
			},
			"target": {
				"h5_path": str(inputs.h5_path),
				"stream_id": str(inputs.stream_id),
				"well_out_dir": str(well_out_dir),
			},
			"paths": {
				"preprocess_out_dir": str(preprocess_out_dir),
				"legacy_preprocess_out_dir": str(legacy_out_dir),
			},
			"inputs": _json_ready(asdict(inputs)),
			"common_electrodes": {
				"count": int(len(common_electrodes)),
				"preview": common_preview,
				"preview_truncated": bool(len(common_electrodes) > len(common_preview)),
			},
			"event_count": int(len(event_records)),
			"outputs": manifest_outputs,
		}
		if environment_payload is not None:
			manifest_payload["runtime_environment"] = _json_ready(environment_payload)
		if error_payload is not None:
			manifest_payload["error"] = _json_ready(error_payload)
		_write_json(run_manifest_path, _json_ready(manifest_payload))

	return extra_outputs


def _prepare_phase_logger(inputs: PreprocessInputs, paths: _PreprocessPathSet) -> logging.Logger | None:
	if not bool(inputs.logging_enabled):
		return None
	try:
		return setup_pipeline_logger(
			log_file=paths.stage_log_source,
			logger_name=f"axon_reconstructor.{paths.stage_log_source.stem}",
			verbose=bool(inputs.logging_verbose),
		)
	except Exception:
		return LOGGER


def _coalesce_config_value(*values: Any) -> Any:
	for value in values:
		if value is None:
			continue
		if isinstance(value, str) and not str(value).strip():
			continue
		return value
	return None


def _apply_plot_input_fallback(
	*,
	phase_plot: PreprocessPlotConfig,
	input_plot: PreprocessPlotConfig,
	default_plot: PreprocessPlotConfig,
) -> PreprocessPlotConfig:
	def _prefer_input(phase_value: Any, input_value: Any, default_value: Any) -> Any:
		if phase_value == default_value and input_value != default_value:
			return input_value
		return phase_value

	return PreprocessPlotConfig(
		disable_all_png_diagnostics=_prefer_input(
			phase_plot.disable_all_png_diagnostics,
			input_plot.disable_all_png_diagnostics,
			default_plot.disable_all_png_diagnostics,
		),
		layouts=bool(_prefer_input(phase_plot.layouts, input_plot.layouts, default_plot.layouts)),
		concat_trace=bool(_prefer_input(phase_plot.concat_trace, input_plot.concat_trace, default_plot.concat_trace)),
		segment_traces=bool(_prefer_input(phase_plot.segment_traces, input_plot.segment_traces, default_plot.segment_traces)),
		output_dir=_prefer_input(phase_plot.output_dir, input_plot.output_dir, default_plot.output_dir),
		epoch_markers_output_dir=_prefer_input(
			phase_plot.epoch_markers_output_dir,
			input_plot.epoch_markers_output_dir,
			default_plot.epoch_markers_output_dir,
		),
		assay_stats_relpath=str(
			_prefer_input(phase_plot.assay_stats_relpath, input_plot.assay_stats_relpath, default_plot.assay_stats_relpath)
		),
		channel_layouts_subdir=str(
			_prefer_input(
				phase_plot.channel_layouts_subdir,
				input_plot.channel_layouts_subdir,
				default_plot.channel_layouts_subdir,
			)
		),
		segment_traces_subdir=str(
			_prefer_input(
				phase_plot.segment_traces_subdir,
				input_plot.segment_traces_subdir,
				default_plot.segment_traces_subdir,
			)
		),
		concat_trace_relpath=str(
			_prefer_input(
				phase_plot.concat_trace_relpath,
				input_plot.concat_trace_relpath,
				default_plot.concat_trace_relpath,
			)
		),
		n_representative_channels=int(
			_prefer_input(
				phase_plot.n_representative_channels,
				input_plot.n_representative_channels,
				default_plot.n_representative_channels,
			)
		),
		concat_trace_n_reps=int(
			_prefer_input(
				phase_plot.concat_trace_n_reps,
				input_plot.concat_trace_n_reps,
				default_plot.concat_trace_n_reps,
			)
		),
		segment_trace_n_reps=int(
			_prefer_input(
				phase_plot.segment_trace_n_reps,
				input_plot.segment_trace_n_reps,
				default_plot.segment_trace_n_reps,
			)
		),
		n_jobs=_prefer_input(phase_plot.n_jobs, input_plot.n_jobs, default_plot.n_jobs),
		trace_downsample_hz=_prefer_input(
			phase_plot.trace_downsample_hz,
			input_plot.trace_downsample_hz,
			default_plot.trace_downsample_hz,
		),
		trace_max_points=int(
			_prefer_input(
				phase_plot.trace_max_points,
				input_plot.trace_max_points,
				default_plot.trace_max_points,
			)
		),
	)


def _resolve_effective_plot_config(inputs: PreprocessInputs, *, selected_phase: str | None) -> PreprocessPlotConfig:
	input_plot = PreprocessPlotConfig(
		disable_all_png_diagnostics=None,
		layouts=bool(inputs.plot_layouts),
		concat_trace=bool(inputs.plot_concat_trace),
		segment_traces=bool(inputs.plot_segment_traces),
		output_dir=inputs.plot_output_dir,
		epoch_markers_output_dir=inputs.epoch_markers_output_dir,
		assay_stats_relpath=inputs.assay_stats_relpath,
		channel_layouts_subdir=inputs.channel_layouts_subdir,
		segment_traces_subdir=inputs.segment_traces_subdir,
		concat_trace_relpath=inputs.concat_trace_relpath,
		n_representative_channels=int(inputs.n_representative_channels),
		concat_trace_n_reps=int(inputs.concat_trace_n_reps),
		segment_trace_n_reps=int(inputs.segment_trace_n_reps),
		n_jobs=int(inputs.plot_n_jobs),
		trace_downsample_hz=inputs.trace_downsample_hz,
		trace_max_points=int(inputs.trace_max_points),
	)
	segment_plot = _apply_plot_input_fallback(
		phase_plot=inputs.phases.preprocess_segments.plot,
		input_plot=input_plot,
		default_plot=PreprocessSegmentsPhaseConfig().plot,
	)
	concat_plot = _apply_plot_input_fallback(
		phase_plot=inputs.phases.concatenate_preprocessed_recordings.plot,
		input_plot=input_plot,
		default_plot=PreprocessConcatenatePreprocessedRecordingsPhaseConfig().plot,
	)
	primary = segment_plot
	secondary = concat_plot
	layouts = bool(segment_plot.layouts)
	segment_traces = bool(segment_plot.segment_traces)
	concat_trace = bool(concat_plot.concat_trace)

	if selected_phase in {"preprocess_segments", "save_segment_recordings", "build_preprocessed_recording"}:
		primary = segment_plot
		secondary = concat_plot
		layouts = bool(segment_plot.layouts)
		segment_traces = bool(segment_plot.segment_traces)
		concat_trace = bool(segment_plot.concat_trace)
	elif selected_phase in {"concatenate_preprocessed_recordings", "save_concatenated_recording"}:
		primary = concat_plot
		secondary = segment_plot
		layouts = bool(concat_plot.layouts)
		segment_traces = bool(concat_plot.segment_traces)
		concat_trace = bool(concat_plot.concat_trace)
	elif selected_phase == "save_common_electrodes":
		primary = concat_plot
		secondary = segment_plot
		layouts = False
		segment_traces = False
		concat_trace = False
	else:
		primary = segment_plot if bool(inputs.phases.preprocess_segments.enabled) else concat_plot
		secondary = concat_plot if primary is segment_plot else segment_plot
		layouts = bool(inputs.phases.preprocess_segments.enabled and segment_plot.layouts)
		segment_traces = bool(inputs.phases.preprocess_segments.enabled and segment_plot.segment_traces)
		concat_trace = bool(
			(inputs.phases.concatenate_preprocessed_recordings.enabled and concat_plot.concat_trace)
			or (
				not inputs.phases.concatenate_preprocessed_recordings.enabled
				and inputs.phases.preprocess_segments.enabled
				and segment_plot.concat_trace
			)
		)

	disable_all_png_diagnostics = _coalesce_config_value(
		primary.disable_all_png_diagnostics,
		secondary.disable_all_png_diagnostics,
	)
	if disable_all_png_diagnostics is not None:
		force_all_plots_enabled = not bool(disable_all_png_diagnostics)
		layouts = bool(force_all_plots_enabled)
		segment_traces = bool(force_all_plots_enabled)
		concat_trace = bool(force_all_plots_enabled)

	plot_n_jobs = _coalesce_config_value(primary.n_jobs, secondary.n_jobs, inputs.plot_n_jobs)
	trace_downsample_hz = _coalesce_config_value(
		primary.trace_downsample_hz,
		secondary.trace_downsample_hz,
		inputs.trace_downsample_hz,
	)
	trace_max_points = _coalesce_config_value(
		primary.trace_max_points,
		secondary.trace_max_points,
		inputs.trace_max_points,
	)
	return PreprocessPlotConfig(
		disable_all_png_diagnostics=(
			None if disable_all_png_diagnostics is None else bool(disable_all_png_diagnostics)
		),
		layouts=bool(layouts),
		concat_trace=bool(concat_trace),
		segment_traces=bool(segment_traces),
		output_dir=_coalesce_config_value(primary.output_dir, secondary.output_dir, inputs.plot_output_dir),
		epoch_markers_output_dir=_coalesce_config_value(
			primary.epoch_markers_output_dir,
			secondary.epoch_markers_output_dir,
			inputs.epoch_markers_output_dir,
		),
		assay_stats_relpath=str(
			_coalesce_config_value(primary.assay_stats_relpath, secondary.assay_stats_relpath, inputs.assay_stats_relpath)
		),
		channel_layouts_subdir=str(
			_coalesce_config_value(
				primary.channel_layouts_subdir,
				secondary.channel_layouts_subdir,
				inputs.channel_layouts_subdir,
			)
		),
		segment_traces_subdir=str(
			_coalesce_config_value(
				primary.segment_traces_subdir,
				secondary.segment_traces_subdir,
				inputs.segment_traces_subdir,
			)
		),
		concat_trace_relpath=str(
			_coalesce_config_value(
				concat_plot.concat_trace_relpath,
				primary.concat_trace_relpath,
				secondary.concat_trace_relpath,
				inputs.concat_trace_relpath,
			)
		),
		n_representative_channels=int(
			_coalesce_config_value(
				primary.n_representative_channels,
				secondary.n_representative_channels,
				inputs.n_representative_channels,
			)
		),
		concat_trace_n_reps=int(
			_coalesce_config_value(concat_plot.concat_trace_n_reps, primary.concat_trace_n_reps, inputs.concat_trace_n_reps)
		),
		segment_trace_n_reps=int(
			_coalesce_config_value(segment_plot.segment_trace_n_reps, primary.segment_trace_n_reps, inputs.segment_trace_n_reps)
		),
		n_jobs=(int(plot_n_jobs) if plot_n_jobs is not None else None),
		trace_downsample_hz=(float(trace_downsample_hz) if trace_downsample_hz is not None else None),
		trace_max_points=int(trace_max_points),
	)


def _resolve_phase_output_dir(*, preprocess_out_dir: Path, raw: str | None, default: str) -> Path:
	token = str(raw).strip() if raw is not None else ""
	if not token:
		token = str(default)
	path = Path(token).expanduser()
	if path.is_absolute():
		return Path(os.path.abspath(str(path)))
	return Path(os.path.abspath(str(preprocess_out_dir / path)))


def _resolve_preprocess_paths(inputs: PreprocessInputs, *, plot_cfg: PreprocessPlotConfig) -> _PreprocessPathSet:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	preprocess_out_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
	plot_output_dir: Path | None = None
	if bool(plot_cfg.layouts or plot_cfg.concat_trace or plot_cfg.segment_traces):
		if plot_cfg.output_dir is not None and str(plot_cfg.output_dir).strip():
			plot_output_dir = _resolve_path_from_root(
				root_dir=well_out_dir,
				raw=plot_cfg.output_dir,
				default=PREPROCESS_OUTPUTS_DIRNAME,
			)
		else:
			plot_output_dir = preprocess_out_dir
	epoch_markers_output_dir: Path | None
	if plot_cfg.epoch_markers_output_dir is not None and str(plot_cfg.epoch_markers_output_dir).strip():
		epoch_markers_output_dir = _resolve_path_from_root(
			root_dir=well_out_dir,
			raw=plot_cfg.epoch_markers_output_dir,
			default=PREPROCESS_OUTPUTS_DIRNAME,
		)
	else:
		epoch_markers_output_dir = preprocess_out_dir
	return _PreprocessPathSet(
		well_out_dir=well_out_dir,
		preprocess_out_dir=preprocess_out_dir,
		legacy_out_dir=preprocess_out_dir,
		recording_dir=_resolve_phase_output_dir(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.concatenate_preprocessed_recordings.rel_output_root,
			default="preprocessed_recording",
		),
		common_electrodes_path=preprocess_out_dir / "common_electrodes.npy",
		per_segment_preprocessed_dir=_resolve_phase_output_dir(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.preprocess_segments.rel_output_root,
			default="per_segment_preprocessed",
		),
		per_segment_manifest_path=_resolve_phase_output_dir(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.preprocess_segments.rel_output_root,
			default="per_segment_preprocessed",
		) / "manifest.json",
		preprocess_config_path=preprocess_out_dir / "preprocess_config.json",
		stage_summary_json=preprocess_out_dir / "preprocess_summary.json",
		stage_log_source=_resolve_stage_log_source(inputs=inputs, well_out_dir=well_out_dir),
		plot_output_dir=plot_output_dir,
		epoch_markers_output_dir=epoch_markers_output_dir,
	)


def _build_requested_preprocess_cfg(inputs: PreprocessInputs, *, plot_cfg: PreprocessPlotConfig) -> dict[str, Any]:
	return {
		"temporal_resample_factor": (int(inputs.temporal_resample_factor) if inputs.temporal_resample_factor is not None else None),
		"temporal_resample_rate_hz": (int(inputs.temporal_resample_rate_hz) if inputs.temporal_resample_rate_hz is not None else None),
		"temporal_resample_margin_ms": float(inputs.temporal_resample_margin_ms),
		"temporal_resample_dtype": (str(inputs.temporal_resample_dtype) if inputs.temporal_resample_dtype is not None else None),
		"limit_segments_per_well": (
			int(inputs.debug_limit_segments_per_well)
			if inputs.debug_limit_segments_per_well is not None
			else None
		),
		"plot_layouts": bool(plot_cfg.layouts),
		"plot_concat_trace": bool(plot_cfg.concat_trace),
		"plot_segment_traces": bool(plot_cfg.segment_traces),
		"n_representative_channels": int(plot_cfg.n_representative_channels),
		"concat_trace_n_reps": int(plot_cfg.concat_trace_n_reps),
		"segment_trace_n_reps": int(plot_cfg.segment_trace_n_reps),
		"plot_n_jobs": int(max(1, int(plot_cfg.n_jobs or inputs.plot_n_jobs))),
	}


def _normalize_phase_timing(artifacts: dict[str, object]) -> dict[str, float]:
	phase_timing_s_raw = artifacts.get("phase_timing_s", {}) if isinstance(artifacts, dict) else {}
	phase_timing_s: dict[str, float] = {}
	if isinstance(phase_timing_s_raw, dict):
		for key, value in phase_timing_s_raw.items():
			try:
				phase_timing_s[str(key)] = float(value)
			except Exception:
				continue
	return phase_timing_s


def _resolve_save_workers(inputs: PreprocessInputs) -> tuple[int, int]:
	def _resolve_jobs(raw: int | None, *, fallback: int) -> int:
		try:
			parsed = int(raw) if raw is not None else int(fallback)
		except Exception:
			parsed = int(fallback)
		return max(1, int(parsed))

	return (
		_resolve_jobs(inputs.phases.concatenate_preprocessed_recordings.outputs.concat_save_n_jobs, fallback=int(inputs.n_jobs)),
		_resolve_jobs(inputs.phases.preprocess_segments.outputs.segment_save_n_jobs, fallback=int(inputs.n_jobs)),
	)


def _resolve_phase_execution_flags(
	inputs: PreprocessInputs,
	*,
	selected_phase: str | None = None,
) -> dict[str, bool]:
	if selected_phase == "copy_src_to_scratch":
		return {
			"build_required": False,
			"save_concatenated_recording": False,
			"save_segment_recordings": False,
			"save_common_electrodes": False,
		}
	if selected_phase == "build_preprocessed_recording":
		return {
			"build_required": True,
			"save_concatenated_recording": False,
			"save_segment_recordings": False,
			"save_common_electrodes": False,
		}
	if selected_phase == "save_rec_metadata":
		return {
			"build_required": False,
			"save_concatenated_recording": False,
			"save_segment_recordings": False,
			"save_common_electrodes": False,
		}
	if selected_phase == "wipe_src_scratch":
		return {
			"build_required": False,
			"save_concatenated_recording": False,
			"save_segment_recordings": False,
			"save_common_electrodes": False,
		}
	if selected_phase == "preprocess_segments":
		return {
			"build_required": True,
			"save_concatenated_recording": False,
			"save_segment_recordings": True,
			"save_common_electrodes": False,
		}
	if selected_phase == "save_concatenated_recording":
		return {
			"build_required": True,
			"save_concatenated_recording": True,
			"save_segment_recordings": False,
			"save_common_electrodes": False,
		}
	if selected_phase == "concatenate_preprocessed_recordings":
		return {
			"build_required": True,
			"save_concatenated_recording": True,
			"save_segment_recordings": False,
			"save_common_electrodes": bool(inputs.phases.concatenate_preprocessed_recordings.save_common_electrodes.enabled),
		}
	if selected_phase == "save_segment_recordings":
		return {
			"build_required": True,
			"save_concatenated_recording": False,
			"save_segment_recordings": True,
			"save_common_electrodes": False,
		}
	if selected_phase == "save_common_electrodes":
		return {
			"build_required": True,
			"save_concatenated_recording": False,
			"save_segment_recordings": False,
			"save_common_electrodes": True,
		}
	save_concat_enabled = bool(inputs.phases.concatenate_preprocessed_recordings.enabled) and bool(inputs.save_concat_recording)
	save_segment_enabled = bool(inputs.phases.preprocess_segments.enabled) and bool(inputs.save_segment_recordings)
	save_common_enabled = bool(inputs.phases.concatenate_preprocessed_recordings.save_common_electrodes.enabled) and bool(
		inputs.save_recording or save_concat_enabled or save_segment_enabled
	)
	build_required = bool(save_concat_enabled or save_segment_enabled or save_common_enabled)
	return {
		"build_required": bool(build_required),
		"save_concatenated_recording": bool(save_concat_enabled),
		"save_segment_recordings": bool(save_segment_enabled),
		"save_common_electrodes": bool(save_common_enabled),
	}


def _copy_phase_requested(inputs: PreprocessInputs, *, selected_phase: str | None) -> bool:
	if selected_phase == "copy_src_to_scratch":
		return True
	if selected_phase is not None:
		return False
	return bool(inputs.phases.copy_src_to_scratch.enabled)


def _recording_metadata_phase_requested(inputs: PreprocessInputs, *, selected_phase: str | None) -> bool:
	if selected_phase == "save_rec_metadata":
		return True
	if selected_phase is not None:
		return False
	return bool(inputs.phases.save_rec_metadata.enabled)


def _wipe_src_scratch_phase_requested(inputs: PreprocessInputs, *, selected_phase: str | None) -> bool:
	if selected_phase == "wipe_src_scratch":
		return True
	if selected_phase is not None:
		return False
	return bool(inputs.phases.wipe_src_scratch.enabled)


def _validate_copy_phase_requirements(inputs: PreprocessInputs, *, selected_phase: str | None) -> None:
	if not _copy_phase_requested(inputs, selected_phase=selected_phase):
		return
	if bool(inputs.phases.copy_src_to_scratch.requires_use_scratch_root) and not bool(inputs.copied_to_scratch):
		raise RuntimeError(
			"preprocess copy_src_to_scratch phase requires scratch input materialization, but the selected target is using the source h5 path"
		)


def _validate_wipe_phase_requirements(inputs: PreprocessInputs, *, selected_phase: str | None) -> None:
	if not _wipe_src_scratch_phase_requested(inputs, selected_phase=selected_phase):
		return
	if bool(inputs.phases.wipe_src_scratch.requires_use_scratch_root) and not bool(inputs.copied_to_scratch):
		raise RuntimeError(
			"preprocess wipe_src_scratch phase requires scratch input materialization, but the selected target is using the source h5 path"
		)


def _build_copy_src_to_scratch_phase_payload(inputs: PreprocessInputs) -> dict[str, Any]:
	return {
		"phase": "copy_src_to_scratch",
		"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
		"resolved_h5_path": str(inputs.h5_path),
		"copied_to_scratch": bool(inputs.copied_to_scratch),
		"requires_use_scratch_root": bool(inputs.phases.copy_src_to_scratch.requires_use_scratch_root),
	}


def _scratch_input_usage_key(inputs: PreprocessInputs) -> str | None:
	if not bool(inputs.copied_to_scratch):
		return None
	try:
		return str(Path(os.path.abspath(str(Path(inputs.h5_path).expanduser()))))
	except Exception:
		return str(inputs.h5_path)


def _acquire_scratch_input_usage(inputs: PreprocessInputs, *, selected_phase: str | None) -> str | None:
	if not _wipe_src_scratch_phase_requested(inputs, selected_phase=selected_phase):
		return None
	usage_key = _scratch_input_usage_key(inputs)
	if usage_key is None:
		return None
	with _SCRATCH_INPUT_USAGE_LOCK:
		_SCRATCH_INPUT_ACTIVE_COUNTS[usage_key] = int(_SCRATCH_INPUT_ACTIVE_COUNTS.get(usage_key, 0)) + 1
	return usage_key


def _release_scratch_input_usage(usage_key: str | None) -> int:
	if usage_key is None:
		return 0
	with _SCRATCH_INPUT_USAGE_LOCK:
		current = int(_SCRATCH_INPUT_ACTIVE_COUNTS.get(usage_key, 0))
		if current <= 1:
			_SCRATCH_INPUT_ACTIVE_COUNTS.pop(usage_key, None)
			return 0
		current -= 1
		_SCRATCH_INPUT_ACTIVE_COUNTS[usage_key] = int(current)
		return int(current)


def _candidate_wipe_src_scratch_paths(inputs: PreprocessInputs) -> list[Path]:
	if not bool(inputs.copied_to_scratch):
		return []
	resolved_h5_path = Path(inputs.h5_path).expanduser()
	out: list[Path] = [resolved_h5_path]
	try:
		out.extend(sorted(resolved_h5_path.parent.glob("*.cfg")))
	except Exception:
		pass
	unique: list[Path] = []
	seen: set[str] = set()
	for path in out:
		key = str(path)
		if key in seen:
			continue
		seen.add(key)
		unique.append(path)
	return unique


def _execute_wipe_src_scratch_phase(inputs: PreprocessInputs, *, usage_key: str | None) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"phase": "wipe_src_scratch",
		"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
		"resolved_h5_path": str(inputs.h5_path),
		"copied_to_scratch": bool(inputs.copied_to_scratch),
		"dry_run": bool(inputs.phases.wipe_src_scratch.dry_run),
		"requires_use_scratch_root": bool(inputs.phases.wipe_src_scratch.requires_use_scratch_root),
	}
	if not bool(inputs.copied_to_scratch):
		payload.update(
			{
				"status": "skipped",
				"reason": "selected_target_did_not_use_scratch_input_root",
				"removed_paths": [],
				"would_remove_paths": [],
				"missing_paths": [],
			}
		)
		return payload

	active_shared_users_remaining = _release_scratch_input_usage(usage_key)
	payload["active_shared_users_remaining"] = int(active_shared_users_remaining)
	if int(active_shared_users_remaining) > 0:
		payload.update(
			{
				"status": "deferred",
				"reason": "shared_scratch_input_still_in_use",
				"removed_paths": [],
				"would_remove_paths": [],
				"missing_paths": [],
			}
		)
		return payload

	removed_paths: list[str] = []
	would_remove_paths: list[str] = []
	missing_paths: list[str] = []
	errors: list[str] = []
	for path in _candidate_wipe_src_scratch_paths(inputs):
		try:
			if not path.exists() and not path.is_symlink():
				missing_paths.append(str(path))
				continue
			if bool(inputs.phases.wipe_src_scratch.dry_run):
				would_remove_paths.append(str(path))
				continue
			path.unlink()
			removed_paths.append(str(path))
		except FileNotFoundError:
			missing_paths.append(str(path))
		except Exception as exc:
			errors.append(f"{path}: {type(exc).__name__}: {exc}")

	payload["removed_paths"] = list(removed_paths)
	payload["would_remove_paths"] = list(would_remove_paths)
	payload["missing_paths"] = list(missing_paths)
	if errors:
		raise RuntimeError("wipe_src_scratch failed: " + "; ".join(errors))
	if bool(inputs.phases.wipe_src_scratch.dry_run):
		if would_remove_paths:
			LOGGER.info(
				"wipe_src_scratch dry run for %s would remove %d path(s): %s",
				inputs.stream_id,
				len(would_remove_paths),
				", ".join(would_remove_paths),
			)
			payload["status"] = "dry_run"
		else:
			payload["status"] = "already_missing"
		return payload
	if removed_paths:
		payload["status"] = "ok"
	else:
		payload["status"] = "already_missing"
	return payload


def _parse_recording_context_from_path(path: Path) -> dict[str, str]:
	parts = list(path.parts)
	out: dict[str, str] = {}
	try:
		out["filename"] = path.name
		out["parent_dir"] = path.parent.name
		if len(parts) >= 6:
			out["dataset"] = parts[-6]
			out["date"] = parts[-5]
			out["plate"] = parts[-4]
			out["assay"] = parts[-3]
			out["run"] = parts[-2]
	except Exception:
		pass
	return out


def _try_get_spikeinterface_recording_info(
	*,
	h5_path: Path,
	stream_id: str,
	suppress_h5_plugin_messages: bool,
) -> tuple[dict[str, Any], str | None]:
	try:
		import spikeinterface.extractors as se  # type: ignore[import-not-found]
	except Exception as exc:
		return {}, f"SpikeInterface import failed: {exc}"

	try:
		with contextlib.ExitStack() as stack:
			if bool(suppress_h5_plugin_messages):
				suppressed_stream = io.StringIO()
				stack.enter_context(contextlib.redirect_stdout(suppressed_stream))
				stack.enter_context(contextlib.redirect_stderr(suppressed_stream))
			recording = se.read_maxwell(h5_path, stream_id=stream_id)
	except Exception as exc:
		return {}, f"SpikeInterface read_maxwell failed: {exc}"

	info: dict[str, Any] = {}
	try:
		info["sampling_frequency_hz"] = float(recording.get_sampling_frequency())
	except Exception:
		pass
	try:
		info["num_channels"] = int(recording.get_num_channels())
	except Exception:
		pass
	try:
		info["num_segments"] = int(recording.get_num_segments())
	except Exception:
		pass
	try:
		get_dtype = getattr(recording, "get_dtype", None)
		if callable(get_dtype):
			info["dtype"] = str(get_dtype())
	except Exception:
		pass
	try:
		fs = float(recording.get_sampling_frequency())
		nseg = int(recording.get_num_segments())
		segment_frames = [int(recording.get_num_frames(segment_index=segment_index)) for segment_index in range(nseg)]
		if segment_frames and fs > 0:
			total_frames = int(sum(segment_frames))
			info["duration_s_total"] = float(total_frames / fs)
			if nseg == 1:
				info["num_frames"] = int(segment_frames[0])
			else:
				info["num_frames_total"] = int(total_frames)
				info["num_frames_by_segment"] = [int(value) for value in segment_frames]
	except Exception:
		pass
	try:
		get_channel_locations = getattr(recording, "get_channel_locations", None)
		if callable(get_channel_locations):
			locations = get_channel_locations()
			shape = getattr(locations, "shape", None)
			if shape is not None:
				info["channel_locations_shape"] = [int(value) for value in list(shape)]
	except Exception:
		pass
	return info, None


def _build_recording_metadata_phase_payload(inputs: PreprocessInputs) -> dict[str, Any]:
	source_h5_path = Path(inputs.source_h5_path or inputs.h5_path)
	resolved_h5_path = Path(inputs.h5_path)
	recording_info, recording_info_error = _try_get_spikeinterface_recording_info(
		h5_path=resolved_h5_path,
		stream_id=str(inputs.stream_id),
		suppress_h5_plugin_messages=bool(inputs.logging_suppress_h5_plugin_messages),
	)
	payload: dict[str, Any] = {
		"source_h5_path": str(source_h5_path),
		"resolved_h5_path": str(resolved_h5_path),
		"copied_to_scratch": bool(inputs.copied_to_scratch),
		"source_path_context": _parse_recording_context_from_path(source_h5_path),
		"source_file": _collect_path_details(source_h5_path),
		"recording_info": recording_info,
	}
	if not _paths_equal_no_resolve(left=source_h5_path, right=resolved_h5_path):
		payload["resolved_path_context"] = _parse_recording_context_from_path(resolved_h5_path)
		payload["resolved_file"] = _collect_path_details(resolved_h5_path)
	if recording_info_error is not None:
		payload["recording_info_error"] = str(recording_info_error)
	return payload


def _write_preprocess_config_json(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	cfg_discovery_summary: dict[str, Any],
	phase_timing_s: dict[str, float],
	effective_save_recording: bool,
	effective_save_concat_recording: bool,
	effective_save_segment_recordings: bool,
	concat_jobs: int,
	segment_jobs: int,
	build_plot_cfg: PreprocessPlotConfig,
) -> None:
	payload = {
		"generated_utc": _utc_now_iso(),
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"cfg_discovery_summary": cfg_discovery_summary,
		"logging_cfg": {
			"enabled": bool(inputs.logging_enabled),
			"verbose": bool(inputs.logging_verbose),
			"file_relpath": inputs.logging_file_relpath,
			"suppress_h5_plugin_messages": bool(inputs.logging_suppress_h5_plugin_messages),
			"phase_dividers": bool(inputs.logging_phase_dividers),
		},
		"requested_cfg": _build_requested_preprocess_cfg(inputs, plot_cfg=build_plot_cfg),
		"plot_cfg": {
			"plot_layouts": bool(build_plot_cfg.layouts),
			"plot_concat_trace": bool(build_plot_cfg.concat_trace),
			"plot_segment_traces": bool(build_plot_cfg.segment_traces),
			"limit_segments_per_well": (
				int(inputs.debug_limit_segments_per_well)
				if inputs.debug_limit_segments_per_well is not None
				else None
			),
			"n_representative_channels": int(build_plot_cfg.n_representative_channels),
			"concat_trace_n_reps": int(build_plot_cfg.concat_trace_n_reps),
			"segment_trace_n_reps": int(build_plot_cfg.segment_trace_n_reps),
			"plot_n_jobs": max(1, int(build_plot_cfg.n_jobs or inputs.plot_n_jobs)),
			"trace_downsample_hz": build_plot_cfg.trace_downsample_hz,
			"trace_max_points": _normalize_trace_max_points(build_plot_cfg.trace_max_points),
		},
		"save_cfg": {
			"save_recording": bool(effective_save_recording),
			"overwrite_saved_recording": bool(inputs.overwrite_saved_recording),
			"save_concat_recording": bool(effective_save_concat_recording),
			"save_segment_recordings": bool(effective_save_segment_recordings),
			"concatenate_preprocessed_recordings": {
				"save_chunk_duration": str(inputs.phases.concatenate_preprocessed_recordings.outputs.save_chunk_duration),
				"save_progress_bar": bool(inputs.phases.concatenate_preprocessed_recordings.outputs.save_progress_bar),
				"concat_save_n_jobs": int(concat_jobs),
				"print_n_jobs_used": bool(inputs.phases.concatenate_preprocessed_recordings.outputs.print_n_jobs_used),
			},
			"preprocess_segments": {
				"save_chunk_duration": str(inputs.phases.preprocess_segments.outputs.save_chunk_duration),
				"save_progress_bar": bool(inputs.phases.preprocess_segments.outputs.save_progress_bar),
				"segment_save_n_jobs": int(segment_jobs),
				"print_n_jobs_used": bool(inputs.phases.preprocess_segments.outputs.print_n_jobs_used),
			},
		},
		"phase_cfg": _json_ready(asdict(inputs.phases)),
		"phase_timing_s": {str(key): float(value) for key, value in phase_timing_s.items()},
	}
	_write_json(paths.preprocess_config_path, _json_ready(payload))


def _write_phase_summary(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	phase_name: str,
	summary_json_relpath: str,
	outputs: dict[str, str],
	extra_payload: dict[str, Any],
) -> dict[str, Any]:
	summary_json = paths.preprocess_out_dir / Path(str(summary_json_relpath)).expanduser()
	payload: dict[str, Any] = {
		"phase": phase_name,
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"well_out_dir": str(paths.well_out_dir),
		"preprocess_out_dir": str(paths.preprocess_out_dir),
		"outputs": dict(outputs),
	}
	payload.update(extra_payload)
	_write_json(summary_json, _json_ready(payload))
	payload["summary_json"] = str(summary_json)
	return payload


def _execute_preprocess_phase_work(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	selected_phase: str | None,
) -> tuple[dict[str, bool], dict[str, str], dict[str, Any], list[int], dict[str, float], dict[str, Any]]:
	_validate_copy_phase_requirements(inputs, selected_phase=selected_phase)
	_validate_wipe_phase_requirements(inputs, selected_phase=selected_phase)
	phase_flags = _resolve_phase_execution_flags(inputs, selected_phase=selected_phase)
	outputs: dict[str, str] = {
		"legacy.preprocess_out_dir": str(paths.legacy_out_dir),
		"preprocessed_recording_dir": str(paths.recording_dir),
		"common_electrodes_path": str(paths.common_electrodes_path),
		"per_segment_manifest_json": str(paths.per_segment_manifest_path),
		"per_segment_preprocessed_dir": str(paths.per_segment_preprocessed_dir),
	}
	if bool(inputs.logging_enabled) or _safe_path_exists(paths.stage_log_source):
		outputs["pipeline_log"] = str(paths.stage_log_source)
	if paths.plot_output_dir is not None:
		outputs["plot_output_dir"] = str(paths.plot_output_dir)
	if paths.epoch_markers_output_dir is not None:
		outputs["epoch_markers_output_dir"] = str(paths.epoch_markers_output_dir)

	phase_results: dict[str, Any] = {}
	common_electrodes: list[int] = []
	phase_timing_s: dict[str, float] = {}
	cfg_summary = {}
	if not bool(phase_flags.get("build_required", False)):
		return phase_flags, outputs, phase_results, common_electrodes, phase_timing_s, cfg_summary

	paths.preprocess_out_dir.mkdir(parents=True, exist_ok=True)
	phase_logger = _prepare_phase_logger(inputs, paths)
	build_plot_cfg = _resolve_effective_plot_config(inputs, selected_phase=selected_phase)
	plan = build_preprocess_plan(h5_path=inputs.h5_path, stream_id=inputs.stream_id)
	cfg_summary = plan.cfg_discovery_summary if isinstance(plan.cfg_discovery_summary, dict) else {}
	multirecording, common_electrodes, build_artifacts = run_build_preprocessed_recording_core(
		h5_path=inputs.h5_path,
		stream_id=inputs.stream_id,
		n_jobs=max(1, int(inputs.n_jobs)),
		plot_output_dir=paths.plot_output_dir,
		plot_layouts=bool(build_plot_cfg.layouts),
		plot_concat_trace=bool(build_plot_cfg.concat_trace),
		plot_segment_traces=bool(build_plot_cfg.segment_traces),
		epoch_markers_output_dir=paths.epoch_markers_output_dir,
		assay_stats_relpath=build_plot_cfg.assay_stats_relpath,
		channel_layouts_subdir=build_plot_cfg.channel_layouts_subdir,
		segment_traces_subdir=build_plot_cfg.segment_traces_subdir,
		concat_trace_relpath=build_plot_cfg.concat_trace_relpath,
		n_representative_channels=int(build_plot_cfg.n_representative_channels),
		concat_trace_n_reps=int(build_plot_cfg.concat_trace_n_reps),
		segment_trace_n_reps=int(build_plot_cfg.segment_trace_n_reps),
		plot_n_jobs=max(1, int(build_plot_cfg.n_jobs or inputs.plot_n_jobs)),
		trace_downsample_hz=build_plot_cfg.trace_downsample_hz,
		trace_max_points=_normalize_trace_max_points(build_plot_cfg.trace_max_points),
		limit_segments_per_well=(
			int(inputs.debug_limit_segments_per_well)
			if inputs.debug_limit_segments_per_well is not None
			else None
		),
		temporal_resample_factor=inputs.temporal_resample_factor,
		temporal_resample_rate_hz=inputs.temporal_resample_rate_hz,
		temporal_resample_margin_ms=float(inputs.temporal_resample_margin_ms),
		temporal_resample_dtype=inputs.temporal_resample_dtype,
		phase_dividers=bool(inputs.logging_phase_dividers),
		suppress_h5_plugin_messages=bool(inputs.logging_suppress_h5_plugin_messages),
		logger=phase_logger,
	)
	phase_timing_s = _normalize_phase_timing(build_artifacts)
	phase_results["build_preprocessed_recording"] = {
		"segment_count": int(len(list(build_artifacts.get("rec_names", []) or []))),
		"rec_names": [str(value) for value in list(build_artifacts.get("rec_names", []) or [])],
		"phase_timing_s": phase_timing_s,
		"n_common_electrodes": int(len(common_electrodes)),
	}
	concat_jobs, segment_jobs = _resolve_save_workers(inputs)
	_write_preprocess_config_json(
		inputs=inputs,
		paths=paths,
		cfg_discovery_summary=cfg_summary,
		phase_timing_s=phase_timing_s,
		effective_save_recording=bool(
			phase_flags["save_concatenated_recording"]
			or phase_flags["save_segment_recordings"]
			or phase_flags["save_common_electrodes"]
		),
		effective_save_concat_recording=bool(phase_flags["save_concatenated_recording"]),
		effective_save_segment_recordings=bool(phase_flags["save_segment_recordings"]),
		concat_jobs=concat_jobs,
		segment_jobs=segment_jobs,
		build_plot_cfg=build_plot_cfg,
	)
	if bool(phase_flags["save_concatenated_recording"]):
		phase_results["save_concatenated_recording"] = run_save_concatenated_recording_core(
			multirecording=multirecording,
			recording_dir=paths.recording_dir,
			overwrite_saved_recording=bool(inputs.overwrite_saved_recording),
			n_jobs=concat_jobs,
			chunk_duration=str(inputs.phases.concatenate_preprocessed_recordings.outputs.save_chunk_duration),
			progress_bar=bool(inputs.phases.concatenate_preprocessed_recordings.outputs.save_progress_bar),
			logger=phase_logger,
		)
	if bool(phase_flags["save_segment_recordings"]):
		phase_results["save_segment_recordings"] = run_save_segment_recordings_core(
			segment_recordings=list(build_artifacts.get("segment_recordings_preprocessed", []) or []),
			segment_names=[str(value) for value in list(build_artifacts.get("rec_names", []) or [])],
			segment_stats=[
				dict(item) if isinstance(item, dict) else {}
				for item in list(build_artifacts.get("segment_stats", []) or [])
			],
			output_dir=paths.per_segment_preprocessed_dir,
			manifest_path=paths.per_segment_manifest_path,
			overwrite_saved_recording=bool(inputs.overwrite_saved_recording),
			n_jobs=segment_jobs,
			chunk_duration=str(inputs.phases.preprocess_segments.outputs.save_chunk_duration),
			progress_bar=bool(inputs.phases.preprocess_segments.outputs.save_progress_bar),
			logger=phase_logger,
		)
	if bool(phase_flags["save_common_electrodes"]):
		phase_results["save_common_electrodes"] = run_save_common_electrodes_core(
			common_electrodes=common_electrodes,
			output_path=paths.common_electrodes_path,
			logger=phase_logger,
		)
	return phase_flags, outputs, phase_results, common_electrodes, phase_timing_s, cfg_summary


def _write_enabled_phase_summaries(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	phase_results: dict[str, Any],
	outputs: dict[str, str],
	common_electrodes: list[int],
	phase_timing_s: dict[str, float],
	selected_phase: str | None,
	wipe_src_scratch_payload: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
	phase_summaries: dict[str, dict[str, Any]] = {}
	if _copy_phase_requested(inputs, selected_phase=selected_phase):
		phase_summaries["copy_src_to_scratch"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="copy_src_to_scratch",
			summary_json_relpath=str(inputs.phases.copy_src_to_scratch.summary_json_relpath),
			outputs={
				"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
				"resolved_h5_path": str(inputs.h5_path),
			},
			extra_payload=_build_copy_src_to_scratch_phase_payload(inputs),
		)
	if _recording_metadata_phase_requested(inputs, selected_phase=selected_phase):
		phase_summaries["save_rec_metadata"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="save_rec_metadata",
			summary_json_relpath=str(inputs.phases.save_rec_metadata.summary_json_relpath),
			outputs={
				"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
				"resolved_h5_path": str(inputs.h5_path),
			},
			extra_payload=_build_recording_metadata_phase_payload(inputs),
		)
	if _wipe_src_scratch_phase_requested(inputs, selected_phase=selected_phase) and wipe_src_scratch_payload is not None:
		phase_summaries["wipe_src_scratch"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="wipe_src_scratch",
			summary_json_relpath=str(inputs.phases.wipe_src_scratch.summary_json_relpath),
			outputs={
				"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
				"resolved_h5_path": str(inputs.h5_path),
			},
			extra_payload=dict(wipe_src_scratch_payload),
		)
	if ("build_preprocessed_recording" in phase_results or "save_segment_recordings" in phase_results) and (
		selected_phase in {"build_preprocessed_recording", "save_segment_recordings", "preprocess_segments"}
		or (selected_phase is None and bool(inputs.phases.preprocess_segments.enabled))
	):
		segment_payload = dict(phase_results.get("save_segment_recordings", {}))
		segment_payload.update(
			{
				"n_common_electrodes": int(len(common_electrodes)),
				"segment_count": int(phase_results.get("build_preprocessed_recording", {}).get("segment_count", 0)),
				"rec_names": list(phase_results.get("build_preprocessed_recording", {}).get("rec_names", [])),
				"phase_timing_s": phase_timing_s,
			}
		)
		phase_summaries["preprocess_segments"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="preprocess_segments",
			summary_json_relpath=str(inputs.phases.preprocess_segments.summary_json_relpath),
			outputs={
				"per_segment_preprocessed_dir": outputs["per_segment_preprocessed_dir"],
				"per_segment_manifest_json": outputs["per_segment_manifest_json"],
				"plot_output_dir": outputs.get("plot_output_dir", outputs["legacy.preprocess_out_dir"]),
				"epoch_markers_output_dir": outputs.get("epoch_markers_output_dir", outputs["legacy.preprocess_out_dir"]),
			},
			extra_payload=segment_payload,
		)
	if "save_concatenated_recording" in phase_results and (
		selected_phase in {"save_concatenated_recording", "concatenate_preprocessed_recordings"}
		or (selected_phase is None and bool(inputs.phases.concatenate_preprocessed_recordings.enabled))
	):
		phase_summaries["concatenate_preprocessed_recordings"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="concatenate_preprocessed_recordings",
			summary_json_relpath=str(inputs.phases.concatenate_preprocessed_recordings.summary_json_relpath),
			outputs={"preprocessed_recording_dir": outputs["preprocessed_recording_dir"]},
			extra_payload=dict(phase_results["save_concatenated_recording"]),
		)
	if "save_common_electrodes" in phase_results and (
		selected_phase == "save_common_electrodes"
		or selected_phase == "concatenate_preprocessed_recordings"
		or (
			selected_phase is None
			and bool(inputs.phases.concatenate_preprocessed_recordings.save_common_electrodes.enabled)
		)
	):
		phase_summaries["save_common_electrodes"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="save_common_electrodes",
			summary_json_relpath=str(inputs.phases.concatenate_preprocessed_recordings.save_common_electrodes.summary_json_relpath),
			outputs={"common_electrodes_path": outputs["common_electrodes_path"]},
			extra_payload=dict(phase_results["save_common_electrodes"]),
		)
	for phase_name, payload in phase_summaries.items():
		outputs[f"{phase_name}_summary_json"] = str(payload["summary_json"])
	return phase_summaries

def run_preprocess_stage(inputs: PreprocessInputs) -> PreprocessResult:
	build_plot_cfg = _resolve_effective_plot_config(inputs, selected_phase=None)
	paths = _resolve_preprocess_paths(inputs, plot_cfg=build_plot_cfg)
	legacy_out_dir = paths.legacy_out_dir
	if str(inputs.output_rel_root).strip() != PREPROCESS_OUTPUTS_DIRNAME:
		LOGGER.info(
			"Ignoring preprocess output_rel_root=%s; using canonical directory=%s",
			inputs.output_rel_root,
			PREPROCESS_OUTPUTS_DIRNAME,
		)

	stage_started_utc = _utc_now_iso()
	stage_t0 = time.perf_counter()
	event_records: list[dict[str, Any]] = [
		{
			"event": "stage_start",
			"utc": stage_started_utc,
			"details": {
				"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
				"h5_path": str(inputs.h5_path),
				"copied_to_scratch": bool(inputs.copied_to_scratch),
				"stream_id": str(inputs.stream_id),
			},
		}
	]
	if _copy_phase_requested(inputs, selected_phase=None):
		event_records.append(
			{
				"event": "copy_src_to_scratch",
				"utc": _utc_now_iso(),
				"details": _build_copy_src_to_scratch_phase_payload(inputs),
			}
		)

	stage_log_source = paths.stage_log_source
	_clear_self_referential_symlink(stage_log_source)
	scratch_usage_key = _acquire_scratch_input_usage(inputs, selected_phase=None)
	scratch_usage_released = False
	wipe_src_scratch_payload: dict[str, Any] | None = None

	try:
		phase_flags, outputs, phase_results, common_electrodes, phase_timing_s, _cfg_summary = _execute_preprocess_phase_work(
			inputs=inputs,
			paths=paths,
			selected_phase=None,
		)
		if _wipe_src_scratch_phase_requested(inputs, selected_phase=None):
			wipe_src_scratch_payload = _execute_wipe_src_scratch_phase(inputs, usage_key=scratch_usage_key)
			scratch_usage_released = True
			event_records.append(
				{
					"event": "wipe_src_scratch",
					"utc": _utc_now_iso(),
					"details": dict(wipe_src_scratch_payload),
				}
			)
		phase_summaries = _write_enabled_phase_summaries(
			inputs=inputs,
			paths=paths,
			phase_results=phase_results,
			outputs=outputs,
			common_electrodes=common_electrodes,
			phase_timing_s=phase_timing_s,
			selected_phase=None,
			wipe_src_scratch_payload=wipe_src_scratch_payload,
		)
		event_records.append(
			{
				"event": "phase_execution_complete",
				"utc": _utc_now_iso(),
				"details": {
					"phase_count": int(len(phase_summaries)),
					"n_common_electrodes": int(len(common_electrodes)),
				},
			}
		)
	except Exception as exc:
		event_records.append(
			{
				"event": "preprocess_stage_failed",
				"utc": _utc_now_iso(),
				"details": {
					"error_type": type(exc).__name__,
					"error_message": str(exc),
				},
			}
		)
		if _is_observability_enabled(inputs):
			stage_elapsed_s = float(max(0.0, time.perf_counter() - stage_t0))
			legacy_out_dir.mkdir(parents=True, exist_ok=True)
			try:
				outputs_on_error: dict[str, str] = {
					"legacy.preprocess_out_dir": str(legacy_out_dir),
				}
				obs_outputs = _write_observability_artifacts(
					inputs=inputs,
					well_out_dir=paths.well_out_dir,
					preprocess_out_dir=legacy_out_dir,
					legacy_out_dir=legacy_out_dir,
					outputs=outputs_on_error,
					summary_json=None,
					event_records=event_records,
					common_electrodes=[],
					stage_status="error",
					stage_started_utc=stage_started_utc,
					stage_elapsed_s=stage_elapsed_s,
					stage_error=exc,
					stage_log_source=(stage_log_source if _safe_path_exists(stage_log_source) else None),
				)
				if obs_outputs:
					failure_summary_json = legacy_out_dir / "preprocess_failure_summary.json"
					_write_json(
						failure_summary_json,
						{
							"stage": "preprocess",
							"status": "error",
							"h5_path": str(inputs.h5_path),
							"stream_id": str(inputs.stream_id),
							"error": {
								"type": type(exc).__name__,
								"message": str(exc),
							},
							"observability_outputs": obs_outputs,
						},
					)
			except Exception:
				LOGGER.warning("Failed writing preprocess failure observability artifacts", exc_info=True)
		raise
	finally:
		if scratch_usage_key is not None and not bool(scratch_usage_released):
			_release_scratch_input_usage(scratch_usage_key)

	stage_elapsed_s = float(max(0.0, time.perf_counter() - stage_t0))
	event_records.append(
		{
			"event": "stage_complete",
			"utc": _utc_now_iso(),
			"details": {
				"status": "ok",
				"duration_seconds": stage_elapsed_s,
				"n_common_electrodes": int(len(common_electrodes)),
			},
		}
	)

	summary_payload = {
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"well_out_dir": str(paths.well_out_dir),
		"preprocess_out_dir": str(paths.preprocess_out_dir),
		"legacy_preprocess_out_dir": str(legacy_out_dir),
		"output_rel_root": str(inputs.output_rel_root),
		"n_common_electrodes": int(len(common_electrodes)),
		"timing": {
			"started_utc": stage_started_utc,
			"ended_utc": _utc_now_iso(),
			"duration_seconds": stage_elapsed_s,
		},
		"inputs": {
			"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
			"copied_to_scratch": bool(inputs.copied_to_scratch),
			"force_restart": bool(inputs.force_restart),
			"force_replot": bool(inputs.force_replot),
			"debug_limit_segments_per_well": (
				int(inputs.debug_limit_segments_per_well)
				if inputs.debug_limit_segments_per_well is not None
				else None
			),
			"logging_enabled": bool(inputs.logging_enabled),
			"logging_verbose": bool(inputs.logging_verbose),
			"logging_file_relpath": inputs.logging_file_relpath,
			"logging_suppress_h5_plugin_messages": bool(inputs.logging_suppress_h5_plugin_messages),
			"logging_phase_dividers": bool(inputs.logging_phase_dividers),
			"enable_checkpointing": bool(inputs.enable_checkpointing),
			"n_jobs": int(max(1, int(inputs.n_jobs))),
			"plot_layouts": bool(inputs.plot_layouts),
			"plot_concat_trace": bool(inputs.plot_concat_trace),
			"plot_segment_traces": bool(inputs.plot_segment_traces),
			"plot_output_dir": inputs.plot_output_dir,
			"epoch_markers_output_dir": inputs.epoch_markers_output_dir,
			"assay_stats_relpath": inputs.assay_stats_relpath,
			"channel_layouts_subdir": inputs.channel_layouts_subdir,
			"segment_traces_subdir": inputs.segment_traces_subdir,
			"concat_trace_relpath": inputs.concat_trace_relpath,
			"n_representative_channels": int(inputs.n_representative_channels),
			"concat_trace_n_reps": int(inputs.concat_trace_n_reps),
			"segment_trace_n_reps": int(inputs.segment_trace_n_reps),
			"plot_n_jobs": max(1, int(inputs.plot_n_jobs)),
			"trace_downsample_hz": inputs.trace_downsample_hz,
			"trace_max_points": _normalize_trace_max_points(inputs.trace_max_points),
			"observability_mode": _normalize_observability_mode(inputs.observability_mode),
			"observability_output_subdir": inputs.observability_output_subdir,
			"observability_save_run_manifest": bool(inputs.observability_save_run_manifest),
			"observability_save_event_timeline": bool(inputs.observability_save_event_timeline),
			"observability_save_environment": bool(inputs.observability_save_environment),
			"observability_save_artifact_inventory": bool(inputs.observability_save_artifact_inventory),
			"observability_save_stage_log": bool(inputs.observability_save_stage_log),
			"observability_stage_log_relpath": inputs.observability_stage_log_relpath,
			"temporal_resample_factor": inputs.temporal_resample_factor,
			"temporal_resample_rate_hz": inputs.temporal_resample_rate_hz,
			"temporal_resample_margin_ms": float(inputs.temporal_resample_margin_ms),
			"temporal_resample_dtype": inputs.temporal_resample_dtype,
			"save_recording": bool(inputs.save_recording),
			"overwrite_saved_recording": bool(inputs.overwrite_saved_recording),
			"save_concat_recording": bool(inputs.save_concat_recording),
			"save_segment_recordings": bool(inputs.save_segment_recordings),
			"save_chunk_duration": str(inputs.save_chunk_duration),
			"save_progress_bar": bool(inputs.save_progress_bar),
			"concat_save_n_jobs": (int(inputs.concat_save_n_jobs) if inputs.concat_save_n_jobs is not None else None),
			"segment_save_n_jobs": (int(inputs.segment_save_n_jobs) if inputs.segment_save_n_jobs is not None else None),
			"print_n_jobs_used": bool(inputs.print_n_jobs_used),
			"phases": _json_ready(asdict(inputs.phases)),
		},
		"phase_timing_s": phase_timing_s,
		"phase_summaries": {name: str(payload["summary_json"]) for name, payload in phase_summaries.items()},
		"outputs": outputs,
	}

	_write_json(paths.stage_summary_json, _json_ready(summary_payload))

	if _is_observability_enabled(inputs):
		try:
			obs_outputs = _write_observability_artifacts(
				inputs=inputs,
				well_out_dir=paths.well_out_dir,
				preprocess_out_dir=paths.preprocess_out_dir,
				legacy_out_dir=legacy_out_dir,
				outputs=outputs,
				summary_json=paths.stage_summary_json,
				event_records=event_records,
				common_electrodes=[int(x) for x in common_electrodes],
				stage_status="ok",
				stage_started_utc=stage_started_utc,
				stage_elapsed_s=stage_elapsed_s,
				stage_error=None,
				stage_log_source=(stage_log_source if _safe_path_exists(stage_log_source) else None),
			)
			if obs_outputs:
				outputs.update(obs_outputs)
				summary_payload["outputs"] = outputs
				summary_payload["observability"] = {
					"enabled": True,
					"mode": _normalize_observability_mode(inputs.observability_mode),
				}
				_write_json(paths.stage_summary_json, _json_ready(summary_payload))
		except Exception:
			LOGGER.warning("Failed writing preprocess observability artifacts", exc_info=True)

	return PreprocessResult(
		well_out_dir=paths.well_out_dir,
		preprocess_out_dir=paths.preprocess_out_dir,
		summary_json=paths.stage_summary_json,
		outputs=outputs,
	)


def _run_preprocess_selected_phase(inputs: PreprocessInputs, *, selected_phase: str) -> dict[str, Any]:
	build_plot_cfg = _resolve_effective_plot_config(inputs, selected_phase=selected_phase)
	paths = _resolve_preprocess_paths(inputs, plot_cfg=build_plot_cfg)
	paths.preprocess_out_dir.mkdir(parents=True, exist_ok=True)
	_clear_self_referential_symlink(paths.stage_log_source)
	scratch_usage_key = _acquire_scratch_input_usage(inputs, selected_phase=selected_phase)
	scratch_usage_released = False
	try:
		phase_flags, outputs, phase_results, common_electrodes, phase_timing_s, _cfg_summary = _execute_preprocess_phase_work(
			inputs=inputs,
			paths=paths,
			selected_phase=selected_phase,
		)
		_ = phase_flags
		wipe_src_scratch_payload: dict[str, Any] | None = None
		if _wipe_src_scratch_phase_requested(inputs, selected_phase=selected_phase):
			wipe_src_scratch_payload = _execute_wipe_src_scratch_phase(inputs, usage_key=scratch_usage_key)
			scratch_usage_released = True
		phase_summaries = _write_enabled_phase_summaries(
			inputs=inputs,
			paths=paths,
			phase_results=phase_results,
			outputs=outputs,
			common_electrodes=common_electrodes,
			phase_timing_s=phase_timing_s,
			selected_phase=selected_phase,
			wipe_src_scratch_payload=wipe_src_scratch_payload,
		)
	finally:
		if scratch_usage_key is not None and not bool(scratch_usage_released):
			_release_scratch_input_usage(scratch_usage_key)
	requested_phase_key = {
		"build_preprocessed_recording": "preprocess_segments",
		"save_rec_metadata": "save_rec_metadata",
		"wipe_src_scratch": "wipe_src_scratch",
		"save_segment_recordings": "preprocess_segments",
		"preprocess_segments": "preprocess_segments",
		"save_concatenated_recording": "concatenate_preprocessed_recordings",
		"concatenate_preprocessed_recordings": "concatenate_preprocessed_recordings",
		"copy_src_to_scratch": "copy_src_to_scratch",
		"save_common_electrodes": "save_common_electrodes",
	}.get(selected_phase, selected_phase)
	payload = dict(phase_summaries.get(requested_phase_key, {}))
	if not payload:
		raise RuntimeError(f"No preprocess phase summary was written for phase '{selected_phase}'")
	payload.setdefault("preprocess_out_dir", str(paths.preprocess_out_dir))
	payload.setdefault("well_out_dir", str(paths.well_out_dir))
	return payload


def run_preprocess_copy_src_to_scratch_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="copy_src_to_scratch")


def run_preprocess_save_rec_metadata_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="save_rec_metadata")


def run_preprocess_wipe_src_scratch_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="wipe_src_scratch")


def run_preprocess_preprocess_segments_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="preprocess_segments")


def run_preprocess_concatenate_preprocessed_recordings_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="concatenate_preprocessed_recordings")


def run_preprocess_build_preprocessed_recording_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return run_preprocess_preprocess_segments_phase(inputs)


def run_preprocess_save_concatenated_recording_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return run_preprocess_concatenate_preprocessed_recordings_phase(inputs)


def run_preprocess_save_segment_recordings_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return run_preprocess_preprocess_segments_phase(inputs)


def run_preprocess_save_common_electrodes_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="save_common_electrodes")
