from __future__ import annotations

import contextlib
import datetime as dt
import getpass
import io
import json
import logging
import math
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
from axon_reconstructor.pipeline.stg1_preprocessing.h5_helpers import (
	_print_assay_settings,
	_print_data_store_start_stop_durations,
	_read_well_rec_frame_nos_and_trigger_settings,
	_tee_stdout_to_file,
)
from axon_reconstructor.pipeline.stg1_preprocessing.planning import build_preprocess_plan

from ...shared.sampling import read_maxwell_sampling_frequency_hz
from .core import (
	run_build_preprocessed_recording_core,
	run_save_common_electrodes_core,
	run_save_concatenated_recording_core,
	run_save_segment_recordings_core,
)
from .models.inputs import (
	PreprocessConcatenateRecordingsPhaseConfig,
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


@dataclass(frozen=True)
class _RecordingMetadataPathSet:
	segment_epochs_path: Path
	contiguous_epochs_path: Path
	sampling_metadata_path: Path
	assay_stats_path: Path


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


def _as_positive_float_or_none(value: Any) -> float | None:
	try:
		out = float(value)
	except Exception:
		return None
	if not math.isfinite(out) or out <= 0.0:
		return None
	return out


def _resolve_path_from_root(*, root_dir: Path, raw: str | None, default: str) -> Path:
	token = str(raw).strip() if raw is not None else ""
	if not token:
		token = str(default)
	path = Path(token).expanduser()
	if path.is_absolute():
		return Path(os.path.abspath(str(path)))
	return Path(os.path.abspath(str(root_dir / path)))


def _resolve_stream_path_from_root(*, root_dir: Path, raw: str | None, default: str, stream_id: str) -> Path:
	token = str(raw).strip() if raw is not None else ""
	if not token:
		token = str(default)
	try:
		token = token.format(stream_id=str(stream_id))
	except Exception:
		pass
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


def _is_concatenate_recordings_phase(selected_phase: str | None) -> bool:
	return selected_phase in {
		"concatenate_recordings",
		"concatenate_preprocessed_recordings",
		"save_concatenated_recording",
	}


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
		phase_plot=inputs.phases.concatenate_recordings.plot,
		input_plot=input_plot,
		default_plot=PreprocessConcatenateRecordingsPhaseConfig().plot,
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
	elif _is_concatenate_recordings_phase(selected_phase):
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
			(inputs.phases.concatenate_recordings.enabled and concat_plot.concat_trace)
			or (
				not inputs.phases.concatenate_recordings.enabled
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
			raw=inputs.phases.concatenate_recordings.rel_output_root,
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
		_resolve_jobs(inputs.phases.concatenate_recordings.outputs.concat_save_n_jobs, fallback=int(inputs.n_jobs)),
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
	if _is_concatenate_recordings_phase(selected_phase):
		return {
			"build_required": True,
			"save_concatenated_recording": True,
			"save_segment_recordings": False,
			"save_common_electrodes": bool(inputs.phases.concatenate_recordings.save_common_electrodes.enabled),
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
	save_concat_enabled = bool(inputs.phases.concatenate_recordings.enabled) and bool(inputs.save_concat_recording)
	save_segment_enabled = bool(inputs.phases.preprocess_segments.enabled) and bool(inputs.save_segment_recordings)
	save_common_enabled = bool(inputs.phases.concatenate_recordings.save_common_electrodes.enabled) and bool(
		inputs.save_recording or save_concat_enabled or save_segment_enabled
	)
	build_required = bool(save_concat_enabled or save_segment_enabled or save_common_enabled)
	return {
		"build_required": bool(build_required),
		"save_concatenated_recording": bool(save_concat_enabled),
		"save_segment_recordings": bool(save_segment_enabled),
		"save_common_electrodes": bool(save_common_enabled),
	}


def _concatenate_segment_recordings(segment_recordings: list[Any]) -> Any:
	if not segment_recordings:
		raise RuntimeError("No segment recordings available for concatenation")
	if len(segment_recordings) == 1:
		return segment_recordings[0]
	try:
		import spikeinterface.full as si  # type: ignore[import-not-found]
	except Exception as exc:
		raise RuntimeError(f"SpikeInterface import failed while concatenating segment recordings: {exc}") from exc
	return si.concatenate_recordings(segment_recordings)


def _resolve_concatenated_recording_for_save(
	*,
	built_recording: Any,
	build_artifacts: dict[str, Any],
	inputs: PreprocessInputs,
) -> tuple[Any, str, int]:
	if bool(inputs.phases.concatenate_recordings.concatenate_preprocessed_recordings):
		segment_recordings = list(build_artifacts.get("segment_recordings_preprocessed_concat", []) or [])
		if not segment_recordings:
			segment_recordings = list(build_artifacts.get("segment_recordings_preprocessed", []) or [])
		return built_recording, "preprocessed", int(len(segment_recordings))

	raw_segment_recordings = list(build_artifacts.get("segment_recordings_raw_concat", []) or [])
	if not raw_segment_recordings:
		raw_segment_recordings = list(build_artifacts.get("segment_recordings_raw", []) or [])
	if not raw_segment_recordings:
		raise RuntimeError(
			"concatenate_recordings configured concatenate_preprocessed_recordings=false, but no raw segment recordings were available"
		)
	return _concatenate_segment_recordings(raw_segment_recordings), "raw", int(len(raw_segment_recordings))


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


def _resolve_recording_metadata_paths(*, inputs: PreprocessInputs, well_out_dir: Path) -> _RecordingMetadataPathSet:
	return _RecordingMetadataPathSet(
		segment_epochs_path=_resolve_path_from_root(
			root_dir=well_out_dir,
			raw=inputs.phases.save_rec_metadata.segment_epochs_relpath,
			default="segment_epochs.json",
		),
		contiguous_epochs_path=_resolve_path_from_root(
			root_dir=well_out_dir,
			raw=inputs.phases.save_rec_metadata.contiguous_epochs_relpath,
			default="continuous_epochs.json",
		),
		sampling_metadata_path=_resolve_path_from_root(
			root_dir=well_out_dir,
			raw=inputs.phases.save_rec_metadata.sampling_metadata_relpath,
			default="sampling_rate_metadata.json",
		),
		assay_stats_path=_resolve_stream_path_from_root(
			root_dir=well_out_dir,
			raw=inputs.assay_stats_relpath,
			default="assay_stats_{stream_id}.txt",
			stream_id=str(inputs.stream_id),
		),
	)


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


def _list_maxwell_recording_names(*, h5_path: Path, stream_id: str) -> tuple[list[str], str | None]:
	try:
		import h5py
	except Exception as exc:
		return [], f"h5py import failed: {exc}"

	try:
		with h5py.File(str(Path(h5_path).expanduser().resolve()), "r") as h5:
			if "wells" not in h5:
				return [], "Missing /wells group in Maxwell H5"
			wells = h5["wells"]
			if str(stream_id) not in wells:
				return [], f"Stream '{stream_id}' not found under /wells"
			return [str(name) for name in list(wells[str(stream_id)].keys())], None
	except Exception as exc:
		return [], f"Failed reading stream segments from Maxwell H5: {exc}"


def _try_get_spikeinterface_segment_infos(
	*,
	h5_path: Path,
	stream_id: str,
	rec_names: list[str],
	suppress_h5_plugin_messages: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
	try:
		import spikeinterface.extractors as se  # type: ignore[import-not-found]
	except Exception as exc:
		return [], [f"SpikeInterface import failed while reading per-segment metadata: {exc}"]

	segment_infos: list[dict[str, Any]] = []
	warnings: list[str] = []
	for segment_index, rec_name in enumerate(rec_names):
		try:
			with contextlib.ExitStack() as stack:
				if bool(suppress_h5_plugin_messages):
					suppressed_stream = io.StringIO()
					stack.enter_context(contextlib.redirect_stdout(suppressed_stream))
					stack.enter_context(contextlib.redirect_stderr(suppressed_stream))
				try:
					recording = se.read_maxwell(h5_path, stream_id=stream_id, rec_name=rec_name)
				except TypeError:
					recording = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
		except Exception as exc:
			warnings.append(f"SpikeInterface read_maxwell failed for segment '{rec_name}': {exc}")
			continue

		entry: dict[str, Any] = {
			"segment_index": int(segment_index),
			"rec_name": str(rec_name),
			"source": "spikeinterface",
		}
		try:
			entry["sampling_frequency_hz"] = float(recording.get_sampling_frequency())
		except Exception:
			pass
		try:
			entry["num_channels"] = int(recording.get_num_channels())
		except Exception:
			pass
		try:
			get_num_samples = getattr(recording, "get_num_samples", None)
			if callable(get_num_samples):
				entry["num_samples"] = int(get_num_samples())
			else:
				entry["num_samples"] = int(recording.get_num_frames())
		except Exception:
			pass
		fs_hz = _as_positive_float_or_none(entry.get("sampling_frequency_hz", None))
		if fs_hz is not None and entry.get("num_samples", None) is not None:
			try:
				entry["duration_samples_s"] = float(int(entry["num_samples"]) / float(fs_hz))
			except Exception:
				pass
		segment_infos.append(entry)
	return segment_infos, warnings


def _infer_epoch_divisor_to_seconds(values: list[int]) -> tuple[float, str]:
	if not values:
		return 1.0, "s"
	vmax = max(int(v) for v in values)
	if vmax >= 10**18:
		return 1e9, "ns"
	if vmax >= 10**15:
		return 1e6, "us"
	if vmax >= 10**12:
		return 1e3, "ms"
	return 1.0, "s"


def _format_epoch_iso_utc(value: Any, *, divisor: float) -> str | None:
	try:
		if value is None:
			return None
		return dt.datetime.fromtimestamp(float(value) / float(divisor), tz=dt.timezone.utc).isoformat()
	except Exception:
		return None


def _build_recording_metadata_phase_payload(inputs: PreprocessInputs) -> dict[str, Any]:
	source_h5_path = Path(inputs.source_h5_path or inputs.h5_path)
	resolved_h5_path = Path(inputs.h5_path)
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	metadata_paths = _resolve_recording_metadata_paths(inputs=inputs, well_out_dir=well_out_dir)
	segment_epochs_path = metadata_paths.segment_epochs_path
	contiguous_epochs_path = metadata_paths.contiguous_epochs_path
	sampling_metadata_path = metadata_paths.sampling_metadata_path
	assay_stats_path = metadata_paths.assay_stats_path
	recording_info, recording_info_error = _try_get_spikeinterface_recording_info(
		h5_path=resolved_h5_path,
		stream_id=str(inputs.stream_id),
		suppress_h5_plugin_messages=bool(inputs.logging_suppress_h5_plugin_messages),
	)
	rec_names, rec_names_error = _list_maxwell_recording_names(
		h5_path=resolved_h5_path,
		stream_id=str(inputs.stream_id),
	)
	segment_info_entries, segment_info_warnings = _try_get_spikeinterface_segment_infos(
		h5_path=resolved_h5_path,
		stream_id=str(inputs.stream_id),
		rec_names=list(rec_names),
		suppress_h5_plugin_messages=bool(inputs.logging_suppress_h5_plugin_messages),
	)
	segment_info_by_name = {
		str(item.get("rec_name")): dict(item)
		for item in segment_info_entries
		if str(item.get("rec_name", "")).strip()
	}
	stream_sampling_hz = read_maxwell_sampling_frequency_hz(
		h5_path=resolved_h5_path,
		stream_id=str(inputs.stream_id),
	)
	metadata_warnings: list[str] = []
	if rec_names_error is not None:
		metadata_warnings.append(str(rec_names_error))
	metadata_warnings.extend(str(item) for item in segment_info_warnings if str(item).strip())

	segment_epochs: list[dict[str, Any]] = []
	contiguous_epochs: list[dict[str, Any]] = []
	raw_epoch_values: list[int] = []
	segment_start_seconds_by_name: dict[str, float] = {}
	try:
		import numpy as np
	except Exception as exc:
		metadata_warnings.append(f"numpy import failed while reading recording metadata: {exc}")
		np = None  # type: ignore[assignment]

	if np is not None:
		for segment_index, rec_name in enumerate(rec_names):
			segment_si_info = dict(segment_info_by_name.get(str(rec_name), {}))
			sampling_hz = _as_positive_float_or_none(segment_si_info.get("sampling_frequency_hz", None))
			if sampling_hz is None:
				sampling_hz = _as_positive_float_or_none(stream_sampling_hz)
			try:
				segment_info = _read_well_rec_frame_nos_and_trigger_settings(
					h5_path=resolved_h5_path,
					stream_id=str(inputs.stream_id),
					rec_name=str(rec_name),
				)
			except Exception as exc:
				metadata_warnings.append(f"Failed reading frame/timing metadata for segment '{rec_name}': {exc}")
				continue

			start_raw = int(segment_info["start_ms"])
			stop_raw = int(segment_info["stop_ms"])
			raw_epoch_values.extend([int(start_raw), int(stop_raw)])
			frame_nos = np.asarray(segment_info.get("frame_nos", []), dtype=np.int64)
			frame_count = int(frame_nos.size)
			segment_payload: dict[str, Any] = {
				"segment_index": int(segment_index),
				"rec_name": str(rec_name),
				"start_timestamp_raw": int(start_raw),
				"stop_timestamp_raw": int(stop_raw),
				"n_samples": int(frame_count),
			}
			if frame_count > 0:
				segment_payload["frame_no_start"] = int(frame_nos[0])
				segment_payload["frame_no_end"] = int(frame_nos[-1])
			if sampling_hz is not None:
				segment_payload["sampling_frequency_hz"] = float(sampling_hz)
				segment_payload["duration_samples_s"] = float(frame_count / float(sampling_hz))
				if frame_count > 0:
					segment_payload["frame_span_s"] = float((int(frame_nos[-1]) - int(frame_nos[0])) / float(sampling_hz))
			for key in ("triggered", "trigger_pre", "trigger_post", "trigger_minamp", "trigger_maxamp"):
				value = segment_info.get(key, None)
				if value is not None:
					segment_payload[str(key)] = value
			if segment_si_info.get("num_channels", None) is not None:
				segment_payload["num_channels"] = int(segment_si_info["num_channels"])
			segment_epochs.append(segment_payload)

			if frame_count <= 0:
				continue

			diffs = np.diff(frame_nos)
			split_points = np.flatnonzero(diffs != 1) + 1
			run_starts = np.concatenate(([0], split_points))
			run_ends = np.concatenate((split_points, [frame_count]))
			frame0 = int(frame_nos[0])
			for epoch_index, (run_start, run_end) in enumerate(zip(run_starts, run_ends, strict=False)):
				run_start_i = int(run_start)
				run_end_i = int(run_end)
				if run_end_i <= run_start_i:
					continue
				epoch_payload: dict[str, Any] = {
					"segment_index": int(segment_index),
					"rec_name": str(rec_name),
					"epoch_index": int(epoch_index),
					"segment_start_sample": int(run_start_i),
					"segment_end_sample": int(run_end_i),
					"n_samples": int(run_end_i - run_start_i),
					"frame_no_start": int(frame_nos[run_start_i]),
					"frame_no_end": int(frame_nos[run_end_i - 1]),
				}
				if sampling_hz is not None:
					segment_relative_start_s = float((int(frame_nos[run_start_i]) - frame0) / float(sampling_hz))
					segment_relative_end_s = float(((int(frame_nos[run_end_i - 1]) - frame0) + 1) / float(sampling_hz))
					epoch_payload["segment_relative_start_s"] = segment_relative_start_s
					epoch_payload["segment_relative_end_s"] = segment_relative_end_s
					epoch_payload["duration_s"] = float(max(0.0, segment_relative_end_s - segment_relative_start_s))
				contiguous_epochs.append(epoch_payload)

	divisor_to_seconds, epoch_unit = _infer_epoch_divisor_to_seconds(raw_epoch_values)
	timestamp_unit = f"{epoch_unit}_since_epoch"
	segment_epoch_by_name = {str(item.get("rec_name")): item for item in segment_epochs}
	for segment_payload in segment_epochs:
		segment_payload["timestamp_unit"] = timestamp_unit
		start_raw = segment_payload.get("start_timestamp_raw", None)
		stop_raw = segment_payload.get("stop_timestamp_raw", None)
		if start_raw is not None:
			start_s = float(start_raw) / float(divisor_to_seconds)
			segment_payload["start_time_seconds_since_epoch"] = start_s
			segment_payload["start_time_utc"] = _format_epoch_iso_utc(start_raw, divisor=float(divisor_to_seconds))
			segment_start_seconds_by_name[str(segment_payload.get("rec_name", ""))] = start_s
		if stop_raw is not None:
			stop_s = float(stop_raw) / float(divisor_to_seconds)
			segment_payload["stop_time_seconds_since_epoch"] = stop_s
			segment_payload["stop_time_utc"] = _format_epoch_iso_utc(stop_raw, divisor=float(divisor_to_seconds))
		if start_raw is not None and stop_raw is not None:
			segment_payload["duration_wall_clock_s"] = float((float(stop_raw) - float(start_raw)) / float(divisor_to_seconds))

	for epoch_payload in contiguous_epochs:
		epoch_payload["timestamp_unit"] = timestamp_unit
		segment_name = str(epoch_payload.get("rec_name", ""))
		segment_start_s = segment_start_seconds_by_name.get(segment_name, None)
		segment_relative_start_s = epoch_payload.get("segment_relative_start_s", None)
		segment_relative_end_s = epoch_payload.get("segment_relative_end_s", None)
		if segment_start_s is not None and segment_relative_start_s is not None:
			epoch_start_s = float(segment_start_s + float(segment_relative_start_s))
			epoch_payload["start_time_seconds_since_epoch"] = epoch_start_s
			epoch_payload["start_time_utc"] = _format_epoch_iso_utc(
				epoch_start_s * float(divisor_to_seconds),
				divisor=float(divisor_to_seconds),
			)
		if segment_start_s is not None and segment_relative_end_s is not None:
			epoch_end_s = float(segment_start_s + float(segment_relative_end_s))
			epoch_payload["end_time_seconds_since_epoch"] = epoch_end_s
			epoch_payload["end_time_utc"] = _format_epoch_iso_utc(
				epoch_end_s * float(divisor_to_seconds),
				divisor=float(divisor_to_seconds),
			)

	sampling_segments: list[dict[str, Any]] = []
	distinct_sampling_rates_hz: list[float] = []
	for segment_index, rec_name in enumerate(rec_names):
		segment_si_info = dict(segment_info_by_name.get(str(rec_name), {}))
		sampling_hz = _as_positive_float_or_none(segment_si_info.get("sampling_frequency_hz", None))
		sampling_source = str(segment_si_info.get("source", "spikeinterface")) if sampling_hz is not None else None
		if sampling_hz is None:
			sampling_hz = _as_positive_float_or_none(stream_sampling_hz)
			if sampling_hz is not None:
				sampling_source = "data_store_or_attrs"
		segment_payload = {
			"segment_index": int(segment_index),
			"rec_name": str(rec_name),
			"sampling_frequency_hz": (None if sampling_hz is None else float(sampling_hz)),
			"source": sampling_source,
		}
		segment_epoch_payload = segment_epoch_by_name.get(str(rec_name), None)
		if segment_epoch_payload is not None:
			for key in (
				"n_samples",
				"frame_no_start",
				"frame_no_end",
				"duration_samples_s",
				"duration_wall_clock_s",
			):
				if key in segment_epoch_payload:
					segment_payload[str(key)] = segment_epoch_payload[key]
		if segment_si_info.get("num_channels", None) is not None:
			segment_payload["num_channels"] = int(segment_si_info["num_channels"])
		if sampling_hz is not None:
			distinct_sampling_rates_hz.append(float(sampling_hz))
		sampling_segments.append(segment_payload)

	distinct_sampling_rates_hz = sorted({round(float(value), 9) for value in distinct_sampling_rates_hz})
	sampling_summary = {
		"stream_sampling_frequency_hz": (
			None if _as_positive_float_or_none(stream_sampling_hz) is None else float(stream_sampling_hz)
		),
		"recording_info_sampling_frequency_hz": (
			None
			if _as_positive_float_or_none(recording_info.get("sampling_frequency_hz", None)) is None
			else float(recording_info["sampling_frequency_hz"])
		),
		"distinct_sampling_frequency_hz": [float(value) for value in distinct_sampling_rates_hz],
		"all_segments_match": bool(len(distinct_sampling_rates_hz) <= 1),
	}

	segment_epochs_payload = {
		"h5_path": str(resolved_h5_path),
		"source_h5_path": str(source_h5_path),
		"stream_id": str(inputs.stream_id),
		"timestamp_unit": timestamp_unit,
		"segment_count": int(len(segment_epochs)),
		"segments": segment_epochs,
	}
	if metadata_warnings:
		segment_epochs_payload["warnings"] = list(metadata_warnings)

	contiguous_epochs_payload = {
		"h5_path": str(resolved_h5_path),
		"source_h5_path": str(source_h5_path),
		"stream_id": str(inputs.stream_id),
		"timestamp_unit": timestamp_unit,
		"segment_count": int(len(segment_epochs)),
		"contiguous_epoch_count": int(len(contiguous_epochs)),
		"epochs": contiguous_epochs,
	}
	if metadata_warnings:
		contiguous_epochs_payload["warnings"] = list(metadata_warnings)

	sampling_metadata_payload = {
		"h5_path": str(resolved_h5_path),
		"source_h5_path": str(source_h5_path),
		"stream_id": str(inputs.stream_id),
		"segment_count": int(len(sampling_segments)),
		"sampling_summary": sampling_summary,
		"segments": sampling_segments,
	}
	if metadata_warnings:
		sampling_metadata_payload["warnings"] = list(metadata_warnings)

	_write_json(segment_epochs_path, _json_ready(segment_epochs_payload))
	_write_json(contiguous_epochs_path, _json_ready(contiguous_epochs_payload))
	_write_json(sampling_metadata_path, _json_ready(sampling_metadata_payload))
	try:
		with _tee_stdout_to_file(assay_stats_path) as written_path:
			print(
				f"[axon_reconstructor][DEBUG] assay_stats file: {written_path} "
				f"(generated {dt.datetime.now(dt.timezone.utc).isoformat()})",
				flush=True,
			)
			print(
				f"[axon_reconstructor][DEBUG] assay_stats context: h5={resolved_h5_path} stream={inputs.stream_id}",
				flush=True,
			)
			_print_assay_settings(h5_path=resolved_h5_path)
			_print_data_store_start_stop_durations(
				h5_path=resolved_h5_path,
				target_stream_id=str(inputs.stream_id),
			)
	except Exception as exc:
		metadata_warnings.append(f"Failed writing assay stats artifact '{assay_stats_path}': {exc}")
	payload: dict[str, Any] = {
		"phase": "save_rec_metadata",
		"source_h5_path": str(source_h5_path),
		"resolved_h5_path": str(resolved_h5_path),
		"copied_to_scratch": bool(inputs.copied_to_scratch),
		"verbose": bool(inputs.phases.save_rec_metadata.verbose),
		"source_path_context": _parse_recording_context_from_path(source_h5_path),
		"source_file": (
			_collect_path_details(source_h5_path)
			if _paths_equal_no_resolve(left=source_h5_path, right=resolved_h5_path)
			else {
				"path": str(source_h5_path),
				"metadata_source": "path_only_prefer_resolved_h5_path",
			}
		),
		"recording_info": recording_info,
		"segment_count": int(len(segment_epochs)),
		"contiguous_epoch_count": int(len(contiguous_epochs)),
		"segment_epochs_json": str(segment_epochs_path),
		"contiguous_epochs_json": str(contiguous_epochs_path),
		"sampling_metadata_json": str(sampling_metadata_path),
		"assay_stats_txt": str(assay_stats_path),
		"sampling_summary": sampling_summary,
	}
	payload["resolved_file"] = _collect_path_details(resolved_h5_path)
	if not _paths_equal_no_resolve(left=source_h5_path, right=resolved_h5_path):
		payload["resolved_path_context"] = _parse_recording_context_from_path(resolved_h5_path)
	if recording_info_error is not None:
		payload["recording_info_error"] = str(recording_info_error)
	if metadata_warnings:
		payload["metadata_warnings"] = list(metadata_warnings)
	if bool(inputs.phases.save_rec_metadata.verbose):
		payload["segment_epochs_preview"] = list(segment_epochs[: min(len(segment_epochs), 10)])
		payload["contiguous_epochs_preview"] = list(contiguous_epochs[: min(len(contiguous_epochs), 20)])
		payload["sampling_segments"] = list(sampling_segments)
		LOGGER.info(
			"Saved recording metadata stream_id=%s segment_count=%d contiguous_epoch_count=%d segment_epochs=%s contiguous_epochs=%s sampling_metadata=%s assay_stats=%s",
			str(inputs.stream_id),
			int(len(segment_epochs)),
			int(len(contiguous_epochs)),
			segment_epochs_path,
			contiguous_epochs_path,
			sampling_metadata_path,
			assay_stats_path,
		)
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
			"concatenate_recordings": {
				"concatenate_preprocessed_recordings": bool(inputs.phases.concatenate_recordings.concatenate_preprocessed_recordings),
				"segment_source": (
					"preprocessed"
					if bool(inputs.phases.concatenate_recordings.concatenate_preprocessed_recordings)
					else "raw"
				),
				"save_chunk_duration": str(inputs.phases.concatenate_recordings.outputs.save_chunk_duration),
				"save_progress_bar": bool(inputs.phases.concatenate_recordings.outputs.save_progress_bar),
				"concat_save_n_jobs": int(concat_jobs),
				"print_n_jobs_used": bool(inputs.phases.concatenate_recordings.outputs.print_n_jobs_used),
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
	saved_assay_stats_path: Path | None,
) -> tuple[dict[str, bool], dict[str, str], dict[str, Any], list[int], dict[str, float], dict[str, Any]]:
	_validate_copy_phase_requirements(inputs, selected_phase=selected_phase)
	_validate_wipe_phase_requirements(inputs, selected_phase=selected_phase)
	phase_flags = _resolve_phase_execution_flags(inputs, selected_phase=selected_phase)
	outputs: dict[str, str] = {
		"legacy.preprocess_out_dir": str(paths.legacy_out_dir),
		"concatenated_recording_dir": str(paths.recording_dir),
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
		saved_assay_stats_path=saved_assay_stats_path,
		require_saved_assay_stats=True,
		phase_dividers=bool(inputs.logging_phase_dividers),
		emit_phase_dividers_to_stdout=bool(inputs.logging_subphase_dividers_to_stdout),
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
		concat_recording, concat_segment_source, concat_segment_count = _resolve_concatenated_recording_for_save(
			built_recording=multirecording,
			build_artifacts=build_artifacts,
			inputs=inputs,
		)
		phase_results["save_concatenated_recording"] = run_save_concatenated_recording_core(
			multirecording=concat_recording,
			recording_dir=paths.recording_dir,
			overwrite_saved_recording=bool(inputs.overwrite_saved_recording),
			n_jobs=concat_jobs,
			chunk_duration=str(inputs.phases.concatenate_recordings.outputs.save_chunk_duration),
			progress_bar=bool(inputs.phases.concatenate_recordings.outputs.save_progress_bar),
			logger=phase_logger,
		)
		phase_results["save_concatenated_recording"].update(
			{
				"segment_source": str(concat_segment_source),
				"concatenate_preprocessed_recordings": bool(
					inputs.phases.concatenate_recordings.concatenate_preprocessed_recordings
				),
				"source_segment_count": int(concat_segment_count),
			}
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
	recording_metadata_payload: dict[str, Any] | None = None,
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
		recording_metadata_payload = (
			dict(recording_metadata_payload)
			if recording_metadata_payload is not None
			else _build_recording_metadata_phase_payload(inputs)
		)
		recording_metadata_outputs = {
			"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
			"resolved_h5_path": str(inputs.h5_path),
		}
		for output_key in ("segment_epochs_json", "contiguous_epochs_json", "sampling_metadata_json", "assay_stats_txt"):
			value = recording_metadata_payload.get(output_key, None)
			if value is None:
				continue
			recording_metadata_outputs[str(output_key)] = str(value)
		outputs.update(recording_metadata_outputs)
		phase_summaries["save_rec_metadata"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="save_rec_metadata",
			summary_json_relpath=str(inputs.phases.save_rec_metadata.summary_json_relpath),
			outputs=recording_metadata_outputs,
			extra_payload=recording_metadata_payload,
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
		_is_concatenate_recordings_phase(selected_phase)
		or (selected_phase is None and bool(inputs.phases.concatenate_recordings.enabled))
	):
		phase_summaries["concatenate_recordings"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="concatenate_recordings",
			summary_json_relpath=str(inputs.phases.concatenate_recordings.summary_json_relpath),
			outputs={
				"concatenated_recording_dir": outputs["concatenated_recording_dir"],
				"preprocessed_recording_dir": outputs["preprocessed_recording_dir"],
			},
			extra_payload=dict(phase_results["save_concatenated_recording"]),
		)
	if "save_common_electrodes" in phase_results and (
		selected_phase == "save_common_electrodes"
		or _is_concatenate_recordings_phase(selected_phase)
		or (
			selected_phase is None
			and bool(inputs.phases.concatenate_recordings.save_common_electrodes.enabled)
		)
	):
		phase_summaries["save_common_electrodes"] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name="save_common_electrodes",
			summary_json_relpath=str(inputs.phases.concatenate_recordings.save_common_electrodes.summary_json_relpath),
			outputs={"common_electrodes_path": outputs["common_electrodes_path"]},
			extra_payload=dict(phase_results["save_common_electrodes"]),
		)
	for phase_name, payload in phase_summaries.items():
		outputs[f"{phase_name}_summary_json"] = str(payload["summary_json"])
	if "concatenate_recordings" in phase_summaries:
		outputs["concatenate_preprocessed_recordings_summary_json"] = str(
			phase_summaries["concatenate_recordings"]["summary_json"]
		)
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
	recording_metadata_paths = _resolve_recording_metadata_paths(inputs=inputs, well_out_dir=paths.well_out_dir)
	recording_metadata_payload: dict[str, Any] | None = None

	try:
		if _recording_metadata_phase_requested(inputs, selected_phase=None):
			recording_metadata_payload = _build_recording_metadata_phase_payload(inputs)
		phase_flags, outputs, phase_results, common_electrodes, phase_timing_s, _cfg_summary = _execute_preprocess_phase_work(
			inputs=inputs,
			paths=paths,
			selected_phase=None,
			saved_assay_stats_path=recording_metadata_paths.assay_stats_path,
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
			recording_metadata_payload=recording_metadata_payload,
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
			"logging_subphase_dividers_to_stdout": bool(inputs.logging_subphase_dividers_to_stdout),
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
	recording_metadata_paths = _resolve_recording_metadata_paths(inputs=inputs, well_out_dir=paths.well_out_dir)
	recording_metadata_payload: dict[str, Any] | None = None
	try:
		if _recording_metadata_phase_requested(inputs, selected_phase=selected_phase):
			recording_metadata_payload = _build_recording_metadata_phase_payload(inputs)
		phase_flags, outputs, phase_results, common_electrodes, phase_timing_s, _cfg_summary = _execute_preprocess_phase_work(
			inputs=inputs,
			paths=paths,
			selected_phase=selected_phase,
			saved_assay_stats_path=recording_metadata_paths.assay_stats_path,
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
			recording_metadata_payload=recording_metadata_payload,
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
		"save_concatenated_recording": "concatenate_recordings",
		"concatenate_recordings": "concatenate_recordings",
		"concatenate_preprocessed_recordings": "concatenate_recordings",
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


def run_preprocess_concatenate_recordings_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="concatenate_recordings")


def run_preprocess_concatenate_preprocessed_recordings_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return run_preprocess_concatenate_recordings_phase(inputs)


def run_preprocess_build_preprocessed_recording_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return run_preprocess_preprocess_segments_phase(inputs)


def run_preprocess_save_concatenated_recording_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return run_preprocess_concatenate_recordings_phase(inputs)


def run_preprocess_save_segment_recordings_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return run_preprocess_preprocess_segments_phase(inputs)


def run_preprocess_save_common_electrodes_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="save_common_electrodes")
