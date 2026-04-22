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

from .constants import PREPROCESS_OUTPUTS_DIRNAME
from .core.save_rec_metadata import (
	_print_assay_settings,
	_print_data_store_start_stop_durations,
	_read_well_rec_frame_nos_and_trigger_settings,
	_tee_stdout_to_file,
)

from ...shared.sampling import read_maxwell_sampling_frequency_hz
from .core import (
	load_common_electrodes,
	load_concat_manifest,
	load_recording_metadata,
	load_saved_recording,
	load_segment_manifest,
	read_json,
	run_concat_segments_core,
	run_copy_src_to_scratch_core,
	run_plot_concat_traces_core,
	run_plot_segment_traces_core,
	run_preprocess_segments_core,
	run_save_concatenated_recording_core,
	run_save_rec_metadata_core,
	run_save_segment_recordings_core,
	run_wipe_src_scratch_core,
)
from .models.inputs import (
	PreprocessConcatSegmentsPhaseConfig,
	PreprocessInputs,
	PreprocessPlotConcatTracesPhaseConfig,
	PreprocessPlotConfig,
	PreprocessPlotSegmentTracesPhaseConfig,
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
	concat_manifest_path: Path
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
	common_electrodes_path: Path


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


def _strip_preprocess_root_prefix(path: Path) -> Path:
	parts = list(path.parts)
	if parts and parts[0] == PREPROCESS_OUTPUTS_DIRNAME:
		if len(parts) == 1:
			return Path(".")
		return Path(*parts[1:])
	return path


def _resolve_preprocess_path_from_root(*, preprocess_out_dir: Path, raw: str | None, default: str) -> Path:
	token = str(raw).strip() if raw is not None else ""
	if not token:
		token = str(default)
	path = Path(token).expanduser()
	if path.is_absolute():
		return Path(os.path.abspath(str(path)))
	path = _strip_preprocess_root_prefix(path)
	return Path(os.path.abspath(str(preprocess_out_dir / path)))


def _resolve_preprocess_stream_path_from_root(
	*,
	preprocess_out_dir: Path,
	raw: str | None,
	default: str,
	stream_id: str,
) -> Path:
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
	path = _strip_preprocess_root_prefix(path)
	return Path(os.path.abspath(str(preprocess_out_dir / path)))


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


def _safe_file_size(path: Path) -> int:
	try:
		return int(path.stat().st_size)
	except Exception:
		return 0


def _is_nonempty_file(path: Path) -> bool:
	return bool(_safe_path_exists(path) and _safe_file_size(path) > 0)


def _is_complete_png(path: Path) -> bool:
	if not _is_nonempty_file(path):
		return False
	try:
		with open(path, "rb") as handle:
			header = handle.read(8)
			if header != b"\x89PNG\r\n\x1a\n":
				return False
			handle.seek(-12, os.SEEK_END)
			tail = handle.read(12)
		return tail == b"\x00\x00\x00\x00IEND\xaeB`\x82"
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


def _resolve_stage_log_source(*, inputs: PreprocessInputs, preprocess_out_dir: Path) -> Path:
	if inputs.logging_file_relpath is not None and str(inputs.logging_file_relpath).strip():
		return _resolve_preprocess_path_from_root(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.logging_file_relpath,
			default="logs/preprocess_pipeline.log",
		)
	return compute_pipeline_log_file(
		well_out_dir=preprocess_out_dir,
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


def _reset_preprocess_output_root_for_force_restart(
	*,
	inputs: PreprocessInputs,
	preprocess_out_dir: Path,
	selected_phase: str | None,
) -> bool:
	if not bool(inputs.force_restart) or selected_phase is not None:
		return False
	if not _safe_path_exists(preprocess_out_dir) and not preprocess_out_dir.is_symlink():
		return False
	if preprocess_out_dir.is_dir() and not preprocess_out_dir.is_symlink():
		shutil.rmtree(preprocess_out_dir)
	else:
		preprocess_out_dir.unlink()
	LOGGER.info("Force restart cleared preprocess output dir: %s", preprocess_out_dir)
	return True


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


def _is_concat_segments_phase(selected_phase: str | None) -> bool:
	return selected_phase == "concat_segments"


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
		phase_plot=inputs.phases.plot_segment_traces.plot,
		input_plot=input_plot,
		default_plot=PreprocessPlotSegmentTracesPhaseConfig().plot,
	)
	concat_plot = _apply_plot_input_fallback(
		phase_plot=inputs.phases.plot_concat_traces.plot,
		input_plot=input_plot,
		default_plot=PreprocessPlotConcatTracesPhaseConfig().plot,
	)
	primary = segment_plot
	secondary = concat_plot
	layouts = False
	segment_traces = False
	concat_trace = False

	if selected_phase == "preprocess_segments":
		primary = segment_plot
		secondary = concat_plot
		layouts = False
		segment_traces = False
		concat_trace = False
	elif selected_phase == "plot_segment_traces":
		primary = segment_plot
		secondary = concat_plot
		layouts = bool(segment_plot.layouts)
		segment_traces = bool(segment_plot.segment_traces)
		concat_trace = False
	elif _is_concat_segments_phase(selected_phase):
		primary = concat_plot
		secondary = segment_plot
		layouts = False
		segment_traces = False
		concat_trace = False
	elif selected_phase == "plot_concat_traces":
		primary = concat_plot
		secondary = segment_plot
		layouts = False
		segment_traces = False
		concat_trace = bool(concat_plot.concat_trace)
	else:
		primary = segment_plot if bool(inputs.phases.plot_segment_traces.enabled) else concat_plot
		secondary = concat_plot if primary is segment_plot else segment_plot
		layouts = bool(inputs.phases.plot_segment_traces.enabled and segment_plot.layouts)
		segment_traces = bool(inputs.phases.plot_segment_traces.enabled and segment_plot.segment_traces)
		concat_trace = bool(inputs.phases.plot_concat_traces.enabled and concat_plot.concat_trace)

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
			plot_output_dir = _resolve_preprocess_path_from_root(
				preprocess_out_dir=preprocess_out_dir,
				raw=plot_cfg.output_dir,
				default=".",
			)
		else:
			plot_output_dir = preprocess_out_dir
	epoch_markers_output_dir: Path | None
	if plot_cfg.epoch_markers_output_dir is not None and str(plot_cfg.epoch_markers_output_dir).strip():
		epoch_markers_output_dir = _resolve_preprocess_path_from_root(
			preprocess_out_dir=preprocess_out_dir,
			raw=plot_cfg.epoch_markers_output_dir,
			default=".",
		)
	else:
		epoch_markers_output_dir = preprocess_out_dir
	return _PreprocessPathSet(
		well_out_dir=well_out_dir,
		preprocess_out_dir=preprocess_out_dir,
		legacy_out_dir=preprocess_out_dir,
		recording_dir=_resolve_phase_output_dir(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.concat_segments.rel_output_root,
			default="concatenated_recording",
		),
		concat_manifest_path=_resolve_phase_output_dir(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.concat_segments.manifest_relpath,
			default="context/concat_segments_manifest.json",
		),
		common_electrodes_path=_resolve_preprocess_path_from_root(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.save_rec_metadata.common_electrodes_relpath,
			default="common_electrodes.npy",
		),
		per_segment_preprocessed_dir=_resolve_phase_output_dir(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.preprocess_segments.rel_output_root,
			default="preprocessed_segments",
		),
		per_segment_manifest_path=_resolve_phase_output_dir(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.preprocess_segments.rel_output_root,
			default="preprocessed_segments",
		) / "manifest.json",
		preprocess_config_path=preprocess_out_dir / "preprocess_config.json",
		stage_summary_json=preprocess_out_dir / "preprocess_summary.json",
		stage_log_source=_resolve_stage_log_source(inputs=inputs, preprocess_out_dir=preprocess_out_dir),
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


def _resolve_recording_metadata_paths(*, inputs: PreprocessInputs, preprocess_out_dir: Path) -> _RecordingMetadataPathSet:
	return _RecordingMetadataPathSet(
		segment_epochs_path=_resolve_preprocess_path_from_root(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.save_rec_metadata.segment_epochs_relpath,
			default="segment_epochs.json",
		),
		contiguous_epochs_path=_resolve_preprocess_path_from_root(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.save_rec_metadata.contiguous_epochs_relpath,
			default="continuous_epochs.json",
		),
		sampling_metadata_path=_resolve_preprocess_path_from_root(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.save_rec_metadata.sampling_metadata_relpath,
			default="sampling_rate_metadata.json",
		),
		assay_stats_path=_resolve_preprocess_stream_path_from_root(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.assay_stats_relpath,
			default="assay_stats_{stream_id}.txt",
			stream_id=str(inputs.stream_id),
		),
		common_electrodes_path=_resolve_preprocess_path_from_root(
			preprocess_out_dir=preprocess_out_dir,
			raw=inputs.phases.save_rec_metadata.common_electrodes_relpath,
			default="common_electrodes.npy",
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


def _normalize_requested_preprocess_phase(selected_phase: str | None) -> str | None:
	return str(selected_phase) if selected_phase is not None else None


def _phase_enabled(inputs: PreprocessInputs, phase_name: str) -> bool:
	if phase_name == "copy_src_to_scratch":
		return bool(inputs.phases.copy_src_to_scratch.enabled)
	if phase_name == "save_rec_metadata":
		return bool(inputs.phases.save_rec_metadata.enabled)
	if phase_name == "preprocess_segments":
		return bool(inputs.phases.preprocess_segments.enabled) and bool(inputs.save_segment_recordings)
	if phase_name == "plot_segment_traces":
		return bool(inputs.phases.plot_segment_traces.enabled)
	if phase_name == "concat_segments":
		return bool(inputs.phases.concat_segments.enabled) and bool(inputs.save_concat_recording)
	if phase_name == "plot_concat_traces":
		return bool(inputs.phases.plot_concat_traces.enabled)
	if phase_name == "wipe_src_scratch":
		return bool(inputs.phases.wipe_src_scratch.enabled)
	return False


def _summary_relpath_for_phase(inputs: PreprocessInputs, phase_name: str) -> str:
	if phase_name == "copy_src_to_scratch":
		return str(inputs.phases.copy_src_to_scratch.summary_json_relpath)
	if phase_name == "save_rec_metadata":
		return str(inputs.phases.save_rec_metadata.summary_json_relpath)
	if phase_name == "preprocess_segments":
		return str(inputs.phases.preprocess_segments.summary_json_relpath)
	if phase_name == "plot_segment_traces":
		return str(inputs.phases.plot_segment_traces.summary_json_relpath)
	if phase_name == "concat_segments":
		return str(inputs.phases.concat_segments.summary_json_relpath)
	if phase_name == "plot_concat_traces":
		return str(inputs.phases.plot_concat_traces.summary_json_relpath)
	if phase_name == "wipe_src_scratch":
		return str(inputs.phases.wipe_src_scratch.summary_json_relpath)
	return "context/preprocess_phase_summary.json"


def _summary_outputs_for_phase(
	*,
	phase_name: str,
	paths: _PreprocessPathSet,
	recording_metadata_paths: _RecordingMetadataPathSet,
	payload: dict[str, Any],
	outputs: dict[str, str],
) -> dict[str, str]:
	if phase_name == "copy_src_to_scratch":
		return {
			"source_h5_path": str(payload.get("source_h5_path", "")),
			"resolved_h5_path": str(payload.get("resolved_h5_path", "")),
		}
	if phase_name == "save_rec_metadata":
		return {
			"segment_epochs_json": str(recording_metadata_paths.segment_epochs_path),
			"contiguous_epochs_json": str(recording_metadata_paths.contiguous_epochs_path),
			"sampling_metadata_json": str(recording_metadata_paths.sampling_metadata_path),
			"assay_stats_txt": str(recording_metadata_paths.assay_stats_path),
			"common_electrodes_path": str(recording_metadata_paths.common_electrodes_path),
		}
	if phase_name == "preprocess_segments":
		return {
			"per_segment_preprocessed_dir": str(paths.per_segment_preprocessed_dir),
			"per_segment_manifest_json": str(paths.per_segment_manifest_path),
		}
	if phase_name == "plot_segment_traces":
		return {
			"plot_output_dir": outputs.get("plot_output_dir", str(paths.preprocess_out_dir)),
		}
	if phase_name == "concat_segments":
		return {
			"concatenated_recording_dir": str(paths.recording_dir),
			"preprocessed_recording_dir": str(paths.recording_dir),
			"concat_manifest_path": str(paths.concat_manifest_path),
		}
	if phase_name == "plot_concat_traces":
		return {
			"plot_output_dir": outputs.get("plot_output_dir", str(paths.preprocess_out_dir)),
			"concatenated_recording_dir": str(paths.recording_dir),
			"concat_manifest_path": str(paths.concat_manifest_path),
		}
	if phase_name == "wipe_src_scratch":
		return {
			"source_h5_path": str(payload.get("source_h5_path", "")),
			"resolved_h5_path": str(payload.get("resolved_h5_path", "")),
		}
	return {}


def _load_common_electrodes_or_empty(path: Path) -> list[int]:
	try:
		return load_common_electrodes(path)
	except Exception:
		return []


def _phase_summary_path(*, inputs: PreprocessInputs, paths: _PreprocessPathSet, phase_name: str) -> Path:
	return paths.preprocess_out_dir / Path(str(_summary_relpath_for_phase(inputs, phase_name))).expanduser()


def _load_existing_phase_payload(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	phase_name: str,
) -> dict[str, Any] | None:
	summary_path = _phase_summary_path(inputs=inputs, paths=paths, phase_name=phase_name)
	if not _safe_path_exists(summary_path):
		return None
	try:
		payload = read_json(summary_path)
	except Exception:
		return None
	if not isinstance(payload, dict):
		return None
	if str(payload.get("phase", "")).strip() != str(phase_name):
		return None
	return {
		str(key): value
		for key, value in dict(payload).items()
		if str(key) not in {"summary_json", "outputs", "h5_path", "stream_id", "well_out_dir", "preprocess_out_dir"}
	}


def _build_resumed_phase_payload(
	*,
	phase_name: str,
	existing_payload: dict[str, Any] | None,
	payload_updates: dict[str, Any],
) -> dict[str, Any]:
	payload: dict[str, Any] = {}
	if isinstance(existing_payload, dict):
		payload.update({str(key): value for key, value in existing_payload.items()})
	payload.update({str(key): value for key, value in payload_updates.items()})
	payload["phase"] = str(phase_name)
	payload["status"] = "skipped"
	payload["reused_existing_artifacts"] = True
	payload["phase_elapsed_s"] = 0.0
	return payload


def _resume_artifact_log_details(payload: dict[str, Any]) -> str:
	details: list[str] = []
	for key in (
		"resolved_h5_path",
		"segment_epochs_json",
		"contiguous_epochs_json",
		"sampling_metadata_json",
		"assay_stats_txt",
		"common_electrodes_path",
		"output_dir",
		"manifest_path",
		"recording_dir",
		"concat_manifest_path",
		"trace_plot_path",
	):
		value = str(payload.get(key, "")).strip()
		if value:
			details.append(f"{key}={value}")
	for key in ("layout_plot_paths", "segment_trace_paths", "missing_paths", "removed_paths", "would_remove_paths"):
		value = payload.get(key, [])
		if isinstance(value, list) and value:
			details.append(f"{key}={len(value)}")
	return "; ".join(details)


def _resume_copy_phase_payload_if_complete(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
) -> dict[str, Any] | None:
	existing_payload = _load_existing_phase_payload(inputs=inputs, paths=paths, phase_name="copy_src_to_scratch")
	if existing_payload is None:
		return None
	resolved_h5_path = Path(inputs.h5_path).expanduser().resolve()
	source_h5_path = Path(inputs.source_h5_path or inputs.h5_path).expanduser().resolve()
	if bool(inputs.copied_to_scratch) and not _safe_path_exists(resolved_h5_path):
		return None
	if not _safe_path_exists(source_h5_path):
		return None
	return _build_resumed_phase_payload(
		phase_name="copy_src_to_scratch",
		existing_payload=existing_payload,
		payload_updates={
			"source_h5_path": str(source_h5_path),
			"resolved_h5_path": str(resolved_h5_path),
			"copied_to_scratch": bool(inputs.copied_to_scratch),
		},
	)


def _resume_save_rec_metadata_payload_if_complete(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	recording_metadata_paths: _RecordingMetadataPathSet,
) -> dict[str, Any] | None:
	try:
		segment_epochs_payload, contiguous_epochs_payload, sampling_metadata_payload = load_recording_metadata(
			segment_epochs_path=recording_metadata_paths.segment_epochs_path,
			contiguous_epochs_path=recording_metadata_paths.contiguous_epochs_path,
			sampling_metadata_path=recording_metadata_paths.sampling_metadata_path,
		)
		common_electrodes = load_common_electrodes(recording_metadata_paths.common_electrodes_path)
	except Exception:
		return None
	if not _is_nonempty_file(recording_metadata_paths.assay_stats_path):
		return None
	segments = segment_epochs_payload.get("segments", [])
	if not isinstance(segments, list) or not segments:
		return None
	epochs = contiguous_epochs_payload.get("epochs", [])
	if not isinstance(epochs, list):
		return None
	sampling_summary = sampling_metadata_payload.get("sampling_summary", {})
	existing_payload = _load_existing_phase_payload(inputs=inputs, paths=paths, phase_name="save_rec_metadata")
	return _build_resumed_phase_payload(
		phase_name="save_rec_metadata",
		existing_payload=existing_payload,
		payload_updates={
			"source_h5_path": str(Path(inputs.source_h5_path or inputs.h5_path).expanduser().resolve()),
			"resolved_h5_path": str(Path(inputs.h5_path).expanduser().resolve()),
			"verbose": bool(inputs.phases.save_rec_metadata.verbose),
			"segment_count": int(len(segments)),
			"contiguous_epoch_count": int(len(epochs)),
			"segment_epochs_json": str(recording_metadata_paths.segment_epochs_path),
			"contiguous_epochs_json": str(recording_metadata_paths.contiguous_epochs_path),
			"sampling_metadata_json": str(recording_metadata_paths.sampling_metadata_path),
			"assay_stats_txt": str(recording_metadata_paths.assay_stats_path),
			"common_electrodes_path": str(recording_metadata_paths.common_electrodes_path),
			"common_electrode_count": int(len(common_electrodes)),
			"sampling_summary": (dict(sampling_summary) if isinstance(sampling_summary, dict) else {}),
		},
	)


def _resume_preprocess_segments_payload_if_complete(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
) -> dict[str, Any] | None:
	try:
		segment_entries = load_segment_manifest(paths.per_segment_manifest_path)
	except Exception:
		return None
	if not segment_entries:
		return None
	rec_names: list[str] = []
	for entry in segment_entries:
		folder = Path(str(entry.get("folder", ""))).expanduser().resolve()
		if not _safe_path_exists(folder):
			return None
		try:
			load_saved_recording(folder)
		except Exception:
			return None
		rec_name = str(entry.get("rec_name", folder.name)).strip()
		rec_names.append(rec_name or folder.name)
	existing_payload = _load_existing_phase_payload(inputs=inputs, paths=paths, phase_name="preprocess_segments")
	return _build_resumed_phase_payload(
		phase_name="preprocess_segments",
		existing_payload=existing_payload,
		payload_updates={
			"segment_count": int(len(segment_entries)),
			"rec_names": [str(value) for value in rec_names],
			"output_dir": str(paths.per_segment_preprocessed_dir),
			"manifest_path": str(paths.per_segment_manifest_path),
		},
	)


def _resume_plot_segment_traces_payload_if_complete(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	build_plot_cfg: PreprocessPlotConfig,
) -> dict[str, Any] | None:
	try:
		segment_entries = load_segment_manifest(paths.per_segment_manifest_path)
	except Exception:
		return None
	if not segment_entries:
		return None
	plot_root = paths.plot_output_dir or paths.preprocess_out_dir
	layout_paths: list[str] = []
	if bool(build_plot_cfg.layouts):
		layout_path = Path(plot_root) / str(build_plot_cfg.channel_layouts_subdir) / f"common_channel_layout_{inputs.stream_id}.png"
		if not _is_complete_png(layout_path):
			return None
		layout_paths.append(str(layout_path))
	segment_trace_paths: list[str] = []
	if bool(build_plot_cfg.segment_traces):
		for entry in segment_entries:
			rec_name = str(entry.get("rec_name", "segment")).strip() or "segment"
			trace_path = Path(plot_root) / str(build_plot_cfg.segment_traces_subdir) / f"segment_trace_{inputs.stream_id}_{rec_name}.png"
			if not _is_complete_png(trace_path):
				return None
			segment_trace_paths.append(str(trace_path))
	existing_payload = _load_existing_phase_payload(inputs=inputs, paths=paths, phase_name="plot_segment_traces")
	return _build_resumed_phase_payload(
		phase_name="plot_segment_traces",
		existing_payload=existing_payload,
		payload_updates={
			"segment_count": int(len(segment_entries)),
			"layout_plot_paths": list(layout_paths),
			"segment_trace_paths": list(segment_trace_paths),
		},
	)


def _resume_concat_segments_payload_if_complete(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
) -> dict[str, Any] | None:
	try:
		concat_manifest = load_concat_manifest(paths.concat_manifest_path)
		load_saved_recording(paths.recording_dir)
	except Exception:
		return None

	segment_entries = [
		dict(item)
		for item in concat_manifest.get("segment_entries", [])
		if isinstance(item, dict)
	]
	if not segment_entries:
		return None
	stitch_frames = [int(value) for value in concat_manifest.get("stitch_frames", []) if value is not None]
	if len(stitch_frames) not in {0, max(0, int(len(segment_entries) - 1))}:
		return None
	existing_payload = _load_existing_phase_payload(inputs=inputs, paths=paths, phase_name="concat_segments")
	return _build_resumed_phase_payload(
		phase_name="concat_segments",
		existing_payload=existing_payload,
		payload_updates={
			"segment_count": int(len(segment_entries)),
			"segment_source": str(concat_manifest.get("segment_source", "preprocessed") or "preprocessed"),
			"source_segment_count": int(len(segment_entries)),
			"concatenate_preprocessed_recordings": bool(inputs.phases.concat_segments.concatenate_preprocessed_recordings),
			"concat_manifest_path": str(paths.concat_manifest_path),
			"recording_dir": str(paths.recording_dir),
			"stitch_frame_count": int(len(stitch_frames)),
		},
	)


def _resume_plot_concat_traces_payload_if_complete(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	build_plot_cfg: PreprocessPlotConfig,
) -> dict[str, Any] | None:
	try:
		concat_manifest = load_concat_manifest(paths.concat_manifest_path)
		load_saved_recording(paths.recording_dir)
	except Exception:
		return None
	segment_entries = [
		dict(item)
		for item in concat_manifest.get("segment_entries", [])
		if isinstance(item, dict)
	]
	if not segment_entries:
		return None
	trace_plot_path = Path(paths.plot_output_dir or paths.preprocess_out_dir) / str(build_plot_cfg.concat_trace_relpath)
	if bool(build_plot_cfg.concat_trace) and not _is_complete_png(trace_plot_path):
		return None
	stitch_frames = [int(value) for value in concat_manifest.get("stitch_frames", []) if value is not None]
	existing_payload = _load_existing_phase_payload(inputs=inputs, paths=paths, phase_name="plot_concat_traces")
	return _build_resumed_phase_payload(
		phase_name="plot_concat_traces",
		existing_payload=existing_payload,
		payload_updates={
			"segment_count": int(len(segment_entries)),
			"trace_plot_path": str(trace_plot_path),
			"stitch_frame_count": int(len(stitch_frames)),
		},
	)


def _resume_wipe_src_scratch_payload_if_complete(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
) -> dict[str, Any] | None:
	if bool(inputs.phases.wipe_src_scratch.dry_run):
		existing_payload = _load_existing_phase_payload(inputs=inputs, paths=paths, phase_name="wipe_src_scratch")
		if existing_payload is None:
			return None
		return _build_resumed_phase_payload(
			phase_name="wipe_src_scratch",
			existing_payload=existing_payload,
			payload_updates={
				"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
				"resolved_h5_path": str(inputs.h5_path),
				"dry_run": True,
			},
		)
	if any(_safe_path_exists(path) for path in _candidate_wipe_src_scratch_paths(inputs)):
		return None
	existing_payload = _load_existing_phase_payload(inputs=inputs, paths=paths, phase_name="wipe_src_scratch")
	if existing_payload is None:
		return None
	return _build_resumed_phase_payload(
		phase_name="wipe_src_scratch",
		existing_payload=existing_payload,
		payload_updates={
			"source_h5_path": str(inputs.source_h5_path or inputs.h5_path),
			"resolved_h5_path": str(inputs.h5_path),
			"dry_run": False,
			"removed_paths": [],
			"would_remove_paths": [],
			"missing_paths": [str(path) for path in _candidate_wipe_src_scratch_paths(inputs)],
			"status": "already_missing",
		},
	)


def _resume_phase_payload_if_complete(
	*,
	phase_name: str,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	recording_metadata_paths: _RecordingMetadataPathSet,
	build_plot_cfg: PreprocessPlotConfig,
) -> dict[str, Any] | None:
	if bool(inputs.force_restart):
		return None
	if phase_name == "copy_src_to_scratch":
		return _resume_copy_phase_payload_if_complete(inputs=inputs, paths=paths)
	if phase_name == "save_rec_metadata":
		return _resume_save_rec_metadata_payload_if_complete(
			inputs=inputs,
			paths=paths,
			recording_metadata_paths=recording_metadata_paths,
		)
	if phase_name == "preprocess_segments":
		return _resume_preprocess_segments_payload_if_complete(inputs=inputs, paths=paths)
	if phase_name == "plot_segment_traces":
		return _resume_plot_segment_traces_payload_if_complete(
			inputs=inputs,
			paths=paths,
			build_plot_cfg=build_plot_cfg,
		)
	if phase_name == "concat_segments":
		return _resume_concat_segments_payload_if_complete(inputs=inputs, paths=paths)
	if phase_name == "plot_concat_traces":
		return _resume_plot_concat_traces_payload_if_complete(
			inputs=inputs,
			paths=paths,
			build_plot_cfg=build_plot_cfg,
		)
	if phase_name == "wipe_src_scratch":
		return _resume_wipe_src_scratch_payload_if_complete(inputs=inputs, paths=paths)
	return None


def _run_preprocess_phase_sequence(
	*,
	inputs: PreprocessInputs,
	paths: _PreprocessPathSet,
	recording_metadata_paths: _RecordingMetadataPathSet,
	selected_phase: str | None,
	event_records: list[dict[str, Any]] | None = None,
	scratch_usage_key: str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[int], bool]:
	canonical_selected_phase = _normalize_requested_preprocess_phase(selected_phase)
	outputs: dict[str, str] = {
		"legacy.preprocess_out_dir": str(paths.legacy_out_dir),
		"concatenated_recording_dir": str(paths.recording_dir),
		"preprocessed_recording_dir": str(paths.recording_dir),
		"concat_manifest_path": str(paths.concat_manifest_path),
		"common_electrodes_path": str(recording_metadata_paths.common_electrodes_path),
		"per_segment_manifest_json": str(paths.per_segment_manifest_path),
		"per_segment_preprocessed_dir": str(paths.per_segment_preprocessed_dir),
	}
	if bool(inputs.logging_enabled) or _safe_path_exists(paths.stage_log_source):
		outputs["pipeline_log"] = str(paths.stage_log_source)
	build_plot_cfg = _resolve_effective_plot_config(inputs, selected_phase=canonical_selected_phase)
	if paths.plot_output_dir is not None:
		outputs["plot_output_dir"] = str(paths.plot_output_dir)
	if paths.epoch_markers_output_dir is not None:
		outputs["epoch_markers_output_dir"] = str(paths.epoch_markers_output_dir)
	phase_logger = _prepare_phase_logger(inputs, paths)
	phase_summaries: dict[str, dict[str, Any]] = {}
	scratch_usage_released = False
	all_phases = [
		"copy_src_to_scratch",
		"save_rec_metadata",
		"preprocess_segments",
		"plot_segment_traces",
		"concat_segments",
		"plot_concat_traces",
		"wipe_src_scratch",
	]
	phases_to_run = [canonical_selected_phase] if canonical_selected_phase is not None else [
		phase_name for phase_name in all_phases if _phase_enabled(inputs, phase_name)
	]
	if phase_logger is not None and phases_to_run:
		phase_logger.info(
			"Starting preprocess work for well=%s phase_count=%d selected_phase=%s",
			str(inputs.stream_id),
			int(len(phases_to_run)),
			(str(canonical_selected_phase) if canonical_selected_phase is not None else "all"),
		)

	for phase_index, phase_name in enumerate(phases_to_run, start=1):
		phase_t0 = time.perf_counter()
		if phase_logger is not None:
			phase_logger.info(
				"Starting preprocess phase %d/%d for well=%s phase=%s",
				int(phase_index),
				int(len(phases_to_run)),
				str(inputs.stream_id),
				str(phase_name),
			)
		if phase_name == "copy_src_to_scratch":
			_validate_copy_phase_requirements(inputs, selected_phase=selected_phase)
		elif phase_name == "wipe_src_scratch":
			_validate_wipe_phase_requirements(inputs, selected_phase=selected_phase)

		payload = _resume_phase_payload_if_complete(
			phase_name=phase_name,
			inputs=inputs,
			paths=paths,
			recording_metadata_paths=recording_metadata_paths,
			build_plot_cfg=build_plot_cfg,
		)
		if payload is not None and phase_logger is not None:
			resume_artifacts = _resume_artifact_log_details(payload)
			phase_logger.info(
				"Resuming preprocess phase for well=%s phase=%s; found existing complete artifacts%s%s",
				str(inputs.stream_id),
				str(phase_name),
				(": " if resume_artifacts else ""),
				str(resume_artifacts),
			)
		if payload is None:
			if phase_name == "copy_src_to_scratch":
				payload = run_copy_src_to_scratch_core(
					h5_path=inputs.h5_path,
					source_h5_path=(inputs.source_h5_path or inputs.h5_path),
					stream_id=str(inputs.stream_id),
					copied_to_scratch=bool(inputs.copied_to_scratch),
					requires_use_scratch_root=bool(inputs.phases.copy_src_to_scratch.requires_use_scratch_root),
				)
			elif phase_name == "save_rec_metadata":
				payload = run_save_rec_metadata_core(
					h5_path=inputs.h5_path,
					source_h5_path=(inputs.source_h5_path or inputs.h5_path),
					stream_id=str(inputs.stream_id),
					segment_epochs_path=recording_metadata_paths.segment_epochs_path,
					contiguous_epochs_path=recording_metadata_paths.contiguous_epochs_path,
					sampling_metadata_path=recording_metadata_paths.sampling_metadata_path,
					assay_stats_path=recording_metadata_paths.assay_stats_path,
					common_electrodes_path=recording_metadata_paths.common_electrodes_path,
					verbose=bool(inputs.phases.save_rec_metadata.verbose),
					suppress_h5_plugin_messages=bool(inputs.logging_suppress_h5_plugin_messages),
					logger=phase_logger,
				)
			elif phase_name == "preprocess_segments":
				payload = run_preprocess_segments_core(
					h5_path=inputs.h5_path,
					stream_id=str(inputs.stream_id),
					n_jobs=max(1, int(inputs.n_jobs)),
					segment_epochs_path=recording_metadata_paths.segment_epochs_path,
					contiguous_epochs_path=recording_metadata_paths.contiguous_epochs_path,
					sampling_metadata_path=recording_metadata_paths.sampling_metadata_path,
					common_electrodes_path=recording_metadata_paths.common_electrodes_path,
					output_dir=paths.per_segment_preprocessed_dir,
					manifest_path=paths.per_segment_manifest_path,
					overwrite_saved_recording=bool(inputs.overwrite_saved_recording),
					save_n_jobs=max(1, int(inputs.phases.preprocess_segments.outputs.segment_save_n_jobs or inputs.n_jobs)),
					chunk_duration=str(inputs.phases.preprocess_segments.outputs.save_chunk_duration),
					progress_bar=bool(inputs.phases.preprocess_segments.outputs.save_progress_bar),
					limit_segments_per_well=inputs.debug_limit_segments_per_well,
					logger=phase_logger,
					run_save_segment_recordings_core=run_save_segment_recordings_core,
				)
			elif phase_name == "plot_segment_traces":
				payload = run_plot_segment_traces_core(
					stream_id=str(inputs.stream_id),
					segment_manifest_path=paths.per_segment_manifest_path,
					segment_epochs_path=recording_metadata_paths.segment_epochs_path,
					contiguous_epochs_path=recording_metadata_paths.contiguous_epochs_path,
					sampling_metadata_path=recording_metadata_paths.sampling_metadata_path,
					plot_output_dir=(paths.plot_output_dir or paths.preprocess_out_dir),
					channel_layouts_subdir=build_plot_cfg.channel_layouts_subdir,
					segment_traces_subdir=build_plot_cfg.segment_traces_subdir,
					plot_layouts=bool(build_plot_cfg.layouts),
					plot_segment_traces=bool(build_plot_cfg.segment_traces),
					segment_trace_n_reps=int(build_plot_cfg.segment_trace_n_reps),
					plot_n_jobs=max(1, int(build_plot_cfg.n_jobs or inputs.plot_n_jobs)),
					trace_downsample_hz=build_plot_cfg.trace_downsample_hz,
					trace_max_points=_normalize_trace_max_points(build_plot_cfg.trace_max_points),
					logger=phase_logger,
				)
			elif phase_name == "concat_segments":
				payload = run_concat_segments_core(
					stream_id=str(inputs.stream_id),
					segment_manifest_path=paths.per_segment_manifest_path,
					recording_dir=paths.recording_dir,
					concat_manifest_path=paths.concat_manifest_path,
					overwrite_saved_recording=bool(inputs.overwrite_saved_recording),
					n_jobs=max(1, int(inputs.phases.concat_segments.outputs.concat_save_n_jobs or inputs.n_jobs)),
					chunk_duration=str(inputs.phases.concat_segments.outputs.save_chunk_duration),
					progress_bar=bool(inputs.phases.concat_segments.outputs.save_progress_bar),
					logger=phase_logger,
					run_save_concatenated_recording_core=run_save_concatenated_recording_core,
				)
			elif phase_name == "plot_concat_traces":
				payload = run_plot_concat_traces_core(
					stream_id=str(inputs.stream_id),
					recording_dir=paths.recording_dir,
					concat_manifest_path=paths.concat_manifest_path,
					segment_epochs_path=recording_metadata_paths.segment_epochs_path,
					contiguous_epochs_path=recording_metadata_paths.contiguous_epochs_path,
					sampling_metadata_path=recording_metadata_paths.sampling_metadata_path,
					plot_output_dir=(paths.plot_output_dir or paths.preprocess_out_dir),
					concat_trace_relpath=build_plot_cfg.concat_trace_relpath,
					plot_concat_trace=bool(build_plot_cfg.concat_trace),
					concat_trace_n_reps=int(build_plot_cfg.concat_trace_n_reps),
					plot_n_jobs=max(1, int(build_plot_cfg.n_jobs or inputs.plot_n_jobs)),
					trace_downsample_hz=build_plot_cfg.trace_downsample_hz,
					trace_max_points=_normalize_trace_max_points(build_plot_cfg.trace_max_points),
					logger=phase_logger,
				)
			elif phase_name == "wipe_src_scratch":
				active_shared_users_remaining = _release_scratch_input_usage(scratch_usage_key)
				scratch_usage_released = True
				payload = run_wipe_src_scratch_core(
					h5_path=inputs.h5_path,
					source_h5_path=(inputs.source_h5_path or inputs.h5_path),
					copied_to_scratch=bool(inputs.copied_to_scratch),
					dry_run=bool(inputs.phases.wipe_src_scratch.dry_run),
					requires_use_scratch_root=bool(inputs.phases.wipe_src_scratch.requires_use_scratch_root),
					active_shared_users_remaining=int(active_shared_users_remaining),
				)
			else:
				raise RuntimeError(f"Unsupported preprocess phase: {phase_name}")

		payload = dict(payload)
		payload.setdefault("phase_elapsed_s", float(max(0.0, time.perf_counter() - phase_t0)))
		if event_records is not None:
			event_records.append(
				{
					"event": f"phase_complete:{phase_name}",
					"utc": _utc_now_iso(),
					"details": {
						"elapsed_s": float(payload.get("phase_elapsed_s", 0.0) or 0.0),
						"reused_existing_artifacts": bool(payload.get("reused_existing_artifacts", False)),
					},
				}
			)
		summary_outputs = _summary_outputs_for_phase(
			phase_name=phase_name,
			paths=paths,
			recording_metadata_paths=recording_metadata_paths,
			payload=payload,
			outputs=outputs,
		)
		outputs.update({str(key): str(value) for key, value in summary_outputs.items()})
		phase_summaries[phase_name] = _write_phase_summary(
			inputs=inputs,
			paths=paths,
			phase_name=phase_name,
			summary_json_relpath=_summary_relpath_for_phase(inputs, phase_name),
			outputs=summary_outputs,
			extra_payload=payload,
		)
		outputs[f"{phase_name}_summary_json"] = str(phase_summaries[phase_name]["summary_json"])
	common_electrodes = _load_common_electrodes_or_empty(recording_metadata_paths.common_electrodes_path)
	return outputs, phase_summaries, common_electrodes, scratch_usage_released

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
	if _reset_preprocess_output_root_for_force_restart(
		inputs=inputs,
		preprocess_out_dir=paths.preprocess_out_dir,
		selected_phase=None,
	):
		event_records.append(
			{
				"event": "force_restart_reset_preprocess_outputs",
				"utc": _utc_now_iso(),
				"details": {
					"preprocess_out_dir": str(paths.preprocess_out_dir),
				},
			}
		)

	stage_log_source = paths.stage_log_source
	_clear_self_referential_symlink(stage_log_source)
	scratch_usage_key = _acquire_scratch_input_usage(inputs, selected_phase=None)
	scratch_usage_released = False
	recording_metadata_paths = _resolve_recording_metadata_paths(inputs=inputs, preprocess_out_dir=paths.preprocess_out_dir)
	outputs: dict[str, str] = {"legacy.preprocess_out_dir": str(legacy_out_dir)}
	phase_summaries: dict[str, dict[str, Any]] = {}
	common_electrodes: list[int] = []

	try:
		outputs, phase_summaries, common_electrodes, scratch_usage_released = _run_preprocess_phase_sequence(
			inputs=inputs,
			paths=paths,
			recording_metadata_paths=recording_metadata_paths,
			selected_phase=None,
			event_records=event_records,
			scratch_usage_key=scratch_usage_key,
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
	phase_timing_s = {
		str(name): float(payload.get("phase_elapsed_s", 0.0) or 0.0)
		for name, payload in phase_summaries.items()
	}
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
	canonical_selected_phase = _normalize_requested_preprocess_phase(selected_phase)
	build_plot_cfg = _resolve_effective_plot_config(inputs, selected_phase=canonical_selected_phase)
	paths = _resolve_preprocess_paths(inputs, plot_cfg=build_plot_cfg)
	paths.preprocess_out_dir.mkdir(parents=True, exist_ok=True)
	_clear_self_referential_symlink(paths.stage_log_source)
	scratch_usage_key = _acquire_scratch_input_usage(inputs, selected_phase=selected_phase)
	scratch_usage_released = False
	recording_metadata_paths = _resolve_recording_metadata_paths(inputs=inputs, preprocess_out_dir=paths.preprocess_out_dir)
	try:
		outputs, phase_summaries, _common_electrodes, scratch_usage_released = _run_preprocess_phase_sequence(
			inputs=inputs,
			paths=paths,
			recording_metadata_paths=recording_metadata_paths,
			selected_phase=selected_phase,
			scratch_usage_key=scratch_usage_key,
		)
	finally:
		if scratch_usage_key is not None and not bool(scratch_usage_released):
			_release_scratch_input_usage(scratch_usage_key)
	payload = dict(phase_summaries.get(selected_phase, {}))
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


def run_preprocess_plot_segment_traces_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="plot_segment_traces")


def run_preprocess_concat_segments_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="concat_segments")


def run_preprocess_plot_concat_traces_phase(inputs: PreprocessInputs) -> dict[str, Any]:
	return _run_preprocess_selected_phase(inputs, selected_phase="plot_concat_traces")
