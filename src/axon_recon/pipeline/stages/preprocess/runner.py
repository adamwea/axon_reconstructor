from __future__ import annotations

import datetime as dt
import getpass
import json
import logging
import os
import platform
import shutil
import socket
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_reconstructor.pipeline.pipeline_logging import compute_pipeline_log_file
from axon_reconstructor.pipeline.stg1_preprocessing.constants import PREPROCESS_OUTPUTS_DIRNAME
from axon_reconstructor.pipeline.stg1_preprocessing.main import run_preprocess_stage as run_legacy_preprocess_stage

from .models.inputs import PreprocessInputs
from .models.results import PreprocessResult


LOGGER = logging.getLogger("axon_recon.preprocess")


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

def run_preprocess_stage(inputs: PreprocessInputs) -> PreprocessResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	legacy_out_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
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
				"h5_path": str(inputs.h5_path),
				"stream_id": str(inputs.stream_id),
			},
		}
	]

	stage_log_source = _resolve_stage_log_source(inputs=inputs, well_out_dir=well_out_dir)
	_clear_self_referential_symlink(stage_log_source)

	try:
		effective_trace_max_points = _normalize_trace_max_points(inputs.trace_max_points)
		_, common_electrodes = run_legacy_preprocess_stage(
			h5_path=inputs.h5_path,
			stream_id=inputs.stream_id,
			mea_output_root=inputs.mea_output_root,
			force_restart=bool(inputs.force_restart or inputs.force_replot),
			limit_segments_per_well=(
				int(inputs.debug_limit_segments_per_well)
				if inputs.debug_limit_segments_per_well is not None
				else None
			),
			log_enabled=bool(inputs.logging_enabled),
			log_verbose=bool(inputs.logging_verbose),
			log_file_override=inputs.logging_file_relpath,
			suppress_h5_plugin_messages=bool(inputs.logging_suppress_h5_plugin_messages),
			phase_dividers=bool(inputs.logging_phase_dividers),
			enable_checkpointing=bool(inputs.enable_checkpointing),
			n_jobs=max(1, int(inputs.n_jobs)),
			plot_layouts=bool(inputs.plot_layouts),
			plot_concat_trace=bool(inputs.plot_concat_trace),
			plot_segment_traces=bool(inputs.plot_segment_traces),
			plot_output_dir_override=inputs.plot_output_dir,
			epoch_markers_output_dir_override=inputs.epoch_markers_output_dir,
			assay_stats_relpath=inputs.assay_stats_relpath,
			channel_layouts_subdir=inputs.channel_layouts_subdir,
			segment_traces_subdir=inputs.segment_traces_subdir,
			concat_trace_relpath=inputs.concat_trace_relpath,
			n_representative_channels=int(inputs.n_representative_channels),
			concat_trace_n_reps=int(inputs.concat_trace_n_reps),
			segment_trace_n_reps=int(inputs.segment_trace_n_reps),
			plot_n_jobs=max(1, int(inputs.plot_n_jobs)),
			trace_downsample_hz=inputs.trace_downsample_hz,
			trace_max_points=effective_trace_max_points,
			temporal_resample_factor=inputs.temporal_resample_factor,
			temporal_resample_rate_hz=inputs.temporal_resample_rate_hz,
			temporal_resample_margin_ms=float(inputs.temporal_resample_margin_ms),
			temporal_resample_dtype=inputs.temporal_resample_dtype,
			save_recording=bool(inputs.save_recording),
			overwrite_saved_recording=bool(inputs.overwrite_saved_recording),
			save_concat_recording=bool(inputs.save_concat_recording),
			save_segment_recordings=bool(inputs.save_segment_recordings),
			save_chunk_duration=str(inputs.save_chunk_duration),
			save_progress_bar=bool(inputs.save_progress_bar),
			concat_save_n_jobs=(int(inputs.concat_save_n_jobs) if inputs.concat_save_n_jobs is not None else None),
			segment_save_n_jobs=(int(inputs.segment_save_n_jobs) if inputs.segment_save_n_jobs is not None else None),
			print_n_jobs_used=bool(inputs.print_n_jobs_used),
			logger=LOGGER,
		)
		event_records.append(
			{
				"event": "legacy_stage_complete",
				"utc": _utc_now_iso(),
				"details": {
					"n_common_electrodes": int(len(common_electrodes)),
				},
			}
		)
	except Exception as exc:
		event_records.append(
			{
				"event": "legacy_stage_failed",
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
					well_out_dir=well_out_dir,
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

	preprocess_out_dir = legacy_out_dir

	recording_dir = legacy_out_dir / "preprocessed_recording"
	common_electrodes_path = legacy_out_dir / "common_electrodes.npy"
	per_segment_manifest = legacy_out_dir / "per_segment_preprocessed" / "manifest.json"
	summary_json = preprocess_out_dir / "preprocess_summary.json"

	outputs: dict[str, str] = {
		"legacy.preprocess_out_dir": str(legacy_out_dir),
		"preprocessed_recording_dir": str(recording_dir),
		"common_electrodes_path": str(common_electrodes_path),
		"per_segment_manifest_json": str(per_segment_manifest),
	}
	if _safe_path_exists(stage_log_source):
		outputs["pipeline_log"] = str(stage_log_source)

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
		"well_out_dir": str(well_out_dir),
		"preprocess_out_dir": str(preprocess_out_dir),
		"legacy_preprocess_out_dir": str(legacy_out_dir),
		"output_rel_root": str(inputs.output_rel_root),
		"n_common_electrodes": int(len(common_electrodes)),
		"timing": {
			"started_utc": stage_started_utc,
			"ended_utc": _utc_now_iso(),
			"duration_seconds": stage_elapsed_s,
		},
		"inputs": {
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
		},
		"outputs": outputs,
	}

	_write_json(summary_json, _json_ready(summary_payload))

	if _is_observability_enabled(inputs):
		try:
			obs_outputs = _write_observability_artifacts(
				inputs=inputs,
				well_out_dir=well_out_dir,
				preprocess_out_dir=preprocess_out_dir,
				legacy_out_dir=legacy_out_dir,
				outputs=outputs,
				summary_json=summary_json,
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
				_write_json(summary_json, _json_ready(summary_payload))
		except Exception:
			LOGGER.warning("Failed writing preprocess observability artifacts", exc_info=True)

	return PreprocessResult(
		well_out_dir=well_out_dir,
		preprocess_out_dir=preprocess_out_dir,
		summary_json=summary_json,
		outputs=outputs,
	)
