from __future__ import annotations

import contextlib
import io
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from .artifacts import load_raw_binary_manifest, load_saved_recording, write_json
from .save_rec_metadata import _ensure_maxwell_hdf5_plugin_path, _list_maxwell_recording_names
from .save_segment_recordings import run_save_segment_recordings_core



def _read_maxwell_recording(
	*,
	h5_path: Path,
	stream_id: str,
	rec_name: str,
	suppress_h5_plugin_messages: bool,
) -> Any:
	try:
		import spikeinterface.extractors as se
	except Exception as exc:  # pragma: no cover
		raise RuntimeError("raw binary preparation requires spikeinterface.extractors") from exc

	_ensure_maxwell_hdf5_plugin_path(suppress_messages=bool(suppress_h5_plugin_messages))
	with contextlib.ExitStack() as stack:
		if bool(suppress_h5_plugin_messages):
			suppressed_stream = io.StringIO()
			stack.enter_context(contextlib.redirect_stdout(suppressed_stream))
			stack.enter_context(contextlib.redirect_stderr(suppressed_stream))
		try:
			return se.read_maxwell(h5_path, stream_id=stream_id, rec_name=str(rec_name))
		except TypeError:
			return se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=str(rec_name))


def _recording_segment_frames(recording: Any) -> list[int]:
	try:
		segment_count = int(recording.get_num_segments())
	except Exception:
		segment_count = 1
	frames: list[int] = []
	for segment_index in range(max(1, int(segment_count))):
		count: int | None = None
		for getter_name in ("get_num_frames", "get_num_samples"):
			getter = getattr(recording, getter_name, None)
			if not callable(getter):
				continue
			for args, kwargs in (
				((segment_index,), {}),
				((), {"segment_index": segment_index}),
				((), {}),
			):
				try:
					count = int(getter(*args, **kwargs))
					break
				except Exception:
					continue
			if count is not None:
				break
		frames.append(int(count or 0))
	return frames


def _build_manifest_payload(
	*,
	h5_path: Path,
	source_h5_path: Path,
	stream_id: str,
	rec_names: list[str],
	recording_dir: Path,
	segment_entries: list[dict[str, Any]],
) -> dict[str, Any]:
	segment_frames = [int(item.get("n_samples", 0) or 0) for item in list(segment_entries)]
	num_channels_by_segment = [int(item.get("n_channels", 0) or 0) for item in list(segment_entries)]
	sampling_frequency_hz_by_segment = [float(item.get("fs_hz", 0.0) or 0.0) for item in list(segment_entries)]
	unique_channels = {int(value) for value in num_channels_by_segment if int(value) > 0}
	unique_sampling = {float(value) for value in sampling_frequency_hz_by_segment if float(value) > 0.0}
	return {
		"version": 1,
		"stream_id": str(stream_id),
		"rec_names": [str(value) for value in list(rec_names)],
		"source_h5_path": str(Path(source_h5_path).expanduser().resolve()),
		"resolved_h5_path": str(Path(h5_path).expanduser().resolve()),
		"recording_dir": str(Path(recording_dir).expanduser().resolve()),
		"segment_count": int(len(segment_entries)),
		"num_channels": int(next(iter(unique_channels))) if len(unique_channels) == 1 else 0,
		"num_channels_by_segment": [int(value) for value in num_channels_by_segment],
		"sampling_frequency_hz": float(next(iter(unique_sampling))) if len(unique_sampling) == 1 else 0.0,
		"sampling_frequency_hz_by_segment": [float(value) for value in sampling_frequency_hz_by_segment],
		"num_frames_by_segment": [int(value) for value in segment_frames],
		"segments": [dict(item) for item in list(segment_entries)],
	}


def _manifest_segment_entries(manifest_payload: dict[str, Any]) -> list[dict[str, Any]]:
	segments = manifest_payload.get("segments", [])
	if not isinstance(segments, list):
		return []
	return [dict(item) for item in segments if isinstance(item, dict)]


def _manifest_rec_names(manifest_payload: dict[str, Any]) -> list[str]:
	rec_names = [str(value) for value in list(manifest_payload.get("rec_names", [])) if str(value).strip()]
	if rec_names:
		return rec_names
	return [
		str(item.get("rec_name"))
		for item in _manifest_segment_entries(manifest_payload)
		if str(item.get("rec_name", "")).strip()
	]


def _validate_saved_raw_binary_artifact(*, recording_dir: Path, manifest_payload: dict[str, Any]) -> None:
	segment_entries = _manifest_segment_entries(manifest_payload)
	if segment_entries:
		for entry in segment_entries:
			folder = Path(str(entry.get("folder", ""))).expanduser().resolve()
			if not str(folder):
				raise RuntimeError("Raw binary manifest entry is missing folder")
			load_saved_recording(folder)
		return
	load_saved_recording(recording_dir)


def _segment_stats_payload(*, rec_name: str, recording: Any) -> dict[str, Any]:
	segment_frames = _recording_segment_frames(recording)
	try:
		sampling_frequency_hz = float(recording.get_sampling_frequency())
	except Exception:
		sampling_frequency_hz = 0.0
	try:
		n_channels = int(recording.get_num_channels())
	except Exception:
		n_channels = 0
	return {
		"rec_name": str(rec_name),
		"fs": float(sampling_frequency_hz),
		"n_samples": int(sum(int(value) for value in segment_frames)),
		"n_channels": int(n_channels),
	}


def run_prepare_raw_binaries_core(
	*,
	h5_path: Path,
	source_h5_path: Path,
	stream_id: str,
	recording_dir: Path,
	manifest_path: Path,
	overwrite_saved_recording: bool,
	n_jobs: int,
	chunk_duration: str,
	progress_bar: bool,
	suppress_h5_plugin_messages: bool = False,
	logger: logging.Logger | None,
) -> dict[str, object]:
	t0 = time.perf_counter()
	recording_dir = Path(recording_dir).expanduser().resolve()
	manifest_path = Path(manifest_path).expanduser().resolve()
	resolved_h5_path = Path(h5_path).expanduser().resolve()
	resolved_source_h5_path = Path(source_h5_path).expanduser().resolve()
	rec_names, rec_names_error = _list_maxwell_recording_names(
		h5_path=resolved_h5_path,
		stream_id=str(stream_id),
	)
	if rec_names_error is not None:
		raise RuntimeError(str(rec_names_error))
	if not rec_names:
		raise RuntimeError(f"No Maxwell recording IDs found for stream={stream_id} in {resolved_h5_path}")

	if recording_dir.exists() or manifest_path.exists():
		if bool(overwrite_saved_recording):
			if recording_dir.exists():
				shutil.rmtree(recording_dir)
			if manifest_path.exists():
				manifest_path.unlink()
		else:
			try:
				manifest_payload = load_raw_binary_manifest(manifest_path)
				_validate_saved_raw_binary_artifact(recording_dir=recording_dir, manifest_payload=manifest_payload)
				if logger is not None:
					logger.info("prepare_raw_binaries reused existing artifact: %s", recording_dir)
				return {
					"phase": "prepare_raw_binaries",
					"recording_dir": str(recording_dir),
					"manifest_path": str(manifest_path),
					"raw_binary_recording_dir": str(recording_dir),
					"raw_binary_manifest_path": str(manifest_path),
					"saved": False,
					"reused_existing": True,
					"segment_count": int(manifest_payload.get("segment_count", 0) or 0),
					"rec_names": [str(value) for value in _manifest_rec_names(manifest_payload)],
					"num_channels": int(manifest_payload.get("num_channels", 0) or 0),
					"sampling_frequency_hz": float(manifest_payload.get("sampling_frequency_hz", 0.0) or 0.0),
					"num_frames_by_segment": [
						int(value)
						for value in list(manifest_payload.get("num_frames_by_segment", []))
					],
					"source_h5_path": str(resolved_source_h5_path),
					"resolved_h5_path": str(resolved_h5_path),
					"phase_timing_s": {
						"prepare_raw_binaries": float(max(0.0, time.perf_counter() - t0)),
					},
				}
			except Exception:
				if recording_dir.exists():
					shutil.rmtree(recording_dir)
				if manifest_path.exists():
					manifest_path.unlink()

	if logger is not None:
		logger.info(
			"Preparing raw binary recording for well=%s source=%s rec_count=%d out=%s",
			str(stream_id),
			resolved_h5_path,
			int(len(rec_names)),
			recording_dir,
		)
	segment_recordings: list[Any] = []
	segment_stats: list[dict[str, Any]] = []
	for segment_index, rec_name in enumerate(rec_names, start=1):
		recording = _read_maxwell_recording(
			h5_path=resolved_h5_path,
			stream_id=str(stream_id),
			rec_name=str(rec_name),
			suppress_h5_plugin_messages=bool(suppress_h5_plugin_messages),
		)
		segment_recordings.append(recording)
		segment_stats.append(_segment_stats_payload(rec_name=str(rec_name), recording=recording))
		if logger is not None:
			logger.info(
				"prepare_raw_binaries progress well=%s loaded=%d/%d rec_name=%s",
				str(stream_id),
				int(segment_index),
				int(len(rec_names)),
				str(rec_name),
			)
	save_result = run_save_segment_recordings_core(
		segment_recordings=list(segment_recordings),
		segment_names=[str(value) for value in rec_names],
		segment_stats=[dict(item) for item in segment_stats],
		output_dir=recording_dir,
		manifest_path=manifest_path,
		overwrite_saved_recording=bool(overwrite_saved_recording),
		n_jobs=max(1, int(n_jobs)),
		chunk_duration=str(chunk_duration),
		progress_bar=bool(progress_bar),
		logger=logger,
	)
	saved_manifest_payload = load_raw_binary_manifest(manifest_path)
	segment_entries = _manifest_segment_entries(saved_manifest_payload)
	manifest_payload = _build_manifest_payload(
		h5_path=resolved_h5_path,
		source_h5_path=resolved_source_h5_path,
		stream_id=str(stream_id),
		rec_names=list(rec_names),
		recording_dir=recording_dir,
		segment_entries=segment_entries,
	)
	write_json(manifest_path, manifest_payload)
	if logger is not None:
		logger.info(
			"Prepared raw binary recording for well=%s segment_count=%d out=%s",
			str(stream_id),
			int(manifest_payload.get("segment_count", 0) or 0),
			recording_dir,
		)
	return {
		"phase": "prepare_raw_binaries",
		"recording_dir": str(recording_dir),
		"manifest_path": str(manifest_path),
		"raw_binary_recording_dir": str(recording_dir),
		"raw_binary_manifest_path": str(manifest_path),
		"output_dir": str(save_result.get("output_dir", recording_dir)),
		"saved": True,
		"reused_existing": False,
		"segment_count": int(manifest_payload.get("segment_count", 0) or 0),
		"rec_names": [str(value) for value in list(manifest_payload.get("rec_names", []))],
		"num_channels": int(manifest_payload.get("num_channels", 0) or 0),
		"sampling_frequency_hz": float(manifest_payload.get("sampling_frequency_hz", 0.0) or 0.0),
		"num_frames_by_segment": [int(value) for value in list(manifest_payload.get("num_frames_by_segment", []))],
		"source_h5_path": str(resolved_source_h5_path),
		"resolved_h5_path": str(resolved_h5_path),
		"phase_timing_s": {
			"prepare_raw_binaries": float(max(0.0, time.perf_counter() - t0)),
		},
	}