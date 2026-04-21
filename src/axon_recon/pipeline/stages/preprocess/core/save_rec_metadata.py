from __future__ import annotations

import contextlib
import datetime as dt
import io
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Optional

from axon_recon.pipeline.shared.sampling import read_maxwell_sampling_frequency_hz
from .artifacts import write_json


_STDOUT_TEE_LOCK = threading.Lock()


def _ensure_maxwell_hdf5_plugin_path(*, prefix: str = "[axon_reconstructor]", suppress_messages: bool = False) -> None:
	env = os.environ.get("HDF5_PLUGIN_PATH")
	if env:
		try:
			if not Path(env).expanduser().exists():
				if not bool(suppress_messages):
					print(f"{prefix}[WARN] HDF5_PLUGIN_PATH points to missing dir: {env}; ignoring", flush=True)
				os.environ.pop("HDF5_PLUGIN_PATH", None)
		except Exception:
			pass

	if os.environ.get("HDF5_PLUGIN_PATH"):
		return

	here = Path(__file__).resolve()
	for parent in [here] + list(here.parents):
		cand_dir = parent / "vendor" / "maxwell_hdf5_plugin" / "Linux"
		if (cand_dir / "libcompression.so").exists():
			os.environ["HDF5_PLUGIN_PATH"] = str(cand_dir)
			if not bool(suppress_messages):
				print(f"{prefix}[DEBUG] set HDF5_PLUGIN_PATH={cand_dir}", flush=True)
			return


def _print_assay_settings(*, h5_path: Path, prefix: str = "[axon_reconstructor]") -> None:
	try:
		import h5py
	except Exception:
		print(f"{prefix} assay settings: h5py not available", flush=True)
		return

	h5_path = Path(h5_path).expanduser().resolve()
	try:
		with h5py.File(h5_path, "r") as h5:
			if "assay" not in h5:
				print(f"{prefix} assay settings: no /assay group", flush=True)
				return
			assay = h5["assay"]
			keys = list(assay.keys())
			print(f"{prefix} assay settings: /assay keys={keys}", flush=True)
	except Exception as exc:
		print(f"{prefix} assay settings: failed to read: {exc}", flush=True)


def _print_data_store_start_stop_durations(
	*,
	h5_path: Path,
	target_stream_id: Optional[str] = None,
	prefix: str = "[axon_reconstructor]",
) -> None:
	try:
		import h5py
		import numpy as np
	except Exception:
		print(f"{prefix} data_store: h5py/numpy not available", flush=True)
		return

	h5_path = Path(h5_path).expanduser().resolve()
	try:
		with h5py.File(h5_path, "r") as h5:
			if "data_store" not in h5:
				print(f"{prefix} data_store: no /data_store group", flush=True)
				return

			data_store = h5["data_store"]
			stream_ids = sorted(data_store.keys())
			if target_stream_id is not None:
				stream_ids = [stream for stream in stream_ids if str(stream) == str(target_stream_id)]
				if not stream_ids:
					print(f"{prefix} data_store: target_stream_id={target_stream_id} not found", flush=True)
					return

			for stream_id in stream_ids:
				stream = data_store[str(stream_id)]
				cfg_names = sorted(stream.keys())
				for cfg_name in cfg_names:
					cfg = stream[str(cfg_name)]

					def _read_ms(name: str) -> int | None:
						if name not in cfg:
							return None
						try:
							return int(np.asarray(cfg[name][()]).ravel()[0])
						except Exception:
							return None

					start_ms = _read_ms("start_time")
					stop_ms = _read_ms("stop_time")
					if start_ms is None or stop_ms is None:
						print(
							f"{prefix} data_store: stream={stream_id} cfg={cfg_name} start/stop unavailable",
							flush=True,
						)
						continue

					duration_s = (stop_ms - start_ms) / 1000.0
					print(
						f"{prefix} data_store: stream={stream_id} cfg={cfg_name} "
						f"start_ms={start_ms} stop_ms={stop_ms} dur_s={duration_s:.3f}",
						flush=True,
					)
	except Exception as exc:
		print(f"{prefix} data_store: failed to read: {exc}", flush=True)


def _read_well_rec_frame_nos_and_trigger_settings(*, h5_path: Path, stream_id: str, rec_name: str) -> dict[str, Any]:
	try:
		import h5py
		import numpy as np
	except Exception as exc:  # pragma: no cover
		raise RuntimeError("reading well timing requires h5py/numpy") from exc

	h5_path = Path(h5_path).expanduser().resolve()
	with h5py.File(h5_path, "r") as h5:
		rec = h5["wells"][str(stream_id)][str(rec_name)]
		start_raw = rec["start_time"][()]
		stop_raw = rec["stop_time"][()]
		start_ms = int(np.asarray(start_raw).ravel()[0])
		stop_ms = int(np.asarray(stop_raw).ravel()[0])

		routed = rec["groups"]["routed"]
		frame_nos = np.asarray(routed["frame_nos"], dtype=np.int64)

		def _read_int_1(name: str) -> int | None:
			if name not in routed:
				return None
			try:
				return int(np.asarray(routed[name][()]).ravel()[0])
			except Exception:
				return None

		def _read_float_1(name: str) -> float | None:
			if name not in routed:
				return None
			try:
				return float(np.asarray(routed[name][()]).ravel()[0])
			except Exception:
				return None

		triggered = _read_int_1("triggered")
		trigger_pre = _read_int_1("trigger_pre")
		trigger_post = _read_int_1("trigger_post")
		trigger_minamp = _read_float_1("trigger_minamp")
		trigger_maxamp = _read_float_1("trigger_maxamp")

	return {
		"start_ms": int(start_ms),
		"stop_ms": int(stop_ms),
		"frame_nos": frame_nos,
		"triggered": triggered,
		"trigger_pre": trigger_pre,
		"trigger_post": trigger_post,
		"trigger_minamp": trigger_minamp,
		"trigger_maxamp": trigger_maxamp,
	}


@contextlib.contextmanager
def _tee_stdout_to_file(out_path: Path):
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with _STDOUT_TEE_LOCK, open(out_path, "w", encoding="utf-8") as file_handle:

		class _Tee(io.TextIOBase):
			def __init__(self, terminal: Any, file_stream: Any):
				self._terminal = terminal
				self._file_stream = file_stream

			def write(self, text: str) -> int:
				self._terminal.write(text)
				try:
					self._file_stream.write(text)
				except Exception:
					pass
				return len(text)

			def flush(self) -> None:
				try:
					self._terminal.flush()
				finally:
					try:
						self._file_stream.flush()
					except Exception:
						pass

		tee = _Tee(sys.stdout, file_handle)
		with contextlib.redirect_stdout(tee):
			yield out_path


def find_common_electrodes_from_segments(*, h5_path: Path, stream_id: str) -> tuple[list[str], list[int]]:
	try:
		import h5py
		import spikeinterface.extractors as se
	except Exception as exc:  # pragma: no cover
		raise RuntimeError("raw preprocessing requires `h5py` and `spikeinterface` installed") from exc

	_ensure_maxwell_hdf5_plugin_path()

	h5_path = Path(h5_path)
	with h5py.File(h5_path, "r") as h5:
		rec_names = list(h5["wells"][stream_id].keys())

	common: set[int] | None = None
	for rec_name in rec_names:
		if hasattr(se, "read_maxwell"):
			recording = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
		else:  # pragma: no cover
			recording = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)
		electrodes = recording.get_property("contact_vector")["electrode"]
		electrode_set = set(int(value) for value in electrodes)
		if common is None:
			common = electrode_set
		else:
			common &= electrode_set

	return [str(name) for name in rec_names], sorted(common or set())


def _as_positive_float_or_none(value: Any) -> float | None:
	try:
		out = float(value)
	except Exception:
		return None
	return out if out > 0.0 else None


def _parse_recording_context_from_path(path: Path) -> dict[str, str]:
	parts = list(path.parts)
	out: dict[str, str] = {"filename": path.name, "parent_dir": path.parent.name}
	try:
		if len(parts) >= 6:
			out["dataset"] = parts[-6]
			out["date"] = parts[-5]
			out["plate"] = parts[-4]
			out["assay"] = parts[-3]
			out["run"] = parts[-2]
	except Exception:
		pass
	return out


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
		_ensure_maxwell_hdf5_plugin_path(suppress_messages=bool(suppress_h5_plugin_messages))
		with contextlib.ExitStack() as stack:
			if bool(suppress_h5_plugin_messages):
				suppressed_stream = io.StringIO()
				stack.enter_context(contextlib.redirect_stdout(suppressed_stream))
				stack.enter_context(contextlib.redirect_stderr(suppressed_stream))
			recording = se.read_maxwell(h5_path, stream_id=stream_id)
	except Exception as exc:
		return {}, f"SpikeInterface read_maxwell failed: {exc}"

	info: dict[str, Any] = {}
	for key, getter in (
		("sampling_frequency_hz", lambda: float(recording.get_sampling_frequency())),
		("num_channels", lambda: int(recording.get_num_channels())),
		("num_segments", lambda: int(recording.get_num_segments())),
	):
		try:
			info[key] = getter()
		except Exception:
			continue
	try:
		get_dtype = getattr(recording, "get_dtype", None)
		if callable(get_dtype):
			info["dtype"] = str(get_dtype())
	except Exception:
		pass
	try:
		fs_hz = float(recording.get_sampling_frequency())
		n_segments = int(recording.get_num_segments())
		segment_frames = [int(recording.get_num_frames(segment_index=segment_index)) for segment_index in range(n_segments)]
		if segment_frames and fs_hz > 0.0:
			total_frames = int(sum(segment_frames))
			info["duration_s_total"] = float(total_frames / fs_hz)
			info["num_frames_total"] = int(total_frames)
			info["num_frames_by_segment"] = [int(value) for value in segment_frames]
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
			_ensure_maxwell_hdf5_plugin_path(suppress_messages=bool(suppress_h5_plugin_messages))
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
		for key, getter in (
			("sampling_frequency_hz", lambda: float(recording.get_sampling_frequency())),
			("num_channels", lambda: int(recording.get_num_channels())),
		):
			try:
				entry[key] = getter()
			except Exception:
				continue
		try:
			get_num_samples = getattr(recording, "get_num_samples", None)
			entry["num_samples"] = int(get_num_samples()) if callable(get_num_samples) else int(recording.get_num_frames())
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


def run_save_rec_metadata_core(
	*,
	h5_path: Path,
	source_h5_path: Path,
	stream_id: str,
	segment_epochs_path: Path,
	contiguous_epochs_path: Path,
	sampling_metadata_path: Path,
	assay_stats_path: Path,
	common_electrodes_path: Path,
	verbose: bool,
	suppress_h5_plugin_messages: bool,
	logger: logging.Logger | None,
) -> dict[str, Any]:
	source_h5_path = Path(source_h5_path).expanduser().resolve()
	resolved_h5_path = Path(h5_path).expanduser().resolve()
	recording_info, recording_info_error = _try_get_spikeinterface_recording_info(
		h5_path=resolved_h5_path,
		stream_id=str(stream_id),
		suppress_h5_plugin_messages=bool(suppress_h5_plugin_messages),
	)
	rec_names, rec_names_error = _list_maxwell_recording_names(
		h5_path=resolved_h5_path,
		stream_id=str(stream_id),
	)
	segment_info_entries, segment_info_warnings = _try_get_spikeinterface_segment_infos(
		h5_path=resolved_h5_path,
		stream_id=str(stream_id),
		rec_names=list(rec_names),
		suppress_h5_plugin_messages=bool(suppress_h5_plugin_messages),
	)
	segment_info_by_name = {
		str(item.get("rec_name")): dict(item)
		for item in segment_info_entries
		if str(item.get("rec_name", "")).strip()
	}
	stream_sampling_hz = read_maxwell_sampling_frequency_hz(
		h5_path=resolved_h5_path,
		stream_id=str(stream_id),
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
					stream_id=str(stream_id),
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
			if segment_si_info.get("num_channels", None) is not None:
				segment_payload["num_channels"] = int(segment_si_info["num_channels"])
			segment_epochs.append(segment_payload)

			if frame_count <= 0 or sampling_hz is None:
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
			epoch_payload["start_time_utc"] = _format_epoch_iso_utc(epoch_start_s * float(divisor_to_seconds), divisor=float(divisor_to_seconds))
		if segment_start_s is not None and segment_relative_end_s is not None:
			epoch_end_s = float(segment_start_s + float(segment_relative_end_s))
			epoch_payload["end_time_seconds_since_epoch"] = epoch_end_s
			epoch_payload["end_time_utc"] = _format_epoch_iso_utc(epoch_end_s * float(divisor_to_seconds), divisor=float(divisor_to_seconds))

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
			for key in ("n_samples", "frame_no_start", "frame_no_end", "duration_samples_s", "duration_wall_clock_s"):
				if key in segment_epoch_payload:
					segment_payload[str(key)] = segment_epoch_payload[key]
		if segment_si_info.get("num_channels", None) is not None:
			segment_payload["num_channels"] = int(segment_si_info["num_channels"])
		if sampling_hz is not None:
			distinct_sampling_rates_hz.append(float(sampling_hz))
		sampling_segments.append(segment_payload)

	distinct_sampling_rates_hz = sorted({round(float(value), 9) for value in distinct_sampling_rates_hz})
	sampling_summary = {
		"stream_sampling_frequency_hz": (None if _as_positive_float_or_none(stream_sampling_hz) is None else float(stream_sampling_hz)),
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
		"stream_id": str(stream_id),
		"timestamp_unit": timestamp_unit,
		"segment_count": int(len(segment_epochs)),
		"segments": segment_epochs,
	}
	if metadata_warnings:
		segment_epochs_payload["warnings"] = list(metadata_warnings)

	contiguous_epochs_payload = {
		"h5_path": str(resolved_h5_path),
		"source_h5_path": str(source_h5_path),
		"stream_id": str(stream_id),
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
		"stream_id": str(stream_id),
		"segment_count": int(len(sampling_segments)),
		"sampling_summary": sampling_summary,
		"segments": sampling_segments,
	}
	if metadata_warnings:
		sampling_metadata_payload["warnings"] = list(metadata_warnings)

	write_json(segment_epochs_path, segment_epochs_payload)
	write_json(contiguous_epochs_path, contiguous_epochs_payload)
	write_json(sampling_metadata_path, sampling_metadata_payload)
	try:
		with _tee_stdout_to_file(assay_stats_path) as written_path:
			print(
				f"[axon_reconstructor][DEBUG] assay_stats file: {written_path} (generated {dt.datetime.now(dt.timezone.utc).isoformat()})",
				flush=True,
			)
			print(
				f"[axon_reconstructor][DEBUG] assay_stats context: h5={resolved_h5_path} stream={stream_id}",
				flush=True,
			)
			_print_assay_settings(h5_path=resolved_h5_path)
			_print_data_store_start_stop_durations(h5_path=resolved_h5_path, target_stream_id=str(stream_id))
	except Exception as exc:
		metadata_warnings.append(f"Failed writing assay stats artifact '{assay_stats_path}': {exc}")

	common_electrodes: list[int] = []
	try:
		common_names, common_electrodes = find_common_electrodes_from_segments(
			h5_path=resolved_h5_path,
			stream_id=str(stream_id),
		)
		_ = common_names
		if common_electrodes_path.suffix != ".npy":
			common_electrodes_path = common_electrodes_path.with_suffix(".npy")
		common_electrodes_path.parent.mkdir(parents=True, exist_ok=True)
		import numpy as np
		np.save(common_electrodes_path, np.asarray([int(value) for value in common_electrodes], dtype=np.int64))
	except Exception as exc:
		metadata_warnings.append(f"Failed computing/saving common electrodes: {exc}")

	payload: dict[str, Any] = {
		"phase": "save_rec_metadata",
		"source_h5_path": str(source_h5_path),
		"resolved_h5_path": str(resolved_h5_path),
		"verbose": bool(verbose),
		"source_path_context": _parse_recording_context_from_path(source_h5_path),
		"recording_info": recording_info,
		"segment_count": int(len(segment_epochs)),
		"contiguous_epoch_count": int(len(contiguous_epochs)),
		"segment_epochs_json": str(segment_epochs_path),
		"contiguous_epochs_json": str(contiguous_epochs_path),
		"sampling_metadata_json": str(sampling_metadata_path),
		"assay_stats_txt": str(assay_stats_path),
		"common_electrodes_path": str(common_electrodes_path),
		"common_electrode_count": int(len(common_electrodes)),
		"sampling_summary": sampling_summary,
	}
	if recording_info_error is not None:
		payload["recording_info_error"] = str(recording_info_error)
	if metadata_warnings:
		payload["metadata_warnings"] = list(metadata_warnings)
	if bool(verbose):
		payload["segment_epochs_preview"] = list(segment_epochs[: min(len(segment_epochs), 10)])
		payload["contiguous_epochs_preview"] = list(contiguous_epochs[: min(len(contiguous_epochs), 20)])
		payload["sampling_segments"] = list(sampling_segments)
		payload["common_electrodes_preview"] = [int(value) for value in list(common_electrodes)[:64]]
	if logger is not None:
		logger.info(
			"Saved recording metadata stream_id=%s segment_count=%d contiguous_epoch_count=%d common_electrodes=%d segment_epochs=%s contiguous_epochs=%s sampling_metadata=%s assay_stats=%s",
			str(stream_id),
			int(len(segment_epochs)),
			int(len(contiguous_epochs)),
			int(len(common_electrodes)),
			segment_epochs_path,
			contiguous_epochs_path,
			sampling_metadata_path,
			assay_stats_path,
		)
	return payload