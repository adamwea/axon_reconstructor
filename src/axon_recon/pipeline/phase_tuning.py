from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

from axon_recon.runtime_config import RuntimeConfig

from .logging.setup import current_pipeline_logging_config
from .resources import (
	PhaseResourceClassConfig,
	ResourcesConfig,
	get_active_resource_profile,
	get_phase_resource_class_config,
	parse_resources_config,
)


LOGGER = logging.getLogger("axon_recon.pipeline.phase_tuning")


@dataclass(frozen=True)
class PhaseTuningConfig:
	output_relpath: str = "resource_tuning"
	ram_safety_factor: float = 1.5
	cpu_safety_factor: float = 1.25
	min_observations_for_underuse: int = 5
	disk_benchmark_enabled: bool = True
	disk_benchmark_size_mb: int = 64
	disk_benchmark_chunk_mb: int = 4
	disk_underuse_fraction: float = 0.50
	disk_overuse_fraction: float = 0.90
	disk_target_fraction: float = 0.75
	require_limits_unless_confirmed: bool = True
	write_recommendations: bool = True
	update_runtime_yml: bool = False


def _as_mapping(value: Any) -> dict[str, Any]:
	return dict(value) if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
	if value is None:
		return bool(default)
	if isinstance(value, bool):
		return bool(value)
	token = str(value).strip().lower()
	if token in {"1", "true", "yes", "on"}:
		return True
	if token in {"0", "false", "no", "off"}:
		return False
	return bool(default)


def _as_float(value: Any, default: float) -> float:
	try:
		return float(value)
	except Exception:
		return float(default)


def _as_int(value: Any, default: int) -> int:
	try:
		return int(value)
	except Exception:
		return int(default)


def parse_phase_tuning_config(runtime_config: RuntimeConfig) -> PhaseTuningConfig:
	block = _as_mapping(runtime_config.get("resources.tuning", None))
	resources_block = _as_mapping(runtime_config.get("resources", None))
	if not block:
		block = _as_mapping(resources_block.get("tuning", None))
	disk_underuse_fraction = max(
		0.01,
		min(0.95, _as_float(block.get("disk_underuse_fraction", 0.50), 0.50)),
	)
	disk_overuse_fraction = max(
		disk_underuse_fraction,
		min(2.0, _as_float(block.get("disk_overuse_fraction", 0.90), 0.90)),
	)
	disk_target_fraction = max(
		disk_underuse_fraction,
		min(disk_overuse_fraction, _as_float(block.get("disk_target_fraction", 0.75), 0.75)),
	)
	return PhaseTuningConfig(
		output_relpath=str(block.get("output_relpath", "resource_tuning") or "resource_tuning"),
		ram_safety_factor=max(1.0, _as_float(block.get("ram_safety_factor", 1.5), 1.5)),
		cpu_safety_factor=max(1.0, _as_float(block.get("cpu_safety_factor", 1.25), 1.25)),
		min_observations_for_underuse=max(
			1,
			_as_int(block.get("min_observations_for_underuse", 5), 5),
		),
		disk_benchmark_enabled=_as_bool(block.get("disk_benchmark_enabled", True), True),
		disk_benchmark_size_mb=max(1, _as_int(block.get("disk_benchmark_size_mb", 64), 64)),
		disk_benchmark_chunk_mb=max(1, _as_int(block.get("disk_benchmark_chunk_mb", 4), 4)),
		disk_underuse_fraction=disk_underuse_fraction,
		disk_overuse_fraction=disk_overuse_fraction,
		disk_target_fraction=disk_target_fraction,
		require_limits_unless_confirmed=_as_bool(block.get("require_limits_unless_confirmed", True), True),
		write_recommendations=_as_bool(block.get("write_recommendations", True), True),
		update_runtime_yml=_as_bool(block.get("update_runtime_yml", False), False),
	)


def _jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
	if not path.exists():
		return
	for line in path.read_text(encoding="utf-8").splitlines():
		text = line.strip()
		if not text:
			continue
		try:
			payload = json.loads(text)
		except Exception:
			continue
		if isinstance(payload, dict):
			yield payload


def _metric(value: Any) -> float | None:
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _int_metric(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except Exception:
		return None


def _resource_gate_key_path(resource_gate: dict[str, Any], resource_name: str) -> str | None:
	keyed_requests = _as_mapping(resource_gate.get("keyed_requests", None))
	request = _as_mapping(keyed_requests.get(str(resource_name), None))
	value = request.get("key", None)
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _phase_read_h5_path_from_record(record: dict[str, Any], resource_gate: dict[str, Any]) -> str | None:
	for value in (
		record.get("phase_read_h5_path", None),
		_resource_gate_key_path(resource_gate, "source_h5_path"),
		record.get("source_h5_path", None),
	):
		if value is None:
			continue
		text = str(value).strip()
		if text:
			return text
	return None


def collect_phase_resource_observations_from_jsonl(
	path: str | Path,
	*,
	run_id: str | None = None,
	selected_stages: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
	jsonl_path = Path(path).expanduser()
	selected = {str(stage) for stage in (selected_stages or ()) if str(stage).strip()}
	observations: list[dict[str, Any]] = []
	for record in _jsonl_records(jsonl_path):
		if record.get("event") != "phase_resource_usage":
			continue
		if run_id is not None and str(record.get("run_id", "")) != str(run_id):
			continue
		stage = str(record.get("stage", "") or "")
		if selected and stage not in selected and not any(stage.startswith(f"{item}.") for item in selected):
			continue
		usage = record.get("resource_usage", None)
		if not isinstance(usage, dict):
			continue
		wall_time_s = _metric(usage.get("wall_time_s", None))
		disk_read_gb = _metric(usage.get("disk_read_gb", None))
		disk_write_gb = _metric(usage.get("disk_write_gb", None))
		read_gb_per_s = None
		write_gb_per_s = None
		if wall_time_s is not None and wall_time_s > 0:
			if disk_read_gb is not None:
				read_gb_per_s = max(0.0, float(disk_read_gb) / float(wall_time_s))
			if disk_write_gb is not None:
				write_gb_per_s = max(0.0, float(disk_write_gb) / float(wall_time_s))
		resource_gate = _as_mapping(record.get("resource_gate", None))
		resource_gate_wait_s = _metric(resource_gate.get("wait_s", None))
		if resource_gate_wait_s is None:
			resource_gate_wait_s = _metric(record.get("resource_gate_wait_s", None))
		resource_gate_wait_s = max(0.0, float(resource_gate_wait_s or 0.0))
		phase_read_h5_path = _phase_read_h5_path_from_record(record, resource_gate)
		observations.append(
			{
				"run_id": record.get("run_id", None),
				"timestamp": record.get("timestamp", None),
				"stage": stage or None,
				"phase": record.get("phase", None),
				"resource_class": record.get("resource_class", None),
				"dataset": record.get("dataset_id", None),
				"recording": record.get("recording_id", None),
				"well": record.get("well_id", None),
				"source_h5_path": record.get("source_h5_path", None),
				"phase_read_h5_path": phase_read_h5_path,
				"status": record.get("status", None),
				"exception_type": record.get("exception_type", None),
				"wall_time_s": wall_time_s,
				"process_peak_rss_gb": _metric(usage.get("process_peak_rss_gb", None)),
				"child_peak_rss_gb": _metric(usage.get("child_peak_rss_gb", None)),
				"total_peak_rss_gb": _metric(usage.get("total_peak_rss_gb", None)),
				"cpu_time_user_s": _metric(usage.get("cpu_time_user_s", None)),
				"cpu_time_system_s": _metric(usage.get("cpu_time_system_s", None)),
				"max_threads": _int_metric(usage.get("max_threads", None)),
				"observed_process_max_threads": _int_metric(usage.get("observed_process_max_threads", None)),
				"child_process_count_max": _int_metric(usage.get("child_process_count_max", None)),
				"disk_read_gb": disk_read_gb,
				"disk_write_gb": disk_write_gb,
				"disk_read_gb_per_s": read_gb_per_s,
				"disk_write_gb_per_s": write_gb_per_s,
				"gpu_peak_memory_gb": _metric(usage.get("gpu_peak_memory_gb", None)),
				"gpu_utilization_max_pct": _metric(usage.get("gpu_utilization_max_pct", None)),
				"resource_gate": resource_gate,
				"resource_gate_wait_s": resource_gate_wait_s,
				"resource_gate_waited": bool(resource_gate.get("waited", bool(resource_gate_wait_s > 0.0))),
				"resource_gate_slot_demands": _as_mapping(resource_gate.get("slot_demands", None)),
				"resource_gate_keyed_requests": _as_mapping(resource_gate.get("keyed_requests", None)),
			}
		)
	return observations


def _nonnull_float_values(items: Iterable[Any]) -> list[float]:
	values: list[float] = []
	for item in items:
		value = _metric(item)
		if value is not None:
			values.append(float(value))
	return values


def _nonnull_int_values(items: Iterable[Any]) -> list[int]:
	values: list[int] = []
	for item in items:
		value = _int_metric(item)
		if value is not None:
			values.append(int(value))
	return values


def _phase_class(resources: ResourcesConfig, resource_class: str | None) -> PhaseResourceClassConfig | None:
	return get_phase_resource_class_config(resources, resource_class)


def _cpu_parallelism_estimate(observation: dict[str, Any]) -> float | None:
	wall_time_s = _metric(observation.get("wall_time_s", None))
	if wall_time_s is None or wall_time_s <= 0:
		return None
	user_s = _metric(observation.get("cpu_time_user_s", None)) or 0.0
	system_s = _metric(observation.get("cpu_time_system_s", None)) or 0.0
	return max(0.0, float(user_s + system_s) / float(wall_time_s))


def _group_key_for_observation(observation: dict[str, Any]) -> tuple[str, str, str]:
	return (
		str(observation.get("stage", "unknown") or "unknown"),
		str(observation.get("phase", "unknown") or "unknown"),
		str(observation.get("resource_class", "") or ""),
	)


def _timestamp_to_epoch_s(value: Any) -> float | None:
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	if text.endswith("Z"):
		text = f"{text[:-1]}+00:00"
	try:
		parsed = datetime.fromisoformat(text)
	except Exception:
		return None
	if parsed.tzinfo is None:
		parsed = parsed.replace(tzinfo=timezone.utc)
	return float(parsed.timestamp())


def _fmt_float(value: Any, *, digits: int = 3) -> str:
	parsed = _metric(value)
	if parsed is None:
		return "unavailable"
	return f"{float(parsed):.{int(digits)}f}"


def _max_overlapping_slot_demand(
	*,
	observations: list[dict[str, Any]],
	recommendations_by_group: dict[tuple[str, str, str], dict[str, Any]],
	demand_key: str,
	include_gate_wait: bool = False,
) -> tuple[int | None, int]:
	events: list[tuple[float, int]] = []
	interval_count = 0
	for observation in observations:
		wall_time_s = _metric(observation.get("wall_time_s", None))
		end_s = _timestamp_to_epoch_s(observation.get("timestamp", None))
		if wall_time_s is None or wall_time_s <= 0 or end_s is None:
			continue
		recommendation = recommendations_by_group.get(_group_key_for_observation(observation), {})
		demand = _int_metric(recommendation.get(demand_key, None))
		if demand is None or demand <= 0:
			continue
		gate_wait_s = _metric(observation.get("resource_gate_wait_s", None)) if include_gate_wait else 0.0
		start_s = max(0.0, float(end_s) - float(wall_time_s) - float(gate_wait_s or 0.0))
		events.append((start_s, int(demand)))
		events.append((float(end_s), -int(demand)))
		interval_count += 1
	if not events:
		return None, interval_count
	events.sort(key=lambda item: (float(item[0]), 0 if int(item[1]) < 0 else 1))
	active_demand = 0
	max_demand = 0
	for _event_time, delta in events:
		active_demand = max(0, int(active_demand) + int(delta))
		max_demand = max(int(max_demand), int(active_demand))
	return int(max_demand), interval_count


def _nearest_existing_path(path: str | Path | None) -> Path | None:
	if path is None:
		return None
	candidate = Path(path).expanduser()
	for item in (candidate, *candidate.parents):
		try:
			if item.exists():
				return item
		except Exception:
			continue
	return None


def _device_id_for_path(path: str | Path | None) -> str | None:
	existing = _nearest_existing_path(path)
	if existing is None:
		return None
	try:
		return str(existing.stat().st_dev)
	except Exception:
		return None


def _gb_per_s(byte_count: int, elapsed_s: float) -> float | None:
	if byte_count <= 0 or elapsed_s <= 0:
		return None
	return float(byte_count) / float(elapsed_s) / float(1024**3)


def _benchmark_file_read(path: Path, *, byte_count: int, chunk_bytes: int) -> tuple[float | None, int, str | None]:
	try:
		file_size = max(0, int(path.stat().st_size))
	except Exception as exc:
		return None, 0, f"read benchmark stat failed: {type(exc).__name__}: {exc}"
	bytes_to_read = min(int(byte_count), int(file_size))
	if bytes_to_read <= 0:
		return None, 0, "read benchmark skipped because file is empty"
	read_bytes = 0
	started = time.perf_counter()
	try:
		with path.open("rb", buffering=0) as handle:
			while read_bytes < bytes_to_read:
				chunk = handle.read(min(int(chunk_bytes), int(bytes_to_read - read_bytes)))
				if not chunk:
					break
				read_bytes += len(chunk)
	except Exception as exc:
		return None, int(read_bytes), f"read benchmark failed: {type(exc).__name__}: {exc}"
	return _gb_per_s(read_bytes, max(0.0, time.perf_counter() - started)), int(read_bytes), None


def _benchmark_directory_read_write(
	directory: Path,
	*,
	byte_count: int,
	chunk_bytes: int,
) -> tuple[float | None, float | None, int, list[str], list[str]]:
	notes: list[str] = []
	warnings: list[str] = []
	directory.mkdir(parents=True, exist_ok=True)
	path = directory / f".axon_recon_phase_tune_io_{os.getpid()}.tmp"
	chunk = b"\0" * int(chunk_bytes)
	written_bytes = 0
	write_started = time.perf_counter()
	try:
		with path.open("wb", buffering=0) as handle:
			while written_bytes < int(byte_count):
				to_write = min(int(chunk_bytes), int(byte_count) - int(written_bytes))
				handle.write(chunk[:to_write])
				written_bytes += int(to_write)
			os.fsync(handle.fileno())
	except Exception as exc:
		warnings.append(f"write benchmark failed: {type(exc).__name__}: {exc}")
		try:
			path.unlink(missing_ok=True)
		except Exception:
			pass
		return None, None, int(written_bytes), notes, warnings
	write_gb_per_s = _gb_per_s(written_bytes, max(0.0, time.perf_counter() - write_started))
	read_gb_per_s, read_bytes, read_warning = _benchmark_file_read(
		path,
		byte_count=min(int(byte_count), int(written_bytes)),
		chunk_bytes=int(chunk_bytes),
	)
	if read_warning is not None:
		warnings.append(read_warning)
	elif read_bytes > 0:
		notes.append("temporary-file read benchmark may be influenced by OS cache")
	try:
		path.unlink(missing_ok=True)
	except Exception as exc:
		warnings.append(f"temporary benchmark file cleanup failed: {type(exc).__name__}: {exc}")
	return write_gb_per_s, read_gb_per_s, int(written_bytes), notes, warnings


def collect_disk_bandwidth_measurements(
	*,
	run_root: str | Path,
	tuning_config: PhaseTuningConfig,
	observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
	if not bool(tuning_config.disk_benchmark_enabled):
		return []
	benchmark_bytes = int(tuning_config.disk_benchmark_size_mb) * 1024 * 1024
	chunk_bytes = int(tuning_config.disk_benchmark_chunk_mb) * 1024 * 1024
	output_dir = (Path(run_root).expanduser() / tuning_config.output_relpath).resolve()
	measurements: list[dict[str, Any]] = []
	write_gb_per_s, read_gb_per_s, byte_count, notes, warnings = _benchmark_directory_read_write(
		output_dir,
		byte_count=benchmark_bytes,
		chunk_bytes=chunk_bytes,
	)
	measurements.append(
		{
			"path": str(output_dir),
			"path_kind": "run_output",
			"device_id": _device_id_for_path(output_dir),
			"read_capacity_gb_per_s": read_gb_per_s,
			"write_capacity_gb_per_s": write_gb_per_s,
			"benchmark_bytes": byte_count,
			"notes": notes,
			"warnings": warnings,
		}
	)
	device_measurements: dict[str, dict[str, Any]] = {}
	if measurements[0].get("device_id") is not None:
		device_measurements[str(measurements[0].get("device_id"))] = measurements[0]
	seen_phase_read_devices: set[str] = set()
	for observation in observations:
		source_path = observation.get("phase_read_h5_path", None) or observation.get("source_h5_path", None)
		if source_path is None:
			continue
		existing = _nearest_existing_path(source_path)
		if existing is None or not existing.is_file():
			continue
		device_id = _device_id_for_path(existing)
		device_key = None if device_id is None else str(device_id)
		if device_key is not None and device_key in seen_phase_read_devices:
			continue
		measurement_notes: list[str] = []
		measurement_warnings: list[str] = []
		if device_key is not None and device_key in device_measurements:
			device_measurement = device_measurements[device_key]
			read_capacity = device_measurement.get("read_capacity_gb_per_s", None)
			read_bytes = 0
			measurement_notes.append(f"reused device benchmark from {device_measurement.get('path')}")
		else:
			read_capacity, read_bytes, read_warning = _benchmark_file_read(
				existing,
				byte_count=benchmark_bytes,
				chunk_bytes=chunk_bytes,
			)
			if read_warning is not None:
				measurement_warnings.append(read_warning)
		measurement = {
			"path": str(existing),
			"path_kind": "phase_read_h5",
			"device_id": device_id,
			"read_capacity_gb_per_s": read_capacity,
			"write_capacity_gb_per_s": None,
			"benchmark_bytes": read_bytes,
			"notes": measurement_notes,
			"warnings": measurement_warnings,
		}
		measurements.append(measurement)
		if device_key is not None:
			device_measurements.setdefault(device_key, measurement)
			seen_phase_read_devices.add(device_key)
	return measurements


def _measurement_for_path(
	path: str | Path | None,
	measurements: list[dict[str, Any]],
	*,
	preferred_path_kind: str | None = None,
) -> dict[str, Any] | None:
	if path is None:
		return None
	path_text = str(Path(path).expanduser())
	for measurement in measurements:
		if str(measurement.get("path", "")) == path_text:
			return measurement
	device_id = _device_id_for_path(path)
	if device_id is not None:
		if preferred_path_kind is not None:
			for measurement in measurements:
				if str(measurement.get("path_kind", "")) != str(preferred_path_kind):
					continue
				if str(measurement.get("device_id", "")) == str(device_id):
					return measurement
		for measurement in measurements:
			if str(measurement.get("device_id", "")) == str(device_id):
				return measurement
	return None


def _observation_interval(observation: dict[str, Any]) -> tuple[float, float] | None:
	wall_time_s = _metric(observation.get("wall_time_s", None))
	end_s = _timestamp_to_epoch_s(observation.get("timestamp", None))
	if wall_time_s is None or wall_time_s <= 0 or end_s is None:
		return None
	return max(0.0, float(end_s) - float(wall_time_s)), float(end_s)


def _disk_bandwidth_pressure_by_path(
	*,
	observations: list[dict[str, Any]],
	run_root: str | Path | None,
	disk_measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
	if not disk_measurements:
		return []
	write_path = None if run_root is None else str(Path(run_root).expanduser())
	events_by_path: dict[str, list[tuple[float, float, float]]] = {}
	measurement_by_key: dict[str, dict[str, Any]] = {}
	observation_counts: dict[str, int] = {}
	for observation in observations:
		interval = _observation_interval(observation)
		if interval is None:
			continue
		start_s, end_s = interval
		read_rate = _metric(observation.get("disk_read_gb_per_s", None)) or 0.0
		write_rate = _metric(observation.get("disk_write_gb_per_s", None)) or 0.0
		read_path = observation.get("phase_read_h5_path", None) or observation.get("source_h5_path", None) or write_path
		for path, rate, is_read in ((read_path, read_rate, True), (write_path, write_rate, False)):
			if path is None or rate <= 0:
				continue
			measurement = _measurement_for_path(
				path,
				disk_measurements,
				preferred_path_kind="phase_read_h5" if is_read else "run_output",
			)
			if measurement is None:
				continue
			key = str(measurement.get("path") or path)
			measurement_by_key[key] = measurement
			observation_counts[key] = int(observation_counts.get(key, 0)) + 1
			read_delta = float(rate) if is_read else 0.0
			write_delta = 0.0 if is_read else float(rate)
			events_by_path.setdefault(key, []).append((start_s, read_delta, write_delta))
			events_by_path.setdefault(key, []).append((end_s, -read_delta, -write_delta))
	pressure: list[dict[str, Any]] = []
	for key, events in sorted(events_by_path.items()):
		events.sort(key=lambda item: (float(item[0]), 0 if float(item[1] + item[2]) < 0 else 1))
		active_read = 0.0
		active_write = 0.0
		max_read = 0.0
		max_write = 0.0
		max_combined_utilization: float | None = None
		measurement = measurement_by_key[key]
		read_capacity = _metric(measurement.get("read_capacity_gb_per_s", None))
		write_capacity = _metric(measurement.get("write_capacity_gb_per_s", None))
		for _event_time, read_delta, write_delta in events:
			active_read = max(0.0, float(active_read) + float(read_delta))
			active_write = max(0.0, float(active_write) + float(write_delta))
			max_read = max(float(max_read), float(active_read))
			max_write = max(float(max_write), float(active_write))
			read_util = None if read_capacity is None or read_capacity <= 0 else float(active_read) / float(read_capacity)
			write_util = None if write_capacity is None or write_capacity <= 0 else float(active_write) / float(write_capacity)
			combined = sum(value for value in (read_util, write_util) if value is not None)
			if read_util is not None or write_util is not None:
				max_combined_utilization = max(float(max_combined_utilization or 0.0), float(combined))
		max_read_utilization = None if read_capacity is None or read_capacity <= 0 else float(max_read) / float(read_capacity)
		max_write_utilization = None if write_capacity is None or write_capacity <= 0 else float(max_write) / float(write_capacity)
		pressure.append(
			{
				"path": key,
				"path_kind": measurement.get("path_kind", None),
				"device_id": measurement.get("device_id", None),
				"read_capacity_gb_per_s": read_capacity,
				"write_capacity_gb_per_s": write_capacity,
				"max_observed_read_gb_per_s": max_read,
				"max_observed_write_gb_per_s": max_write,
				"max_read_utilization": max_read_utilization,
				"max_write_utilization": max_write_utilization,
				"max_combined_utilization": max_combined_utilization,
				"observation_count": int(observation_counts.get(key, 0)),
			}
		)
	return pressure


def _recommend_for_group(
	*,
	resources: ResourcesConfig,
	tuning_config: PhaseTuningConfig,
	group_key: tuple[str, str, str],
	observations: list[dict[str, Any]],
) -> dict[str, Any]:
	stage, phase, resource_class = group_key
	phase_class = _phase_class(resources, resource_class)
	notes: list[str] = []
	warnings: list[str] = []
	observation_count = len(observations)
	if phase_class is None:
		warnings.append("resource_class is missing or undefined; recommendations are limited")

	current_ram_gb = None if phase_class is None else float(phase_class.ram_gb)
	current_cpu_cores = None if phase_class is None else int(phase_class.cpu_cores)
	current_h5_read_slots = None if phase_class is None else int(phase_class.h5_read_slots)
	current_disk_heavy_slots = None if phase_class is None else int(phase_class.disk_heavy_slots)
	current_gpu_sort_slots = None if phase_class is None else int(phase_class.gpu_sort_slots)
	current_plot_slots = None if phase_class is None else int(phase_class.plot_slots)
	current_analyzer_slots = None if phase_class is None else int(phase_class.analyzer_slots)

	peak_values = _nonnull_float_values(item.get("total_peak_rss_gb", None) for item in observations)
	max_peak_ram = max(peak_values) if peak_values else None
	recommended_ram_gb = current_ram_gb
	if max_peak_ram is not None:
		ram_candidate = int(max(1, math.ceil(float(max_peak_ram) * float(tuning_config.ram_safety_factor))))
		if current_ram_gb is None or current_ram_gb <= 0:
			recommended_ram_gb = ram_candidate
		elif ram_candidate > current_ram_gb:
			recommended_ram_gb = float(ram_candidate)
			warnings.append("observed peak RAM is close to or above the configured class estimate")
		elif ram_candidate < current_ram_gb:
			if observation_count >= int(tuning_config.min_observations_for_underuse):
				recommended_ram_gb = float(max(1, ram_candidate))
				notes.append("class may be overestimated; verify with representative non-debug runs before lowering")
			else:
				notes.append("RAM looked underused, but observation count is too small for a lowering recommendation")
		else:
			notes.append("current RAM estimate matches the observed peak plus safety factor")
	else:
		notes.append("RAM metrics were unavailable")

	planned_threads = _nonnull_int_values(item.get("max_threads", None) for item in observations)
	observed_threads = _nonnull_int_values(item.get("observed_process_max_threads", None) for item in observations)
	cpu_parallelism = _nonnull_float_values(_cpu_parallelism_estimate(item) for item in observations)
	max_planned_threads = max(planned_threads) if planned_threads else None
	max_observed_threads = max(observed_threads) if observed_threads else None
	max_cpu_parallelism = max(cpu_parallelism) if cpu_parallelism else None
	cpu_basis = max(
		[value for value in (max_planned_threads, max_cpu_parallelism) if value is not None],
		default=None,
	)
	recommended_cpu_cores = current_cpu_cores
	if cpu_basis is not None:
		cpu_candidate = int(max(1, math.ceil(float(cpu_basis) * float(tuning_config.cpu_safety_factor))))
		if current_cpu_cores is None or current_cpu_cores <= 0:
			recommended_cpu_cores = cpu_candidate
		elif float(cpu_basis) > float(current_cpu_cores) * 1.10 and cpu_candidate > current_cpu_cores:
			recommended_cpu_cores = cpu_candidate
			warnings.append("observed planned threads or CPU time exceed the configured CPU estimate")
		else:
			notes.append("current CPU estimate covers observed pipeline thread demand")
	else:
		notes.append("CPU parallelism metrics were unavailable")
	if max_observed_threads is not None and current_cpu_cores is not None and max_observed_threads > max(1, current_cpu_cores) * 2:
		notes.append("native/library thread count exceeded configured CPU cores; treat as diagnostic, not direct demand")

	max_read_rate = max(_nonnull_float_values(item.get("disk_read_gb_per_s", None) for item in observations), default=None)
	max_write_rate = max(_nonnull_float_values(item.get("disk_write_gb_per_s", None) for item in observations), default=None)
	max_read_gb = max(_nonnull_float_values(item.get("disk_read_gb", None) for item in observations), default=None)
	max_write_gb = max(_nonnull_float_values(item.get("disk_write_gb", None) for item in observations), default=None)
	recommended_h5_read_slots = current_h5_read_slots
	recommended_disk_heavy_slots = current_disk_heavy_slots
	if max_read_rate is not None and max_read_rate > 0.05 and current_h5_read_slots == 0:
		recommended_h5_read_slots = 1
		notes.append("phase performed measurable reads; consider h5_read_slots=1 when concurrent reads share a source file")
	if (
		((max_read_rate is not None and max_read_rate > 0.25) or (max_write_rate is not None and max_write_rate > 0.1))
		and current_disk_heavy_slots == 0
	):
		recommended_disk_heavy_slots = 1
		notes.append("phase showed measurable disk throughput; consider disk_heavy_slots=1 before increasing phase concurrency")
	elif max_read_rate is not None or max_write_rate is not None:
		notes.append("disk read/write rates are observed phase throughput, not a destructive disk benchmark")

	return {
		"stage": stage,
		"phase": phase,
		"resource_class": resource_class or None,
		"observations": observation_count,
		"max_total_peak_rss_gb": max_peak_ram,
		"max_pipeline_threads": max_planned_threads,
		"max_observed_process_threads": max_observed_threads,
		"max_cpu_parallelism_estimate": max_cpu_parallelism,
		"max_disk_read_gb": max_read_gb,
		"max_disk_write_gb": max_write_gb,
		"max_disk_read_gb_per_s": max_read_rate,
		"max_disk_write_gb_per_s": max_write_rate,
		"current_class_ram_gb": current_ram_gb,
		"recommended_class_ram_gb": recommended_ram_gb,
		"current_class_cpu_cores": current_cpu_cores,
		"recommended_class_cpu_cores": recommended_cpu_cores,
		"current_h5_read_slots": current_h5_read_slots,
		"recommended_h5_read_slots": recommended_h5_read_slots,
		"current_disk_heavy_slots": current_disk_heavy_slots,
		"recommended_disk_heavy_slots": recommended_disk_heavy_slots,
		"current_gpu_sort_slots": current_gpu_sort_slots,
		"recommended_gpu_sort_slots": current_gpu_sort_slots,
		"current_plot_slots": current_plot_slots,
		"recommended_plot_slots": current_plot_slots,
		"current_analyzer_slots": current_analyzer_slots,
		"recommended_analyzer_slots": current_analyzer_slots,
		"notes": notes,
		"warnings": warnings,
	}


def _qualified_stage_phase(stage: Any, phase: Any) -> str:
	stage_text = str(stage or "unknown").strip() or "unknown"
	phase_text = str(phase or "unknown").strip() or "unknown"
	if stage_text == phase_text or stage_text.endswith(f".{phase_text}"):
		return stage_text
	return f"{stage_text}.{phase_text}"


def _profile_io_slot_recommendation(
	*,
	resources: ResourcesConfig,
	tuning_config: PhaseTuningConfig,
	observations: list[dict[str, Any]],
	recommendations: list[dict[str, Any]],
	run_root: str | Path | None,
	disk_measurements: list[dict[str, Any]],
) -> dict[str, Any]:
	profile = get_active_resource_profile(resources)
	profile_name = resources.active_profile
	if profile is None:
		return {
			"profile": profile_name,
			"notes": ["active resource profile is undefined; profile slot recommendations are unavailable"],
			"warnings": [],
		}
	recommendations_by_group = {
		(
			str(recommendation.get("stage", "unknown") or "unknown"),
			str(recommendation.get("phase", "unknown") or "unknown"),
			str(recommendation.get("resource_class", "") or ""),
		): recommendation
		for recommendation in recommendations
	}
	notes: list[str] = []
	warnings: list[str] = []
	current_h5_read_slots = max(0, int(profile.h5_read_slots))
	current_disk_heavy_slots = max(0, int(profile.disk_heavy_slots))
	current_h5_active_demand, h5_active_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="current_h5_read_slots",
	)
	recommended_h5_active_demand, recommended_h5_active_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="recommended_h5_read_slots",
	)
	current_disk_active_demand, disk_active_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="current_disk_heavy_slots",
	)
	recommended_disk_active_demand, recommended_disk_active_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="recommended_disk_heavy_slots",
	)
	current_h5_demand, h5_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="current_h5_read_slots",
		include_gate_wait=True,
	)
	recommended_h5_demand, recommended_h5_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="recommended_h5_read_slots",
		include_gate_wait=True,
	)
	current_disk_demand, disk_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="current_disk_heavy_slots",
		include_gate_wait=True,
	)
	recommended_disk_demand, recommended_disk_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="recommended_disk_heavy_slots",
		include_gate_wait=True,
	)
	gate_wait_values = _nonnull_float_values(item.get("resource_gate_wait_s", None) for item in observations)
	positive_gate_wait_values = [value for value in gate_wait_values if float(value) > 0.0]
	disk_bandwidth_pressure = _disk_bandwidth_pressure_by_path(
		observations=observations,
		run_root=run_root,
		disk_measurements=disk_measurements,
	)
	h5_read_utilization_values = [
		float(item["max_read_utilization"])
		for item in disk_bandwidth_pressure
		if item.get("max_read_utilization", None) is not None
	]
	disk_heavy_utilization_values = [
		float(item["max_combined_utilization"])
		for item in disk_bandwidth_pressure
		if item.get("max_combined_utilization", None) is not None
	]
	max_h5_read_bandwidth_utilization = max(h5_read_utilization_values, default=None)
	max_disk_heavy_bandwidth_utilization = max(disk_heavy_utilization_values, default=None)

	def _recommend_profile_slots(
		*,
		dimension: str,
		current_slots: int,
		observed_recommended_demand: int | None,
		bandwidth_utilization: float | None,
	) -> int:
		if observed_recommended_demand is None or observed_recommended_demand <= 0:
			notes.append(
				f"{dimension}: selected observations did not show slot-consuming demand; keeping current profile slots ({current_slots})"
			)
			return int(current_slots)
		if bandwidth_utilization is None:
			notes.append(
				f"{dimension}: disk bandwidth capacity was unavailable; keeping current profile slots ({current_slots})"
			)
			if int(observed_recommended_demand) > int(current_slots):
				notes.append(
					f"{dimension}: peak requested recommended slot demand including gate waits ({observed_recommended_demand}) exceeded current profile slots ({current_slots}), but bandwidth utilization is needed before recommending a profile change"
				)
			return int(current_slots)
		if float(bandwidth_utilization) < float(tuning_config.disk_underuse_fraction):
			if int(observed_recommended_demand) <= int(current_slots):
				notes.append(
					f"{dimension}: disk bandwidth was underused (utilization={_fmt_float(bandwidth_utilization)} below underuse_threshold={_fmt_float(tuning_config.disk_underuse_fraction)}), but peak requested recommended slot demand including gate waits ({observed_recommended_demand}) did not exceed current profile slots ({current_slots}); increasing {dimension} would not change this run. Increase selected target concurrency or inspect other limits if you expected more IO pressure."
				)
				return int(current_slots)
			scaled_candidate = int(
				math.ceil(
					max(1.0, float(current_slots or 1))
					* float(tuning_config.disk_target_fraction)
					/ max(0.01, float(bandwidth_utilization))
				)
			)
			recommended = max(int(current_slots) + 1, min(int(observed_recommended_demand), int(scaled_candidate)))
			notes.append(
				f"{dimension}: disk bandwidth was underused (utilization={_fmt_float(bandwidth_utilization)} below underuse_threshold={_fmt_float(tuning_config.disk_underuse_fraction)}) and peak requested recommended slot demand including gate waits ({observed_recommended_demand}) exceeded current profile slots ({current_slots}); recommend increasing to {recommended} so more IO work can run concurrently."
			)
			return int(recommended)
		if float(bandwidth_utilization) > float(tuning_config.disk_overuse_fraction):
			if int(current_slots) <= 1 or int(observed_recommended_demand) <= 1:
				warnings.append(
					f"{dimension}: disk bandwidth looked saturated (utilization={_fmt_float(bandwidth_utilization)} above overuse_threshold={_fmt_float(tuning_config.disk_overuse_fraction)}), but current observed demand cannot be reduced below one slot"
				)
				return int(current_slots)
			scaled_candidate = int(
				math.floor(
					float(current_slots)
					* float(tuning_config.disk_target_fraction)
					/ max(0.01, float(bandwidth_utilization))
				)
			)
			recommended = max(1, min(int(current_slots) - 1, int(scaled_candidate)))
			warnings.append(
				f"{dimension}: disk bandwidth looked saturated (utilization={_fmt_float(bandwidth_utilization)} above overuse_threshold={_fmt_float(tuning_config.disk_overuse_fraction)}); recommend decreasing to {recommended} to reduce IO contention"
			)
			return int(recommended)
		notes.append(
			f"{dimension}: observed disk bandwidth utilization ({_fmt_float(bandwidth_utilization)}) is within the target range"
		)
		return int(current_slots)

	recommended_profile_h5_read_slots = _recommend_profile_slots(
		dimension="h5_read_slots",
		current_slots=current_h5_read_slots,
		observed_recommended_demand=recommended_h5_demand,
		bandwidth_utilization=max_h5_read_bandwidth_utilization,
	)
	recommended_profile_disk_heavy_slots = _recommend_profile_slots(
		dimension="disk_heavy_slots",
		current_slots=current_disk_heavy_slots,
		observed_recommended_demand=recommended_disk_demand,
		bandwidth_utilization=max_disk_heavy_bandwidth_utilization,
	)
	if observations:
		notes.append("profile slot recommendations compare observed phase IO rates with measured disk bandwidth")
	if positive_gate_wait_values:
		notes.append("resource gate waits were observed; peak requested demand includes queued phases before acquisition")
	return {
		"profile": profile_name,
		"current_h5_read_slots": current_h5_read_slots,
		"recommended_h5_read_slots": recommended_profile_h5_read_slots,
		"current_disk_heavy_slots": current_disk_heavy_slots,
		"recommended_disk_heavy_slots": recommended_profile_disk_heavy_slots,
		"max_current_h5_read_slot_demand": current_h5_demand,
		"max_recommended_h5_read_slot_demand": recommended_h5_demand,
		"max_active_current_h5_read_slot_demand": current_h5_active_demand,
		"max_active_recommended_h5_read_slot_demand": recommended_h5_active_demand,
		"max_requested_current_h5_read_slot_demand": current_h5_demand,
		"max_requested_recommended_h5_read_slot_demand": recommended_h5_demand,
		"h5_read_interval_observations": h5_interval_count,
		"recommended_h5_read_interval_observations": recommended_h5_interval_count,
		"h5_read_active_interval_observations": h5_active_interval_count,
		"recommended_h5_read_active_interval_observations": recommended_h5_active_interval_count,
		"max_current_disk_heavy_slot_demand": current_disk_demand,
		"max_recommended_disk_heavy_slot_demand": recommended_disk_demand,
		"max_active_current_disk_heavy_slot_demand": current_disk_active_demand,
		"max_active_recommended_disk_heavy_slot_demand": recommended_disk_active_demand,
		"max_requested_current_disk_heavy_slot_demand": current_disk_demand,
		"max_requested_recommended_disk_heavy_slot_demand": recommended_disk_demand,
		"disk_heavy_interval_observations": disk_interval_count,
		"recommended_disk_heavy_interval_observations": recommended_disk_interval_count,
		"disk_heavy_active_interval_observations": disk_active_interval_count,
		"recommended_disk_heavy_active_interval_observations": recommended_disk_active_interval_count,
		"resource_gate_wait_observations": len(positive_gate_wait_values),
		"max_resource_gate_wait_s": max(positive_gate_wait_values, default=0.0),
		"total_resource_gate_wait_s": sum(positive_gate_wait_values),
		"max_h5_read_bandwidth_utilization": max_h5_read_bandwidth_utilization,
		"max_disk_heavy_bandwidth_utilization": max_disk_heavy_bandwidth_utilization,
		"disk_bandwidth_pressure": disk_bandwidth_pressure,
		"notes": notes,
		"warnings": warnings,
	}


def build_phase_tuning_summary(
	*,
	resources: ResourcesConfig,
	tuning_config: PhaseTuningConfig,
	observations: list[dict[str, Any]],
	selected_stages: Iterable[str],
	run_id: str | None,
	run_root: str | Path | None = None,
	disk_measurements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
	groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
	for observation in observations:
		key = (
			str(observation.get("stage", "unknown") or "unknown"),
			str(observation.get("phase", "unknown") or "unknown"),
			str(observation.get("resource_class", "") or ""),
		)
		groups.setdefault(key, []).append(observation)
	recommendations = [
		_recommend_for_group(
			resources=resources,
			tuning_config=tuning_config,
			group_key=key,
			observations=group_observations,
		)
		for key, group_observations in sorted(groups.items())
	]
	profile = get_active_resource_profile(resources)
	active_profile_recommendation = _profile_io_slot_recommendation(
		resources=resources,
		tuning_config=tuning_config,
		observations=observations,
		recommendations=recommendations,
		run_root=run_root,
		disk_measurements=list(disk_measurements or []),
	)
	return {
		"run_id": run_id,
		"selected_stages": list(selected_stages),
		"active_profile": resources.active_profile,
		"active_profile_capacity": None if profile is None else profile.__dict__,
		"active_profile_recommendation": active_profile_recommendation,
		"disk_bandwidth_measurements": list(disk_measurements or []),
		"observation_count": len(observations),
		"recommendation_count": len(recommendations),
		"update_runtime_yml": False,
		"advisory_only": True,
		"recommendations": recommendations,
	}


def format_phase_tuning_report(summary: dict[str, Any]) -> str:
	lines: list[str] = [
		"# Resource Tuning Report",
		"",
		f"Run ID: {summary.get('run_id') or 'unknown'}",
		f"Active profile: {summary.get('active_profile') or 'none'}",
		f"Observations: {summary.get('observation_count', 0)}",
		"Advisory only: runtime YAML was not modified.",
		"",
		"## Active Profile IO Slots",
	]
	profile_recommendation = summary.get("active_profile_recommendation", {})
	if isinstance(profile_recommendation, dict):
		lines.extend(
			[
				f"- profile: {profile_recommendation.get('profile') or 'none'}",
				f"- h5_read_slots: {profile_recommendation.get('current_h5_read_slots')} -> {profile_recommendation.get('recommended_h5_read_slots')}",
				f"- disk_heavy_slots: {profile_recommendation.get('current_disk_heavy_slots')} -> {profile_recommendation.get('recommended_disk_heavy_slots')}",
				f"- max_requested_current_h5_read_slot_demand: {profile_recommendation.get('max_requested_current_h5_read_slot_demand')}",
				f"- max_requested_recommended_h5_read_slot_demand: {profile_recommendation.get('max_requested_recommended_h5_read_slot_demand')}",
				f"- max_active_recommended_h5_read_slot_demand: {profile_recommendation.get('max_active_recommended_h5_read_slot_demand')}",
				f"- h5_read_requested_interval_observations: {profile_recommendation.get('recommended_h5_read_interval_observations')}",
				f"- max_requested_current_disk_heavy_slot_demand: {profile_recommendation.get('max_requested_current_disk_heavy_slot_demand')}",
				f"- max_requested_recommended_disk_heavy_slot_demand: {profile_recommendation.get('max_requested_recommended_disk_heavy_slot_demand')}",
				f"- max_active_recommended_disk_heavy_slot_demand: {profile_recommendation.get('max_active_recommended_disk_heavy_slot_demand')}",
				f"- disk_heavy_requested_interval_observations: {profile_recommendation.get('recommended_disk_heavy_interval_observations')}",
				f"- resource_gate_wait_observations: {profile_recommendation.get('resource_gate_wait_observations')}",
				f"- max_resource_gate_wait_s: {profile_recommendation.get('max_resource_gate_wait_s')}",
				f"- total_resource_gate_wait_s: {profile_recommendation.get('total_resource_gate_wait_s')}",
				f"- max_h5_read_bandwidth_utilization: {profile_recommendation.get('max_h5_read_bandwidth_utilization')}",
				f"- max_disk_heavy_bandwidth_utilization: {profile_recommendation.get('max_disk_heavy_bandwidth_utilization')}",
			]
		)
		for warning in profile_recommendation.get("warnings", []) or []:
			lines.append(f"- warning: {warning}")
		for note in profile_recommendation.get("notes", []) or []:
			lines.append(f"- note: {note}")
	lines.extend(["", "## Disk Bandwidth Measurements"])
	measurements = summary.get("disk_bandwidth_measurements", []) or []
	if not measurements:
		lines.append("- none")
	else:
		for measurement in measurements:
			if not isinstance(measurement, dict):
				continue
			lines.extend(
				[
					f"- path: {measurement.get('path')}",
					f"  kind: {measurement.get('path_kind')}",
					f"  read_capacity_gb_per_s: {measurement.get('read_capacity_gb_per_s')}",
					f"  write_capacity_gb_per_s: {measurement.get('write_capacity_gb_per_s')}",
					f"  benchmark_bytes: {measurement.get('benchmark_bytes')}",
				]
			)
			for warning in measurement.get("warnings", []) or []:
				lines.append(f"  warning: {warning}")
			for note in measurement.get("notes", []) or []:
				lines.append(f"  note: {note}")
	lines.extend(["", "## Disk Bandwidth Utilization"])
	pressure_items = []
	if isinstance(profile_recommendation, dict):
		pressure_items = list(profile_recommendation.get("disk_bandwidth_pressure", []) or [])
	if not pressure_items:
		lines.append("- none")
	else:
		for item in pressure_items:
			if not isinstance(item, dict):
				continue
			lines.extend(
				[
					f"- path: {item.get('path')}",
					f"  kind: {item.get('path_kind')}",
					f"  max_observed_read_gb_per_s: {item.get('max_observed_read_gb_per_s')}",
					f"  max_observed_write_gb_per_s: {item.get('max_observed_write_gb_per_s')}",
					f"  max_read_utilization: {item.get('max_read_utilization')}",
					f"  max_write_utilization: {item.get('max_write_utilization')}",
					f"  max_combined_utilization: {item.get('max_combined_utilization')}",
					f"  observation_count: {item.get('observation_count')}",
				]
			)
	lines.extend(
		[
			"",
			"## Recommendations",
		]
	)
	for recommendation in summary.get("recommendations", []) or []:
		qualified_name = _qualified_stage_phase(recommendation.get("stage"), recommendation.get("phase"))
		lines.extend(
			[
				"",
				f"### {qualified_name}",
				f"- resource_class: {recommendation.get('resource_class') or 'none'}",
				f"- observations: {recommendation.get('observations', 0)}",
				f"- ram_gb: {recommendation.get('current_class_ram_gb')} -> {recommendation.get('recommended_class_ram_gb')}",
				f"- cpu_cores: {recommendation.get('current_class_cpu_cores')} -> {recommendation.get('recommended_class_cpu_cores')}",
				f"- h5_read_slots: {recommendation.get('current_h5_read_slots')} -> {recommendation.get('recommended_h5_read_slots')}",
				f"- disk_heavy_slots: {recommendation.get('current_disk_heavy_slots')} -> {recommendation.get('recommended_disk_heavy_slots')}",
				f"- max_total_peak_rss_gb: {recommendation.get('max_total_peak_rss_gb')}",
				f"- max_disk_read_gb_per_s: {recommendation.get('max_disk_read_gb_per_s')}",
				f"- max_disk_write_gb_per_s: {recommendation.get('max_disk_write_gb_per_s')}",
			]
		)
		for warning in recommendation.get("warnings", []) or []:
			lines.append(f"- warning: {warning}")
		for note in recommendation.get("notes", []) or []:
			lines.append(f"- note: {note}")
	lines.append("")
	return "\n".join(lines)


def write_phase_tuning_artifacts(
	*,
	run_root: Path,
	tuning_config: PhaseTuningConfig,
	observations: list[dict[str, Any]],
	summary: dict[str, Any],
) -> dict[str, Path]:
	output_dir = (run_root / tuning_config.output_relpath).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	observations_path = output_dir / "resource_usage_observations.jsonl"
	summary_path = output_dir / "resource_tuning_summary.json"
	report_path = output_dir / "resource_tuning_report.md"
	observations_path.write_text(
		"".join(json.dumps(item, sort_keys=True, default=str) + "\n" for item in observations),
		encoding="utf-8",
	)
	summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
	report_path.write_text(format_phase_tuning_report(summary), encoding="utf-8")
	return {
		"observations": observations_path,
		"summary": summary_path,
		"report": report_path,
	}


def emit_phase_tuning_recommendations(
	*,
	config_path: str | Path,
	selected_stages: Iterable[str],
) -> dict[str, Any]:
	logging_config = current_pipeline_logging_config()
	if logging_config is None:
		raise RuntimeError("phase tuning requires pipeline logging to be configured")
	for handler in logging.getLogger().handlers:
		try:
			handler.flush()
		except Exception:
			continue
	runtime_config = RuntimeConfig.load(Path(config_path).expanduser().resolve())
	tuning_config = parse_phase_tuning_config(runtime_config)
	resources = parse_resources_config(runtime_config=runtime_config, logger=LOGGER)
	structured_path = logging_config.structured.path
	if structured_path is None:
		observations: list[dict[str, Any]] = []
	else:
		observations = collect_phase_resource_observations_from_jsonl(
			structured_path,
			run_id=logging_config.run_id,
			selected_stages=selected_stages,
		)
	disk_measurements = collect_disk_bandwidth_measurements(
		run_root=logging_config.run_root,
		tuning_config=tuning_config,
		observations=observations,
	)
	summary = build_phase_tuning_summary(
		resources=resources,
		tuning_config=tuning_config,
		observations=observations,
		selected_stages=selected_stages,
		run_id=logging_config.run_id,
		run_root=logging_config.run_root,
		disk_measurements=disk_measurements,
	)
	paths = write_phase_tuning_artifacts(
		run_root=logging_config.run_root,
		tuning_config=tuning_config,
		observations=observations,
		summary=summary,
	)
	for recommendation in summary.get("recommendations", []) or []:
		qualified_name = _qualified_stage_phase(recommendation.get("stage"), recommendation.get("phase"))
		LOGGER.info(
			"Resource tuning recommendation: %s resource_class=%s observations=%s ram_gb=%s->%s cpu_cores=%s->%s h5_read_slots=%s->%s disk_heavy_slots=%s->%s",
			qualified_name,
			recommendation.get("resource_class") or "none",
			recommendation.get("observations", 0),
			recommendation.get("current_class_ram_gb"),
			recommendation.get("recommended_class_ram_gb"),
			recommendation.get("current_class_cpu_cores"),
			recommendation.get("recommended_class_cpu_cores"),
			recommendation.get("current_h5_read_slots"),
			recommendation.get("recommended_h5_read_slots"),
			recommendation.get("current_disk_heavy_slots"),
			recommendation.get("recommended_disk_heavy_slots"),
			extra={"event": "phase_tuning_recommendation"},
		)
	profile_recommendation = summary.get("active_profile_recommendation", {})
	if isinstance(profile_recommendation, dict):
		LOGGER.info(
			"Resource profile tuning recommendation: profile=%s h5_read_slots=%s->%s disk_heavy_slots=%s->%s h5_requested_demand=%s disk_requested_demand=%s gate_wait_max_s=%s h5_read_utilization=%s disk_heavy_utilization=%s",
			profile_recommendation.get("profile") or "none",
			profile_recommendation.get("current_h5_read_slots"),
			profile_recommendation.get("recommended_h5_read_slots"),
			profile_recommendation.get("current_disk_heavy_slots"),
			profile_recommendation.get("recommended_disk_heavy_slots"),
			profile_recommendation.get("max_requested_recommended_h5_read_slot_demand"),
			profile_recommendation.get("max_requested_recommended_disk_heavy_slot_demand"),
			profile_recommendation.get("max_resource_gate_wait_s"),
			profile_recommendation.get("max_h5_read_bandwidth_utilization"),
			profile_recommendation.get("max_disk_heavy_bandwidth_utilization"),
			extra={"event": "phase_tuning_profile_recommendation"},
		)
		for item in profile_recommendation.get("disk_bandwidth_pressure", []) or []:
			if not isinstance(item, dict):
				continue
			LOGGER.info(
				"Resource profile disk bandwidth utilization: path=%s kind=%s read_gb_per_s=%s read_capacity_gb_per_s=%s read_utilization=%s write_gb_per_s=%s write_capacity_gb_per_s=%s write_utilization=%s combined_utilization=%s observations=%s",
				item.get("path"),
				item.get("path_kind"),
				item.get("max_observed_read_gb_per_s"),
				item.get("read_capacity_gb_per_s"),
				item.get("max_read_utilization"),
				item.get("max_observed_write_gb_per_s"),
				item.get("write_capacity_gb_per_s"),
				item.get("max_write_utilization"),
				item.get("max_combined_utilization"),
				item.get("observation_count"),
				extra={
					"event": "phase_tuning_disk_bandwidth_utilization",
					"disk_bandwidth_pressure": item,
				},
			)
		for warning in profile_recommendation.get("warnings", []) or []:
			LOGGER.warning(
				"Resource profile tuning warning: %s",
				warning,
				extra={
					"event": "phase_tuning_profile_recommendation_warning",
					"profile": profile_recommendation.get("profile"),
					"recommendation_message": warning,
				},
			)
		for note in profile_recommendation.get("notes", []) or []:
			LOGGER.info(
				"Resource profile tuning note: %s",
				note,
				extra={
					"event": "phase_tuning_profile_recommendation_note",
					"profile": profile_recommendation.get("profile"),
					"recommendation_message": note,
				},
			)
	LOGGER.info(
		"Finished resource tuning run observations_written=%d summary_path=%s recommendations_path=%s",
		len(observations),
		str(paths["summary"]),
		str(paths["report"]),
		extra={"event": "phase_tuning_completed"},
	)
	return {"summary": summary, "paths": {key: str(value) for key, value in paths.items()}}