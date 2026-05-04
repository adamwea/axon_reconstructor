from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

from .resources import (
	RESOURCE_CAPACITY_DIMENSIONS,
	estimate_phase_resource_class_capacity,
	get_active_resource_profile,
	get_phase_resource_class_config,
	get_phase_resource_demand_units,
	get_profile_budget_units,
	ResourcesConfig,
)


try:  # pragma: no cover - exercised in environments with psutil installed.
	import psutil  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - keep reporting best-effort.
	psutil = None


_NVML_INIT_LOCK = threading.Lock()
_NVML_STATE: dict[str, Any] = {"module": None, "initialized": False, "unavailable": False}
_RESOURCE_UNDERUSE_OBSERVATIONS: dict[tuple[str, str, str, str], int] = {}
_RESOURCE_DIMENSION_LABELS: dict[str, str] = {
	"cpu_cores": "cpu_cores",
	"ram_gb": "ram_gb",
	"gpu_sort_slots": "gpu_sort_slots",
	"h5_read_slots": "h5_read_slots",
	"disk_heavy_slots": "disk_heavy_slots",
	"plot_slots": "plot_slots",
	"analyzer_slots": "analyzer_slots",
}


def _to_gib(value: int | float | None) -> float | None:
	if value is None:
		return None
	try:
		return float(value) / float(1024 ** 3)
	except Exception:
		return None


def _coerce_nonnegative(value: float | None) -> float | None:
	if value is None:
		return None
	try:
		return max(0.0, float(value))
	except Exception:
		return None


def _coerce_nonnegative_int(value: int | None) -> int | None:
	if value is None:
		return None
	try:
		return max(0, int(value))
	except Exception:
		return None


def _resolve_summary_json_path(source: Any) -> Path | None:
	if source is None:
		return None
	if isinstance(source, (str, Path)):
		return Path(str(source)).expanduser()
	if isinstance(source, dict):
		return _resolve_summary_json_path(source.get("summary_json", None))
	return _resolve_summary_json_path(getattr(source, "summary_json", None))


def _gpu_stats_for_pids(pids: set[int]) -> tuple[int | None, float | None]:
	if not pids:
		return 0, 0.0
	with _NVML_INIT_LOCK:
		if _NVML_STATE["unavailable"]:
			return None, None
		if _NVML_STATE["module"] is None:
			try:
				import pynvml  # type: ignore[import-not-found]
			except Exception:
				_NVML_STATE["unavailable"] = True
				return None, None
			_NVML_STATE["module"] = pynvml
		if not _NVML_STATE["initialized"]:
			try:
				_NVML_STATE["module"].nvmlInit()
			except Exception:
				_NVML_STATE["unavailable"] = True
				return None, None
			_NVML_STATE["initialized"] = True

	pynvml = _NVML_STATE["module"]
	try:
		device_count = int(pynvml.nvmlDeviceGetCount())
	except Exception:
		return None, None

	process_fns = (
		getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
		getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
		getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
	)
	process_fn = next((fn for fn in process_fns if callable(fn)), None)
	if process_fn is None:
		return None, None
	utilization_fn = getattr(pynvml, "nvmlDeviceGetUtilizationRates", None)

	total_bytes = 0
	max_utilization_pct: float | None = None
	matched_device = False
	for device_index in range(device_count):
		try:
			handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
			processes = process_fn(handle)
		except Exception:
			continue
		device_matched = False
		for process in processes or ():
			try:
				pid = int(getattr(process, "pid", -1))
			except Exception:
				continue
			if pid not in pids:
				continue
			try:
				used_bytes = int(getattr(process, "usedGpuMemory", 0) or 0)
			except Exception:
				used_bytes = 0
			total_bytes += max(0, used_bytes)
			device_matched = True
		if device_matched:
			matched_device = True
			if callable(utilization_fn):
				try:
					utilization = float(getattr(utilization_fn(handle), "gpu", 0.0) or 0.0)
				except Exception:
					utilization = None
				if utilization is not None:
					utilization = max(0.0, utilization)
					max_utilization_pct = (
						utilization
						if max_utilization_pct is None
						else max(float(max_utilization_pct), utilization)
					)
	if not matched_device:
		return 0, 0.0 if callable(utilization_fn) else None
	return total_bytes, max_utilization_pct


@dataclass(frozen=True)
class PhaseResourceUsage:
	wall_time_s: float | None = None
	process_peak_rss_gb: float | None = None
	child_peak_rss_gb: float | None = None
	total_peak_rss_gb: float | None = None
	cpu_time_user_s: float | None = None
	cpu_time_system_s: float | None = None
	max_threads: int | None = None
	observed_process_max_threads: int | None = None
	child_process_count_max: int | None = None
	gpu_peak_memory_gb: float | None = None
	gpu_utilization_max_pct: float | None = None
	disk_read_gb: float | None = None
	disk_write_gb: float | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"wall_time_s": self.wall_time_s,
			"process_peak_rss_gb": self.process_peak_rss_gb,
			"child_peak_rss_gb": self.child_peak_rss_gb,
			"total_peak_rss_gb": self.total_peak_rss_gb,
			"cpu_time_user_s": self.cpu_time_user_s,
			"cpu_time_system_s": self.cpu_time_system_s,
			"max_threads": self.max_threads,
			"observed_process_max_threads": self.observed_process_max_threads,
			"child_process_count_max": self.child_process_count_max,
			"gpu_peak_memory_gb": self.gpu_peak_memory_gb,
			"gpu_utilization_max_pct": self.gpu_utilization_max_pct,
			"disk_read_gb": self.disk_read_gb,
			"disk_write_gb": self.disk_write_gb,
		}


class PhaseResourceMonitor:
	def __init__(
		self,
		*,
		include_children: bool = True,
		sample_interval_s: float = 0.5,
		include_gpu: bool = True,
		include_disk_io: bool = False,
		pipeline_thread_count: int | None = None,
	) -> None:
		self.include_children = bool(include_children)
		self.sample_interval_s = max(0.05, float(sample_interval_s))
		self.include_gpu = bool(include_gpu)
		self.include_disk_io = bool(include_disk_io)
		try:
			normalized_pipeline_thread_count = (
				int(pipeline_thread_count) if pipeline_thread_count is not None else 1
			)
		except Exception:
			normalized_pipeline_thread_count = 1
		self._pipeline_thread_count = max(1, int(normalized_pipeline_thread_count))
		self._process = None if psutil is None else psutil.Process(os.getpid())
		self._start_perf = time.perf_counter()
		self._start_cpu: dict[int, tuple[float, float]] = {}
		self._latest_cpu: dict[int, tuple[float, float]] = {}
		self._peak_process_rss_bytes = 0
		self._peak_child_rss_bytes = 0
		self._peak_total_rss_bytes = 0
		self._observed_process_max_threads = 0
		self._child_process_count_max = 0
		self._gpu_peak_bytes: int | None = None
		self._gpu_utilization_max_pct: float | None = None
		self._start_io: dict[int, tuple[int, int]] = {}
		self._max_io: dict[int, tuple[int, int]] = {}
		self._lock = threading.RLock()
		self._stop_event = threading.Event()
		self._thread: threading.Thread | None = None

	def _capture_sample(self, *, initial: bool = False) -> None:
		if self._process is None:
			return
		try:
			procs = [self._process]
			if self.include_children:
				procs.extend(self._process.children(recursive=True))
		except Exception:
			procs = [self._process]

		with self._lock:
			self._child_process_count_max = max(self._child_process_count_max, max(0, len(procs) - 1))
			parent_rss = None
			total_threads = 0
			try:
				parent_rss = int(self._process.memory_info().rss)
			except Exception:
				parent_rss = None
			if parent_rss is not None:
				self._peak_process_rss_bytes = max(self._peak_process_rss_bytes, max(0, parent_rss))

			child_rss_total = 0
			for index, proc in enumerate(procs):
				is_parent = bool(index == 0)
				try:
					total_threads += max(0, int(proc.num_threads()))
				except Exception:
					pass
				try:
					cpu_times = proc.cpu_times()
					current_cpu = (
						float(getattr(cpu_times, "user", 0.0) or 0.0),
						float(getattr(cpu_times, "system", 0.0) or 0.0),
					)
					pid = int(proc.pid)
					if pid not in self._start_cpu:
						self._start_cpu[pid] = current_cpu if initial or is_parent else (0.0, 0.0)
					last_cpu = self._latest_cpu.get(pid, self._start_cpu[pid])
					self._latest_cpu[pid] = (
						max(float(last_cpu[0]), float(current_cpu[0])),
						max(float(last_cpu[1]), float(current_cpu[1])),
					)
				except Exception:
					pass
				if is_parent:
					continue
				try:
					child_rss_total += max(0, int(proc.memory_info().rss))
				except Exception:
					continue
			self._peak_child_rss_bytes = max(self._peak_child_rss_bytes, child_rss_total)
			self._peak_total_rss_bytes = max(
				self._peak_total_rss_bytes,
				max(0, int(parent_rss or 0)) + max(0, int(child_rss_total)),
			)
			self._observed_process_max_threads = max(
				self._observed_process_max_threads,
				max(0, int(total_threads)),
			)

			if self.include_disk_io:
				for index, proc in enumerate(procs):
					is_parent = bool(index == 0)
					try:
						io_counters = proc.io_counters()
						current = (
							max(0, int(getattr(io_counters, "read_bytes", 0) or 0)),
							max(0, int(getattr(io_counters, "write_bytes", 0) or 0)),
						)
						pid = int(proc.pid)
					except Exception:
						continue
					if pid not in self._start_io:
						self._start_io[pid] = current if initial or is_parent else (0, 0)
					baseline = self._max_io.get(pid, self._start_io[pid])
					self._max_io[pid] = (
						max(int(baseline[0]), int(current[0])),
						max(int(baseline[1]), int(current[1])),
					)

			if self.include_gpu:
				gpu_bytes, gpu_utilization = _gpu_stats_for_pids(
					{int(proc.pid) for proc in procs if getattr(proc, "pid", None) is not None}
				)
				if gpu_bytes is not None:
					self._gpu_peak_bytes = max(int(self._gpu_peak_bytes or 0), int(gpu_bytes))
				if gpu_utilization is not None:
					self._gpu_utilization_max_pct = max(
						float(self._gpu_utilization_max_pct or 0.0),
						max(0.0, float(gpu_utilization)),
					)

	def start(self) -> None:
		self._start_perf = time.perf_counter()
		self._capture_sample(initial=True)
		self._thread = threading.Thread(target=self._run_sampler, name="phase-resource-monitor", daemon=True)
		self._thread.start()

	def _run_sampler(self) -> None:
		while not self._stop_event.wait(self.sample_interval_s):
			self._capture_sample(initial=False)

	def stop(self) -> PhaseResourceUsage:
		self._stop_event.set()
		if self._thread is not None:
			self._thread.join(timeout=max(0.1, self.sample_interval_s * 4.0))
		self._capture_sample(initial=False)

		wall_time_s = max(0.0, float(time.perf_counter() - self._start_perf))
		cpu_user_s: float | None = None
		cpu_system_s: float | None = None
		with self._lock:
			if self._start_cpu or self._latest_cpu:
				total_cpu_user = 0.0
				total_cpu_system = 0.0
				for pid in set(self._start_cpu) | set(self._latest_cpu):
					start_cpu = self._start_cpu.get(pid, (0.0, 0.0))
					latest_cpu = self._latest_cpu.get(pid, start_cpu)
					total_cpu_user += max(0.0, float(latest_cpu[0]) - float(start_cpu[0]))
					total_cpu_system += max(0.0, float(latest_cpu[1]) - float(start_cpu[1]))
				cpu_user_s = _coerce_nonnegative(total_cpu_user)
				cpu_system_s = _coerce_nonnegative(total_cpu_system)

		disk_read_gb: float | None = None
		disk_write_gb: float | None = None
		if self.include_disk_io:
			total_read = 0
			total_write = 0
			with self._lock:
				for pid, max_io in self._max_io.items():
					start_io = self._start_io.get(pid, (0, 0))
					total_read += max(0, int(max_io[0]) - int(start_io[0]))
					total_write += max(0, int(max_io[1]) - int(start_io[1]))
			disk_read_gb = _to_gib(total_read)
			disk_write_gb = _to_gib(total_write)

		gpu_peak_memory_gb = _to_gib(self._gpu_peak_bytes) if self.include_gpu else None
		return PhaseResourceUsage(
			wall_time_s=_coerce_nonnegative(wall_time_s),
			process_peak_rss_gb=_to_gib(self._peak_process_rss_bytes),
			child_peak_rss_gb=(_to_gib(self._peak_child_rss_bytes) if self.include_children else None),
			total_peak_rss_gb=_to_gib(
				self._peak_total_rss_bytes if self.include_children else self._peak_process_rss_bytes
			),
			cpu_time_user_s=_coerce_nonnegative(cpu_user_s),
			cpu_time_system_s=_coerce_nonnegative(cpu_system_s),
			max_threads=_coerce_nonnegative_int(self._pipeline_thread_count),
			observed_process_max_threads=_coerce_nonnegative_int(self._observed_process_max_threads),
			child_process_count_max=(
				_coerce_nonnegative_int(self._child_process_count_max) if self.include_children else None
			),
			gpu_peak_memory_gb=_coerce_nonnegative(gpu_peak_memory_gb),
			gpu_utilization_max_pct=(
				_coerce_nonnegative(self._gpu_utilization_max_pct) if self.include_gpu else None
			),
			disk_read_gb=_coerce_nonnegative(disk_read_gb),
			disk_write_gb=_coerce_nonnegative(disk_write_gb),
		)


def start_phase_resource_monitor(
	resource_usage_config: Any | None,
	*,
	pipeline_thread_count: int | None = None,
) -> PhaseResourceMonitor | None:
	if resource_usage_config is None or not bool(getattr(resource_usage_config, "enabled", False)):
		return None
	monitor = PhaseResourceMonitor(
		include_children=bool(getattr(resource_usage_config, "include_children", True)),
		sample_interval_s=float(getattr(resource_usage_config, "sample_interval_s", 0.5) or 0.5),
		include_gpu=bool(getattr(resource_usage_config, "include_gpu", True)),
		include_disk_io=bool(getattr(resource_usage_config, "include_disk_io", False)),
		pipeline_thread_count=pipeline_thread_count,
	)
	monitor.start()
	return monitor


def update_phase_summary_metadata(
	*,
	summary_source: Any,
	resource_class: str | None = None,
	resource_usage: PhaseResourceUsage | None = None,
	resource_gate: dict[str, Any] | None = None,
) -> bool:
	summary_json = _resolve_summary_json_path(summary_source)
	if summary_json is None or not summary_json.exists():
		return False
	try:
		payload = json.loads(summary_json.read_text(encoding="utf-8"))
	except Exception:
		return False
	if not isinstance(payload, dict):
		return False
	if resource_class is not None:
		payload["resource_class"] = str(resource_class)
	if resource_usage is not None:
		payload["resource_usage"] = resource_usage.to_dict()
	if resource_gate is not None:
		payload["resource_gate"] = dict(resource_gate)
	summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return True


def _display_field(value: Any) -> str:
	if value is None:
		return "null"
	text = str(value).strip()
	return text if text else "null"


def _display_metric(value: Any) -> str:
	if value is None:
		return "null"
	if isinstance(value, bool):
		return "true" if bool(value) else "false"
	if isinstance(value, int):
		return str(value)
	try:
		return f"{float(value):.6f}"
	except Exception:
		return str(value)


def _qualified_phase_name(stage_name: Any, phase_name: Any) -> str:
	stage = str(stage_name).strip() if stage_name is not None else ""
	phase = str(phase_name).strip() if phase_name is not None else ""
	if stage and phase:
		return f"{stage}.{phase}"
	return stage or phase or "unknown"


def format_phase_message(
	*,
	action: str,
	stage_name: Any,
	phase_name: Any,
	dataset_id: Any = None,
	recording_id: Any = None,
	well_id: Any = None,
	resource_class: Any = None,
	exception_type: str | None = None,
) -> str:
	parts = [
		f"dataset={_display_field(dataset_id)}",
		f"recording={_display_field(recording_id)}",
		f"well={_display_field(well_id)}",
		f"resource_class={_display_field(resource_class)}",
	]
	if exception_type is not None:
		parts.append(f"exception_type={_display_field(exception_type)}")
	return f"{str(action).strip()} phase: {_qualified_phase_name(stage_name, phase_name)} [{', '.join(parts)}]"


def format_phase_resource_usage_message(
	*,
	stage_name: Any,
	phase_name: Any,
	dataset_id: Any = None,
	recording_id: Any = None,
	well_id: Any = None,
	resource_class: Any = None,
	status: str,
	resource_usage: PhaseResourceUsage,
	exception_type: str | None = None,
) -> str:
	lines = [
		f"Phase resource usage: {_qualified_phase_name(stage_name, phase_name)}",
		f"  dataset={_display_field(dataset_id)}",
		f"  recording={_display_field(recording_id)}",
		f"  well={_display_field(well_id)}",
		f"  resource_class={_display_field(resource_class)}",
		f"  status={_display_field(status)}",
	]
	if exception_type is not None:
		lines.append(f"  exception_type={_display_field(exception_type)}")
	for key, value in resource_usage.to_dict().items():
		lines.append(f"  {key}={_display_metric(value)}")
	return "\n".join(lines)


def _resource_warning_key(stage_name: Any, phase_name: Any, resource_class: Any, metric: str) -> tuple[str, str, str, str]:
	return (
		str(stage_name or "unknown"),
		str(phase_name or "unknown"),
		str(resource_class or "unknown"),
		str(metric),
	)


def log_phase_resource_plan_warnings(
	*,
	logger: logging.Logger | None,
	resource_usage_config: Any,
	resources: ResourcesConfig | None,
	stage_name: Any,
	phase_name: Any,
	resource_class: str | None,
	well_workers: int,
	planned_target_count: int,
) -> None:
	if logger is None or resources is None or resource_class is None:
		return
	warning_config = getattr(resource_usage_config, "warnings", None)
	if warning_config is not None and not bool(getattr(warning_config, "enabled", True)):
		return
	phase_config = get_phase_resource_class_config(resources, resource_class)
	profile = get_active_resource_profile(resources)
	if phase_config is None or profile is None:
		return
	level = int(getattr(warning_config, "level", logging.WARNING)) if warning_config is not None else logging.WARNING
	plan_fraction_threshold = (
		float(getattr(warning_config, "plan_fraction_threshold", 0.8))
		if warning_config is not None
		else 0.8
	)
	estimated_capacity = estimate_phase_resource_class_capacity(resources=resources, resource_class=resource_class)
	budgets = get_profile_budget_units(resources=resources, dimensions=RESOURCE_CAPACITY_DIMENSIONS)
	demands = get_phase_resource_demand_units(
		resources=resources,
		resource_class=resource_class,
		dimensions=RESOURCE_CAPACITY_DIMENSIONS,
	)
	budget_pressure = [
		f"{_RESOURCE_DIMENSION_LABELS.get(str(dimension), str(dimension))}={int(demands.get(str(dimension), 0))}/{int(budgets.get(str(dimension), 0))}"
		for dimension in RESOURCE_CAPACITY_DIMENSIONS
		if int(demands.get(str(dimension), 0)) > 0
		and int(budgets.get(str(dimension), 0)) > 0
		and (float(demands.get(str(dimension), 0)) / float(budgets.get(str(dimension), 0))) >= float(plan_fraction_threshold)
	]
	qualified_phase_name = _qualified_phase_name(stage_name, phase_name)
	profile_name = str(resources.active_profile or "unknown")
	if estimated_capacity is not None and int(well_workers) > int(estimated_capacity):
		logger.log(
			level,
			"Phase resource plan warning: %s resource_class=%s profile=%s current_well_workers=%d estimated_capacity=%d planned_targets=%d budget_pressure=%s",
			qualified_phase_name,
			str(resource_class),
			profile_name,
			max(1, int(well_workers)),
			max(0, int(estimated_capacity)),
			max(0, int(planned_target_count)),
			(", ".join(budget_pressure) if budget_pressure else "none"),
			extra={"event": "phase_resource_plan_warning"},
		)
		return
	if budget_pressure and estimated_capacity == 1:
		logger.log(
			level,
			"Phase resource plan warning: %s resource_class=%s profile=%s estimated_capacity=1 budget_pressure=%s",
			qualified_phase_name,
			str(resource_class),
			profile_name,
			", ".join(budget_pressure),
			extra={"event": "phase_resource_plan_warning"},
		)


def log_phase_resource_observation_warnings(
	*,
	logger: logging.Logger | None,
	resource_usage_config: Any,
	resources: ResourcesConfig | None,
	stage_name: Any,
	phase_name: Any,
	resource_class: str | None,
	resource_usage: PhaseResourceUsage | None,
) -> None:
	if logger is None or resources is None or resource_class is None or resource_usage is None:
		return
	warning_config = getattr(resource_usage_config, "warnings", None)
	if warning_config is not None and not bool(getattr(warning_config, "enabled", True)):
		return
	phase_config = get_phase_resource_class_config(resources, resource_class)
	if phase_config is None:
		return
	level = int(getattr(warning_config, "level", logging.WARNING)) if warning_config is not None else logging.WARNING
	ram_warn_fraction = (
		float(getattr(warning_config, "observed_ram_warn_fraction", 1.25))
		if warning_config is not None
		else 1.25
	)
	thread_warn_fraction = (
		float(getattr(warning_config, "observed_thread_warn_fraction", 1.5))
		if warning_config is not None
		else 1.5
	)
	underuse_fraction = (
		float(getattr(warning_config, "underuse_fraction", 0.25))
		if warning_config is not None
		else 0.25
	)
	underuse_observation_count = (
		max(1, int(getattr(warning_config, "underuse_observation_count", 5)))
		if warning_config is not None
		else 5
	)
	qualified_phase_name = _qualified_phase_name(stage_name, phase_name)

	estimated_ram_gb = float(getattr(phase_config, "ram_gb", 0.0) or 0.0)
	observed_total_peak_rss_gb = resource_usage.total_peak_rss_gb
	if estimated_ram_gb > 0.0 and observed_total_peak_rss_gb is not None:
		ram_ratio = float(observed_total_peak_rss_gb) / float(estimated_ram_gb)
		ram_key = _resource_warning_key(stage_name, phase_name, resource_class, "ram_gb")
		if ram_ratio >= float(ram_warn_fraction):
			logger.log(
				level,
				"Phase resource usage warning: %s resource_class=%s observed_total_peak_rss_gb=%.6f estimated_ram_gb=%.6f ratio=%.2fx",
				qualified_phase_name,
				str(resource_class),
				float(observed_total_peak_rss_gb),
				float(estimated_ram_gb),
				float(ram_ratio),
				extra={"event": "phase_resource_usage_warning"},
			)
			_RESOURCE_UNDERUSE_OBSERVATIONS.pop(ram_key, None)
		elif ram_ratio <= float(underuse_fraction):
			observation_count = int(_RESOURCE_UNDERUSE_OBSERVATIONS.get(ram_key, 0)) + 1
			if observation_count >= int(underuse_observation_count):
				logger.info(
					"Phase resource tuning note: %s resource_class=%s observed_total_peak_rss_gb=%.6f has stayed below estimated_ram_gb=%.6f for %d observation(s); consider tightening the class.",
					qualified_phase_name,
					str(resource_class),
					float(observed_total_peak_rss_gb),
					float(estimated_ram_gb),
					int(underuse_observation_count),
					extra={"event": "phase_resource_tuning_note"},
				)
				observation_count = 0
			_RESOURCE_UNDERUSE_OBSERVATIONS[ram_key] = observation_count
		else:
			_RESOURCE_UNDERUSE_OBSERVATIONS.pop(ram_key, None)

	estimated_cpu_cores = max(0, int(getattr(phase_config, "cpu_cores", 0) or 0))
	pipeline_max_threads = resource_usage.max_threads
	if estimated_cpu_cores > 0 and pipeline_max_threads is not None:
		thread_ratio = float(pipeline_max_threads) / float(estimated_cpu_cores)
		if thread_ratio >= float(thread_warn_fraction):
			logger.log(
				level,
				"Phase resource usage warning: %s resource_class=%s pipeline_max_threads=%d estimated_cpu_cores=%d ratio=%.2fx",
				qualified_phase_name,
				str(resource_class),
				int(pipeline_max_threads),
				int(estimated_cpu_cores),
				float(thread_ratio),
				extra={"event": "phase_resource_usage_warning"},
			)
