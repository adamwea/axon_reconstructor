from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .resources import (
	RESOURCE_CAPACITY_DIMENSIONS,
	ResourcesConfig,
	estimate_phase_resource_class_capacity,
	get_active_resource_profile,
	get_phase_resource_class_config,
	get_phase_resource_demand_units,
	get_profile_budget_units,
)

try:  # pragma: no cover - exercised in environments with psutil installed.
	import psutil  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - keep reporting best-effort.
	psutil = None


_NVML_INIT_LOCK = threading.Lock()
_NVML_STATE: dict[str, Any] = {"module": None, "initialized": False, "unavailable": False}
_PHASE_TUNE_LOCK = threading.Lock()
_PHASE_TUNE_CONFIG: dict[str, Any] = {
	"enabled": False,
	"system_tools_enabled": True,
	"system_tool_interval_s": 1.0,
	"output_relpath": "resource_tuning",
	"write_tool_logs": True,
	"tuning_config": None,
}
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


def configure_phase_tuning_monitoring(
	*,
	enabled: bool,
	system_tools_enabled: bool = True,
	system_tool_interval_s: float = 1.0,
	output_relpath: str = "resource_tuning",
	write_tool_logs: bool = True,
	tuning_config: Any | None = None,
) -> None:
	with _PHASE_TUNE_LOCK:
		_PHASE_TUNE_CONFIG.update(
			{
				"enabled": bool(enabled),
				"system_tools_enabled": bool(system_tools_enabled),
				"system_tool_interval_s": max(1.0, float(system_tool_interval_s or 1.0)),
				"output_relpath": str(output_relpath or "resource_tuning"),
				"write_tool_logs": bool(write_tool_logs),
				"tuning_config": tuning_config if bool(enabled) else None,
			}
		)


def phase_tuning_monitoring_enabled() -> bool:
	with _PHASE_TUNE_LOCK:
		return bool(_PHASE_TUNE_CONFIG.get("enabled", False))


def _phase_tuning_monitoring_config() -> dict[str, Any]:
	with _PHASE_TUNE_LOCK:
		return dict(_PHASE_TUNE_CONFIG)


def current_phase_tuning_config() -> Any | None:
	with _PHASE_TUNE_LOCK:
		return _PHASE_TUNE_CONFIG.get("tuning_config", None)


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
	process_peak_pss_gb: float | None = None
	child_peak_pss_gb: float | None = None
	total_peak_pss_gb: float | None = None
	cpu_time_user_s: float | None = None
	cpu_time_system_s: float | None = None
	max_threads: int | None = None
	observed_process_max_threads: int | None = None
	child_process_count_max: int | None = None
	gpu_peak_memory_gb: float | None = None
	gpu_utilization_max_pct: float | None = None
	disk_read_gb: float | None = None
	disk_write_gb: float | None = None
	phase_tune_tools: tuple[str, ...] = ()
	phase_tune_log_dir: str | None = None
	phase_tune_sample_count: int | None = None
	phase_tune_avg_cpu_pct: float | None = None
	phase_tune_peak_cpu_pct: float | None = None
	phase_tune_peak_rss_gb: float | None = None
	phase_tune_avg_read_gb_per_s: float | None = None
	phase_tune_avg_write_gb_per_s: float | None = None
	phase_tune_peak_read_gb_per_s: float | None = None
	phase_tune_peak_write_gb_per_s: float | None = None
	phase_tune_peak_device_read_mb_per_s: float | None = None
	phase_tune_peak_device_write_mb_per_s: float | None = None
	phase_tune_peak_device_await_ms: float | None = None
	phase_tune_peak_device_util_pct: float | None = None
	phase_tune_warnings: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		payload: dict[str, Any] = {
			"wall_time_s": self.wall_time_s,
			"process_peak_rss_gb": self.process_peak_rss_gb,
			"child_peak_rss_gb": self.child_peak_rss_gb,
			"total_peak_rss_gb": self.total_peak_rss_gb,
			"process_peak_pss_gb": self.process_peak_pss_gb,
			"child_peak_pss_gb": self.child_peak_pss_gb,
			"total_peak_pss_gb": self.total_peak_pss_gb,
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
		phase_tune_fields = {
			"phase_tune_tools": list(self.phase_tune_tools),
			"phase_tune_log_dir": self.phase_tune_log_dir,
			"phase_tune_sample_count": self.phase_tune_sample_count,
			"phase_tune_avg_cpu_pct": self.phase_tune_avg_cpu_pct,
			"phase_tune_peak_cpu_pct": self.phase_tune_peak_cpu_pct,
			"phase_tune_peak_rss_gb": self.phase_tune_peak_rss_gb,
			"phase_tune_avg_read_gb_per_s": self.phase_tune_avg_read_gb_per_s,
			"phase_tune_avg_write_gb_per_s": self.phase_tune_avg_write_gb_per_s,
			"phase_tune_peak_read_gb_per_s": self.phase_tune_peak_read_gb_per_s,
			"phase_tune_peak_write_gb_per_s": self.phase_tune_peak_write_gb_per_s,
			"phase_tune_peak_device_read_mb_per_s": self.phase_tune_peak_device_read_mb_per_s,
			"phase_tune_peak_device_write_mb_per_s": self.phase_tune_peak_device_write_mb_per_s,
			"phase_tune_peak_device_await_ms": self.phase_tune_peak_device_await_ms,
			"phase_tune_peak_device_util_pct": self.phase_tune_peak_device_util_pct,
			"phase_tune_warnings": list(self.phase_tune_warnings),
		}
		if any(value not in (None, [], ()) for value in phase_tune_fields.values()):
			payload.update(phase_tune_fields)
		return payload


def _to_gib_from_kb(value: int | float | None) -> float | None:
	if value is None:
		return None
	try:
		return float(value) * 1024.0 / float(1024**3)
	except Exception:
		return None


def _kb_per_s_to_gib_per_s(value: int | float | None) -> float | None:
	if value is None:
		return None
	try:
		return float(value) / float(1024**2)
	except Exception:
		return None


def _safe_file_token(value: Any) -> str:
	text = str(value or "unknown").strip() or "unknown"
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"


def _as_tool_float(value: Any) -> float | None:
	if value is None:
		return None
	text = str(value).strip()
	if not text or text == "-":
		return None
	try:
		return float(text)
	except Exception:
		return None


def _tool_rows_with_header(text: str, *, header_field: str) -> list[tuple[tuple[str, ...], dict[str, str]]]:
	rows: list[tuple[tuple[str, ...], dict[str, str]]] = []
	current_header: list[str] | None = None
	for raw_line in str(text or "").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("Linux ") or line.startswith("Average:"):
			continue
		parts = line.split()
		if header_field in parts:
			current_header = parts[parts.index(header_field) :]
			continue
		if current_header is None or len(parts) < len(current_header):
			continue
		values = parts[-len(current_header) :]
		prefix = tuple(parts[: len(parts) - len(current_header)])
		rows.append((prefix, dict(zip(current_header, values))))
	return rows


def parse_pidstat_phase_tune_output(text: str) -> dict[str, Any]:
	cpu_by_sample: dict[tuple[str, ...], float] = {}
	read_by_sample: dict[tuple[str, ...], float] = {}
	write_by_sample: dict[tuple[str, ...], float] = {}
	rss_by_sample: dict[tuple[str, ...], float] = {}
	for sample_key, row in _tool_rows_with_header(text, header_field="UID"):
		cpu_pct = _as_tool_float(row.get("%CPU"))
		if cpu_pct is not None:
			cpu_by_sample[sample_key] = float(cpu_by_sample.get(sample_key, 0.0)) + float(cpu_pct)
		read_kb_per_s = _as_tool_float(row.get("kB_rd/s"))
		if read_kb_per_s is not None:
			read_by_sample[sample_key] = float(read_by_sample.get(sample_key, 0.0)) + float(read_kb_per_s)
		write_kb_per_s = _as_tool_float(row.get("kB_wr/s"))
		if write_kb_per_s is not None:
			write_by_sample[sample_key] = float(write_by_sample.get(sample_key, 0.0)) + float(write_kb_per_s)
		rss_kb = _as_tool_float(row.get("RSS"))
		if rss_kb is not None:
			rss_by_sample[sample_key] = float(rss_by_sample.get(sample_key, 0.0)) + float(rss_kb)

	def _average(values: dict[tuple[str, ...], float]) -> float | None:
		if not values:
			return None
		return float(sum(values.values())) / float(len(values))

	peak_read_kb_per_s = max(read_by_sample.values(), default=None)
	peak_write_kb_per_s = max(write_by_sample.values(), default=None)
	avg_read_kb_per_s = _average(read_by_sample)
	avg_write_kb_per_s = _average(write_by_sample)
	return {
		"sample_count": max(len(cpu_by_sample), len(read_by_sample), len(write_by_sample), len(rss_by_sample)),
		"avg_cpu_pct": _average(cpu_by_sample),
		"peak_cpu_pct": max(cpu_by_sample.values(), default=None),
		"peak_rss_gb": _to_gib_from_kb(max(rss_by_sample.values(), default=None)),
		"avg_read_gb_per_s": _kb_per_s_to_gib_per_s(avg_read_kb_per_s),
		"avg_write_gb_per_s": _kb_per_s_to_gib_per_s(avg_write_kb_per_s),
		"peak_read_gb_per_s": _kb_per_s_to_gib_per_s(peak_read_kb_per_s),
		"peak_write_gb_per_s": _kb_per_s_to_gib_per_s(peak_write_kb_per_s),
	}


def parse_iostat_phase_tune_output(text: str) -> dict[str, Any]:
	peak_read_mb_per_s: float | None = None
	peak_write_mb_per_s: float | None = None
	peak_await_ms: float | None = None
	peak_util_pct: float | None = None
	for _sample_key, row in _tool_rows_with_header(text, header_field="Device"):
		read_mb_per_s = _as_tool_float(row.get("rMB/s"))
		if read_mb_per_s is None:
			read_kb_per_s = _as_tool_float(row.get("rkB/s"))
			read_mb_per_s = None if read_kb_per_s is None else float(read_kb_per_s) / 1024.0
		write_mb_per_s = _as_tool_float(row.get("wMB/s"))
		if write_mb_per_s is None:
			write_kb_per_s = _as_tool_float(row.get("wkB/s"))
			write_mb_per_s = None if write_kb_per_s is None else float(write_kb_per_s) / 1024.0
		await_candidates = [
			_as_tool_float(row.get("await")),
			_as_tool_float(row.get("r_await")),
			_as_tool_float(row.get("w_await")),
		]
		await_ms = max((value for value in await_candidates if value is not None), default=None)
		util_pct = _as_tool_float(row.get("%util"))
		if read_mb_per_s is not None:
			peak_read_mb_per_s = max(float(peak_read_mb_per_s or 0.0), float(read_mb_per_s))
		if write_mb_per_s is not None:
			peak_write_mb_per_s = max(float(peak_write_mb_per_s or 0.0), float(write_mb_per_s))
		if await_ms is not None:
			peak_await_ms = max(float(peak_await_ms or 0.0), float(await_ms))
		if util_pct is not None:
			peak_util_pct = max(float(peak_util_pct or 0.0), float(util_pct))
	return {
		"peak_device_read_mb_per_s": peak_read_mb_per_s,
		"peak_device_write_mb_per_s": peak_write_mb_per_s,
		"peak_device_await_ms": peak_await_ms,
		"peak_device_util_pct": peak_util_pct,
	}


class _ExternalPhaseTuneMonitor:
	def __init__(
		self,
		*,
		tool_interval_s: float,
		tool_log_dir: Path | None,
		write_tool_logs: bool,
	) -> None:
		self.tool_interval_s = max(1, int(round(float(tool_interval_s or 1.0))))
		self.tool_log_dir = tool_log_dir
		self.write_tool_logs = bool(write_tool_logs)
		self._processes: dict[str, subprocess.Popen[str]] = {}
		self._warnings: list[str] = []
		self._tools: list[str] = []

	def _start_tool(self, tool_name: str, args: list[str]) -> None:
		path = shutil.which(tool_name)
		if path is None:
			self._warnings.append(f"{tool_name} unavailable")
			return
		try:
			self._processes[tool_name] = subprocess.Popen(
				[path, *args],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
			)
		except Exception as exc:
			self._warnings.append(f"{tool_name} failed to start: {type(exc).__name__}: {exc}")
			return
		self._tools.append(tool_name)

	def start(self) -> None:
		interval = str(self.tool_interval_s)
		self._start_tool("pidstat", ["-h", "-u", "-r", "-d", "-p", "ALL", interval])
		self._start_tool("iostat", ["-x", "-d", "-m", "-y", interval])

	def _stop_process(self, tool_name: str, proc: subprocess.Popen[str]) -> str:
		if proc.poll() is None:
			try:
				proc.terminate()
			except Exception:
				pass
		try:
			stdout, _stderr = proc.communicate(timeout=3.0)
		except subprocess.TimeoutExpired:
			try:
				proc.kill()
			except Exception:
				pass
			stdout, _stderr = proc.communicate(timeout=3.0)
		if self.write_tool_logs and self.tool_log_dir is not None:
			try:
				self.tool_log_dir.mkdir(parents=True, exist_ok=True)
				(self.tool_log_dir / f"{tool_name}.txt").write_text(str(stdout or ""), encoding="utf-8")
			except Exception as exc:
				self._warnings.append(f"{tool_name} log write failed: {type(exc).__name__}: {exc}")
		if proc.returncode not in (0, None, -15):
			self._warnings.append(f"{tool_name} exited with code {proc.returncode}")
		return str(stdout or "")

	def stop(self) -> dict[str, Any]:
		outputs = {
			tool_name: self._stop_process(tool_name, proc)
			for tool_name, proc in list(self._processes.items())
		}
		pidstat_metrics = parse_pidstat_phase_tune_output(outputs.get("pidstat", ""))
		iostat_metrics = parse_iostat_phase_tune_output(outputs.get("iostat", ""))
		return {
			"tools": tuple(self._tools),
			"log_dir": None if self.tool_log_dir is None else str(self.tool_log_dir),
			"warnings": tuple(self._warnings),
			**pidstat_metrics,
			**iostat_metrics,
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
		include_phase_tune_tools: bool = False,
		phase_tune_tool_interval_s: float = 1.0,
		phase_tune_tool_log_dir: Path | None = None,
		phase_tune_write_tool_logs: bool = True,
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
		self._peak_process_pss_bytes: int | None = None
		self._peak_child_pss_bytes: int | None = None
		self._peak_total_pss_bytes: int | None = None
		self._observed_process_max_threads = 0
		self._child_process_count_max = 0
		self._gpu_peak_bytes: int | None = None
		self._gpu_utilization_max_pct: float | None = None
		self._start_io: dict[int, tuple[int, int]] = {}
		self._max_io: dict[int, tuple[int, int]] = {}
		self._lock = threading.RLock()
		self._stop_event = threading.Event()
		self._thread: threading.Thread | None = None
		self._phase_tune_external_monitor = (
			_ExternalPhaseTuneMonitor(
				tool_interval_s=phase_tune_tool_interval_s,
				tool_log_dir=phase_tune_tool_log_dir,
				write_tool_logs=phase_tune_write_tool_logs,
			)
			if bool(include_phase_tune_tools)
			else None
		)

	def _capture_sample(self, *, initial: bool = False) -> None:
		if self._process is None:
			return
		try:
			procs = [self._process]
			if self.include_children:
				procs.extend(self._process.children(recursive=True))
		except Exception:
			procs = [self._process]

		def _read_memory_bytes(proc: Any) -> tuple[int | None, int | None]:
			rss_bytes: int | None = None
			pss_bytes: int | None = None
			try:
				rss_bytes = max(0, int(proc.memory_info().rss))
			except Exception:
				rss_bytes = None
			try:
				full_info = proc.memory_full_info()
				pss = getattr(full_info, "pss", None)
				if pss is not None:
					pss_bytes = max(0, int(pss))
			except Exception:
				pss_bytes = None
			return rss_bytes, pss_bytes

		with self._lock:
			self._child_process_count_max = max(self._child_process_count_max, max(0, len(procs) - 1))
			parent_rss = None
			parent_pss = None
			total_threads = 0
			parent_rss, parent_pss = _read_memory_bytes(self._process)
			if parent_rss is not None:
				self._peak_process_rss_bytes = max(self._peak_process_rss_bytes, max(0, parent_rss))
			if parent_pss is not None:
				self._peak_process_pss_bytes = max(int(self._peak_process_pss_bytes or 0), max(0, parent_pss))

			child_rss_total = 0
			child_pss_total: int | None = 0 if parent_pss is not None else None
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
				child_rss, child_pss = _read_memory_bytes(proc)
				if child_rss is not None:
					child_rss_total += max(0, int(child_rss))
				if child_pss_total is not None:
					if child_pss is None:
						child_pss_total = None
					else:
						child_pss_total += max(0, int(child_pss))
			self._peak_child_rss_bytes = max(self._peak_child_rss_bytes, child_rss_total)
			self._peak_total_rss_bytes = max(
				self._peak_total_rss_bytes,
				max(0, int(parent_rss or 0)) + max(0, int(child_rss_total)),
			)
			if child_pss_total is not None:
				self._peak_child_pss_bytes = max(int(self._peak_child_pss_bytes or 0), child_pss_total)
				self._peak_total_pss_bytes = max(
					int(self._peak_total_pss_bytes or 0),
					max(0, int(parent_pss or 0)) + max(0, int(child_pss_total)),
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
		if self._phase_tune_external_monitor is not None:
			self._phase_tune_external_monitor.start()
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
		phase_tune_metrics = (
			{} if self._phase_tune_external_monitor is None else self._phase_tune_external_monitor.stop()
		)

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
			process_peak_pss_gb=_to_gib(self._peak_process_pss_bytes),
			child_peak_pss_gb=(_to_gib(self._peak_child_pss_bytes) if self.include_children else None),
			total_peak_pss_gb=_to_gib(
				self._peak_total_pss_bytes if self.include_children else self._peak_process_pss_bytes
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
			phase_tune_tools=tuple(phase_tune_metrics.get("tools", ()) or ()),
			phase_tune_log_dir=phase_tune_metrics.get("log_dir", None),
			phase_tune_sample_count=_coerce_nonnegative_int(phase_tune_metrics.get("sample_count", None)),
			phase_tune_avg_cpu_pct=_coerce_nonnegative(phase_tune_metrics.get("avg_cpu_pct", None)),
			phase_tune_peak_cpu_pct=_coerce_nonnegative(phase_tune_metrics.get("peak_cpu_pct", None)),
			phase_tune_peak_rss_gb=_coerce_nonnegative(phase_tune_metrics.get("peak_rss_gb", None)),
			phase_tune_avg_read_gb_per_s=_coerce_nonnegative(phase_tune_metrics.get("avg_read_gb_per_s", None)),
			phase_tune_avg_write_gb_per_s=_coerce_nonnegative(phase_tune_metrics.get("avg_write_gb_per_s", None)),
			phase_tune_peak_read_gb_per_s=_coerce_nonnegative(phase_tune_metrics.get("peak_read_gb_per_s", None)),
			phase_tune_peak_write_gb_per_s=_coerce_nonnegative(phase_tune_metrics.get("peak_write_gb_per_s", None)),
			phase_tune_peak_device_read_mb_per_s=_coerce_nonnegative(
				phase_tune_metrics.get("peak_device_read_mb_per_s", None)
			),
			phase_tune_peak_device_write_mb_per_s=_coerce_nonnegative(
				phase_tune_metrics.get("peak_device_write_mb_per_s", None)
			),
			phase_tune_peak_device_await_ms=_coerce_nonnegative(
				phase_tune_metrics.get("peak_device_await_ms", None)
			),
			phase_tune_peak_device_util_pct=_coerce_nonnegative(
				phase_tune_metrics.get("peak_device_util_pct", None)
			),
			phase_tune_warnings=tuple(str(item) for item in (phase_tune_metrics.get("warnings", ()) or ())),
		)


def start_phase_resource_monitor(
	resource_usage_config: Any | None,
	*,
	pipeline_thread_count: int | None = None,
	run_root: str | Path | None = None,
	run_id: str | None = None,
	stage_name: Any = None,
	phase_name: Any = None,
	target_label: str | None = None,
) -> PhaseResourceMonitor | None:
	phase_tune_config = _phase_tuning_monitoring_config()
	phase_tune_enabled = bool(phase_tune_config.get("enabled", False))
	if not phase_tune_enabled and (
		resource_usage_config is None or not bool(getattr(resource_usage_config, "enabled", False))
	):
		return None
	tool_log_dir = None
	if phase_tune_enabled and bool(phase_tune_config.get("write_tool_logs", True)) and run_root is not None:
		tool_log_dir = (
			Path(run_root)
			/ str(phase_tune_config.get("output_relpath", "resource_tuning") or "resource_tuning")
			/ "raw"
			/ _safe_file_token(run_id or os.getpid())
			/ _safe_file_token(stage_name)
			/ _safe_file_token(phase_name)
			/ f"{time.time_ns()}_{_safe_file_token(target_label)}_{os.getpid()}"
		)
	monitor = PhaseResourceMonitor(
		include_children=bool(getattr(resource_usage_config, "include_children", True)),
		sample_interval_s=float(getattr(resource_usage_config, "sample_interval_s", 0.5) or 0.5),
		include_gpu=bool(getattr(resource_usage_config, "include_gpu", True)),
		include_disk_io=phase_tune_enabled or bool(getattr(resource_usage_config, "include_disk_io", False)),
		pipeline_thread_count=pipeline_thread_count,
		include_phase_tune_tools=phase_tune_enabled and bool(phase_tune_config.get("system_tools_enabled", True)),
		phase_tune_tool_interval_s=float(phase_tune_config.get("system_tool_interval_s", 1.0) or 1.0),
		phase_tune_tool_log_dir=tool_log_dir,
		phase_tune_write_tool_logs=bool(phase_tune_config.get("write_tool_logs", True)),
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


def _phase_tune_recommendation_lines(recommendation: dict[str, Any] | None) -> list[str]:
	if not isinstance(recommendation, dict) or not recommendation:
		return []
	lines = ["  phase_tune_recommendation:"]
	for label, current_key, recommended_key in (
		("ram_gb", "current_class_ram_gb", "recommended_class_ram_gb"),
		("cpu_cores", "current_class_cpu_cores", "recommended_class_cpu_cores"),
		("h5_read_slots", "current_h5_read_slots", "recommended_h5_read_slots"),
		("disk_heavy_slots", "current_disk_heavy_slots", "recommended_disk_heavy_slots"),
	):
		current = recommendation.get(current_key, None)
		recommended = recommendation.get(recommended_key, None)
		if current is None and recommended is None:
			continue
		lines.append(f"    {label}={_display_metric(current)}->{_display_metric(recommended)}")
	for key in (
		"observations",
		"max_memory_peak_gb",
		"memory_peak_basis",
		"max_total_peak_rss_gb",
		"max_total_peak_pss_gb",
		"max_cpu_parallelism_estimate",
		"max_disk_read_gb_per_s",
		"max_disk_write_gb_per_s",
	):
		if recommendation.get(key, None) is not None:
			lines.append(f"    {key}={_display_metric(recommendation.get(key))}")
	for warning in recommendation.get("phase_tune_warnings", []) or []:
		lines.append(f"    phase_tune_warning={warning}")
	for warning in recommendation.get("warnings", []) or []:
		lines.append(f"    warning={warning}")
	for note in recommendation.get("notes", []) or []:
		lines.append(f"    note={note}")
	return lines


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
	phase_tune_recommendation: dict[str, Any] | None = None,
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
	lines.extend(_phase_tune_recommendation_lines(phase_tune_recommendation))
	return "\n".join(lines)


def _resource_warning_key(stage_name: Any, phase_name: Any, resource_class: Any, metric: str) -> tuple[str, str, str, str]:
	return (
		str(stage_name or "unknown"),
		str(phase_name or "unknown"),
		str(resource_class or "unknown"),
		str(metric),
	)


def _resource_usage_ram_peak_for_capacity(resource_usage: PhaseResourceUsage) -> tuple[float | None, str]:
	if resource_usage.total_peak_pss_gb is not None:
		return resource_usage.total_peak_pss_gb, "total_peak_pss_gb"
	return resource_usage.total_peak_rss_gb, "total_peak_rss_gb"


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
	observed_peak_ram_gb, observed_peak_metric = _resource_usage_ram_peak_for_capacity(resource_usage)
	if estimated_ram_gb > 0.0 and observed_peak_ram_gb is not None:
		ram_ratio = float(observed_peak_ram_gb) / float(estimated_ram_gb)
		ram_key = _resource_warning_key(stage_name, phase_name, resource_class, "ram_gb")
		if ram_ratio >= float(ram_warn_fraction):
			logger.log(
				level,
				"Phase resource usage warning: %s resource_class=%s observed_%s=%.6f estimated_ram_gb=%.6f ratio=%.2fx",
				qualified_phase_name,
				str(resource_class),
				str(observed_peak_metric),
				float(observed_peak_ram_gb),
				float(estimated_ram_gb),
				float(ram_ratio),
				extra={"event": "phase_resource_usage_warning"},
			)
			_RESOURCE_UNDERUSE_OBSERVATIONS.pop(ram_key, None)
		elif ram_ratio <= float(underuse_fraction):
			observation_count = int(_RESOURCE_UNDERUSE_OBSERVATIONS.get(ram_key, 0)) + 1
			if observation_count >= int(underuse_observation_count):
				logger.info(
					"Phase resource tuning note: %s resource_class=%s observed_%s=%.6f has stayed below estimated_ram_gb=%.6f for %d observation(s); consider tightening the class.",
					qualified_phase_name,
					str(resource_class),
					str(observed_peak_metric),
					float(observed_peak_ram_gb),
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
