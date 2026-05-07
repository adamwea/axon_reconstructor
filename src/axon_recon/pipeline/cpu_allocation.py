from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Callable, Iterable


LOGGER = logging.getLogger("axon_recon.pipeline.cpu_allocation")


@dataclass(frozen=True)
class CpuCoreTopology:
	package_id: int
	core_id: int
	logical_cpus: tuple[int, ...]


@dataclass(frozen=True)
class CpuTopology:
	visible_cpus: tuple[int, ...]
	cores: tuple[CpuCoreTopology, ...]
	source: str = "sysfs"
	warning: str | None = None

	@property
	def logical_cpu_count(self) -> int:
		return len(self.visible_cpus)

	@property
	def physical_core_count(self) -> int:
		return len(self.cores)

	@property
	def package_ids(self) -> tuple[int, ...]:
		ordered: list[int] = []
		seen: set[int] = set()
		for core in self.cores:
			package_id = int(core.package_id)
			if package_id in seen:
				continue
			seen.add(package_id)
			ordered.append(package_id)
		return tuple(ordered)

	@property
	def package_count(self) -> int:
		return len(self.package_ids)

	@property
	def visible_threads_per_core(self) -> tuple[int, ...]:
		thread_counts = {
			len(core.logical_cpus)
			for core in self.cores
			if len(core.logical_cpus) > 0
		}
		return tuple(sorted(int(count) for count in thread_counts))


def _default_affinity_getter(pid: int) -> Iterable[int]:
	sched_getaffinity = getattr(os, "sched_getaffinity", None)
	if callable(sched_getaffinity):
		return sched_getaffinity(pid)
	cpu_count = max(1, int(os.cpu_count() or 1))
	return range(cpu_count)


def _normalize_cpu_ids(values: Iterable[int]) -> tuple[int, ...]:
	parsed: list[int] = []
	seen: set[int] = set()
	for raw_value in values:
		cpu_id = int(raw_value)
		if cpu_id < 0 or cpu_id in seen:
			continue
		seen.add(cpu_id)
		parsed.append(cpu_id)
	if not parsed:
		raise ValueError("No visible CPUs detected")
	return tuple(sorted(parsed))


def _parse_cpu_list_spec(raw: str) -> tuple[int, ...]:
	text = str(raw).strip()
	if not text:
		raise ValueError("Empty CPU list spec")
	values: list[int] = []
	for token in text.split(","):
		item = token.strip()
		if not item:
			continue
		if "-" in item:
			start_text, end_text = item.split("-", 1)
			start = int(start_text.strip())
			end = int(end_text.strip())
			if end < start:
				raise ValueError(f"Invalid CPU range {item!r}")
			values.extend(range(start, end + 1))
			continue
		values.append(int(item))
	return _normalize_cpu_ids(values)


def format_cpu_set(cpus: Iterable[int]) -> str:
	values = _normalize_cpu_ids(cpus)
	ranges: list[str] = []
	start = values[0]
	end = values[0]
	for cpu_id in values[1:]:
		if cpu_id == end + 1:
			end = cpu_id
			continue
		ranges.append(f"{start}-{end}" if start != end else str(start))
		start = cpu_id
		end = cpu_id
	ranges.append(f"{start}-{end}" if start != end else str(start))
	return ",".join(ranges)


def _read_required_text(path: Path) -> str:
	return path.read_text(encoding="utf-8").strip()


def _logical_cpu_fallback(*, visible_cpus: tuple[int, ...], warning: str | None) -> CpuTopology:
	return CpuTopology(
		visible_cpus=visible_cpus,
		cores=tuple(
			CpuCoreTopology(package_id=0, core_id=int(cpu_id), logical_cpus=(int(cpu_id),))
			for cpu_id in visible_cpus
		),
		source="logical_fallback",
		warning=warning,
	)


def _topology_from_sysfs(*, visible_cpus: tuple[int, ...], sysfs_root: Path) -> CpuTopology:
	visible_set = set(visible_cpus)
	grouped_cores: dict[tuple[int, int], set[int]] = {}
	for cpu_id in visible_cpus:
		topology_root = sysfs_root / f"cpu{cpu_id}" / "topology"
		package_id = int(_read_required_text(topology_root / "physical_package_id"))
		core_id = int(_read_required_text(topology_root / "core_id"))
		sibling_ids = tuple(
			sibling_id
			for sibling_id in _parse_cpu_list_spec(_read_required_text(topology_root / "thread_siblings_list"))
			if sibling_id in visible_set
		)
		if cpu_id not in sibling_ids:
			sibling_ids = tuple(sorted({int(cpu_id), *sibling_ids}))
		grouped_cores.setdefault((package_id, core_id), set()).update(sibling_ids)

	cores = tuple(
		CpuCoreTopology(
			package_id=int(package_id),
			core_id=int(core_id),
			logical_cpus=tuple(sorted(int(cpu_id) for cpu_id in logical_cpus)),
		)
		for (package_id, core_id), logical_cpus in sorted(
			grouped_cores.items(),
			key=lambda item: (int(item[0][0]), min(int(cpu_id) for cpu_id in item[1]), int(item[0][1])),
		)
	)
	return CpuTopology(visible_cpus=visible_cpus, cores=cores, source="sysfs", warning=None)


def detect_cpu_topology(
	*,
	sysfs_root: str | Path = "/sys/devices/system/cpu",
	affinity_getter: Callable[[int], Iterable[int]] | None = None,
	logger: logging.Logger | None = None,
) -> CpuTopology:
	resolved_logger = logger or LOGGER
	visible_cpus = _normalize_cpu_ids((affinity_getter or _default_affinity_getter)(0))
	resolved_sysfs_root = Path(sysfs_root)
	try:
		return _topology_from_sysfs(visible_cpus=visible_cpus, sysfs_root=resolved_sysfs_root)
	except Exception as exc:
		warning = (
			"CPU topology sysfs data unavailable under "
			f"{resolved_sysfs_root}; falling back to visible logical CPUs only ({exc})"
		)
		resolved_logger.warning(warning)
		return _logical_cpu_fallback(visible_cpus=visible_cpus, warning=warning)


def format_cpu_topology(topology: CpuTopology) -> str:
	thread_counts = topology.visible_threads_per_core
	if not thread_counts:
		thread_summary = "0"
	elif len(thread_counts) == 1:
		thread_summary = str(thread_counts[0])
	else:
		thread_summary = ",".join(str(value) for value in thread_counts)

	lines = [
		"CPU topology",
		f"source: {topology.source}",
		f"visible_cpus: {format_cpu_set(topology.visible_cpus)}",
		f"logical_cpu_count: {topology.logical_cpu_count}",
		f"physical_core_count: {topology.physical_core_count}",
		f"package_count: {topology.package_count}",
		f"visible_threads_per_core: {thread_summary}",
	]
	if topology.warning:
		lines.append(f"warning: {topology.warning}")
	lines.append("core_map:")
	active_package: int | None = None
	for core in topology.cores:
		if active_package != int(core.package_id):
			active_package = int(core.package_id)
			lines.append(f"  package {active_package}:")
		lines.append(f"    core {int(core.core_id)} -> {format_cpu_set(core.logical_cpus)}")
	return "\n".join(lines)
