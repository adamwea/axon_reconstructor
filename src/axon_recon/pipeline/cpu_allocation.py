from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .resources import PhaseResourceClassConfig, ResourceProfileConfig, TaskAllocationConfig


LOGGER = logging.getLogger("axon_recon.pipeline.cpu_allocation")
_CURRENT_TASK_SLOT: ContextVar["TaskSlot | None"] = ContextVar("axon_recon_task_slot", default=None)
_CURRENT_TASK_ALLOCATION_CONTEXT: ContextVar["dict[str, Any] | None"] = ContextVar(
	"axon_recon_task_allocation_context", default=None
)
_CURRENT_PHASE_BUDGETS: ContextVar["dict[str, PhaseResourceClassConfig] | None"] = ContextVar(
	"axon_recon_phase_budgets", default=None
)


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


@dataclass(frozen=True)
class TaskSlot:
	slot_id: int
	logical_cpus: tuple[int, ...]
	core_ids: tuple[int, ...]
	package_ids: tuple[int, ...]

	@property
	def cpu_count(self) -> int:
		return len(self.logical_cpus)

	@property
	def physical_core_count(self) -> int:
		return len(self.core_ids)


def current_task_slot() -> TaskSlot | None:
	return _CURRENT_TASK_SLOT.get()


@contextmanager
def task_slot_context(slot: TaskSlot | None) -> Iterator[None]:
	token = _CURRENT_TASK_SLOT.set(slot)
	try:
		yield
	finally:
		_CURRENT_TASK_SLOT.reset(token)


def current_task_allocation_context() -> dict[str, Any] | None:
	return _CURRENT_TASK_ALLOCATION_CONTEXT.get()


@contextmanager
def task_allocation_context(metadata: dict[str, Any] | None) -> Iterator[None]:
	token = _CURRENT_TASK_ALLOCATION_CONTEXT.set(metadata)
	try:
		yield
	finally:
		_CURRENT_TASK_ALLOCATION_CONTEXT.reset(token)


def current_phase_budgets() -> dict[str, PhaseResourceClassConfig] | None:
	"""Return the active phase_budgets mapping (stage.phase -> config), or None if unset."""
	return _CURRENT_PHASE_BUDGETS.get()


@contextmanager
def phase_budgets_context(
	budgets: dict[str, PhaseResourceClassConfig] | None,
) -> Iterator[None]:
	"""Bind the active resources.phase_budgets dict to the current context."""
	token = _CURRENT_PHASE_BUDGETS.set(budgets)
	try:
		yield
	finally:
		_CURRENT_PHASE_BUDGETS.reset(token)


def current_phase_budget(stage: str, phase: str) -> PhaseResourceClassConfig | None:
	"""Look up the phase budget entry for ``<stage>.<phase>`` in the active context.

	Returns None if no budget context is active or no entry is configured for the
	given stage/phase pair. Callers must tolerate None — phases without budgets
	default to inheriting the active task slot's CPU count.
	"""
	budgets = _CURRENT_PHASE_BUDGETS.get()
	if budgets is None:
		return None
	key = f"{stage}.{phase}"
	return budgets.get(key)


def resolve_inner_worker_count(
	*,
	nested_shape: str,
	phase_cpus_per_task: int | None,
	yaml_n_jobs_override: int | None,
	work_item_count: int | None,
	stage_name: str | None = None,
	phase_name: str | None = None,
) -> int:
	"""Compute the per-well inner worker count from the active task slot.

	Rules:
		- If ``nested_shape == "serial"``, return 1 unconditionally.
		- Base = ``current_task_slot().cpu_count``, or 1 if no slot is active.
		- If ``phase_cpus_per_task`` is a positive int, base = min(base, phase_cpus_per_task).
		- If ``yaml_n_jobs_override`` is a positive int, base = min(base, override).
		- If ``work_item_count`` is a positive int, base = min(base, work_item_count).
		- Result is always >= 1.

	Phase RAM is *not* a clamp here; it is enforced by the live-resource gate at
	phase entry. Inner thread count is purely a CPU question.
	"""
	if str(nested_shape).strip() == "serial":
		result = 1
	else:
		slot = _CURRENT_TASK_SLOT.get()
		if slot is not None and int(slot.cpu_count) > 0:
			base = int(slot.cpu_count)
		else:
			base = 1
		if isinstance(phase_cpus_per_task, int) and int(phase_cpus_per_task) > 0:
			base = min(base, int(phase_cpus_per_task))
		if isinstance(yaml_n_jobs_override, int) and int(yaml_n_jobs_override) > 0:
			base = min(base, int(yaml_n_jobs_override))
		if isinstance(work_item_count, int) and int(work_item_count) > 0:
			base = min(base, int(work_item_count))
		result = max(1, int(base))
	if stage_name is not None and phase_name is not None:
		slot = _CURRENT_TASK_SLOT.get()
		slot_cpus = int(slot.cpu_count) if slot is not None and int(slot.cpu_count) > 0 else 1
		LOGGER.info(
			"phase parallelism stage=%s phase=%s nested_shape=%s slot_cpus=%d phase_cap=%s effective=%d",
			str(stage_name),
			str(phase_name),
			str(nested_shape),
			slot_cpus,
			str(phase_cpus_per_task) if phase_cpus_per_task is not None else "none",
			result,
			extra={
				"event": "phase_parallelism",
				"phase_parallelism_stage": str(stage_name),
				"phase_parallelism_phase": str(phase_name),
				"nested_shape": str(nested_shape),
				"slot_cpus": slot_cpus,
				"phase_cap": phase_cpus_per_task,
				"effective_workers": result,
			},
		)
	return result


@dataclass(frozen=True)
class TaskAllocationPlan:
	backend: str
	bind: str
	use_hyperthreads: bool
	topology: CpuTopology
	cpus_per_task: int
	cpus_per_task_source: str
	requested_tasks_per_node: int | str
	effective_tasks_per_node: int
	slot_capacity: int
	cpu_capacity_tasks: int
	ram_capacity_tasks: int | None
	shm_capacity_tasks: int | None
	available_unit_count: int
	reserved_unit_count: int
	set_thread_env: bool = False
	nested_thread_policy: str = "preserve_existing"
	task_unit: str = "well"
	target_count: int | None = None
	keyed_read_cap: int | None = None
	slots: tuple[TaskSlot, ...] = ()


def _default_affinity_getter(pid: int) -> Iterable[int]:
	sched_getaffinity = getattr(os, "sched_getaffinity", None)
	if callable(sched_getaffinity):
		return sched_getaffinity(pid)
	cpu_count = max(1, int(os.cpu_count() or 1))
	return range(cpu_count)


def _default_affinity_setter(pid: int, cpus: Iterable[int]) -> None:
	sched_setaffinity = getattr(os, "sched_setaffinity", None)
	if not callable(sched_setaffinity):
		raise RuntimeError("os.sched_setaffinity is not available on this platform")
	sched_setaffinity(pid, set(int(cpu_id) for cpu_id in cpus))


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


@contextmanager
def task_slot_affinity_context(
	slot: TaskSlot | None,
	*,
	enabled: bool,
	soft_failure: bool = True,
	logger: logging.Logger | None = None,
	affinity_getter: Callable[[int], Iterable[int]] | None = None,
	affinity_setter: Callable[[int, Iterable[int]], None] | None = None,
) -> Iterator[None]:
	if slot is None or not bool(enabled):
		yield
		return
	log = logger if logger is not None else LOGGER
	get_affinity = affinity_getter or _default_affinity_getter
	set_affinity = affinity_setter or _default_affinity_setter
	requested_cpus = tuple(int(cpu_id) for cpu_id in slot.logical_cpus)
	previous_cpus: tuple[int, ...] | None = None
	applied = False
	try:
		previous_cpus = _normalize_cpu_ids(get_affinity(0))
		set_affinity(0, requested_cpus)
		applied = True
		log.info(
			"Applied task CPU affinity task_slot=%d cpus=%s previous_cpus=%s",
			int(slot.slot_id),
			format_cpu_set(requested_cpus),
			format_cpu_set(previous_cpus),
			extra={
				"event": "task_affinity_applied",
				"task_slot_id": int(slot.slot_id),
				"task_cpu_set": format_cpu_set(requested_cpus),
				"task_previous_cpu_set": format_cpu_set(previous_cpus),
			},
		)
	except Exception as exc:
		message = f"Failed to apply task CPU affinity task_slot={int(slot.slot_id)} cpus={format_cpu_set(requested_cpus)} error={exc}"
		if not bool(soft_failure):
			raise RuntimeError(message) from exc
		log.warning(
			message,
			extra={
				"event": "task_affinity_apply_failed",
				"task_slot_id": int(slot.slot_id),
				"task_cpu_set": format_cpu_set(requested_cpus),
				"task_affinity_error": str(exc),
			},
		)
	try:
		yield
	finally:
		if applied and previous_cpus is not None:
			try:
				set_affinity(0, previous_cpus)
			except Exception as exc:
				log.warning(
					"Failed to restore task CPU affinity task_slot=%d cpus=%s error=%s",
					int(slot.slot_id),
					format_cpu_set(previous_cpus),
					str(exc),
					extra={
						"event": "task_affinity_restore_failed",
						"task_slot_id": int(slot.slot_id),
						"task_cpu_set": format_cpu_set(previous_cpus),
						"task_affinity_error": str(exc),
					},
				)


_THREAD_ENV_VARS: tuple[str, ...] = (
	"OMP_NUM_THREADS",
	"MKL_NUM_THREADS",
	"OPENBLAS_NUM_THREADS",
	"NUMEXPR_NUM_THREADS",
	"VECLIB_MAXIMUM_THREADS",
	"NUMBA_NUM_THREADS",
)


@contextmanager
def apply_thread_env_context(
	slot: TaskSlot | None,
	*,
	enabled: bool,
	policy: str,
	logger: logging.Logger | None = None,
) -> Iterator[None]:
	"""Optionally set native thread-count env vars for the duration of a worker call.

	Args:
		slot: The task slot for the current worker, used to derive the CPU count.
		enabled: Whether thread-env application is enabled (``set_thread_env`` config).
		policy: One of ``match_cpus_per_task``, ``force_1``, or ``preserve_existing``.
		logger: Logger to use; falls back to the module logger.
	"""
	log = logger if logger is not None else LOGGER
	if not bool(enabled) or str(policy).strip() == "preserve_existing":
		if bool(enabled) and str(policy).strip() == "preserve_existing":
			existing = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
			log.info(
				"Thread env policy=preserve_existing; logging existing values: %s",
				" ".join(
					f"{var}={val if val is not None else '(unset)'}"
					for var, val in existing.items()
				),
				extra={"event": "thread_env_preserved", "thread_env_policy": "preserve_existing"},
			)
		yield
		return

	if str(policy).strip() == "force_1":
		thread_count = 1
	elif str(policy).strip() == "match_cpus_per_task":
		thread_count = int(slot.cpu_count) if slot is not None else 1
	else:
		log.warning(
			"Unknown nested_thread_policy=%r; skipping thread env application.",
			str(policy),
			extra={"event": "thread_env_unknown_policy", "thread_env_policy": str(policy)},
		)
		yield
		return

	previous: dict[str, str | None] = {}
	applied_vars: dict[str, str] = {}
	for var in _THREAD_ENV_VARS:
		previous[var] = os.environ.get(var)
		os.environ[var] = str(thread_count)
		applied_vars[var] = str(thread_count)

	log.info(
		"Applied thread env policy=%s thread_count=%d task_slot=%s vars=%s",
		str(policy),
		int(thread_count),
		str(slot.slot_id) if slot is not None else "none",
		" ".join(f"{var}={val}" for var, val in applied_vars.items()),
		extra={
			"event": "thread_env_applied",
			"thread_env_policy": str(policy),
			"thread_env_count": int(thread_count),
		},
	)
	try:
		yield
	finally:
		for var in _THREAD_ENV_VARS:
			prev = previous.get(var)
			if prev is None:
				os.environ.pop(var, None)
			else:
				os.environ[var] = prev


def capture_sample_worker_environment(
	slot: TaskSlot | None,
	*,
	plan: Any | None = None,
	logger: logging.Logger | None = None,
) -> dict[str, str]:
	"""Capture what environment would be set for a sample worker in this slot.

	Enters the task allocation contexts and captures the resulting environment state.
	Used for validation during --alloc preview mode.
	"""
	if plan is None:
		return {}

	apply_affinity = bool(
		str(getattr(plan, "backend", "none")) == "local_affinity"
		and str(getattr(plan, "bind", "none")) != "none"
	)
	set_thread_env = bool(getattr(plan, "set_thread_env", False))
	thread_policy = str(getattr(plan, "nested_thread_policy", "preserve_existing") or "preserve_existing")

	captured = {}
	try:
		with task_slot_affinity_context(
			slot,
			enabled=apply_affinity,
			soft_failure=True,
			logger=logger,
		), apply_thread_env_context(
			slot,
			enabled=set_thread_env,
			policy=thread_policy,
			logger=logger,
		):
			# Capture the environment state inside the contexts
			captured["thread_env"] = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
			if slot is not None:
				captured["cpu_affinity"] = format_cpu_set(slot.logical_cpus)
				captured["cpu_count"] = int(slot.cpu_count)
				captured["slot_id"] = int(slot.slot_id)
	except Exception as exc:
		captured["error"] = str(exc)

	return captured


def _read_required_text(path: Path) -> str:
	return path.read_text(encoding="utf-8").strip()


def _as_optional_positive_int(value: int | None) -> int | None:
	if value is None:
		return None
	parsed = int(value)
	if parsed <= 0:
		return None
	return parsed


def _capacity_limit_from_float(*, budget: float | None, demand: float | None) -> int | None:
	if budget is None or demand is None:
		return None
	budget_value = float(budget)
	demand_value = float(demand)
	if budget_value <= 0.0 or demand_value <= 0.0:
		return None
	return max(0, int(budget_value // demand_value))


def _derive_cpus_per_task(
	*,
	config: TaskAllocationConfig,
) -> tuple[int, str]:
	configured = getattr(config, "cpus_per_task", "auto")
	if isinstance(configured, int):
		return max(1, int(configured)), "task_allocation.cpus_per_task"
	if str(configured).strip().lower() != "auto":
		return max(1, int(configured)), "task_allocation.cpus_per_task"
	return 1, "default"


def _logical_cpu_owner_map(topology: CpuTopology) -> dict[int, CpuCoreTopology]:
	owners: dict[int, CpuCoreTopology] = {}
	for core in topology.cores:
		for cpu_id in core.logical_cpus:
			owners[int(cpu_id)] = core
	return owners


def _allocation_units(
	*,
	topology: CpuTopology,
	bind: str,
	use_hyperthreads: bool,
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
	if str(bind) == "logical_cpus":
		owners = _logical_cpu_owner_map(topology)
		units: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
		for cpu_id in topology.visible_cpus:
			owner = owners.get(int(cpu_id))
			core_ids = () if owner is None else (int(owner.core_id),)
			package_ids = () if owner is None else (int(owner.package_id),)
			units.append(((int(cpu_id),), core_ids, package_ids))
		return units
	units = []
	for core in topology.cores:
		logical_cpus = core.logical_cpus if bool(use_hyperthreads) else (int(core.logical_cpus[0]),)
		units.append((logical_cpus, (int(core.core_id),), (int(core.package_id),)))
	return units


def _limit_units_by_profile_cpu(
	*,
	units: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
	resource_profile: ResourceProfileConfig | None,
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
	if resource_profile is None or resource_profile.cpu_cores is None:
		return list(units)
	limit = max(0, int(resource_profile.cpu_cores))
	if limit <= 0:
		return []
	return list(units[:limit])


def _build_slot(
	*,
	slot_id: int,
	units: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> TaskSlot:
	logical_cpus: list[int] = []
	core_ids: list[int] = []
	package_ids: list[int] = []
	seen_cpus: set[int] = set()
	seen_core_ids: set[int] = set()
	seen_package_ids: set[int] = set()
	for logical_cpu_group, core_group, package_group in units:
		for cpu_id in logical_cpu_group:
			if int(cpu_id) in seen_cpus:
				continue
			seen_cpus.add(int(cpu_id))
			logical_cpus.append(int(cpu_id))
		for core_id in core_group:
			if int(core_id) in seen_core_ids:
				continue
			seen_core_ids.add(int(core_id))
			core_ids.append(int(core_id))
		for package_id in package_group:
			if int(package_id) in seen_package_ids:
				continue
			seen_package_ids.add(int(package_id))
			package_ids.append(int(package_id))
	return TaskSlot(
		slot_id=int(slot_id),
		logical_cpus=tuple(sorted(logical_cpus)),
		core_ids=tuple(core_ids),
		package_ids=tuple(package_ids),
	)


def build_task_allocation_plan(
	*,
	config: TaskAllocationConfig,
	topology: CpuTopology,
	target_count: int | None = None,
	keyed_read_cap: int | None = None,
	resource_profile: ResourceProfileConfig | None = None,
	available_shm_gb: float | None = None,
) -> TaskAllocationPlan | None:
	if not bool(getattr(config, "enabled", False)):
		return None
	backend = str(getattr(config, "backend", "none") or "none")
	if backend == "none":
		return None
	if backend == "mpi":
		# MPI backend: rank partitioning happens at distributor level.
		# No local allocation plan needed here; each rank will manage its own CPUs.
		return None
	if backend != "local_affinity":
		raise ValueError(
			f"Task allocation plan builder currently supports backend='local_affinity' or 'mpi', got {backend!r}"
		)

	cpus_per_task, cpus_per_task_source = _derive_cpus_per_task(
		config=config,
	)
	allocation_units = _allocation_units(
		topology=topology,
		bind=str(getattr(config, "bind", "none") or "none"),
		use_hyperthreads=bool(getattr(config, "use_hyperthreads", False)),
	)
	allocation_units = _limit_units_by_profile_cpu(units=allocation_units, resource_profile=resource_profile)
	reserved_unit_count = min(len(allocation_units), max(0, int(getattr(config, "reserve_cpus", 0) or 0)))
	allocatable_units = allocation_units[reserved_unit_count:]
	available_unit_count = len(allocatable_units)
	slot_capacity = max(0, int(available_unit_count // max(1, int(cpus_per_task))))
	cpu_capacity_tasks = int(slot_capacity)
	ram_capacity_tasks = _capacity_limit_from_float(
		budget=None if resource_profile is None else resource_profile.ram_gb,
		demand=getattr(config, "ram_gb_per_task", None),
	)
	shm_capacity_tasks = _capacity_limit_from_float(
		budget=available_shm_gb,
		demand=getattr(config, "shm_gb_per_task", None),
	)

	requested_tasks_per_node = getattr(config, "tasks_per_node", "auto")
	if isinstance(requested_tasks_per_node, int):
		effective_task_limit = max(0, int(requested_tasks_per_node))
	else:
		effective_task_limit = int(cpu_capacity_tasks)
	if ram_capacity_tasks is not None:
		effective_task_limit = min(int(effective_task_limit), int(ram_capacity_tasks))
	if shm_capacity_tasks is not None:
		effective_task_limit = min(int(effective_task_limit), int(shm_capacity_tasks))
	resolved_target_count = _as_optional_positive_int(target_count)
	if resolved_target_count is not None:
		effective_task_limit = min(int(effective_task_limit), int(resolved_target_count))
	resolved_keyed_read_cap = _as_optional_positive_int(keyed_read_cap)
	if resolved_keyed_read_cap is not None:
		effective_task_limit = min(int(effective_task_limit), int(resolved_keyed_read_cap))
	effective_task_limit = min(int(effective_task_limit), int(slot_capacity))
	effective_task_limit = max(0, int(effective_task_limit))

	slots = tuple(
		_build_slot(
			slot_id=slot_id,
			units=list(allocatable_units[slot_id * cpus_per_task : (slot_id + 1) * cpus_per_task]),
		)
		for slot_id in range(int(effective_task_limit))
	)
	return TaskAllocationPlan(
		backend=backend,
		bind=str(getattr(config, "bind", "none") or "none"),
		use_hyperthreads=bool(getattr(config, "use_hyperthreads", False)),
		topology=topology,
		cpus_per_task=int(cpus_per_task),
		cpus_per_task_source=str(cpus_per_task_source),
		requested_tasks_per_node=requested_tasks_per_node,
		effective_tasks_per_node=int(effective_task_limit),
		slot_capacity=int(slot_capacity),
		cpu_capacity_tasks=int(cpu_capacity_tasks),
		ram_capacity_tasks=ram_capacity_tasks,
		shm_capacity_tasks=shm_capacity_tasks,
		available_unit_count=int(available_unit_count),
		reserved_unit_count=int(reserved_unit_count),
		set_thread_env=bool(getattr(config, "set_thread_env", False)),
		nested_thread_policy=str(getattr(config, "nested_thread_policy", "preserve_existing") or "preserve_existing"),
		task_unit=str(getattr(config, "task_unit", "well") or "well"),
		target_count=resolved_target_count,
		keyed_read_cap=resolved_keyed_read_cap,
		slots=slots,
	)


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


@dataclass(frozen=True)
class ContainerAffinityReadiness:
	"""Diagnostic result from probing whether local affinity can work in the current environment."""

	affinity_api_available: bool
	"""True when os.sched_getaffinity / os.sched_setaffinity exist on this platform."""
	sysfs_topology_readable: bool
	"""True when /sys/devices/system/cpu exists and cpu0/topology is accessible."""
	sysfs_root: str
	"""Path that was probed for sysfs topology."""
	shm_available_gb: float | None
	"""Usable bytes in /dev/shm (or the probed shm_path) as GiB, None if unavailable."""
	shm_path: str
	"""Path that was probed for shared memory sizing."""
	visible_cpu_count: int
	"""Number of CPUs visible to the current process via sched_getaffinity (or os.cpu_count fallback)."""
	warnings: tuple[str, ...]
	"""Human-readable warnings about conditions that may limit local affinity."""

	def to_dict(self) -> dict[str, Any]:
		return {
			"affinity_api_available": self.affinity_api_available,
			"sysfs_topology_readable": self.sysfs_topology_readable,
			"sysfs_root": self.sysfs_root,
			"shm_available_gb": self.shm_available_gb,
			"shm_path": self.shm_path,
			"visible_cpu_count": self.visible_cpu_count,
			"warnings": list(self.warnings),
		}


def probe_container_readiness(
	*,
	sysfs_root: str | Path = "/sys/devices/system/cpu",
	shm_path: str | Path = "/dev/shm",
	affinity_getter: Callable[[int], Iterable[int]] | None = None,
) -> ContainerAffinityReadiness:
	"""Probe the current process environment for local affinity prerequisites.

	Designed to be called at stage start when backend=local_affinity is configured,
	so operator logs record exactly what the container-visible topology and /dev/shm
	look like before work starts.
	"""
	warnings: list[str] = []

	# --- affinity API ---
	get_affinity = affinity_getter or _default_affinity_getter
	affinity_api_available = callable(getattr(os, "sched_getaffinity", None)) and callable(
		getattr(os, "sched_setaffinity", None)
	)
	if not affinity_api_available:
		warnings.append("os.sched_getaffinity / os.sched_setaffinity are not available on this platform; CPU affinity will not be applied")

	# --- visible CPUs ---
	visible_cpu_count = 0
	try:
		visible_cpu_count = len(tuple(get_affinity(0)))
	except Exception:
		try:
			visible_cpu_count = int(os.cpu_count() or 0)
		except Exception:
			pass
	if visible_cpu_count == 0:
		warnings.append("could not determine visible CPU count; sched_getaffinity returned empty or failed")

	# --- sysfs topology ---
	resolved_sysfs = Path(sysfs_root)
	sysfs_topology_readable = False
	try:
		cpu0_topology = resolved_sysfs / "cpu0" / "topology"
		_ = (cpu0_topology / "physical_package_id").read_text(encoding="utf-8")
		sysfs_topology_readable = True
	except Exception:
		warnings.append(
			f"sysfs topology is not readable under {resolved_sysfs}; "
			"physical-core grouping will fall back to logical CPUs only"
		)

	# --- /dev/shm ---
	shm_available_gb: float | None = None
	resolved_shm = Path(shm_path)
	try:
		stats = os.statvfs(str(resolved_shm))
		shm_available_gb = float(int(stats.f_bavail) * int(stats.f_frsize)) / float(1024**3)
	except Exception:
		warnings.append(
			f"could not read /dev/shm size from {resolved_shm}; "
			"shm_gb_per_task capacity planning will not be validated"
		)

	return ContainerAffinityReadiness(
		affinity_api_available=affinity_api_available,
		sysfs_topology_readable=sysfs_topology_readable,
		sysfs_root=str(resolved_sysfs),
		shm_available_gb=shm_available_gb,
		shm_path=str(resolved_shm),
		visible_cpu_count=visible_cpu_count,
		warnings=tuple(warnings),
	)
