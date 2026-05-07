from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any

from axon_recon.runtime_config import RuntimeConfig


LOGGER = logging.getLogger("axon_recon.pipeline.resources")

_RESERVED_RESOURCE_KEYS: frozenset[str] = frozenset(
	{
		"active_profile",
		"profiles",
		"phase_resource_classes",
		"keyed_resource_limits",
		"defaults",
		"container_caps",
		"task_allocation",
	}
)
_LEGACY_RESOURCE_DEFAULT_KEYS: tuple[str, ...] = (
	"chunk_duration",
	"max_simultaneous_well_reads_per_dataset",
	"max_simultaneous_well_reads_per_h5_file",
)
_WARNED_LEGACY_RESOURCE_DEFAULTS = False
_WARNED_LEGACY_KEYED_RESOURCE_LIMITS = False
_LEGACY_SOURCE_H5_LIMIT_KEYS: tuple[str, ...] = (
	"max_simultaneous_well_reads_per_h5_file",
	"max_simultaneous_well_reads_per_dataset",
)
SOURCE_H5_PATH_KEYED_RESOURCE = "source_h5_path"
RESOURCE_CAPACITY_DIMENSIONS: tuple[str, ...] = (
	"cpu_cores",
	"ram_gb",
	"gpu_sort_slots",
	"h5_read_slots",
	"disk_heavy_slots",
	"plot_slots",
	"analyzer_slots",
)
RESOURCE_SLOT_DIMENSIONS: tuple[str, ...] = (
	"gpu_sort_slots",
	"h5_read_slots",
	"disk_heavy_slots",
	"plot_slots",
	"analyzer_slots",
)
TASK_ALLOCATION_BACKENDS: frozenset[str] = frozenset({"local_affinity", "none", "mpi", "slurm"})
TASK_ALLOCATION_UNITS: frozenset[str] = frozenset({"well"})
TASK_ALLOCATION_BIND_MODES: frozenset[str] = frozenset({"physical_cores", "logical_cpus", "none"})
TASK_ALLOCATION_THREAD_POLICIES: frozenset[str] = frozenset(
	{"match_cpus_per_task", "force_1", "preserve_existing"}
)


@dataclass(frozen=True)
class ResourceProfileConfig:
	cpu_cores: int | None = None
	ram_gb: float | None = None
	gpu_sort_slots: int = 0
	h5_read_slots: int = 0
	disk_heavy_slots: int = 0
	plot_slots: int = 0
	analyzer_slots: int = 0
	default_chunk_duration: str | None = None


@dataclass(frozen=True)
class KeyedResourceLimitConfig:
	description: str | None = None
	max_concurrent: int = 1
	applies_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseResourceClassConfig:
	description: str | None = None
	bottleneck: str | None = None
	cpu_cores: int = 0
	ram_gb: float = 0.0
	gpu_sort_slots: int = 0
	h5_read_slots: int = 0
	disk_heavy_slots: int = 0
	plot_slots: int = 0
	analyzer_slots: int = 0
	keyed_resources: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ContainerCapsConfig:
	shm_size: str | None = None
	shm_size_configured: bool = False
	memory: str | None = None
	memory_configured: bool = False
	memory_reservation: str | None = None
	memory_reservation_configured: bool = False
	memory_swap: str | None = None
	memory_swap_configured: bool = False
	ipc: str | None = None
	ipc_configured: bool = False


@dataclass(frozen=True)
class TaskAllocationConfig:
	enabled: bool = False
	backend: str = "none"
	task_unit: str = "well"
	cpus_per_task: int | str = "auto"
	tasks_per_node: int | str = "auto"
	bind: str = "none"
	use_hyperthreads: bool = False
	reserve_cpus: int = 0
	set_thread_env: bool = False
	nested_thread_policy: str = "preserve_existing"
	ram_gb_per_task: float | None = None
	shm_gb_per_task: float | None = None


@dataclass(frozen=True)
class ResourcesConfig:
	active_profile: str | None = None
	profiles: dict[str, ResourceProfileConfig] = field(default_factory=dict)
	keyed_resource_limits: dict[str, KeyedResourceLimitConfig] = field(default_factory=dict)
	phase_resource_classes: dict[str, PhaseResourceClassConfig] = field(default_factory=dict)
	container_caps: ContainerCapsConfig = field(default_factory=ContainerCapsConfig)
	task_allocation: TaskAllocationConfig = field(default_factory=TaskAllocationConfig)
	defaults: dict[str, Any] = field(default_factory=dict)


def _as_mapping(value: Any) -> dict[str, Any]:
	return dict(value) if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool = False) -> bool:
	if value is None:
		return bool(default)
	if isinstance(value, bool):
		return value
	token = str(value).strip().lower()
	if token in {"1", "true", "yes", "on"}:
		return True
	if token in {"0", "false", "no", "off"}:
		return False
	return bool(default)


def _as_optional_name(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text if text else None


def _as_name_tuple(value: Any) -> tuple[str, ...]:
	if value is None:
		return ()
	if isinstance(value, (list, tuple, set)):
		items = list(value)
	else:
		items = [value]
	ordered: list[str] = []
	seen: set[str] = set()
	for item in items:
		name = _as_optional_name(item)
		if name is None or name in seen:
			continue
		seen.add(name)
		ordered.append(name)
	return tuple(ordered)


def _as_optional_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except Exception:
		return None


def _as_int(value: Any, default: int = 0) -> int:
	parsed = _as_optional_int(value)
	return int(default if parsed is None else parsed)


def _as_optional_float(value: Any) -> float | None:
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _as_float(value: Any, default: float = 0.0) -> float:
	parsed = _as_optional_float(value)
	return float(default if parsed is None else parsed)


def _require_choice(
	*,
	value: Any,
	field_name: str,
	choices: frozenset[str],
	default: str,
) -> str:
	if value is None:
		return str(default)
	resolved = _as_optional_name(value)
	if resolved is None:
		return str(default)
	if resolved not in choices:
		raise ValueError(
			f"resources.task_allocation.{field_name} must be one of {sorted(choices)!r}; got {resolved!r}"
		)
	return str(resolved)


def _require_auto_or_positive_int(*, value: Any, field_name: str) -> int | str:
	if value is None:
		return "auto"
	if isinstance(value, str) and value.strip().lower() == "auto":
		return "auto"
	parsed = _as_optional_int(value)
	if parsed is None or parsed <= 0:
		raise ValueError(
			f"resources.task_allocation.{field_name} must be a positive integer or 'auto'; got {value!r}"
		)
	return int(parsed)


def _require_nonnegative_int(*, value: Any, field_name: str, default: int = 0) -> int:
	if value is None:
		return int(default)
	parsed = _as_optional_int(value)
	if parsed is None or parsed < 0:
		raise ValueError(
			f"resources.task_allocation.{field_name} must be a non-negative integer; got {value!r}"
		)
	return int(parsed)


def _require_optional_positive_float(*, value: Any, field_name: str) -> float | None:
	if value is None:
		return None
	parsed = _as_optional_float(value)
	if parsed is None or parsed <= 0.0:
		raise ValueError(
			f"resources.task_allocation.{field_name} must be a positive number when set; got {value!r}"
		)
	return float(parsed)


def _parse_resource_profile(raw: Any) -> ResourceProfileConfig:
	block = _as_mapping(raw)
	return ResourceProfileConfig(
		cpu_cores=_as_optional_int(block.get("cpu_cores", block.get("cpu", None))),
		ram_gb=_as_optional_float(block.get("ram_gb", None)),
		gpu_sort_slots=max(0, _as_int(block.get("gpu_sort_slots", block.get("gpu", 0)), 0)),
		h5_read_slots=max(0, _as_int(block.get("h5_read_slots", 0), 0)),
		disk_heavy_slots=max(0, _as_int(block.get("disk_heavy_slots", 0), 0)),
		plot_slots=max(0, _as_int(block.get("plot_slots", 0), 0)),
		analyzer_slots=max(0, _as_int(block.get("analyzer_slots", 0), 0)),
		default_chunk_duration=_as_optional_name(
			block.get("default_chunk_duration", block.get("chunk_duration", None))
		),
	)


def _parse_keyed_resource_limit(raw: Any) -> KeyedResourceLimitConfig:
	block = _as_mapping(raw)
	if not block and raw is not None and not isinstance(raw, dict):
		block = {"max_concurrent": raw}
	return KeyedResourceLimitConfig(
		description=_as_optional_name(block.get("description", block.get("note", None))),
		max_concurrent=max(1, _as_int(block.get("max_concurrent", 1), 1)),
		applies_to=_as_name_tuple(block.get("applies_to", ())),
	)


def _parse_keyed_resource_demands(raw: Any) -> dict[str, int]:
	block = _as_mapping(raw)
	demands: dict[str, int] = {}
	for name, value in block.items():
		resource_name = _as_optional_name(name)
		if resource_name is None:
			continue
		if isinstance(value, str) and value.strip().lower() in {"read", "shared", "true", "yes", "on"}:
			demand = 1
		else:
			demand = max(0, _as_int(value, 0))
		if demand > 0:
			demands[resource_name] = int(demand)
	return demands


def _parse_phase_resource_class(raw: Any) -> PhaseResourceClassConfig:
	block = _as_mapping(raw)
	return PhaseResourceClassConfig(
		description=_as_optional_name(block.get("description", block.get("note", None))),
		bottleneck=_as_optional_name(block.get("bottleneck", None)),
		cpu_cores=max(0, _as_int(block.get("cpu_cores", block.get("cpu", 0)), 0)),
		ram_gb=max(0.0, _as_float(block.get("ram_gb", 0.0), 0.0)),
		gpu_sort_slots=max(0, _as_int(block.get("gpu_sort_slots", block.get("gpu", 0)), 0)),
		h5_read_slots=max(0, _as_int(block.get("h5_read_slots", 0), 0)),
		disk_heavy_slots=max(0, _as_int(block.get("disk_heavy_slots", 0), 0)),
		plot_slots=max(0, _as_int(block.get("plot_slots", 0), 0)),
		analyzer_slots=max(0, _as_int(block.get("analyzer_slots", 0), 0)),
		keyed_resources=_parse_keyed_resource_demands(
			block.get("keyed_resources", block.get("keyed_locks", {}))
		),
	)


def _parse_container_cap_value(block: dict[str, Any], key: str) -> tuple[str | None, bool]:
	if key not in block:
		return None, False
	return _as_optional_name(block.get(key, None)), True


def _parse_container_caps(raw: Any) -> ContainerCapsConfig:
	block = _as_mapping(raw)
	shm_size, shm_size_configured = _parse_container_cap_value(block, "shm_size")
	memory, memory_configured = _parse_container_cap_value(block, "memory")
	memory_reservation, memory_reservation_configured = _parse_container_cap_value(block, "memory_reservation")
	memory_swap, memory_swap_configured = _parse_container_cap_value(block, "memory_swap")
	ipc, ipc_configured = _parse_container_cap_value(block, "ipc")
	return ContainerCapsConfig(
		shm_size=shm_size,
		shm_size_configured=shm_size_configured,
		memory=memory,
		memory_configured=memory_configured,
		memory_reservation=memory_reservation,
		memory_reservation_configured=memory_reservation_configured,
		memory_swap=memory_swap,
		memory_swap_configured=memory_swap_configured,
		ipc=ipc,
		ipc_configured=ipc_configured,
	)


def _parse_task_allocation(raw: Any) -> TaskAllocationConfig:
	block = _as_mapping(raw)
	return TaskAllocationConfig(
		enabled=_as_bool(block.get("enabled", False), False),
		backend=_require_choice(
			value=block.get("backend", None),
			field_name="backend",
			choices=TASK_ALLOCATION_BACKENDS,
			default="none",
		),
		task_unit=_require_choice(
			value=block.get("task_unit", None),
			field_name="task_unit",
			choices=TASK_ALLOCATION_UNITS,
			default="well",
		),
		cpus_per_task=_require_auto_or_positive_int(
			value=block.get("cpus_per_task", None),
			field_name="cpus_per_task",
		),
		tasks_per_node=_require_auto_or_positive_int(
			value=block.get("tasks_per_node", None),
			field_name="tasks_per_node",
		),
		bind=_require_choice(
			value=block.get("bind", None),
			field_name="bind",
			choices=TASK_ALLOCATION_BIND_MODES,
			default="none",
		),
		use_hyperthreads=_as_bool(block.get("use_hyperthreads", False), False),
		reserve_cpus=_require_nonnegative_int(
			value=block.get("reserve_cpus", None),
			field_name="reserve_cpus",
			default=0,
		),
		set_thread_env=_as_bool(block.get("set_thread_env", False), False),
		nested_thread_policy=_require_choice(
			value=block.get("nested_thread_policy", None),
			field_name="nested_thread_policy",
			choices=TASK_ALLOCATION_THREAD_POLICIES,
			default="preserve_existing",
		),
		ram_gb_per_task=_require_optional_positive_float(
			value=block.get("ram_gb_per_task", None),
			field_name="ram_gb_per_task",
		),
		shm_gb_per_task=_require_optional_positive_float(
			value=block.get("shm_gb_per_task", None),
			field_name="shm_gb_per_task",
		),
	)


def _warn_legacy_resource_defaults(*, logger: logging.Logger, keys: tuple[str, ...]) -> None:
	global _WARNED_LEGACY_RESOURCE_DEFAULTS
	if _WARNED_LEGACY_RESOURCE_DEFAULTS or not keys:
		return
	logger.warning(
		"Legacy top-level resources keys detected; mapping %s into resources.defaults for compatibility.",
		", ".join(sorted(keys)),
	)
	_WARNED_LEGACY_RESOURCE_DEFAULTS = True


def _warn_legacy_keyed_resource_limit(*, logger: logging.Logger, keys: tuple[str, ...]) -> None:
	global _WARNED_LEGACY_KEYED_RESOURCE_LIMITS
	if _WARNED_LEGACY_KEYED_RESOURCE_LIMITS or not keys:
		return
	logger.warning(
		"Legacy H5 read cap keys detected; mapping %s into resources.keyed_resource_limits.%s.max_concurrent for compatibility.",
		", ".join(sorted(keys)),
		SOURCE_H5_PATH_KEYED_RESOURCE,
	)
	_WARNED_LEGACY_KEYED_RESOURCE_LIMITS = True


def _legacy_source_h5_limit_value(
	*,
	resources_block: dict[str, Any],
	defaults: dict[str, Any],
) -> tuple[int | None, tuple[str, ...]]:
	for key in _LEGACY_SOURCE_H5_LIMIT_KEYS:
		candidate = defaults.get(key, resources_block.get(key, None))
		parsed = _as_optional_int(candidate)
		if parsed is None or parsed <= 0:
			continue
		return int(parsed), (str(key),)
	return None, ()


def parse_resources_config(
	*,
	runtime_config: RuntimeConfig,
	logger: logging.Logger | None = None,
) -> ResourcesConfig:
	resources_block = _as_mapping(runtime_config.get("resources", None))
	defaults = _as_mapping(resources_block.get("defaults", {}))
	legacy_defaults: list[str] = []
	for key in _LEGACY_RESOURCE_DEFAULT_KEYS:
		if key in resources_block and key not in defaults:
			defaults[key] = resources_block[key]
			legacy_defaults.append(key)
	if legacy_defaults:
		_warn_legacy_resource_defaults(logger=(logger or LOGGER), keys=tuple(legacy_defaults))

	profiles_raw = _as_mapping(resources_block.get("profiles", {}))
	profiles = {
		str(name): _parse_resource_profile(value)
		for name, value in profiles_raw.items()
	}
	keyed_limits_raw = _as_mapping(resources_block.get("keyed_resource_limits", {}))
	keyed_resource_limits = {
		str(name): _parse_keyed_resource_limit(value)
		for name, value in keyed_limits_raw.items()
	}
	phase_classes_raw = _as_mapping(resources_block.get("phase_resource_classes", {}))
	phase_resource_classes = {
		str(name): _parse_phase_resource_class(value)
		for name, value in phase_classes_raw.items()
	}
	container_caps = _parse_container_caps(resources_block.get("container_caps", {}))
	task_allocation = _parse_task_allocation(resources_block.get("task_allocation", {}))
	legacy_source_h5_limit, legacy_source_h5_keys = _legacy_source_h5_limit_value(
		resources_block=resources_block,
		defaults=defaults,
	)
	if (
		legacy_source_h5_limit is not None
		and SOURCE_H5_PATH_KEYED_RESOURCE not in keyed_resource_limits
	):
		keyed_resource_limits[SOURCE_H5_PATH_KEYED_RESOURCE] = KeyedResourceLimitConfig(
			description="Legacy source H5 concurrency limit.",
			max_concurrent=int(legacy_source_h5_limit),
		)
		_warn_legacy_keyed_resource_limit(
			logger=(logger or LOGGER),
			keys=legacy_source_h5_keys,
		)
	active_profile = _as_optional_name(resources_block.get("active_profile", None))
	if active_profile is not None and active_profile not in profiles:
		raise ValueError(
			"resources.active_profile references an undefined profile: "
			f"{active_profile!r}"
		)
	return ResourcesConfig(
		active_profile=active_profile,
		profiles=profiles,
		keyed_resource_limits=keyed_resource_limits,
		phase_resource_classes=phase_resource_classes,
		container_caps=container_caps,
		task_allocation=task_allocation,
		defaults=defaults,
	)


def validate_phase_resource_class(
	*,
	resource_class: Any,
	resources: ResourcesConfig,
	phase_name: str,
) -> str | None:
	resolved = _as_optional_name(resource_class)
	if resolved is None:
		return None
	if resolved not in resources.phase_resource_classes:
		raise ValueError(
			f"Unknown resource_class for {phase_name}: {resolved!r}. "
			"Define it under resources.phase_resource_classes."
		)
	return resolved


def get_active_resource_profile(resources: ResourcesConfig) -> ResourceProfileConfig | None:
	active_profile = _as_optional_name(resources.active_profile)
	if active_profile is None:
		return None
	return resources.profiles.get(active_profile, None)


def get_phase_resource_class_config(
	resources: ResourcesConfig,
	resource_class: str | None,
) -> PhaseResourceClassConfig | None:
	resolved = _as_optional_name(resource_class)
	if resolved is None:
		return None
	return resources.phase_resource_classes.get(resolved, None)


def get_keyed_resource_limit_config(
	resources: ResourcesConfig,
	resource_name: str | None,
) -> KeyedResourceLimitConfig | None:
	resolved = _as_optional_name(resource_name)
	if resolved is None:
		return None
	return resources.keyed_resource_limits.get(resolved, None)


def get_keyed_resource_limit_max_concurrent(
	resources: ResourcesConfig,
	resource_name: str | None,
) -> int | None:
	config = get_keyed_resource_limit_config(resources, resource_name)
	if config is None:
		return None
	return max(1, int(config.max_concurrent))


def get_phase_keyed_resource_demands(
	resources: ResourcesConfig,
	resource_class: str | None,
) -> dict[str, int]:
	phase_config = get_phase_resource_class_config(resources, resource_class)
	if phase_config is None:
		return {}
	return {
		str(resource_name): max(0, int(demand))
		for resource_name, demand in phase_config.keyed_resources.items()
		if int(demand or 0) > 0
	}


def _resource_value_to_units(*, dimension: str, value: Any, budget: bool) -> int | None:
	if value is None:
		return None
	if str(dimension) == "ram_gb":
		try:
			parsed = float(value)
		except Exception:
			return None
		if parsed <= 0.0:
			return 0
		return int(max(0, math.floor(parsed) if budget else math.ceil(parsed)))
	try:
		parsed_int = int(value)
	except Exception:
		return None
	return max(0, parsed_int)


def get_profile_budget_units(
	*,
	resources: ResourcesConfig,
	dimensions: tuple[str, ...] = RESOURCE_CAPACITY_DIMENSIONS,
) -> dict[str, int]:
	profile = get_active_resource_profile(resources)
	if profile is None:
		return {str(dimension): 0 for dimension in dimensions}
	budgets: dict[str, int] = {}
	for dimension in dimensions:
		budget_units = _resource_value_to_units(
			dimension=str(dimension),
			value=getattr(profile, str(dimension), None),
			budget=True,
		)
		budgets[str(dimension)] = max(0, int(budget_units or 0))
	return budgets


def get_phase_resource_demand_units(
	*,
	resources: ResourcesConfig,
	resource_class: str | None,
	dimensions: tuple[str, ...] = RESOURCE_CAPACITY_DIMENSIONS,
) -> dict[str, int]:
	phase_config = get_phase_resource_class_config(resources, resource_class)
	if phase_config is None:
		return {str(dimension): 0 for dimension in dimensions}
	demands: dict[str, int] = {}
	for dimension in dimensions:
		demand_units = _resource_value_to_units(
			dimension=str(dimension),
			value=getattr(phase_config, str(dimension), None),
			budget=False,
		)
		demands[str(dimension)] = max(0, int(demand_units or 0))
	return demands


def estimate_phase_resource_class_capacity(
	*,
	resources: ResourcesConfig,
	resource_class: str | None,
	dimensions: tuple[str, ...] = RESOURCE_CAPACITY_DIMENSIONS,
) -> int | None:
	resolved = _as_optional_name(resource_class)
	if resolved is None:
		return None
	profile = get_active_resource_profile(resources)
	phase_config = get_phase_resource_class_config(resources, resolved)
	if profile is None or phase_config is None:
		return None
	budgets = get_profile_budget_units(resources=resources, dimensions=dimensions)
	demands = get_phase_resource_demand_units(
		resources=resources,
		resource_class=resolved,
		dimensions=dimensions,
	)
	capacities: list[int] = []
	for dimension in dimensions:
		budget_units = max(0, int(budgets.get(str(dimension), 0)))
		demand_units = max(0, int(demands.get(str(dimension), 0)))
		if demand_units <= 0 or budget_units <= 0:
			continue
		capacities.append(max(0, int(budget_units // demand_units)))
	if not capacities:
		return None
	return max(0, min(capacities))


def get_max_phase_resource_demands(
	*,
	resources: ResourcesConfig,
	resource_classes: list[str] | tuple[str, ...],
	dimensions: tuple[str, ...],
) -> dict[str, int]:
	max_demands = {str(dimension): 0 for dimension in dimensions}
	for resource_class in resource_classes:
		demands = get_phase_resource_demand_units(
			resources=resources,
			resource_class=resource_class,
			dimensions=dimensions,
		)
		for dimension in dimensions:
			max_demands[str(dimension)] = max(
				int(max_demands.get(str(dimension), 0)),
				int(demands.get(str(dimension), 0)),
			)
	return max_demands


def get_resource_default(
	*,
	runtime_config: RuntimeConfig,
	key: str,
	default: Any = None,
	logger: logging.Logger | None = None,
) -> Any:
	resources = parse_resources_config(runtime_config=runtime_config, logger=logger)
	if key in resources.defaults:
		return resources.defaults[key]
	if str(key) == "chunk_duration" and resources.active_profile is not None:
		active_profile = resources.profiles.get(str(resources.active_profile), None)
		if active_profile is not None and active_profile.default_chunk_duration is not None:
			return active_profile.default_chunk_duration
	resources_block = _as_mapping(runtime_config.get("resources", None))
	if key in resources_block and key not in _RESERVED_RESOURCE_KEYS:
		return resources_block[key]
	return default
