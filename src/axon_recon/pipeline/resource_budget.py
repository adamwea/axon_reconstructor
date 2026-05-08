from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .resources import (
	RESOURCE_CAPACITY_DIMENSIONS,
	SOURCE_H5_PATH_KEYED_RESOURCE,
	ResourcesConfig,
	estimate_phase_resource_class_capacity,
	get_active_profile,
	get_phase_keyed_resource_demands,
	get_phase_resource_demand_units,
	get_profile_budget_units,
)


def _normalize_keyed_resource_value(value: Any) -> str | None:
	if value is None:
		return None
	try:
		return str(Path(value).expanduser())
	except Exception:
		text = str(value).strip()
		return text if text else None


def _resource_key_context_value(*, resource_name: str, resource_key_context: Any) -> Any:
	if resource_key_context is None:
		return None
	if isinstance(resource_key_context, dict):
		value = resource_key_context.get(resource_name, None)
		if value is None and str(resource_name) == SOURCE_H5_PATH_KEYED_RESOURCE:
			value = resource_key_context.get("h5_path", None)
		return value
	value = getattr(resource_key_context, resource_name, None)
	if value is None and str(resource_name) == SOURCE_H5_PATH_KEYED_RESOURCE:
		value = getattr(resource_key_context, SOURCE_H5_PATH_KEYED_RESOURCE, None)
		if value is None:
			value = getattr(resource_key_context, "h5_path", None)
	return value


class ResourceBudgetManager:
	def __init__(
		self,
		*,
		resources: ResourcesConfig,
		planned_target_count: int,
		well_workers: int,
	) -> None:
		self.resources = resources
		self.planned_target_count = max(0, int(planned_target_count))
		self.well_workers = max(1, int(well_workers))
		self._total_slot_budget = get_profile_budget_units(
			resources=resources,
			dimensions=RESOURCE_CAPACITY_DIMENSIONS,
		)
		self._available_slot_budget = dict(self._total_slot_budget)
		_active_prof = get_active_profile(self.resources)
		if _active_prof is not None and hasattr(_active_prof, "keyed_resource_limits"):
			_keyed_limits = _active_prof.keyed_resource_limits
		else:
			_keyed_limits = self.resources._legacy_keyed_resource_limits
		self._keyed_resource_limits = {
			str(resource_name): max(1, int(limit.max_concurrent))
			for resource_name, limit in _keyed_limits.items()
			if int(limit.max_concurrent) > 0
		}
		self._active_keyed_resource_counts: dict[str, dict[str, int]] = {}
		self._condition = threading.Condition()

	@property
	def profile_name(self) -> str | None:
		return None if self.resources.active_profile is None else str(self.resources.active_profile)

	def phase_capacity(self, resource_class: str | None) -> int | None:
		return estimate_phase_resource_class_capacity(
			resources=self.resources,
			resource_class=resource_class,
		)

	def slot_demands(self, resource_class: str | None) -> dict[str, int]:
		return get_phase_resource_demand_units(
			resources=self.resources,
			resource_class=resource_class,
			dimensions=RESOURCE_CAPACITY_DIMENSIONS,
		)

	def phase_cpu_cores(self, resource_class: str | None) -> int:
		demands = get_phase_resource_demand_units(
			resources=self.resources,
			resource_class=resource_class,
			dimensions=("cpu_cores",),
		)
		return max(0, int(demands.get("cpu_cores", 0) or 0))

	def phase_worker_count(self, resource_class: str | None, *, fallback_workers: int = 1) -> int:
		cpu_cores = self.phase_cpu_cores(resource_class)
		if cpu_cores > 0:
			return int(cpu_cores)
		return max(1, int(fallback_workers))

	def keyed_resource_demands(self, resource_class: str | None) -> dict[str, int]:
		return get_phase_keyed_resource_demands(self.resources, resource_class)

	@contextmanager
	def phase_budget(
		self,
		*,
		resource_class: str | None,
		logger: Any | None = None,
		phase_name: str | None = None,
		target_label: str | None = None,
		resource_key_context: Any | None = None,
	) -> Iterator[dict[str, Any]]:
		slot_demands = {
			str(dimension): max(0, int(demand))
			for dimension, demand in self.slot_demands(resource_class).items()
			if int(demand or 0) > 0
		}
		keyed_demands = {
			str(resource_name): max(0, int(demand))
			for resource_name, demand in self.keyed_resource_demands(resource_class).items()
			if int(demand or 0) > 0
		}
		keyed_requests: dict[str, tuple[str, int]] = {}
		missing_keyed_resources: list[str] = []
		for resource_name, demand in keyed_demands.items():
			limit = max(0, int(self._keyed_resource_limits.get(str(resource_name), 0)))
			if limit <= 0:
				continue
			key_value = _normalize_keyed_resource_value(
				_resource_key_context_value(
					resource_name=str(resource_name),
					resource_key_context=resource_key_context,
				)
			)
			if key_value is None:
				missing_keyed_resources.append(str(resource_name))
				continue
			keyed_requests[str(resource_name)] = (str(key_value), int(demand))
		if logger is not None and missing_keyed_resources:
			logger.warning(
				"Phase resource gate missing keyed resource values: phase=%s target=%s resource_class=%s resources=%s",
				str(phase_name or "unknown"),
				str(target_label or "unknown"),
				str(resource_class or "null"),
				", ".join(sorted(set(missing_keyed_resources))),
			)
		keyed_request_payload = {
			str(resource_name): {"key": str(key_value), "demand": int(demand)}
			for resource_name, (key_value, demand) in sorted(keyed_requests.items())
		}
		keyed_limit_payload = {
			str(resource_name): int(self._keyed_resource_limits.get(str(resource_name), 0))
			for resource_name in sorted(keyed_requests)
		}
		if not slot_demands and not keyed_requests:
			yield {
				"wait_s": 0.0,
				"waited": False,
				"slot_demands": {},
				"keyed_requests": {},
				"keyed_limits": {},
			}
			return

		wait_started = time.perf_counter()
		wait_logged = False
		first_wait_snapshot: dict[str, Any] | None = None
		with self._condition:
			while True:
				has_slot_budget = all(
					int(self._available_slot_budget.get(str(dimension), 0)) >= int(demand)
					for dimension, demand in slot_demands.items()
				)
				has_keyed_budget = all(
					int(self._active_keyed_resource_counts.get(str(resource_name), {}).get(str(key_value), 0))
					+ int(demand)
					<= int(self._keyed_resource_limits.get(str(resource_name), 0))
					for resource_name, (key_value, demand) in keyed_requests.items()
				)
				if has_slot_budget and has_keyed_budget:
					break
				if first_wait_snapshot is None:
					first_wait_snapshot = {
						"available_slot_budget": {
							str(dimension): int(self._available_slot_budget.get(str(dimension), 0))
							for dimension in sorted(slot_demands)
						},
						"blocked_slot_dimensions": [
							str(dimension)
							for dimension, demand in sorted(slot_demands.items())
							if int(self._available_slot_budget.get(str(dimension), 0)) < int(demand)
						],
						"keyed_active": {
							str(resource_name): int(
								self._active_keyed_resource_counts.get(str(resource_name), {}).get(str(key_value), 0)
							)
							for resource_name, (key_value, _demand) in sorted(keyed_requests.items())
						},
						"blocked_keyed_resources": [
							str(resource_name)
							for resource_name, (key_value, demand) in sorted(keyed_requests.items())
							if int(self._active_keyed_resource_counts.get(str(resource_name), {}).get(str(key_value), 0))
							+ int(demand)
							> int(self._keyed_resource_limits.get(str(resource_name), 0))
						],
					}
				if logger is not None and not wait_logged:
					available_slot_budget = {
						str(dimension): int(self._available_slot_budget.get(str(dimension), 0))
						for dimension in sorted(slot_demands)
					}
					keyed_request_payload = {
						str(resource_name): {"key": str(key_value), "demand": int(demand)}
						for resource_name, (key_value, demand) in sorted(keyed_requests.items())
					}
					keyed_active_payload = {
						str(resource_name): int(
							self._active_keyed_resource_counts.get(str(resource_name), {}).get(str(key_value), 0)
						)
						for resource_name, (key_value, _demand) in sorted(keyed_requests.items())
					}
					keyed_limit_wait_payload = {
						str(resource_name): int(self._keyed_resource_limits.get(str(resource_name), 0))
						for resource_name in sorted(keyed_requests)
					}
					wait_payload = {
						"waited": True,
						"wait_s": 0.0,
						"slot_demands": dict(slot_demands),
						"slot_budget_total": dict(self._total_slot_budget),
						"slot_available_at_wait": available_slot_budget,
						"keyed_requests": keyed_request_payload,
						"keyed_limits": keyed_limit_wait_payload,
						"keyed_active_at_wait": keyed_active_payload,
						"first_wait_snapshot": first_wait_snapshot,
					}
					logger.warning(
						"Phase resource gate waiting for slots: phase=%s target=%s resource_class=%s well_workers=%d planned_targets=%d slot_demands=%s available=%s keyed_requests=%s keyed_active=%s keyed_limits=%s",
						str(phase_name or "unknown"),
						str(target_label or "unknown"),
						str(resource_class or "null"),
						int(self.well_workers),
						int(self.planned_target_count),
						slot_demands,
						available_slot_budget,
						keyed_request_payload,
						keyed_active_payload,
						keyed_limit_wait_payload,
						extra={
							"event": "phase_resource_gate_waiting",
							"resource_gate": wait_payload,
							"well_workers": int(self.well_workers),
							"target_count": int(self.planned_target_count),
						},
					)
					wait_logged = True
				self._condition.wait(timeout=0.25)
			slot_available_at_acquire = {
				str(dimension): int(self._available_slot_budget.get(str(dimension), 0))
				for dimension in sorted(slot_demands)
			}
			keyed_active_at_acquire = {
				str(resource_name): int(
					self._active_keyed_resource_counts.get(str(resource_name), {}).get(str(key_value), 0)
				)
				for resource_name, (key_value, _demand) in sorted(keyed_requests.items())
			}
			for dimension, demand in slot_demands.items():
				self._available_slot_budget[str(dimension)] = max(
					0,
					int(self._available_slot_budget.get(str(dimension), 0)) - int(demand),
				)
			for resource_name, (key_value, demand) in keyed_requests.items():
				active_counts = self._active_keyed_resource_counts.setdefault(str(resource_name), {})
				active_counts[str(key_value)] = int(active_counts.get(str(key_value), 0)) + int(demand)
		wait_s = max(0.0, float(time.perf_counter() - wait_started)) if first_wait_snapshot is not None else 0.0
		acquisition = {
			"wait_s": wait_s,
			"waited": bool(first_wait_snapshot is not None),
			"slot_demands": dict(slot_demands),
			"slot_budget_total": dict(self._total_slot_budget),
			"slot_available_at_acquire": slot_available_at_acquire,
			"keyed_requests": keyed_request_payload,
			"keyed_limits": keyed_limit_payload,
			"keyed_active_at_acquire": keyed_active_at_acquire,
			"first_wait_snapshot": first_wait_snapshot,
		}
		try:
			yield acquisition
		finally:
			with self._condition:
				for dimension, demand in slot_demands.items():
					self._available_slot_budget[str(dimension)] = min(
						int(self._total_slot_budget.get(str(dimension), 0)),
						int(self._available_slot_budget.get(str(dimension), 0)) + int(demand),
					)
				for resource_name, (key_value, demand) in keyed_requests.items():
					active_counts = self._active_keyed_resource_counts.get(str(resource_name), {})
					remaining = int(active_counts.get(str(key_value), 0)) - int(demand)
					if remaining > 0:
						active_counts[str(key_value)] = int(remaining)
					else:
						active_counts.pop(str(key_value), None)
					if not active_counts:
						self._active_keyed_resource_counts.pop(str(resource_name), None)
				self._condition.notify_all()


_CURRENT_STAGE_RESOURCE_BUDGET: ResourceBudgetManager | None = None
_CURRENT_STAGE_RESOURCE_BUDGET_LOCK = threading.RLock()


@contextmanager
def stage_resource_budget_context(manager: ResourceBudgetManager | None) -> Iterator[None]:
	global _CURRENT_STAGE_RESOURCE_BUDGET
	with _CURRENT_STAGE_RESOURCE_BUDGET_LOCK:
		previous = _CURRENT_STAGE_RESOURCE_BUDGET
		_CURRENT_STAGE_RESOURCE_BUDGET = manager
	try:
		yield
	finally:
		with _CURRENT_STAGE_RESOURCE_BUDGET_LOCK:
			_CURRENT_STAGE_RESOURCE_BUDGET = previous


def current_stage_resource_budget_manager() -> ResourceBudgetManager | None:
	with _CURRENT_STAGE_RESOURCE_BUDGET_LOCK:
		return _CURRENT_STAGE_RESOURCE_BUDGET


def current_phase_worker_allocation(
	*,
	resource_class: str | None,
	fallback_workers: int = 1,
) -> tuple[int, str]:
	manager = current_stage_resource_budget_manager()
	if manager is None:
		return max(1, int(fallback_workers)), "inputs.n_jobs"
	cpu_cores = manager.phase_cpu_cores(resource_class)
	if cpu_cores > 0:
		return int(cpu_cores), "resource_class.cpu_cores"
	return max(1, int(fallback_workers)), "inputs.n_jobs"
