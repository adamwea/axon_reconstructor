from __future__ import annotations

import logging
import threading
from pathlib import Path

from axon_recon.pipeline.resource_budget import ResourceBudgetManager
from axon_recon.pipeline.resources import parse_resources_config
from axon_recon.runtime_config import RuntimeConfig


def _resource_budget_payload(*, h5_read_slots: int = 2, source_h5_max_concurrent: int = 1) -> dict[str, object]:
	return {
		"resources": {
			"active_profile": "lab_server_safe",
			"profiles": {
				"lab_server_safe": {
					"cpu_cores": 8,
					"ram_gb": 32,
					"h5_read_slots": h5_read_slots,
				},
			},
			"keyed_resource_limits": {
				"source_h5_path": {
					"max_concurrent": source_h5_max_concurrent,
				},
			},
			"phase_resource_classes": {
				"preprocess_segments": {
					"h5_read_slots": 1,
					"keyed_resources": {"source_h5_path": 1},
				},
			},
		},
	}


def test_phase_budget_limits_same_source_h5_path_but_allows_other_files() -> None:
	resources = parse_resources_config(runtime_config=RuntimeConfig(_resource_budget_payload()))
	manager = ResourceBudgetManager(resources=resources, planned_target_count=3, well_workers=3)
	condition = threading.Condition()
	release_workers = threading.Event()
	started: list[str] = []
	completed: list[str] = []
	leases: list[tuple[str, dict[str, object]]] = []
	errors: list[BaseException] = []

	def worker(label: str, source_h5_path: Path) -> None:
		try:
			with manager.phase_budget(
				resource_class="preprocess_segments",
				phase_name="preprocess_segments",
				target_label=label,
				resource_key_context={"source_h5_path": source_h5_path},
			) as lease:
				with condition:
					leases.append((label, lease))
					started.append(label)
					condition.notify_all()
				assert release_workers.wait(timeout=5)
				completed.append(label)
		except BaseException as exc:
			with condition:
				errors.append(exc)
				condition.notify_all()

	threads = [
		threading.Thread(target=worker, args=("same-a", Path("/tmp/file_0.raw.h5"))),
		threading.Thread(target=worker, args=("same-b", Path("/tmp/file_0.raw.h5"))),
		threading.Thread(target=worker, args=("other", Path("/tmp/file_1.raw.h5"))),
	]
	for thread in threads:
		thread.start()

	with condition:
		assert condition.wait_for(lambda: len(started) >= 2, timeout=5)
		condition.wait(timeout=0.1)
		started_snapshot = list(started)

	assert "other" in started_snapshot
	assert not {"same-a", "same-b"}.issubset(set(started_snapshot))

	release_workers.set()
	for thread in threads:
		thread.join(timeout=5)
		assert not thread.is_alive()

	assert not errors
	assert sorted(completed) == ["other", "same-a", "same-b"]
	assert len(leases) == 3
	assert all(item[1]["slot_demands"] == {"h5_read_slots": 1} for item in leases)
	assert all(item[1]["keyed_requests"]["source_h5_path"]["demand"] == 1 for item in leases)
	assert any(bool(item[1]["waited"]) and float(item[1]["wait_s"]) > 0.0 for item in leases)


def test_phase_budget_limits_cpu_and_ram_capacity() -> None:
	resources = parse_resources_config(
		runtime_config=RuntimeConfig(
			{
				"resources": {
					"active_profile": "lab_server_safe",
					"profiles": {"lab_server_safe": {"cpu_cores": 6, "ram_gb": 16}},
					"phase_resource_classes": {
						"template_build": {"cpu_cores": 5, "ram_gb": 8},
					},
				}
			}
		)
	)
	manager = ResourceBudgetManager(resources=resources, planned_target_count=2, well_workers=2)
	condition = threading.Condition()
	release_workers = threading.Event()
	started: list[str] = []
	leases: list[tuple[str, dict[str, object]]] = []
	errors: list[BaseException] = []

	def worker(label: str) -> None:
		try:
			with manager.phase_budget(resource_class="template_build", phase_name="build_templates", target_label=label) as lease:
				with condition:
					started.append(label)
					leases.append((label, lease))
					condition.notify_all()
				assert release_workers.wait(timeout=5)
		except BaseException as exc:
			with condition:
				errors.append(exc)
				condition.notify_all()

	threads = [threading.Thread(target=worker, args=("well000",)), threading.Thread(target=worker, args=("well001",))]
	for thread in threads:
		thread.start()

	with condition:
		assert condition.wait_for(lambda: len(started) == 1, timeout=5)
		condition.wait(timeout=0.1)
		assert len(started) == 1

	release_workers.set()
	for thread in threads:
		thread.join(timeout=5)
		assert not thread.is_alive()

	assert not errors
	assert len(leases) == 2
	assert all(item[1]["slot_demands"] == {"cpu_cores": 5, "ram_gb": 8} for item in leases)
	assert any(bool(item[1]["waited"]) and float(item[1]["wait_s"]) > 0.0 for item in leases)
	assert manager.phase_worker_count("template_build") == 5


def test_phase_budget_logs_structured_wait_warning_for_queued_worker() -> None:
	resources = parse_resources_config(runtime_config=RuntimeConfig(_resource_budget_payload(h5_read_slots=1)))
	manager = ResourceBudgetManager(resources=resources, planned_target_count=2, well_workers=2)
	logger = logging.getLogger("axon_recon.tests.resource_budget.waiting")
	logger.setLevel(logging.INFO)
	logger.propagate = False
	condition = threading.Condition()
	records: list[logging.LogRecord] = []

	class _CaptureHandler(logging.Handler):
		def emit(self, record: logging.LogRecord) -> None:
			with condition:
				records.append(record)
				condition.notify_all()

	handler = _CaptureHandler()
	logger.addHandler(handler)
	holder_entered = threading.Event()
	release_holder = threading.Event()
	errors: list[BaseException] = []

	def holder() -> None:
		try:
			with manager.phase_budget(
				resource_class="preprocess_segments",
				logger=logger,
				phase_name="save_rec_metadata",
				target_label="well000",
				resource_key_context={"source_h5_path": Path("/tmp/source.raw.h5")},
			):
				holder_entered.set()
				assert release_holder.wait(timeout=5)
		except BaseException as exc:
			errors.append(exc)

	def waiter() -> None:
		try:
			with manager.phase_budget(
				resource_class="preprocess_segments",
				logger=logger,
				phase_name="preprocess_segments",
				target_label="well001",
				resource_key_context={"source_h5_path": Path("/tmp/other.raw.h5")},
			):
				pass
		except BaseException as exc:
			errors.append(exc)

	try:
		holder_thread = threading.Thread(target=holder)
		waiter_thread = threading.Thread(target=waiter)
		holder_thread.start()
		assert holder_entered.wait(timeout=5)
		waiter_thread.start()
		with condition:
			assert condition.wait_for(
				lambda: any(getattr(record, "event", None) == "phase_resource_gate_waiting" for record in records),
				timeout=5,
			)
		release_holder.set()
		holder_thread.join(timeout=5)
		waiter_thread.join(timeout=5)
		assert not holder_thread.is_alive()
		assert not waiter_thread.is_alive()
	finally:
		release_holder.set()
		logger.removeHandler(handler)

	assert not errors
	wait_records = [record for record in records if getattr(record, "event", None) == "phase_resource_gate_waiting"]
	assert len(wait_records) == 1
	wait_record = wait_records[0]
	assert wait_record.levelno == logging.WARNING
	assert "waiting for slots" in wait_record.getMessage()
	assert wait_record.well_workers == 2
	assert wait_record.target_count == 2
	resource_gate = wait_record.resource_gate
	assert resource_gate["waited"] is True
	assert resource_gate["slot_demands"] == {"h5_read_slots": 1}
	assert resource_gate["slot_available_at_wait"] == {"h5_read_slots": 0}
	assert resource_gate["first_wait_snapshot"]["blocked_slot_dimensions"] == ["h5_read_slots"]