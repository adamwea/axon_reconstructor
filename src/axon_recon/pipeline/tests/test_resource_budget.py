from __future__ import annotations

from pathlib import Path
import threading

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