from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.cpu_allocation import (
	build_task_allocation_plan,
	detect_cpu_topology,
	format_cpu_set,
	format_cpu_topology,
)
from axon_recon.pipeline.execution.context import StageParallelism
from axon_recon.pipeline.resources import ResourceProfileConfig, TaskAllocationConfig


def _write_cpu_topology(
	*,
	sysfs_root: Path,
	cpu_id: int,
	package_id: int,
	core_id: int,
	siblings: str,
) -> None:
	topology_root = sysfs_root / f"cpu{cpu_id}" / "topology"
	topology_root.mkdir(parents=True, exist_ok=True)
	(topology_root / "physical_package_id").write_text(f"{package_id}\n", encoding="utf-8")
	(topology_root / "core_id").write_text(f"{core_id}\n", encoding="utf-8")
	(topology_root / "thread_siblings_list").write_text(f"{siblings}\n", encoding="utf-8")


def _write_single_socket_hyperthreaded_topology(sysfs_root: Path, *, core_count: int, threads_per_core: int) -> None:
	assert threads_per_core == 2
	for core_id in range(core_count):
		logical_start = core_id * threads_per_core
		siblings = f"{logical_start}-{logical_start + 1}"
		_write_cpu_topology(
			sysfs_root=sysfs_root,
			cpu_id=logical_start,
			package_id=0,
			core_id=core_id,
			siblings=siblings,
		)
		_write_cpu_topology(
			sysfs_root=sysfs_root,
			cpu_id=logical_start + 1,
			package_id=0,
			core_id=core_id,
			siblings=siblings,
		)


def _lab_topology(tmp_path: Path):
	sysfs_root = tmp_path / "sys" / "devices" / "system" / "cpu"
	_write_single_socket_hyperthreaded_topology(sysfs_root, core_count=24, threads_per_core=2)
	return detect_cpu_topology(
		sysfs_root=sysfs_root,
		affinity_getter=lambda _pid: set(range(48)),
	)


def _stage_parallelism(*, well_workers: int, max_stage_workers: int = 24) -> StageParallelism:
	return StageParallelism(
		max_workers=max_stage_workers,
		max_stage_workers=max_stage_workers,
		well_workers=well_workers,
		unit_workers=max(1, int(max_stage_workers // max(1, int(well_workers)))),
	)


def test_format_cpu_set_compacts_ranges() -> None:
	assert format_cpu_set((0, 1, 2, 4, 5, 7, 9, 10)) == "0-2,4-5,7,9-10"


def test_detect_cpu_topology_reads_visible_sysfs_topology_for_single_socket_hyperthreads(tmp_path: Path) -> None:
	sysfs_root = tmp_path / "sys" / "devices" / "system" / "cpu"
	_write_single_socket_hyperthreaded_topology(sysfs_root, core_count=24, threads_per_core=2)

	topology = detect_cpu_topology(
		sysfs_root=sysfs_root,
		affinity_getter=lambda _pid: set(range(48)),
	)

	assert topology.source == "sysfs"
	assert topology.visible_cpus == tuple(range(48))
	assert topology.logical_cpu_count == 48
	assert topology.physical_core_count == 24
	assert topology.package_count == 1
	assert topology.visible_threads_per_core == (2,)
	assert topology.cores[0].logical_cpus == (0, 1)
	assert topology.cores[-1].logical_cpus == (46, 47)

	formatted = format_cpu_topology(topology)
	assert "source: sysfs" in formatted
	assert "visible_cpus: 0-47" in formatted
	assert "physical_core_count: 24" in formatted
	assert "core 0 -> 0-1" in formatted


def test_detect_cpu_topology_respects_cpuset_restricted_visibility(tmp_path: Path) -> None:
	sysfs_root = tmp_path / "sys" / "devices" / "system" / "cpu"
	_write_single_socket_hyperthreaded_topology(sysfs_root, core_count=24, threads_per_core=2)
	visible_cpus = {0, 1, 2, 3, 8, 9}

	topology = detect_cpu_topology(
		sysfs_root=sysfs_root,
		affinity_getter=lambda _pid: visible_cpus,
	)

	assert topology.source == "sysfs"
	assert topology.visible_cpus == (0, 1, 2, 3, 8, 9)
	assert topology.logical_cpu_count == 6
	assert topology.physical_core_count == 3
	assert topology.package_count == 1
	assert topology.visible_threads_per_core == (2,)
	assert [core.logical_cpus for core in topology.cores] == [(0, 1), (2, 3), (8, 9)]


def test_detect_cpu_topology_falls_back_to_logical_visible_cpus_when_sysfs_missing(
	tmp_path: Path,
	caplog: pytest.LogCaptureFixture,
) -> None:
	sysfs_root = tmp_path / "missing" / "cpu"
	caplog.set_level("WARNING", logger="axon_recon.pipeline.cpu_allocation")

	topology = detect_cpu_topology(
		sysfs_root=sysfs_root,
		affinity_getter=lambda _pid: {0, 2, 4},
	)

	assert topology.source == "logical_fallback"
	assert topology.visible_cpus == (0, 2, 4)
	assert topology.logical_cpu_count == 3
	assert topology.physical_core_count == 3
	assert topology.package_count == 1
	assert topology.visible_threads_per_core == (1,)
	assert topology.warning is not None
	assert "falling back to visible logical CPUs only" in str(topology.warning)
	assert any("falling back to visible logical CPUs only" in record.getMessage() for record in caplog.records)


def test_build_task_allocation_plan_returns_none_when_disabled(tmp_path: Path) -> None:
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(enabled=False),
		topology=topology,
		target_count=8,
		stage_parallelism=_stage_parallelism(well_workers=8),
	)

	assert plan is None


def test_build_task_allocation_plan_uses_physical_core_capacity_without_hyperthreads(tmp_path: Path) -> None:
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=4,
			tasks_per_node="auto",
			use_hyperthreads=False,
		),
		topology=topology,
		target_count=12,
		stage_parallelism=_stage_parallelism(well_workers=12),
	)

	assert plan is not None
	assert plan.cpu_capacity_tasks == 6
	assert plan.slot_capacity == 6
	assert plan.effective_tasks_per_node == 6
	assert len(plan.slots) == 6
	assert plan.cpus_per_task == 4
	assert plan.slots[0].logical_cpus == (0, 2, 4, 6)
	assert plan.slots[0].physical_core_count == 4


def test_build_task_allocation_plan_reserve_cpus_reduces_slot_capacity(tmp_path: Path) -> None:
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=4,
			tasks_per_node="auto",
			use_hyperthreads=False,
			reserve_cpus=2,
		),
		topology=topology,
		target_count=10,
		stage_parallelism=_stage_parallelism(well_workers=10),
	)

	assert plan is not None
	assert plan.reserved_unit_count == 2
	assert plan.available_unit_count == 22
	assert plan.cpu_capacity_tasks == 5
	assert len(plan.slots) == 5
	assert plan.slots[0].logical_cpus == (4, 6, 8, 10)


def test_build_task_allocation_plan_clamps_explicit_tasks_per_node_to_capacity(tmp_path: Path) -> None:
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=4,
			tasks_per_node=99,
			use_hyperthreads=False,
			reserve_cpus=2,
		),
		topology=topology,
		target_count=99,
		stage_parallelism=_stage_parallelism(well_workers=99),
	)

	assert plan is not None
	assert plan.requested_tasks_per_node == 99
	assert plan.cpu_capacity_tasks == 5
	assert plan.effective_tasks_per_node == 5
	assert len(plan.slots) == 5


def test_build_task_allocation_plan_reduces_tasks_by_ram_and_shm_capacity(tmp_path: Path) -> None:
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=4,
			tasks_per_node="auto",
			use_hyperthreads=False,
			ram_gb_per_task=20.0,
			shm_gb_per_task=10.0,
		),
		topology=topology,
		target_count=99,
		stage_parallelism=_stage_parallelism(well_workers=99),
		resource_profile=ResourceProfileConfig(cpu_cores=24, ram_gb=55.0),
		available_shm_gb=25.0,
	)

	assert plan is not None
	assert plan.cpu_capacity_tasks == 6
	assert plan.ram_capacity_tasks == 2
	assert plan.shm_capacity_tasks == 2
	assert plan.effective_tasks_per_node == 2
	assert len(plan.slots) == 2


def test_build_task_allocation_plan_includes_sibling_threads_when_enabled(tmp_path: Path) -> None:
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=2,
			tasks_per_node="auto",
			use_hyperthreads=True,
		),
		topology=topology,
		target_count=12,
		stage_parallelism=_stage_parallelism(well_workers=12),
	)

	assert plan is not None
	assert plan.cpu_capacity_tasks == 12
	assert plan.slots[0].logical_cpus == (0, 1, 2, 3)
	assert plan.slots[0].physical_core_count == 2


def test_build_task_allocation_plan_derives_auto_cpus_per_task_from_stage_parallelism(tmp_path: Path) -> None:
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task="auto",
			tasks_per_node="auto",
			use_hyperthreads=False,
		),
		topology=topology,
		target_count=8,
		stage_parallelism=_stage_parallelism(well_workers=6, max_stage_workers=24),
	)

	assert plan is not None
	assert plan.cpus_per_task == 4
	assert plan.cpus_per_task_source == "stage_parallelism.max_stage_workers/well_workers"
	assert plan.cpu_capacity_tasks == 6
	assert plan.effective_tasks_per_node == 6
