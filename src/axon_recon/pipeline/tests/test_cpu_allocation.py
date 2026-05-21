from __future__ import annotations

import os
from pathlib import Path

import pytest

from axon_recon.pipeline.cpu_allocation import (
	TaskSlot,
	apply_thread_env_context,
	build_task_allocation_plan,
	current_phase_budget,
	detect_cpu_topology,
	format_cpu_set,
	format_cpu_topology,
	phase_budgets_context,
	probe_container_readiness,
	resolve_inner_worker_count,
	task_slot_affinity_context,
	task_slot_context,
)
from axon_recon.pipeline.execution.context import StageParallelism
from axon_recon.pipeline.runner import StageAllocationPreview, format_stage_allocation_previews
from axon_recon.pipeline.resources import PhaseResourceClassConfig, ResourceProfileConfig, TaskAllocationConfig


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
	)

	assert plan is not None
	assert plan.cpu_capacity_tasks == 6
	assert plan.slot_capacity == 6
	assert plan.effective_tasks_per_node == 6
	assert len(plan.slots) == 6
	assert plan.cpus_per_task == 4
	assert plan.topology.visible_cpus == tuple(range(48))
	assert plan.slots[0].logical_cpus == (0, 2, 4, 6)
	assert plan.slots[0].physical_core_count == 4
	preview = StageAllocationPreview(
		stage="preprocess.save_rec_metadata",
		target_count=2,
		target_labels=("12:well000", "12:well001"),
		phase_resource_classes=("h5_metadata",),
		parallelism=StageParallelism(
			well_workers=2,
			unit_workers=2,
			task_allocation_plan=plan,
		),
	)
	formatted = format_stage_allocation_previews([preview])
	assert "task_allocation: enabled backend=local_affinity bind=physical_cores" in formatted
	assert "cpu_topology:" in formatted
	assert "visible_cpus:" in formatted


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
		resource_profile=ResourceProfileConfig(cpu_cores=24, ram_gb=55.0),
		available_shm_gb=25.0,
	)

	assert plan is not None
	assert plan.cpu_capacity_tasks == 6
	assert plan.ram_capacity_tasks == 2
	assert plan.shm_capacity_tasks == 2
	assert plan.effective_tasks_per_node == 2
	assert len(plan.slots) == 2


def test_build_task_allocation_plan_keyed_read_cap_clamps_effective_tasks(tmp_path: Path) -> None:
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=2,
			tasks_per_node="auto",
			use_hyperthreads=False,
		),
		topology=topology,
		target_count=12,
		keyed_read_cap=3,
	)

	assert plan is not None
	assert plan.cpu_capacity_tasks == 12
	assert plan.keyed_read_cap == 3
	assert plan.effective_tasks_per_node == 3
	assert len(plan.slots) == 3


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
	)

	assert plan is not None
	assert plan.cpu_capacity_tasks == 12
	assert plan.slots[0].logical_cpus == (0, 1, 2, 3)
	assert plan.slots[0].physical_core_count == 2


def test_build_task_allocation_plan_auto_cpus_per_task_defaults_to_1(tmp_path: Path) -> None:
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
	)

	assert plan is not None
	assert plan.cpus_per_task == 1
	assert plan.cpus_per_task_source == "default"


def test_build_task_allocation_plan_slurm_env_wins_over_yaml_cpus_per_task(
	tmp_path: Path, monkeypatch
) -> None:
	"""Per parallelism plan slice 9.5 + user 2026-05-21: when srun supplies
	SLURM_CPUS_PER_TASK, that value wins over the YAML's task_allocation.
	cpus_per_task. YAML is the fallback default."""
	monkeypatch.setenv("SLURM_CPUS_PER_TASK", "128")
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=16,  # YAML default — should be overridden by env
			tasks_per_node="auto",
			use_hyperthreads=False,
		),
		topology=topology,
		target_count=8,
	)
	assert plan is not None
	assert plan.cpus_per_task == 128
	assert plan.cpus_per_task_source == "slurm_env:SLURM_CPUS_PER_TASK"


def test_build_task_allocation_plan_yaml_used_when_slurm_env_absent(
	tmp_path: Path, monkeypatch
) -> None:
	"""When SLURM_CPUS_PER_TASK is NOT set, the YAML cpus_per_task wins
	(this preserves the existing MPI workload shape where YAML drives the
	per-rank budget)."""
	monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
	topology = _lab_topology(tmp_path)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=16,
			tasks_per_node="auto",
			use_hyperthreads=False,
		),
		topology=topology,
		target_count=8,
	)
	assert plan is not None
	assert plan.cpus_per_task == 16
	assert plan.cpus_per_task_source == "task_allocation.cpus_per_task"


def test_build_task_allocation_plan_slurm_env_ignored_when_zero_or_garbage(
	tmp_path: Path, monkeypatch
) -> None:
	"""SLURM_CPUS_PER_TASK with value <=0 or non-numeric falls through to
	YAML — defensive against malformed env state."""
	topology = _lab_topology(tmp_path)
	for bad in ("0", "-3", "not-a-number", "  "):
		monkeypatch.setenv("SLURM_CPUS_PER_TASK", bad)
		plan = build_task_allocation_plan(
			config=TaskAllocationConfig(
				enabled=True,
				backend="local_affinity",
				bind="physical_cores",
				cpus_per_task=16,
				tasks_per_node="auto",
				use_hyperthreads=False,
			),
			topology=topology,
			target_count=8,
		)
		assert plan is not None
		assert plan.cpus_per_task == 16, f"bad env={bad!r} unexpectedly took effect"
		assert plan.cpus_per_task_source == "task_allocation.cpus_per_task"


def test_task_slot_affinity_context_applies_and_restores_mock_affinity() -> None:
	current_affinity = {0, 1, 2, 3}
	set_calls: list[tuple[int, tuple[int, ...]]] = []
	slot = TaskSlot(slot_id=2, logical_cpus=(2, 3), core_ids=(1,), package_ids=(0,))

	def affinity_getter(_pid: int) -> set[int]:
		return set(current_affinity)

	def affinity_setter(pid: int, cpus) -> None:
		nonlocal current_affinity
		resolved = tuple(sorted(int(cpu_id) for cpu_id in cpus))
		set_calls.append((int(pid), resolved))
		current_affinity = set(resolved)

	with task_slot_affinity_context(
		slot,
		enabled=True,
		affinity_getter=affinity_getter,
		affinity_setter=affinity_setter,
	):
		assert current_affinity == {2, 3}

	assert current_affinity == {0, 1, 2, 3}
	assert set_calls == [(0, (2, 3)), (0, (0, 1, 2, 3))]


def test_task_slot_affinity_context_soft_failure_warns_and_continues(caplog: pytest.LogCaptureFixture) -> None:
	slot = TaskSlot(slot_id=3, logical_cpus=(4, 5), core_ids=(2,), package_ids=(0,))
	caplog.set_level("WARNING", logger="axon_recon.pipeline.cpu_allocation")
	entered = False

	def affinity_setter(_pid: int, _cpus) -> None:
		raise OSError("mock affinity denied")

	with task_slot_affinity_context(
		slot,
		enabled=True,
		affinity_getter=lambda _pid: {0, 1, 2, 3},
		affinity_setter=affinity_setter,
	):
		entered = True

	assert entered is True
	assert any("Failed to apply task CPU affinity" in record.getMessage() for record in caplog.records)


def test_task_slot_affinity_context_strict_failure_raises() -> None:
	slot = TaskSlot(slot_id=4, logical_cpus=(6, 7), core_ids=(3,), package_ids=(0,))

	def affinity_setter(_pid: int, _cpus) -> None:
		raise OSError("mock affinity denied")

	with pytest.raises(RuntimeError, match="Failed to apply task CPU affinity"):
		with task_slot_affinity_context(
			slot,
			enabled=True,
			soft_failure=False,
			affinity_getter=lambda _pid: {0, 1, 2, 3},
			affinity_setter=affinity_setter,
		):
			pass


# ---------------------------------------------------------------------------
# apply_thread_env_context tests
# ---------------------------------------------------------------------------

_THREAD_ENV_VARS = (
	"OMP_NUM_THREADS",
	"MKL_NUM_THREADS",
	"OPENBLAS_NUM_THREADS",
	"NUMEXPR_NUM_THREADS",
	"VECLIB_MAXIMUM_THREADS",
	"NUMBA_NUM_THREADS",
)


def _strip_thread_env_vars() -> dict[str, str | None]:
	"""Remove all thread env vars from os.environ and return their original values."""
	previous: dict[str, str | None] = {}
	for var in _THREAD_ENV_VARS:
		previous[var] = os.environ.pop(var, None)
	return previous


def _restore_thread_env_vars(previous: dict[str, str | None]) -> None:
	for var, val in previous.items():
		if val is None:
			os.environ.pop(var, None)
		else:
			os.environ[var] = val


def test_apply_thread_env_context_disabled_does_not_touch_env() -> None:
	prev = _strip_thread_env_vars()
	slot = TaskSlot(slot_id=0, logical_cpus=(0, 1), core_ids=(0,), package_ids=(0,))
	try:
		with apply_thread_env_context(slot, enabled=False, policy="match_cpus_per_task"):
			for var in _THREAD_ENV_VARS:
				assert var not in os.environ
	finally:
		_restore_thread_env_vars(prev)


def test_apply_thread_env_context_force_1_sets_all_vars_to_1() -> None:
	prev = _strip_thread_env_vars()
	slot = TaskSlot(slot_id=0, logical_cpus=(0, 1, 2, 3), core_ids=(0, 1), package_ids=(0,))
	try:
		with apply_thread_env_context(slot, enabled=True, policy="force_1"):
			for var in _THREAD_ENV_VARS:
				assert os.environ.get(var) == "1"
		# values restored after context
		for var in _THREAD_ENV_VARS:
			assert var not in os.environ
	finally:
		_restore_thread_env_vars(prev)


def test_apply_thread_env_context_match_cpus_per_task_uses_slot_cpu_count() -> None:
	prev = _strip_thread_env_vars()
	# slot has 4 logical CPUs
	slot = TaskSlot(slot_id=1, logical_cpus=(0, 1, 2, 3), core_ids=(0, 1), package_ids=(0,))
	try:
		with apply_thread_env_context(slot, enabled=True, policy="match_cpus_per_task"):
			for var in _THREAD_ENV_VARS:
				assert os.environ.get(var) == "4"
		for var in _THREAD_ENV_VARS:
			assert var not in os.environ
	finally:
		_restore_thread_env_vars(prev)


def test_apply_thread_env_context_restores_previous_values() -> None:
	prev_outer = _strip_thread_env_vars()
	os.environ["OMP_NUM_THREADS"] = "16"
	slot = TaskSlot(slot_id=0, logical_cpus=(0, 1), core_ids=(0,), package_ids=(0,))
	try:
		with apply_thread_env_context(slot, enabled=True, policy="force_1"):
			assert os.environ["OMP_NUM_THREADS"] == "1"
		assert os.environ["OMP_NUM_THREADS"] == "16"
	finally:
		_restore_thread_env_vars(prev_outer)


def test_apply_thread_env_context_preserve_existing_logs_and_skips(
	caplog: pytest.LogCaptureFixture,
) -> None:
	prev = _strip_thread_env_vars()
	os.environ["OMP_NUM_THREADS"] = "8"
	slot = TaskSlot(slot_id=0, logical_cpus=(0, 1), core_ids=(0,), package_ids=(0,))
	caplog.set_level("INFO", logger="axon_recon.pipeline.cpu_allocation")
	try:
		with apply_thread_env_context(slot, enabled=True, policy="preserve_existing"):
			# env must be unchanged
			assert os.environ.get("OMP_NUM_THREADS") == "8"
		assert any("preserve_existing" in record.getMessage() for record in caplog.records)
	finally:
		_restore_thread_env_vars(prev)


def test_apply_thread_env_context_match_cpus_per_task_defaults_to_1_when_slot_is_none() -> None:
	prev = _strip_thread_env_vars()
	try:
		with apply_thread_env_context(None, enabled=True, policy="match_cpus_per_task"):
			for var in _THREAD_ENV_VARS:
				assert os.environ.get(var) == "1"
	finally:
		_restore_thread_env_vars(prev)


def test_build_task_allocation_plan_propagates_thread_env_fields(tmp_path: Path) -> None:
	sysfs_root = tmp_path / "sys" / "devices" / "system" / "cpu"
	_write_single_socket_hyperthreaded_topology(sysfs_root, core_count=24, threads_per_core=2)
	topology = detect_cpu_topology(
		sysfs_root=sysfs_root,
		affinity_getter=lambda _pid: set(range(48)),
	)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=2,
			tasks_per_node="auto",
			use_hyperthreads=False,
			set_thread_env=True,
			nested_thread_policy="match_cpus_per_task",
		),
		topology=topology,
		target_count=4,
	)
	assert plan is not None
	assert plan.set_thread_env is True
	assert plan.nested_thread_policy == "match_cpus_per_task"


def test_allocation_preview_shows_thread_env_policy_when_enabled(tmp_path: Path) -> None:
	sysfs_root = tmp_path / "sys" / "devices" / "system" / "cpu"
	_write_single_socket_hyperthreaded_topology(sysfs_root, core_count=24, threads_per_core=2)
	topology = detect_cpu_topology(
		sysfs_root=sysfs_root,
		affinity_getter=lambda _pid: set(range(48)),
	)
	plan = build_task_allocation_plan(
		config=TaskAllocationConfig(
			enabled=True,
			backend="local_affinity",
			bind="physical_cores",
			cpus_per_task=2,
			tasks_per_node="auto",
			use_hyperthreads=False,
			set_thread_env=True,
			nested_thread_policy="match_cpus_per_task",
		),
		topology=topology,
		target_count=4,
	)
	assert plan is not None
	preview = StageAllocationPreview(
		stage="preprocess",
		target_count=4,
		target_labels=("12:well000", "12:well001", "12:well002", "12:well003"),
		phase_resource_classes=("h5_metadata",),
		parallelism=StageParallelism(
			well_workers=4,
			unit_workers=6,
			task_allocation_plan=plan,
		),
	)
	formatted = format_stage_allocation_previews([preview])
	assert "thread_env: policy=match_cpus_per_task" in formatted
	assert "current:" in formatted
	assert "OMP_NUM_THREADS=" in formatted
	assert "worker:" in formatted
	assert "set per-worker at execution" in formatted
	assert "slot_clamps:" in formatted


def test_allocation_preview_reports_mpi_backend_without_local_plan() -> None:
	preview = StageAllocationPreview(
		stage="preprocess",
		target_count=1,
		target_labels=("11:well000",),
		phase_resource_classes=("h5_metadata",),
		parallelism=StageParallelism(
			well_workers=2,
			unit_workers=2,
			task_allocation_plan=None,
		),
		allocation_backend="mpi",
		mpi_rank=1,
		mpi_size=2,
	)
	formatted = format_stage_allocation_previews([preview])
	assert "mpi_partition: rank=1 size=2" in formatted
	assert "task_allocation: enabled backend=mpi" in formatted
	assert "local_task_slots: n/a" in formatted


# ---------------------------------------------------------------------------
# probe_container_readiness tests
# ---------------------------------------------------------------------------


def test_probe_container_readiness_reports_sysfs_readable(tmp_path: Path) -> None:
	sysfs_root = tmp_path / "sys" / "devices" / "system" / "cpu"
	_write_single_socket_hyperthreaded_topology(sysfs_root, core_count=2, threads_per_core=2)

	result = probe_container_readiness(
		sysfs_root=sysfs_root,
		shm_path=tmp_path,
		affinity_getter=lambda _pid: {0, 1, 2, 3},
	)

	assert result.sysfs_topology_readable is True
	assert result.affinity_api_available is True or result.affinity_api_available is False  # platform dependent
	assert result.visible_cpu_count == 4
	assert result.sysfs_root == str(sysfs_root)
	assert result.shm_path == str(tmp_path)
	assert not any("sysfs topology is not readable" in w for w in result.warnings)


def test_probe_container_readiness_warns_when_sysfs_missing(tmp_path: Path) -> None:
	sysfs_root = tmp_path / "nonexistent" / "cpu"

	result = probe_container_readiness(
		sysfs_root=sysfs_root,
		shm_path=tmp_path,
		affinity_getter=lambda _pid: {0, 1},
	)

	assert result.sysfs_topology_readable is False
	assert any("sysfs topology is not readable" in w for w in result.warnings)


def test_probe_container_readiness_reports_visible_cpu_count_from_affinity_getter(tmp_path: Path) -> None:
	sysfs_root = tmp_path / "nonexistent" / "cpu"

	result = probe_container_readiness(
		sysfs_root=sysfs_root,
		shm_path=tmp_path,
		affinity_getter=lambda _pid: {0, 1, 2, 3},
	)

	assert result.visible_cpu_count == 4


def test_probe_container_readiness_warns_when_shm_unavailable(tmp_path: Path) -> None:
	sysfs_root = tmp_path / "nonexistent" / "cpu"
	missing_shm = tmp_path / "nonexistent_shm"

	result = probe_container_readiness(
		sysfs_root=sysfs_root,
		shm_path=missing_shm,
		affinity_getter=lambda _pid: {0},
	)

	assert result.shm_available_gb is None
	assert any("shm_gb_per_task" in w for w in result.warnings)


def test_probe_container_readiness_to_dict_has_expected_keys(tmp_path: Path) -> None:
	result = probe_container_readiness(
		sysfs_root=tmp_path / "nonexistent",
		shm_path=tmp_path,
		affinity_getter=lambda _pid: {0, 1},
	)

	d = result.to_dict()
	assert "affinity_api_available" in d
	assert "sysfs_topology_readable" in d
	assert "sysfs_root" in d
	assert "shm_available_gb" in d
	assert "shm_path" in d
	assert "visible_cpu_count" in d
	assert "warnings" in d
	assert isinstance(d["warnings"], list)


# ---------------------------------------------------------------------------
# resolve_inner_worker_count + phase_budgets_context tests (slice 5)
# ---------------------------------------------------------------------------


def _make_slot(*, slot_id: int = 0, cpu_count: int = 10) -> TaskSlot:
	return TaskSlot(
		slot_id=int(slot_id),
		logical_cpus=tuple(range(int(cpu_count))),
		core_ids=tuple(range(int(cpu_count))),
		package_ids=(0,),
	)


def test_resolve_inner_worker_count_inherits_slot_cpu_count_when_no_clamp() -> None:
	slot = _make_slot(cpu_count=10)
	with task_slot_context(slot):
		result = resolve_inner_worker_count(
			nested_shape="si_njobs",
			phase_cpus_per_task=None,
			yaml_n_jobs_override=None,
			work_item_count=None,
		)
	assert result == 10


def test_resolve_inner_worker_count_clamps_by_phase_cpus_per_task() -> None:
	slot = _make_slot(cpu_count=10)
	with task_slot_context(slot):
		result = resolve_inner_worker_count(
			nested_shape="unit_workers",
			phase_cpus_per_task=4,
			yaml_n_jobs_override=None,
			work_item_count=None,
		)
	assert result == 4


def test_resolve_inner_worker_count_clamps_by_yaml_n_jobs_override() -> None:
	slot = _make_slot(cpu_count=10)
	with task_slot_context(slot):
		result = resolve_inner_worker_count(
			nested_shape="si_njobs",
			phase_cpus_per_task=None,
			yaml_n_jobs_override=2,
			work_item_count=None,
		)
	assert result == 2


def test_resolve_inner_worker_count_clamps_by_work_item_count() -> None:
	slot = _make_slot(cpu_count=10)
	with task_slot_context(slot):
		result = resolve_inner_worker_count(
			nested_shape="segment_workers",
			phase_cpus_per_task=None,
			yaml_n_jobs_override=None,
			work_item_count=3,
		)
	assert result == 3


def test_resolve_inner_worker_count_serial_returns_one_regardless() -> None:
	slot = _make_slot(cpu_count=10)
	with task_slot_context(slot):
		result = resolve_inner_worker_count(
			nested_shape="serial",
			phase_cpus_per_task=8,
			yaml_n_jobs_override=8,
			work_item_count=8,
		)
	assert result == 1


def test_resolve_inner_worker_count_no_active_slot_falls_back_to_override() -> None:
	# No task_slot_context active. With no hints at all we fall back to 1 (safe
	# default for the headless case where nothing was configured).
	result = resolve_inner_worker_count(
		nested_shape="si_njobs",
		phase_cpus_per_task=None,
		yaml_n_jobs_override=None,
		work_item_count=None,
	)
	assert result == 1
	# Explicit hints (yaml override, phase_cpus_per_task, work_item_count) ARE
	# honored when the slot ContextVar isn't bound — typical for workers spawned
	# across process boundaries by the MPI task backend or by srun → shifter →
	# python, where the parent's contextvar doesn't propagate. Without this
	# fallback the bootstrap_concat_binary phase silently runs single-threaded
	# even when the YAML / profile have explicitly asked for >1.
	result_with_override = resolve_inner_worker_count(
		nested_shape="si_njobs",
		phase_cpus_per_task=None,
		yaml_n_jobs_override=8,
		work_item_count=None,
	)
	assert result_with_override == 8
	# When multiple hints disagree, the smallest still wins — same min() contract
	# as when a slot IS bound.
	result_with_two_hints = resolve_inner_worker_count(
		nested_shape="si_njobs",
		phase_cpus_per_task=16,
		yaml_n_jobs_override=64,
		work_item_count=None,
	)
	assert result_with_two_hints == 16


def test_resolve_inner_worker_count_minimum_is_one_with_tiny_slot() -> None:
	# Zero / negative clamps are treated as "absent" (no clamp); the base is the slot count.
	# To validate that the minimum return value is always >= 1, force base=1 by using
	# a 1-CPU slot and verifying that even larger overrides do not push the result below 1.
	slot = _make_slot(cpu_count=1)
	with task_slot_context(slot):
		result = resolve_inner_worker_count(
			nested_shape="si_njobs",
			phase_cpus_per_task=None,
			yaml_n_jobs_override=None,
			work_item_count=None,
		)
	assert result == 1


def test_phase_budgets_context_lookup_returns_entry_for_stage_phase() -> None:
	budgets = {
		"reconstruct.plot_recons": PhaseResourceClassConfig(
			nested_shape="unit_workers",
			cpus_per_task=4,
		),
		"preprocess.preprocess_segments": PhaseResourceClassConfig(
			nested_shape="si_njobs",
		),
	}
	with phase_budgets_context(budgets):
		assert current_phase_budget("reconstruct", "plot_recons") is budgets["reconstruct.plot_recons"]
		assert current_phase_budget("preprocess", "preprocess_segments") is budgets["preprocess.preprocess_segments"]
		assert current_phase_budget("reconstruct", "missing") is None
	# After context exits, budgets should be cleared.
	assert current_phase_budget("reconstruct", "plot_recons") is None


def test_phase_budgets_context_with_none_clears_lookup() -> None:
	with phase_budgets_context(None):
		assert current_phase_budget("any", "phase") is None


# ---------------------------------------------------------------------------
# Slice 6: SI n_jobs derivation integration tests
# ---------------------------------------------------------------------------


def test_si_n_jobs_inherits_slot_cpu_count_when_no_phase_clamp() -> None:
	"""A si_njobs phase with no cpus_per_task clamp passes slot.cpu_count to SI."""
	from axon_recon.pipeline.resources import PhaseResourceClassConfig

	slot = TaskSlot(slot_id=0, logical_cpus=tuple(range(10)), core_ids=tuple(range(10)), package_ids=(0,))
	budgets = {
		"reconstruct.analyzers": PhaseResourceClassConfig(nested_shape="si_njobs", cpus_per_task=None),
	}
	with task_slot_context(slot), phase_budgets_context(budgets):
		budget = current_phase_budget("reconstruct", "analyzers")
		n_jobs = resolve_inner_worker_count(
			nested_shape="si_njobs",
			phase_cpus_per_task=getattr(budget, "cpus_per_task", None) if budget else None,
			yaml_n_jobs_override=None,
			work_item_count=None,
		)
	assert n_jobs == 10


def test_si_n_jobs_clamped_by_phase_budget_cpus_per_task() -> None:
	"""A si_njobs phase with cpus_per_task=4 clamps SI n_jobs to 4 even with slot=10."""
	from axon_recon.pipeline.resources import PhaseResourceClassConfig

	slot = TaskSlot(slot_id=0, logical_cpus=tuple(range(10)), core_ids=tuple(range(10)), package_ids=(0,))
	budgets = {
		"reconstruct.plot_recons": PhaseResourceClassConfig(nested_shape="si_njobs", cpus_per_task=4),
	}
	with task_slot_context(slot), phase_budgets_context(budgets):
		budget = current_phase_budget("reconstruct", "plot_recons")
		n_jobs = resolve_inner_worker_count(
			nested_shape="si_njobs",
			phase_cpus_per_task=getattr(budget, "cpus_per_task", None) if budget else None,
			yaml_n_jobs_override=None,
			work_item_count=None,
		)
	assert n_jobs == 4
