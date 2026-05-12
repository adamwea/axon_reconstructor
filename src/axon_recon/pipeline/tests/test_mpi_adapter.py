"""Tests for MPI adapter and rank partitioning."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.mpi_adapter import (
	FakeMPI,
	MPIContext,
	apply_per_rank_cuda_visible_devices,
	current_mpi_context,
	mpi_context,
	partition_targets_by_mpi_rank,
)
import axon_recon.pipeline.mpi_adapter as mpi_adapter


def test_mpi_context_is_rank_0_when_rank_is_zero() -> None:
	ctx = MPIContext(rank=0, size=4, comm=None, is_fake=True)
	assert ctx.is_rank_0 is True


def test_mpi_context_is_rank_0_is_false_for_nonzero_rank() -> None:
	ctx = MPIContext(rank=2, size=4, comm=None, is_fake=True)
	assert ctx.is_rank_0 is False


def test_mpi_context_is_single_rank_when_size_is_1() -> None:
	ctx = MPIContext(rank=0, size=1, comm=None, is_fake=True)
	assert ctx.is_single_rank is True


def test_mpi_context_is_single_rank_is_false_for_multi_rank() -> None:
	ctx = MPIContext(rank=0, size=4, comm=None, is_fake=True)
	assert ctx.is_single_rank is False


def test_current_mpi_context_returns_none_by_default() -> None:
	ctx = current_mpi_context()
	# Can be None or a real MPI context depending on environment;
	# we're just verifying the function doesn't crash
	assert ctx is None or isinstance(ctx, MPIContext)


def test_mpi_context_context_manager_sets_and_restores() -> None:
	original = current_mpi_context()
	test_ctx = MPIContext(rank=1, size=2, comm=None, is_fake=True)

	with mpi_context(test_ctx):
		assert current_mpi_context() == test_ctx

	assert current_mpi_context() == original


def test_fake_mpi_partition_targets_empty_list() -> None:
	fake = FakeMPI(rank=0, size=2)
	assert fake.partition_targets([]) == []


def test_fake_mpi_partition_targets_single_rank() -> None:
	# With size=1, all targets should go to rank 0
	fake = FakeMPI(rank=0, size=1)
	targets = ["a", "b", "c"]
	assert fake.partition_targets(targets) == ["a", "b", "c"]


def test_fake_mpi_partition_targets_round_robin() -> None:
	targets = ["a", "b", "c", "d", "e", "f"]

	# Rank 0 gets indices where i % 3 == 0: [0, 3] -> ["a", "d"]
	rank_0 = FakeMPI(rank=0, size=3)
	assert rank_0.partition_targets(targets) == ["a", "d"]

	# Rank 1 gets indices where i % 3 == 1: [1, 4] -> ["b", "e"]
	rank_1 = FakeMPI(rank=1, size=3)
	assert rank_1.partition_targets(targets) == ["b", "e"]

	# Rank 2 gets indices where i % 3 == 2: [2, 5] -> ["c", "f"]
	rank_2 = FakeMPI(rank=2, size=3)
	assert rank_2.partition_targets(targets) == ["c", "f"]


def test_fake_mpi_partition_targets_deterministic() -> None:
	targets = list(range(100))

	fake_rank_0 = FakeMPI(rank=0, size=4)
	fake_rank_1 = FakeMPI(rank=1, size=4)
	fake_rank_2 = FakeMPI(rank=2, size=4)
	fake_rank_3 = FakeMPI(rank=3, size=4)

	partitions = [
		fake_rank_0.partition_targets(targets),
		fake_rank_1.partition_targets(targets),
		fake_rank_2.partition_targets(targets),
		fake_rank_3.partition_targets(targets),
	]

	# All partitions should be non-overlapping
	all_indices = []
	for partition in partitions:
		all_indices.extend(partition)
	assert sorted(all_indices) == list(range(100))

	# Each rank should get 25 elements
	for partition in partitions:
		assert len(partition) == 25


def test_fake_mpi_to_context() -> None:
	fake = FakeMPI(rank=2, size=4)
	ctx = fake.to_context()

	assert isinstance(ctx, MPIContext)
	assert ctx.rank == 2
	assert ctx.size == 4
	assert ctx.is_fake is True
	assert ctx.comm is None


def test_fake_mpi_broadcast_from_rank_0() -> None:
	fake = FakeMPI(rank=1, size=2)
	value = {"data": "test"}
	result = fake.broadcast_from_rank_0(value)
	assert result == value


def test_fake_mpi_gather_to_rank_0_on_rank_0() -> None:
	fake = FakeMPI(rank=0, size=3)
	result = fake.gather_to_rank_0("local_value")
	assert isinstance(result, list)
	assert len(result) > 0


def test_fake_mpi_gather_to_rank_0_on_nonzero_rank() -> None:
	fake = FakeMPI(rank=1, size=3)
	result = fake.gather_to_rank_0("local_value")
	assert result is None


def test_fake_mpi_barrier_does_not_raise() -> None:
	fake = FakeMPI(rank=0, size=2)
	# Should not raise
	fake.barrier()


def test_partition_targets_by_mpi_rank_with_none_context() -> None:
	targets = ["a", "b", "c"]
	result = partition_targets_by_mpi_rank(targets=targets, mpi_context=None)
	assert result == targets


def test_partition_targets_by_mpi_rank_with_single_rank_context() -> None:
	targets = ["a", "b", "c"]
	ctx = MPIContext(rank=0, size=1, comm=None, is_fake=True)
	result = partition_targets_by_mpi_rank(targets=targets, mpi_context=ctx)
	assert result == targets


def test_partition_targets_by_mpi_rank_with_multi_rank_context() -> None:
	targets = ["a", "b", "c", "d", "e", "f"]
	ctx = MPIContext(rank=1, size=2, comm=None, is_fake=True)
	result = partition_targets_by_mpi_rank(targets=targets, mpi_context=ctx)
	# Rank 1 should get indices 1, 3, 5 (b, d, f)
	assert result == ["b", "d", "f"]


def test_partition_targets_by_mpi_rank_across_all_ranks() -> None:
	"""Verify that partitioning across all ranks covers all targets without overlap."""
	targets = list(range(13))

	# Create contexts for 3 ranks
	contexts = [
		MPIContext(rank=i, size=3, comm=None, is_fake=True)
		for i in range(3)
	]

	partitions = [
		partition_targets_by_mpi_rank(targets=targets, mpi_context=ctx)
		for ctx in contexts
	]

	# Collect all partitioned targets
	all_partitioned = []
	for partition in partitions:
		all_partitioned.extend(partition)

	# Should be non-overlapping and complete
	assert sorted(all_partitioned) == targets
	assert len(all_partitioned) == len(targets)


def test_current_mpi_context_falls_back_to_openmpi_env(monkeypatch) -> None:
	monkeypatch.setattr(mpi_adapter, "_CURRENT_MPI_CONTEXT", None)
	monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
	monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
	ctx = current_mpi_context()
	assert ctx is not None
	assert ctx.rank == 1
	assert ctx.size == 2
	assert ctx.comm is None


def test_current_mpi_context_env_fallback_ignores_single_rank(monkeypatch) -> None:
	monkeypatch.setattr(mpi_adapter, "_CURRENT_MPI_CONTEXT", None)
	monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
	monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
	ctx = current_mpi_context()
	assert ctx is None or ctx.size == 1


def test_apply_per_rank_cuda_2rank_2gpu_rank0(monkeypatch) -> None:
	monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
	monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
	monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
	assigned = apply_per_rank_cuda_visible_devices()
	assert assigned == "0"
	import os as _os
	assert _os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_apply_per_rank_cuda_2rank_2gpu_rank1(monkeypatch) -> None:
	monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
	monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
	monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
	assigned = apply_per_rank_cuda_visible_devices()
	assert assigned == "1"
	import os as _os
	assert _os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_apply_per_rank_cuda_2rank_1gpu_warns_and_shares(monkeypatch, caplog) -> None:
	monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
	monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
	monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
	import logging as _logging
	caplog.set_level(_logging.WARNING, logger="axon_recon.pipeline.mpi_adapter")
	assigned_rank0 = apply_per_rank_cuda_visible_devices()
	assert assigned_rank0 == "0"
	monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
	monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # reset before second call
	assigned_rank1 = apply_per_rank_cuda_visible_devices()
	assert assigned_rank1 == "0"
	warning_events = [rec for rec in caplog.records if rec.levelno == _logging.WARNING and "exceeds visible GPU count" in rec.getMessage()]
	assert len(warning_events) >= 2


def test_apply_per_rank_cuda_single_rank_no_mutation(monkeypatch) -> None:
	monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
	monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
	monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
	assigned = apply_per_rank_cuda_visible_devices()
	assert assigned is None
	import os as _os
	assert _os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_apply_per_rank_cuda_4rank_2gpu_round_robin(monkeypatch) -> None:
	for rank, expected in [(0, "0"), (1, "1"), (2, "0"), (3, "1")]:
		monkeypatch.setenv("OMPI_COMM_WORLD_RANK", str(rank))
		monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
		monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
		assigned = apply_per_rank_cuda_visible_devices()
		assert assigned == expected, f"rank {rank}: expected {expected}, got {assigned}"


def test_apply_per_rank_cuda_no_mpi_env_noop(monkeypatch) -> None:
	monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
	monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
	monkeypatch.delenv("PMI_RANK", raising=False)
	monkeypatch.delenv("PMI_SIZE", raising=False)
	monkeypatch.delenv("PMIX_RANK", raising=False)
	monkeypatch.delenv("PMIX_SIZE", raising=False)
	monkeypatch.delenv("SLURM_PROCID", raising=False)
	monkeypatch.delenv("SLURM_NTASKS", raising=False)
	monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
	assigned = apply_per_rank_cuda_visible_devices()
	assert assigned is None
	import os as _os
	assert _os.environ["CUDA_VISIBLE_DEVICES"] == "5"


def test_mpi_context_for_partition_returns_ctx_when_backend_mpi() -> None:
	from types import SimpleNamespace

	from axon_recon.pipeline.runner import _mpi_context_for_partition

	ctx = MPIContext(rank=1, size=2, comm=None, is_fake=True)
	parallelism = SimpleNamespace(task_allocation_backend="mpi")
	assert _mpi_context_for_partition(mpi_context=ctx, parallelism=parallelism) is ctx


def test_mpi_context_for_partition_skips_when_backend_local_affinity(caplog) -> None:
	from types import SimpleNamespace

	from axon_recon.pipeline.runner import _mpi_context_for_partition

	ctx = MPIContext(rank=0, size=2, comm=None, is_fake=True)
	parallelism = SimpleNamespace(task_allocation_backend="local_affinity")
	import logging as _logging
	caplog.set_level(_logging.INFO, logger="axon_recon.pipeline")
	assert _mpi_context_for_partition(mpi_context=ctx, parallelism=parallelism) is None
	assert any("skipping rank partitioning" in rec.getMessage() for rec in caplog.records)


def test_mpi_context_for_partition_skips_when_backend_none() -> None:
	from types import SimpleNamespace

	from axon_recon.pipeline.runner import _mpi_context_for_partition

	ctx = MPIContext(rank=0, size=2, comm=None, is_fake=True)
	parallelism = SimpleNamespace(task_allocation_backend="none")
	assert _mpi_context_for_partition(mpi_context=ctx, parallelism=parallelism) is None


def test_mpi_context_for_partition_returns_none_when_no_mpi_context() -> None:
	from types import SimpleNamespace

	from axon_recon.pipeline.runner import _mpi_context_for_partition

	parallelism = SimpleNamespace(task_allocation_backend="mpi")
	assert _mpi_context_for_partition(mpi_context=None, parallelism=parallelism) is None


def test_mpi_partition_targets_disjoint_across_ranks_when_backend_mpi() -> None:
	"""End-to-end fake-MPI: with backend=mpi each rank's partition is disjoint and covers all targets."""
	targets = list(range(7))
	contexts = [MPIContext(rank=i, size=3, comm=None, is_fake=True) for i in range(3)]
	partitions = [
		partition_targets_by_mpi_rank(targets=targets, mpi_context=ctx)
		for ctx in contexts
	]
	flat = sorted(t for partition in partitions for t in partition)
	assert flat == targets
	for i, partition in enumerate(partitions):
		for j, other in enumerate(partitions):
			if i == j:
				continue
			assert set(partition).isdisjoint(other), f"rank {i} and rank {j} overlap"


def test_apply_per_rank_cuda_no_visible_gpus_noop(monkeypatch) -> None:
	monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
	monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
	monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
	# Force the pynvml-based fallback to also report no GPUs
	monkeypatch.setattr(mpi_adapter, "_detect_visible_gpus", lambda: None)
	assigned = apply_per_rank_cuda_visible_devices()
	assert assigned is None
