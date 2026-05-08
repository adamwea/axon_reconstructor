"""MPI adapter for optional multi-rank distributed execution.

This module provides:
- Safe MPI detection and initialization (does not require mpi4py at import time).
- Rank/size detection and communicator access behind a context manager.
- FakeMPI for testing deterministic rank partitioning without mpirun.
- Structured logging for rank metadata when MPI is active.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
import logging
import os


LOGGER = logging.getLogger("axon_recon.pipeline.mpi_adapter")


@dataclass(frozen=True)
class MPIContext:
	"""Snapshot of current MPI state."""

	rank: int
	"""Rank of this process (0-based)."""
	size: int
	"""Total number of ranks."""
	comm: Any | None
	"""MPI communicator object (mpi4py.MPI.Comm), or None if fake/unavailable."""
	is_fake: bool
	"""True if this is a FakeMPI context for testing."""

	@property
	def is_rank_0(self) -> bool:
		return int(self.rank) == 0

	@property
	def is_single_rank(self) -> bool:
		return int(self.size) == 1


def _is_mpi4py_available() -> bool:
	"""Check whether mpi4py can be imported."""
	try:
		import mpi4py  # noqa: F401
		return True
	except ImportError:
		return False


def _parse_int_env(name: str) -> int | None:
	raw = os.environ.get(name)
	if raw is None:
		return None
	text = str(raw).strip()
	if not text:
		return None
	try:
		return int(text)
	except Exception:
		return None


def _context_from_mpi_env() -> MPIContext | None:
	"""Detect MPI rank/size from common launcher environment variables."""
	pairs: tuple[tuple[str, str], ...] = (
		("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
		("PMI_RANK", "PMI_SIZE"),
		("PMIX_RANK", "PMIX_SIZE"),
		("SLURM_PROCID", "SLURM_NTASKS"),
	)
	for rank_var, size_var in pairs:
		rank = _parse_int_env(rank_var)
		size = _parse_int_env(size_var)
		if rank is None or size is None:
			continue
		if int(size) <= 1:
			continue
		if int(rank) < 0 or int(rank) >= int(size):
			continue
		return MPIContext(rank=int(rank), size=int(size), comm=None, is_fake=False)
	return None


def _get_mpi_context() -> MPIContext | None:
	"""Detect and return current MPI context if running under mpirun.

	Returns None if MPI is not available or process is not an MPI rank.
	"""
	# Prefer launcher-provided environment variables first. This avoids importing
	# mpi4py in mixed MPI stacks (e.g., OpenMPI launcher with MPICH-linked mpi4py),
	# which can abort the process before Python can catch an exception.
	env_context = _context_from_mpi_env()
	if env_context is not None:
		return env_context
	if _is_mpi4py_available():
		try:
			from mpi4py import MPI

			comm = MPI.COMM_WORLD
			rank = int(comm.Get_rank())
			size = int(comm.Get_size())
			if int(size) > 1:
				return MPIContext(rank=rank, size=size, comm=comm, is_fake=False)
		except Exception:
			pass
	return None


_CURRENT_MPI_CONTEXT: MPIContext | None = None


@contextmanager
def mpi_context(context: MPIContext | None) -> Iterator[None]:
	"""Set the current MPI context for this scope (for testing and management)."""
	global _CURRENT_MPI_CONTEXT
	old_context = _CURRENT_MPI_CONTEXT
	try:
		_CURRENT_MPI_CONTEXT = context
		yield
	finally:
		_CURRENT_MPI_CONTEXT = old_context


def current_mpi_context() -> MPIContext | None:
	"""Return the current MPI context (auto-detect if not explicitly set)."""
	global _CURRENT_MPI_CONTEXT
	if _CURRENT_MPI_CONTEXT is not None:
		return _CURRENT_MPI_CONTEXT
	return _get_mpi_context()


@dataclass(frozen=True)
class FakeMPI:
	"""Fake MPI context for deterministic testing without mpirun.

	Generates rank-partitioned target lists and simulates collective operations.
	"""

	rank: int
	"""Rank of this simulated process."""
	size: int
	"""Total number of simulated ranks."""

	def to_context(self) -> MPIContext:
		"""Convert to MPIContext for use in code expecting MPI."""
		return MPIContext(rank=int(self.rank), size=int(self.size), comm=None, is_fake=True)

	def partition_targets(self, targets: list[Any]) -> list[Any]:
		"""Partition a flat target list round-robin by rank.

		Returns only the targets assigned to this rank.
		"""
		target_count = len(targets)
		if target_count == 0:
			return []
		assigned = [targets[i] for i in range(target_count) if i % int(self.size) == int(self.rank)]
		return assigned

	def broadcast_from_rank_0(self, value: Any) -> Any:
		"""Simulate MPI broadcast from rank 0 (return value unchanged)."""
		return value

	def gather_to_rank_0(self, local_value: Any) -> list[Any] | None:
		"""Simulate MPI gather to rank 0 (return dict on rank 0, None elsewhere)."""
		if int(self.rank) == 0:
			return [local_value]  # In real MPI, this would be all gathered values
		return None

	def barrier(self) -> None:
		"""Simulate MPI barrier (no-op in fake MPI)."""
		pass


def partition_targets_by_mpi_rank(*, targets: list[Any], mpi_context: MPIContext | None) -> list[Any]:
	"""Partition targets round-robin by MPI rank.

	If mpi_context is None or rank/size is invalid, returns all targets.
	"""
	if mpi_context is None or int(mpi_context.size) <= 1:
		return list(targets)
	return [targets[i] for i in range(len(targets)) if i % int(mpi_context.size) == int(mpi_context.rank)]


def log_mpi_context(logger: logging.Logger | None = None, context: MPIContext | None = None) -> None:
	"""Log the current MPI context if active."""
	resolved_logger = logger or LOGGER
	resolved_context = context or current_mpi_context()
	if resolved_context is None or resolved_context.is_single_rank:
		return
	resolved_logger.info(
		"MPI context: rank=%d size=%d is_rank_0=%s is_fake=%s",
		int(resolved_context.rank),
		int(resolved_context.size),
		resolved_context.is_rank_0,
		resolved_context.is_fake,
		extra={
			"event": "mpi_context",
			"mpi_rank": int(resolved_context.rank),
			"mpi_size": int(resolved_context.size),
			"mpi_is_rank_0": resolved_context.is_rank_0,
			"mpi_is_fake": resolved_context.is_fake,
		},
	)
