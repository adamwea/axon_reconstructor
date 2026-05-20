from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context import ExecutionTarget


@dataclass(frozen=True)
class TargetStageResult:
    target: ExecutionTarget
    status: str
    result: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class MultiTargetStageResult:
    stage: str
    total_targets: int
    succeeded_targets: int
    failed_targets: int
    target_results: list[TargetStageResult] = field(default_factory=list)


def stage_aggregate_summary_lines(agg: MultiTargetStageResult) -> list[str]:
    """Return the standard header lines for an end-of-stage aggregate summary.

    Each target is one (dataset, well) pair, so wells_succeeded mirrors
    succeeded_targets; datasets_succeeded counts unique dataset_indices with
    at least one ok target. Callers emit via print() or LOGGER.info() as
    appropriate.

    Note: in MPI mode each rank reaches this helper with its own partition
    slice (no gather across ranks). That predates this helper and applies
    equally to the per-target lines callers append after these headers.
    """
    ok_datasets = {item.target.dataset_index for item in agg.target_results if item.status == "ok"}
    all_datasets = {item.target.dataset_index for item in agg.target_results}
    return [
        f"stage: {agg.stage}",
        f"targets_total: {agg.total_targets}",
        f"targets_succeeded: {agg.succeeded_targets}",
        f"targets_failed: {agg.failed_targets}",
        f"datasets_succeeded: {len(ok_datasets)}/{len(all_datasets)}",
        f"wells_succeeded: {agg.succeeded_targets}/{agg.total_targets}",
    ]


def stage_aggregate_exit_code(agg: MultiTargetStageResult) -> int:
    """Compute a CLI exit code from a multi-target aggregate.

    Per `trackers/issues.md` §"Stage exits 0 when all targets fail — breaks
    `afterok` chains": when `succeeded_targets == 0 and total_targets > 0`,
    the stage produced no useful output and downstream `afterok` chains
    should NOT proceed. Return non-zero (2) so `sbatch --dependency=afterok`
    skips the dependent job. The 0-targets case (e.g. phase plan empty or
    YAML disabled all phases) is still "ok" — nothing to fail at.

    Return codes:
      0 — succeeded_targets > 0, OR total_targets == 0 (nothing to do).
      2 — total_targets > 0 AND succeeded_targets == 0 (all targets failed).

    Note: in MPI mode each rank computes this against its own partition
    slice (no gather across ranks; see `stage_aggregate_summary_lines`).
    `srun` propagates `max(rank_exit_codes)` to slurm. So the exit code is
    conservative-asymmetric: if any single rank's partition fully fails,
    the whole stage reports 2 even when other ranks succeeded. In the
    common all-targets-fail case (the bug this fix was for — systemic
    environment / code failures hit every rank), every rank returns 2 and
    srun exits 2. The asymmetric false-positive case (1 of N ranks fully
    fails) blocks `afterok` chains correctly-from-a-data-completeness lens
    (the failed partition's outputs are missing) but could be tightened
    once the multi-rank gather lands (tracker §"Multi-rank stage summary
    shows per-rank slice only").
    """

    total = int(getattr(agg, "total_targets", 0) or 0)
    succeeded = int(getattr(agg, "succeeded_targets", 0) or 0)
    if total > 0 and succeeded == 0:
        return 2
    return 0
