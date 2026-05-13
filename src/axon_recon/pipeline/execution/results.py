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
