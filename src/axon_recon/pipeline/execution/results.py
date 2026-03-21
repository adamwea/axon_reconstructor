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
