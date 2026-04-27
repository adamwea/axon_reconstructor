from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class PhaseDescriptor:
    name: str
    runner: Callable[[], Any]
    enabled: bool = True


@dataclass(frozen=True)
class PhaseOutcome:
    name: str
    status: str
    result: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class PhaseChainResult:
    result: Any | None = None
    outcomes: tuple[PhaseOutcome, ...] = field(default_factory=tuple)


class PhaseChainError(RuntimeError):
    def __init__(self, *, phase_name: str, error: BaseException, outcomes: Sequence[PhaseOutcome]) -> None:
        self.phase_name = str(phase_name)
        self.original_error = error
        self.outcomes = tuple(outcomes)
        super().__init__(f"{self.phase_name} failed: {error}")


def run_phase_chain(
    *,
    phases: Sequence[PhaseDescriptor],
    logger: logging.Logger | None = None,
    target_label: str | None = None,
) -> PhaseChainResult:
    outcomes: list[PhaseOutcome] = []
    last_result: Any | None = None
    for phase in phases:
        if not bool(phase.enabled):
            continue
        if logger is not None:
            logger.info(
                "Phase chain start target=%s phase=%s",
                str(target_label or "unknown"),
                str(phase.name),
            )
        try:
            last_result = phase.runner()
        except Exception as exc:
            outcome = PhaseOutcome(name=str(phase.name), status="error", error=str(exc))
            outcomes.append(outcome)
            if logger is not None:
                logger.exception(
                    "Phase chain failed target=%s phase=%s",
                    str(target_label or "unknown"),
                    str(phase.name),
                )
            raise PhaseChainError(phase_name=str(phase.name), error=exc, outcomes=tuple(outcomes)) from exc
        outcomes.append(PhaseOutcome(name=str(phase.name), status="ok", result=last_result))
        if logger is not None:
            logger.info(
                "Phase chain complete target=%s phase=%s",
                str(target_label or "unknown"),
                str(phase.name),
            )
    return PhaseChainResult(result=last_result, outcomes=tuple(outcomes))
