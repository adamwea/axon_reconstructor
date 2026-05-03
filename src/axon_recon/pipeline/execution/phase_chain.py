from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Sequence

from .logging_context import current_log_context, pipeline_log_context
from ..resource_budget import current_stage_resource_budget_manager
from ..resource_usage import (
	format_phase_message,
    format_phase_resource_usage_message,
    log_phase_resource_observation_warnings,
    log_phase_resource_plan_warnings,
    start_phase_resource_monitor,
    update_phase_summary_metadata,
)


@dataclass(frozen=True)
class PhaseDescriptor:
    name: str
    runner: Callable[[], Any]
    enabled: bool = True
    resource_class: str | None = None
    pipeline_thread_count: int | None = None


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
    resource_key_context: Any | None = None,
) -> PhaseChainResult:
    outcomes: list[PhaseOutcome] = []
    last_result: Any | None = None
    from ..logging.setup import current_pipeline_logging_config, emit_pipeline_console_blank_line

    logging_config = current_pipeline_logging_config()
    resource_budget_manager = current_stage_resource_budget_manager()
    resources_config = None if resource_budget_manager is None else resource_budget_manager.resources
    resource_usage_config = None if logging_config is None else logging_config.resource_usage
    resource_usage_log_level = logging.INFO if resource_usage_config is None else int(resource_usage_config.level)
    blank_line_after_phase = bool(
        logging_config is not None and bool(logging_config.console.blank_line_after_phase)
    )
    for phase in phases:
        if not bool(phase.enabled):
            if logger is not None:
                with pipeline_log_context(phase=phase.name, resource_class=phase.resource_class):
                    logger.info(
                        "Phase chain skipped target=%s phase=%s",
                        str(target_label or "unknown"),
                        str(phase.name),
                        extra={"event": "phase_skipped"},
                    )
            continue
        with pipeline_log_context(phase=phase.name, resource_class=phase.resource_class):
            context = current_log_context()
            stage_name = context.get("stage", None)
            dataset_id = context.get("dataset_id", None)
            recording_id = context.get("recording_id", None)
            well_id = context.get("well_id", None)
            log_phase_resource_plan_warnings(
                logger=logger,
                resource_usage_config=resource_usage_config,
                resources=resources_config,
                stage_name=stage_name,
                phase_name=phase.name,
                resource_class=phase.resource_class,
                well_workers=(1 if resource_budget_manager is None else int(resource_budget_manager.well_workers)),
                planned_target_count=(
                    0 if resource_budget_manager is None else int(resource_budget_manager.planned_target_count)
                ),
            )
            phase_budget_context = (
                nullcontext()
                if resource_budget_manager is None
                else resource_budget_manager.phase_budget(
                    resource_class=phase.resource_class,
                    logger=logger,
                    phase_name=str(phase.name),
                    target_label=str(target_label or "unknown"),
                    resource_key_context=resource_key_context,
                )
            )
            with phase_budget_context:
                phase_t0 = time.perf_counter()
                resource_monitor = start_phase_resource_monitor(
                    resource_usage_config,
                    pipeline_thread_count=phase.pipeline_thread_count,
                )
                if logger is not None:
                    logger.info(
                        format_phase_message(
					action="Starting",
					stage_name=stage_name,
					phase_name=phase.name,
					dataset_id=dataset_id,
					recording_id=recording_id,
					well_id=well_id,
					resource_class=phase.resource_class,
				),
                        extra={"event": "phase_started"},
                    )
                try:
                    last_result = phase.runner()
                except Exception as exc:
                    resource_usage = None if resource_monitor is None else resource_monitor.stop()
                    outcome = PhaseOutcome(name=str(phase.name), status="error", error=str(exc))
                    outcomes.append(outcome)
                    exception_type = type(exc).__name__
                    if logger is not None:
                        logger.exception(
                            format_phase_message(
						action="Failed",
						stage_name=stage_name,
						phase_name=phase.name,
						dataset_id=dataset_id,
						recording_id=recording_id,
						well_id=well_id,
						resource_class=phase.resource_class,
						exception_type=exception_type,
					),
					extra={
						"event": "phase_failed",
						"elapsed_s": float(max(0.0, time.perf_counter() - phase_t0)),
						"exception_type": exception_type,
					},
                        )
                        if resource_usage is not None:
                            logger.log(
                                resource_usage_log_level,
							format_phase_resource_usage_message(
								stage_name=stage_name,
								phase_name=phase.name,
								dataset_id=dataset_id,
								recording_id=recording_id,
								well_id=well_id,
								resource_class=phase.resource_class,
								status="failed",
								resource_usage=resource_usage,
								exception_type=exception_type,
							),
                                extra={
                                    "event": "phase_resource_usage",
								"status": "failed",
								"exception_type": exception_type,
                                    "resource_usage": resource_usage.to_dict(),
                                },
                            )
                            log_phase_resource_observation_warnings(
                                logger=logger,
                                resource_usage_config=resource_usage_config,
                                resources=resources_config,
                                stage_name=stage_name,
                                phase_name=phase.name,
                                resource_class=phase.resource_class,
                                resource_usage=resource_usage,
                            )
                    if blank_line_after_phase:
                        emit_pipeline_console_blank_line()
                    raise PhaseChainError(phase_name=str(phase.name), error=exc, outcomes=tuple(outcomes)) from exc
                outcomes.append(PhaseOutcome(name=str(phase.name), status="ok", result=last_result))
                resource_usage = None if resource_monitor is None else resource_monitor.stop()
                update_phase_summary_metadata(
                    summary_source=last_result,
                    resource_class=phase.resource_class,
                    resource_usage=(resource_usage if bool(getattr(resource_usage_config, "write_to_phase_summary", True)) else None),
                )
                if logger is not None:
                    logger.info(
                        format_phase_message(
                            action="Finished",
                            stage_name=stage_name,
                            phase_name=phase.name,
                            dataset_id=dataset_id,
                            recording_id=recording_id,
                            well_id=well_id,
                            resource_class=phase.resource_class,
                        ),
                        extra={"event": "phase_completed", "elapsed_s": float(max(0.0, time.perf_counter() - phase_t0))},
                    )
                    if resource_usage is not None:
                        logger.log(
                            resource_usage_log_level,
                            format_phase_resource_usage_message(
                                stage_name=stage_name,
                                phase_name=phase.name,
                                dataset_id=dataset_id,
                                recording_id=recording_id,
                                well_id=well_id,
                                resource_class=phase.resource_class,
                                status="success",
                                resource_usage=resource_usage,
                            ),
                            extra={
                                "event": "phase_resource_usage",
                                "status": "success",
                                "resource_usage": resource_usage.to_dict(),
                            },
                        )
                        log_phase_resource_observation_warnings(
                            logger=logger,
                            resource_usage_config=resource_usage_config,
                            resources=resources_config,
                            stage_name=stage_name,
                            phase_name=phase.name,
                            resource_class=phase.resource_class,
                            resource_usage=resource_usage,
                        )
                if blank_line_after_phase:
                    emit_pipeline_console_blank_line()
    return PhaseChainResult(result=last_result, outcomes=tuple(outcomes))
