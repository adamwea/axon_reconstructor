from __future__ import annotations

import concurrent.futures
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable

from .context import ExecutionTarget
from .read_groups import target_read_group_key
from .results import TargetStageResult

if TYPE_CHECKING:
    from ..cpu_allocation import TaskSlot


TargetCompleteCallback = Callable[[TargetStageResult], None]


def _notify_target_complete(callback: TargetCompleteCallback | None, result: TargetStageResult) -> None:
    if callback is None:
        return
    try:
        callback(result)
    except Exception:
        return


def _target_stage_result_from_future(
    *,
    future: concurrent.futures.Future[Any],
    target: ExecutionTarget,
) -> TargetStageResult:
    try:
        result = future.result()
        return TargetStageResult(target=target, status="ok", result=result, error=None)
    except Exception as exc:
        return TargetStageResult(target=target, status="error", result=None, error=str(exc))


def _run_worker_with_task_slot(
    *,
    worker_fn: Callable[[ExecutionTarget], Any],
    target: ExecutionTarget,
    task_slot: TaskSlot | None,
) -> Any:
    from ..cpu_allocation import task_slot_context

    with task_slot_context(task_slot):
        return worker_fn(target)


def _coerce_positive_optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _distribute_targets_with_read_group_cap(
    *,
    targets: list[ExecutionTarget],
    well_workers: int,
    max_simultaneous_well_reads_per_dataset: int,
    worker_fn: Callable[[ExecutionTarget], Any],
    on_target_complete: TargetCompleteCallback | None = None,
    task_slots: tuple[TaskSlot, ...] = (),
) -> list[TargetStageResult]:
    group_order: list[Any] = []
    queues: dict[Any, deque[ExecutionTarget]] = {}
    for target in targets:
        key = target_read_group_key(target)
        if key not in queues:
            queues[key] = deque()
            group_order.append(key)
        queues[key].append(target)

    active_counts: dict[Any, int] = defaultdict(int)
    max_workers = int(max(1, int(well_workers)))
    available_slots: deque[TaskSlot] | None = None
    if task_slots:
        max_workers = min(max_workers, len(task_slots))
        available_slots = deque(task_slots[:max_workers])
    read_cap = int(max(1, int(max_simultaneous_well_reads_per_dataset)))
    remaining = len(targets)
    next_group_index = 0
    futures: dict[concurrent.futures.Future[Any], tuple[ExecutionTarget, Any, TaskSlot | None]] = {}
    out: list[TargetStageResult] = []

    def _submit_available(pool: concurrent.futures.ThreadPoolExecutor) -> None:
        nonlocal next_group_index, remaining
        while len(futures) < max_workers and remaining > 0:
            if available_slots is not None and not available_slots:
                return
            submitted = False
            for _ in range(len(group_order)):
                key = group_order[next_group_index % len(group_order)]
                next_group_index = (next_group_index + 1) % len(group_order)
                if not queues[key]:
                    continue
                if int(active_counts[key]) >= read_cap:
                    continue
                target = queues[key].popleft()
                remaining -= 1
                active_counts[key] += 1
                task_slot = available_slots.popleft() if available_slots is not None else None
                futures[
                    pool.submit(
                        _run_worker_with_task_slot,
                        worker_fn=worker_fn,
                        target=target,
                        task_slot=task_slot,
                    )
                ] = (target, key, task_slot)
                submitted = True
                break
            if not submitted:
                return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        _submit_available(pool)
        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                target, key, task_slot = futures.pop(future)
                active_counts[key] = max(0, int(active_counts[key]) - 1)
                if available_slots is not None and task_slot is not None:
                    available_slots.append(task_slot)
                result = _target_stage_result_from_future(future=future, target=target)
                out.append(result)
                _notify_target_complete(on_target_complete, result)
            _submit_available(pool)

    out.sort(key=lambda item: (item.target.dataset_index, item.target.stream_id))
    return out


def distribute_targets(
    *,
    targets: list[ExecutionTarget],
    well_workers: int,
    worker_fn: Callable[[ExecutionTarget], Any],
    max_simultaneous_well_reads_per_dataset: int | None = None,
    on_target_complete: TargetCompleteCallback | None = None,
    task_slots: tuple[TaskSlot, ...] | list[TaskSlot] | None = None,
) -> list[TargetStageResult]:
    resolved_task_slots = tuple(task_slots or ())
    effective_well_workers = int(max(1, int(well_workers)))
    if resolved_task_slots:
        effective_well_workers = min(effective_well_workers, len(resolved_task_slots))

    if effective_well_workers <= 1:
        out: list[TargetStageResult] = []
        task_slot = resolved_task_slots[0] if resolved_task_slots else None
        for target in targets:
            try:
                result = _run_worker_with_task_slot(
                    worker_fn=worker_fn,
                    target=target,
                    task_slot=task_slot,
                )
                target_result = TargetStageResult(target=target, status="ok", result=result, error=None)
            except Exception as exc:
                target_result = TargetStageResult(target=target, status="error", result=None, error=str(exc))
            out.append(target_result)
            _notify_target_complete(on_target_complete, target_result)
        return out

    read_cap = _coerce_positive_optional_int(max_simultaneous_well_reads_per_dataset)
    if read_cap is not None:
        return _distribute_targets_with_read_group_cap(
            targets=targets,
            well_workers=effective_well_workers,
            max_simultaneous_well_reads_per_dataset=int(read_cap),
            worker_fn=worker_fn,
            on_target_complete=on_target_complete,
            task_slots=resolved_task_slots,
        )

    if resolved_task_slots:
        pending = deque(targets)
        available_slots = deque(resolved_task_slots[:effective_well_workers])
        futures: dict[concurrent.futures.Future[Any], tuple[ExecutionTarget, TaskSlot]] = {}
        out: list[TargetStageResult] = []

        def _submit_available(pool: concurrent.futures.ThreadPoolExecutor) -> None:
            while pending and available_slots and len(futures) < effective_well_workers:
                target = pending.popleft()
                task_slot = available_slots.popleft()
                futures[
                    pool.submit(
                        _run_worker_with_task_slot,
                        worker_fn=worker_fn,
                        target=target,
                        task_slot=task_slot,
                    )
                ] = (target, task_slot)

        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_well_workers) as pool:
            _submit_available(pool)
            while futures:
                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    target, task_slot = futures.pop(future)
                    available_slots.append(task_slot)
                    result = _target_stage_result_from_future(future=future, target=target)
                    out.append(result)
                    _notify_target_complete(on_target_complete, result)
                _submit_available(pool)

        out.sort(key=lambda item: (item.target.dataset_index, item.target.stream_id))
        return out

    futures: dict[concurrent.futures.Future[Any], ExecutionTarget] = {}
    out: list[TargetStageResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_well_workers) as pool:
        for target in targets:
            fut = pool.submit(worker_fn, target)
            futures[fut] = target

        for fut in concurrent.futures.as_completed(futures):
            target = futures[fut]
            result = _target_stage_result_from_future(future=fut, target=target)
            out.append(result)
            _notify_target_complete(on_target_complete, result)

    out.sort(key=lambda item: (item.target.dataset_index, item.target.stream_id))
    return out
