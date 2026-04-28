from __future__ import annotations

import concurrent.futures
from collections import defaultdict, deque
from typing import Any, Callable

from .context import ExecutionTarget
from .read_groups import target_read_group_key
from .results import TargetStageResult


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
    read_cap = int(max(1, int(max_simultaneous_well_reads_per_dataset)))
    remaining = len(targets)
    next_group_index = 0
    futures: dict[concurrent.futures.Future[Any], tuple[ExecutionTarget, Any]] = {}
    out: list[TargetStageResult] = []

    def _submit_available(pool: concurrent.futures.ThreadPoolExecutor) -> None:
        nonlocal next_group_index, remaining
        while len(futures) < max_workers and remaining > 0:
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
                futures[pool.submit(worker_fn, target)] = (target, key)
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
                target, key = futures.pop(future)
                active_counts[key] = max(0, int(active_counts[key]) - 1)
                out.append(_target_stage_result_from_future(future=future, target=target))
            _submit_available(pool)

    out.sort(key=lambda item: (item.target.dataset_index, item.target.stream_id))
    return out


def distribute_targets(
    *,
    targets: list[ExecutionTarget],
    well_workers: int,
    worker_fn: Callable[[ExecutionTarget], Any],
    max_simultaneous_well_reads_per_dataset: int | None = None,
) -> list[TargetStageResult]:
    if int(max(1, int(well_workers))) <= 1:
        out: list[TargetStageResult] = []
        for target in targets:
            try:
                result = worker_fn(target)
                out.append(TargetStageResult(target=target, status="ok", result=result, error=None))
            except Exception as exc:
                out.append(TargetStageResult(target=target, status="error", result=None, error=str(exc)))
        return out

    read_cap = _coerce_positive_optional_int(max_simultaneous_well_reads_per_dataset)
    if read_cap is not None:
        return _distribute_targets_with_read_group_cap(
            targets=targets,
            well_workers=well_workers,
            max_simultaneous_well_reads_per_dataset=int(read_cap),
            worker_fn=worker_fn,
        )

    futures: dict[concurrent.futures.Future[Any], ExecutionTarget] = {}
    out: list[TargetStageResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(max(1, int(well_workers)))) as pool:
        for target in targets:
            fut = pool.submit(worker_fn, target)
            futures[fut] = target

        for fut in concurrent.futures.as_completed(futures):
            target = futures[fut]
            out.append(_target_stage_result_from_future(future=fut, target=target))

    out.sort(key=lambda item: (item.target.dataset_index, item.target.stream_id))
    return out
