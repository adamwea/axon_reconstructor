from __future__ import annotations

import concurrent.futures
from typing import Any, Callable

from .context import ExecutionTarget
from .results import TargetStageResult


def distribute_targets(
    *,
    targets: list[ExecutionTarget],
    well_workers: int,
    worker_fn: Callable[[ExecutionTarget], Any],
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

    futures: dict[concurrent.futures.Future[Any], ExecutionTarget] = {}
    out: list[TargetStageResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(max(1, int(well_workers)))) as pool:
        for target in targets:
            fut = pool.submit(worker_fn, target)
            futures[fut] = target

        for fut in concurrent.futures.as_completed(futures):
            target = futures[fut]
            try:
                result = fut.result()
                out.append(TargetStageResult(target=target, status="ok", result=result, error=None))
            except Exception as exc:
                out.append(TargetStageResult(target=target, status="error", result=None, error=str(exc)))

    out.sort(key=lambda item: (item.target.dataset_index, item.target.stream_id))
    return out
