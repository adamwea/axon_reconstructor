from __future__ import annotations

from pathlib import Path
import threading

from axon_recon.pipeline.execution.context import ExecutionTarget
from axon_recon.pipeline.execution.distributor import distribute_targets


def _target(i: int) -> ExecutionTarget:
    return ExecutionTarget(
        dataset_index=i,
        dataset_id=f"dataset_{i}",
        h5_path=Path(f"/tmp/ds{i}.h5"),
        stream_id="well001",
        mea_output_root=Path("/tmp/out"),
    )


def _target_for(dataset_index: int, stream_id: str) -> ExecutionTarget:
    return ExecutionTarget(
        dataset_index=dataset_index,
        dataset_id=f"dataset_{dataset_index}",
        h5_path=Path(f"/tmp/ds{dataset_index}.h5"),
        stream_id=stream_id,
        mea_output_root=Path("/tmp/out"),
    )


def _blocked_distribution_snapshot(
    *,
    targets: list[ExecutionTarget],
    well_workers: int,
    read_cap: int,
    expected_initial_starts: int,
) -> tuple[list[tuple[int, str]], dict[Path, int], list[str]]:
    condition = threading.Condition()
    release_workers = threading.Event()
    active_counts: dict[Path, int] = {}
    max_counts: dict[Path, int] = {}
    started: list[tuple[int, str]] = []
    results: list[str] = []
    errors: list[BaseException] = []

    def worker(target: ExecutionTarget) -> str:
        with condition:
            active_counts[target.h5_path] = active_counts.get(target.h5_path, 0) + 1
            max_counts[target.h5_path] = max(max_counts.get(target.h5_path, 0), active_counts[target.h5_path])
            started.append((target.dataset_index, target.stream_id))
            condition.notify_all()
        try:
            release_workers.wait(timeout=5)
            return f"ok-{target.dataset_index}-{target.stream_id}"
        finally:
            with condition:
                active_counts[target.h5_path] = max(0, active_counts.get(target.h5_path, 0) - 1)
                condition.notify_all()

    def run_distribution() -> None:
        try:
            output = distribute_targets(
                targets=targets,
                well_workers=well_workers,
                worker_fn=worker,
                max_simultaneous_well_reads_per_dataset=read_cap,
            )
            results.extend(str(item.result) for item in output)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_distribution)
    thread.start()
    with condition:
        assert condition.wait_for(lambda: len(started) >= expected_initial_starts, timeout=5)
        condition.wait(timeout=0.1)
        started_snapshot = list(started)
        max_snapshot = dict(max_counts)
    release_workers.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    return started_snapshot, max_snapshot, results


def test_distributor_serial_continue_on_error() -> None:
    targets = [_target(0), _target(1)]

    def worker(t: ExecutionTarget) -> str:
        if t.dataset_index == 0:
            raise RuntimeError("boom")
        return "ok"

    out = distribute_targets(targets=targets, well_workers=1, worker_fn=worker)
    assert len(out) == 2
    assert out[0].status == "error"
    assert out[1].status == "ok"


def test_distributor_threaded_continue_on_error() -> None:
    targets = [_target(0), _target(1), _target(2)]

    def worker(t: ExecutionTarget) -> str:
        if t.dataset_index == 1:
            raise RuntimeError("boom")
        return f"ok-{t.dataset_index}"

    out = distribute_targets(targets=targets, well_workers=2, worker_fn=worker)
    assert len(out) == 3
    errors = [item for item in out if item.status == "error"]
    oks = [item for item in out if item.status == "ok"]
    assert len(errors) == 1
    assert len(oks) == 2


def test_distributor_notifies_target_completion() -> None:
    targets = [_target(0), _target(1)]
    completed: list[tuple[int, str]] = []

    def worker(target: ExecutionTarget) -> str:
        return f"ok-{target.dataset_index}"

    out = distribute_targets(
        targets=targets,
        well_workers=2,
        worker_fn=worker,
        on_target_complete=lambda result: completed.append((result.target.dataset_index, result.status)),
    )

    assert [item.status for item in out] == ["ok", "ok"]
    assert sorted(completed) == [(0, "ok"), (1, "ok")]


def test_distributor_read_cap_starts_one_well_per_h5_before_reusing_h5() -> None:
    targets = [
        _target_for(0, "well000"),
        _target_for(0, "well001"),
        _target_for(1, "well000"),
        _target_for(1, "well001"),
        _target_for(2, "well000"),
        _target_for(2, "well001"),
    ]

    started, max_counts, results = _blocked_distribution_snapshot(
        targets=targets,
        well_workers=3,
        read_cap=1,
        expected_initial_starts=3,
    )

    assert set(started[:3]) == {(0, "well000"), (1, "well000"), (2, "well000")}
    assert max(max_counts.values()) == 1
    assert len(results) == len(targets)


def test_distributor_read_cap_limits_single_h5_concurrency() -> None:
    targets = [_target_for(0, f"well00{index}") for index in range(5)]

    started, max_counts, results = _blocked_distribution_snapshot(
        targets=targets,
        well_workers=4,
        read_cap=2,
        expected_initial_starts=2,
    )

    assert len(started) == 2
    assert max(max_counts.values()) == 2
    assert len(results) == len(targets)
