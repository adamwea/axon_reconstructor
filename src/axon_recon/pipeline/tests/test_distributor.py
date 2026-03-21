from __future__ import annotations

from pathlib import Path

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
