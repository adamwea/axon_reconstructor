from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from axon_recon.pipeline.cpu_allocation import TaskSlot, current_task_slot
from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.execution.logging_context import (
    ensure_pipeline_target_in_format,
    install_pipeline_log_record_factory,
    pipeline_log_context,
)
from axon_recon.pipeline.runner import _distribute_runtime_targets


def _target(*, dataset_index: int, dataset_id: str, stream_id: str) -> ExecutionTarget:
    return ExecutionTarget(
        dataset_index=dataset_index,
        dataset_id=dataset_id,
        h5_path=Path(f"/tmp/{dataset_id}.h5"),
        stream_id=stream_id,
        mea_output_root=Path("/tmp/out"),
    )


def test_pipeline_log_context_defaults_and_overrides(caplog):
    install_pipeline_log_record_factory()
    logger = logging.getLogger("axon_recon.tests.logging_context.defaults")
    caplog.set_level(logging.INFO, logger=logger.name)

    logger.info("outside target")
    with pipeline_log_context(dataset_id="dataset-a", dataset_index=3, well="well007"):
        logger.info("inside target")

    default_record = caplog.records[-2]
    target_record = caplog.records[-1]
    assert default_record.pipeline_target == "dataset=- idx=- well=-"
    assert target_record.pipeline_dataset_id == "dataset-a"
    assert target_record.pipeline_dataset_index == "3"
    assert target_record.pipeline_well == "well007"
    assert target_record.pipeline_target == "dataset=dataset-a idx=3 well=well007"


def test_ensure_pipeline_target_in_format_inserts_before_message():
    assert ensure_pipeline_target_in_format("[%(levelname)s] %(message)s") == (
        "[%(levelname)s] [%(pipeline_target)s] %(message)s"
    )
    assert ensure_pipeline_target_in_format("%(levelname)s %(pipeline_well)s %(message)s") == (
        "%(levelname)s %(pipeline_well)s %(message)s"
    )


def test_runtime_target_distribution_sets_log_context(caplog):
    install_pipeline_log_record_factory()
    logger = logging.getLogger("axon_recon.tests.logging_context.distribution")
    caplog.set_level(logging.INFO, logger=logger.name)
    targets = [
        _target(dataset_index=0, dataset_id="dataset-a", stream_id="well001"),
        _target(dataset_index=1, dataset_id="dataset-b", stream_id="well002"),
    ]
    parallelism = StageParallelism(
        max_workers=2,
        max_stage_workers=2,
        well_workers=2,
        unit_workers=1,
    )

    def worker(target: ExecutionTarget) -> str:
        logger.info("working")
        return target.stream_id

    results = _distribute_runtime_targets(targets=targets, parallelism=parallelism, worker_fn=worker)

    assert [result.status for result in results] == ["ok", "ok"]
    contexts = {(record.pipeline_dataset_id, record.pipeline_well) for record in caplog.records}
    assert contexts == {("dataset-a", "well001"), ("dataset-b", "well002")}


def test_runtime_target_distribution_sets_task_slot_context() -> None:
    targets = [
        _target(dataset_index=0, dataset_id="dataset-a", stream_id="well001"),
        _target(dataset_index=1, dataset_id="dataset-b", stream_id="well002"),
    ]
    task_slots = (
        TaskSlot(slot_id=0, logical_cpus=(0, 1), core_ids=(0,), package_ids=(0,)),
        TaskSlot(slot_id=1, logical_cpus=(2, 3), core_ids=(1,), package_ids=(0,)),
    )
    parallelism = StageParallelism(
        max_workers=2,
        max_stage_workers=2,
        well_workers=2,
        unit_workers=1,
        task_allocation_plan=SimpleNamespace(slots=task_slots),
    )
    seen_slot_ids: list[int] = []

    def worker(target: ExecutionTarget) -> str:
        task_slot = current_task_slot()
        assert task_slot is not None
        seen_slot_ids.append(task_slot.slot_id)
        return target.stream_id

    results = _distribute_runtime_targets(targets=targets, parallelism=parallelism, worker_fn=worker)

    assert [result.status for result in results] == ["ok", "ok"]
    assert set(seen_slot_ids) == {0, 1}
    assert current_task_slot() is None
