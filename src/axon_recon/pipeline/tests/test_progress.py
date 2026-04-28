from __future__ import annotations

import io
import logging
from pathlib import Path
from types import SimpleNamespace

import axon_recon.pipeline.execution.progress as progress_module
from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.execution.progress import (
    PipelineProgress,
    PipelineProgressStreamHandler,
    ProgressSpec,
    add_current_progress_total,
    advance_current_progress,
    pipeline_progress_context,
)
from axon_recon.pipeline.runner import _distribute_runtime_targets


def _target(dataset_index: int, stream_id: str) -> ExecutionTarget:
    return ExecutionTarget(
        dataset_index=dataset_index,
        dataset_id=f"dataset-{dataset_index}",
        h5_path=Path(f"/tmp/dataset-{dataset_index}.h5"),
        stream_id=stream_id,
        mea_output_root=Path("/tmp/out"),
    )


def test_pipeline_progress_context_tracks_dynamic_totals() -> None:
    progress = PipelineProgress(ProgressSpec(label="test units", total=1, unit="unit", enabled=False))

    with progress, pipeline_progress_context(progress):
        add_current_progress_total(2)
        advance_current_progress()
        advance_current_progress(2)

    assert progress.total == 3
    assert progress.completed == 3


def test_progress_stream_handler_writes_through_tqdm(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        progress_module,
        "_tqdm",
        SimpleNamespace(write=lambda message, file=None: messages.append(str(message))),
    )
    progress = PipelineProgress(ProgressSpec(label="test", total=1, unit="unit", enabled=False))
    logger = logging.Logger("axon_recon.tests.progress_handler")
    stream = io.StringIO()
    handler = PipelineProgressStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    with pipeline_progress_context(progress):
        logger.info("hello")

    assert messages == ["INFO: hello"]
    assert stream.getvalue() == ""


def test_runtime_distribution_advances_target_progress() -> None:
    targets = [_target(0, "well001"), _target(1, "well002")]
    parallelism = StageParallelism(
        max_workers=2,
        max_stage_workers=2,
        well_workers=2,
        unit_workers=1,
    )
    progress = PipelineProgress(ProgressSpec(label="test wells", total=len(targets), unit="well", enabled=False))

    results = _distribute_runtime_targets(
        targets=targets,
        parallelism=parallelism,
        worker_fn=lambda target: target.stream_id,
        progress=progress,
        advance_progress_on_target_complete=True,
    )

    assert [result.status for result in results] == ["ok", "ok"]
    assert progress.total == 2
    assert progress.completed == 2