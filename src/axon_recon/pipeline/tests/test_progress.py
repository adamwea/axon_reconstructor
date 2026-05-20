from __future__ import annotations

from contextlib import contextmanager
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

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


def test_pipeline_progress_skips_tqdm_logging_redirect_for_rich_handler(monkeypatch) -> None:
    entered: list[str] = []
    writes: list[tuple[str, object | None]] = []

    class RichHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            return None

    class _Bar:
        fp = object()

        def update(self, amount: int) -> None:
            return None

        def close(self) -> None:
            return None

    class _FakeTqdm:
        def __call__(self, *args: object, **kwargs: object) -> _Bar:
            return _Bar()

        def write(self, message: str, file: object | None = None) -> None:
            writes.append((message, file))

    @contextmanager
    def _fake_redirect() -> Iterator[None]:
        entered.append("redirect")
        yield

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    root.handlers = [RichHandler()]
    bar_tqdm = _FakeTqdm()
    monkeypatch.setattr(progress_module, "_tqdm", bar_tqdm)
    monkeypatch.setattr(progress_module, "_logging_redirect_tqdm", _fake_redirect)
    progress = PipelineProgress(ProgressSpec(label="test", total=1, unit="item", enabled=True))

    try:
        with progress:
            pass
    finally:
        root.handlers = original_handlers

    assert entered == []
    # `_Bar.fp` is a class-level sentinel that every `_Bar()` instance shares;
    # `progress._bar` is None after `__exit__` (execution/progress.py:94), so
    # we have to reach for the class attr to get the same object the
    # `_FakeTqdm.write(..., file=…)` call captured.
    assert writes == [("", _Bar.fp)]


def test_runtime_distribution_advances_target_progress() -> None:
    targets = [_target(0, "well001"), _target(1, "well002")]
    parallelism = StageParallelism(
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