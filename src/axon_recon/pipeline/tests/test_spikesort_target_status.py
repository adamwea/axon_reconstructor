from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.execution.results import MultiTargetStageResult, TargetStageResult
from axon_recon.runtime_config import RuntimeConfig
from axon_recon.pipeline.runner import (
    run_spikesort_bombcell_label_from_runtime,
    run_spikesort_cleanup_concat_binary_from_runtime,
    run_spikesort_from_runtime,
    run_spikesort_merge_from_runtime,
    run_spikesort_sort_from_runtime,
    run_spikesort_summarize_sort_from_runtime,
)
from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs
from axon_recon.pipeline.stages.spikesort.orchestrators.merge_slay import (
    run_spikesort_merge_slay_from_runtime,
)
from axon_recon.pipeline.stages.spikesort.orchestrators import bombcell_label as bombcell_label_orchestrator
from axon_recon.pipeline.stages.spikesort.orchestrators import cleanup_concat_binary as cleanup_concat_binary_orchestrator
from axon_recon.pipeline.stages.spikesort.orchestrators import sort as sort_orchestrator
from axon_recon.pipeline.stages.spikesort.models.results import (
    SpikesortBombcellResult,
    SpikesortMergeResult,
    SpikesortResult,
)


def test_run_spikesort_sort_from_runtime_wraps_direct_phase_in_resource_chain(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )
    result = SpikesortResult(
        well_out_dir=tmp_path / "well_out",
        spikesort_out_dir=tmp_path / "well_out" / "spikesort_outputs",
        summary_json=tmp_path / "sort_summary.json",
        outputs={},
    )
    seen_descriptors: list[object] = []

    class _DummyBundle:
        runtime_config = RuntimeConfig(
            {
                "resources": {
                    "active_profile": "test_profile",
                    "profiles": {"test_profile": {"cpu_cores": 8, "ram_gb": 32}},
                    "phase_resource_classes": {"sort_class": {"cpu_cores": 4, "ram_gb": 12}},
                }
            }
        )
        data_config = RuntimeConfig({"publish_outputs": False})

    def _fake_run_phase_chain(*, phases, logger, target_label, resource_key_context):
        _ = logger, target_label, resource_key_context
        descriptor = list(phases)[0]
        seen_descriptors.append(descriptor)
        return SimpleNamespace(result=descriptor.runner(), outcomes=())

    def _fake_distribute_targets(*, targets, worker_fn, **kwargs):
        return [TargetStageResult(target=item, status="ok", result=worker_fn(item)) for item in targets]

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", lambda *, config_path: _DummyBundle())
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", lambda *, bundle, **kwargs: [target])
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_stage_parallelism",
        lambda **kwargs: StageParallelism(max_workers=4, max_stage_workers=4, well_workers=1, unit_workers=4),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "parse_spikesort_stage_config",
        lambda **kwargs: SimpleNamespace(
            debug_limit_datasets=None,
            debug_limit_wells=None,
            debug_limit_wells_per_dataset=None,
            sort_debug_mode_enabled=False,
            sort_resource_class="sort_class",
            output_rel_root="spikesort_outputs",
            force_restart=False,
            force_replot=False,
        ),
    )
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(pipeline_runner, "run_spikesort", lambda inputs: result)
    monkeypatch.setattr(pipeline_runner, "run_phase_chain", _fake_run_phase_chain)
    monkeypatch.setattr(pipeline_runner, "distribute_targets", _fake_distribute_targets)

    agg = run_spikesort_sort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.succeeded_targets == 1
    assert seen_descriptors
    descriptor = seen_descriptors[0]
    assert descriptor.name == "sort"
    assert descriptor.resource_class == "sort_class"
    assert descriptor.pipeline_thread_count == 4


@pytest.mark.parametrize(
    ("orchestrator", "entry_name", "runtime_name"),
    [
        (sort_orchestrator, "_run_sort_from_args", "run_spikesort_sort_from_runtime"),
        (bombcell_label_orchestrator, "_run_bombcell_label_from_args", "run_spikesort_bombcell_label_from_runtime"),
        (
            cleanup_concat_binary_orchestrator,
            "_run_cleanup_concat_binary_from_args",
            "run_spikesort_cleanup_concat_binary_from_runtime",
        ),
    ],
)
def test_spikesort_direct_phase_from_args_forwards_debug_limits(
    monkeypatch,
    tmp_path: Path,
    orchestrator,
    entry_name: str,
    runtime_name: str,
) -> None:
    seen: dict[str, object] = {}

    def _fake_runtime(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stage="spikesort.test", total_targets=0, succeeded_targets=0, failed_targets=0, target_results=[])

    monkeypatch.setattr(orchestrator, runtime_name, _fake_runtime)
    if orchestrator is sort_orchestrator:
        monkeypatch.setattr(orchestrator, "_debug_outputs_enabled_for_config", lambda config_path: False)

    rc = getattr(orchestrator, entry_name)(
        SimpleNamespace(
            config=str(tmp_path / "runtime.yml"),
            limit_segments=2,
            limit_datasets=3,
            limit_wells_per_dataset=1,
            force_restart=True,
            force_replot=False,
        )
    )

    assert rc == 0
    assert seen["limit_segments_override"] == 2
    assert seen["limit_datasets_override"] == 3
    assert seen["limit_wells_per_dataset_override"] == 1
    assert seen["force_restart_override"] is True


def test_run_spikesort_from_runtime_marks_target_ok(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = SpikesortInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_spikesort_inputs_for_target(*, target, stage_config, unit_workers: int):
        return dummy_inputs

    def _fake_run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
        return SpikesortResult(
            well_out_dir=tmp_path / "well_out",
            spikesort_out_dir=tmp_path / "spikesort_out",
            summary_json=tmp_path / "spikesort_summary.json",
            outputs={"sorter_output_dir": "sorter_output"},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", _fake_build_spikesort_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_spikesort", _fake_run_spikesort)

    agg = run_spikesort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].status == "ok"
    assert agg.target_results[0].result is not None


def test_run_spikesort_from_runtime_logs_stage_topology(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = SpikesortInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", lambda *, config_path: _DummyBundle())
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", lambda *, bundle: [target])
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_stage_parallelism",
        lambda *, bundle, stage_name: StageParallelism(max_workers=8, max_stage_workers=8, well_workers=2, unit_workers=4),
    )
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", lambda **kwargs: SimpleNamespace(debug_limit_wells=None))
    monkeypatch.setattr(
        pipeline_runner,
        "build_spikesort_inputs_for_target",
        lambda *, target, stage_config, unit_workers: dummy_inputs,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "run_spikesort",
        lambda inputs: SpikesortResult(
            well_out_dir=tmp_path / "well_out",
            spikesort_out_dir=tmp_path / "spikesort_out",
            summary_json=tmp_path / "spikesort_summary.json",
            outputs={},
        ),
    )

    with caplog.at_level(logging.INFO):
        run_spikesort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    messages = [record.getMessage() for record in caplog.records]
    assert "Starting stage: spikesort" in messages
    assert "Execution topology: stage_global_order=true, well_local_phase_sequence=true" in messages
    assert "Selected wells: 1" in messages
    assert "well_workers=2 max_stage_workers=8" in messages


def test_run_spikesort_from_runtime_marks_target_error(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = SpikesortInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_spikesort_inputs_for_target(*, target, stage_config, unit_workers: int):
        return dummy_inputs

    def _fake_run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
        raise RuntimeError("spikesort failed")

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", _fake_build_spikesort_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_spikesort", _fake_run_spikesort)

    agg = run_spikesort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 0
    assert agg.failed_targets == 1
    assert agg.target_results[0].status == "error"
    assert "spikesort failed" in str(agg.target_results[0].error)


def test_run_spikesort_from_runtime_applies_segment_limit_to_bootstrap_phase(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = RuntimeConfig(
            {
                "stages": {
                    "spikesort": {
                        "phases": {
                            "bootstrap_concat_binary": {"enabled": True},
                            "sort": {"enabled": False},
                            "cleanup_concat_binary": {"enabled": False},
                        }
                    }
                }
            }
        )
        data_config = object()

    seen: dict[str, object] = {}

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_distribute_targets(*, targets, well_workers: int, worker_fn, **kwargs):
        return [TargetStageResult(target=item, status="ok", result=worker_fn(item)) for item in targets]

    def _fake_bootstrap_target(*, target, stage_config, unit_workers: int):
        seen["debug_limit_segments_per_well"] = getattr(stage_config, "debug_limit_segments_per_well", None)
        seen["bootstrap_concat_binary_debug_limit_segments_per_well"] = getattr(
            stage_config,
            "bootstrap_concat_binary_debug_limit_segments_per_well",
            None,
        )
        return SpikesortResult(
            well_out_dir=tmp_path / "well_out",
            spikesort_out_dir=tmp_path / "well_out" / "spikesort_outputs",
            summary_json=tmp_path / "bootstrap.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "distribute_targets", _fake_distribute_targets)
    monkeypatch.setattr(pipeline_runner, "_run_spikesort_bootstrap_concat_binary_target", _fake_bootstrap_target)

    agg = run_spikesort_from_runtime(config_path=str(tmp_path / "runtime.yml"), limit_segments_override=2)

    assert agg.total_targets == 1
    assert seen == {
        "debug_limit_segments_per_well": 2,
        "bootstrap_concat_binary_debug_limit_segments_per_well": 2,
    }


def test_run_spikesort_from_runtime_applies_debug_well_limit(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target_a = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_a.h5",
        h5_path=tmp_path / "test_a.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )
    target_b = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_b.h5",
        h5_path=tmp_path / "test_b.h5",
        stream_id="well002",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(**kwargs):
        return [target_a, target_b]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=1)

    def _fake_build_spikesort_inputs_for_target(*, target, stage_config, unit_workers: int):
        return SpikesortInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
        )

    def _fake_run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
        return SpikesortResult(
            well_out_dir=tmp_path / f"well_out_{inputs.stream_id}",
            spikesort_out_dir=tmp_path / f"spikesort_out_{inputs.stream_id}",
            summary_json=tmp_path / f"spikesort_summary_{inputs.stream_id}.json",
            outputs={"sorter_output_dir": f"sorter_output_{inputs.stream_id}"},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", _fake_build_spikesort_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_spikesort", _fake_run_spikesort)

    agg = run_spikesort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.stream_id == "well001"


def test_run_spikesort_from_runtime_applies_global_debug_dataset_and_well_limits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    targets = [
        ExecutionTarget(
            dataset_index=0,
            dataset_id="dataset_000:test_a.h5",
            h5_path=tmp_path / "test_a.h5",
            stream_id="well001",
            mea_output_root=tmp_path,
        ),
        ExecutionTarget(
            dataset_index=0,
            dataset_id="dataset_000:test_a.h5",
            h5_path=tmp_path / "test_a.h5",
            stream_id="well002",
            mea_output_root=tmp_path,
        ),
        ExecutionTarget(
            dataset_index=0,
            dataset_id="dataset_000:test_a.h5",
            h5_path=tmp_path / "test_a.h5",
            stream_id="well003",
            mea_output_root=tmp_path,
        ),
        ExecutionTarget(
            dataset_index=1,
            dataset_id="dataset_001:test_b.h5",
            h5_path=tmp_path / "test_b.h5",
            stream_id="well001",
            mea_output_root=tmp_path,
        ),
    ]

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    built_targets: list[tuple[int, str]] = []
    select_kwargs: dict[str, object] = {}

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(**kwargs):
        select_kwargs.update(kwargs)
        return list(targets)

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str, target_count: int | None = None):
        _ = bundle, stage_name, target_count
        return StageParallelism(max_workers=2, max_stage_workers=2, well_workers=2, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_datasets=1,
            debug_limit_wells=2,
            output_rel_root="spikesort_outputs",
            phase_sequence=("sort",),
            sort_enabled=True,
            sort_debug_mode_enabled=False,
        )

    def _fake_build_spikesort_inputs_for_target(*, target, stage_config, unit_workers: int):
        _ = stage_config, unit_workers
        built_targets.append((int(target.dataset_index), str(target.stream_id)))
        return SpikesortInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
        )

    def _fake_run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
        return SpikesortResult(
            well_out_dir=tmp_path / f"well_out_{inputs.stream_id}",
            spikesort_out_dir=tmp_path / f"spikesort_out_{inputs.stream_id}",
            summary_json=tmp_path / f"spikesort_summary_{inputs.stream_id}.json",
            outputs={"sorter_output_dir": f"sorter_output_{inputs.stream_id}"},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", _fake_build_spikesort_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_spikesort", _fake_run_spikesort)

    agg = run_spikesort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 2
    assert agg.succeeded_targets == 2
    assert agg.failed_targets == 0
    assert built_targets == [(0, "well001"), (0, "well002")]
    assert select_kwargs["limit_datasets"] == 1
    assert select_kwargs["limit_wells"] == 2


def test_spikesort_debug_limits_select_first_wells_per_dataset(tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    targets = [
        ExecutionTarget(
            dataset_index=dataset_index,
            dataset_id=f"dataset_{dataset_index:03d}:test.h5",
            h5_path=tmp_path / f"test_{dataset_index}.h5",
            stream_id=f"well{well_index:03d}",
            mea_output_root=tmp_path,
        )
        for dataset_index in range(3)
        for well_index in range(1, 4)
    ]

    limited = pipeline_runner._apply_spikesort_debug_target_limits(
        stage_name="spikesort",
        targets=targets,
        limit_datasets=2,
        limit_wells=None,
        limit_wells_per_dataset=2,
    )

    assert [(int(target.dataset_index), str(target.stream_id)) for target in limited] == [
        (0, "well001"),
        (0, "well002"),
        (1, "well001"),
        (1, "well002"),
    ]


def test_run_spikesort_from_runtime_runs_enabled_phases_in_lifecycle_order(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    calls: list[str] = []
    n_jobs_seen: list[int | None] = []
    distribute_calls: list[tuple[list[ExecutionTarget], int]] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=12, max_stage_workers=12, well_workers=2, unit_workers=6)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            output_rel_root="spikesort_outputs",
            bootstrap_concat_binary_enabled=True,
            sort_enabled=True,
            summarize_sort_enabled=False,
            bombcell_label_enabled=True,
            merge_slay_enabled=True,
            merge_si_auto_enabled=False,
            merge_unitmatch_enabled=False,
            cleanup_concat_binary_enabled=True,
        )

    def _fake_target_phase_runner(stage_name: str):
        def _runner(**kwargs):
            calls.append(stage_name)
            n_jobs_seen.append(getattr(kwargs["stage_config"], "n_jobs", None))
            return SpikesortResult(
                well_out_dir=tmp_path / "well_out",
                spikesort_out_dir=tmp_path / "well_out" / "spikesort_outputs",
                summary_json=tmp_path / f"{stage_name.replace('.', '_')}.json",
                outputs={},
            )

        return _runner

    def _fake_distribute_targets(*, targets, well_workers: int, worker_fn, **kwargs):
        distribute_calls.append((list(targets), int(well_workers)))
        return [TargetStageResult(target=item, status="ok", result=worker_fn(item)) for item in targets]

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(
        pipeline_runner,
        "_run_spikesort_bootstrap_concat_binary_target",
        _fake_target_phase_runner("spikesort.bootstrap_concat_binary"),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_run_spikesort_sort_target",
        _fake_target_phase_runner("spikesort.sort"),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_run_spikesort_bombcell_label_target",
        _fake_target_phase_runner("spikesort.bombcell_label"),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_run_spikesort_merge_slay_target",
        _fake_target_phase_runner("spikesort.merge_SLAy"),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_run_spikesort_cleanup_concat_binary_target",
        _fake_target_phase_runner("spikesort.cleanup_concat_binary"),
    )
    monkeypatch.setattr(pipeline_runner, "distribute_targets", _fake_distribute_targets)

    with caplog.at_level(logging.INFO):
        agg = run_spikesort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    messages = [record.getMessage() for record in caplog.records]

    assert len(distribute_calls) == 1
    assert distribute_calls[0][0] == [target]
    assert calls == [
        "spikesort.bootstrap_concat_binary",
        "spikesort.sort",
        "spikesort.bombcell_label",
        "spikesort.merge_SLAy",
        "spikesort.cleanup_concat_binary",
    ]
    assert n_jobs_seen == [6, 6, 6, 6, 6]
    assert any(
        "Spikesort phase worker allocation stage=spikesort phase=sort target=0:well001 stage_workers=12 well_workers=2 n_jobs=6 n_jobs_source=derived"
        in message
        for message in messages
    )
    assert not any("unit_workers" in message for message in messages if "worker allocation" in message)
    assert agg.stage == "spikesort"
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0


def test_run_spikesort_from_runtime_gates_only_sort_phase_across_wells_via_resource_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    targets = [
        ExecutionTarget(
            dataset_index=0,
            dataset_id="dataset_000:test_a.h5",
            h5_path=tmp_path / "test_a.h5",
            stream_id="well001",
            mea_output_root=tmp_path,
        ),
        ExecutionTarget(
            dataset_index=0,
            dataset_id="dataset_000:test_b.h5",
            h5_path=tmp_path / "test_b.h5",
            stream_id="well002",
            mea_output_root=tmp_path,
        ),
    ]

    class _DummyBundle:
        runtime_config = RuntimeConfig(
            {
                "resources": {
                    "active_profile": "test_profile",
                    "profiles": {
                        "test_profile": {
                            "cpu_cores": 4,
                            "ram_gb": 16,
                            "gpu_sort_slots": 1,
                        }
                    },
                    "phase_resource_classes": {
                        "bootstrap_concat_binary": {
                            "cpu_cores": 1,
                            "ram_gb": 1,
                        },
                        "kilosort4": {
                            "cpu_cores": 1,
                            "ram_gb": 1,
                            "gpu_sort_slots": 1,
                        },
                    },
                }
            }
        )
        data_config = object()

    active_counts = {"bootstrap_concat_binary": 0, "sort": 0}
    max_active_counts = {"bootstrap_concat_binary": 0, "sort": 0}
    active_lock = threading.Lock()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return targets

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=4, max_stage_workers=4, well_workers=2, unit_workers=2)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            output_rel_root="spikesort_outputs",
            force_single_well_sort=False,
            phase_sequence=("bootstrap_concat_binary", "sort"),
            bootstrap_concat_binary_enabled=True,
            bootstrap_concat_binary_resource_class="bootstrap_concat_binary",
            sort_enabled=True,
            sort_resource_class="kilosort4",
            summarize_sort_enabled=False,
            bombcell_label_enabled=False,
            merge_slay_enabled=False,
            merge_si_auto_enabled=False,
            merge_unitmatch_enabled=False,
            cleanup_concat_binary_enabled=False,
        )

    def _fake_phase_runner(phase_name: str):
        def _runner(**kwargs):
            with active_lock:
                active_counts[phase_name] += 1
                max_active_counts[phase_name] = max(max_active_counts[phase_name], active_counts[phase_name])
            time.sleep(0.05)
            with active_lock:
                active_counts[phase_name] -= 1
            return SpikesortResult(
                well_out_dir=tmp_path / str(kwargs["target"].stream_id),
                spikesort_out_dir=tmp_path / str(kwargs["target"].stream_id) / "spikesort_outputs",
                summary_json=tmp_path / f"{phase_name}_{kwargs['target'].stream_id}.json",
                outputs={},
            )

        return _runner

    def _fake_distribute_targets(*, targets, well_workers: int, worker_fn, **kwargs):
        target_list = list(targets)
        start_barrier = threading.Barrier(len(target_list))

        def _run_target(item: ExecutionTarget) -> TargetStageResult:
            start_barrier.wait(timeout=1.0)
            try:
                return TargetStageResult(target=item, status="ok", result=worker_fn(item))
            except Exception as exc:
                return TargetStageResult(target=item, status="error", error=str(exc))

        with ThreadPoolExecutor(max_workers=int(well_workers)) as executor:
            return list(executor.map(_run_target, target_list))

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(
        pipeline_runner,
        "_run_spikesort_bootstrap_concat_binary_target",
        _fake_phase_runner("bootstrap_concat_binary"),
    )
    monkeypatch.setattr(pipeline_runner, "_run_spikesort_sort_target", _fake_phase_runner("sort"))
    monkeypatch.setattr(pipeline_runner, "distribute_targets", _fake_distribute_targets)

    agg = run_spikesort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 2
    assert agg.succeeded_targets == 2
    assert agg.failed_targets == 0
    assert max_active_counts["bootstrap_concat_binary"] == 2
    assert max_active_counts["sort"] == 1


def test_enabled_spikesort_runtime_phase_plan_uses_configured_sequence_and_skips_disabled() -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    stage_config = SimpleNamespace(
        bootstrap_concat_binary_enabled=True,
        sort_enabled=True,
        summarize_sort_enabled=False,
        bombcell_label_enabled=True,
        merge_slay_enabled=True,
        merge_si_auto_enabled=False,
        merge_unitmatch_enabled=False,
        cleanup_concat_binary_enabled=True,
        phase_sequence=("bombcell_label", "merge_unitmatch", "sort", "cleanup", "merge_slay", "bootstrap"),
    )

    phase_plan = pipeline_runner._enabled_spikesort_runtime_phase_plan(stage_config)

    assert [phase.name for phase in phase_plan] == [
        "spikesort.bombcell_label",
        "spikesort.sort",
        "spikesort.cleanup_concat_binary",
        "spikesort.merge_SLAy",
        "spikesort.bootstrap_concat_binary",
    ]


def test_run_spikesort_sort_from_runtime_ignores_phase_debug_limits(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target_a = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_a.h5",
        h5_path=tmp_path / "test_a.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )
    target_b = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_b.h5",
        h5_path=tmp_path / "test_b.h5",
        stream_id="well002",
        mea_output_root=tmp_path,
    )
    target_c = ExecutionTarget(
        dataset_index=1,
        dataset_id="dataset_001:test_c.h5",
        h5_path=tmp_path / "test_c.h5",
        stream_id="well003",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target_a, target_b, target_c]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            sort_debug_mode_enabled=True,
            sort_debug_limit_datasets=1,
            sort_debug_limit_wells=1,
        )

    def _fake_build_spikesort_inputs_for_target(*, target, stage_config, unit_workers: int):
        return SpikesortInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
        )

    def _fake_run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
        return SpikesortResult(
            well_out_dir=tmp_path / f"well_out_{inputs.stream_id}",
            spikesort_out_dir=tmp_path / f"spikesort_out_{inputs.stream_id}",
            summary_json=tmp_path / f"spikesort_summary_{inputs.stream_id}.json",
            outputs={"sorter_output_dir": f"sorter_output_{inputs.stream_id}"},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", _fake_build_spikesort_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_spikesort", _fake_run_spikesort)

    agg = run_spikesort_sort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.stage == "spikesort.sort"
    assert agg.total_targets == 3
    assert agg.succeeded_targets == 3
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.dataset_index == 0
    assert agg.target_results[0].target.stream_id == "well001"
    assert agg.target_results[-1].target.dataset_index == 1
    assert agg.target_results[-1].target.stream_id == "well003"


def test_run_spikesort_sort_from_runtime_applies_global_debug_dataset_and_well_limits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    targets = [
        ExecutionTarget(
            dataset_index=0,
            dataset_id="dataset_000:test_a.h5",
            h5_path=tmp_path / "test_a.h5",
            stream_id="well001",
            mea_output_root=tmp_path,
        ),
        ExecutionTarget(
            dataset_index=0,
            dataset_id="dataset_000:test_a.h5",
            h5_path=tmp_path / "test_a.h5",
            stream_id="well002",
            mea_output_root=tmp_path,
        ),
        ExecutionTarget(
            dataset_index=0,
            dataset_id="dataset_000:test_a.h5",
            h5_path=tmp_path / "test_a.h5",
            stream_id="well003",
            mea_output_root=tmp_path,
        ),
        ExecutionTarget(
            dataset_index=1,
            dataset_id="dataset_001:test_b.h5",
            h5_path=tmp_path / "test_b.h5",
            stream_id="well001",
            mea_output_root=tmp_path,
        ),
    ]

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    built_targets: list[tuple[int, str]] = []
    select_kwargs: dict[str, object] = {}

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(**kwargs):
        select_kwargs.update(kwargs)
        return list(targets)

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str, target_count: int | None = None):
        _ = bundle, stage_name, target_count
        return StageParallelism(max_workers=2, max_stage_workers=2, well_workers=2, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_datasets=1,
            debug_limit_wells=2,
            sort_debug_mode_enabled=False,
        )

    def _fake_build_spikesort_inputs_for_target(*, target, stage_config, unit_workers: int):
        _ = stage_config, unit_workers
        built_targets.append((int(target.dataset_index), str(target.stream_id)))
        return SpikesortInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
        )

    def _fake_run_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
        return SpikesortResult(
            well_out_dir=tmp_path / f"well_out_{inputs.stream_id}",
            spikesort_out_dir=tmp_path / f"spikesort_out_{inputs.stream_id}",
            summary_json=tmp_path / f"spikesort_summary_{inputs.stream_id}.json",
            outputs={"sorter_output_dir": f"sorter_output_{inputs.stream_id}"},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", _fake_build_spikesort_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_spikesort", _fake_run_spikesort)

    agg = run_spikesort_sort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.stage == "spikesort.sort"
    assert agg.total_targets == 2
    assert agg.succeeded_targets == 2
    assert agg.failed_targets == 0
    assert built_targets == [(0, "well001"), (0, "well002")]
    assert select_kwargs["limit_datasets"] == 1
    assert select_kwargs["limit_wells"] == 2


def test_run_spikesort_summarize_sort_from_runtime_marks_target_ok(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = SpikesortInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
        summarize_sort_enabled=True,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_spikesort_inputs_for_target(*, target, stage_config, unit_workers: int):
        return dummy_inputs

    def _fake_summarize_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
        return SpikesortResult(
            well_out_dir=tmp_path / "well_out",
            spikesort_out_dir=tmp_path / "spikesort_out",
            summary_json=tmp_path / "summarize_sort_summary.json",
            outputs={"summarize_sort.summary_json": str(tmp_path / "summarize_sort_summary.json")},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", _fake_build_spikesort_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "summarize_spikesort", _fake_summarize_spikesort)

    agg = run_spikesort_summarize_sort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.stage == "spikesort.summarize_sort"
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].status == "ok"


def test_run_spikesort_summarize_sort_from_runtime_ignores_phase_debug_limits(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target_a = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_a.h5",
        h5_path=tmp_path / "test_a.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )
    target_b = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_b.h5",
        h5_path=tmp_path / "test_b.h5",
        stream_id="well002",
        mea_output_root=tmp_path,
    )
    target_c = ExecutionTarget(
        dataset_index=1,
        dataset_id="dataset_001:test_c.h5",
        h5_path=tmp_path / "test_c.h5",
        stream_id="well003",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target_a, target_b, target_c]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            summarize_sort_debug_mode_enabled=True,
            summarize_sort_debug_limit_datasets=1,
            summarize_sort_debug_limit_wells=1,
        )

    def _fake_build_spikesort_inputs_for_target(*, target, stage_config, unit_workers: int):
        return SpikesortInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
            summarize_sort_enabled=True,
        )

    def _fake_summarize_spikesort(inputs: SpikesortInputs) -> SpikesortResult:
        return SpikesortResult(
            well_out_dir=tmp_path / f"well_out_{inputs.stream_id}",
            spikesort_out_dir=tmp_path / f"spikesort_out_{inputs.stream_id}",
            summary_json=tmp_path / f"summarize_sort_summary_{inputs.stream_id}.json",
            outputs={"summarize_sort.summary_json": str(tmp_path / f"summarize_sort_summary_{inputs.stream_id}.json")},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_spikesort_inputs_for_target", _fake_build_spikesort_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "summarize_spikesort", _fake_summarize_spikesort)

    agg = run_spikesort_summarize_sort_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.stage == "spikesort.summarize_sort"
    assert agg.total_targets == 3
    assert agg.succeeded_targets == 3
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.dataset_index == 0
    assert agg.target_results[0].target.stream_id == "well001"
    assert agg.target_results[-1].target.dataset_index == 1
    assert agg.target_results[-1].target.stream_id == "well003"


def test_run_spikesort_bombcell_label_from_runtime_marks_target_ok(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            output_rel_root="spikesort_outputs",
            force_restart=False,
            bombcell_label_debug_mode_enabled=False,
            bombcell_label_debug_limit_datasets=None,
            bombcell_label_debug_limit_wells=None,
        )

    def _fake_run_spikesort_bombcell(**kwargs) -> SpikesortBombcellResult:
        return SpikesortBombcellResult(
            well_out_dir=tmp_path / "well_out",
            bombcell_out_dir=tmp_path / "bombcell_out",
            summary_json=tmp_path / "bombcell_summary.json",
            outputs={"bombcell_label.summary_json": str(tmp_path / "bombcell_summary.json")},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "run_spikesort_bombcell", _fake_run_spikesort_bombcell)

    agg = run_spikesort_bombcell_label_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.stage == "spikesort.bombcell_label"
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].status == "ok"


def test_run_spikesort_bombcell_label_from_runtime_ignores_phase_debug_limits(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target_a = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_a.h5",
        h5_path=tmp_path / "test_a.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )
    target_b = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_b.h5",
        h5_path=tmp_path / "test_b.h5",
        stream_id="well002",
        mea_output_root=tmp_path,
    )
    target_c = ExecutionTarget(
        dataset_index=1,
        dataset_id="dataset_001:test_c.h5",
        h5_path=tmp_path / "test_c.h5",
        stream_id="well003",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target_a, target_b, target_c]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            output_rel_root="spikesort_outputs",
            force_restart=False,
            bombcell_label_debug_mode_enabled=True,
            bombcell_label_debug_limit_datasets=1,
            bombcell_label_debug_limit_wells=1,
        )

    def _fake_run_spikesort_bombcell(**kwargs) -> SpikesortBombcellResult:
        stream_id = str(kwargs.get("stream_id"))
        return SpikesortBombcellResult(
            well_out_dir=tmp_path / f"well_out_{stream_id}",
            bombcell_out_dir=tmp_path / f"bombcell_out_{stream_id}",
            summary_json=tmp_path / f"bombcell_summary_{stream_id}.json",
            outputs={"bombcell_label.summary_json": str(tmp_path / f"bombcell_summary_{stream_id}.json")},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "run_spikesort_bombcell", _fake_run_spikesort_bombcell)

    agg = run_spikesort_bombcell_label_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.stage == "spikesort.bombcell_label"
    assert agg.total_targets == 3
    assert agg.succeeded_targets == 3
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.dataset_index == 0
    assert agg.target_results[0].target.stream_id == "well001"
    assert agg.target_results[-1].target.dataset_index == 1
    assert agg.target_results[-1].target.stream_id == "well003"


def test_run_spikesort_merge_from_runtime_inherits_template_heatmap_probe_dimensions(
    monkeypatch, tmp_path: Path
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    captured_stage_configs: list[SimpleNamespace] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            output_rel_root="spikesort_outputs",
            force_restart=False,
            force_replot=False,
            merge_reports_2panel_inherit_probe_dimensions=False,
            merge_reports_2panel_probe_dim_x_um=None,
            merge_reports_2panel_probe_dim_y_um=None,
            merge_reports_template_heatmaps_inherit_probe_dimensions=True,
            merge_reports_template_heatmaps_probe_dim_x_um=None,
            merge_reports_template_heatmaps_probe_dim_y_um=None,
            merge_reports_template_heatmaps_probe_pitch_um=None,
            merge_reports_template_heatmaps_probe_electrode_size_um_x=None,
            merge_reports_template_heatmaps_probe_electrode_size_um_y=None,
        )

    def _fake_parse_probe_geometry_from_data_config(*, data_config):
        return SimpleNamespace(
            active_area_um_x=3850.0,
            active_area_um_y=2100.0,
            pitch_um=17.5,
            electrode_size_um_x=12.0,
            electrode_size_um_y=8.8,
        )

    def _fake_run_spikesort_merge(
        *,
        h5_path: Path,
        stream_id: str,
        mea_output_root: Path,
        output_rel_root: str,
        stage_config,
        force_restart: bool,
        force_replot: bool = False,
    ) -> SpikesortMergeResult:
        captured_stage_configs.append(stage_config)
        return SpikesortMergeResult(
            well_out_dir=tmp_path / "well_out",
            merge_out_dir=tmp_path / "merge_out",
            summary_json=tmp_path / "merge_summary.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(
        pipeline_runner,
        "parse_probe_geometry_from_data_config",
        _fake_parse_probe_geometry_from_data_config,
    )
    monkeypatch.setattr(pipeline_runner, "run_spikesort_merge", _fake_run_spikesort_merge)

    agg = run_spikesort_merge_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert len(captured_stage_configs) == 1
    assert captured_stage_configs[0].merge_reports_template_heatmaps_probe_dim_x_um == 3850.0
    assert captured_stage_configs[0].merge_reports_template_heatmaps_probe_dim_y_um == 2100.0
    assert captured_stage_configs[0].merge_reports_template_heatmaps_probe_pitch_um == 17.5
    assert captured_stage_configs[0].merge_reports_template_heatmaps_probe_electrode_size_um_x == 12.0
    assert captured_stage_configs[0].merge_reports_template_heatmaps_probe_electrode_size_um_y == 8.8


def test_run_spikesort_merge_slay_from_runtime_applies_phase_workspace_config(
    monkeypatch, tmp_path: Path
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    captured_stage_configs: list[SimpleNamespace] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str, **kwargs):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            output_rel_root="spikesort_outputs",
            merge_sequence=("SLAy", "auto_merge"),
            merge_units_enabled=True,
            merge_rel_output_root="merge_output",
            merge_delete_outputs_on_force_restart=False,
            merge_force_restart=False,
            merge_force_replot=False,
            cache_sorting_outputs_before_merge=True,
            cache_sorting_outputs_before_merge_use_canonical_workspace=False,
            cache_sorting_outputs_before_merge_canonical_workspace_relpath="cache/merge_workspace",
            cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run=False,
            cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer=False,
            cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success=True,
            cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure=True,
            cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace=False,
            slay_enabled=False,
            merge_slay_enabled=True,
            merge_slay_rel_output_root="merge_SLAy",
            merge_slay_delete_outputs_on_force_restart=True,
            merge_slay_force_restart=False,
            merge_slay_force_replot=False,
            merge_slay_use_canonical_workspace=True,
            merge_slay_canonical_workspace_relpath="cache/slay_workspace",
            merge_slay_canonical_workspace_refresh_on_run=True,
            merge_slay_canonical_workspace_rebuild_analyzer=False,
            merge_slay_publish_canonical_to_stage_outputs_on_success=True,
            merge_slay_publish_canonical_to_stage_outputs_on_failure=False,
            merge_slay_assert_uses_canonical_workspace=True,
            merge_phase_runtime_overrides={
                "merge_slay": {
                    "merge_analyzer_n_jobs": 7,
                    "merge_reports_enabled": False,
                    "pre_merge_metadata_enabled": False,
                }
            },
            merge_reports_2panel_inherit_probe_dimensions=False,
            merge_reports_template_heatmaps_inherit_probe_dimensions=False,
            force_restart=False,
            force_replot=False,
        )

    def _fake_run_spikesort_merge(*, h5_path, stream_id, mea_output_root, output_rel_root, stage_config, force_restart, force_replot):
        captured_stage_configs.append(stage_config)
        return SpikesortMergeResult(
            well_out_dir=tmp_path / "well_out",
            merge_out_dir=tmp_path / "merge_out",
            summary_json=tmp_path / "merge_summary.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "run_spikesort_merge", _fake_run_spikesort_merge)

    agg = run_spikesort_merge_slay_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert len(captured_stage_configs) == 1
    mapped_stage_config = captured_stage_configs[0]
    assert mapped_stage_config.merge_sequence == ("SLAy",)
    assert mapped_stage_config.merge_units_enabled is True
    assert mapped_stage_config.merge_rel_output_root == "merge_SLAy"
    assert mapped_stage_config.merge_delete_outputs_on_force_restart is True
    assert mapped_stage_config.cache_sorting_outputs_before_merge is False
    assert mapped_stage_config.cache_sorting_outputs_before_merge_use_canonical_workspace is True
    assert mapped_stage_config.cache_sorting_outputs_before_merge_canonical_workspace_relpath == "cache/slay_workspace"
    assert mapped_stage_config.cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run is True
    assert mapped_stage_config.cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer is False
    assert mapped_stage_config.cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success is True
    assert mapped_stage_config.cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure is False
    assert mapped_stage_config.cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace is True
    assert mapped_stage_config.slay_enabled is True
    assert mapped_stage_config.merge_analyzer_n_jobs == 7
    assert mapped_stage_config.merge_reports_enabled is False
    assert mapped_stage_config.pre_merge_metadata_enabled is False


def test_run_spikesort_merge_slay_from_runtime_ignores_phase_debug_limits(
    monkeypatch, tmp_path: Path
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target_a = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_a.h5",
        h5_path=tmp_path / "test_a.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )
    target_b = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test_b.h5",
        h5_path=tmp_path / "test_b.h5",
        stream_id="well002",
        mea_output_root=tmp_path,
    )
    target_c = ExecutionTarget(
        dataset_index=1,
        dataset_id="dataset_001:test_c.h5",
        h5_path=tmp_path / "test_c.h5",
        stream_id="well003",
        mea_output_root=tmp_path,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target_a, target_b, target_c]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str, **kwargs):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_spikesort_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            output_rel_root="spikesort_outputs",
            merge_sequence=("SLAy", "auto_merge"),
            merge_units_enabled=True,
            merge_rel_output_root="merge_output",
            merge_delete_outputs_on_force_restart=False,
            merge_force_restart=False,
            merge_force_replot=False,
            cache_sorting_outputs_before_merge=True,
            cache_sorting_outputs_before_merge_use_canonical_workspace=False,
            cache_sorting_outputs_before_merge_canonical_workspace_relpath="cache/merge_workspace",
            cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run=False,
            cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer=False,
            cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success=True,
            cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure=True,
            cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace=False,
            slay_enabled=False,
            merge_slay_enabled=True,
            merge_slay_rel_output_root="merge_SLAy",
            merge_slay_delete_outputs_on_force_restart=True,
            merge_slay_force_restart=False,
            merge_slay_force_replot=False,
            merge_slay_use_canonical_workspace=True,
            merge_slay_canonical_workspace_relpath="cache/slay_workspace",
            merge_slay_canonical_workspace_refresh_on_run=True,
            merge_slay_canonical_workspace_rebuild_analyzer=False,
            merge_slay_publish_canonical_to_stage_outputs_on_success=True,
            merge_slay_publish_canonical_to_stage_outputs_on_failure=False,
            merge_slay_assert_uses_canonical_workspace=True,
            merge_slay_debug_mode_enabled=True,
            merge_slay_debug_limit_datasets=1,
            merge_slay_debug_limit_wells=1,
            merge_phase_runtime_overrides=None,
            merge_reports_2panel_inherit_probe_dimensions=False,
            merge_reports_template_heatmaps_inherit_probe_dimensions=False,
            force_restart=False,
            force_replot=False,
        )

    def _fake_run_spikesort_merge(*, h5_path, stream_id, mea_output_root, output_rel_root, stage_config, force_restart, force_replot):
        return SpikesortMergeResult(
            well_out_dir=tmp_path / f"well_out_{stream_id}",
            merge_out_dir=tmp_path / f"merge_out_{stream_id}",
            summary_json=tmp_path / f"merge_summary_{stream_id}.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_spikesort_stage_config", _fake_parse_spikesort_stage_config)
    monkeypatch.setattr(pipeline_runner, "run_spikesort_merge", _fake_run_spikesort_merge)

    agg = run_spikesort_merge_slay_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 3
    assert agg.succeeded_targets == 3
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.dataset_index == 0
    assert agg.target_results[0].target.stream_id == "well001"
    assert agg.target_results[-1].target.dataset_index == 1
    assert agg.target_results[-1].target.stream_id == "well003"
