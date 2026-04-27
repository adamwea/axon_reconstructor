from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.execution.results import MultiTargetStageResult, TargetStageResult
from axon_recon.pipeline.runner import (
    run_spikesort_bombcell_label_from_runtime,
    run_spikesort_from_runtime,
    run_spikesort_merge_from_runtime,
    run_spikesort_sort_from_runtime,
    run_spikesort_summarize_sort_from_runtime,
)
from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs
from axon_recon.pipeline.stages.spikesort.orchestrators.merge_slay import (
    run_spikesort_merge_slay_from_runtime,
)
from axon_recon.pipeline.stages.spikesort.models.results import (
    SpikesortBombcellResult,
    SpikesortMergeResult,
    SpikesortResult,
)


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

    def _fake_select_execution_targets(*, bundle):
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

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
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

    def _fake_distribute_targets(*, targets, well_workers: int, worker_fn):
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


def test_run_spikesort_sort_from_runtime_applies_phase_debug_limits(monkeypatch, tmp_path: Path) -> None:
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
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.dataset_index == 0
    assert agg.target_results[0].target.stream_id == "well001"


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

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
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


def test_run_spikesort_summarize_sort_from_runtime_applies_phase_debug_limits(monkeypatch, tmp_path: Path) -> None:
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
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.dataset_index == 0
    assert agg.target_results[0].target.stream_id == "well001"


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


def test_run_spikesort_bombcell_label_from_runtime_applies_phase_debug_limits(monkeypatch, tmp_path: Path) -> None:
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
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.dataset_index == 0
    assert agg.target_results[0].target.stream_id == "well001"


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


def test_run_spikesort_merge_slay_from_runtime_applies_phase_debug_limits(
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

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.dataset_index == 0
    assert agg.target_results[0].target.stream_id == "well001"
