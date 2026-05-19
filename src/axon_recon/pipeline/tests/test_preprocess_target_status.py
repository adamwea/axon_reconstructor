from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.runner import (
    run_preprocess_from_runtime,
    run_preprocess_plot_raster_threshold_from_runtime,
    run_preprocess_plot_segment_channel_layouts_from_runtime,
    run_preprocess_plot_segment_traces_from_runtime,
    run_preprocess_preprocess_segments_from_runtime,
    run_preprocess_save_rec_metadata_from_runtime,
)
from axon_recon.pipeline.stages.preprocess.models.inputs import PreprocessInputs
from axon_recon.pipeline.stages.preprocess.models.results import PreprocessResult


def test_run_preprocess_from_runtime_marks_target_ok(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = PreprocessInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    select_calls: list[bool] = []
    unit_worker_calls: list[int] = []
    divider_stdout_calls: list[bool] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        select_calls.append(bool(materialize_scratch_inputs))
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        unit_worker_calls.append(int(unit_workers))
        return dummy_inputs

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        divider_stdout_calls.append(bool(inputs.logging_subphase_dividers_to_stdout))
        return PreprocessResult(
            well_out_dir=tmp_path / "well_out",
            preprocess_out_dir=tmp_path / "preprocess_out",
            summary_json=tmp_path / "preprocess_summary.json",
            outputs={"preprocessed_recording_dir": "preprocessed_recording"},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    agg = run_preprocess_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].status == "ok"
    assert agg.target_results[0].result is not None
    assert select_calls == [False]
    assert unit_worker_calls == [12]
    assert divider_stdout_calls == [False]


def test_run_preprocess_from_runtime_logs_stage_topology(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = PreprocessInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", lambda *, config_path: _DummyBundle())
    monkeypatch.setattr(
        pipeline_runner,
        "select_execution_targets",
        lambda *, bundle, materialize_scratch_inputs=False: [target],
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_stage_parallelism",
        lambda *, bundle, stage_name: StageParallelism(well_workers=2, unit_workers=12),
    )
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", lambda **kwargs: SimpleNamespace(debug_limit_wells=None))
    monkeypatch.setattr(
        pipeline_runner,
        "build_preprocess_inputs_for_target",
        lambda *, target, stage_config, unit_workers: dummy_inputs,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "run_preprocess",
        lambda inputs: PreprocessResult(
            well_out_dir=tmp_path / "well_out",
            preprocess_out_dir=tmp_path / "preprocess_out",
            summary_json=tmp_path / "preprocess_summary.json",
            outputs={},
        ),
    )

    with caplog.at_level(logging.INFO):
        run_preprocess_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    messages = [record.getMessage() for record in caplog.records]
    assert "Starting stage: preprocess" in messages
    assert "Execution topology: stage_global_order=true, well_local_phase_sequence=true" in messages
    assert "Selected wells: 1" in messages
    assert "well_workers=2" in messages


def test_run_preprocess_from_runtime_applies_force_restart_override_to_stage_inputs(
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
        runtime_config = object()
        data_config = object()

    parse_force_restart_overrides: list[bool | None] = []
    captured_force_restart_inputs: list[bool] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        _ = bundle, materialize_scratch_inputs
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        _ = bundle, stage_name
        return StageParallelism(well_workers=1, unit_workers=8)

    def _fake_parse_preprocess_stage_config(**kwargs):
        parse_force_restart_overrides.append(kwargs.get("force_restart_override"))
        return SimpleNamespace(debug_limit_wells=None, force_restart=True)

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        _ = unit_workers
        return PreprocessInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
            force_restart=bool(getattr(stage_config, "force_restart", False)),
        )

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        captured_force_restart_inputs.append(bool(inputs.force_restart))
        return PreprocessResult(
            well_out_dir=tmp_path / "well_out",
            preprocess_out_dir=tmp_path / "preprocess_out",
            summary_json=tmp_path / "preprocess_summary.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    agg = run_preprocess_from_runtime(
        config_path=str(tmp_path / "runtime.yml"),
        force_restart_override=True,
    )

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert parse_force_restart_overrides == [True]
    assert captured_force_restart_inputs == [True]


def test_run_preprocess_from_runtime_marks_target_error(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = PreprocessInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    select_calls: list[bool] = []
    unit_worker_calls: list[int] = []
    divider_stdout_calls: list[bool] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        select_calls.append(bool(materialize_scratch_inputs))
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        unit_worker_calls.append(int(unit_workers))
        return dummy_inputs

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        divider_stdout_calls.append(bool(inputs.logging_subphase_dividers_to_stdout))
        raise RuntimeError("preprocess failed")

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    agg = run_preprocess_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 0
    assert agg.failed_targets == 1
    assert agg.target_results[0].status == "error"
    assert "preprocess failed" in str(agg.target_results[0].error)
    assert select_calls == [False]
    assert unit_worker_calls == [12]
    assert divider_stdout_calls == [False]


def test_run_preprocess_from_runtime_applies_debug_well_limit(monkeypatch, tmp_path: Path) -> None:
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

    select_calls: list[bool] = []
    select_kwargs: list[dict[str, object]] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(**kwargs):
        select_kwargs.append(dict(kwargs))
        materialize_scratch_inputs = bool(kwargs.get("materialize_scratch_inputs", False))
        select_calls.append(bool(materialize_scratch_inputs))
        return [target_a, target_b]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=1,
            phases=SimpleNamespace(),
        )

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        return PreprocessInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
        )

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        return PreprocessResult(
            well_out_dir=tmp_path / f"well_out_{inputs.stream_id}",
            preprocess_out_dir=tmp_path / f"preprocess_out_{inputs.stream_id}",
            summary_json=tmp_path / f"preprocess_summary_{inputs.stream_id}.json",
            outputs={"preprocessed_recording_dir": f"preprocessed_recording_{inputs.stream_id}"},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    agg = run_preprocess_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].target.stream_id == "well001"
    assert select_calls == [False]
    assert select_kwargs[0]["limit_wells"] == 1


def test_run_preprocess_from_runtime_applies_global_debug_dataset_and_well_limits(
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
    select_kwargs: list[dict[str, object]] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(**kwargs):
        select_kwargs.append(dict(kwargs))
        return list(targets)

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        _ = bundle, stage_name
        return StageParallelism(well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_datasets=1,
            debug_limit_wells=2,
            phases=SimpleNamespace(),
        )

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        _ = stage_config, unit_workers
        built_targets.append((int(target.dataset_index), str(target.stream_id)))
        return PreprocessInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
        )

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        return PreprocessResult(
            well_out_dir=tmp_path / f"well_out_{inputs.stream_id}",
            preprocess_out_dir=tmp_path / f"preprocess_out_{inputs.stream_id}",
            summary_json=tmp_path / f"preprocess_summary_{inputs.stream_id}.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    agg = run_preprocess_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 2
    assert agg.succeeded_targets == 2
    assert agg.failed_targets == 0
    assert built_targets == [(0, "well001"), (0, "well002")]
    assert select_kwargs[0]["limit_datasets"] == 1
    assert select_kwargs[0]["limit_wells"] == 2


def test_preprocess_debug_limits_select_first_wells_per_dataset(tmp_path: Path) -> None:
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

    limited = pipeline_runner._apply_preprocess_debug_target_limits(
        stage_name="preprocess",
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


@pytest.mark.parametrize(
    (
        "runner_fn",
        "runner_symbol",
        "api_symbol",
        "stage_name",
        "phase_name",
        "expected_unit_workers",
        "expected_divider_stdout",
    ),
    [
        (
            run_preprocess_save_rec_metadata_from_runtime,
            "run_preprocess_save_rec_metadata_from_runtime",
            "run_preprocess_save_rec_metadata",
            "preprocess.save_rec_metadata",
            "save_rec_metadata",
            1,
            True,
        ),
        (
            run_preprocess_preprocess_segments_from_runtime,
            "run_preprocess_preprocess_segments_from_runtime",
            "run_preprocess_preprocess_segments",
            "preprocess.preprocess_segments",
            "preprocess_segments",
            12,
            False,
        ),
        (
            run_preprocess_plot_segment_traces_from_runtime,
            "run_preprocess_plot_segment_traces_from_runtime",
            "run_preprocess_plot_segment_traces",
            "preprocess.plot_segment_traces",
            "plot_segment_traces",
            1,
            True,
        ),
        (
            run_preprocess_plot_segment_channel_layouts_from_runtime,
            "run_preprocess_plot_segment_channel_layouts_from_runtime",
            "run_preprocess_plot_segment_channel_layouts",
            "preprocess.plot_segment_channel_layouts",
            "plot_segment_channel_layouts",
            1,
            True,
        ),
        # plot_concat_traces / plot_concat_channel_layout moved to spikesort
        # in phase_roster_cleanup_plan slice 8; their target-status checks
        # belong in test_spikesort_target_status.py (or its successor) now.
        (
            run_preprocess_plot_raster_threshold_from_runtime,
            "run_preprocess_plot_raster_threshold_from_runtime",
            "run_preprocess_plot_raster_threshold",
            "preprocess.plot_raster_threshold",
            "plot_raster_threshold",
            1,
            True,
        ),
    ],
)
def test_run_preprocess_substage_from_runtime_marks_target_ok(
    monkeypatch,
    tmp_path: Path,
    runner_fn,
    runner_symbol: str,
    api_symbol: str,
    stage_name: str,
    phase_name: str,
    expected_unit_workers: int,
    expected_divider_stdout: bool,
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = PreprocessInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    select_calls: list[bool] = []
    select_kwargs: list[dict[str, object]] = []
    stage_configs_seen: list[object] = []
    unit_worker_calls: list[int] = []
    divider_stdout_calls: list[bool] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(**kwargs):
        select_kwargs.append(dict(kwargs))
        materialize_scratch_inputs = bool(kwargs.get("materialize_scratch_inputs", False))
        select_calls.append(bool(materialize_scratch_inputs))
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        stage_configs_seen.append(stage_config)
        unit_worker_calls.append(int(unit_workers))
        return dummy_inputs

    def _fake_run_substage(inputs: PreprocessInputs) -> dict[str, object]:
        divider_stdout_calls.append(bool(inputs.logging_subphase_dividers_to_stdout))
        return {
            "phase": phase_name,
            "preprocess_out_dir": str(tmp_path / "preprocess_out"),
            "summary_json": str(tmp_path / f"{phase_name}_summary.json"),
            "outputs": {},
        }

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, api_symbol, _fake_run_substage)

    agg = runner_fn(
        config_path=str(tmp_path / "runtime.yml"),
        limit_segments_override=2,
        limit_datasets_override=1,
        target_datasets_override=[1],
        limit_wells_per_dataset_override=1,
    )

    assert agg.stage == stage_name
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].status == "ok"
    assert agg.target_results[0].result == {
        "phase": phase_name,
        "preprocess_out_dir": str(tmp_path / "preprocess_out"),
        "summary_json": str(tmp_path / f"{phase_name}_summary.json"),
        "outputs": {},
    }
    # After slice 5, no preprocess substage triggers scratch materialization
    # (copy_src_to_scratch moved to the init stage).
    expected_materialize = False
    assert select_calls == [expected_materialize]
    assert stage_configs_seen
    assert getattr(stage_configs_seen[0], "debug_limit_segments_per_well") == 2
    assert getattr(stage_configs_seen[0], "debug_limit_datasets") == 1
    assert getattr(stage_configs_seen[0], "debug_limit_wells_per_dataset") == 1
    assert select_kwargs[0]["limit_datasets"] == 1
    assert select_kwargs[0]["target_datasets"] == [1]
    assert select_kwargs[0]["limit_wells_per_dataset"] == 1
    assert unit_worker_calls == [expected_unit_workers]
    assert divider_stdout_calls == [expected_divider_stdout]


def test_preprocess_phase_from_args_forwards_debug_limits(monkeypatch, tmp_path: Path) -> None:
    from axon_recon.pipeline.stages.preprocess.orchestrators import _shared as preprocess_shared

    runtime_cfg = tmp_path / "runtime.yml"
    runtime_cfg.write_text("{}\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def _fake_print_preprocess_aggregate(agg: object) -> int:
        seen["aggregate"] = agg
        return 0

    def _fake_runtime_runner(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            stage="preprocess.preprocess_segments",
            total_targets=0,
            succeeded_targets=0,
            failed_targets=0,
            target_results=[],
        )

    monkeypatch.setattr(preprocess_shared, "print_preprocess_aggregate", _fake_print_preprocess_aggregate)

    rc = preprocess_shared.run_preprocess_phase_from_args(
        SimpleNamespace(
            config=str(runtime_cfg),
            limit_segments=2,
            limit_datasets=1,
            target_datasets=["1,", "3"],
            limit_wells_per_dataset=1,
            force_restart=True,
            replot=False,
        ),
        runtime_runner=_fake_runtime_runner,
    )

    assert rc == 0
    assert seen["config_path"] == str(runtime_cfg)
    assert seen["limit_segments_override"] == 2
    assert seen["limit_datasets_override"] == 1
    assert seen["target_datasets_override"] == [1, 3]
    assert seen["limit_wells_per_dataset_override"] == 1
    assert seen["force_restart_override"] is True


@pytest.mark.parametrize(
    ("module_name", "runner_symbol", "phase_name"),
    [
        ("save_rec_metadata", "run_preprocess_save_rec_metadata_from_runtime", "save_rec_metadata"),
        ("preprocess_segments", "run_preprocess_preprocess_segments_from_runtime", "preprocess_segments"),
        ("plot_segment_traces", "run_preprocess_plot_segment_traces_from_runtime", "plot_segment_traces"),
        (
            "plot_segment_channel_layouts",
            "run_preprocess_plot_segment_channel_layouts_from_runtime",
            "plot_segment_channel_layouts",
        ),
        # plot_concat_traces / plot_concat_channel_layout moved to spikesort
        # in slice 8; their orchestrator wrappers live under spikesort now.
        ("plot_raster_threshold", "run_preprocess_plot_raster_threshold_from_runtime", "plot_raster_threshold"),
    ],
)
def test_preprocess_phase_runtime_wrappers_forward_debug_limits(
    monkeypatch,
    module_name: str,
    runner_symbol: str,
    phase_name: str,
) -> None:
    module = importlib.import_module(f"axon_recon.pipeline.stages.preprocess.orchestrators.{module_name}")
    seen: dict[str, object] = {}

    def _fake_run_preprocess_phase_from_runtime(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stage=f"preprocess.{phase_name}")

    monkeypatch.setattr(module, "run_preprocess_phase_from_runtime", _fake_run_preprocess_phase_from_runtime)

    aggregate = getattr(module, runner_symbol)(
        config_path="runtime.yml",
        limit_segments_override=2,
        limit_datasets_override=1,
        target_datasets_override=[1, 3],
        limit_wells_per_dataset_override=1,
        force_restart_override=True,
    )

    assert aggregate.stage == f"preprocess.{phase_name}"
    assert seen["phase_name"] == phase_name
    assert seen["config_path"] == "runtime.yml"
    assert seen["limit_segments_override"] == 2
    assert seen["limit_datasets_override"] == 1
    assert seen["target_datasets_override"] == [1, 3]
    assert seen["limit_wells_per_dataset_override"] == 1
    assert seen["force_restart_override"] is True


def test_run_preprocess_from_runtime_does_not_materialize_scratch_inputs(monkeypatch, tmp_path: Path) -> None:
    """After slice 5, preprocess never materializes scratch inputs.

    The `copy_src_to_scratch` phase moved to the init stage; preprocess
    consumes whatever h5_path was resolved upstream (either the source path
    or, when init ran, the scratch-staged path). This test locks in that
    `run_preprocess_from_runtime` always passes `materialize_scratch_inputs=False`
    to `select_execution_targets`.
    """
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = PreprocessInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    select_calls: list[bool] = []
    unit_worker_calls: list[int] = []
    divider_stdout_calls: list[bool] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        select_calls.append(bool(materialize_scratch_inputs))
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            phases=SimpleNamespace(),
        )

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        unit_worker_calls.append(int(unit_workers))
        return dummy_inputs

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        divider_stdout_calls.append(bool(inputs.logging_subphase_dividers_to_stdout))
        return PreprocessResult(
            well_out_dir=tmp_path / "well_out",
            preprocess_out_dir=tmp_path / "preprocess_out",
            summary_json=tmp_path / "preprocess_summary.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    agg = run_preprocess_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.succeeded_targets == 1
    # Preprocess never materializes scratch inputs after slice 5.
    assert select_calls == [False]
    assert unit_worker_calls == [1]
    assert divider_stdout_calls == [True]


def test_run_preprocess_from_runtime_passes_debug_limits_to_target_selection(
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

    dummy_inputs = PreprocessInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    select_kwargs: list[dict[str, object]] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(**kwargs):
        select_kwargs.append(dict(kwargs))
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_datasets=None,
            debug_limit_wells=None,
            debug_limit_wells_per_dataset=None,
            phases=SimpleNamespace(),
        )

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        _ = target, stage_config, unit_workers
        return dummy_inputs

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        _ = inputs
        return PreprocessResult(
            well_out_dir=tmp_path / "well_out",
            preprocess_out_dir=tmp_path / "preprocess_out",
            summary_json=tmp_path / "preprocess_summary.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    agg = run_preprocess_from_runtime(
        config_path=str(tmp_path / "runtime.yml"),
        limit_datasets_override=1,
        limit_wells_per_dataset_override=1,
    )

    assert agg.succeeded_targets == 1
    assert select_kwargs
    # Preprocess never materializes scratch inputs after slice 5.
    assert select_kwargs[0]["materialize_scratch_inputs"] is False
    assert select_kwargs[0]["limit_datasets"] == 1
    assert select_kwargs[0]["limit_wells_per_dataset"] == 1


def test_run_preprocess_from_runtime_uses_nested_workers_when_heavy_phases_enabled(
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

    dummy_inputs = PreprocessInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    unit_worker_calls: list[int] = []
    divider_stdout_calls: list[bool] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            phases=SimpleNamespace(
                preprocess_segments=SimpleNamespace(enabled=True),
            ),
        )

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        unit_worker_calls.append(int(unit_workers))
        return dummy_inputs

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        divider_stdout_calls.append(bool(inputs.logging_subphase_dividers_to_stdout))
        return PreprocessResult(
            well_out_dir=tmp_path / "well_out",
            preprocess_out_dir=tmp_path / "preprocess_out",
            summary_json=tmp_path / "preprocess_summary.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    with caplog.at_level(logging.INFO):
        agg = run_preprocess_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    messages = [record.getMessage() for record in caplog.records]

    assert agg.succeeded_targets == 1
    assert unit_worker_calls == [12]
    assert divider_stdout_calls == [False]
    assert any(
        "Preprocess worker allocation stage=preprocess well_workers=2 n_jobs=12 n_jobs_source=derived"
        in message
        for message in messages
    )
    assert not any("unit_workers" in message for message in messages if "worker allocation" in message)


def test_run_preprocess_from_runtime_uses_phase_sequence_for_scratch_and_worker_decisions(
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

    dummy_inputs = PreprocessInputs(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=target.mea_output_root,
    )

    class _DummyBundle:
        runtime_config = object()
        data_config = object()

    select_calls: list[bool] = []
    unit_worker_calls: list[int] = []
    divider_stdout_calls: list[bool] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        select_calls.append(bool(materialize_scratch_inputs))
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            phase_sequence=("save_rec_metadata",),
            phases=SimpleNamespace(
                preprocess_segments=SimpleNamespace(enabled=True),
            ),
        )

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        unit_worker_calls.append(int(unit_workers))
        return dummy_inputs

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
        divider_stdout_calls.append(bool(inputs.logging_subphase_dividers_to_stdout))
        return PreprocessResult(
            well_out_dir=tmp_path / "well_out",
            preprocess_out_dir=tmp_path / "preprocess_out",
            summary_json=tmp_path / "preprocess_summary.json",
            outputs={},
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess", _fake_run_preprocess)

    agg = run_preprocess_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.succeeded_targets == 1
    assert select_calls == [False]
    assert unit_worker_calls == [1]
    assert divider_stdout_calls == [True]


def test_run_preprocess_save_rec_metadata_from_runtime_ignores_phase_debug_dataset_and_well_limits(
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
            dataset_index=0,
            dataset_id="dataset_000:test_a.h5",
            h5_path=tmp_path / "test_a.h5",
            stream_id="well004",
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

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        _ = bundle, materialize_scratch_inputs
        return list(targets)

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        _ = bundle, stage_name
        return StageParallelism(well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_wells=None,
            phases=SimpleNamespace(
                save_rec_metadata=SimpleNamespace(
                    debug_mode_enabled=True,
                    debug_limit_datasets=1,
                    debug_limit_wells=3,
                    report_step_timers=True,
                )
            ),
        )

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        _ = stage_config, unit_workers
        built_targets.append((int(target.dataset_index), str(target.stream_id)))
        return PreprocessInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
        )

    def _fake_run_substage(inputs: PreprocessInputs) -> dict[str, object]:
        return {
            "phase": "save_rec_metadata",
            "preprocess_out_dir": str(tmp_path / f"preprocess_out_{inputs.stream_id}"),
            "summary_json": str(tmp_path / f"save_rec_metadata_{inputs.stream_id}.json"),
            "outputs": {},
        }

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess_save_rec_metadata", _fake_run_substage)

    agg = run_preprocess_save_rec_metadata_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 5
    assert agg.succeeded_targets == 5
    assert agg.failed_targets == 0
    assert built_targets == [
        (0, "well001"),
        (0, "well002"),
        (0, "well003"),
        (0, "well004"),
        (1, "well001"),
    ]


def test_run_preprocess_plot_raster_threshold_from_runtime_ignores_phase_debug_dataset_and_well_limits(
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

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        _ = bundle, materialize_scratch_inputs
        return list(targets)

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        _ = bundle, stage_name
        return StageParallelism(well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_wells=None,
            phases=SimpleNamespace(
                plot_raster_threshold=SimpleNamespace(
                    debug_mode_enabled=True,
                    debug_limit_datasets=1,
                    debug_limit_wells=2,
                    report_step_timers=False,
                )
            ),
        )

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        _ = stage_config, unit_workers
        built_targets.append((int(target.dataset_index), str(target.stream_id)))
        return PreprocessInputs(
            h5_path=target.h5_path,
            stream_id=target.stream_id,
            mea_output_root=target.mea_output_root,
        )

    def _fake_run_substage(inputs: PreprocessInputs) -> dict[str, object]:
        return {
            "phase": "plot_raster_threshold",
            "preprocess_out_dir": str(tmp_path / f"preprocess_out_{inputs.stream_id}"),
            "summary_json": str(tmp_path / f"plot_raster_threshold_{inputs.stream_id}.json"),
            "outputs": {},
        }

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess_plot_raster_threshold", _fake_run_substage)

    agg = run_preprocess_plot_raster_threshold_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 3
    assert agg.succeeded_targets == 3
    assert agg.failed_targets == 0
    assert built_targets == [(0, "well001"), (0, "well002"), (1, "well001")]

