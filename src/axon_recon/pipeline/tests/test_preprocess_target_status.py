from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.runner import (
    run_preprocess_concat_segments_from_runtime,
    run_preprocess_copy_src_to_scratch_from_runtime,
    run_preprocess_from_runtime,
    run_preprocess_prepare_raw_binaries_from_runtime,
    run_preprocess_plot_concat_traces_from_runtime,
    run_preprocess_plot_raster_threshold_from_runtime,
    run_preprocess_plot_segment_traces_from_runtime,
    run_preprocess_preprocess_segments_from_runtime,
    run_preprocess_save_rec_metadata_from_runtime,
    run_preprocess_wipe_src_scratch_from_runtime,
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
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

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
        lambda *, bundle, stage_name: StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12),
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
    assert "well_workers=2 max_stage_workers=24" in messages


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
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=1, unit_workers=8)

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
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

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

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        select_calls.append(bool(materialize_scratch_inputs))
        return [target_a, target_b]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=1,
            phases=SimpleNamespace(copy_src_to_scratch=SimpleNamespace(enabled=False)),
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

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        _ = bundle, materialize_scratch_inputs
        return list(targets)

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        _ = bundle, stage_name
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_datasets=1,
            debug_limit_wells=2,
            phases=SimpleNamespace(copy_src_to_scratch=SimpleNamespace(enabled=False)),
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
            run_preprocess_copy_src_to_scratch_from_runtime,
            "run_preprocess_copy_src_to_scratch_from_runtime",
            "run_preprocess_copy_src_to_scratch",
            "preprocess.copy_src_to_scratch",
            "copy_src_to_scratch",
            1,
            True,
        ),
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
            run_preprocess_prepare_raw_binaries_from_runtime,
            "run_preprocess_prepare_raw_binaries_from_runtime",
            "run_preprocess_prepare_raw_binaries",
            "preprocess.prepare_raw_binaries",
            "prepare_raw_binaries",
            12,
            False,
        ),
        (
            run_preprocess_wipe_src_scratch_from_runtime,
            "run_preprocess_wipe_src_scratch_from_runtime",
            "run_preprocess_wipe_src_scratch",
            "preprocess.wipe_src_scratch",
            "wipe_src_scratch",
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
            run_preprocess_concat_segments_from_runtime,
            "run_preprocess_concat_segments_from_runtime",
            "run_preprocess_concat_segments",
            "preprocess.concat_segments",
            "concat_segments",
            12,
            False,
        ),
        (
            run_preprocess_plot_concat_traces_from_runtime,
            "run_preprocess_plot_concat_traces_from_runtime",
            "run_preprocess_plot_concat_traces",
            "preprocess.plot_concat_traces",
            "plot_concat_traces",
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
    unit_worker_calls: list[int] = []
    divider_stdout_calls: list[bool] = []

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        select_calls.append(bool(materialize_scratch_inputs))
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
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

    agg = runner_fn(config_path=str(tmp_path / "runtime.yml"))

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
    expected_materialize = (stage_name == "preprocess.copy_src_to_scratch")
    assert select_calls == [expected_materialize]
    assert unit_worker_calls == [expected_unit_workers]
    assert divider_stdout_calls == [expected_divider_stdout]


def test_run_preprocess_from_runtime_materializes_inputs_when_copy_phase_enabled(monkeypatch, tmp_path: Path) -> None:
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
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            phases=SimpleNamespace(copy_src_to_scratch=SimpleNamespace(enabled=True)),
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
    assert select_calls == [True]
    assert unit_worker_calls == [1]
    assert divider_stdout_calls == [True]


def test_run_preprocess_from_runtime_passes_debug_limits_to_scratch_materializing_target_selection(
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
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_datasets=None,
            debug_limit_wells=None,
            debug_limit_wells_per_dataset=None,
            phases=SimpleNamespace(copy_src_to_scratch=SimpleNamespace(enabled=True)),
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
    assert select_kwargs[0]["materialize_scratch_inputs"] is True
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
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            phases=SimpleNamespace(
                copy_src_to_scratch=SimpleNamespace(enabled=False),
                preprocess_segments=SimpleNamespace(enabled=True),
                concat_segments=SimpleNamespace(enabled=True),
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
        "Preprocess worker allocation stage=preprocess stage_workers=24 well_workers=2 n_jobs=12 n_jobs_source=derived"
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
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(
            debug_limit_wells=None,
            phase_sequence=("save_rec_metadata",),
            phases=SimpleNamespace(
                copy_src_to_scratch=SimpleNamespace(enabled=True),
                prepare_raw_binaries=SimpleNamespace(enabled=True),
                preprocess_segments=SimpleNamespace(enabled=True),
                concat_segments=SimpleNamespace(enabled=True),
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
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

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
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

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


def test_run_preprocess_concat_segments_from_runtime_ignores_phase_debug_dataset_and_well_limits(
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

    def _fake_select_execution_targets(*, bundle, materialize_scratch_inputs: bool = False):
        _ = bundle, materialize_scratch_inputs
        return list(targets)

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        _ = bundle, stage_name
        return StageParallelism(max_workers=24, max_stage_workers=24, well_workers=2, unit_workers=12)

    def _fake_parse_preprocess_stage_config(**kwargs):
        _ = kwargs
        return SimpleNamespace(
            debug_limit_wells=None,
            phases=SimpleNamespace(
                concat_segments=SimpleNamespace(
                    debug_mode_enabled=True,
                    debug_limit_datasets=1,
                    debug_limit_wells=2,
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
            "phase": "concat_segments",
            "preprocess_out_dir": str(tmp_path / f"preprocess_out_{inputs.stream_id}"),
            "summary_json": str(tmp_path / f"concat_segments_{inputs.stream_id}.json"),
            "outputs": {},
        }

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_preprocess_stage_config", _fake_parse_preprocess_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_preprocess_inputs_for_target", _fake_build_preprocess_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_preprocess_concat_segments", _fake_run_substage)

    agg = run_preprocess_concat_segments_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 4
    assert agg.succeeded_targets == 4
    assert agg.failed_targets == 0
    assert built_targets == [(0, "well001"), (0, "well002"), (0, "well003"), (1, "well001")]
