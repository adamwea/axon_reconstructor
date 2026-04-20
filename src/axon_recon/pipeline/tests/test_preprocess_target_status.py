from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.runner import (
    run_preprocess_concatenate_preprocessed_recordings_from_runtime,
    run_preprocess_copy_src_to_scratch_from_runtime,
    run_preprocess_from_runtime,
    run_preprocess_preprocess_segments_from_runtime,
    run_preprocess_save_rec_metadata_from_runtime,
    run_preprocess_save_common_electrodes_from_runtime,
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

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        return dummy_inputs

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
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

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        return dummy_inputs

    def _fake_run_preprocess(inputs: PreprocessInputs) -> PreprocessResult:
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

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target_a, target_b]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=1)

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


@pytest.mark.parametrize(
    ("runner_fn", "runner_symbol", "api_symbol", "stage_name", "phase_name"),
    [
        (
            run_preprocess_copy_src_to_scratch_from_runtime,
            "run_preprocess_copy_src_to_scratch_from_runtime",
            "run_preprocess_copy_src_to_scratch",
            "preprocess.copy_src_to_scratch",
            "copy_src_to_scratch",
        ),
        (
            run_preprocess_save_rec_metadata_from_runtime,
            "run_preprocess_save_rec_metadata_from_runtime",
            "run_preprocess_save_rec_metadata",
            "preprocess.save_rec_metadata",
            "save_rec_metadata",
        ),
        (
            run_preprocess_wipe_src_scratch_from_runtime,
            "run_preprocess_wipe_src_scratch_from_runtime",
            "run_preprocess_wipe_src_scratch",
            "preprocess.wipe_src_scratch",
            "wipe_src_scratch",
        ),
        (
            run_preprocess_preprocess_segments_from_runtime,
            "run_preprocess_preprocess_segments_from_runtime",
            "run_preprocess_preprocess_segments",
            "preprocess.preprocess_segments",
            "preprocess_segments",
        ),
        (
            run_preprocess_concatenate_preprocessed_recordings_from_runtime,
            "run_preprocess_concatenate_preprocessed_recordings_from_runtime",
            "run_preprocess_concatenate_preprocessed_recordings",
            "preprocess.concatenate_preprocessed_recordings",
            "concatenate_preprocessed_recordings",
        ),
        (
            run_preprocess_save_common_electrodes_from_runtime,
            "run_preprocess_save_common_electrodes_from_runtime",
            "run_preprocess_save_common_electrodes",
            "preprocess.save_common_electrodes",
            "save_common_electrodes",
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

    def _fake_load_pipeline_runtime_bundle(*, config_path: str):
        return _DummyBundle()

    def _fake_select_execution_targets(*, bundle):
        return [target]

    def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
        return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

    def _fake_parse_preprocess_stage_config(**kwargs):
        return SimpleNamespace(debug_limit_wells=None)

    def _fake_build_preprocess_inputs_for_target(*, target, stage_config, unit_workers: int):
        return dummy_inputs

    def _fake_run_substage(inputs: PreprocessInputs) -> dict[str, object]:
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
