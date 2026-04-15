from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.runner import (
    run_reconstruct_from_runtime,
    run_reconstruct_generate_gtrs_from_runtime,
    run_reconstruct_plot_recons_from_runtime,
    run_reconstruct_report_recons_from_runtime,
)
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs
from axon_recon.pipeline.stages.reconstruct.models.results import ReconstructionResult, UnitReconstructionResult


def test_run_reconstruct_from_runtime_marks_target_ok_when_any_unit_succeeds(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = ReconstructionInputs(
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

    def _fake_parse_reconstruction_stage_config(**kwargs):
        return object()

    def _fake_build_reconstruction_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
        _ = probe_geometry
        return dummy_inputs

    def _fake_run_reconstruct(inputs: ReconstructionInputs) -> ReconstructionResult:
        return ReconstructionResult(
            well_out_dir=tmp_path / "well_out",
            reconstruction_out_dir=tmp_path / "recon_out",
            summary_json=tmp_path / "summary.json",
            units=[
                UnitReconstructionResult(
                    unit_id=93,
                    status="ok",
                    outputs={"circle_recon_png": "ok.png"},
                    error=None,
                ),
                UnitReconstructionResult(
                    unit_id=94,
                    status="error",
                    outputs={},
                    error="No branches found",
                )
            ],
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", lambda *, data_config: None)
    monkeypatch.setattr(pipeline_runner, "parse_reconstruction_stage_config", _fake_parse_reconstruction_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_reconstruction_inputs_for_target", _fake_build_reconstruction_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_reconstruct", _fake_run_reconstruct)

    agg = run_reconstruct_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].status == "ok"
    assert agg.target_results[0].result is not None


def test_run_reconstruct_from_runtime_marks_target_error_when_no_units_succeed(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = ReconstructionInputs(
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

    def _fake_parse_reconstruction_stage_config(**kwargs):
        return object()

    def _fake_build_reconstruction_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
        _ = probe_geometry
        return dummy_inputs

    def _fake_run_reconstruct(inputs: ReconstructionInputs) -> ReconstructionResult:
        return ReconstructionResult(
            well_out_dir=tmp_path / "well_out",
            reconstruction_out_dir=tmp_path / "recon_out",
            summary_json=tmp_path / "summary.json",
            units=[
                UnitReconstructionResult(
                    unit_id=94,
                    status="error",
                    outputs={},
                    error="No branches found",
                )
            ],
        )

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", lambda *, data_config: None)
    monkeypatch.setattr(pipeline_runner, "parse_reconstruction_stage_config", _fake_parse_reconstruction_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_reconstruction_inputs_for_target", _fake_build_reconstruction_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_reconstruct", _fake_run_reconstruct)

    agg = run_reconstruct_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 0
    assert agg.failed_targets == 1
    assert agg.target_results[0].status == "error"
    assert "No branches found" in str(agg.target_results[0].error)


@pytest.mark.parametrize(
    ("wrapper", "runner_attr", "expected_stage", "phase_name"),
    [
        (run_reconstruct_generate_gtrs_from_runtime, "run_reconstruct_generate_gtrs", "reconstruct.generate_gtrs", "generate_gtrs"),
        (run_reconstruct_plot_recons_from_runtime, "run_reconstruct_plot_recons", "reconstruct.plot_recons", "plot_recons"),
        (run_reconstruct_report_recons_from_runtime, "run_reconstruct_report_recons", "reconstruct.report_recons", "report_recons"),
    ],
)
def test_run_reconstruct_phase_from_runtime_marks_target_ok(
    monkeypatch,
    tmp_path: Path,
    wrapper,
    runner_attr: str,
    expected_stage: str,
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

    dummy_inputs = ReconstructionInputs(
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

    def _fake_parse_reconstruction_stage_config(**kwargs):
        return object()

    def _fake_build_reconstruction_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
        _ = probe_geometry
        return dummy_inputs

    def _fake_phase_runner(inputs: ReconstructionInputs):
        assert inputs is dummy_inputs
        return {
            "phase": phase_name,
            "reconstruction_out_dir": str(tmp_path / "recon_out"),
            "summary_json": str(tmp_path / f"{phase_name}_summary.json"),
            "units_ok": 1,
            "units_error": 0,
            "units": [{"unit_id": 94, "status": "ok", "outputs": {}, "error": None}],
        }

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", lambda *, data_config: None)
    monkeypatch.setattr(pipeline_runner, "parse_reconstruction_stage_config", _fake_parse_reconstruction_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_reconstruction_inputs_for_target", _fake_build_reconstruction_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, runner_attr, _fake_phase_runner)

    agg = wrapper(config_path=str(tmp_path / "runtime.yml"))

    assert agg.stage == expected_stage
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].status == "ok"
    assert agg.target_results[0].result == {
        "phase": phase_name,
        "reconstruction_out_dir": str(tmp_path / "recon_out"),
        "summary_json": str(tmp_path / f"{phase_name}_summary.json"),
        "units_ok": 1,
        "units_error": 0,
        "units": [{"unit_id": 94, "status": "ok", "outputs": {}, "error": None}],
    }


def test_run_reconstruct_plot_recons_from_runtime_marks_target_error_when_no_units_succeed(monkeypatch, tmp_path: Path) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    target = ExecutionTarget(
        dataset_index=0,
        dataset_id="dataset_000:test.h5",
        h5_path=tmp_path / "test.h5",
        stream_id="well001",
        mea_output_root=tmp_path,
    )

    dummy_inputs = ReconstructionInputs(
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

    def _fake_parse_reconstruction_stage_config(**kwargs):
        return object()

    def _fake_build_reconstruction_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
        _ = probe_geometry
        return dummy_inputs

    def _fake_run_reconstruct_plot_recons(inputs: ReconstructionInputs):
        return {
            "phase": "plot_recons",
            "reconstruction_out_dir": str(tmp_path / "recon_out"),
            "summary_json": str(tmp_path / "plot_recons_summary.json"),
            "units_ok": 0,
            "units_error": 1,
            "units": [{"unit_id": 94, "status": "error", "outputs": {}, "error": "No branches found"}],
        }

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", lambda *, data_config: None)
    monkeypatch.setattr(pipeline_runner, "parse_reconstruction_stage_config", _fake_parse_reconstruction_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_reconstruction_inputs_for_target", _fake_build_reconstruction_inputs_for_target)
    monkeypatch.setattr(pipeline_runner, "run_reconstruct_plot_recons", _fake_run_reconstruct_plot_recons)

    agg = run_reconstruct_plot_recons_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.total_targets == 1
    assert agg.succeeded_targets == 0
    assert agg.failed_targets == 1
    assert agg.target_results[0].status == "error"
    assert "No branches found" in str(agg.target_results[0].error)
