from __future__ import annotations

import logging
from pathlib import Path

import pytest

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.runner import (
    run_reconstruct_clear_templates_cache_from_runtime,
    run_reconstruct_templates_analyzers_from_runtime,
    run_reconstruct_templates_build_templates_from_runtime,
    run_reconstruct_templates_compute_template_similarity_from_runtime,
    run_reconstruct_templates_extract_template_segments_from_runtime,
    run_reconstruct_templates_plot_templates_from_runtime,
    run_reconstruct_templates_report_templates_from_runtime,
    run_reconstruct_templates_reports_from_runtime,
    run_reconstruct_templates_resolve_sources_from_runtime,
    run_reconstruct_from_runtime,
    run_reconstruct_generate_gtrs_from_runtime,
    run_reconstruct_plot_branch_propagations_from_runtime,
    run_reconstruct_plot_branch_velocities_from_runtime,
    run_reconstruct_plot_unit_summary_from_runtime,
    run_reconstruct_plot_recons_from_runtime,
    run_reconstruct_report_full_chip_layout_from_runtime,
    run_reconstruct_report_recon_grid_from_runtime,
    run_reconstruct_report_recons_from_runtime,
    run_reconstruct_report_summaries_from_runtime,
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


def test_run_reconstruct_from_runtime_logs_stage_topology(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
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

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", lambda *, config_path: _DummyBundle())
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", lambda *, bundle: [target])
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_stage_parallelism",
        lambda *, bundle, stage_name: StageParallelism(max_workers=6, max_stage_workers=6, well_workers=2, unit_workers=3),
    )
    monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", lambda *, data_config: None)
    monkeypatch.setattr(pipeline_runner, "parse_reconstruction_stage_config", lambda **kwargs: object())
    monkeypatch.setattr(
        pipeline_runner,
        "build_reconstruction_inputs_for_target",
        lambda *, target, stage_config, unit_workers, probe_geometry: dummy_inputs,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "run_reconstruct",
        lambda inputs: ReconstructionResult(
            well_out_dir=tmp_path / "well_out",
            reconstruction_out_dir=tmp_path / "recon_out",
            summary_json=tmp_path / "summary.json",
            units=[UnitReconstructionResult(unit_id=93, status="ok", outputs={}, error=None)],
        ),
    )

    with caplog.at_level(logging.INFO):
        run_reconstruct_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    messages = [record.getMessage() for record in caplog.records]
    assert "Starting stage: reconstruct" in messages
    assert "Execution topology: stage_global_order=true, well_local_phase_sequence=true" in messages
    assert "Selected wells: 1" in messages
    assert "well_workers=2 max_stage_workers=6" in messages


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
        (
            run_reconstruct_plot_branch_propagations_from_runtime,
            "run_reconstruct_plot_branch_propagations",
            "reconstruct.plot_branch_propagations",
            "plot_branch_propagations",
        ),
        (
            run_reconstruct_plot_branch_velocities_from_runtime,
            "run_reconstruct_plot_branch_velocities",
            "reconstruct.plot_branch_velocities",
            "plot_branch_velocities",
        ),
        (
            run_reconstruct_plot_unit_summary_from_runtime,
            "run_reconstruct_plot_unit_summary",
            "reconstruct.plot_unit_summary",
            "plot_unit_summary",
        ),
        (
            run_reconstruct_report_full_chip_layout_from_runtime,
            "run_reconstruct_report_full_chip_layout",
            "reconstruct.report_full_chip_layout",
            "report_full_chip_layout",
        ),
        (run_reconstruct_report_recons_from_runtime, "run_reconstruct_report_recons", "reconstruct.report_recons", "report_recons"),
        (
            run_reconstruct_report_recon_grid_from_runtime,
            "run_reconstruct_report_recon_grid",
            "reconstruct.report_recon_grid",
            "report_recon_grid",
        ),
        (
            run_reconstruct_report_summaries_from_runtime,
            "run_reconstruct_report_summaries",
            "reconstruct.report_summaries",
            "report_summaries",
        ),
        (
            run_reconstruct_clear_templates_cache_from_runtime,
            "run_reconstruct_clear_templates_cache",
            "reconstruct.clear_templates_cache",
            "clear_templates_cache",
        ),
        (
            run_reconstruct_templates_resolve_sources_from_runtime,
            "run_reconstruct_templates_resolve_sources",
            "reconstruct.resolve_sources",
            "resolve_sources",
        ),
        (
            run_reconstruct_templates_analyzers_from_runtime,
            "run_reconstruct_templates_analyzers",
            "reconstruct.analyzers",
            "analyzers",
        ),
        (
            run_reconstruct_templates_extract_template_segments_from_runtime,
            "run_reconstruct_templates_extract_template_segments",
            "reconstruct.extract_template_segments",
            "extract_template_segments",
        ),
        (
            run_reconstruct_templates_build_templates_from_runtime,
            "run_reconstruct_templates_build_templates",
            "reconstruct.build_templates",
            "build_templates",
        ),
        (
            run_reconstruct_templates_compute_template_similarity_from_runtime,
            "run_reconstruct_templates_compute_template_similarity",
            "reconstruct.compute_template_similarity",
            "compute_template_similarity",
        ),
        (
            run_reconstruct_templates_plot_templates_from_runtime,
            "run_reconstruct_templates_plot_templates",
            "reconstruct.plot_templates",
            "plot_templates",
        ),
        (
            run_reconstruct_templates_report_templates_from_runtime,
            "run_reconstruct_templates_report_templates",
            "reconstruct.report_templates",
            "report_templates",
        ),
        (
            run_reconstruct_templates_reports_from_runtime,
            "run_reconstruct_templates_reports",
            "reconstruct.reports",
            "reports",
        ),
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


@pytest.mark.parametrize(
    ("wrapper", "expected_stage"),
    [
        (run_reconstruct_templates_resolve_sources_from_runtime, "reconstruct.resolve_sources"),
        (run_reconstruct_templates_analyzers_from_runtime, "reconstruct.analyzers"),
        (run_reconstruct_templates_extract_template_segments_from_runtime, "reconstruct.extract_template_segments"),
        (run_reconstruct_templates_build_templates_from_runtime, "reconstruct.build_templates"),
        (run_reconstruct_templates_compute_template_similarity_from_runtime, "reconstruct.compute_template_similarity"),
        (run_reconstruct_templates_plot_templates_from_runtime, "reconstruct.plot_templates"),
        (run_reconstruct_templates_report_templates_from_runtime, "reconstruct.report_templates"),
        (run_reconstruct_templates_reports_from_runtime, "reconstruct.reports"),
        (run_reconstruct_generate_gtrs_from_runtime, "reconstruct.generate_gtrs"),
        (run_reconstruct_plot_recons_from_runtime, "reconstruct.plot_recons"),
        (run_reconstruct_plot_branch_propagations_from_runtime, "reconstruct.plot_branch_propagations"),
        (run_reconstruct_plot_branch_velocities_from_runtime, "reconstruct.plot_branch_velocities"),
        (run_reconstruct_plot_unit_summary_from_runtime, "reconstruct.plot_unit_summary"),
        (run_reconstruct_report_recons_from_runtime, "reconstruct.report_recons"),
        (run_reconstruct_report_recon_grid_from_runtime, "reconstruct.report_recon_grid"),
        (run_reconstruct_report_full_chip_layout_from_runtime, "reconstruct.report_full_chip_layout"),
        (run_reconstruct_report_summaries_from_runtime, "reconstruct.report_summaries"),
        (run_reconstruct_clear_templates_cache_from_runtime, "reconstruct.clear_templates_cache"),
    ],
)
def test_reconstruct_phase_wrappers_forward_dataset_and_well_limits(
    monkeypatch,
    tmp_path: Path,
    wrapper,
    expected_stage: str,
) -> None:
    import axon_recon.pipeline.runner as pipeline_runner

    seen: dict[str, object] = {}
    sentinel = object()

    def _fake_substage(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(pipeline_runner, "_run_reconstruct_substage_from_runtime", _fake_substage)

    result = wrapper(
        config_path=str(tmp_path / "runtime.yml"),
        limit_segments_override=2,
        limit_datasets_override=3,
        limit_wells_per_dataset_override=1,
    )

    assert result is sentinel
    assert seen["stage_name"] == expected_stage
    assert seen["limit_segments_override"] == 2
    assert seen["limit_datasets_override"] == 3
    assert seen["limit_wells_per_dataset_override"] == 1


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


def test_run_reconstruct_clear_templates_cache_from_runtime_marks_target_ok_when_skipped(
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

    def _fake_run_reconstruct_clear_templates_cache(inputs: ReconstructionInputs):
        assert inputs is dummy_inputs
        return {
            "phase": "clear_templates_cache",
            "summary_json": str(tmp_path / "clear_templates_cache_summary.json"),
            "skipped": True,
            "reason": "disabled",
        }

    monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
    monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
    monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
    monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", lambda *, data_config: None)
    monkeypatch.setattr(pipeline_runner, "parse_reconstruction_stage_config", _fake_parse_reconstruction_stage_config)
    monkeypatch.setattr(pipeline_runner, "build_reconstruction_inputs_for_target", _fake_build_reconstruction_inputs_for_target)
    monkeypatch.setattr(
        pipeline_runner,
        "run_reconstruct_clear_templates_cache",
        _fake_run_reconstruct_clear_templates_cache,
    )

    agg = run_reconstruct_clear_templates_cache_from_runtime(config_path=str(tmp_path / "runtime.yml"))

    assert agg.stage == "reconstruct.clear_templates_cache"
    assert agg.total_targets == 1
    assert agg.succeeded_targets == 1
    assert agg.failed_targets == 0
    assert agg.target_results[0].status == "ok"
    assert agg.target_results[0].result == {
        "phase": "clear_templates_cache",
        "summary_json": str(tmp_path / "clear_templates_cache_summary.json"),
        "skipped": True,
        "reason": "disabled",
    }
