from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.runner import run_spikesort_from_runtime
from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs
from axon_recon.pipeline.stages.spikesort.models.results import SpikesortResult


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
