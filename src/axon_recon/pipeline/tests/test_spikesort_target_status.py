from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.runner import (
    run_spikesort_bombcell_label_from_runtime,
    run_spikesort_from_runtime,
    run_spikesort_merge_from_runtime,
    run_spikesort_sort_from_runtime,
    run_spikesort_summarize_sort_from_runtime,
)
from axon_recon.pipeline.stages.spikesort.models.inputs import SpikesortInputs
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
