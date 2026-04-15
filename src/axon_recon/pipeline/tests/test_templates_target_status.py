from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from axon_recon.pipeline.execution.context import ExecutionTarget, StageParallelism
from axon_recon.pipeline.runner import (
	run_templates_analyzers_from_runtime,
	run_templates_build_templates_from_runtime,
	run_templates_plot_templates_from_runtime,
	run_templates_report_templates_from_runtime,
	run_templates_reports_overlays_from_runtime,
)
from axon_recon.pipeline.stages.templates.models.inputs import TemplatesInputs


def _target(tmp_path: Path) -> ExecutionTarget:
	return ExecutionTarget(
		dataset_index=0,
		dataset_id="dataset_000:test.h5",
		h5_path=tmp_path / "test.h5",
		stream_id="well001",
		mea_output_root=tmp_path,
	)


def _dummy_inputs(target: ExecutionTarget) -> TemplatesInputs:
	return TemplatesInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
	)


def test_run_templates_analyzers_from_runtime_marks_target_ok(monkeypatch, tmp_path: Path) -> None:
	import axon_recon.pipeline.runner as pipeline_runner

	target = _target(tmp_path)
	dummy_inputs = _dummy_inputs(target)

	class _DummyBundle:
		runtime_config = object()
		data_config = object()

	def _fake_load_pipeline_runtime_bundle(*, config_path: str):
		return _DummyBundle()

	def _fake_select_execution_targets(*, bundle):
		return [target]

	def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
		return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

	def _fake_parse_probe_geometry_from_data_config(*, data_config):
		return None

	def _fake_parse_templates_stage_config(**kwargs):
		return SimpleNamespace(debug_limit_wells=None)

	def _fake_build_templates_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
		return dummy_inputs

	def _fake_run_templates_analyzers(inputs: TemplatesInputs, *, source_scope: str | None = None):
		assert inputs is dummy_inputs
		assert source_scope is None
		return {
			"phase": "analyzers",
			"templates_out_dir": str(tmp_path / "templates_out"),
			"summary_json": str(tmp_path / "analyzers_summary.json"),
		}

	monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
	monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
	monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
	monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", _fake_parse_probe_geometry_from_data_config)
	monkeypatch.setattr(pipeline_runner, "parse_templates_stage_config", _fake_parse_templates_stage_config)
	monkeypatch.setattr(pipeline_runner, "build_templates_inputs_for_target", _fake_build_templates_inputs_for_target)
	monkeypatch.setattr(pipeline_runner, "run_templates_analyzers", _fake_run_templates_analyzers)

	agg = run_templates_analyzers_from_runtime(config_path=str(tmp_path / "runtime.yml"))

	assert agg.stage == "templates.analyzers"
	assert agg.total_targets == 1
	assert agg.succeeded_targets == 1
	assert agg.failed_targets == 0
	assert agg.target_results[0].status == "ok"
	assert agg.target_results[0].result == {
		"phase": "analyzers",
		"templates_out_dir": str(tmp_path / "templates_out"),
		"summary_json": str(tmp_path / "analyzers_summary.json"),
	}


def test_run_templates_reports_overlays_from_runtime_uses_leaf_scope(monkeypatch, tmp_path: Path) -> None:
	import axon_recon.pipeline.runner as pipeline_runner

	target = _target(tmp_path)
	dummy_inputs = _dummy_inputs(target)
	calls: list[str | None] = []

	class _DummyBundle:
		runtime_config = object()
		data_config = object()

	def _fake_load_pipeline_runtime_bundle(*, config_path: str):
		return _DummyBundle()

	def _fake_select_execution_targets(*, bundle):
		return [target]

	def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
		return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

	def _fake_parse_probe_geometry_from_data_config(*, data_config):
		return None

	def _fake_parse_templates_stage_config(**kwargs):
		return SimpleNamespace(debug_limit_wells=None)

	def _fake_build_templates_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
		return dummy_inputs

	def _fake_run_templates_reports(inputs: TemplatesInputs, *, report_scope: str | None = None):
		assert inputs is dummy_inputs
		calls.append(report_scope)
		return {
			"phase": f"reports.{report_scope}",
			"templates_out_dir": str(tmp_path / "templates_out"),
			"summary_json": str(tmp_path / "reports_summary.json"),
		}

	monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
	monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
	monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
	monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", _fake_parse_probe_geometry_from_data_config)
	monkeypatch.setattr(pipeline_runner, "parse_templates_stage_config", _fake_parse_templates_stage_config)
	monkeypatch.setattr(pipeline_runner, "build_templates_inputs_for_target", _fake_build_templates_inputs_for_target)
	monkeypatch.setattr(pipeline_runner, "run_templates_reports", _fake_run_templates_reports)

	agg = run_templates_reports_overlays_from_runtime(config_path=str(tmp_path / "runtime.yml"))

	assert calls == ["overlays"]
	assert agg.stage == "templates.reports.overlays"
	assert agg.total_targets == 1
	assert agg.succeeded_targets == 1
	assert agg.failed_targets == 0
	assert agg.target_results[0].status == "ok"
	assert agg.target_results[0].result == {
		"phase": "reports.overlays",
		"templates_out_dir": str(tmp_path / "templates_out"),
		"summary_json": str(tmp_path / "reports_summary.json"),
	}


def test_run_templates_build_templates_from_runtime_marks_target_ok(monkeypatch, tmp_path: Path) -> None:
	import axon_recon.pipeline.runner as pipeline_runner

	target = _target(tmp_path)
	dummy_inputs = _dummy_inputs(target)

	class _DummyBundle:
		runtime_config = object()
		data_config = object()

	def _fake_load_pipeline_runtime_bundle(*, config_path: str):
		return _DummyBundle()

	def _fake_select_execution_targets(*, bundle):
		return [target]

	def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
		return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

	def _fake_parse_probe_geometry_from_data_config(*, data_config):
		return None

	def _fake_parse_templates_stage_config(**kwargs):
		return SimpleNamespace(debug_limit_wells=None)

	def _fake_build_templates_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
		return dummy_inputs

	def _fake_run_templates_build_templates(inputs: TemplatesInputs):
		assert inputs is dummy_inputs
		return {
			"phase": "build_templates",
			"templates_out_dir": str(tmp_path / "templates_out"),
			"summary_json": str(tmp_path / "build_templates_summary.json"),
		}

	monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
	monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
	monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
	monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", _fake_parse_probe_geometry_from_data_config)
	monkeypatch.setattr(pipeline_runner, "parse_templates_stage_config", _fake_parse_templates_stage_config)
	monkeypatch.setattr(pipeline_runner, "build_templates_inputs_for_target", _fake_build_templates_inputs_for_target)
	monkeypatch.setattr(pipeline_runner, "run_templates_build_templates", _fake_run_templates_build_templates)

	agg = run_templates_build_templates_from_runtime(config_path=str(tmp_path / "runtime.yml"))

	assert agg.stage == "templates.build_templates"
	assert agg.total_targets == 1
	assert agg.succeeded_targets == 1
	assert agg.failed_targets == 0
	assert agg.target_results[0].status == "ok"
	assert agg.target_results[0].result == {
		"phase": "build_templates",
		"templates_out_dir": str(tmp_path / "templates_out"),
		"summary_json": str(tmp_path / "build_templates_summary.json"),
	}


def test_run_templates_plot_templates_from_runtime_marks_target_ok(monkeypatch, tmp_path: Path) -> None:
	import axon_recon.pipeline.runner as pipeline_runner

	target = _target(tmp_path)
	dummy_inputs = _dummy_inputs(target)

	class _DummyBundle:
		runtime_config = object()
		data_config = object()

	def _fake_load_pipeline_runtime_bundle(*, config_path: str):
		return _DummyBundle()

	def _fake_select_execution_targets(*, bundle):
		return [target]

	def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
		return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

	def _fake_parse_probe_geometry_from_data_config(*, data_config):
		return None

	def _fake_parse_templates_stage_config(**kwargs):
		return SimpleNamespace(debug_limit_wells=None)

	def _fake_build_templates_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
		return dummy_inputs

	def _fake_run_templates_plot_templates(inputs: TemplatesInputs):
		assert inputs is dummy_inputs
		return {
			"phase": "plot_templates",
			"templates_out_dir": str(tmp_path / "templates_out"),
			"summary_json": str(tmp_path / "plot_templates_summary.json"),
		}

	monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
	monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
	monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
	monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", _fake_parse_probe_geometry_from_data_config)
	monkeypatch.setattr(pipeline_runner, "parse_templates_stage_config", _fake_parse_templates_stage_config)
	monkeypatch.setattr(pipeline_runner, "build_templates_inputs_for_target", _fake_build_templates_inputs_for_target)
	monkeypatch.setattr(pipeline_runner, "run_templates_plot_templates", _fake_run_templates_plot_templates)

	agg = run_templates_plot_templates_from_runtime(config_path=str(tmp_path / "runtime.yml"))

	assert agg.stage == "templates.plot_templates"
	assert agg.total_targets == 1
	assert agg.succeeded_targets == 1
	assert agg.failed_targets == 0
	assert agg.target_results[0].status == "ok"
	assert agg.target_results[0].result == {
		"phase": "plot_templates",
		"templates_out_dir": str(tmp_path / "templates_out"),
		"summary_json": str(tmp_path / "plot_templates_summary.json"),
	}


def test_run_templates_report_templates_from_runtime_marks_target_ok(monkeypatch, tmp_path: Path) -> None:
	import axon_recon.pipeline.runner as pipeline_runner

	target = _target(tmp_path)
	dummy_inputs = _dummy_inputs(target)

	class _DummyBundle:
		runtime_config = object()
		data_config = object()

	def _fake_load_pipeline_runtime_bundle(*, config_path: str):
		return _DummyBundle()

	def _fake_select_execution_targets(*, bundle):
		return [target]

	def _fake_resolve_stage_parallelism(*, bundle, stage_name: str):
		return StageParallelism(max_workers=1, max_stage_workers=1, well_workers=1, unit_workers=1)

	def _fake_parse_probe_geometry_from_data_config(*, data_config):
		return None

	def _fake_parse_templates_stage_config(**kwargs):
		return SimpleNamespace(debug_limit_wells=None)

	def _fake_build_templates_inputs_for_target(*, target, stage_config, unit_workers: int, probe_geometry):
		return dummy_inputs

	def _fake_run_templates_report_templates(inputs: TemplatesInputs):
		assert inputs is dummy_inputs
		return {
			"phase": "report_templates",
			"templates_out_dir": str(tmp_path / "templates_out"),
			"summary_json": str(tmp_path / "report_templates_summary.json"),
		}

	monkeypatch.setattr(pipeline_runner, "load_pipeline_runtime_bundle", _fake_load_pipeline_runtime_bundle)
	monkeypatch.setattr(pipeline_runner, "select_execution_targets", _fake_select_execution_targets)
	monkeypatch.setattr(pipeline_runner, "resolve_stage_parallelism", _fake_resolve_stage_parallelism)
	monkeypatch.setattr(pipeline_runner, "parse_probe_geometry_from_data_config", _fake_parse_probe_geometry_from_data_config)
	monkeypatch.setattr(pipeline_runner, "parse_templates_stage_config", _fake_parse_templates_stage_config)
	monkeypatch.setattr(pipeline_runner, "build_templates_inputs_for_target", _fake_build_templates_inputs_for_target)
	monkeypatch.setattr(pipeline_runner, "run_templates_report_templates", _fake_run_templates_report_templates)

	agg = run_templates_report_templates_from_runtime(config_path=str(tmp_path / "runtime.yml"))

	assert agg.stage == "templates.report_templates"
	assert agg.total_targets == 1
	assert agg.succeeded_targets == 1
	assert agg.failed_targets == 0
	assert agg.target_results[0].status == "ok"
	assert agg.target_results[0].result == {
		"phase": "report_templates",
		"templates_out_dir": str(tmp_path / "templates_out"),
		"summary_json": str(tmp_path / "report_templates_summary.json"),
	}
