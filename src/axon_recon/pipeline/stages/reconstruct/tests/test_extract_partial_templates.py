"""Tests for the templates extract_partial_templates / build_templates split.

Slice 2 of the parallelism migration plan splits the legacy ``build_templates``
phase into two phases:

* ``extract_partial_templates`` — opens each segment analyzer once and writes
  per-(unit, source) partial payloads under
  ``cache/source_payloads/<source>/<unit>/``.
* ``build_templates`` — reads the on-disk partial payloads and merges them per
  unit. It MUST NOT reopen segment analyzers.

These tests exercise the split using lightweight synthetic payloads written to
a tmp_path so we never need real Maxwell H5 inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner
from axon_recon.pipeline.stages.reconstruct.models.inputs import (
	ReconstructionInputs,
	ReconstructionPhasesConfig,
	ReconstructionClearTemplatesCachePhaseConfig,
	ReconstructionGenerateGtrsPhaseConfig,
	ReconstructionPlotBranchPropagationsPhaseConfig,
	ReconstructionPlotBranchVelocitiesPhaseConfig,
	ReconstructionPlotReconsPhaseConfig,
	ReconstructionPlotUnitSummaryPhaseConfig,
	ReconstructionReportFullChipLayoutPhaseConfig,
	ReconstructionReportReconGridPhaseConfig,
	ReconstructionReportReconsPhaseConfig,
	ReconstructionReportSummariesPhaseConfig,
)
from axon_recon.pipeline.stages.reconstruct.runner import (
	DEFAULT_INTERNAL_RECONSTRUCTION_PHASE_SEQUENCE,
	_normalize_reconstruct_stage_phase_name,
	_reconstruct_stage_phase_runner,
	run_reconstruct_templates_build_templates_phase,
	run_reconstruct_templates_extract_partial_templates_phase,
)
from axon_recon.pipeline.stages.reconstruct.templates.io import (
	SOURCE_PAYLOADS_CACHE_RELPATH,
	resolve_materialized_source_payload_unit_dir,
	write_materialized_source_payload,
)
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import (
	TemplateBuildTemplatesPhaseConfig,
	TemplateExtractPartialTemplatesPhaseConfig,
	TemplatesInputs,
	TemplatesPhasesConfig,
)


def test_extract_partial_templates_appears_before_build_templates_in_default_sequence() -> None:
	sequence = DEFAULT_INTERNAL_RECONSTRUCTION_PHASE_SEQUENCE
	assert "templates_extract_partial_templates" in sequence
	assert "templates_build_templates" in sequence
	assert sequence.index("templates_extract_partial_templates") < sequence.index(
		"templates_build_templates"
	)


def test_extract_partial_templates_phase_aliases_normalize() -> None:
	assert (
		_normalize_reconstruct_stage_phase_name("extract_partial_templates")
		== "templates_extract_partial_templates"
	)
	assert (
		_normalize_reconstruct_stage_phase_name("templates.extract_partial_templates")
		== "templates_extract_partial_templates"
	)
	assert (
		_reconstruct_stage_phase_runner("templates_extract_partial_templates")
		is run_reconstruct_templates_extract_partial_templates_phase
	)


def test_extract_partial_templates_runs_before_build_templates(monkeypatch, tmp_path: Path) -> None:
	"""Both phases enabled → extract runs before build."""
	order: list[str] = []
	collected_result = SimpleNamespace(summary_json=tmp_path / "summary.json")

	def _record(phase_name: str):
		def _run(inputs: ReconstructionInputs) -> dict[str, str]:
			order.append(phase_name)
			return {"phase": phase_name, "summary_json": str(tmp_path / f"{phase_name}.json")}

		return _run

	# Replace every phase runner referenced by the canonical sequence so
	# nothing actually executes; we just want to capture order.
	phase_runner_attrs = {
		"templates_resolve_sources": "run_reconstruct_templates_resolve_sources_phase",
		"templates_analyzers": "run_reconstruct_templates_analyzers_phase",
		"templates_extract_partial_templates": "run_reconstruct_templates_extract_partial_templates_phase",
		"templates_build_templates": "run_reconstruct_templates_build_templates_phase",
		"templates_compute_template_similarity": "run_reconstruct_templates_compute_template_similarity_phase",
		"templates_plot_templates_v2": "run_reconstruct_templates_plot_templates_v2_phase",
		"templates_report_templates": "run_reconstruct_templates_report_templates_phase",
		"axon_velocity_gtrs": "run_reconstruct_axon_velocity_gtrs_phase",
		"plot_recons": "run_reconstruct_plot_recons_phase",
		"plot_branch_propagations": "run_reconstruct_plot_branch_propagations_phase",
		"plot_branch_velocities": "run_reconstruct_plot_branch_velocities_phase",
		"plot_unit_summary": "run_reconstruct_plot_unit_summary_phase",
		"report_recons": "run_reconstruct_report_recons_phase",
		"report_recon_grid": "run_reconstruct_report_recon_grid_phase",
		"report_full_chip_layout": "run_reconstruct_report_full_chip_layout_phase",
		"report_summaries": "run_reconstruct_report_summaries_phase",
		"clear_templates_cache": "run_reconstruct_clear_templates_cache_phase",
	}
	for phase_name, attr_name in phase_runner_attrs.items():
		monkeypatch.setattr(reconstruct_runner, attr_name, _record(phase_name))
	monkeypatch.setattr(
		reconstruct_runner, "collect_reconstruct_result_from_outputs", lambda inputs: collected_result
	)

	templates_inputs = TemplatesInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		phases=TemplatesPhasesConfig(
			extract_partial_templates=TemplateExtractPartialTemplatesPhaseConfig(enabled=True),
			build_templates=TemplateBuildTemplatesPhaseConfig(enabled=True),
		),
	)
	inputs = ReconstructionInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		phase_sequence=None,
		templates_inputs=templates_inputs,
		phases=ReconstructionPhasesConfig(
			clear_templates_cache=ReconstructionClearTemplatesCachePhaseConfig(enabled=True),
			axon_velocity_gtrs=ReconstructionGenerateGtrsPhaseConfig(enabled=True),
			plot_recons=ReconstructionPlotReconsPhaseConfig(enabled=True),
			plot_branch_propagations=ReconstructionPlotBranchPropagationsPhaseConfig(enabled=True),
			plot_branch_velocities=ReconstructionPlotBranchVelocitiesPhaseConfig(enabled=True),
			plot_unit_summary=ReconstructionPlotUnitSummaryPhaseConfig(enabled=True),
			report_recons=ReconstructionReportReconsPhaseConfig(enabled=True),
			report_recon_grid=ReconstructionReportReconGridPhaseConfig(enabled=True),
			report_full_chip_layout=ReconstructionReportFullChipLayoutPhaseConfig(enabled=True),
			report_summaries=ReconstructionReportSummariesPhaseConfig(enabled=True),
		),
	)

	reconstruct_runner.run_reconstruct_stage(inputs)

	# extract should appear immediately before build, and both must be present.
	assert "templates_extract_partial_templates" in order
	assert "templates_build_templates" in order
	assert order.index("templates_extract_partial_templates") < order.index(
		"templates_build_templates"
	)


def _write_dummy_partial_payload(
	*,
	templates_out_dir: Path,
	source_name: str,
	unit_id: int,
	channel_count: int = 4,
	waveform_count: int = 7,
) -> Path:
	"""Write a synthetic partial payload to disk in the layout extract produces."""
	template_c_by_t = np.zeros((channel_count, 3), dtype=float)
	template_c_by_t[:, 1] = -float(unit_id) - 1.0  # non-zero peak so merge is happy
	locations_xy = np.asarray(
		[[float(i) * 17.5, 0.0] for i in range(channel_count)], dtype=float
	)
	electrode_ids = np.arange(channel_count, dtype=int)
	channel_ids = np.asarray([f"ch{i}" for i in range(channel_count)], dtype=object)
	write_materialized_source_payload(
		templates_out_dir=templates_out_dir,
		output_rel_root=str(SOURCE_PAYLOADS_CACHE_RELPATH),
		source_name=source_name,
		unit_id=unit_id,
		template_c_by_t=template_c_by_t,
		locations_xy=locations_xy,
		electrode_ids=electrode_ids,
		channel_ids=channel_ids,
		waveform_count=waveform_count,
		sampling_rate_hz=10000.0,
		overlay_waveforms=None,
		top_electrode_id=None,
		total_waveforms_at_channel=None,
	)
	return resolve_materialized_source_payload_unit_dir(
		templates_out_dir=templates_out_dir,
		output_rel_root=str(SOURCE_PAYLOADS_CACHE_RELPATH),
		source_name=source_name,
		unit_id=unit_id,
	)


def test_build_templates_errors_when_payload_root_missing(tmp_path: Path) -> None:
	"""build_templates must NOT bootstrap from analyzers; missing partials → error."""
	from axon_recon.pipeline.stages.reconstruct.phases.build_templates import (
		run_reconstruct_templates_build_templates_phase as run_build_phase_inner,
	)

	templates_inputs = TemplatesInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		phases=TemplatesPhasesConfig(
			extract_partial_templates=TemplateExtractPartialTemplatesPhaseConfig(),
			build_templates=TemplateBuildTemplatesPhaseConfig(),
		),
	)

	with pytest.raises(FileNotFoundError) as excinfo:
		run_build_phase_inner(templates_inputs)

	assert "extract_partial_templates" in str(excinfo.value)
	assert "build_templates" in str(excinfo.value)


def test_build_templates_phase_does_not_call_analyzer_loaders(monkeypatch, tmp_path: Path) -> None:
	"""When partials are on disk, build phase merges without reopening analyzers.

	We monkeypatch every analyzer-load entry point in build_templates and
	confirm none of them get hit.
	"""
	from axon_recon.pipeline.stages.reconstruct.phases import build_templates as build_module

	templates_inputs = TemplatesInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		phases=TemplatesPhasesConfig(
			extract_partial_templates=TemplateExtractPartialTemplatesPhaseConfig(),
			build_templates=TemplateBuildTemplatesPhaseConfig(),
		),
	)

	# Resolve where build_templates would expect to find the payload root, then
	# write two synthetic partial payloads (unit 1, unit 2) under one source.
	context = build_module._resolve_build_templates_context(templates_inputs)
	for unit_id in (1, 2):
		_write_dummy_partial_payload(
			templates_out_dir=context.templates_out_dir,
			source_name="segment_0000",
			unit_id=unit_id,
		)

	# Trip-wires: any of these being called means build_templates is reopening
	# analyzers, which is forbidden by the slice 2 contract.
	def _forbidden(*_args, **_kwargs):
		raise AssertionError(
			"build_templates must not load cached analyzers; it must read partials only"
		)

	monkeypatch.setattr(build_module, "_discover_cached_analyzer_sources", _forbidden)
	monkeypatch.setattr(build_module, "_load_or_create_source_unit_manifests", _forbidden)
	monkeypatch.setattr(build_module, "_materialize_cached_analyzers_by_unit", _forbidden)
	monkeypatch.setattr(build_module, "_materialize_cached_analyzers_by_source", _forbidden)
	monkeypatch.setattr(build_module, "_load_requested_cached_source", _forbidden)

	# Stub the merge call to avoid invoking real spike-interface paths.
	captured: dict[str, object] = {}

	def _fake_build(*, inputs, well_out_dir, templates_out_dir, unit_ids, source_names, payload_root, payload_materialization_mode, payload_loader):
		captured["payload_materialization_mode"] = payload_materialization_mode
		captured["unit_ids"] = list(unit_ids)
		captured["source_names"] = list(source_names)
		captured["payload_root"] = payload_root
		return {
			"phase": "build_templates",
			"stream_id": str(inputs.stream_id),
			"unit_count": len(list(unit_ids)),
			"built_units": list(unit_ids),
			"reused_units": [],
			"skipped_units": [],
			"source_count": len(list(source_names)),
		}

	monkeypatch.setattr(build_module, "build_templates_phase_from_unit_payloads", _fake_build)

	# The label filter triggers if any spikesort labels are configured. Disable.
	templates_inputs = TemplatesInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		unit_label_filter_labels=(),
		unit_label_filter_required=False,
		phases=TemplatesPhasesConfig(
			extract_partial_templates=TemplateExtractPartialTemplatesPhaseConfig(),
			build_templates=TemplateBuildTemplatesPhaseConfig(),
		),
	)

	summary = build_module.run_reconstruct_templates_build_templates_phase(templates_inputs)

	assert captured["payload_materialization_mode"] == "partial_payloads"
	# Both partials should have been discovered from disk.
	assert sorted(int(u) for u in captured["unit_ids"]) == [1, 2]
	assert captured["source_names"] == ["segment_0000"]
	# Summary written and contains the synthesized fields.
	assert summary["phase"] == "build_templates"
	assert summary["unit_count"] == 2
	# A summary file should also exist on disk.
	assert Path(summary["summary_json"]).exists()


def test_extract_partial_templates_summary_written(monkeypatch, tmp_path: Path) -> None:
	"""extract phase writes a summary JSON listing the materialized units/sources."""
	from axon_recon.pipeline.stages.reconstruct.phases import (
		extract_partial_templates as extract_module,
	)
	from axon_recon.pipeline.stages.reconstruct.phases import build_templates as build_module

	templates_inputs = TemplatesInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		unit_label_filter_labels=(),
		unit_label_filter_required=False,
		phases=TemplatesPhasesConfig(
			extract_partial_templates=TemplateExtractPartialTemplatesPhaseConfig(),
			build_templates=TemplateBuildTemplatesPhaseConfig(),
		),
	)

	context = build_module._resolve_build_templates_context(templates_inputs)

	# Force the analyzer discovery + materialization helpers to behave as if
	# there is exactly one cached segment source with two units.
	def _fake_discover(*, inputs, context):
		analyzer_cache_dir = context.templates_out_dir / "fake_analyzer_cache"
		analyzer_cache_dir.mkdir(parents=True, exist_ok=True)
		return context.well_out_dir, analyzer_cache_dir, ["segment_0000"]

	def _fake_load_manifests(**_kwargs):
		return {"segment_0000": [1, 2]}

	def _fake_materialize_by_source(**_kwargs):
		# Pretend the by-source materializer wrote payloads to disk (we'll
		# write them here so the summary captures real values).
		for unit_id in (1, 2):
			_write_dummy_partial_payload(
				templates_out_dir=context.templates_out_dir,
				source_name="segment_0000",
				unit_id=unit_id,
			)
		return [1, 2], {
			"segment_0000": {
				"units_materialized": [1, 2],
				"units_reused": [],
				"units_skipped_absent": [],
				"unit_count": 2,
			}
		}

	monkeypatch.setattr(extract_module, "_discover_cached_analyzer_sources", _fake_discover)
	monkeypatch.setattr(extract_module, "_load_or_create_source_unit_manifests", _fake_load_manifests)
	monkeypatch.setattr(extract_module, "_materialize_cached_analyzers_by_source", _fake_materialize_by_source)

	summary = extract_module.run_reconstruct_templates_extract_partial_templates_phase(
		templates_inputs
	)

	assert summary["phase"] == "extract_partial_templates"
	assert summary["source_count"] == 1
	assert summary["source_names"] == ["segment_0000"]
	summary_json_path = Path(summary["summary_json"])
	assert summary_json_path.exists()
	on_disk = json.loads(summary_json_path.read_text(encoding="utf-8"))
	assert on_disk["phase"] == "extract_partial_templates"
	assert on_disk["source_count"] == 1
