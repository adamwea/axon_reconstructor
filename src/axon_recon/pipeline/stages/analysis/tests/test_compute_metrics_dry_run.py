"""Tests for `analysis.compute_metrics` phase dry-run short-circuit.

When `get_dry_run_override()` is True, the phase writes a `dry_run_ok`
manifest at the standard location WITHOUT scanning recon_outputs/units/
or building any parquet tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def test_compute_metrics_dry_run_writes_manifest_and_skips_scan(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override
	from axon_recon.pipeline.stages.analysis import runner as analysis_runner

	well_out_dir = tmp_path / "well"

	monkeypatch.setattr(
		analysis_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)

	# These should NOT be called during dry-run.
	scan_called = {"count": 0}

	def _fail_iter(_unit_dir):
		scan_called["count"] += 1
		raise AssertionError("iter_unit_dirs should not run in dry-run")

	monkeypatch.setattr(analysis_runner, "iter_unit_dirs", _fail_iter)

	stage_config = SimpleNamespace(
		manifest_relpath="manifest.json",
		tables_relpath="tables",
		compute_metrics_enabled=True,
		recon_output_rel_root="recon_outputs",
		dataset_id_field=None,
		treatment_field=None,
	)

	set_dry_run_override(True)

	result = analysis_runner.run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=tmp_path / "data.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	assert scan_called["count"] == 0

	# Manifest written with dry_run_ok.
	manifest_path = well_out_dir / "analysis_outputs" / "manifest.json"
	assert manifest_path.exists()
	on_disk = json.loads(manifest_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	assert on_disk["phase"] == "analysis.compute_metrics"

	# Inputs reported.
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert "recon_outputs_dir" in by_name
	assert "spikesort_outputs_dir" in by_name
	# Both should be reported as not existing in this tmp_path fixture.
	assert by_name["recon_outputs_dir"]["exists"] is False

	# Missing recon_outputs warning surfaced.
	assert any(
		"recon_outputs_dir not found" in w
		for w in on_disk["validation"]["warnings"]
	)

	# Outputs reported.
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "manifest_json" in output_names
	assert "units_parquet" in output_names
	assert "well_summary_parquet" in output_names

	# Return type is AnalysisResult — verify the relevant fields.
	assert str(result.manifest_json) == str(manifest_path)
	assert str(result.well_out_dir) == str(well_out_dir)


def test_compute_metrics_dry_run_reports_existing_recon_outputs(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override
	from axon_recon.pipeline.stages.analysis import runner as analysis_runner

	well_out_dir = tmp_path / "well"
	recon_outputs_dir = well_out_dir / "recon_outputs"
	recon_outputs_dir.mkdir(parents=True, exist_ok=True)

	monkeypatch.setattr(
		analysis_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)

	monkeypatch.setattr(
		analysis_runner,
		"iter_unit_dirs",
		lambda _: (_ for _ in ()).throw(AssertionError("should not run")),
	)

	stage_config = SimpleNamespace(
		manifest_relpath="manifest.json",
		tables_relpath="tables",
		compute_metrics_enabled=True,
		recon_output_rel_root="recon_outputs",
		dataset_id_field=None,
		treatment_field=None,
	)

	set_dry_run_override(True)

	analysis_runner.run_analysis_compute_metrics_stage(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=tmp_path / "data.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)

	manifest_path = well_out_dir / "analysis_outputs" / "manifest.json"
	on_disk = json.loads(manifest_path.read_text())
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["recon_outputs_dir"]["exists"] is True
	# No missing-recon warning when recon_outputs exists.
	assert not any(
		"recon_outputs_dir not found" in w
		for w in on_disk["validation"]["warnings"]
	)
