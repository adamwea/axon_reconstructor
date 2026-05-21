"""Tests for `analysis.unitmatch` phase dry-run short-circuit.

Mirrors `propagation_video`'s compose-with-OR pattern: dry-run fires
when EITHER `stage_config.dry_run=True` OR the process-wide
`get_dry_run_override()` is set.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.analysis.orchestrators import unitmatch as unitmatch_phase


@pytest.fixture(autouse=True)
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def _stub_compute_mea_analysis_output_dir(monkeypatch, well_out_dir: Path) -> None:
	monkeypatch.setattr(
		unitmatch_phase,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)


def _build_stage_config(*, unitmatch_enabled: bool = True) -> SimpleNamespace:
	"""Build a minimal SimpleNamespace with the fields the dry-run path reads."""

	return SimpleNamespace(
		unitmatch_enabled=unitmatch_enabled,
		unitmatch_rel_output_root="unitmatch",
		well_metadata_lookup={
			(0, "well000"): {"chip_id": "M08073"},
		},
	)


def test_unitmatch_dry_run_via_process_wide_override(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well_out"
	_stub_compute_mea_analysis_output_dir(monkeypatch, well_out_dir)

	# Make sure the heavy unitlink.match() resolver is NEVER reached.
	def _fail_resolve(_stage_config):
		raise AssertionError("_resolve_unitlink_call should not run under --dry-run")

	monkeypatch.setattr(unitmatch_phase, "_resolve_unitlink_call", _fail_resolve)

	stage_config = _build_stage_config(unitmatch_enabled=True)
	set_dry_run_override(True)

	result = unitmatch_phase.run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=tmp_path / "data.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "analysis.unitmatch"

	# Summary persisted at the per-target location.
	target_summary_path = well_out_dir / "analysis_outputs" / "context" / "unitmatch_summary.json"
	assert target_summary_path.exists()
	on_disk = json.loads(target_summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	# chip_id reported as input + group_dir as would-be output.
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["well_metadata_chip_id"]["path"] == "M08073"
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "group_dir" in output_names


def test_unitmatch_dry_run_disabled_phase_warning(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	"""When unitmatch_enabled=False, a clear warning is surfaced."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well_out"
	_stub_compute_mea_analysis_output_dir(monkeypatch, well_out_dir)
	monkeypatch.setattr(
		unitmatch_phase,
		"_resolve_unitlink_call",
		lambda _: (_ for _ in ()).throw(AssertionError("should not run")),
	)

	stage_config = _build_stage_config(unitmatch_enabled=False)
	set_dry_run_override(True)

	result = unitmatch_phase.run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=tmp_path / "data.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	assert result["status"] == "dry_run_ok"

	target_summary_path = well_out_dir / "analysis_outputs" / "context" / "unitmatch_summary.json"
	on_disk = json.loads(target_summary_path.read_text())
	assert any(
		"unitmatch_enabled=False" in w
		for w in on_disk["validation"]["warnings"]
	)


def test_unitmatch_dry_run_via_stage_config_dry_run(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	"""Per-stage_config `dry_run=True` ALSO triggers the short-circuit,
	without needing the process-wide override (compose-with-OR)."""

	well_out_dir = tmp_path / "well_out"
	_stub_compute_mea_analysis_output_dir(monkeypatch, well_out_dir)
	monkeypatch.setattr(
		unitmatch_phase,
		"_resolve_unitlink_call",
		lambda _: (_ for _ in ()).throw(AssertionError("should not run")),
	)

	stage_config = _build_stage_config(unitmatch_enabled=True)
	stage_config.dry_run = True  # per-stage_config trigger

	result = unitmatch_phase.run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=tmp_path / "data.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		output_rel_root="analysis_outputs",
		stage_config=stage_config,
		force_restart=False,
	)
	assert result["status"] == "dry_run_ok"
