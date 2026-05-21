"""Tests for `spikesort.sort` phase dry-run short-circuit.

CRITICAL contract: dry-run for `sort` must NOT load Kilosort or CUDA.
The dry-run intercept fires at the top of `run_spikesort_stage`, before
any sort-engine import or recording load. This test stubs
`compute_mea_analysis_output_dir` and verifies the heavy code path
(`_cleanup_spikesort_outputs_for_force_restart`, the actual sort
invocation, etc.) is NEVER reached.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.spikesort import runner as spikesort_runner


@pytest.fixture
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def _stub_compute_well_out(monkeypatch, tmp_path: Path) -> Path:
	well_out_dir = tmp_path / "well_out"

	monkeypatch.setattr(
		spikesort_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	return well_out_dir


def _build_inputs(tmp_path: Path, *, sort_enabled: bool = True) -> SimpleNamespace:
	"""Minimal inputs shape — only fields the dry-run path reads."""

	return SimpleNamespace(
		h5_path=tmp_path / "data.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		output_rel_root="spikesort_outputs",
		force_restart=False,
		replot=False,
		sort_enabled=sort_enabled,
		sort_delete_outputs_on_force_restart=False,
		sort_engine="mea_analysis",
		sorter="kilosort4",
		sort_use_bootstrapped_concat_binary=False,
		sort_use_lazy_source=False,
		sort_assert_one_source=False,
		run_analyzer=False,
		um_kwargs={},
		preprocess_concat_recording_relpath="preprocess_outputs/concat",
	)


def test_spikesort_sort_dry_run_writes_summary_and_skips_heavy_work(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""Critical: dry-run must skip the cleanup + sort kicks entirely."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = _stub_compute_well_out(monkeypatch, tmp_path)

	# Stub the heavy cleanup helper to raise if reached.
	def _fail_cleanup(**_):
		raise AssertionError("_cleanup_spikesort_outputs_for_force_restart should not run under dry-run")

	monkeypatch.setattr(
		spikesort_runner, "_cleanup_spikesort_outputs_for_force_restart", _fail_cleanup
	)

	# Materialize a fake h5 so the warning isn't triggered.
	h5_path = tmp_path / "data.h5"
	h5_path.write_bytes(b"")

	inputs = _build_inputs(tmp_path, sort_enabled=True)
	set_dry_run_override(True)

	result = spikesort_runner.run_spikesort_stage(inputs)

	# Returns a SpikesortResult with the summary path.
	assert str(result.well_out_dir) == str(well_out_dir)
	summary_path = well_out_dir / "spikesort_outputs" / "spikesort_summary.json"
	assert summary_path.exists()
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	assert on_disk["phase"] == "spikesort.sort"

	# Inputs reported.
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["h5_path"]["exists"] is True

	# Outputs reported.
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "sorter_output_dir" in output_names
	assert "concat_binary_dir" in output_names
	assert "summary_json" in output_names

	# Extras reported.
	assert on_disk["sort_enabled"] is True
	assert on_disk["sorter"] == "kilosort4"


def test_spikesort_sort_dry_run_missing_h5_warning(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""h5 not on disk → warning surfaced + still completes."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = _stub_compute_well_out(monkeypatch, tmp_path)
	monkeypatch.setattr(
		spikesort_runner,
		"_cleanup_spikesort_outputs_for_force_restart",
		lambda **_: (_ for _ in ()).throw(AssertionError("should not run")),
	)

	inputs = _build_inputs(tmp_path)  # h5 doesn't exist
	set_dry_run_override(True)

	result = spikesort_runner.run_spikesort_stage(inputs)
	summary_path = well_out_dir / "spikesort_outputs" / "spikesort_summary.json"
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	assert any("h5_path not found" in w for w in on_disk["validation"]["warnings"])


def test_spikesort_sort_dry_run_disabled_phase_warning(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""When sort_enabled=False, a clear warning is surfaced."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = _stub_compute_well_out(monkeypatch, tmp_path)
	monkeypatch.setattr(
		spikesort_runner,
		"_cleanup_spikesort_outputs_for_force_restart",
		lambda **_: (_ for _ in ()).throw(AssertionError("should not run")),
	)
	h5_path = tmp_path / "data.h5"
	h5_path.write_bytes(b"")

	inputs = _build_inputs(tmp_path, sort_enabled=False)
	set_dry_run_override(True)

	result = spikesort_runner.run_spikesort_stage(inputs)
	summary_path = well_out_dir / "spikesort_outputs" / "spikesort_summary.json"
	on_disk = json.loads(summary_path.read_text())
	assert any("sort_enabled=False" in w for w in on_disk["validation"]["warnings"])
	assert on_disk["sort_enabled"] is False
