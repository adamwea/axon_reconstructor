"""Tests for `reconstruct.clear_templates_cache` phase dry-run short-circuit.

When `get_dry_run_override()` is True, the phase reports the cache dir
it WOULD remove without invoking the rmtree.

NOTE: The phase module is imported lazily INSIDE each test function
(not at the top of this file). The pre-existing test
`test_reconstruct_clear_templates_cache_phase_uses_templates_output_root`
in `test_runner.py` patches the CORE module's
`run_clear_templates_cache_phase` and relies on the phase module's
binding being established AFTER the patch fires (which it does because
the runner.py wrapper imports the phase module lazily inside the
function body). If we import the phase module eagerly at the top of
this test file, the phase module's binding is captured BEFORE that
test's monkeypatch, and the pre-existing test fails. Keeping imports
lazy here preserves backwards compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def test_clear_templates_cache_dry_run_writes_summary_and_skips_rmtree(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	# Lazy import — see module-level docstring for why.
	from axon_recon.pipeline.config import set_dry_run_override
	from axon_recon.pipeline.stages.reconstruct.phases import (
		clear_templates_cache as clear_phase,
	)

	# Materialize a fake cache dir so we can confirm the dry-run does NOT
	# remove it.
	well_out_dir = tmp_path / "well"
	templates_out_dir = well_out_dir / "templates_outputs"
	cache_dir = templates_out_dir / "cache" / "templates"
	cache_dir.mkdir(parents=True, exist_ok=True)
	(cache_dir / "sentinel.txt").write_text("not deleted")

	monkeypatch.setattr(
		clear_phase,
		"compute_mea_analysis_output_dir",
		lambda **_: well_out_dir,
	)

	# Verify the heavy work (`run_clear_templates_cache_phase`) is NEVER
	# called when dry-run is on. Phase resolves via the core submodule
	# reference, so we patch CORE's attribute (which is also the form the
	# pre-existing test_runner.py test uses).
	from axon_recon.pipeline.stages.reconstruct.core import clear_templates_cache as core_clear

	heavy_called = {"count": 0}

	def _fail_heavy(**_):
		heavy_called["count"] += 1
		raise AssertionError("heavy phase should not run in dry-run")

	monkeypatch.setattr(core_clear, "run_clear_templates_cache_phase", _fail_heavy)

	inputs = SimpleNamespace(
		mea_output_root=tmp_path,
		h5_path=tmp_path / "fake.h5",
		stream_id="well000",
		output_rel_root="recon_outputs",
		templates_inputs=SimpleNamespace(output_rel_root="templates_outputs"),
		phases=SimpleNamespace(
			clear_templates_cache=SimpleNamespace(
				enabled=True,
				keep_merged_per_unit_outputs=True,
				keep_full_channels_templates=False,
				summary_json_relpath="reports/clear_templates_cache_summary.json",
			),
		),
	)

	set_dry_run_override(True)

	result = clear_phase.run_reconstruct_clear_templates_cache_phase(inputs)

	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "reconstruct.clear_templates_cache"
	assert heavy_called["count"] == 0
	# Sentinel is still there.
	assert (cache_dir / "sentinel.txt").exists()

	summary_path = (
		well_out_dir / "recon_outputs" / "reports" / "clear_templates_cache_summary.json"
	)
	assert summary_path.exists()
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	# Cache dir reported as a would-be-removed output.
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "cache_dir_would_be_removed" in output_names
	# Extras reported.
	assert on_disk["phase_enabled"] is True
	assert on_disk["keep_merged_per_unit_outputs"] is True


def test_clear_templates_cache_dry_run_reports_missing_cache(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""Cache dir doesn't exist → reports exists=False."""

	# Lazy import — see module-level docstring for why.
	from axon_recon.pipeline.config import set_dry_run_override
	from axon_recon.pipeline.stages.reconstruct.phases import (
		clear_templates_cache as clear_phase,
	)

	from axon_recon.pipeline.stages.reconstruct.core import clear_templates_cache as core_clear

	well_out_dir = tmp_path / "well"
	monkeypatch.setattr(
		clear_phase,
		"compute_mea_analysis_output_dir",
		lambda **_: well_out_dir,
	)
	monkeypatch.setattr(
		core_clear,
		"run_clear_templates_cache_phase",
		lambda **_: (_ for _ in ()).throw(AssertionError("heavy phase should not run")),
	)

	inputs = SimpleNamespace(
		mea_output_root=tmp_path,
		h5_path=tmp_path / "fake.h5",
		stream_id="well000",
		output_rel_root="recon_outputs",
		templates_inputs=None,
		phases=SimpleNamespace(
			clear_templates_cache=SimpleNamespace(
				enabled=False,
				keep_merged_per_unit_outputs=True,
				keep_full_channels_templates=False,
				summary_json_relpath="reports/clear_templates_cache_summary.json",
			),
		),
	)
	set_dry_run_override(True)

	result = clear_phase.run_reconstruct_clear_templates_cache_phase(inputs)
	assert result["status"] == "dry_run_ok"

	summary_path = (
		well_out_dir / "recon_outputs" / "reports" / "clear_templates_cache_summary.json"
	)
	on_disk = json.loads(summary_path.read_text())
	cache_entries = [
		item for item in on_disk["inputs_resolved"] if item["name"] == "cache_dir"
	]
	assert cache_entries[0]["exists"] is False
	# Phase ships disabled — that's reflected.
	assert on_disk["phase_enabled"] is False
