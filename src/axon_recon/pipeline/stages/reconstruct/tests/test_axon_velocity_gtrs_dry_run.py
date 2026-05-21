"""Tests for `reconstruct.axon_velocity_gtrs` phase dry-run short-circuit.

When `get_dry_run_override()` is True, the phase writes a
`dry_run_ok` summary and returns without invoking the per-unit GTRS
fanout (which is the heaviest compute in the recon stage after
kssynth).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.reconstruct.phases import axon_velocity_gtrs as gtrs_phase


@pytest.fixture
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def test_axon_velocity_gtrs_dry_run_writes_summary_and_skips_impl(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	reconstruction_out_dir = tmp_path / "well" / "recon_outputs"
	reconstruction_out_dir.mkdir(parents=True, exist_ok=True)

	fake_env = SimpleNamespace(
		well_out_dir=well_out_dir,
		reconstruction_out_dir=reconstruction_out_dir,
		unit_ids=[42, 43, 44, 45, 46, 47, 48],
		preserve_stage_reports=False,
	)

	monkeypatch.setattr(
		gtrs_phase.reconstruct_runner,
		"_prepare_reconstruct_phase_environment",
		lambda *, inputs, clear_output_root: fake_env,
	)

	# These should NOT be called during dry-run.
	impl_called = {"count": 0}

	def _fail_impl(**_):
		impl_called["count"] += 1
		raise AssertionError("phase impl should not run in dry-run")

	monkeypatch.setattr(
		gtrs_phase,
		"_run_reconstruct_axon_velocity_gtrs_phase_impl",
		_fail_impl,
	)

	# Build minimal inputs matching the shape the dry-run path reads.
	inputs = SimpleNamespace(
		phases=SimpleNamespace(
			axon_velocity_gtrs=SimpleNamespace(
				summary_json_relpath="context/axon_velocity_gtrs_summary.json"
			)
		),
	)

	set_dry_run_override(True)

	result = gtrs_phase.run_reconstruct_axon_velocity_gtrs_phase(inputs)

	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "reconstruct.axon_velocity_gtrs"
	assert result["n_units"] == 7
	assert impl_called["count"] == 0

	summary_path = reconstruction_out_dir / "context" / "axon_velocity_gtrs_summary.json"
	assert summary_path.exists()
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	assert on_disk["phase"] == "reconstruct.axon_velocity_gtrs"
	# Extra field comes through.
	assert on_disk["n_units_resolved"] == 7

	# Outputs reported.
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "summary_json" in output_names
	assert "reconstruction_out_dir" in output_names


def test_axon_velocity_gtrs_dry_run_zero_units(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""Empty unit_ids still produces a valid dry-run summary."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	reconstruction_out_dir = tmp_path / "well" / "recon_outputs"
	reconstruction_out_dir.mkdir(parents=True, exist_ok=True)

	fake_env = SimpleNamespace(
		well_out_dir=well_out_dir,
		reconstruction_out_dir=reconstruction_out_dir,
		unit_ids=[],
		preserve_stage_reports=False,
	)
	monkeypatch.setattr(
		gtrs_phase.reconstruct_runner,
		"_prepare_reconstruct_phase_environment",
		lambda *, inputs, clear_output_root: fake_env,
	)
	monkeypatch.setattr(
		gtrs_phase,
		"_run_reconstruct_axon_velocity_gtrs_phase_impl",
		lambda **_: (_ for _ in ()).throw(AssertionError("impl should not run")),
	)

	inputs = SimpleNamespace(
		phases=SimpleNamespace(
			axon_velocity_gtrs=SimpleNamespace(
				summary_json_relpath="axon_velocity_gtrs_summary.json"
			)
		),
	)
	set_dry_run_override(True)

	result = gtrs_phase.run_reconstruct_axon_velocity_gtrs_phase(inputs)
	assert result["status"] == "dry_run_ok"
	assert result["n_units"] == 0
