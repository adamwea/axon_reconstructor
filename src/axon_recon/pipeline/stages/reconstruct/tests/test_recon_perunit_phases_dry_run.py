"""Tests for the recon-stage per-unit phases' dry-run short-circuit.

The helper `reconstruct_phase_dry_run_short_circuit` in
`stages/reconstruct/runner.py` is shared by every plot_recons-family
+ report_recons-family phase. This test file exercises the helper
directly + verifies each phase wires through it correctly.
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


def _build_fake_env(tmp_path: Path, *, n_units: int = 5):
	well_out_dir = tmp_path / "well"
	reconstruction_out_dir = tmp_path / "well" / "recon_outputs"
	reconstruction_out_dir.mkdir(parents=True, exist_ok=True)
	return SimpleNamespace(
		well_out_dir=well_out_dir,
		reconstruction_out_dir=reconstruction_out_dir,
		unit_ids=list(range(n_units)),
		preserve_stage_reports=False,
		merged_units_dir=reconstruction_out_dir / "cache/templates/merged",
		full_channels_templates_dir=reconstruction_out_dir / "cache/templates/full",
		existing_stage_outputs={},
	)


def _stub_env_prep(monkeypatch, env):
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	monkeypatch.setattr(
		reconstruct_runner,
		"_prepare_reconstruct_phase_environment",
		lambda *, inputs, clear_output_root: env,
	)


# ---------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------


def test_reconstruct_phase_dry_run_short_circuit_writes_summary(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	env = _build_fake_env(tmp_path, n_units=12)
	_stub_env_prep(monkeypatch, env)

	summary_json = env.reconstruction_out_dir / "context" / "test_summary.json"

	result = reconstruct_runner.reconstruct_phase_dry_run_short_circuit(
		inputs=SimpleNamespace(),
		phase_name="plot_recons",
		summary_json=summary_json,
	)
	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "reconstruct.plot_recons"
	assert result["n_units"] == 12

	assert summary_json.exists()
	on_disk = json.loads(summary_json.read_text())
	assert on_disk["status"] == "dry_run_ok"
	assert on_disk["phase"] == "reconstruct.plot_recons"
	assert on_disk["n_units_resolved"] == 12

	# Standard outputs reported.
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "summary_json" in output_names
	assert "reconstruction_out_dir" in output_names


def test_reconstruct_phase_dry_run_short_circuit_accepts_additional_outputs(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner

	env = _build_fake_env(tmp_path)
	_stub_env_prep(monkeypatch, env)

	summary_json = env.reconstruction_out_dir / "p.json"

	reconstruct_runner.reconstruct_phase_dry_run_short_circuit(
		inputs=SimpleNamespace(),
		phase_name="report_recons",
		summary_json=summary_json,
		additional_outputs=(
			("recon_report_pdf", env.reconstruction_out_dir / "recon_report.pdf"),
		),
	)
	on_disk = json.loads(summary_json.read_text())
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "recon_report_pdf" in output_names


# ---------------------------------------------------------------------
# Per-phase integration: each phase short-circuits via the helper.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
	"phase_module_path,attr_name",
	[
		(
			"axon_recon.pipeline.stages.reconstruct.phases.plot_recons",
			"run_reconstruct_plot_recons_phase",
		),
		(
			"axon_recon.pipeline.stages.reconstruct.phases.plot_branch_propagations",
			"run_reconstruct_plot_branch_propagations_phase",
		),
		(
			"axon_recon.pipeline.stages.reconstruct.phases.plot_branch_velocities",
			"run_reconstruct_plot_branch_velocities_phase",
		),
		(
			"axon_recon.pipeline.stages.reconstruct.phases.plot_unit_summary",
			"run_reconstruct_plot_unit_summary_phase",
		),
		(
			"axon_recon.pipeline.stages.reconstruct.phases.report_recons",
			"run_reconstruct_report_recons_phase",
		),
		(
			"axon_recon.pipeline.stages.reconstruct.phases.report_recon_grid",
			"run_reconstruct_report_recon_grid_phase",
		),
		(
			"axon_recon.pipeline.stages.reconstruct.phases.report_full_chip_layout",
			"run_reconstruct_report_full_chip_layout_phase",
		),
		(
			"axon_recon.pipeline.stages.reconstruct.phases.report_summaries",
			"run_reconstruct_report_summaries_phase",
		),
	],
)
def test_recon_perunit_phase_dry_run_short_circuits(
	phase_module_path: str,
	attr_name: str,
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
	_reset_dry_run_override,
) -> None:
	"""Every per-unit recon phase routes through the shared helper when
	dry-run is set and writes a `dry_run_ok` summary."""

	from importlib import import_module

	from axon_recon.pipeline.config import set_dry_run_override

	module = import_module(phase_module_path)
	phase_fn = getattr(module, attr_name)

	# Derive the summary_json_relpath field name from the phase's name.
	# Each phase reads `inputs.phases.<phase_name>.summary_json_relpath`.
	phase_name = attr_name.replace("run_reconstruct_", "").replace("_phase", "")
	env = _build_fake_env(tmp_path)
	_stub_env_prep(monkeypatch, env)

	inputs = SimpleNamespace(
		phases=SimpleNamespace(
			**{
				phase_name: SimpleNamespace(
					summary_json_relpath=f"{phase_name}_summary.json"
				)
			}
		),
		report_sort_by="unit_id",
	)

	set_dry_run_override(True)

	result = phase_fn(inputs)
	assert result["status"] == "dry_run_ok"
	assert result["phase"] == f"reconstruct.{phase_name}"
