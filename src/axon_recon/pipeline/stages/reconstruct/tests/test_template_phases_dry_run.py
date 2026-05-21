"""Tests for `reconstruct.plot_templates_v2` + `reconstruct.report_templates`
dry-run short-circuits.

Both phases write a `dry_run_ok` summary and return WITHOUT invoking
their heavy per-unit work (the templates render + PDF compose path).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.reconstruct.phases import plot_templates_v2 as plot_phase
from axon_recon.pipeline.stages.reconstruct.phases import report_templates as report_phase


@pytest.fixture
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def _stub_resolve_templates_phase_environment(*, well_out_dir: Path, templates_out_dir: Path):
	def _impl(_inputs):
		return well_out_dir, [], templates_out_dir, None

	return _impl


def test_plot_templates_v2_dry_run_writes_summary_and_skips_body(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	templates_out_dir = tmp_path / "well" / "recon_outputs"

	monkeypatch.setattr(
		plot_phase.templates_runner,
		"_resolve_templates_phase_environment",
		_stub_resolve_templates_phase_environment(
			well_out_dir=well_out_dir, templates_out_dir=templates_out_dir
		),
	)

	# Body should not be called.
	body_called = {"count": 0}

	def _fail_body(**_):
		body_called["count"] += 1
		raise AssertionError("phase body should not run in dry-run")

	monkeypatch.setattr(
		plot_phase,
		"_run_reconstruct_templates_plot_templates_v2_phase_body",
		_fail_body,
	)

	# Stub templates dir resolution — return a path that doesn't exist.
	monkeypatch.setattr(
		plot_phase.templates_runner,
		"_resolve_templates_dirs",
		lambda **_: (
			templates_out_dir / "cache/templates/merged",
			templates_out_dir / "cache/templates/full",
		),
	)

	inputs = SimpleNamespace(
		phases=SimpleNamespace(
			plot_templates_v2=SimpleNamespace(
				summary_json_relpath="plot_templates_v2_summary.json",
				output_relpath="template_circles_v2",
			),
		),
	)

	set_dry_run_override(True)

	result = plot_phase.run_reconstruct_templates_plot_templates_v2_phase(inputs)

	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "reconstruct.plot_templates_v2"
	assert body_called["count"] == 0

	summary_path = templates_out_dir / "plot_templates_v2_summary.json"
	assert summary_path.exists()
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"

	# Cache dir reported as not-existing.
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["merged_units_dir"]["exists"] is False

	# Outputs reported.
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "summary_json" in output_names
	assert "template_plots_root" in output_names


def test_plot_templates_v2_dry_run_surfaces_missing_templates_warning(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""When _resolve_templates_dirs raises FileNotFoundError, the dry-run
	summary captures the missing-templates condition as a warning."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	templates_out_dir = tmp_path / "well" / "recon_outputs"

	monkeypatch.setattr(
		plot_phase.templates_runner,
		"_resolve_templates_phase_environment",
		_stub_resolve_templates_phase_environment(
			well_out_dir=well_out_dir, templates_out_dir=templates_out_dir
		),
	)
	monkeypatch.setattr(
		plot_phase,
		"_run_reconstruct_templates_plot_templates_v2_phase_body",
		lambda **_: (_ for _ in ()).throw(AssertionError("body should not run")),
	)

	def _raise_missing(**_):
		raise FileNotFoundError("nope, build_templates first")

	monkeypatch.setattr(
		plot_phase.templates_runner, "_resolve_templates_dirs", _raise_missing
	)

	inputs = SimpleNamespace(
		phases=SimpleNamespace(
			plot_templates_v2=SimpleNamespace(
				summary_json_relpath="p.json",
				output_relpath="template_circles_v2",
			),
		),
	)
	set_dry_run_override(True)

	result = plot_phase.run_reconstruct_templates_plot_templates_v2_phase(inputs)
	assert result["status"] == "dry_run_ok"

	on_disk = json.loads((templates_out_dir / "p.json").read_text())
	assert any(
		"merged templates dir not found" in w
		for w in on_disk["validation"]["warnings"]
	)


def test_report_templates_dry_run_writes_summary_and_skips_body(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	templates_out_dir = tmp_path / "well" / "recon_outputs"

	monkeypatch.setattr(
		report_phase.templates_runner,
		"_resolve_templates_phase_environment",
		_stub_resolve_templates_phase_environment(
			well_out_dir=well_out_dir, templates_out_dir=templates_out_dir
		),
	)

	body_called = {"count": 0}

	def _fail_body(**_):
		body_called["count"] += 1
		raise AssertionError("phase body should not run in dry-run")

	monkeypatch.setattr(
		report_phase,
		"_run_reconstruct_templates_report_templates_phase_body",
		_fail_body,
	)

	inputs = SimpleNamespace(
		phases=SimpleNamespace(
			report_templates=SimpleNamespace(
				summary_json_relpath="report_templates_summary.json",
				consume="plot_templates_v2",
			),
		),
	)

	set_dry_run_override(True)

	result = report_phase.run_reconstruct_templates_report_templates_phase(inputs)

	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "reconstruct.report_templates"
	assert body_called["count"] == 0

	summary_path = templates_out_dir / "report_templates_summary.json"
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	# Source key resolved cleanly → exists=True.
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["consume_source"]["exists"] is True
	assert on_disk["source_output_key"] == "template_circles_v2_png"


def test_report_templates_dry_run_invalid_consume_surfaces_warning(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""If `consume` is not the allowed `plot_templates_v2` value, the
	dry-run summary captures the misconfiguration as a warning."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	templates_out_dir = tmp_path / "well" / "recon_outputs"

	monkeypatch.setattr(
		report_phase.templates_runner,
		"_resolve_templates_phase_environment",
		_stub_resolve_templates_phase_environment(
			well_out_dir=well_out_dir, templates_out_dir=templates_out_dir
		),
	)
	monkeypatch.setattr(
		report_phase,
		"_run_reconstruct_templates_report_templates_phase_body",
		lambda **_: (_ for _ in ()).throw(AssertionError("body should not run")),
	)

	inputs = SimpleNamespace(
		phases=SimpleNamespace(
			report_templates=SimpleNamespace(
				summary_json_relpath="p.json",
				consume="plot_templates",  # invalid (legacy v1, removed)
			),
		),
	)
	set_dry_run_override(True)

	result = report_phase.run_reconstruct_templates_report_templates_phase(inputs)
	assert result["status"] == "dry_run_ok"

	on_disk = json.loads((templates_out_dir / "p.json").read_text())
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	# consume resolution failed → exists=False
	assert by_name["consume_source"]["exists"] is False
	# Warning about the misconfiguration.
	assert any("plot_templates_v2" in w for w in on_disk["validation"]["warnings"])
