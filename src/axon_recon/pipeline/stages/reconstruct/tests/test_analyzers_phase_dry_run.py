"""Tests for `reconstruct.analyzers` phase dry-run short-circuit.

When `get_dry_run_override()` is True, the phase writes a
`dry_run_ok` summary and returns without calling the heavy
`_run_reconstruct_templates_analyzers_phase_body` (which builds
SortingAnalyzers, scans preprocessed segments, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.reconstruct.phases import analyzers as analyzers_phase


@pytest.fixture
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def test_analyzers_phase_dry_run_writes_summary_and_skips_body(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	templates_out_dir = tmp_path / "well" / "recon_outputs"
	analyzer_cache_dir = templates_out_dir / "cache" / "analyzers"
	h5_path = tmp_path / "fake.h5"

	def _stub_resolve_env(_inputs):
		return well_out_dir, [], templates_out_dir, analyzer_cache_dir

	monkeypatch.setattr(
		analyzers_phase.templates_runner,
		"_resolve_templates_phase_environment",
		_stub_resolve_env,
	)

	# These should NOT be called during dry-run.
	body_called = {"count": 0}

	def _fail_body(**_):
		body_called["count"] += 1
		raise AssertionError("phase body should not run in dry-run")

	monkeypatch.setattr(
		analyzers_phase,
		"_run_reconstruct_templates_analyzers_phase_body",
		_fail_body,
	)

	# Build a minimal SimpleNamespace masquerading as TemplatesInputs —
	# only the fields read by the dry-run path are needed.
	inputs = SimpleNamespace(
		h5_path=h5_path,  # doesn't exist → triggers a warning
		phases=SimpleNamespace(
			analyzers=SimpleNamespace(summary_json_relpath="analyzers_summary.json")
		),
	)

	set_dry_run_override(True)

	result = analyzers_phase.run_reconstruct_templates_analyzers_phase(inputs)

	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "reconstruct.analyzers"
	assert body_called["count"] == 0

	summary_path = templates_out_dir / "analyzers_summary.json"
	assert summary_path.exists()
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	assert on_disk["phase"] == "reconstruct.analyzers"

	# Cache + h5 both reported as missing.
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["analyzer_cache_dir"]["exists"] is False
	assert by_name["h5_path"]["exists"] is False
	# H5 missing should surface as a warning.
	assert any("h5_path not found" in w for w in on_disk["validation"]["warnings"])

	# Outputs that WOULD be produced.
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "analyzer_cache_dir" in output_names
	assert "summary_json" in output_names


def test_analyzers_phase_dry_run_reports_existing_h5(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""H5 file exists → no missing-h5 warning."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	templates_out_dir = tmp_path / "well" / "recon_outputs"
	analyzer_cache_dir = templates_out_dir / "cache" / "analyzers"
	h5_path = tmp_path / "real.h5"
	h5_path.write_bytes(b"")  # exists

	def _stub_resolve_env(_inputs):
		return well_out_dir, [], templates_out_dir, analyzer_cache_dir

	monkeypatch.setattr(
		analyzers_phase.templates_runner,
		"_resolve_templates_phase_environment",
		_stub_resolve_env,
	)

	def _fail_body(**_):
		raise AssertionError("phase body should not run in dry-run")

	monkeypatch.setattr(
		analyzers_phase,
		"_run_reconstruct_templates_analyzers_phase_body",
		_fail_body,
	)

	inputs = SimpleNamespace(
		h5_path=h5_path,
		phases=SimpleNamespace(
			analyzers=SimpleNamespace(summary_json_relpath="analyzers_summary.json")
		),
	)

	set_dry_run_override(True)

	result = analyzers_phase.run_reconstruct_templates_analyzers_phase(inputs)
	assert result["status"] == "dry_run_ok"

	summary_path = templates_out_dir / "analyzers_summary.json"
	on_disk = json.loads(summary_path.read_text())
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["h5_path"]["exists"] is True
	assert not any("h5_path not found" in w for w in on_disk["validation"]["warnings"])


def test_analyzers_phase_dry_run_with_source_scope(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""Phase name in the summary reflects the source_scope kwarg."""

	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = tmp_path / "well"
	templates_out_dir = tmp_path / "well" / "recon_outputs"
	analyzer_cache_dir = templates_out_dir / "cache" / "analyzers"

	def _stub_resolve_env(_inputs):
		return well_out_dir, [], templates_out_dir, analyzer_cache_dir

	monkeypatch.setattr(
		analyzers_phase.templates_runner,
		"_resolve_templates_phase_environment",
		_stub_resolve_env,
	)
	monkeypatch.setattr(
		analyzers_phase,
		"_run_reconstruct_templates_analyzers_phase_body",
		lambda **_: (_ for _ in ()).throw(AssertionError("body should not run")),
	)

	inputs = SimpleNamespace(
		h5_path=None,
		phases=SimpleNamespace(
			analyzers=SimpleNamespace(summary_json_relpath="analyzers_summary.json")
		),
	)
	set_dry_run_override(True)

	result = analyzers_phase.run_reconstruct_templates_analyzers_phase(
		inputs, source_scope="segments"
	)
	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "reconstruct.analyzers.segments"

	summary_path = templates_out_dir / "analyzers_summary.json"
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["phase"] == "reconstruct.analyzers.segments"
