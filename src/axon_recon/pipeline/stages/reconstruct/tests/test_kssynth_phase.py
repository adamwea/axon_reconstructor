"""Tests for the recon-stage `kssynth` phase scaffold (Era 3 slice 1a).

The current scaffold validates the kssynth import path and writes a stub
summary JSON marked `status: scaffold_only`. Slice 1b will replace the
body with the real analyzer-loading + `kssynth.synthesize(...)` call;
these tests will then evolve to mock `kssynth.synthesize` and assert the
WriterResult-to-summary translation.

For now we monkey-patch `_resolve_kssynth_output_dirs` so we don't need a
fully-populated TemplatesInputs fixture (that machinery lives in
`test_extract_partial_templates.py` and is heavy).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.reconstruct.phases import kssynth as kssynth_phase


def _stub_resolve(tmp_path: Path):
	"""Helper: patches _resolve_kssynth_output_dirs to return tmp_path-based dirs."""

	def _impl(inputs):
		well_out_dir = tmp_path / "well_out"
		templates_out_dir = tmp_path / "well_out" / "recon_outputs"
		templates_out_dir.mkdir(parents=True, exist_ok=True)
		(templates_out_dir / kssynth_phase.KSSYNTH_OUTPUT_RELDIR).mkdir(
			parents=True, exist_ok=True
		)
		return well_out_dir, templates_out_dir

	return _impl


def test_kssynth_scaffold_returns_scaffold_only_status(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)
	inputs = SimpleNamespace()
	summary = kssynth_phase.run_reconstruct_kssynth_phase(inputs)

	assert summary["phase"] == "reconstruct.kssynth"
	assert summary["status"] == "scaffold_only"
	assert summary["synth_sorter_output_relpath"] == kssynth_phase.KSSYNTH_OUTPUT_RELDIR
	assert "error" not in summary


def test_kssynth_scaffold_writes_summary_json(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)
	inputs = SimpleNamespace()
	summary = kssynth_phase.run_reconstruct_kssynth_phase(inputs)

	templates_out_dir = tmp_path / "well_out" / "recon_outputs"
	summary_path = templates_out_dir / kssynth_phase.KSSYNTH_SUMMARY_RELPATH
	assert summary_path.exists(), f"missing summary at {summary_path}"

	on_disk = json.loads(summary_path.read_text())
	assert on_disk == summary


def test_kssynth_scaffold_reports_error_when_kssynth_missing(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)

	# Force ImportError by hiding kssynth from sys.modules + sys.path.
	# We add the package name to a finder that always raises ImportError.
	class _BlockKssynthFinder:
		def find_spec(self, fullname, path=None, target=None):
			if fullname.startswith("kssynth"):
				raise ImportError(f"blocked: {fullname}")
			return None

	original_finders = list(sys.meta_path)
	original_modules = {
		k: sys.modules.pop(k)
		for k in list(sys.modules.keys())
		if k == "kssynth" or k.startswith("kssynth.")
	}
	sys.meta_path.insert(0, _BlockKssynthFinder())
	try:
		inputs = SimpleNamespace()
		summary = kssynth_phase.run_reconstruct_kssynth_phase(inputs)
	finally:
		sys.meta_path.remove(_BlockKssynthFinder()) if False else None
		# Restore meta_path
		sys.meta_path[:] = original_finders
		sys.modules.update(original_modules)

	assert summary["status"] == "error"
	assert "kssynth import failed" in summary["error"]
