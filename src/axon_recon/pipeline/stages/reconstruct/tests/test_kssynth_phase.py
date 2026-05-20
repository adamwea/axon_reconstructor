"""Tests for the recon-stage `kssynth` phase (Era 3 slices 1a + 1b).

Slice 1b makes the phase do real work: it calls
`templates.runner._load_templates_phase_analyzers` to get the segment
analyzer set, forwards them to `kssynth.api.synthesize`, then writes a
summary JSON derived from the returned `WriterResult`.

The tests monkey-patch:
  - `_resolve_kssynth_output_dirs` — to avoid populating a full
    `TemplatesInputs` fixture (its config dataclass tree is large).
  - `_load_segment_analyzers` — to return a controlled list of fake
    analyzers without touching the spikeinterface_extract loader.
  - `kssynth.api.synthesize` — to return a fake `WriterResult` shape
    without invoking the real algorithm.

Real-data smoke is slice 3 (login-node, one well from the M08073/well000
80k DMEM cohort).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.reconstruct.phases import kssynth as kssynth_phase


def _stub_resolve(tmp_path: Path):
	def _impl(inputs):
		well_out_dir = tmp_path / "well_out"
		templates_out_dir = tmp_path / "well_out" / "recon_outputs"
		synth_out_dir = templates_out_dir / kssynth_phase.KSSYNTH_OUTPUT_RELDIR
		synth_out_dir.mkdir(parents=True, exist_ok=True)
		return well_out_dir, templates_out_dir, synth_out_dir

	return _impl


def _fake_writer_result(unit_ids=(0, 1, 2), n_channels=42):
	return SimpleNamespace(
		out_folder=Path("/tmp/fake"),
		unit_ids=tuple(unit_ids),
		n_channels=n_channels,
		files_written=("spike_times.npy", "spike_clusters.npy", "templates.npy"),
		channel_grid_mode="union",
		policy="spike_count_weighted_mean",
	)


def test_kssynth_phase_ok_translates_writer_result_to_summary(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)
	# Three fake analyzers — content doesn't matter, kssynth.synthesize is mocked.
	monkeypatch.setattr(
		kssynth_phase, "_load_segment_analyzers", lambda inputs: [object(), object(), object()]
	)
	captured: dict = {}

	def _fake_synthesize(*, analyzers, out_folder, **kwargs):
		captured["analyzers"] = analyzers
		captured["out_folder"] = out_folder
		captured["kwargs"] = kwargs
		return _fake_writer_result(unit_ids=(10, 11, 12, 13), n_channels=128)

	import kssynth.api

	monkeypatch.setattr(kssynth.api, "synthesize", _fake_synthesize)

	summary = kssynth_phase.run_reconstruct_kssynth_phase(SimpleNamespace())

	assert summary["status"] == "ok"
	assert summary["n_units"] == 4
	assert summary["n_analyzers"] == 3
	assert summary["n_channels"] == 128
	assert summary["channel_grid_mode"] == "union"
	assert summary["policy"] == "spike_count_weighted_mean"
	assert "spike_times.npy" in summary["files_written"]

	# kssynth.synthesize was actually called with our 3 analyzers + synth_out_dir.
	assert len(captured["analyzers"]) == 3
	assert captured["out_folder"].name == kssynth_phase.KSSYNTH_OUTPUT_RELDIR


def test_kssynth_phase_writes_summary_to_disk(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)
	monkeypatch.setattr(kssynth_phase, "_load_segment_analyzers", lambda inputs: [object()])

	import kssynth.api

	monkeypatch.setattr(
		kssynth.api, "synthesize", lambda *, analyzers, out_folder, **_: _fake_writer_result()
	)

	summary = kssynth_phase.run_reconstruct_kssynth_phase(SimpleNamespace())

	summary_path = (
		tmp_path / "well_out" / "recon_outputs" / kssynth_phase.KSSYNTH_SUMMARY_RELPATH
	)
	assert summary_path.exists()
	on_disk = json.loads(summary_path.read_text())
	assert on_disk == summary


def test_kssynth_phase_reports_error_on_synthesize_failure(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)
	monkeypatch.setattr(kssynth_phase, "_load_segment_analyzers", lambda inputs: [object(), object()])

	def _boom(*, analyzers, out_folder, **_):
		raise RuntimeError("synthesize-failed-for-test")

	import kssynth.api

	monkeypatch.setattr(kssynth.api, "synthesize", _boom)

	summary = kssynth_phase.run_reconstruct_kssynth_phase(SimpleNamespace())

	assert summary["status"] == "error"
	assert summary["n_analyzers"] == 2
	assert "synthesize-failed-for-test" in summary["error"]


def test_kssynth_phase_reports_error_on_analyzer_load_failure(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)

	def _boom(inputs):
		raise RuntimeError("loader-failed-for-test")

	monkeypatch.setattr(kssynth_phase, "_load_segment_analyzers", _boom)

	summary = kssynth_phase.run_reconstruct_kssynth_phase(SimpleNamespace())

	assert summary["status"] == "error"
	assert "loader-failed-for-test" in summary["error"]


def test_kssynth_phase_reports_error_when_kssynth_missing(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)

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
		summary = kssynth_phase.run_reconstruct_kssynth_phase(SimpleNamespace())
	finally:
		sys.meta_path[:] = original_finders
		sys.modules.update(original_modules)

	assert summary["status"] == "error"
	assert "kssynth import failed" in summary["error"]
