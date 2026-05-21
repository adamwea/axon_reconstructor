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

import numpy as np
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


def _materialize_fake_synth_output(out_folder: Path, *, unit_ids, n_channels=8, n_samples=20):
	"""Write minimal templates.npy + channel_positions.npy so the slice-4 S4-B
	postprocess has something to demux. The templates are deterministic but
	carry a non-zero signal on a SUBSET of channels per unit so the
	sparsification path is exercised."""
	out_folder.mkdir(parents=True, exist_ok=True)
	n_units = len(unit_ids)
	templates = np.zeros((n_units, n_samples, n_channels), dtype=np.float32)
	# Each unit gets non-zero signal on channels [unit_idx, unit_idx+2].
	for i in range(n_units):
		ch_a = i % n_channels
		ch_b = (i + 2) % n_channels
		templates[i, :, ch_a] = np.linspace(-1.0, 1.0, n_samples)
		templates[i, :, ch_b] = np.linspace(0.5, -0.5, n_samples)
	positions = np.column_stack(
		[
			np.arange(n_channels, dtype=np.float32) * 10.0,
			np.zeros(n_channels, dtype=np.float32),
		]
	)
	np.save(out_folder / "templates.npy", templates)
	np.save(out_folder / "channel_positions.npy", positions)


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
		# Materialize templates.npy + channel_positions.npy so the slice-4
		# S4-B postprocess has something to demux.
		_materialize_fake_synth_output(
			Path(out_folder), unit_ids=(10, 11, 12, 13), n_channels=128
		)
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
	# Slice 4 S4-B: per-unit files exist for each unit_id.
	assert summary["per_unit_n_units_written"] == 4
	per_unit_dir = Path(summary["per_unit_dir"])
	for unit_id in (10, 11, 12, 13):
		assert (per_unit_dir / f"unit_{unit_id}" / "merged_template.npy").exists()
		assert (per_unit_dir / f"unit_{unit_id}" / "merged_channel_locations.npy").exists()

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

	def _fake_synthesize(*, analyzers, out_folder, **_):
		_materialize_fake_synth_output(Path(out_folder), unit_ids=(0, 1, 2))
		return _fake_writer_result()

	monkeypatch.setattr(kssynth.api, "synthesize", _fake_synthesize)

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


# ---------------------------------------------------------------------
# Slice 4 S4-B: per-unit postprocess helper (no monkey-patching needed;
# the helper is a pure function over disk artifacts).
# ---------------------------------------------------------------------


def test_per_unit_postprocess_writes_sparse_files(tmp_path: Path) -> None:
	"""Writes per-unit merged_template.npy + merged_channel_locations.npy
	with sparsification applied (only non-zero channels survive)."""

	synth_out = tmp_path / "synth_sorter_output"
	_materialize_fake_synth_output(synth_out, unit_ids=(7, 8, 9), n_channels=10, n_samples=15)

	per_unit_dir, n_written = kssynth_phase._write_per_unit_templates_from_synth_output(
		synth_out_dir=synth_out,
		unit_ids=(7, 8, 9),
	)

	assert n_written == 3
	for unit_id in (7, 8, 9):
		unit_dir = per_unit_dir / f"unit_{unit_id}"
		assert (unit_dir / "merged_template.npy").exists()
		assert (unit_dir / "merged_channel_locations.npy").exists()
		template = np.load(unit_dir / "merged_template.npy")
		locs = np.load(unit_dir / "merged_channel_locations.npy")
		# Sparsified to 2 active channels per unit (the test fixture sets
		# non-zero signal on exactly 2 channels per unit).
		assert template.shape == (2, 15)
		assert locs.shape == (2, 2)


def test_per_unit_postprocess_raises_on_missing_templates(tmp_path: Path) -> None:
	synth_out = tmp_path / "synth_sorter_output"
	synth_out.mkdir(parents=True, exist_ok=True)
	# No templates.npy / channel_positions.npy written.

	with pytest.raises(FileNotFoundError, match="missing"):
		kssynth_phase._write_per_unit_templates_from_synth_output(
			synth_out_dir=synth_out,
			unit_ids=(1, 2, 3),
		)


def test_per_unit_postprocess_raises_on_unit_id_length_mismatch(tmp_path: Path) -> None:
	synth_out = tmp_path / "synth_sorter_output"
	_materialize_fake_synth_output(synth_out, unit_ids=(1, 2, 3), n_channels=8, n_samples=10)

	# Pass mismatched unit_ids count (5 ids vs 3 templates).
	with pytest.raises(ValueError, match="unit_ids length"):
		kssynth_phase._write_per_unit_templates_from_synth_output(
			synth_out_dir=synth_out,
			unit_ids=(1, 2, 3, 4, 5),
		)


# ---------------------------------------------------------------------
# Slice 4e: dry-run short-circuit. When `get_dry_run_override()` is True,
# the phase writes a `dry_run_ok` summary without invoking
# `kssynth.synthesize` or loading analyzers.
# ---------------------------------------------------------------------


@pytest.fixture
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def test_kssynth_phase_dry_run_writes_summary_and_skips_synthesize(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)
	# These should NOT be called during dry-run.
	analyzers_load_called = {"count": 0}
	synthesize_called = {"count": 0}

	def _fail_load(_inputs):
		analyzers_load_called["count"] += 1
		raise AssertionError("analyzer load should not happen in dry-run")

	def _fail_synthesize(**_):
		synthesize_called["count"] += 1
		raise AssertionError("synthesize should not happen in dry-run")

	monkeypatch.setattr(kssynth_phase, "_load_segment_analyzers", _fail_load)

	import kssynth.api

	monkeypatch.setattr(kssynth.api, "synthesize", _fail_synthesize)

	# Stub the build_templates context lookup so the dry-run can fill in
	# the analyzer_cache_dir input without needing a real TemplatesInputs.
	fake_cache_dir = tmp_path / "well_out" / "recon_outputs" / "cache" / "analyzers"

	class _FakeCtx:
		analyzer_cache_dir = fake_cache_dir

	def _fake_resolve_ctx(_inputs):
		return _FakeCtx()

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.phases.build_templates._resolve_build_templates_context",
		_fake_resolve_ctx,
	)

	set_dry_run_override(True)

	result = kssynth_phase.run_reconstruct_kssynth_phase(SimpleNamespace())

	assert result["status"] == "dry_run_ok"
	assert result["phase"] == "reconstruct.kssynth"
	assert analyzers_load_called["count"] == 0
	assert synthesize_called["count"] == 0

	summary_path = (
		tmp_path / "well_out" / "recon_outputs" / kssynth_phase.KSSYNTH_SUMMARY_RELPATH
	)
	assert summary_path.exists()
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	assert on_disk["phase"] == "reconstruct.kssynth"
	# Cache dir is reported as not-existing — fixture didn't create it.
	cache_entries = [
		item for item in on_disk["inputs_resolved"] if item["name"] == "analyzer_cache_dir"
	]
	assert len(cache_entries) == 1
	assert cache_entries[0]["exists"] is False
	# Validation warning about the missing cache is surfaced.
	assert any(
		"analyzer_cache_dir not found" in w
		for w in on_disk["validation"]["warnings"]
	)
	# Outputs that WOULD be produced.
	output_names = [item["name"] for item in on_disk["outputs_would_produce"]]
	assert "synth_sorter_output" in output_names
	assert "per_unit_dir" in output_names
	assert "summary_json" in output_names


def test_kssynth_phase_dry_run_reports_existing_cache(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _reset_dry_run_override
) -> None:
	"""When the analyzer cache directory exists, dry-run reports it
	without the missing-cache warning."""

	from axon_recon.pipeline.config import set_dry_run_override

	monkeypatch.setattr(
		kssynth_phase, "_resolve_kssynth_output_dirs", _stub_resolve(tmp_path)
	)
	monkeypatch.setattr(kssynth_phase, "_load_segment_analyzers", lambda _: [object()])

	fake_cache_dir = tmp_path / "well_out" / "recon_outputs" / "cache" / "analyzers"
	fake_cache_dir.mkdir(parents=True, exist_ok=True)

	class _FakeCtx:
		analyzer_cache_dir = fake_cache_dir

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.reconstruct.phases.build_templates._resolve_build_templates_context",
		lambda _: _FakeCtx(),
	)

	set_dry_run_override(True)

	result = kssynth_phase.run_reconstruct_kssynth_phase(SimpleNamespace())
	assert result["status"] == "dry_run_ok"

	summary_path = (
		tmp_path / "well_out" / "recon_outputs" / kssynth_phase.KSSYNTH_SUMMARY_RELPATH
	)
	on_disk = json.loads(summary_path.read_text())
	cache_entries = [
		item for item in on_disk["inputs_resolved"] if item["name"] == "analyzer_cache_dir"
	]
	assert cache_entries[0]["exists"] is True
	# No missing-cache warning when the cache is present.
	assert not any(
		"analyzer_cache_dir not found" in w
		for w in on_disk["validation"]["warnings"]
	)
