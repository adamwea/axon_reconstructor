"""Tests for `pipeline/dry_run.py::write_dry_run_summary`.

Enforces the schema contract from `guardrails/dry_run.md` §3 so
individual phases consuming this helper can rely on the base fields
without drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from axon_recon.pipeline.dry_run import DRY_RUN_STATUS, write_dry_run_summary


def test_write_dry_run_summary_writes_base_schema(tmp_path: Path) -> None:
	summary_path = tmp_path / "kssynth_summary.json"
	well_out = tmp_path / "well_out"
	stage_root = tmp_path / "well_out" / "recon_outputs"

	returned = write_dry_run_summary(
		phase_name="reconstruct.kssynth",
		well_out_dir=well_out,
		stage_output_root_dir=stage_root,
		summary_json_path=summary_path,
		inputs_resolved=[
			{"name": "segment_analyzers", "path": str(stage_root / "cache/analyzers"), "exists": True},
		],
		outputs_would_produce=[
			{"name": "synth_sorter_output", "path": str(stage_root / "synth_sorter_output")},
			{"name": "summary_json", "path": str(summary_path)},
		],
	)

	assert returned == summary_path
	assert summary_path.exists()
	payload = json.loads(summary_path.read_text())

	assert payload["status"] == DRY_RUN_STATUS
	assert payload["phase"] == "reconstruct.kssynth"
	assert payload["well_out_dir"] == str(well_out)
	assert payload["stage_output_root_dir"] == str(stage_root)
	assert payload["inputs_resolved"] == [
		{"name": "segment_analyzers", "path": str(stage_root / "cache/analyzers"), "exists": True},
	]
	assert payload["outputs_would_produce"] == [
		{"name": "synth_sorter_output", "path": str(stage_root / "synth_sorter_output")},
		{"name": "summary_json", "path": str(summary_path)},
	]
	assert payload["validation"] == {"missing_prerequisites": [], "warnings": []}


def test_write_dry_run_summary_defaults_empty_validation(tmp_path: Path) -> None:
	summary_path = tmp_path / "p.json"
	write_dry_run_summary(
		phase_name="p",
		well_out_dir=tmp_path,
		stage_output_root_dir=tmp_path,
		summary_json_path=summary_path,
		inputs_resolved=[],
		outputs_would_produce=[],
	)
	payload = json.loads(summary_path.read_text())
	assert payload["validation"] == {"missing_prerequisites": [], "warnings": []}


def test_write_dry_run_summary_carries_validation(tmp_path: Path) -> None:
	summary_path = tmp_path / "p.json"
	write_dry_run_summary(
		phase_name="p",
		well_out_dir=tmp_path,
		stage_output_root_dir=tmp_path,
		summary_json_path=summary_path,
		inputs_resolved=[],
		outputs_would_produce=[],
		validation={
			"missing_prerequisites": [{"name": "sorter_output", "path": "/nope"}],
			"warnings": ["preproc segment 3 missing"],
		},
	)
	payload = json.loads(summary_path.read_text())
	assert payload["validation"]["missing_prerequisites"] == [
		{"name": "sorter_output", "path": "/nope"}
	]
	assert payload["validation"]["warnings"] == ["preproc segment 3 missing"]


def test_write_dry_run_summary_creates_parent_dirs(tmp_path: Path) -> None:
	# Nested under several non-existent dirs — helper should mkdir -p.
	summary_path = tmp_path / "a" / "b" / "c" / "summary.json"
	write_dry_run_summary(
		phase_name="p",
		well_out_dir=tmp_path,
		stage_output_root_dir=tmp_path,
		summary_json_path=summary_path,
		inputs_resolved=[],
		outputs_would_produce=[],
	)
	assert summary_path.exists()


def test_write_dry_run_summary_accepts_extra_fields(tmp_path: Path) -> None:
	summary_path = tmp_path / "p.json"
	write_dry_run_summary(
		phase_name="reconstruct.kssynth",
		well_out_dir=tmp_path,
		stage_output_root_dir=tmp_path,
		summary_json_path=summary_path,
		inputs_resolved=[],
		outputs_would_produce=[],
		extra_fields={
			"n_analyzers_resolved": 12,
			"channel_grid_mode": "union",
		},
	)
	payload = json.loads(summary_path.read_text())
	assert payload["n_analyzers_resolved"] == 12
	assert payload["channel_grid_mode"] == "union"
	# Base schema still present.
	assert payload["status"] == DRY_RUN_STATUS
	assert payload["phase"] == "reconstruct.kssynth"


def test_write_dry_run_summary_extras_cant_shadow_base_fields(tmp_path: Path) -> None:
	"""Base schema fields win on conflict — phase-specific extras can't
	clobber `status`, `phase`, `well_out_dir`, etc."""

	summary_path = tmp_path / "p.json"
	write_dry_run_summary(
		phase_name="p",
		well_out_dir=tmp_path,
		stage_output_root_dir=tmp_path,
		summary_json_path=summary_path,
		inputs_resolved=[],
		outputs_would_produce=[],
		extra_fields={
			"status": "ok",  # attempted shadow
			"phase": "different_phase",  # attempted shadow
		},
	)
	payload = json.loads(summary_path.read_text())
	assert payload["status"] == DRY_RUN_STATUS  # base wins
	assert payload["phase"] == "p"  # base wins


def test_write_dry_run_summary_normalizes_inputs_resolved_items(tmp_path: Path) -> None:
	"""Helper coerces input items to the canonical {name, path, exists}
	shape — callers can pass partial dicts and they get sanitized."""

	summary_path = tmp_path / "p.json"
	write_dry_run_summary(
		phase_name="p",
		well_out_dir=tmp_path,
		stage_output_root_dir=tmp_path,
		summary_json_path=summary_path,
		inputs_resolved=[
			{"name": "foo", "path": "/bar"},  # missing `exists` → False
			{"name": "baz", "path": "/qux", "exists": True},
		],
		outputs_would_produce=[],
	)
	payload = json.loads(summary_path.read_text())
	assert payload["inputs_resolved"][0]["exists"] is False
	assert payload["inputs_resolved"][1]["exists"] is True
