"""Slice 14c of phase_roster_cleanup_plan: target-level auto-restart skip
for the reconstruct stage.

When every phase in the reconstruct sequence has an ``ok`` summary on
disk for a given target, ``target_all_reconstruct_phases_succeeded``
returns True so the runtime entry point can skip the monolithic
``run_reconstruct`` call entirely. A single broken / missing summary
returns False so normal dispatch proceeds.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.reconstruct.models.inputs import (
	ReconstructionGenerateGtrsPhaseConfig,
	ReconstructionInputs,
	ReconstructionPhasesConfig,
	ReconstructionPlotReconsPhaseConfig,
)
from axon_recon.pipeline.stages.reconstruct.runner import (
	target_all_reconstruct_phases_succeeded,
)


def _make_inputs(
	tmp_path: Path,
	*,
	enabled_overrides: dict[str, bool] | None = None,
) -> ReconstructionInputs:
	"""Build a minimal ReconstructionInputs over two phases."""

	enabled = {"axon_velocity_gtrs": True, "plot_recons": True}
	if enabled_overrides:
		enabled.update(enabled_overrides)
	phases = ReconstructionPhasesConfig(
		axon_velocity_gtrs=ReconstructionGenerateGtrsPhaseConfig(
			enabled=enabled["axon_velocity_gtrs"],
			summary_json_relpath="context/axon_velocity_gtrs_summary.json",
		),
		plot_recons=ReconstructionPlotReconsPhaseConfig(
			enabled=enabled["plot_recons"],
			summary_json_relpath="context/plot_recons_summary.json",
		),
	)
	return ReconstructionInputs(
		h5_path=tmp_path / "data.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		phase_sequence=("axon_velocity_gtrs", "plot_recons"),
		phases=phases,
	)


def _summary_path(inputs: ReconstructionInputs, phase_name: str) -> Path:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	recon_out_dir = well_out_dir / str(inputs.output_rel_root)
	phase_cfg = getattr(inputs.phases, phase_name)
	return recon_out_dir / Path(phase_cfg.summary_json_relpath)


def _write_status(path: Path, phase_name: str, status: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps({"phase": phase_name, "status": status}),
		encoding="utf-8",
	)


def test_returns_true_when_every_phase_summary_is_ok(tmp_path: Path) -> None:
	inputs = _make_inputs(tmp_path)
	_write_status(_summary_path(inputs, "axon_velocity_gtrs"), "axon_velocity_gtrs", "ok")
	_write_status(_summary_path(inputs, "plot_recons"), "plot_recons", "ok")
	assert target_all_reconstruct_phases_succeeded(inputs) is True


def test_returns_false_when_a_phase_summary_is_missing(tmp_path: Path) -> None:
	inputs = _make_inputs(tmp_path)
	_write_status(_summary_path(inputs, "axon_velocity_gtrs"), "axon_velocity_gtrs", "ok")
	# plot_recons summary intentionally NOT written
	assert target_all_reconstruct_phases_succeeded(inputs) is False


def test_yaml_disabled_phase_does_not_block_skip(tmp_path: Path) -> None:
	inputs = _make_inputs(tmp_path, enabled_overrides={"plot_recons": False})
	_write_status(_summary_path(inputs, "axon_velocity_gtrs"), "axon_velocity_gtrs", "ok")
	assert target_all_reconstruct_phases_succeeded(inputs) is True


def test_returns_false_when_summary_is_error(tmp_path: Path) -> None:
	inputs = _make_inputs(tmp_path)
	_write_status(_summary_path(inputs, "axon_velocity_gtrs"), "axon_velocity_gtrs", "ok")
	_write_status(_summary_path(inputs, "plot_recons"), "plot_recons", "error")
	assert target_all_reconstruct_phases_succeeded(inputs) is False


def test_empty_phase_sequence_returns_false(tmp_path: Path) -> None:
	# None phase_sequence (the monolithic-only dispatch path) means
	# there's nothing to skip on; fall back to normal dispatch.
	inputs = _make_inputs(tmp_path)
	inputs = replace(inputs, phase_sequence=())
	assert target_all_reconstruct_phases_succeeded(inputs) is False


def test_none_phase_sequence_returns_false(tmp_path: Path) -> None:
	inputs = _make_inputs(tmp_path)
	inputs = replace(inputs, phase_sequence=None)
	assert target_all_reconstruct_phases_succeeded(inputs) is False
