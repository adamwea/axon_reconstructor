"""Slice 14c of phase_roster_cleanup_plan: target-level auto-restart skip.

When every phase in the preprocess sequence has an ``ok`` summary on
disk for a given target, ``target_all_preprocess_phases_succeeded``
returns True so the runtime entry point can skip the monolithic
``run_preprocess`` call entirely. A single broken / missing / stale
summary returns False so normal dispatch proceeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axon_recon.pipeline.stages.preprocess.constants import PREPROCESS_OUTPUTS_DIRNAME
from axon_recon.pipeline.stages.preprocess.models.inputs import (
	PreprocessInputs,
	PreprocessPhaseConfig,
	PreprocessPhasesConfig,
)
from axon_recon.pipeline.stages.preprocess.runner import (
	target_all_preprocess_phases_succeeded,
)


def _make_inputs(
	tmp_path: Path,
	*,
	enabled_overrides: dict[str, bool] | None = None,
) -> PreprocessInputs:
	"""Build a minimal PreprocessInputs pointing at ``tmp_path``.

	The two-phase scenario uses ``save_rec_metadata`` and
	``preprocess_segments``; both default to enabled. Callers can pass
	``enabled_overrides`` to flip a phase's ``enabled`` flag.
	"""

	enabled = {"save_rec_metadata": True, "preprocess_segments": True}
	if enabled_overrides:
		enabled.update(enabled_overrides)

	def _phase(name: str, relpath: str) -> PreprocessPhaseConfig:
		return PreprocessPhaseConfig(
			enabled=enabled[name],
			summary_json_relpath=relpath,
		)

	phases = PreprocessPhasesConfig(
		save_rec_metadata=_phase("save_rec_metadata", "save_rec_metadata.json"),
		preprocess_segments=_phase("preprocess_segments", "preprocess_segments.json"),
	)
	return PreprocessInputs(
		h5_path=tmp_path / "data.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		phase_sequence=("save_rec_metadata", "preprocess_segments"),
		phases=phases,
		save_segment_recordings=True,
	)


def _write_ok_summary(path: Path, phase_name: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps({"phase": phase_name, "status": "ok"}), encoding="utf-8"
	)


def _summary_paths_for_inputs(
	inputs: PreprocessInputs, tmp_path: Path
) -> dict[str, Path]:
	from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir

	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	preprocess_out_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
	return {
		"save_rec_metadata": preprocess_out_dir / "save_rec_metadata.json",
		"preprocess_segments": preprocess_out_dir / "preprocess_segments.json",
	}


def test_returns_true_when_every_phase_summary_is_ok(tmp_path: Path) -> None:
	inputs = _make_inputs(tmp_path)
	paths = _summary_paths_for_inputs(inputs, tmp_path)
	_write_ok_summary(paths["save_rec_metadata"], "save_rec_metadata")
	_write_ok_summary(paths["preprocess_segments"], "preprocess_segments")
	assert target_all_preprocess_phases_succeeded(inputs) is True


def test_returns_false_when_a_phase_summary_is_missing(tmp_path: Path) -> None:
	inputs = _make_inputs(tmp_path)
	paths = _summary_paths_for_inputs(inputs, tmp_path)
	_write_ok_summary(paths["save_rec_metadata"], "save_rec_metadata")
	# preprocess_segments summary intentionally NOT written
	assert target_all_preprocess_phases_succeeded(inputs) is False


def test_yaml_disabled_phase_does_not_block_skip(tmp_path: Path) -> None:
	# When `preprocess_segments` is disabled in YAML, its missing summary
	# is intentional — the target should still be skippable if the other
	# phase is ok.
	inputs = _make_inputs(tmp_path, enabled_overrides={"preprocess_segments": False})
	paths = _summary_paths_for_inputs(inputs, tmp_path)
	_write_ok_summary(paths["save_rec_metadata"], "save_rec_metadata")
	assert target_all_preprocess_phases_succeeded(inputs) is True


def test_returns_false_when_summary_is_error(tmp_path: Path) -> None:
	inputs = _make_inputs(tmp_path)
	paths = _summary_paths_for_inputs(inputs, tmp_path)
	_write_ok_summary(paths["save_rec_metadata"], "save_rec_metadata")
	paths["preprocess_segments"].parent.mkdir(parents=True, exist_ok=True)
	paths["preprocess_segments"].write_text(
		json.dumps({"phase": "preprocess_segments", "status": "error"}),
		encoding="utf-8",
	)
	assert target_all_preprocess_phases_succeeded(inputs) is False


def test_empty_phase_sequence_returns_false(tmp_path: Path) -> None:
	# An empty sequence means there's nothing to skip ON, so the runner
	# should proceed normally (which run_preprocess will handle as a
	# no-op via its own internal dispatch).
	inputs = _make_inputs(tmp_path)
	from dataclasses import replace

	inputs = replace(inputs, phase_sequence=())
	assert target_all_preprocess_phases_succeeded(inputs) is False
