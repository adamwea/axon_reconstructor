"""Slice 14c of phase_roster_cleanup_plan: target-level auto-restart skip
for the spikesort stage (partial coverage).

The helper short-circuits the per-target setup cost when every phase in
the phase_plan has a convention-named summary path on stage_config AND
that summary on disk shows ok. Phase plans containing phases without a
convention-named summary relpath (sort, summarize_sort, merge_*,
snapshot_sorter_output, concat_analyzer) fall through to normal
dispatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.spikesort.runner import (
	target_all_spikesort_phases_succeeded,
)


@dataclass
class _ShimTarget:
	dataset_index: int
	stream_id: str
	h5_path: Path
	mea_output_root: Path


@dataclass
class _ShimPhase:
	phase_label: str


_H5_TEMPLATE = "/raw/proj/proj/{date}/{chip}/AxonTracking/{run}/data.raw.h5"


def _make_target(tmp_path: Path) -> _ShimTarget:
	return _ShimTarget(
		dataset_index=0,
		stream_id="well000",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		mea_output_root=tmp_path,
	)


def _stage_output_root(target: _ShimTarget) -> Path:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=target.mea_output_root,
		data_file=target.h5_path,
		well=target.stream_id,
	)
	return well_out_dir / "spikesort_outputs"


def _write_ok(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")


def _make_stage_config(**overrides: Any) -> Any:
	"""SimpleNamespace-like stage config covering all the relpath attrs the
	helper might inspect. Defaults match the production stage_config
	conventions."""

	defaults = {
		"output_rel_root": "spikesort_outputs",
		"concat_binary_summary_json_relpath": "cache/concat_binary/concat_binary_summary.json",
		"plot_concat_traces_summary_json_relpath": "plots/plot_concat_traces_summary.json",
		"plot_concat_channel_layout_summary_json_relpath": "plots/plot_concat_channel_layout_summary.json",
		"cleanup_concat_binary_summary_json_relpath": "cache/concat_binary_cleanup_summary.json",
		"cleanup_analyzers_summary_json_relpath": "cache/cleanup_analyzers_summary.json",
		"bombcell_label_reports_summary_json_relpath": "bombcell/bombcell_label_summary.json",
		"bombcell_label_pass2_summary_json_relpath": "bombcell/bombcell_label_pass2_summary.json",
	}
	defaults.update(overrides)
	from types import SimpleNamespace

	return SimpleNamespace(**defaults)


# --- happy path ---------------------------------------------------------------


def test_returns_true_when_every_phase_summary_is_ok(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _make_stage_config()
	phase_plan = [
		_ShimPhase(phase_label="cleanup_concat_binary"),
		_ShimPhase(phase_label="cleanup_analyzers"),
	]
	stage_root = _stage_output_root(target)
	_write_ok(stage_root / stage_config.cleanup_concat_binary_summary_json_relpath)
	_write_ok(stage_root / stage_config.cleanup_analyzers_summary_json_relpath)
	assert (
		target_all_spikesort_phases_succeeded(
			target=target, stage_config=stage_config, phase_plan=phase_plan
		)
		is True
	)


def test_returns_false_when_one_phase_summary_missing(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _make_stage_config()
	phase_plan = [
		_ShimPhase(phase_label="cleanup_concat_binary"),
		_ShimPhase(phase_label="cleanup_analyzers"),
	]
	stage_root = _stage_output_root(target)
	_write_ok(stage_root / stage_config.cleanup_concat_binary_summary_json_relpath)
	# cleanup_analyzers summary intentionally NOT written
	assert (
		target_all_spikesort_phases_succeeded(
			target=target, stage_config=stage_config, phase_plan=phase_plan
		)
		is False
	)


# --- bombcell coverage (different attribute-name convention) ------------------


def test_bombcell_label_uses_reports_attr(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _make_stage_config()
	phase_plan = [_ShimPhase(phase_label="bombcell_label")]
	stage_root = _stage_output_root(target)
	_write_ok(stage_root / stage_config.bombcell_label_reports_summary_json_relpath)
	assert (
		target_all_spikesort_phases_succeeded(
			target=target, stage_config=stage_config, phase_plan=phase_plan
		)
		is True
	)


# --- fall-through paths -------------------------------------------------------


def test_unmappable_phase_in_plan_returns_false(tmp_path: Path) -> None:
	# The `sort` phase has no convention-named relpath on stage_config,
	# so the helper must fall through (return False) even if all other
	# phases are ok.
	target = _make_target(tmp_path)
	stage_config = _make_stage_config()
	phase_plan = [
		_ShimPhase(phase_label="cleanup_concat_binary"),
		_ShimPhase(phase_label="sort"),  # not in the mapping
	]
	stage_root = _stage_output_root(target)
	_write_ok(stage_root / stage_config.cleanup_concat_binary_summary_json_relpath)
	assert (
		target_all_spikesort_phases_succeeded(
			target=target, stage_config=stage_config, phase_plan=phase_plan
		)
		is False
	)


def test_empty_phase_plan_returns_false(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _make_stage_config()
	assert (
		target_all_spikesort_phases_succeeded(
			target=target, stage_config=stage_config, phase_plan=[]
		)
		is False
	)


def test_missing_relpath_attr_returns_false(tmp_path: Path) -> None:
	# Stage config missing the expected attribute should fall through.
	target = _make_target(tmp_path)
	from types import SimpleNamespace

	stage_config = SimpleNamespace(output_rel_root="spikesort_outputs")
	phase_plan = [_ShimPhase(phase_label="cleanup_concat_binary")]
	assert (
		target_all_spikesort_phases_succeeded(
			target=target, stage_config=stage_config, phase_plan=phase_plan
		)
		is False
	)
