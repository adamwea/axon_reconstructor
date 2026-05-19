"""Slice 14c of phase_roster_cleanup_plan: target-level auto-restart skip
for the analysis stage.

``target_analysis_phase_summary_ok`` returns True iff the per-target
per-phase summary on disk shows the phase is complete (status:
``ok`` / ``skipped`` / ``noop``). The two analysis phases use different
summary file conventions, so the test exercises both layouts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.analysis.runner import (
	target_analysis_phase_summary_ok,
)


@dataclass
class _ShimTarget:
	dataset_index: int
	stream_id: str
	h5_path: Path
	mea_output_root: Path


@dataclass
class _ShimStageConfig:
	output_rel_root: str = "analysis_outputs"
	manifest_relpath: str = "manifest.json"


_H5_TEMPLATE = "/raw/proj/proj/{date}/{chip}/AxonTracking/{run}/data.raw.h5"


def _make_target(tmp_path: Path) -> _ShimTarget:
	return _ShimTarget(
		dataset_index=0,
		stream_id="well000",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		mea_output_root=tmp_path,
	)


def _stage_output_root(target: _ShimTarget, stage_config: _ShimStageConfig) -> Path:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=target.mea_output_root,
		data_file=target.h5_path,
		well=target.stream_id,
	)
	return well_out_dir / stage_config.output_rel_root


def _write_status_file(path: Path, status: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps({"status": status}), encoding="utf-8")


# --- compute_metrics tests ----------------------------------------------------


def test_compute_metrics_ok_returns_true(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	manifest_path = _stage_output_root(target, stage_config) / stage_config.manifest_relpath
	_write_status_file(manifest_path, "ok")
	assert (
		target_analysis_phase_summary_ok(
			phase_name="compute_metrics",
			target=target,
			stage_config=stage_config,
		)
		is True
	)


def test_compute_metrics_missing_returns_false(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	assert (
		target_analysis_phase_summary_ok(
			phase_name="compute_metrics",
			target=target,
			stage_config=stage_config,
		)
		is False
	)


def test_compute_metrics_error_returns_false(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	manifest_path = _stage_output_root(target, stage_config) / stage_config.manifest_relpath
	_write_status_file(manifest_path, "error")
	assert (
		target_analysis_phase_summary_ok(
			phase_name="compute_metrics",
			target=target,
			stage_config=stage_config,
		)
		is False
	)


# --- unitmatch tests ----------------------------------------------------------


def test_unitmatch_ok_returns_true(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	summary_path = _stage_output_root(target, stage_config) / "context" / "unitmatch_summary.json"
	_write_status_file(summary_path, "ok")
	assert (
		target_analysis_phase_summary_ok(
			phase_name="unitmatch",
			target=target,
			stage_config=stage_config,
		)
		is True
	)


def test_unitmatch_noop_returns_true(tmp_path: Path) -> None:
	# When the phase is disabled in YAML, the orchestrator writes
	# status: noop. That counts as "already done" for slice 14c — the
	# runner should skip cleanly without re-dispatching the orchestrator.
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	summary_path = _stage_output_root(target, stage_config) / "context" / "unitmatch_summary.json"
	_write_status_file(summary_path, "noop")
	assert (
		target_analysis_phase_summary_ok(
			phase_name="unitmatch",
			target=target,
			stage_config=stage_config,
		)
		is True
	)


def test_unitmatch_skipped_returns_true(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	summary_path = _stage_output_root(target, stage_config) / "context" / "unitmatch_summary.json"
	_write_status_file(summary_path, "skipped")
	assert (
		target_analysis_phase_summary_ok(
			phase_name="unitmatch",
			target=target,
			stage_config=stage_config,
		)
		is True
	)


def test_unitmatch_missing_returns_false(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	assert (
		target_analysis_phase_summary_ok(
			phase_name="unitmatch",
			target=target,
			stage_config=stage_config,
		)
		is False
	)


# --- unknown phase + malformed summary ----------------------------------------


def test_unknown_phase_returns_false(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	assert (
		target_analysis_phase_summary_ok(
			phase_name="propagation_video",
			target=target,
			stage_config=stage_config,
		)
		is False
	)


def test_malformed_summary_returns_false(tmp_path: Path) -> None:
	target = _make_target(tmp_path)
	stage_config = _ShimStageConfig()
	manifest_path = _stage_output_root(target, stage_config) / stage_config.manifest_relpath
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text("not-json", encoding="utf-8")
	assert (
		target_analysis_phase_summary_ok(
			phase_name="compute_metrics",
			target=target,
			stage_config=stage_config,
		)
		is False
	)
