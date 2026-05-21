"""Tests for `preprocess._run_preprocess_selected_phase` dry-run short-circuit.

Single-intercept dry-run rollout for the preprocess stage: all preprocess
phases (save_rec_metadata, preprocess_segments, plot_segment_traces,
plot_segment_channel_layouts, plot_raster_threshold) dispatch through
`_run_preprocess_selected_phase`, and the dry-run check is at the top
of that helper. One test per phase verifies the wiring + that
`_run_preprocess_phase_sequence` (the heavy fanout) is NOT called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axon_recon.pipeline.stages.preprocess import runner as preprocess_runner
from axon_recon.pipeline.stages.preprocess.models.inputs import (
	PreprocessInputs,
	PreprocessPhasesConfig,
	PreprocessPlotRasterThresholdPhaseConfig,
	PreprocessPlotSegmentChannelLayoutsPhaseConfig,
	PreprocessPlotSegmentTracesPhaseConfig,
	PreprocessSaveRecMetadataPhaseConfig,
	PreprocessSegmentsPhaseConfig,
)


@pytest.fixture(autouse=True)
def _reset_dry_run_override():
	from axon_recon.pipeline.config import set_dry_run_override

	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def _build_inputs(tmp_path: Path) -> PreprocessInputs:
	"""Minimal PreprocessInputs — only what the dry-run path reads."""
	return PreprocessInputs(
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path,
		phases=PreprocessPhasesConfig(
			save_rec_metadata=PreprocessSaveRecMetadataPhaseConfig(enabled=True),
			preprocess_segments=PreprocessSegmentsPhaseConfig(enabled=True),
			plot_segment_traces=PreprocessPlotSegmentTracesPhaseConfig(enabled=True),
			plot_segment_channel_layouts=PreprocessPlotSegmentChannelLayoutsPhaseConfig(
				enabled=True
			),
			plot_raster_threshold=PreprocessPlotRasterThresholdPhaseConfig(enabled=True),
		),
	)


def _stub_compute_well_out(monkeypatch, tmp_path: Path) -> Path:
	well_out_dir = tmp_path / "well_out"

	monkeypatch.setattr(
		preprocess_runner,
		"compute_mea_analysis_output_dir",
		lambda *, output_root, data_file, well: well_out_dir,
	)
	return well_out_dir


def _fail_phase_sequence(monkeypatch):
	"""Stub `_run_preprocess_phase_sequence` to raise — confirms dry-run
	short-circuit means the heavy fanout is NEVER reached."""

	def _fail(**_):
		raise AssertionError("phase sequence should not run under dry-run")

	monkeypatch.setattr(preprocess_runner, "_run_preprocess_phase_sequence", _fail)


@pytest.mark.parametrize(
	"phase_name,phase_relpath",
	[
		("save_rec_metadata", "context/recording_metadata_summary.json"),
		("preprocess_segments", "context/segment_recordings_summary.json"),
		("plot_segment_traces", "context/plot_segment_traces_summary.json"),
		(
			"plot_segment_channel_layouts",
			"context/plot_segment_channel_layouts_summary.json",
		),
		("plot_raster_threshold", "context/plot_raster_threshold_summary.json"),
	],
)
def test_preprocess_phase_dry_run_writes_summary(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
	phase_name: str,
	phase_relpath: str,
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	well_out_dir = _stub_compute_well_out(monkeypatch, tmp_path)
	_fail_phase_sequence(monkeypatch)

	inputs = _build_inputs(tmp_path)
	set_dry_run_override(True)

	result = preprocess_runner._run_preprocess_selected_phase(
		inputs, selected_phase=phase_name
	)
	assert result["status"] == "dry_run_ok"
	assert result["phase"] == f"preprocess.{phase_name}"

	# Summary at the per-phase relpath under preprocess_out_dir.
	preprocess_out_dir = well_out_dir / "preprocess_outputs"
	summary_path = preprocess_out_dir / Path(phase_relpath)
	assert summary_path.exists(), f"missing summary for {phase_name} at {summary_path}"
	on_disk = json.loads(summary_path.read_text())
	assert on_disk["status"] == "dry_run_ok"
	assert on_disk["phase"] == f"preprocess.{phase_name}"


def test_preprocess_dry_run_surfaces_missing_h5_warning(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	_stub_compute_well_out(monkeypatch, tmp_path)
	_fail_phase_sequence(monkeypatch)

	inputs = _build_inputs(tmp_path)  # h5_path doesn't exist
	set_dry_run_override(True)

	result = preprocess_runner._run_preprocess_selected_phase(
		inputs, selected_phase="save_rec_metadata"
	)
	assert result["status"] == "dry_run_ok"

	well_out_dir = tmp_path / "well_out"
	summary_path = (
		well_out_dir / "preprocess_outputs" / "context" / "recording_metadata_summary.json"
	)
	on_disk = json.loads(summary_path.read_text())
	# H5 missing → warning surfaced.
	assert any(
		"h5_path not found" in w for w in on_disk["validation"]["warnings"]
	)
	# inputs_resolved has h5_path entry with exists=False.
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["h5_path"]["exists"] is False


def test_preprocess_dry_run_existing_h5_no_warning(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	from axon_recon.pipeline.config import set_dry_run_override

	_stub_compute_well_out(monkeypatch, tmp_path)
	_fail_phase_sequence(monkeypatch)

	# Materialize h5 file
	h5_path = tmp_path / "input.raw.h5"
	h5_path.write_bytes(b"")

	inputs = _build_inputs(tmp_path)
	set_dry_run_override(True)

	result = preprocess_runner._run_preprocess_selected_phase(
		inputs, selected_phase="preprocess_segments"
	)
	assert result["status"] == "dry_run_ok"

	well_out_dir = tmp_path / "well_out"
	summary_path = (
		well_out_dir / "preprocess_outputs" / "context" / "segment_recordings_summary.json"
	)
	on_disk = json.loads(summary_path.read_text())
	by_name = {item["name"]: item for item in on_disk["inputs_resolved"]}
	assert by_name["h5_path"]["exists"] is True
	assert not any(
		"h5_path not found" in w for w in on_disk["validation"]["warnings"]
	)
