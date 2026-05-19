"""Slice-1 test for `unitmatch_phase_plan.md`: the analysis.unitmatch
phase wires into the pipeline and is a no-op when `enabled: false`.

Slices 2-5 will add behavior; this test just locks in the scaffold
contract so future slices can grow on top without re-litigating wiring.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.stages.analysis.config import (
	AnalysisStageConfig,
	parse_analysis_stage_config,
)
from axon_recon.pipeline.stages.analysis.orchestrators.unitmatch import (
	run_analysis_unitmatch,
)
from axon_recon.runtime_config import RuntimeConfig


def _make_stage_config(*, unitmatch_enabled: bool) -> AnalysisStageConfig:
	"""Build an AnalysisStageConfig with the unitmatch toggle flipped."""

	payload = {
		"stages": {
			"analysis": {
				"phase_sequence": ["compute_metrics", "unitmatch"],
				"phases": {
					"compute_metrics": {"enabled": True},
					"unitmatch": {"enabled": unitmatch_enabled},
				},
			}
		}
	}
	return parse_analysis_stage_config(runtime_config=RuntimeConfig(payload))


def test_parse_unitmatch_phase_block_defaults_disabled() -> None:
	cfg = _make_stage_config(unitmatch_enabled=False)
	assert cfg.unitmatch_enabled is False
	assert cfg.phase_sequence == ("compute_metrics", "unitmatch")


def test_parse_unitmatch_phase_block_can_be_enabled_in_yaml() -> None:
	cfg = _make_stage_config(unitmatch_enabled=True)
	assert cfg.unitmatch_enabled is True


def test_run_analysis_unitmatch_returns_noop_when_disabled(tmp_path: Path) -> None:
	cfg = _make_stage_config(unitmatch_enabled=False)
	result = run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "noop"
	assert result["phase"] == "unitmatch"
	assert result["enabled"] is False
	assert "disabled" in result["reason"]


def test_run_analysis_unitmatch_returns_scaffold_marker_when_enabled(tmp_path: Path) -> None:
	# Slice 1 is scaffold-only — even when enabled, the orchestrator
	# returns a "scaffold" marker until slice 2+ ships real logic.
	cfg = _make_stage_config(unitmatch_enabled=True)
	result = run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "scaffold"
	assert result["enabled"] is True
	assert "scaffold-only" in result["reason"]


def test_analysis_stage_default_sequence_includes_unitmatch() -> None:
	# Default-from-empty-YAML still adds unitmatch to the sequence; its
	# `enabled: false` default keeps it a no-op until the user opts in.
	cfg = parse_analysis_stage_config(runtime_config=RuntimeConfig({"stages": {"analysis": {}}}))
	assert "unitmatch" in cfg.phase_sequence
	assert cfg.unitmatch_enabled is False
