"""Slice 2 of analysis_propagation_video_plan: scaffold tests.

Verifies the phase wires into the analysis stage, defaults to disabled
(noop), and returns a `skipped: not_implemented_yet` marker when
enabled but the core impl (slice 4) hasn't shipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from axon_recon.pipeline.stages.analysis.config import (
	DEFAULT_ANALYSIS_PHASE_SEQUENCE,
	AnalysisStageConfig,
	parse_analysis_stage_config,
)
from axon_recon.pipeline.stages.analysis.orchestrators.propagation_video import (
	run_analysis_propagation_video,
)
from axon_recon.runtime_config import RuntimeConfig


_H5_TEMPLATE = "/pscratch/raw_data/proj/proj/{date}/{chip}/AxonTracking/{run}/data.raw.h5"


@dataclass
class _ShimStageConfig:
	parsed: AnalysisStageConfig
	propagation_video_enabled_override: bool | None = None

	def __getattr__(self, name: str) -> Any:
		if name == "propagation_video_enabled" and self.__dict__["propagation_video_enabled_override"] is not None:
			return self.__dict__["propagation_video_enabled_override"]
		return getattr(self.__dict__["parsed"], name)


def _parse_config(*, enabled: bool) -> _ShimStageConfig:
	payload = {
		"stages": {
			"analysis": {
				"phase_sequence": ["compute_metrics", "unitmatch", "propagation_video"],
				"phases": {
					"compute_metrics": {"enabled": True},
					"unitmatch": {"enabled": False},
					"propagation_video": {"enabled": enabled},
				},
			}
		}
	}
	parsed = parse_analysis_stage_config(runtime_config=RuntimeConfig(payload))
	return _ShimStageConfig(parsed=parsed, propagation_video_enabled_override=enabled)


# --- Slice 1 / 2 wiring tests ---------------------------------------------


def test_propagation_video_in_default_phase_sequence() -> None:
	assert "propagation_video" in DEFAULT_ANALYSIS_PHASE_SEQUENCE


def test_propagation_video_defaults_to_disabled() -> None:
	cfg = parse_analysis_stage_config(
		runtime_config=RuntimeConfig({"stages": {"analysis": {}}}),
	)
	assert cfg.propagation_video_enabled is False
	assert cfg.propagation_video_rel_output_root == "propagation_video"


def test_propagation_video_yaml_enable() -> None:
	cfg = parse_analysis_stage_config(
		runtime_config=RuntimeConfig(
			{
				"stages": {
					"analysis": {
						"phases": {"propagation_video": {"enabled": True}},
					}
				}
			}
		),
	)
	assert cfg.propagation_video_enabled is True


def test_yaml_alias_canonicalizes() -> None:
	# `video` and `prop_video` aliases should both resolve to the
	# canonical phase name in phase_sequence parsing.
	cfg = parse_analysis_stage_config(
		runtime_config=RuntimeConfig(
			{
				"stages": {
					"analysis": {
						"phase_sequence": ["compute_metrics", "video"],
					}
				}
			}
		),
	)
	assert "propagation_video" in cfg.phase_sequence


# --- Slice 2 orchestrator tests --------------------------------------------


def test_orchestrator_noop_when_disabled(tmp_path: Path) -> None:
	cfg = _parse_config(enabled=False)
	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "noop"
	assert result["phase"] == "propagation_video"
	assert "disabled" in result["reason"]


def test_orchestrator_skipped_not_implemented_when_enabled(tmp_path: Path) -> None:
	cfg = _parse_config(enabled=True)
	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "skipped"
	assert "not_implemented_yet" in result["reason"]


def test_orchestrator_writes_per_target_summary_on_disk(tmp_path: Path) -> None:
	cfg = _parse_config(enabled=False)
	run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	# Walk under tmp_path for the summary file.
	matches = list((tmp_path / "outputs").rglob("propagation_video_summary.json"))
	assert len(matches) == 1
	payload = json.loads(matches[0].read_text(encoding="utf-8"))
	assert payload["phase"] == "propagation_video"
	assert payload["status"] == "noop"
