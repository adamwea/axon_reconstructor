from __future__ import annotations

import pytest

from axon_recon.runtime_config import RuntimeConfig

from ..config import CleanupStageConfig, parse_cleanup_stage_config
from ..models.inputs import (
	DEFAULT_CLEANUP_PHASE_SEQUENCE,
	CleanupPhasesConfig,
	CleanupWipeSrcScratchPhaseConfig,
)
from ..runner import run_cleanup_stage


def test_parse_cleanup_stage_config_defaults() -> None:
	"""With no `stages.cleanup` block present, the parser returns all defaults.

	Defaults stay disabled with an empty phase_sequence so the runner's
	no-op branch is hit on legacy configs that predate slice 6.
	"""

	parsed = parse_cleanup_stage_config(runtime_config=RuntimeConfig({}))
	assert isinstance(parsed, CleanupStageConfig)
	assert parsed.enabled is False
	assert parsed.phase_sequence == ()
	assert parsed.output_rel_root == "cleanup_outputs"
	assert parsed.force_restart is False
	assert parsed.replot is False
	assert parsed.phases.wipe_src_scratch.enabled is False
	assert parsed.phases.wipe_src_scratch.dry_run is False
	assert parsed.phases.wipe_src_scratch.requires_use_scratch_root is False
	assert parsed.phases.wipe_src_scratch.summary_json_relpath == "context/wipe_src_scratch_summary.json"


def test_parse_cleanup_stage_config_disabled_yaml() -> None:
	"""Explicit `enabled: false` + empty `phase_sequence: []` round-trip cleanly."""

	cfg = RuntimeConfig(
		{
			"stages": {
				"cleanup": {
					"enabled": False,
					"phase_sequence": [],
					"phases": {},
				}
			}
		}
	)
	parsed = parse_cleanup_stage_config(runtime_config=cfg)
	assert parsed.enabled is False
	assert parsed.phase_sequence == ()


def test_parse_cleanup_stage_config_honors_force_overrides() -> None:
	"""CLI overrides flow through into the dataclass, like the other stages."""

	parsed = parse_cleanup_stage_config(
		runtime_config=RuntimeConfig({}),
		force_restart_override=True,
		replot_override=True,
	)
	assert parsed.force_restart is True
	assert parsed.replot is True


def test_parse_cleanup_stage_config_reads_wipe_phase_overrides() -> None:
	"""`stages.cleanup.phases.wipe_src_scratch` knobs flow into the dataclass."""

	cfg = RuntimeConfig(
		{
			"stages": {
				"cleanup": {
					"enabled": True,
					"phase_sequence": ["wipe_src_scratch"],
					"phases": {
						"wipe_src_scratch": {
							"enabled": True,
							"dry_run": True,
							"requires_use_scratch_root": True,
							"summary_json_relpath": "context/custom_wipe_summary.json",
							"resource_class": "disk_cleanup",
						},
					},
				}
			}
		}
	)
	parsed = parse_cleanup_stage_config(runtime_config=cfg)
	assert parsed.enabled is True
	assert parsed.phase_sequence == ("wipe_src_scratch",)
	assert parsed.phases.wipe_src_scratch.enabled is True
	assert parsed.phases.wipe_src_scratch.dry_run is True
	assert parsed.phases.wipe_src_scratch.requires_use_scratch_root is True
	assert parsed.phases.wipe_src_scratch.summary_json_relpath == "context/custom_wipe_summary.json"
	assert parsed.phases.wipe_src_scratch.resource_class == "disk_cleanup"


def test_parse_cleanup_stage_config_accepts_dotted_phase_names() -> None:
	"""`cleanup.wipe_src_scratch` round-trips to the bare phase name."""

	cfg = RuntimeConfig(
		{
			"stages": {
				"cleanup": {
					"enabled": True,
					"phase_sequence": ["cleanup.wipe_src_scratch"],
				}
			}
		}
	)
	parsed = parse_cleanup_stage_config(runtime_config=cfg)
	assert parsed.phase_sequence == ("wipe_src_scratch",)


def test_parse_cleanup_stage_config_rejects_unknown_phase() -> None:
	"""A phase_sequence entry that doesn't match a known cleanup phase fails loud."""

	cfg = RuntimeConfig(
		{
			"stages": {
				"cleanup": {
					"enabled": True,
					"phase_sequence": ["not_a_real_phase"],
				}
			}
		}
	)
	with pytest.raises(ValueError, match="Unknown cleanup"):
		parse_cleanup_stage_config(runtime_config=cfg)


def test_run_cleanup_stage_disabled_is_noop() -> None:
	"""Disabled-stage invocation returns an empty MultiTargetStageResult."""

	stage_config = parse_cleanup_stage_config(runtime_config=RuntimeConfig({}))
	result = run_cleanup_stage(stage_config)
	assert result.stage == "cleanup"
	assert result.total_targets == 0
	assert result.succeeded_targets == 0
	assert result.failed_targets == 0
	assert result.target_results == []


def test_run_cleanup_stage_enabled_without_phases_is_noop() -> None:
	"""enabled=True but empty phase_sequence still no-ops (defensive)."""

	stage_config = CleanupStageConfig(enabled=True, phase_sequence=())
	result = run_cleanup_stage(stage_config)
	assert result.stage == "cleanup"
	assert result.total_targets == 0
	assert result.target_results == []


def test_run_cleanup_stage_enabled_with_phases_but_no_targets_is_noop() -> None:
	"""Phase configured but caller supplied no targets: clean no-op."""

	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(enabled=True),
		),
	)
	result = run_cleanup_stage(stage_config, targets=[])
	assert result.stage == "cleanup"
	assert result.total_targets == 0


def test_run_cleanup_stage_unknown_phase_raises_not_implemented() -> None:
	"""A phase listed in the sequence with no registered runner fails loud."""

	stage_config = CleanupStageConfig(
		enabled=True,
		phase_sequence=("not_a_phase",),
	)
	# Need a target for the validation branch to run.
	from axon_recon.pipeline.execution.context import ExecutionTarget
	from pathlib import Path

	target = ExecutionTarget(
		dataset_index=0,
		dataset_id="dataset_000:test.h5",
		h5_path=Path("/tmp/test.h5"),
		stream_id="well000",
		mea_output_root=Path("/tmp/out"),
	)
	with pytest.raises(NotImplementedError):
		run_cleanup_stage(stage_config, targets=[target])


def test_default_cleanup_phase_sequence_is_wipe_src_scratch() -> None:
	"""Sanity check: slice 6's only phase is `wipe_src_scratch`."""

	assert DEFAULT_CLEANUP_PHASE_SEQUENCE == ("wipe_src_scratch",)
