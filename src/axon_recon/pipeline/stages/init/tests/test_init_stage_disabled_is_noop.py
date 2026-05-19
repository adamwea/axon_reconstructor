from __future__ import annotations

import pytest

from axon_recon.runtime_config import RuntimeConfig

from ..config import InitStageConfig, parse_init_stage_config
from ..models.inputs import (
	DEFAULT_INIT_PHASE_SEQUENCE,
	InitCopySrcToScratchPhaseConfig,
	InitPhasesConfig,
)
from ..runner import run_init_stage


def test_parse_init_stage_config_defaults() -> None:
	"""With no `stages.init` block present, the parser returns all defaults.

	Defaults stay disabled with an empty phase_sequence so the runner's
	no-op branch is hit on legacy configs that predate slice 5.
	"""

	parsed = parse_init_stage_config(runtime_config=RuntimeConfig({}))
	assert isinstance(parsed, InitStageConfig)
	assert parsed.enabled is False
	assert parsed.phase_sequence == ()
	assert parsed.output_rel_root == "init_outputs"
	assert parsed.force_restart is False
	assert parsed.replot is False
	assert parsed.phases.copy_src_to_scratch.enabled is False
	assert parsed.phases.copy_src_to_scratch.summary_json_relpath == "context/copy_src_to_scratch_summary.json"


def test_parse_init_stage_config_disabled_yaml() -> None:
	"""Explicit `enabled: false` + empty `phase_sequence: []` round-trip cleanly."""

	cfg = RuntimeConfig(
		{
			"stages": {
				"init": {
					"enabled": False,
					"phase_sequence": [],
					"phases": {},
				}
			}
		}
	)
	parsed = parse_init_stage_config(runtime_config=cfg)
	assert parsed.enabled is False
	assert parsed.phase_sequence == ()


def test_parse_init_stage_config_honors_force_overrides() -> None:
	"""CLI overrides flow through into the dataclass, like the other stages."""

	parsed = parse_init_stage_config(
		runtime_config=RuntimeConfig({}),
		force_restart_override=True,
		replot_override=True,
	)
	assert parsed.force_restart is True
	assert parsed.replot is True


def test_parse_init_stage_config_reads_copy_phase_overrides() -> None:
	"""`stages.init.phases.copy_src_to_scratch` knobs flow into the dataclass."""

	cfg = RuntimeConfig(
		{
			"stages": {
				"init": {
					"enabled": True,
					"phase_sequence": ["copy_src_to_scratch"],
					"phases": {
						"copy_src_to_scratch": {
							"enabled": True,
							"requires_use_scratch_root": True,
							"summary_json_relpath": "context/custom_copy_summary.json",
							"resource_class": "h5_to_binary",
						},
					},
				}
			}
		}
	)
	parsed = parse_init_stage_config(runtime_config=cfg)
	assert parsed.enabled is True
	assert parsed.phase_sequence == ("copy_src_to_scratch",)
	assert parsed.phases.copy_src_to_scratch.enabled is True
	assert parsed.phases.copy_src_to_scratch.requires_use_scratch_root is True
	assert parsed.phases.copy_src_to_scratch.summary_json_relpath == "context/custom_copy_summary.json"
	assert parsed.phases.copy_src_to_scratch.resource_class == "h5_to_binary"


def test_parse_init_stage_config_accepts_dotted_phase_names() -> None:
	"""`init.copy_src_to_scratch` round-trips to the bare phase name."""

	cfg = RuntimeConfig(
		{
			"stages": {
				"init": {
					"enabled": True,
					"phase_sequence": ["init.copy_src_to_scratch"],
				}
			}
		}
	)
	parsed = parse_init_stage_config(runtime_config=cfg)
	assert parsed.phase_sequence == ("copy_src_to_scratch",)


def test_parse_init_stage_config_rejects_unknown_phase() -> None:
	"""A phase_sequence entry that doesn't match a known init phase fails loud."""

	cfg = RuntimeConfig(
		{
			"stages": {
				"init": {
					"enabled": True,
					"phase_sequence": ["not_a_real_phase"],
				}
			}
		}
	)
	with pytest.raises(ValueError, match="Unknown init"):
		parse_init_stage_config(runtime_config=cfg)


def test_run_init_stage_disabled_is_noop() -> None:
	"""Disabled-stage invocation returns an empty MultiTargetStageResult."""

	stage_config = parse_init_stage_config(runtime_config=RuntimeConfig({}))
	result = run_init_stage(stage_config)
	assert result.stage == "init"
	assert result.total_targets == 0
	assert result.succeeded_targets == 0
	assert result.failed_targets == 0
	assert result.target_results == []


def test_run_init_stage_enabled_without_phases_is_noop() -> None:
	"""enabled=True but empty phase_sequence still no-ops (defensive)."""

	stage_config = InitStageConfig(enabled=True, phase_sequence=())
	result = run_init_stage(stage_config)
	assert result.stage == "init"
	assert result.total_targets == 0
	assert result.target_results == []


def test_run_init_stage_enabled_with_phases_but_no_targets_is_noop() -> None:
	"""Phase configured but caller supplied no targets: clean no-op."""

	stage_config = InitStageConfig(
		enabled=True,
		phase_sequence=("copy_src_to_scratch",),
		phases=InitPhasesConfig(
			copy_src_to_scratch=InitCopySrcToScratchPhaseConfig(enabled=True),
		),
	)
	result = run_init_stage(stage_config, targets=[])
	assert result.stage == "init"
	assert result.total_targets == 0


def test_run_init_stage_unknown_phase_raises_not_implemented() -> None:
	"""A phase listed in the sequence with no registered runner fails loud."""

	stage_config = InitStageConfig(
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
		run_init_stage(stage_config, targets=[target])


def test_default_init_phase_sequence_is_copy_src_to_scratch() -> None:
	"""Sanity check: slice 5's only phase is `copy_src_to_scratch`."""

	assert DEFAULT_INIT_PHASE_SEQUENCE == ("copy_src_to_scratch",)
