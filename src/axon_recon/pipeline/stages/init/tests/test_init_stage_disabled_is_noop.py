from __future__ import annotations

import pytest

from axon_recon.runtime_config import RuntimeConfig

from ..config import InitStageConfig, parse_init_stage_config
from ..runner import run_init_stage


def test_parse_init_stage_config_defaults() -> None:
	"""With no `stages.init` block present, the parser returns all defaults.

	This is the live shape today: the YAML files don't carry an init block at
	all yet (or they will once slice 5 wires copy_src_to_scratch in). Either
	way the stage must remain disabled with an empty phase_sequence so the
	runner's no-op branch is hit.
	"""

	parsed = parse_init_stage_config(runtime_config=RuntimeConfig({}))
	assert isinstance(parsed, InitStageConfig)
	assert parsed.enabled is False
	assert parsed.phase_sequence == ()
	assert parsed.output_rel_root == "init_outputs"
	assert parsed.force_restart is False
	assert parsed.force_replot is False


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
		force_replot_override=True,
	)
	assert parsed.force_restart is True
	assert parsed.force_replot is True


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


def test_run_init_stage_enabled_with_phases_raises_not_implemented() -> None:
	"""Slice 5 will replace this branch; the error is the half-wired guard."""

	stage_config = InitStageConfig(enabled=True, phase_sequence=("copy_src_to_scratch",))
	with pytest.raises(NotImplementedError):
		run_init_stage(stage_config)
