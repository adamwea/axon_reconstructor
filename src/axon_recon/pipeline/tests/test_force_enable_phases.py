"""Tests for the --force-enable PHASE CLI override and the
`_apply_force_enable_phases` helper that consumes it.

Per USER INJECTION 2026-05-21: when a slice needs to smoke-test a phase
that ships with `enabled: false` in the runtime YAML, the loop sets
`--force-enable <phase>` and the override flips the phase's `enabled`
flag to True after YAML parsing but before phase-roster evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.config import (
	get_force_enable_phases_override,
	set_force_enable_phases_override,
)
from axon_recon.pipeline.runner import _apply_force_enable_phases


@pytest.fixture(autouse=True)
def _reset_force_enable_phases_override():
	set_force_enable_phases_override(None)
	yield
	set_force_enable_phases_override(None)


# ---------------------------------------------------------------------
# Process-wide override getter/setter
# ---------------------------------------------------------------------


def test_force_enable_phases_override_defaults_to_none() -> None:
	assert get_force_enable_phases_override() is None


def test_set_force_enable_phases_override_with_names() -> None:
	set_force_enable_phases_override(["kssynth", "plot_recons"])
	assert get_force_enable_phases_override() == frozenset({"kssynth", "plot_recons"})


def test_set_force_enable_phases_override_normalizes_whitespace_and_case() -> None:
	set_force_enable_phases_override(["  KSSynth  ", "Plot_Recons"])
	assert get_force_enable_phases_override() == frozenset({"kssynth", "plot_recons"})


def test_set_force_enable_phases_override_filters_empty_tokens() -> None:
	set_force_enable_phases_override(["", "   ", "kssynth"])
	assert get_force_enable_phases_override() == frozenset({"kssynth"})


def test_set_force_enable_phases_override_clears_with_none() -> None:
	set_force_enable_phases_override(["kssynth"])
	set_force_enable_phases_override(None)
	assert get_force_enable_phases_override() is None


def test_set_force_enable_phases_override_clears_with_empty_iterable() -> None:
	set_force_enable_phases_override(["kssynth"])
	set_force_enable_phases_override([])
	assert get_force_enable_phases_override() is None


# ---------------------------------------------------------------------
# _apply_force_enable_phases helper
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _FakePhaseCfg:
	enabled: bool = False
	resource_class: str | None = None


@dataclass(frozen=True)
class _FakePhasesCfg:
	kssynth: _FakePhaseCfg = field(default_factory=lambda: _FakePhaseCfg())
	build_templates: _FakePhaseCfg = field(default_factory=lambda: _FakePhaseCfg(enabled=True))
	plot_recons: _FakePhaseCfg = field(default_factory=lambda: _FakePhaseCfg())


@dataclass(frozen=True)
class _FakeStageCfg:
	phases: _FakePhasesCfg = field(default_factory=_FakePhasesCfg)


def test_apply_force_enable_phases_with_no_override_returns_input_unchanged() -> None:
	cfg = _FakeStageCfg()
	result = _apply_force_enable_phases(cfg, phase_names=None)
	assert result is cfg


def test_apply_force_enable_phases_with_empty_set_returns_input_unchanged() -> None:
	cfg = _FakeStageCfg()
	result = _apply_force_enable_phases(cfg, phase_names=frozenset())
	assert result is cfg


def test_apply_force_enable_phases_flips_matching_phase_to_true() -> None:
	cfg = _FakeStageCfg()
	assert cfg.phases.kssynth.enabled is False

	result = _apply_force_enable_phases(cfg, phase_names=frozenset({"kssynth"}))

	# Returned config is a NEW object (frozen dataclass invariant).
	assert result is not cfg
	assert result.phases.kssynth.enabled is True
	# Other phases left alone.
	assert result.phases.build_templates.enabled is True
	assert result.phases.plot_recons.enabled is False


def test_apply_force_enable_phases_flips_multiple_phases() -> None:
	cfg = _FakeStageCfg()
	result = _apply_force_enable_phases(
		cfg, phase_names=frozenset({"kssynth", "plot_recons"})
	)
	assert result.phases.kssynth.enabled is True
	assert result.phases.plot_recons.enabled is True
	# build_templates was already True; still True (idempotent).
	assert result.phases.build_templates.enabled is True


def test_apply_force_enable_phases_skips_unknown_phase_names() -> None:
	cfg = _FakeStageCfg()
	# nonexistent_phase is not a field on _FakePhasesCfg; helper silently
	# ignores it (matching the lookup behavior — unknown phases are no-ops).
	result = _apply_force_enable_phases(
		cfg, phase_names=frozenset({"kssynth", "nonexistent_phase"})
	)
	assert result.phases.kssynth.enabled is True
	# Original cfg untouched.
	assert cfg.phases.kssynth.enabled is False


def test_apply_force_enable_phases_no_change_when_phase_already_enabled() -> None:
	cfg = _FakeStageCfg()
	# build_templates is True by default; asking to force-enable it should
	# be a no-op (no new object returned with that phase as the only "update").
	result = _apply_force_enable_phases(
		cfg, phase_names=frozenset({"build_templates"})
	)
	# Because no actual change needed → may return original or new equivalent.
	# Either way the value stays the same.
	assert result.phases.build_templates.enabled is True


def test_apply_force_enable_phases_handles_stage_config_without_phases_attr() -> None:
	# When stage_config has no `phases` attribute, helper returns it unchanged.
	cfg = SimpleNamespace(force_restart=False)
	result = _apply_force_enable_phases(cfg, phase_names=frozenset({"kssynth"}))
	assert result is cfg


def test_apply_force_enable_phases_handles_none_phases_attr() -> None:
	cfg = SimpleNamespace(phases=None)
	result = _apply_force_enable_phases(cfg, phase_names=frozenset({"kssynth"}))
	assert result is cfg


# ---------------------------------------------------------------------
# Integration with the real ReconstructionInputs tree (loaded from
# `dev/debug_local/debug.runtime.yml`, which has kssynth.enabled: false).
# ---------------------------------------------------------------------


def test_apply_force_enable_phases_flips_real_kssynth_in_recon_inputs() -> None:
	from pathlib import Path

	from axon_recon.pipeline.stages.reconstruct.config import (
		load_reconstruction_inputs_from_runtime,
	)

	repo_root = next(
		parent for parent in Path(__file__).resolve().parents
		if (parent / "dev" / "debug_local" / "debug.runtime.yml").exists()
	)
	inputs = load_reconstruction_inputs_from_runtime(
		config_path=str(repo_root / "dev" / "debug_local" / "debug.runtime.yml")
	)
	# Sanity: kssynth ships disabled in the debug_local YAML (slice 3a).
	# kssynth lives on the OUTER recon stage phases (inputs.phases), not on
	# the templates substage phases (inputs.templates_inputs.phases) — per
	# the slice 2a placement in ReconstructionPhasesConfig.
	assert inputs.phases.kssynth.enabled is False

	# Apply the force-enable override. Returns a NEW ReconstructionInputs
	# (frozen dataclass invariant).
	updated_inputs = _apply_force_enable_phases(
		inputs, phase_names=frozenset({"kssynth"})
	)
	assert updated_inputs.phases.kssynth.enabled is True
	# Original inputs untouched.
	assert inputs.phases.kssynth.enabled is False
	# Other phases unchanged (axon_velocity_gtrs ships enabled=True).
	assert (
		updated_inputs.phases.axon_velocity_gtrs.enabled
		is inputs.phases.axon_velocity_gtrs.enabled
	)
