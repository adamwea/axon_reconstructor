"""Tests for the --dry-run CLI flag + process-wide override.

Per dry_run_rollout_plan slice 1 + guardrails/dry_run.md: --dry-run is
a universal stages-subparser flag that flips a process-wide override.
Phases that have opted into the dry-run contract read it via
`get_dry_run_override()` at the top of their work function and exit
early with a stub `<phase>_summary.json` (status=dry_run_ok). Until a
phase opts in, --dry-run is a no-op for that phase.

This slice ships ONLY the CLI flag + override plumbing. Phase
short-circuits land in dry_run_rollout slices 3-7 (one per stage's
phase set).
"""

from __future__ import annotations

import pytest

from axon_recon.pipeline.config import (
	get_dry_run_override,
	set_dry_run_override,
)


@pytest.fixture(autouse=True)
def _reset_dry_run_override():
	set_dry_run_override(None)
	yield
	set_dry_run_override(None)


def test_dry_run_override_defaults_to_none() -> None:
	assert get_dry_run_override() is None


def test_set_dry_run_override_true() -> None:
	set_dry_run_override(True)
	assert get_dry_run_override() is True


def test_set_dry_run_override_false_is_distinct_from_none() -> None:
	# False is a deliberate "no, don't dry-run" signal, distinct from
	# "unset / honor YAML". The setter respects that — only None clears.
	set_dry_run_override(False)
	assert get_dry_run_override() is False


def test_set_dry_run_override_clears_with_none() -> None:
	set_dry_run_override(True)
	set_dry_run_override(None)
	assert get_dry_run_override() is None


def test_dry_run_cli_flag_parses_through_stages_subparser() -> None:
	from axon_recon.pipeline.cli import build_parser

	parser = build_parser()
	args = parser.parse_args(
		[
			"stages",
			"reconstruct.kssynth",
			"--config",
			"dummy.yml",
			"--dry-run",
		]
	)
	assert bool(getattr(args, "dry_run", False)) is True


def test_dry_run_cli_flag_defaults_to_false_when_absent() -> None:
	from axon_recon.pipeline.cli import build_parser

	parser = build_parser()
	args = parser.parse_args(
		[
			"stages",
			"reconstruct.kssynth",
			"--config",
			"dummy.yml",
		]
	)
	assert bool(getattr(args, "dry_run", False)) is False


def test_dry_run_and_force_enable_can_be_combined() -> None:
	"""--dry-run + --force-enable kssynth: natural combination for slice 3b
	smoke wiring — force-enable the phase so it's in the plan, then dry-run
	to verify input resolution without doing heavy compute."""

	from axon_recon.pipeline.cli import build_parser

	parser = build_parser()
	args = parser.parse_args(
		[
			"stages",
			"reconstruct.kssynth",
			"--config",
			"dummy.yml",
			"--dry-run",
			"--force-enable",
			"kssynth",
		]
	)
	assert bool(getattr(args, "dry_run", False)) is True
	assert getattr(args, "force_enable_phases", None) == "kssynth"
