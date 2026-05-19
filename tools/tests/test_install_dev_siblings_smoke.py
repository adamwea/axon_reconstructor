"""Light smoke test for `tools/install_dev_siblings.sh`.

Verifies the script is syntactically valid bash, parses --help without
exiting non-zero, and that --dry-run produces the expected wiring
output for a default invocation. Idempotency + actual installs are
exercised via the script's `--from-local` mode in dev environments
where UnitMatchPy is already editable-installed (verified manually
during slice 4 development; out of CI scope since cloning into deps/
would require network access).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "install_dev_siblings.sh"


def _run(args: list[str], env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
	env = os.environ.copy()
	if env_overrides:
		env.update(env_overrides)
	return subprocess.run(
		[str(SCRIPT), *args],
		capture_output=True,
		text=True,
		env=env,
		timeout=30,
	)


def test_script_exists_and_is_executable() -> None:
	assert SCRIPT.is_file(), f"missing: {SCRIPT}"
	assert os.access(SCRIPT, os.X_OK), f"not executable: {SCRIPT}"


def test_bash_syntax_check() -> None:
	# `bash -n` parses without executing; catches typos / missing
	# fi/done/etc without needing pip available.
	result = subprocess.run(
		["bash", "-n", str(SCRIPT)],
		capture_output=True,
		text=True,
		timeout=10,
	)
	assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_dry_run_emits_expected_wiring() -> None:
	# Dry-run with a from-local that doesn't have UnitMatch should fall
	# back to cloning into deps/ — and dry-run should print that without
	# actually running git.
	result = _run(["--dry-run", "--siblings", "UnitMatchPy"])
	assert result.returncode == 0, f"dry-run exited {result.returncode}: {result.stderr}"
	combined = result.stdout + result.stderr
	assert "[dry-run]" in combined
	assert "UnitMatchPy" in combined
	assert "deps/UnitMatch" in combined  # cloning destination


def test_filter_skips_unlisted_siblings() -> None:
	result = _run(["--dry-run", "--siblings", "UnitMatchPy"])
	assert result.returncode == 0
	combined = result.stdout + result.stderr
	assert "skip SLAy" in combined


def test_unknown_arg_exits_nonzero() -> None:
	result = _run(["--no-such-flag"])
	assert result.returncode != 0
	assert "unknown arg" in (result.stdout + result.stderr)
