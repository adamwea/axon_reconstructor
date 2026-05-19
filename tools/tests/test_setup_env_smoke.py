"""Light smoke test for `tools/setup_env.sh`.

Verifies the script is syntactically valid bash and parses --help
without actually doing a conda env create. Full smoke (env create from
scratch) is the user's domain — multi-minute, requires conda + network
access, out of automated test scope.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "setup_env.sh"


def test_script_exists_and_is_executable() -> None:
	assert SCRIPT.is_file(), f"missing: {SCRIPT}"
	assert os.access(SCRIPT, os.X_OK), f"not executable: {SCRIPT}"


def test_bash_syntax_check() -> None:
	result = subprocess.run(
		["bash", "-n", str(SCRIPT)],
		capture_output=True,
		text=True,
		timeout=10,
	)
	assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_help_prints_usage() -> None:
	result = subprocess.run(
		[str(SCRIPT), "--help"],
		capture_output=True,
		text=True,
		timeout=10,
	)
	assert result.returncode == 0, f"--help exit code {result.returncode}: {result.stderr}"
	combined = result.stdout + result.stderr
	assert "setup_env.sh" in combined
	assert "--editable-siblings" in combined
	assert "--from-local" in combined
	assert "conda env create" in combined


def test_unknown_arg_exits_nonzero() -> None:
	result = subprocess.run(
		[str(SCRIPT), "--bogus-flag"],
		capture_output=True,
		text=True,
		timeout=10,
	)
	assert result.returncode != 0
	assert "unknown arg" in (result.stdout + result.stderr)
