from __future__ import annotations

import os
import subprocess
from pathlib import Path


class SmokeRunError(RuntimeError):
    pass


def run_bash_script(*, script: Path, cwd: Path, env_overrides: dict[str, str]) -> None:
    if not script.exists():
        raise SmokeRunError(f"Smoke script not found: {script}")

    env = os.environ.copy()
    env.update(env_overrides)

    cmd = ["bash", str(script)]
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SmokeRunError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
