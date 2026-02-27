#!/usr/bin/env python3
"""Run multiple debug steps in sequence.

This is intentionally thin: it shells out to the individual scripts so you can
still attach a debugger to any one step.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent

STEPS = [
    "debug_preprocessing_step.py",
    "debug_spikesorting_step.py",
    "debug_waveforms_step.py",
    "debug_templates_step.py",
    "debug_reconstruction_step.py",
    "debug_analysis_step.py",
    #"debug_best_channel_sources_log.py",
]

STOP_ON_FAILURE = True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all (or selected) debug steps")
    p.add_argument(
        "--env-file",
        type=Path,
        default=(SCRIPT_DIR / "debug.env"),
        help="Env file to pass to every step (default: ./debug.env)",
    )

    # Common dataset/pipeline overrides (passed to every step).
    p.add_argument("--h5-path", type=Path, default=None)
    p.add_argument("--stream-id", type=str, default=None)
    p.add_argument("--mea-output-root", type=Path, default=None)
    p.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force restart (passed to every step that supports it)",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable debug logging (passed to every step)",
    )

    p.add_argument(
        "--steps",
        nargs="*",
        default=None,
        help="Which step scripts to run (default: built-in STEPS list)",
    )
    p.add_argument(
        "--stop-on-failure",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop immediately if a step fails",
    )
    p.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Extra args passed to every step after '--'",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    steps = list(STEPS) if not args.steps else list(args.steps)
    stop_on_failure = bool(STOP_ON_FAILURE) if args.stop_on_failure is None else bool(args.stop_on_failure)
    extra_args = list(args.extra_args) if args.extra_args else []

    common_args: list[str] = ["--env-file", str(Path(args.env_file))]
    if args.debug is not None:
        common_args.append("--debug" if bool(args.debug) else "--no-debug")
    if args.h5_path is not None:
        common_args += ["--h5-path", str(Path(args.h5_path))]
    if args.stream_id is not None:
        common_args += ["--stream-id", str(args.stream_id)]
    if args.mea_output_root is not None:
        common_args += ["--mea-output-root", str(Path(args.mea_output_root))]
    if args.force_restart is not None:
        common_args.append("--force-restart" if bool(args.force_restart) else "--no-force-restart")

    for step in steps:
        step_path = SCRIPT_DIR / step
        if not step_path.exists():
            print(f"Skipping missing step: {step_path}")
            continue
        cmd = [sys.executable, str(step_path)] + common_args + extra_args
        print(f"Running step: {step_path}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Step failed (exit {e.returncode}): {step}")
            if stop_on_failure:
                raise


if __name__ == "__main__":
    main()