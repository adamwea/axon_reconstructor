#!/usr/bin/env python3
"""Project-local entrypoint for stepping through preprocessing in the debugger.

This file is intentionally *simple* and hardcoded for today's dataset so you can:
- hit F5 in VS Code
- step into axon_reconstructor preprocessing code
- get loud validation failures when something is off

Stage API lives in:
    axon_reconstructor.pipeline.pipeline_driver.AxonReconstructor.preprocess_for_spikesorting
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import debug_env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug preprocessing step")
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to env file (default: ./debug.env)",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable debug logging",
    )

    p.add_argument("--h5-path", type=Path, default=None)
    p.add_argument("--stream-id", type=str, default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--mea-output-root", type=Path, default=None)

    p.add_argument(
        "--break-before-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pause (breakpoint) right before running preprocessing",
    )
    p.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignore cached outputs and re-run preprocessing",
    )

    p.add_argument(
        "--temporal-resample-factor",
        type=int,
        default=None,
        help="Multiply sampling rate by this integer factor (e.g. 10)",
    )
    p.add_argument(
        "--temporal-resample-rate-hz",
        type=int,
        default=None,
        help="Explicit target sampling rate in Hz (overrides factor)",
    )
    p.add_argument(
        "--temporal-resample-margin-ms",
        type=float,
        default=None,
        help="Resampling margin (ms) used to reduce edge effects",
    )
    p.add_argument(
        "--temporal-resample-dtype",
        type=str,
        default=None,
        help="Optional dtype for resampled traces (e.g. float32)",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.env_file is not None:
        env_files = [Path(args.env_file)]
    else:
        env_files = debug_env.default_env_paths(script_path=__file__)
    debug_env.load_env_files_into_os(env_files=env_files, override_existing=False)

    debug_enabled = debug_env.env_bool("AXON_RECON_DEBUG", default=False) if args.debug is None else bool(args.debug)
    log_level = logging.DEBUG if debug_enabled else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("projects.debug_preprocessing_step")

    h5_path = Path(args.h5_path) if args.h5_path is not None else debug_env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else debug_env.env_required_str("AXON_RECON_STREAM_ID")
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(debug_env.env_int("AXON_RECON_N_JOBS", default=8) or 8)
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else debug_env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    break_before_run = debug_env.env_bool("AXON_RECON_BREAK_BEFORE_RUN", default=False) if args.break_before_run is None else bool(args.break_before_run)
    force_restart = debug_env.env_bool("AXON_RECON_FORCE_RESTART", default=False) if args.force_restart is None else bool(args.force_restart)

    temporal_resample_factor = (
        int(args.temporal_resample_factor)
        if args.temporal_resample_factor is not None
        else debug_env.env_int("AXON_RECON_TEMPORAL_RESAMPLE_FACTOR", default=None)
    )
    temporal_resample_rate_hz = (
        int(args.temporal_resample_rate_hz)
        if args.temporal_resample_rate_hz is not None
        else debug_env.env_int("AXON_RECON_TEMPORAL_RESAMPLE_RATE_HZ", default=None)
    )
    temporal_resample_margin_ms = (
        float(args.temporal_resample_margin_ms)
        if args.temporal_resample_margin_ms is not None
        else float(debug_env.env_float("AXON_RECON_TEMPORAL_RESAMPLE_MARGIN_MS", default=100.0) or 100.0)
    )
    temporal_resample_dtype = (
        str(args.temporal_resample_dtype)
        if args.temporal_resample_dtype is not None
        else debug_env.env_str("AXON_RECON_TEMPORAL_RESAMPLE_DTYPE", default=None)
    )

    logger.info(
        "Debug preprocessing inputs: h5=%s stream=%s n_jobs=%s output_root=%s",
        h5_path,
        stream_id,
        n_jobs,
        mea_output_root,
    )

    if break_before_run:
        breakpoint()  # noqa: T100

    from axon_reconstructor.pipeline.pipeline_driver import AxonReconstructor

    recon = AxonReconstructor(
        h5_parent_dirs=[h5_path],
        mea_analysis_output_root=str(mea_output_root),
        force_restart=bool(force_restart),
    )

    multirec, common_electrodes = recon.preprocess_for_spikesorting(
        h5_path=h5_path,
        stream_id=stream_id,
        n_jobs=int(n_jobs),
        plot_layouts=True,
        temporal_resample_factor=temporal_resample_factor,
        temporal_resample_rate_hz=temporal_resample_rate_hz,
        temporal_resample_margin_ms=temporal_resample_margin_ms,
        temporal_resample_dtype=temporal_resample_dtype,
        overwrite_saved_recording=bool(force_restart),
    )

    if multirec is None:
        raise RuntimeError("preprocess_for_spikesorting returned no recording")
    if len(common_electrodes) == 0:
        raise RuntimeError("preprocess_for_spikesorting produced zero common electrodes")

    logger.info("Done. Common electrodes: %d", len(common_electrodes))

    # Keep objects visible in debugger watch.
    _ = multirec, common_electrodes


if __name__ == "__main__":
    main()
