#!/usr/bin/env python3
"""Project-local entrypoint for stepping through waveform extraction in the debugger."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import debug_env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug waveforms extraction step")
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
    p.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignore cached outputs and re-run waveforms",
    )
    p.add_argument(
        "--force-replot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rewrite grids/panels while reusing extracted analyzers",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Alias for --force-restart (kept for backwards compatibility)",
    )

    p.add_argument("--h5-path", type=Path, default=None)
    p.add_argument("--stream-id", type=str, default=None)
    p.add_argument("--mea-output-root", type=Path, default=None)

    p.add_argument("--sorter", type=str, default=None)
    p.add_argument("--ms-before", type=float, default=None)
    p.add_argument("--ms-after", type=float, default=None)
    p.add_argument("--max-spikes-per-unit", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=None)

    # Resource controls
    p.add_argument(
        "--chunk-duration",
        type=str,
        default=None,
        help="SpikeInterface chunk duration (e.g. '1s', '2s', '500ms')",
    )
    p.add_argument("--omp-threads", type=int, default=None, help="Set OMP_NUM_THREADS")
    p.add_argument("--mkl-threads", type=int, default=None, help="Set MKL_NUM_THREADS")
    p.add_argument("--openblas-threads", type=int, default=None, help="Set OPENBLAS_NUM_THREADS")
    p.add_argument("--numexpr-threads", type=int, default=None, help="Set NUMEXPR_NUM_THREADS")
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1')",
    )

    p.add_argument(
        "--per-segment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable per-segment waveforms",
    )
    p.add_argument(
        "--filter-by-maxwell-epochs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop spikes whose waveform window crosses Maxwell snippet epochs",
    )

    p.add_argument(
        "--debug-max-units",
        type=int,
        default=None,
        help="Limit waveforms extraction to the first N units (speeds up debugging).",
    )
    p.add_argument(
        "--debug-max-segments",
        type=int,
        default=None,
        help="Limit per-segment waveforms to the first N stitch segments (speeds up debugging).",
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
    logger = logging.getLogger("projects.debug_waveforms_step")

    h5_path = Path(args.h5_path) if args.h5_path is not None else debug_env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else debug_env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else debug_env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    sorter = str(args.sorter) if args.sorter is not None else (debug_env.env_str("AXON_RECON_SORTER", default="kilosort4") or "kilosort4")
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(debug_env.env_int("AXON_RECON_N_JOBS", default=16) or 16)
    chunk_duration = str(args.chunk_duration) if args.chunk_duration is not None else (debug_env.env_str("AXON_RECON_CHUNK_DURATION", default="2s") or "2s")

    omp_threads = int(args.omp_threads) if args.omp_threads is not None else int(debug_env.env_int("AXON_RECON_OMP_THREADS", default=n_jobs) or n_jobs)
    mkl_threads = int(args.mkl_threads) if args.mkl_threads is not None else int(debug_env.env_int("AXON_RECON_MKL_THREADS", default=n_jobs) or n_jobs)
    openblas_threads = int(args.openblas_threads) if args.openblas_threads is not None else int(debug_env.env_int("AXON_RECON_OPENBLAS_THREADS", default=n_jobs) or n_jobs)
    numexpr_threads = int(args.numexpr_threads) if args.numexpr_threads is not None else int(debug_env.env_int("AXON_RECON_NUMEXPR_THREADS", default=n_jobs) or n_jobs)
    cuda_visible_devices = str(args.cuda_visible_devices) if args.cuda_visible_devices is not None else debug_env.env_str("AXON_RECON_CUDA_VISIBLE_DEVICES", default=None)

    # Apply env-level thread controls early.
    debug_env.apply_thread_env(omp=omp_threads, mkl=mkl_threads, openblas=openblas_threads, numexpr=numexpr_threads)
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    from axon_reconstructor.pipeline.waveforms import WaveformExtractInputs, extract_waveforms

    # Configure SpikeInterface global job kwargs (best-effort).
    try:
        import spikeinterface.full as si  # type: ignore[import-not-found]

        if hasattr(si, "set_global_job_kwargs"):
            si.set_global_job_kwargs(
                n_jobs=int(n_jobs),
                chunk_duration=str(chunk_duration),
                progress_bar=bool(debug_enabled),
            )
            logger.info(
                "SpikeInterface global job kwargs: n_jobs=%d chunk_duration=%s",
                int(n_jobs),
                str(chunk_duration),
            )
    except Exception:
        logger.debug("Could not set SpikeInterface global job kwargs", exc_info=True)

    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    elif bool(args.force):
        force_restart = True
    else:
        force_restart = debug_env.env_bool("AXON_RECON_FORCE_RESTART", default=False)

    force_replot = debug_env.env_bool("AXON_RECON_FORCE_REPLOT", default=False) if args.force_replot is None else bool(args.force_replot)

    ms_before = float(args.ms_before) if args.ms_before is not None else float(debug_env.env_float("AXON_RECON_WF_MS_BEFORE", default=1.0) or 1.0)
    ms_after = float(args.ms_after) if args.ms_after is not None else float(debug_env.env_float("AXON_RECON_WF_MS_AFTER", default=2.0) or 2.0)

    max_spikes_per_unit = (
        int(args.max_spikes_per_unit)
        if args.max_spikes_per_unit is not None
        else debug_env.env_int("AXON_RECON_WF_MAX_SPIKES_PER_UNIT", default=None)
    )

    per_segment = debug_env.env_bool("AXON_RECON_WF_PER_SEGMENT", default=True) if args.per_segment is None else bool(args.per_segment)
    filter_by_maxwell_epochs = debug_env.env_bool("AXON_RECON_WF_FILTER_BY_MAXWELL_EPOCHS", default=True) if args.filter_by_maxwell_epochs is None else bool(args.filter_by_maxwell_epochs)

    debug_max_units = int(args.debug_max_units) if args.debug_max_units is not None else debug_env.env_int("AXON_RECON_WF_DEBUG_MAX_UNITS", default=None)
    debug_max_segments = int(args.debug_max_segments) if args.debug_max_segments is not None else debug_env.env_int("AXON_RECON_WF_DEBUG_MAX_SEGMENTS", default=None)

    inputs = WaveformExtractInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        sorter=sorter,
        ms_before=ms_before,
        ms_after=ms_after,
        n_jobs=n_jobs,
        max_spikes_per_unit=(int(max_spikes_per_unit) if max_spikes_per_unit is not None else None),
        per_segment=per_segment,
        filter_by_maxwell_epochs=filter_by_maxwell_epochs,
        force_restart=force_restart,
        force_replot=force_replot,
        debug_max_units=debug_max_units,
        debug_max_segments=debug_max_segments,
    )

    logger.info("Debug waveforms inputs: %s", inputs)
    out = extract_waveforms(inputs=inputs)
    logger.info("Waveforms written under: %s", out.waveforms_out_dir)
    logger.info("Concat waveforms dir: %s", getattr(out, "concat_waveforms_dir", None))
    logger.info("Segment waveforms dir: %s", getattr(out, "segment_waveforms_dir", None))
    logger.info("Done waveforms debugging step.")


if __name__ == "__main__":
    main()
