#!/usr/bin/env python3
"""Project-local entrypoint for stepping through spike sorting in the debugger.

This mirrors `debug_preprocessing_step.py`, but isolates *only* the sorting step.

Prereq:
- Run preprocessing first (or use `debug_preprocessing_step.py`) with `MEA_OUTPUT_ROOT`
  set so the preprocessed recording is saved.

Stage API lives in:
    axon_reconstructor.pipeline.spikesorting
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import debug_env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug spikesorting step")
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

    p.add_argument("--mea-analysis-repo-root", type=Path, default=None)
    p.add_argument("--h5-path", type=Path, default=None)
    p.add_argument("--stream-id", type=str, default=None)
    p.add_argument("--mea-output-root", type=Path, default=None)

    p.add_argument("--sorter", type=str, default=None)
    p.add_argument("--docker-image", type=str, default=None)

    p.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignore cached outputs and re-run spikesorting",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Alias for --force-restart (kept for backwards compatibility)",
    )
    p.add_argument(
        "--break-before-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pause (breakpoint) right before running spikesorting",
    )

    # --- Resource / parallelism controls ---
    p.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="SpikeInterface/analysis CPU parallelism (used for analyzer/reports; best-effort for sorting)",
    )
    p.add_argument(
        "--chunk-duration",
        type=str,
        default=None,
        help="SpikeInterface chunk duration (e.g. '1s', '2s', '500ms') for compute-heavy steps",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Set torch.set_num_threads (affects CPU work in analyzer/reports)",
    )
    p.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Set torch.set_num_interop_threads (affects CPU work in analyzer/reports)",
    )
    p.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="Set OMP_NUM_THREADS",
    )
    p.add_argument(
        "--mkl-threads",
        type=int,
        default=None,
        help="Set MKL_NUM_THREADS",
    )
    p.add_argument(
        "--openblas-threads",
        type=int,
        default=None,
        help="Set OPENBLAS_NUM_THREADS",
    )
    p.add_argument(
        "--numexpr-threads",
        type=int,
        default=None,
        help="Set NUMEXPR_NUM_THREADS",
    )
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1')",
    )

    # --- Kilosort4 tuning (applies inside MEA_Analysis run_sorting) ---
    p.add_argument(
        "--ks-batch-duration-s",
        type=float,
        default=None,
        help="Override Kilosort4 batch duration in seconds (converted to batch_size = fs * seconds).",
    )
    p.add_argument(
        "--ks-batch-size",
        type=int,
        default=None,
        help="Override Kilosort4 batch_size directly (in samples). Overrides --ks-batch-duration-s.",
    )

    p.add_argument(
        "--curation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable spikesorting curation (default: enabled). If disabled, downstream stages can run on uncurated units.",
    )

    # --- Analyzer rerun / post-sorting merge (default-off) ---
    p.add_argument(
        "--rerun-analyzer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force recomputing analyzer_output even if checkpoints say complete (does not force re-sorting).",
    )
    p.add_argument(
        "--auto-merge-units",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional (default off): run SpikeInterface auto_merge_units during analyzer stage.",
    )
    p.add_argument(
        "--auto-merge-template-diff-thresh",
        type=str,
        default=None,
        help="Comma-separated template_diff_thresh values (e.g. '0.05,0.15,0.25') for auto-merge sweep.",
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
    logger = logging.getLogger("projects.debug_spikesorting_step")

    mea_analysis_repo_root = (
        Path(args.mea_analysis_repo_root)
        if args.mea_analysis_repo_root is not None
        else debug_env.env_required_path("AXON_RECON_MEA_ANALYSIS_REPO_ROOT")
    )
    h5_path = Path(args.h5_path) if args.h5_path is not None else debug_env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else debug_env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else debug_env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    sorter = str(args.sorter) if args.sorter is not None else (debug_env.env_str("AXON_RECON_SORTER", default="kilosort4") or "kilosort4")
    docker_image = str(args.docker_image) if args.docker_image is not None else debug_env.env_required_str("AXON_RECON_DOCKER_IMAGE")

    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    elif bool(args.force):
        force_restart = True
    else:
        force_restart = debug_env.env_bool("AXON_RECON_FORCE_RESTART", default=False)

    break_before_run = debug_env.env_bool("AXON_RECON_BREAK_BEFORE_RUN", default=False) if args.break_before_run is None else bool(args.break_before_run)

    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(debug_env.env_int("AXON_RECON_N_JOBS", default=16) or 16)
    chunk_duration = str(args.chunk_duration) if args.chunk_duration is not None else (debug_env.env_str("AXON_RECON_CHUNK_DURATION", default="1s") or "1s")

    # Thread env vars (default to n_jobs unless explicitly overridden).
    omp_threads = int(args.omp_threads) if args.omp_threads is not None else int(debug_env.env_int("AXON_RECON_OMP_THREADS", default=n_jobs) or n_jobs)
    mkl_threads = int(args.mkl_threads) if args.mkl_threads is not None else int(debug_env.env_int("AXON_RECON_MKL_THREADS", default=n_jobs) or n_jobs)
    openblas_threads = int(args.openblas_threads) if args.openblas_threads is not None else int(debug_env.env_int("AXON_RECON_OPENBLAS_THREADS", default=n_jobs) or n_jobs)
    numexpr_threads = int(args.numexpr_threads) if args.numexpr_threads is not None else int(debug_env.env_int("AXON_RECON_NUMEXPR_THREADS", default=n_jobs) or n_jobs)

    torch_threads = int(args.torch_threads) if args.torch_threads is not None else int(debug_env.env_int("AXON_RECON_TORCH_THREADS", default=n_jobs) or n_jobs)
    torch_interop_threads = (
        int(args.torch_interop_threads)
        if args.torch_interop_threads is not None
        else int(debug_env.env_int("AXON_RECON_TORCH_INTEROP_THREADS", default=min(8, max(1, torch_threads // 2))) or min(8, max(1, torch_threads // 2)))
    )

    cuda_visible_devices = str(args.cuda_visible_devices) if args.cuda_visible_devices is not None else debug_env.env_str("AXON_RECON_CUDA_VISIBLE_DEVICES", default=None)

    # Kilosort4 tuning
    env_ks_batch_duration_s = debug_env.env_float("AXON_RECON_KS_BATCH_DURATION_S", default=None)
    env_ks_batch_size = debug_env.env_int("AXON_RECON_KS_BATCH_SIZE", default=None)
    if args.ks_batch_size is not None:
        ks_batch_size = int(args.ks_batch_size)
        ks_batch_duration_s = None
    elif args.ks_batch_duration_s is not None:
        ks_batch_size = None
        ks_batch_duration_s = float(args.ks_batch_duration_s)
    elif env_ks_batch_size is not None:
        ks_batch_size = int(env_ks_batch_size)
        ks_batch_duration_s = None
    else:
        ks_batch_size = None
        ks_batch_duration_s = float(env_ks_batch_duration_s) if env_ks_batch_duration_s is not None else None

    debug_env.apply_thread_env(omp=omp_threads, mkl=mkl_threads, openblas=openblas_threads, numexpr=numexpr_threads)
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    do_curation = (
        debug_env.env_bool("AXON_RECON_SPIKESORT_CURATION", default=True)
        if args.curation is None
        else bool(args.curation)
    )

    force_rerun_analyzer = (
        debug_env.env_bool("AXON_RECON_SPIKESORT_RERUN_ANALYZER", default=False)
        if args.rerun_analyzer is None
        else bool(args.rerun_analyzer)
    )
    auto_merge_units = (
        debug_env.env_bool("AXON_RECON_SPIKESORT_AUTO_MERGE_UNITS", default=False)
        if args.auto_merge_units is None
        else bool(args.auto_merge_units)
    )
    auto_merge_template_diff_thresh = (
        (debug_env.env_str("AXON_RECON_SPIKESORT_AUTO_MERGE_TEMPLATE_DIFF_THRESH", default="0.05,0.15,0.25") or "0.05,0.15,0.25")
        if args.auto_merge_template_diff_thresh is None
        else str(args.auto_merge_template_diff_thresh)
    )

    from axon_reconstructor.pipeline.spikesorting import SpikeSortingInputs, run_spikesorting_stage

    inputs = SpikeSortingInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        mea_analysis_repo_root=mea_analysis_repo_root,
        sorter=sorter,
        docker_image=docker_image,
        force_restart=force_restart,
        verbose=True,
        n_jobs=int(n_jobs),
        chunk_duration=str(chunk_duration),
        torch_threads=int(torch_threads),
        torch_interop_threads=int(torch_interop_threads),
        omp_threads=int(omp_threads),
        mkl_threads=int(mkl_threads),
        openblas_threads=int(openblas_threads),
        numexpr_threads=int(numexpr_threads),
        cuda_visible_devices=cuda_visible_devices,
        ks_batch_duration_s=ks_batch_duration_s,
        ks_batch_size=ks_batch_size,
        run_analyzer=True,
        run_reports=True,
        no_curation=(not bool(do_curation)),
        export_to_phy=False,
        force_rerun_analyzer=bool(force_rerun_analyzer),
        auto_merge_units=bool(auto_merge_units),
        auto_merge_template_diff_thresh=str(auto_merge_template_diff_thresh),
    )

    logger.info("Debug spikesorting inputs: %s", inputs)

    if break_before_run:
        breakpoint()  # noqa: T100

    outputs = run_spikesorting_stage(inputs=inputs, logger=logger)
    logger.info("Done. Sorter output dir: %s", outputs.sorter_output_dir)
    logger.info("Output dir (reports/figures live here): %s", outputs.output_dir)

    # Keep objects visible in debugger watch.
    _ = outputs


if __name__ == "__main__":
    main()
