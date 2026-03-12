from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runner import SpikeSortingInputs


@dataclass(frozen=True)
class SpikesortingDebugConfig:
    inputs: SpikeSortingInputs
    break_before_run: bool


def build_spikesorting_debug_config(*, args: Any, env: Any) -> SpikesortingDebugConfig:
    verbose = (
        env.env_bool("AXON_RECON_SPIKESORT_VERBOSE", default=False)
        if getattr(args, "verbose", None) is None
        else bool(getattr(args, "verbose"))
    )

    mea_analysis_repo_root = (
        Path(args.mea_analysis_repo_root)
        if args.mea_analysis_repo_root is not None
        else env.env_required_path("AXON_RECON_MEA_ANALYSIS_REPO_ROOT")
    )
    h5_path = Path(args.h5_path) if args.h5_path is not None else env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    sorter = str(args.sorter) if args.sorter is not None else (env.env_str("AXON_RECON_SORTER", default="kilosort4") or "kilosort4")
    docker_image = str(args.docker_image) if args.docker_image is not None else env.env_required_str("AXON_RECON_DOCKER_IMAGE")

    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    elif bool(args.force):
        force_restart = True
    else:
        force_restart = env.env_bool("AXON_RECON_FORCE_RESTART", default=False)

    break_before_run = env.env_bool("AXON_RECON_BREAK_BEFORE_RUN", default=False) if args.break_before_run is None else bool(args.break_before_run)

    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(env.env_int("AXON_RECON_N_JOBS", default=16) or 16)
    chunk_duration = str(args.chunk_duration) if args.chunk_duration is not None else (env.env_str("AXON_RECON_CHUNK_DURATION", default="1s") or "1s")

    cuda_visible_devices = str(args.cuda_visible_devices) if args.cuda_visible_devices is not None else env.env_str("AXON_RECON_CUDA_VISIBLE_DEVICES", default=None)

    env_ks_batch_duration_s = env.env_float("AXON_RECON_KS_BATCH_DURATION_S", default=None)
    env_ks_batch_size = env.env_int("AXON_RECON_KS_BATCH_SIZE", default=None)
    ks_th_universal = env.env_float("AXON_RECON_KS_TH_UNIVERSAL", default=None)
    ks_th_learned = env.env_float("AXON_RECON_KS_TH_LEARNED", default=None)
    ks_th_single_ch = env.env_float("AXON_RECON_KS_TH_SINGLE_CH", default=None)
    ks_cluster_downsampling = env.env_int("AXON_RECON_KS_CLUSTER_DOWNSAMPLING", default=None)
    ks_nearest_chans = env.env_int("AXON_RECON_KS_NEAREST_CHANS", default=None)
    ks_max_channel_distance = env.env_float("AXON_RECON_KS_MAX_CHANNEL_DISTANCE", default=None)
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

    do_curation = (
        env.env_bool("AXON_RECON_SPIKESORT_CURATION", default=True)
        if args.curation is None
        else bool(args.curation)
    )

    force_rerun_analyzer = (
        env.env_bool("AXON_RECON_SPIKESORT_RERUN_ANALYZER", default=False)
        if args.rerun_analyzer is None
        else bool(args.rerun_analyzer)
    )
    force_merge_on_resume = env.env_bool("AXON_RECON_SPIKESORT_FORCE_MERGE_ON_RESUME", default=False)
    auto_merge_units = (
        env.env_bool("AXON_RECON_SPIKESORT_AUTO_MERGE_UNITS", default=False)
        if args.auto_merge_units is None
        else bool(args.auto_merge_units)
    )
    auto_merge_template_diff_thresh = (
        (env.env_str("AXON_RECON_SPIKESORT_AUTO_MERGE_TEMPLATE_DIFF_THRESH", default="0.05,0.15,0.25") or "0.05,0.15,0.25")
        if args.auto_merge_template_diff_thresh is None
        else str(args.auto_merge_template_diff_thresh)
    )

    inputs = SpikeSortingInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        mea_analysis_repo_root=mea_analysis_repo_root,
        sorter=sorter,
        docker_image=docker_image,
        force_restart=force_restart,
        verbose=bool(verbose),
        n_jobs=int(n_jobs),
        chunk_duration=str(chunk_duration),
        cuda_visible_devices=cuda_visible_devices,
        ks_batch_duration_s=ks_batch_duration_s,
        ks_batch_size=ks_batch_size,
        ks_th_universal=(float(ks_th_universal) if ks_th_universal is not None else None),
        ks_th_learned=(float(ks_th_learned) if ks_th_learned is not None else None),
        ks_th_single_ch=(float(ks_th_single_ch) if ks_th_single_ch is not None else None),
        ks_cluster_downsampling=(int(ks_cluster_downsampling) if ks_cluster_downsampling is not None else None),
        ks_nearest_chans=(int(ks_nearest_chans) if ks_nearest_chans is not None else None),
        ks_max_channel_distance=(float(ks_max_channel_distance) if ks_max_channel_distance is not None else None),
        run_analyzer=True,
        run_reports=True,
        no_curation=(not bool(do_curation)),
        export_to_phy=False,
        force_rerun_analyzer=bool(force_rerun_analyzer),
        force_merge_on_resume=bool(force_merge_on_resume),
        auto_merge_units=bool(auto_merge_units),
        auto_merge_template_diff_thresh=str(auto_merge_template_diff_thresh),
    )

    return SpikesortingDebugConfig(inputs=inputs, break_before_run=bool(break_before_run))
