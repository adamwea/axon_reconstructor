from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runner import SpikeSortingInputs


@dataclass(frozen=True)
class SpikesortingDebugConfig:
    inputs: SpikeSortingInputs


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
    multiseg_mode = env.env_bool("AXON_RECON_SPIKESORT_MULTISEG_MODE", default=False)
    waveform_prefer_merged_sorting = env.env_bool("AXON_RECON_WF_PREFER_MERGED_SORTING", default=False)
    waveform_merged_sorting_dir = env.env_str("AXON_RECON_WF_MERGED_SORTING_DIR", default=None)
    enable_phase_output_overrides = env.env_bool("AXON_RECON_MEA_ENABLE_PHASE_OUTPUT_OVERRIDES", default=False)
    phase_output_paths_json = env.env_str("AXON_RECON_MEA_PHASE_OUTPUT_PATHS_JSON", default=None)
    phase_output_paths: dict[str, Any] = {}
    if bool(enable_phase_output_overrides) and phase_output_paths_json:
        try:
            parsed = json.loads(str(phase_output_paths_json))
            if isinstance(parsed, dict):
                phase_output_paths = parsed
        except Exception:
            phase_output_paths = {}

    resume_from = (
        str(args.resume_from)
        if getattr(args, "resume_from", None) is not None
        else env.env_str("AXON_RECON_SPIKESORT_RESUME_FROM", default=None)
    )
    unitmatch_merge_units = (
        env.env_bool("AXON_RECON_SPIKESORT_UNITMATCH_MERGE_UNITS", default=False)
        if getattr(args, "unitmatch_merge_units", None) is None
        else bool(getattr(args, "unitmatch_merge_units"))
    )
    unitmatch_dry_run = (
        env.env_bool("AXON_RECON_SPIKESORT_UNITMATCH_DRY_RUN", default=True)
        if getattr(args, "unitmatch_dry_run", None) is None
        else bool(getattr(args, "unitmatch_dry_run"))
    )
    unitmatch_scored_dry_run = (
        env.env_bool("AXON_RECON_SPIKESORT_UNITMATCH_SCORED_DRY_RUN", default=True)
        if getattr(args, "unitmatch_scored_dry_run", None) is None
        else bool(getattr(args, "unitmatch_scored_dry_run"))
    )
    unitmatch_output_subdir_name = (
        str(getattr(args, "unitmatch_output_subdir_name"))
        if getattr(args, "unitmatch_output_subdir_name", None) is not None
        else (env.env_str("AXON_RECON_SPIKESORT_UNITMATCH_OUTPUT_SUBDIR_NAME", default="unitmatch_outputs") or "unitmatch_outputs")
    )
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

    um_kwargs = {
        "merge_units": bool(unitmatch_merge_units),
        "dry_run": bool(unitmatch_dry_run),
        "scored_dry_run": bool(unitmatch_scored_dry_run),
        "output_subdir_name": str(unitmatch_output_subdir_name),
    }
    am_kwargs = {
        "enabled": bool(auto_merge_units),
        "template_diff_thresh": str(auto_merge_template_diff_thresh),
    }
    option_kwargs = {
        "force_rerun_analyzer": bool(force_rerun_analyzer),
        "cuda_visible_devices": cuda_visible_devices,
        "multiseg_mode": bool(multiseg_mode),
        "waveform_prefer_merged_sorting": bool(waveform_prefer_merged_sorting),
        "waveform_merged_sorting_dir": waveform_merged_sorting_dir,
        "phase_output_paths": phase_output_paths,
    }

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
        resume_from=(str(resume_from).strip() if resume_from else None),
        um_kwargs=um_kwargs,
        am_kwargs=am_kwargs,
        option_kwargs=option_kwargs,
    )

    return SpikesortingDebugConfig(inputs=inputs)
