from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .runner import WaveformExtractInputs


@dataclass(frozen=True)
class WaveformsDebugConfig:
    inputs: WaveformExtractInputs
    n_jobs: int
    chunk_duration: str
    cuda_visible_devices: Optional[str]


def build_waveforms_debug_config(*, args: Any, env: Any) -> WaveformsDebugConfig:
    h5_path = Path(args.h5_path) if args.h5_path is not None else env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    sorter = str(args.sorter) if args.sorter is not None else (env.env_str("AXON_RECON_SORTER", default="kilosort4") or "kilosort4")
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(env.env_int("AXON_RECON_N_JOBS", default=16) or 16)
    chunk_duration = str(args.chunk_duration) if args.chunk_duration is not None else (env.env_str("AXON_RECON_CHUNK_DURATION", default="2s") or "2s")

    cuda_visible_devices = str(args.cuda_visible_devices) if args.cuda_visible_devices is not None else env.env_str("AXON_RECON_CUDA_VISIBLE_DEVICES", default=None)

    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    elif bool(args.force):
        force_restart = True
    else:
        force_restart = env.env_bool("AXON_RECON_FORCE_RESTART", default=False)

    force_replot = env.env_bool("AXON_RECON_FORCE_REPLOT", default=False) if args.force_replot is None else bool(args.force_replot)

    ms_before = float(args.ms_before) if args.ms_before is not None else float(env.env_float("AXON_RECON_WF_MS_BEFORE", default=1.0) or 1.0)
    ms_after = float(args.ms_after) if args.ms_after is not None else float(env.env_float("AXON_RECON_WF_MS_AFTER", default=2.0) or 2.0)

    max_spikes_per_unit = (
        int(args.max_spikes_per_unit)
        if args.max_spikes_per_unit is not None
        else env.env_int("AXON_RECON_WF_MAX_SPIKES_PER_UNIT", default=None)
    )
    if max_spikes_per_unit is not None and int(max_spikes_per_unit) < 0:
        max_spikes_per_unit = None

    per_segment = env.env_bool("AXON_RECON_WF_PER_SEGMENT", default=True) if args.per_segment is None else bool(args.per_segment)
    filter_by_maxwell_epochs = env.env_bool("AXON_RECON_WF_FILTER_BY_MAXWELL_EPOCHS", default=True) if args.filter_by_maxwell_epochs is None else bool(args.filter_by_maxwell_epochs)
    filter_by_segment_bounds = env.env_bool("AXON_RECON_WF_FILTER_BY_SEGMENT_BOUNDS", default=True)
    segment_sort_safety_cleanup = env.env_bool("AXON_RECON_WF_SEGMENT_SORT_SAFETY_CLEANUP", default=True)
    recompute_channel_groups_for_reused_segments = env.env_bool(
        "AXON_RECON_WF_RECOMPUTE_CHANNEL_GROUPS_FOR_REUSED_SEGMENTS",
        default=False,
    )
    prefer_merged_sorting = env.env_bool("AXON_RECON_WF_PREFER_MERGED_SORTING", default=False)
    merged_sorting_dir_raw = env.env_str("AXON_RECON_WF_MERGED_SORTING_DIR", default=None)
    merged_sorting_dir = Path(merged_sorting_dir_raw) if merged_sorting_dir_raw else None
    waveforms_variant_name = env.env_str("AXON_RECON_WF_VARIANT_NAME", default=None)

    debug_max_units = int(args.debug_max_units) if args.debug_max_units is not None else env.env_int("AXON_RECON_WF_DEBUG_MAX_UNITS", default=None)
    debug_max_segments = int(args.debug_max_segments) if args.debug_max_segments is not None else env.env_int("AXON_RECON_WF_DEBUG_MAX_SEGMENTS", default=None)

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
        filter_by_segment_bounds=bool(filter_by_segment_bounds),
        segment_sort_safety_cleanup=bool(segment_sort_safety_cleanup),
        recompute_channel_groups_for_reused_segments=bool(recompute_channel_groups_for_reused_segments),
        merged_sorting_dir=merged_sorting_dir,
        prefer_merged_sorting=bool(prefer_merged_sorting),
        waveforms_variant_name=(str(waveforms_variant_name) if waveforms_variant_name is not None else None),
        force_restart=force_restart,
        force_replot=force_replot,
        debug_max_units=debug_max_units,
        debug_max_segments=debug_max_segments,
    )

    return WaveformsDebugConfig(
        inputs=inputs,
        n_jobs=int(n_jobs),
        chunk_duration=str(chunk_duration),
        cuda_visible_devices=(str(cuda_visible_devices) if cuda_visible_devices is not None else None),
    )


def apply_waveforms_runtime_hints(*, config: WaveformsDebugConfig, env: Any, debug_enabled: bool, logger: Any) -> None:
    if config.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)

    try:
        import spikeinterface.full as si  # type: ignore[import-not-found]

        if hasattr(si, "set_global_job_kwargs"):
            si.set_global_job_kwargs(
                n_jobs=int(config.n_jobs),
                chunk_duration=str(config.chunk_duration),
                progress_bar=bool(debug_enabled),
            )
            logger.info(
                "SpikeInterface global job kwargs: n_jobs=%d chunk_duration=%s",
                int(config.n_jobs),
                str(config.chunk_duration),
            )
    except Exception:
        logger.debug("Could not set SpikeInterface global job kwargs", exc_info=True)
