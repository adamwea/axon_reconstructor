"""Spikesorting stage runner.

Contract:
- preprocessing has already saved a SpikeInterface recording at:
    <MEA_OUTPUT_ROOT>/<relative_pattern>/<well>/stg1_preprocess_outputs/preprocessed_recording
- this stage loads that recording and runs MEA_Analysis sorting/analyzer/reports.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..checkpointing import ProcessingStage as AxonProcessingStage, load_checkpoint
from ..pipeline_logging import log_stage_complete, log_stage_failure, log_stage_start
from ..checkpointing import compute_stage_checkpoint_file, save_stage_completed, save_stage_failed, save_stage_started
from ..stg1_preprocessing.constants import PREPROCESS_OUTPUTS_DIRNAME


SPIKESORTING_OUTPUTS_DIRNAME = "stg2_spikesorting_outputs"


def _recording_profile_from_h5(h5_path: Path) -> str:
    path_lower = str(Path(h5_path)).lower()
    if "/network/" in path_lower:
        return "network_single_segment_expected"
    if "/axontracking/" in path_lower:
        return "axontracking_multi_segment_possible"
    return "unknown"


@dataclass(frozen=True)
class SpikeSortingInputs:
    h5_path: Path
    stream_id: str

    mea_output_root: Path

    # Where MEA_Analysis repo lives (needed so imports work when not installed).
    mea_analysis_repo_root: Path

    # MEA_Analysis options
    sorter: str = "kilosort4"
    docker_image: Optional[str] = None
    recording_num: str = "rec0000"
    verbose: bool = False

    # Optional Kilosort tuning (applied via MEA_Analysis sorter_kwargs override).
    # - If both are provided, `ks_batch_size` wins.
    # - `ks_batch_duration_s` is converted using the loaded recording's sampling rate.
    ks_batch_duration_s: Optional[float] = None
    ks_batch_size: Optional[int] = None
    ks_th_universal: Optional[float] = None
    ks_th_learned: Optional[float] = None
    ks_th_single_ch: Optional[float] = None
    ks_cluster_downsampling: Optional[int] = None
    ks_nearest_chans: Optional[int] = None
    ks_max_channel_distance: Optional[float] = None

    # Resource controls (best-effort): these primarily affect analyzer/reporting
    # steps that run in the current Python process.
    n_jobs: Optional[int] = None
    chunk_duration: Optional[str] = None

    # GPU selection (helps multi-GPU nodes / avoiding contention)
    cuda_visible_devices: Optional[str] = None

    # Post-sorting steps (these are where most figures are generated)
    run_analyzer: bool = True
    run_reports: bool = True

    # Report options
    no_curation: bool = False
    export_to_phy: bool = False

    # Analyzer options (default-off)
    force_rerun_analyzer: bool = False
    # Testing override: force merge phase to run on resume by asking MEA_Analysis
    # to rerun analyzer pipeline steps.
    force_merge_on_resume: bool = False
    unitmatch_merge_units: bool = False
    unitmatch_dry_run: bool = True
    auto_merge_units: bool = False
    # CSV string like "0.05,0.15,0.25"; only used if auto_merge_units=True
    auto_merge_template_diff_thresh: str = "0.05,0.15,0.25"

    # If True, ignore existing MEA_Analysis checkpoints for this run.
    force_restart: bool = False


@dataclass(frozen=True)
class SpikeSortingOutputs:
    recording_dir: Path
    sorter_output_dir: Path
    output_dir: Path
    analyzer_dir: Path
    merged_sorter_output_dir: Optional[Path] = None


def _resolve_preprocess_dir(*, well_out_dir: Path) -> Path:
    """Find preprocessing output folder, with backward-compatible fallback."""

    new_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
    if new_dir.exists():
        return new_dir

    legacy_dir = well_out_dir / "axon_reconstructor" / "preprocess"
    if legacy_dir.exists():
        return legacy_dir

    # Default to new location for error messaging.
    return new_dir


def _ensure_mea_analysis_importable(mea_analysis_repo_root: Path) -> None:
    """Make `import MEA_Analysis...` work in ad-hoc debug sessions."""

    mea_analysis_repo_root = Path(mea_analysis_repo_root).expanduser().resolve()
    # To import `MEA_Analysis`, Python must have the *parent* directory on sys.path.
    # Example: /home/.../pkgs is on sys.path, then /home/.../pkgs/MEA_Analysis is importable.
    mea_analysis_parent = mea_analysis_repo_root.parent
    if str(mea_analysis_parent) not in sys.path:
        sys.path.insert(0, str(mea_analysis_parent))


def run_spikesorting_stage(*, inputs: SpikeSortingInputs, logger: logging.Logger) -> SpikeSortingOutputs:
    """Load saved preprocessed recording and run spikesorting stage."""

    _ensure_mea_analysis_importable(inputs.mea_analysis_repo_root)
    logger.debug("MEA_Analysis import path ready from repo root: %s", inputs.mea_analysis_repo_root)

    # Apply environment-level resource controls as early as possible.
    # These can affect subprocess behavior.
    if inputs.cuda_visible_devices is not None:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(inputs.cuda_visible_devices)
        except Exception:
            pass

    def _env_or_none(name: str) -> str | None:
        value = os.environ.get(name)
        if value is None:
            return None
        token = str(value).strip()
        return token if token else None

    logger.info(
        "Spikesort runtime snapshot: pid=%s cpu_count=%s stream_id=%s sorter=%s docker_image=%s",
        os.getpid(),
        os.cpu_count(),
        inputs.stream_id,
        inputs.sorter,
        inputs.docker_image,
    )

    logger.info(
        "Spikesort runtime config: verbose=%s n_jobs=%s chunk_duration=%s force_restart=%s",
        bool(inputs.verbose),
        inputs.n_jobs,
        inputs.chunk_duration,
        bool(inputs.force_restart),
    )
    logger.info(
        "Spikesort resources: cuda_visible_devices=%s",
        inputs.cuda_visible_devices,
    )
    logger.info(
        "Spikesort effective env: CUDA_VISIBLE_DEVICES=%s",
        _env_or_none("CUDA_VISIBLE_DEVICES"),
    )

    import spikeinterface.full as si  # type: ignore[import-not-found]

    # Configure SpikeInterface global job kwargs (affects many `compute()` calls).
    try:
        job_kwargs = {}
        if inputs.n_jobs is not None:
            job_kwargs["n_jobs"] = int(inputs.n_jobs)
        if inputs.chunk_duration is not None:
            job_kwargs["chunk_duration"] = str(inputs.chunk_duration)
        job_kwargs["progress_bar"] = bool(inputs.verbose)

        if job_kwargs and hasattr(si, "set_global_job_kwargs"):
            si.set_global_job_kwargs(**job_kwargs)
            logger.info("SpikeInterface global job kwargs: %s", job_kwargs)
    except Exception:
        logger.debug("Could not set SpikeInterface global job kwargs", exc_info=True)

    # Reuse axon_reconstructor's MEA_Analysis-style output path computation.
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    # Ensure spikesorting status is captured in the same per-well pipeline log as preprocessing.
    try:
        from axon_reconstructor.pipeline.pipeline_logging import build_stage_logger

        logger = build_stage_logger(
            well_out_dir=well_out_dir,
            data_file=inputs.h5_path,
            stream_id=inputs.stream_id,
            stage_name="spikesort",
            logger_name_prefix="axon_reconstructor",
            verbose=inputs.verbose,
        )
    except Exception:
        pass

    axon_ckpt_file = compute_stage_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=inputs.h5_path,
        stream_id=inputs.stream_id,
        stage_name="spikesort",
    )
    axon_ckpt = load_checkpoint(
        checkpoint_file=axon_ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    logger.info(
        "Wrapper checkpoint loaded: file=%s previous_stage=%s force_restart=%s",
        axon_ckpt_file,
        axon_ckpt.stage,
        bool(inputs.force_restart),
    )
    axon_ckpt = save_stage_started(
        checkpoint_file=axon_ckpt_file,
        state=axon_ckpt,
        stage=AxonProcessingStage.SORTING,
        extra_fields={
            "spikesorting_out_dir": str(well_out_dir / SPIKESORTING_OUTPUTS_DIRNAME),
            "recording_profile": str(_recording_profile_from_h5(inputs.h5_path)),
            "checkpoint_owner": "axon_reconstructor_wrapper",
            "delegate_checkpoint_owner": "MEA_Analysis",
        },
    )
    log_stage_start(
        logger=logger,
        stage="spikesort",
        checkpoint_file=axon_ckpt_file,
        note="outer-wrapper; MEA_Analysis checkpoint remains authoritative inside stage",
    )

    recording_profile = _recording_profile_from_h5(inputs.h5_path)
    if recording_profile == "network_single_segment_expected":
        logger.info(
            "Network-scan profile detected: spikesorting uses preprocessed recording as-is "
            "(single-segment handling is applied in preprocessing)."
        )

    preprocess_dir = _resolve_preprocess_dir(well_out_dir=well_out_dir)
    recording_dir = preprocess_dir / "preprocessed_recording"
    logger.info("Resolved preprocessing outputs dir: %s", preprocess_dir)
    if not recording_dir.exists():
        raise FileNotFoundError(
            "Saved preprocessed recording folder not found. "
            "Run preprocessing first with `mea_output_root` set. "
            f"Expected: {recording_dir}"
        )

    logger.info("Loading saved preprocessed recording from %s", recording_dir)
    try:
        recording = si.load(recording_dir)
    except Exception:
        logger.warning("si.load(recording_dir) failed; retrying with si.load_extractor")
        recording = si.load_extractor(recording_dir)

    try:
        logger.info(
            "Loaded recording: segments=%d channels=%d fs_hz=%.3f",
            int(recording.get_num_segments()),
            int(len(recording.get_channel_ids())),
            float(recording.get_sampling_frequency()),
        )
    except Exception:
        logger.debug("Could not summarize loaded recording", exc_info=True)

    # Build optional sorter kwargs override (memory + sensitivity tuning).
    sorter_kwargs: dict = {}
    try:
        if inputs.ks_batch_size is not None:
            sorter_kwargs["batch_size"] = int(inputs.ks_batch_size)
        elif inputs.ks_batch_duration_s is not None:
            fs = float(recording.get_sampling_frequency())
            sorter_kwargs["batch_size"] = int(round(fs * float(inputs.ks_batch_duration_s)))

        if inputs.ks_th_universal is not None:
            sorter_kwargs["Th_universal"] = float(inputs.ks_th_universal)
        if inputs.ks_th_learned is not None:
            sorter_kwargs["Th_learned"] = float(inputs.ks_th_learned)
        if inputs.ks_th_single_ch is not None:
            sorter_kwargs["Th_single_ch"] = float(inputs.ks_th_single_ch)

        if inputs.ks_cluster_downsampling is not None:
            sorter_kwargs["cluster_downsampling"] = int(inputs.ks_cluster_downsampling)
        if inputs.ks_nearest_chans is not None:
            sorter_kwargs["nearest_chans"] = int(inputs.ks_nearest_chans)
        if inputs.ks_max_channel_distance is not None:
            sorter_kwargs["max_channel_distance"] = float(inputs.ks_max_channel_distance)
    except Exception:
        sorter_kwargs = {}

    from MEA_Analysis.IPNAnalysis.mea_analysis_routine import (  # type: ignore[import-not-found]
        MEARunOptions,
        run_mea_pipeline,
        ProcessingStage as MEAProcessingStage,
    )

    logger.info(
        "Merge config: unitmatch_merge_units=%s unitmatch_dry_run=%s auto_merge_units=%s",
        bool(inputs.unitmatch_merge_units),
        bool(inputs.unitmatch_dry_run),
        bool(inputs.auto_merge_units),
    )

    logger.info("Initializing MEA_Analysis pipeline options (sorting/analyzer/reports)")
    run_options = MEARunOptions(
        file_path=str(inputs.h5_path),
        stream_id=inputs.stream_id,
        recording_num=inputs.recording_num,
        output_root=str(inputs.mea_output_root),
        output_subdir_after_well=SPIKESORTING_OUTPUTS_DIRNAME,
        checkpoint_root=None,
        sorter=inputs.sorter,
        docker_image=inputs.docker_image,
        verbose=inputs.verbose,
        cleanup=False,
        force_restart=inputs.force_restart,
        sorter_kwargs=(sorter_kwargs if sorter_kwargs else None),
        unitmatch_merge_units=bool(inputs.unitmatch_merge_units),
        unitmatch_dry_run=bool(inputs.unitmatch_dry_run),
        auto_merge_units=bool(inputs.auto_merge_units),
        force_rerun_analyzer=bool(inputs.force_rerun_analyzer or inputs.force_merge_on_resume),
        preprocessed_recording=recording,
        skip_preprocessing=True,
        n_jobs=inputs.n_jobs,
        chunk_duration=inputs.chunk_duration,
        cuda_visible_devices=inputs.cuda_visible_devices,
        skip_spikesorting=False,
        run_analyzer=bool(inputs.run_analyzer),
        run_reports=bool(inputs.run_reports),
        thresholds=None,
        no_curation=bool(inputs.no_curation),
        export_to_phy=bool(inputs.export_to_phy),
        plot_mode="separate",
        plot_debug=False,
        raster_sort=None,
        fixed_y=False,
        auto_merge_template_diff_thresh=str(inputs.auto_merge_template_diff_thresh),
    )

    if sorter_kwargs:
        logger.info("MEA_Analysis sorter kwargs override: %s", sorter_kwargs)
    else:
        logger.info("MEA_Analysis sorter kwargs override: using defaults")

    logger.info(
        "Starting MEA_Analysis run: sorter=%s docker_image=%s force_restart=%s",
        inputs.sorter,
        inputs.docker_image,
        inputs.force_restart,
    )

    try:
        run_result = run_mea_pipeline(run_options)
        pipeline = run_result.pipeline

        sorter_output_dir = pipeline.output_dir / "sorter_output"
        merged_sorter_output_dir: Optional[Path] = None

        logger.info("Sorting output folder: %s", sorter_output_dir)

        if int(pipeline.state.get("stage", 0)) >= MEAProcessingStage.REPORTS_COMPLETE.value:
            logger.info("MEA_Analysis completed successfully (REPORTS_COMPLETE)")
        elif int(pipeline.state.get("stage", 0)) >= MEAProcessingStage.SORTING_COMPLETE.value:
            logger.info("MEA_Analysis completed sorting successfully (SORTING_COMPLETE)")

        analyzer_dir = pipeline.output_dir / "analyzer_output"
        if inputs.run_reports:
            logger.info("Reports/figures should be under: %s", pipeline.output_dir)

        mea_stage_value = int(pipeline.state.get("stage", 0) or 0)
        if mea_stage_value >= int(AxonProcessingStage.REPORTS_COMPLETE.value):
            completed_stage = AxonProcessingStage.REPORTS_COMPLETE
        elif mea_stage_value >= int(AxonProcessingStage.ANALYZER_COMPLETE.value):
            completed_stage = AxonProcessingStage.ANALYZER_COMPLETE
        elif mea_stage_value >= int(AxonProcessingStage.SORTING_COMPLETE.value):
            completed_stage = AxonProcessingStage.SORTING_COMPLETE
        else:
            completed_stage = AxonProcessingStage.SORTING

        axon_ckpt = save_stage_completed(
            checkpoint_file=axon_ckpt_file,
            state=axon_ckpt,
            stage=completed_stage,
            extra_fields={
                "spikesorting_out_dir": str(pipeline.output_dir),
                "sorter_output_dir": str(sorter_output_dir),
                "analyzer_dir": str(analyzer_dir),
                "recording_profile": str(recording_profile),
                "mea_analysis_checkpoint_file": str(getattr(pipeline, "checkpoint_file", "")),
                "checkpoint_owner": "axon_reconstructor_wrapper",
                "delegate_checkpoint_owner": "MEA_Analysis",
            },
        )
        log_stage_complete(
            logger=logger,
            stage="spikesort",
            checkpoint_file=axon_ckpt_file,
            completed_stage=str(completed_stage.name),
            mea_stage=str(pipeline.state.get("stage")),
        )

        return SpikeSortingOutputs(
            recording_dir=recording_dir,
            sorter_output_dir=sorter_output_dir,
            output_dir=pipeline.output_dir,
            analyzer_dir=analyzer_dir,
            merged_sorter_output_dir=merged_sorter_output_dir,
        )
    except Exception as e:
        logger.exception("Spikesorting stage raised an exception before completion")
        save_stage_failed(
            checkpoint_file=axon_ckpt_file,
            state=axon_ckpt,
            stage=AxonProcessingStage.SORTING,
            failed_stage="SPIKESORT",
            error=e,
            extra_fields={
                "spikesorting_out_dir": str(well_out_dir / SPIKESORTING_OUTPUTS_DIRNAME),
                "recording_profile": str(_recording_profile_from_h5(inputs.h5_path)),
                "checkpoint_owner": "axon_reconstructor_wrapper",
                "delegate_checkpoint_owner": "MEA_Analysis",
            },
        )
        log_stage_failure(
            logger=logger,
            stage="spikesort",
            checkpoint_file=axon_ckpt_file,
            error=e,
        )
        raise


def run_spikesorting_only(*, inputs: SpikeSortingInputs, logger: logging.Logger) -> SpikeSortingOutputs:
    """Backward-compatible alias for older callers."""

    return run_spikesorting_stage(inputs=inputs, logger=logger)
