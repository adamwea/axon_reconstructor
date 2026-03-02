"""Spikesorting stage runner.

Contract:
- preprocessing has already saved a SpikeInterface recording at:
    <MEA_OUTPUT_ROOT>/<relative_pattern>/<well>/preprocess_outputs/preprocessed_recording
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
from ..stage_checkpointing import compute_stage_checkpoint_file, save_stage_completed, save_stage_failed, save_stage_started


PREPROCESS_OUTPUTS_DIRNAME = "preprocess_outputs"
SPIKESORTING_OUTPUTS_DIRNAME = "spikesorting_outputs"


def _compute_spikesort_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    return compute_stage_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=h5_path,
        stream_id=stream_id,
        stage_name="spikesort",
    )


def _resolve_axon_spikesort_completed_stage(stage_value: int) -> AxonProcessingStage:
    if int(stage_value) >= int(AxonProcessingStage.REPORTS_COMPLETE.value):
        return AxonProcessingStage.REPORTS_COMPLETE
    if int(stage_value) >= int(AxonProcessingStage.ANALYZER_COMPLETE.value):
        return AxonProcessingStage.ANALYZER_COMPLETE
    if int(stage_value) >= int(AxonProcessingStage.SORTING_COMPLETE.value):
        return AxonProcessingStage.SORTING_COMPLETE
    if int(stage_value) >= int(AxonProcessingStage.SORTING.value):
        return AxonProcessingStage.SORTING
    return AxonProcessingStage.SORTING


def _setup_well_logger(*, well_out_dir: Path, h5_path: Path, stream_id: str, verbose: bool) -> logging.Logger:
    from axon_reconstructor.pipeline.pipeline_logging import build_stage_logger

    return build_stage_logger(
        well_out_dir=well_out_dir,
        data_file=h5_path,
        stream_id=stream_id,
        logger_name=f"axon_reconstructor.{stream_id}",
        verbose=bool(verbose),
    )


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
    verbose: bool = True

    # Optional Kilosort tuning (applied via MEA_Analysis sorter_kwargs override).
    # - If both are provided, `ks_batch_size` wins.
    # - `ks_batch_duration_s` is converted using the loaded recording's sampling rate.
    ks_batch_duration_s: Optional[float] = None
    ks_batch_size: Optional[int] = None

    # Resource controls (best-effort): these primarily affect analyzer/reporting
    # steps that run in the current Python process.
    n_jobs: Optional[int] = None
    chunk_duration: Optional[str] = None

    # CPU thread controls (applied via env vars)
    omp_threads: Optional[int] = None
    mkl_threads: Optional[int] = None
    openblas_threads: Optional[int] = None
    numexpr_threads: Optional[int] = None

    # Torch CPU threading (analyzer/reports)
    torch_threads: Optional[int] = None
    torch_interop_threads: Optional[int] = None

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


def _relocate_mea_analysis_outputs(*, pipeline, well_out_dir: Path, logger: logging.Logger) -> None:
    """Force MEA_Analysis to write all outputs under <well>/spikesorting_outputs/.

    We keep MEA_Analysis itself unmodified by overriding the pipeline object's
    output_dir + checkpoint_file after construction.
    """

    spikesorting_dir = well_out_dir / SPIKESORTING_OUTPUTS_DIRNAME
    spikesorting_dir.mkdir(parents=True, exist_ok=True)

    pipeline.output_dir = spikesorting_dir

    # Re-home checkpoints into spikesorting_outputs/checkpoints
    ckpt_root = spikesorting_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    pipeline.checkpoint_file = (
        ckpt_root / f"{pipeline.project_name}_{pipeline.run_id}_{pipeline.stream_id}_checkpoint.json"
    )

    # Reload state from the new checkpoint location (if present)
    try:
        pipeline.state = pipeline._load_checkpoint()
    except Exception as e:
        logger.warning("Could not reload MEA_Analysis checkpoint from spikesorting_outputs: %s", e)

    # Re-home the MEA_Analysis log file too (best-effort)
    try:
        log_file = spikesorting_dir / f"{pipeline.run_id}_{pipeline.stream_id}_pipeline.log"
        mea_logger = logging.getLogger(f"mea_{pipeline.stream_id}")
        mea_logger.handlers.clear()
        pipeline.logger = pipeline._setup_logger(log_file)
    except Exception as e:
        logger.debug("Could not reset MEA_Analysis logger handlers: %s", e)


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

    # Apply environment-level resource controls as early as possible.
    # These can affect numpy/scipy/tensor backends and may propagate to subprocesses.
    def _set_env_int(name: str, value: Optional[int]) -> None:
        if value is None:
            return
        try:
            os.environ[name] = str(int(value))
        except Exception:
            return

    _set_env_int("OMP_NUM_THREADS", inputs.omp_threads)
    _set_env_int("MKL_NUM_THREADS", inputs.mkl_threads)
    _set_env_int("OPENBLAS_NUM_THREADS", inputs.openblas_threads)
    _set_env_int("NUMEXPR_NUM_THREADS", inputs.numexpr_threads)
    if inputs.cuda_visible_devices is not None:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(inputs.cuda_visible_devices)
        except Exception:
            pass

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

    # Configure torch CPU thread pool (analyzer/reports). Sorting in docker may ignore this.
    try:
        import torch  # type: ignore[import-not-found]

        if inputs.torch_threads is not None:
            torch.set_num_threads(int(inputs.torch_threads))
        if inputs.torch_interop_threads is not None:
            torch.set_num_interop_threads(int(inputs.torch_interop_threads))
        logger.info(
            "Torch threads: num_threads=%s num_interop_threads=%s",
            str(getattr(torch, "get_num_threads", lambda: None)()),
            str(getattr(torch, "get_num_interop_threads", lambda: None)()),
        )
    except Exception:
        pass

    # Reuse axon_reconstructor's MEA_Analysis-style output path computation.
    from axon_reconstructor.pipeline.pipeline_driver import _compute_mea_analysis_output_dir

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    # Ensure spikesorting status is captured in the same per-well pipeline log as preprocessing.
    try:
        logger = _setup_well_logger(
            well_out_dir=well_out_dir,
            h5_path=inputs.h5_path,
            stream_id=inputs.stream_id,
            verbose=inputs.verbose,
        )
    except Exception:
        pass

    axon_ckpt_file = _compute_spikesort_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    axon_ckpt = load_checkpoint(
        checkpoint_file=axon_ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    axon_ckpt = save_stage_started(
        checkpoint_file=axon_ckpt_file,
        state=axon_ckpt,
        stage=AxonProcessingStage.SORTING,
        extra_fields={
            "spikesorting_out_dir": str(well_out_dir / SPIKESORTING_OUTPUTS_DIRNAME),
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

    preprocess_dir = _resolve_preprocess_dir(well_out_dir=well_out_dir)
    recording_dir = preprocess_dir / "preprocessed_recording"
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
        recording = si.load_extractor(recording_dir)

    # Build optional sorter kwargs override (primarily for Kilosort4 memory tuning).
    sorter_kwargs: dict = {}
    try:
        if inputs.ks_batch_size is not None:
            sorter_kwargs["batch_size"] = int(inputs.ks_batch_size)
        elif inputs.ks_batch_duration_s is not None:
            fs = float(recording.get_sampling_frequency())
            sorter_kwargs["batch_size"] = int(round(fs * float(inputs.ks_batch_duration_s)))
    except Exception:
        sorter_kwargs = {}

    from MEA_Analysis.IPNAnalysis.mea_analysis_routine import (  # type: ignore[import-not-found]
        MEAPipeline,
        ProcessingStage as MEAProcessingStage,
    )

    auto_merge_presets = None
    auto_merge_steps_params = None
    if bool(inputs.auto_merge_units):
        try:
            diffs = [
                float(x.strip())
                for x in str(inputs.auto_merge_template_diff_thresh).split(",")
                if x.strip()
            ]
            if diffs:
                auto_merge_presets = ["x_contaminations"] * len(diffs)
                auto_merge_steps_params = [
                    {"template_similarity": {"template_diff_thresh": float(t)}} for t in diffs
                ]
        except Exception:
            logger.warning(
                "Invalid auto-merge template diff thresh '%s'; using MEA_Analysis defaults",
                str(inputs.auto_merge_template_diff_thresh),
            )

    logger.info("Initializing MEA_Analysis pipeline object (sorting/analyzer/reports)")
    pipeline = MEAPipeline(
        file_path=str(inputs.h5_path),
        stream_id=inputs.stream_id,
        recording_num=inputs.recording_num,
        output_root=str(inputs.mea_output_root),
        checkpoint_root=None,
        sorter=inputs.sorter,
        docker_image=inputs.docker_image,
        verbose=inputs.verbose,
        cleanup=False,
        force_restart=inputs.force_restart,
        sorter_kwargs=(sorter_kwargs if sorter_kwargs else None),
        auto_merge_units=bool(inputs.auto_merge_units),
        auto_merge_presets=auto_merge_presets,
        auto_merge_steps_params=auto_merge_steps_params,
        force_rerun_analyzer=bool(inputs.force_rerun_analyzer),
    )

    if sorter_kwargs:
        logger.info("MEA_Analysis sorter kwargs override: %s", sorter_kwargs)

    # Best-effort: if MEA_Analysis pipeline object supports resource hints, attach them.
    try:
        if inputs.n_jobs is not None:
            setattr(pipeline, "n_jobs", int(inputs.n_jobs))
        if inputs.chunk_duration is not None:
            setattr(pipeline, "chunk_duration", str(inputs.chunk_duration))
    except Exception:
        pass

    _relocate_mea_analysis_outputs(pipeline=pipeline, well_out_dir=well_out_dir, logger=logger)

    # Inject our preprocessed recording; skip MEA_Analysis preprocessing.
    pipeline.recording = recording
    pipeline.state["stage"] = max(int(pipeline.state.get("stage", 0)), MEAProcessingStage.PREPROCESSING_COMPLETE.value)
    pipeline.state["error"] = None

    logger.info(
        "Starting MEA_Analysis run: sorter=%s docker_image=%s force_restart=%s",
        inputs.sorter,
        inputs.docker_image,
        inputs.force_restart,
    )

    try:
        logger.info("Running MEA_Analysis Phase 2 sorting...")
        pipeline.run_sorting()
        logger.info("MEA_Analysis sorting finished; stage=%s", pipeline.state.get("stage"))

        if inputs.run_analyzer:
            logger.info("Running MEA_Analysis Phase 3 analyzer (computes waveforms/templates/metrics)...")
            pipeline.run_analyzer()
            logger.info("MEA_Analysis analyzer finished; stage=%s", pipeline.state.get("stage"))

        if inputs.run_reports:
            if not inputs.run_analyzer and pipeline.analyzer is None:
                raise RuntimeError(
                    "Requested report generation but analyzer was not run and no existing analyzer was loaded. "
                    "Set run_analyzer=True or run once to populate analyzer_output."
                )

            logger.info(
                "Running MEA_Analysis Phase 4 reports (plots/figures)... no_curation=%s export_to_phy=%s",
                inputs.no_curation,
                inputs.export_to_phy,
            )
            pipeline.generate_reports(
                thresholds=None,
                no_curation=inputs.no_curation,
                export_phy=inputs.export_to_phy,
            )
            logger.info("MEA_Analysis reports finished; stage=%s", pipeline.state.get("stage"))

        sorter_output_dir = pipeline.output_dir / "sorter_output"
        logger.info("Sorting output folder: %s", sorter_output_dir)

        if int(pipeline.state.get("stage", 0)) >= MEAProcessingStage.REPORTS_COMPLETE.value:
            logger.info("MEA_Analysis completed successfully (REPORTS_COMPLETE)")
        elif int(pipeline.state.get("stage", 0)) >= MEAProcessingStage.SORTING_COMPLETE.value:
            logger.info("MEA_Analysis completed sorting successfully (SORTING_COMPLETE)")

        analyzer_dir = pipeline.output_dir / "analyzer_output"
        if inputs.run_reports:
            logger.info("Reports/figures should be under: %s", pipeline.output_dir)

        completed_stage = _resolve_axon_spikesort_completed_stage(int(pipeline.state.get("stage", 0) or 0))
        axon_ckpt = save_stage_completed(
            checkpoint_file=axon_ckpt_file,
            state=axon_ckpt,
            stage=completed_stage,
            extra_fields={
                "spikesorting_out_dir": str(pipeline.output_dir),
                "sorter_output_dir": str(sorter_output_dir),
                "analyzer_dir": str(analyzer_dir),
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
        )
    except Exception as e:
        save_stage_failed(
            checkpoint_file=axon_ckpt_file,
            state=axon_ckpt,
            stage=AxonProcessingStage.SORTING,
            failed_stage="SPIKESORT",
            error=e,
            extra_fields={
                "spikesorting_out_dir": str(well_out_dir / SPIKESORTING_OUTPUTS_DIRNAME),
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
