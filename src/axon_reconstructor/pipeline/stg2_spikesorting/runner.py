"""Spikesorting stage runner.

Contract:
- preprocessing has already saved a SpikeInterface recording at:
    <MEA_OUTPUT_ROOT>/<relative_pattern>/<well>/preprocess_outputs/preprocessed_recording
- this stage loads that recording and runs MEA_Analysis sorting/analyzer/reports.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

from ..checkpointing import ProcessingStage as AxonProcessingStage, load_checkpoint
from ..pipeline_logging import log_stage_complete, log_stage_failure, log_stage_start
from ..checkpointing import compute_stage_checkpoint_file, save_stage_completed, save_stage_failed, save_stage_started
from ..stg1_preprocessing.constants import LEGACY_PREPROCESS_OUTPUTS_DIRNAME, PREPROCESS_OUTPUTS_DIRNAME


SPIKESORTING_OUTPUTS_DIRNAME = "spikesort_outputs"
LEGACY_SPIKESORTING_OUTPUTS_DIRNAME = "stg2_spikesorting_outputs"


def _normalize_docker_mount_source(path: str | Path | None) -> str | None:
    if path is None:
        return None
    token = str(path).strip()
    if not token:
        return None
    try:
        return str(Path(token).expanduser().resolve())
    except Exception:
        return str(Path(token).expanduser().absolute())


def _cleanup_interrupted_sorter_containers(
    *,
    docker_image: str | None,
    mount_source: Path,
    logger: logging.Logger,
) -> list[str]:
    image = str(docker_image or "").strip()
    expected_mount_source = _normalize_docker_mount_source(mount_source)
    if not image or expected_mount_source is None or not sys.platform.startswith("linux"):
        return []

    try:
        import docker  # type: ignore[import-not-found]
    except Exception:
        logger.debug("Docker SDK unavailable; interrupted sorter cleanup skipped", exc_info=True)
        return []

    client = None
    removed: list[str] = []
    try:
        client = docker.from_env(timeout=300)
        containers = client.containers.list(all=True, filters={"ancestor": image})
        for container in containers:
            mounts = getattr(container, "attrs", {}).get("Mounts", []) or []
            matches_mount = any(
                _normalize_docker_mount_source(mount.get("Source", None)) == expected_mount_source
                for mount in mounts
                if isinstance(mount, dict)
            )
            if not matches_mount:
                continue
            label = (
                f"{getattr(container, 'name', 'unknown')}"
                f"({getattr(container, 'short_id', getattr(container, 'id', 'unknown'))})"
            )
            try:
                container.remove(force=True)
                removed.append(label)
            except Exception:
                logger.warning("Failed to remove interrupted sorter container %s", label, exc_info=True)
    except Exception:
        logger.warning(
            "Interrupted sorter cleanup failed for image=%s mount_source=%s",
            image,
            expected_mount_source,
            exc_info=True,
        )
    finally:
        try:
            if client is not None:
                client.close()
        except Exception:
            pass

    return removed


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

    # Per-well spikesort output subdirectory (under MEA_Analysis well output root).
    output_subdir_after_well: str = SPIKESORTING_OUTPUTS_DIRNAME

    # Optional per-well preprocess recording override. Relative paths resolve
    # under the computed well output directory.
    preprocess_concat_recording_relpath: Optional[str] = None
    sort_original_preprocess_concat_recording_relpath: Optional[str] = None
    sort_bootstrapped_concat_recording_relpath: Optional[str] = None
    sort_use_bootstrapped_concat_binary: bool = False
    sort_use_lazy_source: bool = True
    sort_assert_one_source: bool = False

    # Logging controls
    log_enabled: bool = True
    log_verbose: bool = False
    log_file_override: Optional[Path | str] = None

    # Debug throttles
    limit_segments_per_well: Optional[int] = None

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

    # Plot options (used by MEA_Analysis report generation)
    plot_mode: str = "separate"
    plot_debug: bool = False
    raster_sort: Optional[str] = None
    fixed_y: bool = False

    # Report options
    no_curation: bool = False
    export_to_phy: bool = False

    # Analyzer options (default-off)
    force_rerun_analyzer: bool = False
    um_kwargs: Optional[dict[str, Any]] = None
    am_kwargs: Optional[dict[str, Any]] = None
    option_kwargs: Optional[dict[str, Any]] = None

    # If True, ignore existing MEA_Analysis checkpoints for this run.
    force_restart: bool = False
    # Optional checkpoint rewind stage in MEA_Analysis (e.g. "merge").
    resume_from: Optional[str] = None


@dataclass(frozen=True)
class SpikeSortingOutputs:
    recording_dir: Path
    sorter_output_dir: Path
    output_dir: Path
    analyzer_dir: Path
    merged_sorting_dir: Optional[Path] = None
    merged_sorter_output_dir: Optional[Path] = None


def _resolve_preprocess_dir(*, well_out_dir: Path) -> Path:
    """Find preprocessing output folder, with backward-compatible fallback."""

    canonical_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
    if canonical_dir.exists():
        return canonical_dir

    legacy_stage_dir = well_out_dir / LEGACY_PREPROCESS_OUTPUTS_DIRNAME
    if legacy_stage_dir.exists():
        return legacy_stage_dir

    legacy_dir = well_out_dir / "axon_reconstructor" / "preprocess"
    if legacy_dir.exists():
        return legacy_dir

    # Default to new location for error messaging.
    return canonical_dir


def _resolve_recording_source_candidate(*, well_out_dir: Path, relpath: str) -> Path:
    candidate = Path(str(relpath)).expanduser()
    if not candidate.is_absolute():
        candidate = (well_out_dir / str(relpath).lstrip("/")).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.name in {"preprocessed_recording", "recording"}:
        return candidate
    nested_recording_dir = candidate / "preprocessed_recording"
    if nested_recording_dir.exists():
        return nested_recording_dir
    return candidate


def _recording_source_relpaths_for_assertion(*, inputs: SpikeSortingInputs) -> list[str]:
    relpaths: list[str] = []
    for value in (
        inputs.sort_original_preprocess_concat_recording_relpath,
        inputs.sort_bootstrapped_concat_recording_relpath,
    ):
        token = str(value or "").strip()
        if token and token not in relpaths:
            relpaths.append(token)
    if not relpaths:
        selected = str(inputs.preprocess_concat_recording_relpath or "").strip()
        if selected:
            relpaths.append(selected)
    return relpaths


def _recording_dir_has_materialized_traces(recording_dir: Path) -> bool:
    if not recording_dir.exists() or not recording_dir.is_dir():
        return False
    trace_suffixes = {".raw", ".dat", ".bin"}
    try:
        for child in recording_dir.rglob("*"):
            if not child.is_file():
                continue
            suffix = child.suffix.lower()
            if suffix in trace_suffixes:
                return True
            if suffix == ".npy" and "trace" in child.stem.lower():
                return True
    except Exception:
        return False
    return False


def run_spikesorting_stage(*, inputs: SpikeSortingInputs, logger: logging.Logger) -> SpikeSortingOutputs:
    """Load saved preprocessed recording and run spikesorting stage."""

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

    try:
        parsed_limit_segments_per_well = (
            int(inputs.limit_segments_per_well) if inputs.limit_segments_per_well is not None else None
        )
    except Exception:
        parsed_limit_segments_per_well = None
    effective_limit_segments_per_well = (
        parsed_limit_segments_per_well
        if parsed_limit_segments_per_well is not None and parsed_limit_segments_per_well > 0
        else None
    )

    # Reuse axon_reconstructor's MEA_Analysis-style output path computation.
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )
    output_subdir_after_well = str(getattr(inputs, "output_subdir_after_well", SPIKESORTING_OUTPUTS_DIRNAME) or "").strip()
    if not output_subdir_after_well:
        output_subdir_after_well = SPIKESORTING_OUTPUTS_DIRNAME
    stage_output_dir = (well_out_dir / output_subdir_after_well).resolve()

    # Configure spikesort stage logging before emitting runtime snapshot lines.
    if bool(inputs.log_enabled):
        try:
            from axon_reconstructor.pipeline.pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger

            if inputs.log_file_override is None:
                log_file = compute_pipeline_log_file(
                    well_out_dir=well_out_dir,
                    data_file=inputs.h5_path,
                    stream_id=inputs.stream_id,
                )
            else:
                log_file = Path(str(inputs.log_file_override)).expanduser()
                if not log_file.is_absolute():
                    log_file = (well_out_dir / log_file).resolve()
                else:
                    log_file = log_file.resolve()

            logger = setup_pipeline_logger(
                log_file=log_file,
                logger_name=f"axon_reconstructor.{log_file.stem}",
                verbose=bool(inputs.log_verbose),
            )
        except Exception:
            pass

    logger.info(
        "Spikesort runtime snapshot: pid=%s cpu_count=%s stream_id=%s sorter=%s docker_image=%s",
        os.getpid(),
        os.cpu_count(),
        inputs.stream_id,
        inputs.sorter,
        inputs.docker_image,
    )

    logger.info(
        "Spikesort runtime config: verbose=%s n_jobs=%s chunk_duration=%s force_restart=%s log_enabled=%s log_verbose=%s",
        bool(inputs.verbose),
        inputs.n_jobs,
        inputs.chunk_duration,
        bool(inputs.force_restart),
        bool(inputs.log_enabled),
        bool(inputs.log_verbose),
    )
    logger.info(
        "Spikesort resources: cuda_visible_devices=%s limit_segments_per_well=%s",
        inputs.cuda_visible_devices,
        (int(effective_limit_segments_per_well) if effective_limit_segments_per_well is not None else None),
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
            "spikesorting_out_dir": str(stage_output_dir),
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

    if bool(inputs.sort_assert_one_source):
        valid_sources: list[Path] = []
        for source_relpath in _recording_source_relpaths_for_assertion(inputs=inputs):
            source_dir = _resolve_recording_source_candidate(well_out_dir=well_out_dir, relpath=source_relpath)
            if source_dir.exists() and source_dir.is_dir() and source_dir not in valid_sources:
                valid_sources.append(source_dir)
        if len(valid_sources) != 1:
            raise RuntimeError(
                "Expected exactly one valid spikesort source recording because sort.assert_one_source=true; "
                f"found={len(valid_sources)} candidates={[str(path) for path in valid_sources]}"
            )

    preprocess_relpath = str(getattr(inputs, "preprocess_concat_recording_relpath", "") or "").strip()
    if preprocess_relpath:
        recording_dir = _resolve_recording_source_candidate(
            well_out_dir=well_out_dir,
            relpath=preprocess_relpath,
        )
        preprocess_dir = recording_dir.parent
    else:
        preprocess_dir = _resolve_preprocess_dir(well_out_dir=well_out_dir)
        recording_dir = preprocess_dir / "preprocessed_recording"
    logger.info(
        "Spikesort source policy: use_bootstrapped_concat_binary=%s use_lazy_source=%s assert_one_source=%s selected_relpath=%s original_relpath=%s bootstrapped_relpath=%s",
        bool(inputs.sort_use_bootstrapped_concat_binary),
        bool(inputs.sort_use_lazy_source),
        bool(inputs.sort_assert_one_source),
        preprocess_relpath or None,
        inputs.sort_original_preprocess_concat_recording_relpath,
        inputs.sort_bootstrapped_concat_recording_relpath,
    )
    logger.info("Resolved preprocessing outputs dir: %s", preprocess_dir)
    if not recording_dir.exists():
        raise FileNotFoundError(
            "Saved preprocessed recording folder not found. "
            "Run preprocessing first with `mea_output_root` set. "
            f"Expected: {recording_dir}"
        )
    if not bool(inputs.sort_use_lazy_source) and not _recording_dir_has_materialized_traces(recording_dir):
        raise FileNotFoundError(
            "Materialized spikesort source recording not found because sort.use_lazy_source=false. "
            f"Expected binary trace files under: {recording_dir}"
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

    if effective_limit_segments_per_well is not None:
        try:
            current_segments = int(recording.get_num_segments())
        except Exception:
            current_segments = 1

        if current_segments > int(effective_limit_segments_per_well):
            if hasattr(si, "select_segment_recording"):
                selected_segments = [int(i) for i in range(int(effective_limit_segments_per_well))]
                recording = si.select_segment_recording(recording=recording, segment_indices=selected_segments)
                logger.info(
                    "Spikesort debug segment limit applied: %d -> %d segment(s)",
                    current_segments,
                    int(effective_limit_segments_per_well),
                )
            else:
                logger.warning(
                    "Spikesort debug segment limit requested (%d) but SpikeInterface has no select_segment_recording; continuing with %d segments",
                    int(effective_limit_segments_per_well),
                    current_segments,
                )

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

    try:
        from IPNAnalysis.mea_analysis_routine import (  # type: ignore[import-not-found]
            MEARunOptions,
            run_mea_pipeline,
            ProcessingStage as MEAProcessingStage,
        )
    except Exception as exc:
        raise RuntimeError(
            "Spikesorting requires MEA_Analysis installed in the active environment "
            "(IPNAnalysis importable)."
        ) from exc

    um_kwargs = dict(inputs.um_kwargs or {})
    am_kwargs = dict(inputs.am_kwargs or {})
    option_kwargs = dict(inputs.option_kwargs or {})
    option_kwargs.setdefault("cuda_visible_devices", inputs.cuda_visible_devices)
    option_kwargs.setdefault("force_rerun_analyzer", bool(inputs.force_rerun_analyzer))
    option_kwargs.setdefault("skip_preprocessing", True)
    option_kwargs.setdefault("preprocessed_recording", recording)

    logger.info(
        "Merge config: unitmatch_merge_units=%s unitmatch_dry_run=%s unitmatch_apply_merges=%s unitmatch_recursive=%s auto_merge_units=%s",
        bool(um_kwargs.get("merge_units", False)),
        bool(um_kwargs.get("dry_run", True)),
        bool(um_kwargs.get("apply_merges", False)),
        bool(um_kwargs.get("recursive", False)),
        bool(am_kwargs.get("enabled", False)),
    )

    logger.info("Initializing MEA_Analysis pipeline options (sorting/analyzer/reports)")
    run_options = MEARunOptions(
        file_path=str(inputs.h5_path),
        stream_id=inputs.stream_id,
        recording_num=inputs.recording_num,
        output_root=str(inputs.mea_output_root),
        output_subdir_after_well=output_subdir_after_well,
        checkpoint_root=None,
        sorter=inputs.sorter,
        docker_image=inputs.docker_image,
        verbose=inputs.verbose,
        cleanup=False,
        force_restart=inputs.force_restart,
        resume_from=inputs.resume_from,
        n_jobs=inputs.n_jobs,
        chunk_duration=inputs.chunk_duration,
        sorter_kwargs=(sorter_kwargs if sorter_kwargs else None),
        um_kwargs=um_kwargs,
        am_kwargs=am_kwargs,
        option_kwargs=option_kwargs,
        skip_spikesorting=False,
        run_analyzer=bool(inputs.run_analyzer),
        run_reports=bool(inputs.run_reports),
        thresholds=None,
        no_curation=bool(inputs.no_curation),
        export_to_phy=bool(inputs.export_to_phy),
        plot_mode=str(inputs.plot_mode or "separate"),
        plot_debug=bool(inputs.plot_debug),
        raster_sort=inputs.raster_sort,
        fixed_y=bool(inputs.fixed_y),
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
        merged_sorting_dir = pipeline.output_dir / "unitmatch_outputs" / "final_merged_sorting"
        if not merged_sorting_dir.exists():
            merged_sorting_dir = None
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
                "merged_sorting_dir": (str(merged_sorting_dir) if merged_sorting_dir is not None else None),
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
            merged_sorting_dir=merged_sorting_dir,
            merged_sorter_output_dir=merged_sorter_output_dir,
        )
    except BaseException as e:
        removed_containers = _cleanup_interrupted_sorter_containers(
            docker_image=inputs.docker_image,
            mount_source=stage_output_dir,
            logger=logger,
        )
        if removed_containers:
            logger.warning(
                "Interrupted sorter cleanup removed %d container(s): %s",
                len(removed_containers),
                removed_containers,
            )
        if not isinstance(e, Exception):
            raise
        logger.exception("Spikesorting stage raised an exception before completion")
        save_stage_failed(
            checkpoint_file=axon_ckpt_file,
            state=axon_ckpt,
            stage=AxonProcessingStage.SORTING,
            failed_stage="SPIKESORT",
            error=e,
            extra_fields={
                "spikesorting_out_dir": str(stage_output_dir),
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
