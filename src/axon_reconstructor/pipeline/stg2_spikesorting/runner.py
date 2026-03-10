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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
        stage_name="spikesort",
        logger_name_prefix="axon_reconstructor",
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
    # Testing override: force merge phase to run on resume by asking MEA_Analysis
    # to rerun analyzer pipeline steps.
    force_merge_on_resume: bool = False
    unitmatch_merge_units: bool = False
    unitmatch_dry_run: bool = True
    auto_merge_units: bool = False
    # CSV string like "0.05,0.15,0.25"; only used if auto_merge_units=True
    auto_merge_template_diff_thresh: str = "0.05,0.15,0.25"

    # Optional post-processing: recursively merge units that fall in the same
    # 4x4 (configurable) channel block based on unit primary channel location.
    post_merge_4x4_units: bool = False
    post_merge_block_size_channels: int = 4
    post_merge_recursive: bool = True
    post_merge_max_iterations: int = 8
    post_merge_channel_pitch_um: float = 17.5

    # If True, ignore existing MEA_Analysis checkpoints for this run.
    force_restart: bool = False


@dataclass(frozen=True)
class SpikeSortingOutputs:
    recording_dir: Path
    sorter_output_dir: Path
    output_dir: Path
    analyzer_dir: Path
    merged_sorter_output_dir: Optional[Path] = None


def _save_sorting_folder(*, sorting: Any, out_dir: Path, overwrite: bool) -> None:
    import shutil

    if out_dir.exists() and bool(overwrite):
        shutil.rmtree(out_dir, ignore_errors=True)

    if out_dir.exists() and (not bool(overwrite)):
        return

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        sorting.save(folder=out_dir)
    except TypeError:
        sorting.save(folder=out_dir, overwrite=True)


def _compute_unit_primary_locations(
    *,
    sorting: Any,
    analyzer: Any,
    logger: logging.Logger,
) -> dict[Any, tuple[float, float]]:
    import numpy as np  # type: ignore[import-not-found]

    from ..stg4_templates.multi_source_utils import _get_unit_template_from_extension, _sparsity_unit_channel_indices

    locations_xy = np.asarray(analyzer.recording.get_channel_locations(), dtype=float)[:, :2]
    channel_ids = list(analyzer.recording.get_channel_ids())

    templates_ext = analyzer.get_extension("templates") if analyzer.has_extension("templates") else None
    if templates_ext is None:
        analyzer.compute(["templates"], verbose=False, n_jobs=1)
        templates_ext = analyzer.get_extension("templates") if analyzer.has_extension("templates") else None
    if templates_ext is None:
        return {}

    sparsity = getattr(analyzer, "sparsity", None)
    by_unit: dict[Any, tuple[float, float]] = {}
    for uid in list(sorting.get_unit_ids()):
        try:
            tmpl = _get_unit_template_from_extension(analyzer=analyzer, templates_ext=templates_ext, unit_id=uid)
            if tmpl is None:
                continue
            tmpl_arr = np.asarray(tmpl, dtype=float)
            if tmpl_arr.ndim != 2 or int(tmpl_arr.shape[1]) <= 0:
                continue

            ptp = np.ptp(tmpl_arr, axis=0)
            local_best = int(np.argmax(ptp))

            ch_indices = _sparsity_unit_channel_indices(sparsity=sparsity, unit_id=uid)
            if ch_indices is not None and int(len(ch_indices)) == int(ptp.shape[0]):
                global_ch_index = int(ch_indices[local_best])
            elif int(ptp.shape[0]) == int(len(channel_ids)):
                global_ch_index = int(local_best)
            else:
                continue

            if global_ch_index < 0 or global_ch_index >= int(locations_xy.shape[0]):
                continue
            loc = locations_xy[global_ch_index]
            by_unit[uid] = (float(loc[0]), float(loc[1]))
        except Exception:
            continue

    logger.info("Post-merge location map: %d/%d units localized", len(by_unit), len(list(sorting.get_unit_ids())))
    return by_unit


def _build_merge_groups_by_block(
    *,
    unit_locations_xy: dict[Any, tuple[float, float]],
    block_size_channels: int,
    channel_pitch_um: float,
) -> list[list[Any]]:
    import math

    block_um = float(max(1, int(block_size_channels))) * float(channel_pitch_um)
    groups_by_key: dict[tuple[int, int], list[Any]] = {}
    for uid, (x_um, y_um) in unit_locations_xy.items():
        key = (int(math.floor(float(x_um) / block_um)), int(math.floor(float(y_um) / block_um)))
        groups_by_key.setdefault(key, []).append(uid)

    groups = [list(v) for v in groups_by_key.values() if int(len(v)) > 1]
    groups.sort(key=lambda g: (-len(g), str(g[0]) if g else ""))
    return groups


def _write_unit_locations_png(
    *,
    unit_locations_xy: dict[Any, tuple[float, float]],
    out_path: Path,
    title: str,
) -> None:
    if not unit_locations_xy:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np  # type: ignore[import-not-found]

    pts = np.asarray(list(unit_locations_xy.values()), dtype=float)
    if pts.ndim != 2 or int(pts.shape[1]) < 2:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=20,
        c="black",
        marker="o",
        linewidths=0,
        alpha=0.85,
    )
    ax.set_title(str(title), fontsize=11)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal", adjustable="box")
    try:
        ax.grid(False)
    except Exception:
        pass

    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _run_post_merge_4x4(
    *,
    sorting: Any,
    recording: Any,
    out_dir: Path,
    logger: logging.Logger,
    block_size_channels: int,
    recursive: bool,
    max_iterations: int,
    channel_pitch_um: float,
    force_restart: bool,
) -> tuple[Any, dict[str, Any]]:
    import spikeinterface.full as si  # type: ignore[import-not-found]
    from spikeinterface.curation import MergeUnitsSorting  # type: ignore[import-not-found]

    meta_json = out_dir.parent / f"{out_dir.name}_meta.json"
    if out_dir.exists() and meta_json.exists() and (not bool(force_restart)):
        try:
            merged = si.load_extractor(out_dir)
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                meta = {}

            existing_locations_png = meta.get("locations_png") if isinstance(meta.get("locations_png"), str) else None
            needs_locations_backfill = True
            if existing_locations_png is not None:
                try:
                    needs_locations_backfill = (not Path(existing_locations_png).exists())
                except Exception:
                    needs_locations_backfill = True

            if needs_locations_backfill:
                try:
                    analyzer_cached = si.create_sorting_analyzer(
                        sorting=merged,
                        recording=recording,
                        format="memory",
                        sparse=True,
                    )
                    try:
                        analyzer_cached.compute(["templates"], verbose=False, n_jobs=1)
                    except Exception:
                        analyzer_cached.compute(["random_spikes", "waveforms", "templates"], verbose=False, n_jobs=1)

                    locs_cached = _compute_unit_primary_locations(
                        sorting=merged,
                        analyzer=analyzer_cached,
                        logger=logger,
                    )
                    locations_png_cached = out_dir.parent / f"locations_{int(len(list(merged.get_unit_ids())))}_units_merged_4x4.png"
                    _write_unit_locations_png(
                        unit_locations_xy=locs_cached,
                        out_path=locations_png_cached,
                        title=f"Merged unit locations ({int(len(list(merged.get_unit_ids())))} units)",
                    )
                    if locations_png_cached.exists():
                        meta["locations_png"] = str(locations_png_cached)
                        try:
                            meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                except Exception:
                    pass

            return merged, meta
        except Exception:
            pass

    current_sorting = sorting
    merges_per_iter: list[int] = []
    units_per_iter: list[int] = [int(len(list(current_sorting.get_unit_ids())))]
    latest_unit_locs: dict[Any, tuple[float, float]] = {}

    max_iter = int(max(1, int(max_iterations))) if bool(recursive) else 1
    for i in range(max_iter):
        analyzer = si.create_sorting_analyzer(
            sorting=current_sorting,
            recording=recording,
            format="memory",
            sparse=True,
        )
        try:
            analyzer.compute(["templates"], verbose=False, n_jobs=1)
        except Exception:
            try:
                analyzer.compute(["random_spikes", "waveforms", "templates"], verbose=False, n_jobs=1)
            except Exception:
                break

        unit_locs = _compute_unit_primary_locations(sorting=current_sorting, analyzer=analyzer, logger=logger)
        latest_unit_locs = dict(unit_locs)
        groups = _build_merge_groups_by_block(
            unit_locations_xy=unit_locs,
            block_size_channels=int(block_size_channels),
            channel_pitch_um=float(channel_pitch_um),
        )

        if not groups:
            merges_per_iter.append(0)
            break

        logger.info("Post-merge iteration %d: merging %d groups", int(i + 1), int(len(groups)))
        try:
            current_sorting = MergeUnitsSorting(current_sorting, units_to_merge=groups)
        except Exception as e:
            logger.warning("Post-merge iteration %d failed: %s", int(i + 1), e)
            break

        try:
            current_sorting = current_sorting.remove_empty_units()
        except Exception:
            pass

        merges_per_iter.append(int(len(groups)))
        units_per_iter.append(int(len(list(current_sorting.get_unit_ids()))))

    _save_sorting_folder(sorting=current_sorting, out_dir=out_dir, overwrite=bool(force_restart))

    if not latest_unit_locs:
        try:
            analyzer_final = si.create_sorting_analyzer(
                sorting=current_sorting,
                recording=recording,
                format="memory",
                sparse=True,
            )
            try:
                analyzer_final.compute(["templates"], verbose=False, n_jobs=1)
            except Exception:
                analyzer_final.compute(["random_spikes", "waveforms", "templates"], verbose=False, n_jobs=1)
            latest_unit_locs = _compute_unit_primary_locations(sorting=current_sorting, analyzer=analyzer_final, logger=logger)
        except Exception:
            latest_unit_locs = {}

    locations_png = out_dir.parent / f"locations_{int(len(list(current_sorting.get_unit_ids())))}_units_merged_4x4.png"
    try:
        _write_unit_locations_png(
            unit_locations_xy=latest_unit_locs,
            out_path=locations_png,
            title=f"Merged unit locations ({int(len(list(current_sorting.get_unit_ids())))} units)",
        )
    except Exception:
        pass

    meta = {
        "enabled": True,
        "block_size_channels": int(block_size_channels),
        "channel_pitch_um": float(channel_pitch_um),
        "recursive": bool(recursive),
        "max_iterations": int(max_iterations),
        "merges_per_iteration": merges_per_iter,
        "unit_counts_per_iteration": units_per_iter,
        "n_units_final": int(len(list(current_sorting.get_unit_ids()))),
        "merged_sorter_output_dir": str(out_dir),
        "locations_png": (str(locations_png) if locations_png.exists() else None),
    }
    try:
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

    return current_sorting, meta


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
    """Force MEA_Analysis to write all outputs under <well>/stg2_spikesorting_outputs/.

    We keep MEA_Analysis itself unmodified by overriding the pipeline object's
    output_dir + checkpoint_file after construction.
    """

    spikesorting_dir = well_out_dir / SPIKESORTING_OUTPUTS_DIRNAME
    spikesorting_dir.mkdir(parents=True, exist_ok=True)

    # Preserve/restore output_root expected by some MEA_Analysis report code paths
    # (e.g., fixed-y burst plotting summary lookup).
    try:
        existing_output_dir = Path(getattr(pipeline, "output_dir"))
        # output_dir is typically: <output_root>/<relative_pattern>/<well>
        inferred_output_root = existing_output_dir.parent.parent
    except Exception:
        inferred_output_root = well_out_dir.parent.parent

    try:
        if getattr(pipeline, "output_root", None) is None:
            setattr(pipeline, "output_root", Path(inferred_output_root))
    except Exception:
        try:
            setattr(pipeline, "output_root", Path(inferred_output_root))
        except Exception:
            logger.debug("Could not set MEA_Analysis output_root attribute", exc_info=True)

    pipeline.output_dir = spikesorting_dir

    # Re-home checkpoints into stg2_spikesorting_outputs/checkpoints
    ckpt_root = spikesorting_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    pipeline.checkpoint_file = (
        ckpt_root / f"{pipeline.project_name}_{pipeline.run_id}_{pipeline.stream_id}_checkpoint.json"
    )

    # Reload state from the new checkpoint location (if present)
    try:
        pipeline.state = pipeline._load_checkpoint()
    except Exception as e:
        logger.warning("Could not reload MEA_Analysis checkpoint from stg2_spikesorting_outputs: %s", e)

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
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    well_out_dir = compute_mea_analysis_output_dir(
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
        unitmatch_merge_units=bool(inputs.unitmatch_merge_units),
        unitmatch_dry_run=bool(inputs.unitmatch_dry_run),
        auto_merge_units=bool(inputs.auto_merge_units),
        auto_merge_presets=auto_merge_presets,
        auto_merge_steps_params=auto_merge_steps_params,
        force_rerun_analyzer=bool(inputs.force_rerun_analyzer or inputs.force_merge_on_resume),
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

        # Run optional merge stage explicitly so UnitMatch/auto-merge logic is
        # exercised when stg2 invokes MEA_Analysis methods directly.
        if hasattr(pipeline, "run_optional_merge_phase"):
            logger.info("Running MEA_Analysis Phase 2.5 optional merge...")
            pipeline.run_optional_merge_phase()
            logger.info("MEA_Analysis merge phase finished; stage=%s", pipeline.state.get("stage"))

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
        merged_sorter_output_dir: Optional[Path] = None
        merged_meta: Optional[dict[str, Any]] = None

        if bool(inputs.post_merge_4x4_units):
            merged_sorter_output_dir = pipeline.output_dir / "sorter_output_merged_4x4"
            logger.info(
                "Running optional post-merge (4x4 blocks): recursive=%s block_size=%d",
                bool(inputs.post_merge_recursive),
                int(inputs.post_merge_block_size_channels),
            )
            _merged_sorting, merged_meta = _run_post_merge_4x4(
                sorting=pipeline.sorting,
                recording=recording,
                out_dir=merged_sorter_output_dir,
                logger=logger,
                block_size_channels=int(inputs.post_merge_block_size_channels),
                recursive=bool(inputs.post_merge_recursive),
                max_iterations=int(inputs.post_merge_max_iterations),
                channel_pitch_um=float(inputs.post_merge_channel_pitch_um),
                force_restart=bool(inputs.force_restart),
            )
            logger.info("Post-merge sorting saved -> %s", merged_sorter_output_dir)

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
                "merged_sorter_output_dir": (str(merged_sorter_output_dir) if merged_sorter_output_dir is not None else None),
                "merged_locations_png": (
                    str((merged_meta or {}).get("locations_png"))
                    if isinstance((merged_meta or {}).get("locations_png"), str)
                    else None
                ),
                "post_merge_4x4": (merged_meta if merged_meta is not None else {"enabled": False}),
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
