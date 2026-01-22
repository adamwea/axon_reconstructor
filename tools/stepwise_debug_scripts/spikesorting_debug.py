#!/usr/bin/env python3
"""Debug harness for running *only* the spike-sorting step.

This is intentionally parallel to `preprocessing_debug.py`:
- project-local wrapper script sets paths and hardcoded dataset
- this module contains the heavier logic

Contract:
- preprocessing has already saved a SpikeInterface recording at:
    <MEA_OUTPUT_ROOT>/<relative_pattern>/<well>/axon_reconstructor/preprocess/preprocessed_recording
- this harness loads that recording and runs MEA_Analysis Phase 2 (sorting)
"""

from __future__ import annotations

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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

    # Post-sorting steps (these are where most figures are generated)
    run_analyzer: bool = True
    run_reports: bool = True

    # Report options
    no_curation: bool = False
    export_to_phy: bool = False

    # If True, ignore existing MEA_Analysis checkpoints for this run.
    force_restart: bool = False


@dataclass(frozen=True)
class SpikeSortingOutputs:
    recording_dir: Path
    sorter_output_dir: Path
    output_dir: Path
    analyzer_dir: Path


def _ensure_mea_analysis_importable(mea_analysis_repo_root: Path) -> None:
    """Make `import MEA_Analysis...` work in ad-hoc debug sessions."""

    mea_analysis_repo_root = Path(mea_analysis_repo_root).expanduser().resolve()
    # To import `MEA_Analysis`, Python must have the *parent* directory on sys.path.
    # Example: /home/.../pkgs is on sys.path, then /home/.../pkgs/MEA_Analysis is importable.
    mea_analysis_parent = mea_analysis_repo_root.parent
    if str(mea_analysis_parent) not in sys.path:
        sys.path.insert(0, str(mea_analysis_parent))


def run_spikesorting_only(*, inputs: SpikeSortingInputs, logger: logging.Logger) -> SpikeSortingOutputs:
    """Load saved preprocessed recording and run MEA_Analysis sorting."""

    _ensure_mea_analysis_importable(inputs.mea_analysis_repo_root)

    import spikeinterface.full as si  # type: ignore[import-not-found]

    # Reuse axon_reconstructor's MEA_Analysis-style output path computation.
    from axon_reconstructor.pipeline.pipeline_driver import _compute_mea_analysis_output_dir

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    preprocess_dir = well_out_dir / "axon_reconstructor" / "preprocess"
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

    from MEA_Analysis.IPNAnalysis.mea_analysis_routine import (  # type: ignore[import-not-found]
        MEAPipeline,
        ProcessingStage,
    )

    logger.info("Initializing MEA_Analysis pipeline object (sorting-only)")
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
    )

    # Inject our preprocessed recording; skip MEA_Analysis preprocessing.
    pipeline.recording = recording
    pipeline.state["stage"] = max(
        int(pipeline.state.get("stage", 0)),
        ProcessingStage.PREPROCESSING_COMPLETE.value,
    )
    pipeline.state["error"] = None

    logger.info("Running MEA_Analysis Phase 2 sorting...")
    pipeline.run_sorting()

    if inputs.run_analyzer:
        logger.info("Running MEA_Analysis Phase 3 analyzer (computes waveforms/templates/metrics)...")
        pipeline.run_analyzer()

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

    sorter_output_dir = pipeline.output_dir / "sorter_output"
    logger.info("Sorting output folder: %s", sorter_output_dir)

    analyzer_dir = pipeline.output_dir / "analyzer_output"
    if inputs.run_reports:
        logger.info("Reports/figures should be under: %s", pipeline.output_dir)

    return SpikeSortingOutputs(
        recording_dir=recording_dir,
        sorter_output_dir=sorter_output_dir,
        output_dir=pipeline.output_dir,
        analyzer_dir=analyzer_dir,
    )
