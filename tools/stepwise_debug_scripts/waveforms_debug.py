#!/usr/bin/env python3
"""Debug harness for running *only* the waveform extraction step.

Contract:
- preprocessing has saved a SpikeInterface recording at:
    <MEA_OUTPUT_ROOT>/<relative_pattern>/<well>/preprocess_outputs/preprocessed_recording
- spikesorting has produced a sorter output at:
    <well>/spikesorting_outputs/sorter_output
- preprocessing has written epoch marker JSONs at:
    <well>/preprocess_outputs/maxwell_contiguous_epochs_<well>.json
    <well>/preprocess_outputs/concatenation_stitch_epochs_<well>.json

This harness extracts waveforms to:
    <well>/waveforms_outputs/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WaveformsInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    sorter: str = "kilosort4"

    ms_before: Optional[float] = None
    ms_after: Optional[float] = None

    n_jobs: int = 8
    max_spikes_per_unit: Optional[int] = None

    per_segment: bool = True
    filter_by_maxwell_epochs: bool = True

    # Debug/perf controls
    debug_max_units: Optional[int] = None
    debug_max_segments: Optional[int] = None

    force_restart: bool = False

    # If True, reuse existing extracted waveforms/analyzers but rewrite grids/panels.
    force_replot: bool = False


def run_waveforms_only(*, inputs: WaveformsInputs, logger: logging.Logger) -> None:
    from axon_reconstructor.pipeline.waveforms import WaveformExtractInputs, extract_waveforms

    out = extract_waveforms(
        inputs=WaveformExtractInputs(
            h5_path=inputs.h5_path,
            stream_id=inputs.stream_id,
            mea_output_root=inputs.mea_output_root,
            sorter=inputs.sorter,
            ms_before=inputs.ms_before,
            ms_after=inputs.ms_after,
            n_jobs=inputs.n_jobs,
            max_spikes_per_unit=inputs.max_spikes_per_unit,
            per_segment=inputs.per_segment,
            force_restart=inputs.force_restart,
            force_replot=inputs.force_replot,
            filter_by_maxwell_epochs=inputs.filter_by_maxwell_epochs,
            debug_max_units=inputs.debug_max_units,
            debug_max_segments=inputs.debug_max_segments,
        )
    )

    logger.info("Waveforms written under: %s", out.waveforms_out_dir)
    logger.info("Concat waveforms: %s", out.concat_waveforms_dir)
    if out.segment_waveforms_dir is not None:
        logger.info("Segment waveforms: %s", out.segment_waveforms_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("Import and call run_waveforms_only() from a project-local debug script.")
