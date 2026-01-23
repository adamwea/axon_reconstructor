#!/usr/bin/env python3
"""Debug harness for running *only* the footprinting step.

Contract:
- waveforms step has produced analyzers at:
    <well>/waveforms_outputs/concat_waveforms/
    <well>/waveforms_outputs/segment_waveforms/segXX_*/

This harness writes footprints under:
    <well>/footprinting_outputs/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FootprintingInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    include_concat: bool = True
    include_segments: bool = True

    plot_concat_footprints_grid_pdf: bool = True
    plot_multi_source_footprints_pdf: bool = True

    n_jobs: int = 8
    unit_limit: Optional[int] = None
    force_restart: bool = False


def run_footprinting_only(*, inputs: FootprintingInputs, logger: logging.Logger) -> None:
    from axon_reconstructor.pipeline.footprinting import FootprintingInputs as StepInputs
    from axon_reconstructor.pipeline.footprinting import run_footprinting

    out = run_footprinting(
        inputs=StepInputs(
            h5_path=inputs.h5_path,
            stream_id=inputs.stream_id,
            mea_output_root=inputs.mea_output_root,
            include_concat=inputs.include_concat,
            include_segments=inputs.include_segments,
            plot_concat_footprints_grid_pdf=inputs.plot_concat_footprints_grid_pdf,
            plot_multi_source_footprints_pdf=inputs.plot_multi_source_footprints_pdf,
            n_jobs=inputs.n_jobs,
            unit_limit=inputs.unit_limit,
            force_restart=inputs.force_restart,
        )
    )

    logger.info("Footprinting outputs under: %s", out.footprinting_out_dir)
    logger.info("Footprinting summary JSON: %s", out.footprinting_summary_json)
    if out.concat_footprints_grid_pdf is not None:
        logger.info("Concat footprints grid PDF: %s", out.concat_footprints_grid_pdf)
    logger.info("multi_source_footprints_dir: %s", out.multi_source_footprints_dir)
    logger.info("multi_source_footprints_summary_json: %s", out.multi_source_footprints_summary_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("Import and call run_footprinting_only() from a project-local debug script.")
