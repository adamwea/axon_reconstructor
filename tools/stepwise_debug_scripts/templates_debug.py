#!/usr/bin/env python3
"""Debug harness for running *only* the template extraction + merging step.

Contract:
- waveforms step has produced analyzers at:
    <well>/waveforms_outputs/concat_waveforms/
    <well>/waveforms_outputs/segment_waveforms/segXX_*/

This harness writes extracted templates + QC PDF to:
    <well>/templates_outputs/extracted_templates/
    <well>/templates_outputs/templates_grid_uncurated.pdf
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TemplatesInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    include_concat: bool = True

    plot_templates_grid_pdf: bool = True
    plot_footprints_grid_pdf: bool = True
    n_jobs: int = 8

    unit_limit: Optional[int] = None
    force_restart: bool = False


def run_templates_only(*, inputs: TemplatesInputs, logger: logging.Logger) -> None:
    from axon_reconstructor.pipeline.templates_extraction import TemplateExtractionInputs, extract_templates

    out = extract_templates(
        inputs=TemplateExtractionInputs(
            h5_path=inputs.h5_path,
            stream_id=inputs.stream_id,
            mea_output_root=inputs.mea_output_root,
            include_concat=inputs.include_concat,
            plot_templates_grid_pdf=inputs.plot_templates_grid_pdf,
            plot_footprints_grid_pdf=inputs.plot_footprints_grid_pdf,
            n_jobs=inputs.n_jobs,
            unit_limit=inputs.unit_limit,
            force_restart=inputs.force_restart,
            plot_multi_source_footprints_pdf=True,
            include_segments=True,
        )
    )

    logger.info("Templates written under: %s", out.templates_out_dir)
    logger.info("Extracted templates: %s", out.extracted_templates_dir)
    logger.info("Summary JSON: %s", out.summary_json)
    if out.templates_grid_pdf is not None:
        logger.info("Templates grid PDF: %s", out.templates_grid_pdf)
    if out.footprints_grid_pdf is not None:
        logger.info("Footprints grid PDF: %s", out.footprints_grid_pdf)
    logger.info("multi_source_footprints_dir: %s", out.multi_source_footprints_dir)
    logger.info("multi_source_footprints_summary_json: %s", out.multi_source_footprints_summary_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("Import and call run_templates_only() from a project-local debug script.")
