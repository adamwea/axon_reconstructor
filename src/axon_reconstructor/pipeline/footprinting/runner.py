from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from ..pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from ..pipeline_driver import _compute_mea_analysis_output_dir

from .plotting import (
    _build_global_channel_layout,
    _write_merged_union_footprints_grid_pdf,
)
from .multi_source import emit_multi_source_outputs
from .utils import (
    _compute_footprinting_checkpoint_file,
    _ensure_analyzer_extensions,
    _get_unit_template_from_extension,
    _infer_location_tolerance,
    _load_curated_unit_ids_from_waveforms_outputs,
    _load_wf_rejection_log_summary,
    _load_waveforms_analyzers,
    _normalize_id_for_compare,
    _try_get_electrode_ids,
    _write_json,
)

FOOTPRINTING_OUTPUTS_DIRNAME = "footprinting_outputs"


@dataclass(frozen=True)
class FootprintingInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    include_concat: bool = True
    include_segments: bool = True

    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    plot_concat_footprints_grid_pdf: bool = True
    plot_multi_source_footprints_pdf: bool = True

    n_jobs: int = 8
    force_restart: bool = False


@dataclass(frozen=True)
class FootprintingOutputs:
    well_out_dir: Path
    templates_out_dir: Path
    footprinting_out_dir: Path

    concat_footprints_grid_pdf: Optional[Path]
    multi_source_footprints_dir: Optional[Path]
    multi_source_footprints_summary_json: Optional[Path]

    footprinting_summary_json: Path
    concat_footprints_grid_curated_pdf: Optional[Path] = None


def run_footprinting(*, inputs: FootprintingInputs, logger_name_prefix: str = "axon_reconstructor") -> FootprintingOutputs:
    """Compute/plot footprints from existing waveforms analyzers.

    This step is intentionally independent of template extraction/merging.
    It reads:
      <well>/waveforms_outputs/concat_waveforms/
      <well>/waveforms_outputs/segment_waveforms/*/

        And writes under:
            <well>/footprinting_outputs/

        Notes:
        - Footprinting uses the waveforms analyzer `templates` extension as the template source.
        - Spike-level exclusions (`wf_exclusions.npz`) are deprecated and are not applied here.
            "Curated" vs "uncurated" differs only by unit list selection.
    """

    import numpy as np  # type: ignore[import-not-found]

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(
        well_out_dir=well_out_dir,
        data_file=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.footprinting",
        verbose=True,
    )

    templates_out_dir = well_out_dir / "templates_outputs"
    footprinting_out_dir = well_out_dir / FOOTPRINTING_OUTPUTS_DIRNAME

    concat_grid_pdf = footprinting_out_dir / "footprints_grid_concat.pdf"
    concat_grid_curated_pdf = footprinting_out_dir / "footprints_grid_concat_curated.pdf"
    multi_source_dir = footprinting_out_dir / "footprints_by_source"
    multi_source_summary_json = multi_source_dir / "footprints_by_source_summary.json"
    multi_source_dir_uncurated = footprinting_out_dir / "footprints_by_source_uncurated"
    multi_source_summary_json_uncurated = multi_source_dir_uncurated / "footprints_by_source_uncurated_summary.json"
    merged_union_dir = footprinting_out_dir / "merged_union_by_unit"
    merged_union_summary_json = merged_union_dir / "merged_union_summary.json"
    merged_union_dir_uncurated = footprinting_out_dir / "merged_union_by_unit_uncurated"
    merged_union_summary_json_uncurated = merged_union_dir_uncurated / "merged_union_uncurated_summary.json"
    summary_json = footprinting_out_dir / "footprinting_summary.json"

    ckpt_file = _compute_footprinting_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # When plotting is enabled, always write BOTH curated+uncurated grids.
    # If waveforms-stage curation exists, "curated" reflects that list; otherwise it's identical to uncurated.
    expect_curated_grid = bool(inputs.plot_concat_footprints_grid_pdf)

    resume_ok = summary_json.exists()
    if inputs.plot_concat_footprints_grid_pdf:
        resume_ok = resume_ok and concat_grid_pdf.exists()
        resume_ok = resume_ok and concat_grid_curated_pdf.exists()
    if inputs.plot_multi_source_footprints_pdf:
        resume_ok = resume_ok and multi_source_summary_json.exists()
        resume_ok = resume_ok and merged_union_summary_json.exists()
        resume_ok = resume_ok and multi_source_summary_json_uncurated.exists()
        resume_ok = resume_ok and merged_union_summary_json_uncurated.exists()

    if not inputs.force_restart and resume_ok:
        logger.info("Resuming footprinting: existing outputs found at %s", footprinting_out_dir)
        return FootprintingOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            footprinting_out_dir=footprinting_out_dir,
            concat_footprints_grid_pdf=(concat_grid_pdf if inputs.plot_concat_footprints_grid_pdf else None),
            concat_footprints_grid_curated_pdf=(
                concat_grid_curated_pdf
                if (inputs.plot_concat_footprints_grid_pdf and concat_grid_curated_pdf.exists())
                else None
            ),
            multi_source_footprints_dir=(multi_source_dir if inputs.plot_multi_source_footprints_pdf else None),
            multi_source_footprints_summary_json=(
                multi_source_summary_json if inputs.plot_multi_source_footprints_pdf else None
            ),
            footprinting_summary_json=summary_json,
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={"footprinting_out_dir": str(footprinting_out_dir)},
    )

    try:
        analyzers = _load_waveforms_analyzers(
            well_out_dir=well_out_dir,
            include_concat=bool(inputs.include_concat),
            include_segments=bool(inputs.include_segments),
            logger=logger,
        )

        # Waveforms-stage spike-level exclusions (wf_exclusions.npz) are deprecated.
        # Footprinting now uses the stored analyzers directly.
        exclusions_by_source: dict[str, dict[Any, set[int]]] = {}

        for _, an in analyzers:
            _ensure_analyzer_extensions(
                analyzer=an,
                extension_names=["random_spikes", "waveforms", "templates"],
                logger=logger,
                n_jobs=int(inputs.n_jobs),
            )

        # Determine unit list from concat if present, else from first source.
        unit_ids: list[Any]
        if inputs.unit_ids is not None:
            unit_ids = list(inputs.unit_ids)
        else:
            unit_ids = list(analyzers[0][1].sorting.unit_ids)

        # Preserve an uncurated copy for QC grid PDFs.
        unit_ids_all = list(unit_ids)

        # If waveforms-stage curation ran, it produced a curated unit list.
        # Use it here so footprinting skips rejected units without re-running curation.
        curation_metrics_xlsx: Optional[Path] = None
        curated_units_norm: Optional[list[Any]] = None
        if inputs.unit_ids is None:
            curated_units_norm, curation_metrics_xlsx = _load_curated_unit_ids_from_waveforms_outputs(
                well_out_dir=well_out_dir,
                logger=logger,
            )
            if curated_units_norm is not None:
                curated_set = set(curated_units_norm)
                before = len(unit_ids)
                unit_ids = [uid for uid in unit_ids if _normalize_id_for_compare(uid) in curated_set]
                logger.info(
                    "Applying waveforms curation: %d -> %d units (from %s)",
                    before,
                    len(unit_ids),
                    curation_metrics_xlsx,
                )

        if inputs.unit_limit is not None:
            unit_ids_all = unit_ids_all[: int(inputs.unit_limit)]
            unit_ids = unit_ids[: int(inputs.unit_limit)]

        wf_rejection_summary: Optional[dict[str, Any]] = None
        if inputs.unit_ids is None:
            wf_rejection_summary, _ = _load_wf_rejection_log_summary(well_out_dir=well_out_dir, logger=logger)
            if wf_rejection_summary is not None:
                try:
                    logger.info(
                        "Found waveforms wf_rejection_log.xlsx (rows=%s)",
                        wf_rejection_summary.get("n_rows"),
                    )
                except Exception:
                    pass

        footprinting_out_dir.mkdir(parents=True, exist_ok=True)

        multi_source_summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "sources": [name for name, _ in analyzers],
            "curation": {
                "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                "applied": bool(curated_units_norm is not None),
                "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
            },
            "waveform_rejections": wf_rejection_summary,
            "units": [],
        }

        multi_source_summary_uncurated: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "sources": [name for name, _ in analyzers],
            "curation": {
                "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                "applied": False,
                "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
            },
            "waveform_rejections": wf_rejection_summary,
            "units": [],
        }

        merged_union_summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "curation": {
                "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                "applied": bool(curated_units_norm is not None),
                "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
            },
            "waveform_rejections": wf_rejection_summary,
            "units": [],
        }

        merged_union_summary_uncurated: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "curation": {
                "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                "applied": False,
                "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
            },
            "waveform_rejections": wf_rejection_summary,
            "units": [],
        }

        # Build a global electrode layout across all sources (lets us show non-contributing channels).
        layout_locs, layout_key_to_index, layout_tol = _build_global_channel_layout(analyzers=analyzers)

        if inputs.plot_multi_source_footprints_pdf:
            # Uncurated: all units, pre-exclusion.
            emit_multi_source_outputs(
                unit_list=unit_ids_all,
                analyzers=analyzers,
                out_dir=multi_source_dir_uncurated,
                out_summary=multi_source_summary_uncurated,
                merged_dir=merged_union_dir_uncurated,
                merged_summary=merged_union_summary_uncurated,
                logger=logger,
                layout_locs=layout_locs,
                layout_key_to_index=layout_key_to_index,
                layout_tol=float(layout_tol),
                unit_limit=(int(inputs.unit_limit) if inputs.unit_limit is not None else None),
                plot_pdfs=bool(inputs.plot_multi_source_footprints_pdf),
            )

            # Curated: curated units, exclusions applied.
            emit_multi_source_outputs(
                unit_list=unit_ids,
                analyzers=analyzers,
                out_dir=multi_source_dir,
                out_summary=multi_source_summary,
                merged_dir=merged_union_dir,
                merged_summary=merged_union_summary,
                logger=logger,
                layout_locs=layout_locs,
                layout_key_to_index=layout_key_to_index,
                layout_tol=float(layout_tol),
                unit_limit=(int(inputs.unit_limit) if inputs.unit_limit is not None else None),
                plot_pdfs=bool(inputs.plot_multi_source_footprints_pdf),
            )

        if inputs.plot_multi_source_footprints_pdf:
            _write_json(multi_source_summary_json, multi_source_summary)
            _write_json(merged_union_summary_json, merged_union_summary)
            _write_json(multi_source_summary_json_uncurated, multi_source_summary_uncurated)
            _write_json(merged_union_summary_json_uncurated, merged_union_summary_uncurated)

        if inputs.plot_concat_footprints_grid_pdf:
            # Uncurated grid: all units, pre-exclusion, merged_union on global layout.
            _write_merged_union_footprints_grid_pdf(
                analyzers=analyzers,
                pdf_path=concat_grid_pdf,
                unit_ids=unit_ids_all,
                exclusions_by_source=exclusions_by_source,
                apply_exclusions=False,
                layout_locs=layout_locs,
                layout_key_to_index=layout_key_to_index,
                layout_tol=float(layout_tol),
                logger=logger,
            )
            # Curated grid: curated units, exclusions applied, merged_union on global layout.
            _write_merged_union_footprints_grid_pdf(
                analyzers=analyzers,
                pdf_path=concat_grid_curated_pdf,
                unit_ids=unit_ids,
                exclusions_by_source=exclusions_by_source,
                apply_exclusions=True,
                layout_locs=layout_locs,
                layout_key_to_index=layout_key_to_index,
                layout_tol=float(layout_tol),
                logger=logger,
            )

        _write_json(
            summary_json,
            {
                "h5_path": str(inputs.h5_path),
                "stream_id": inputs.stream_id,
                "well_out_dir": str(well_out_dir),
                "footprinting_out_dir": str(footprinting_out_dir),
                "sources": [name for name, _ in analyzers],
                "curation": {
                    "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
                    "applied": bool(curated_units_norm is not None),
                    "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
                },
                "concat_footprints_grid_pdf": str(concat_grid_pdf) if inputs.plot_concat_footprints_grid_pdf else None,
                "concat_footprints_grid_curated_pdf": str(concat_grid_curated_pdf)
                if inputs.plot_concat_footprints_grid_pdf
                else None,
                "multi_source_footprints_dir": str(multi_source_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "multi_source_footprints_summary_json": str(multi_source_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "multi_source_footprints_dir_uncurated": str(multi_source_dir_uncurated)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "multi_source_footprints_summary_json_uncurated": str(multi_source_summary_json_uncurated)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "merged_union_by_unit_dir": str(merged_union_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "merged_union_summary_json": str(merged_union_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "merged_union_by_unit_dir_uncurated": str(merged_union_dir_uncurated)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "merged_union_summary_json_uncurated": str(merged_union_summary_json_uncurated)
                if inputs.plot_multi_source_footprints_pdf
                else None,
            },
        )

        ckpt = save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "footprinting_out_dir": str(footprinting_out_dir),
                "concat_footprints_grid_pdf": str(concat_grid_pdf) if inputs.plot_concat_footprints_grid_pdf else None,
                "concat_footprints_grid_curated_pdf": str(concat_grid_curated_pdf)
                if inputs.plot_concat_footprints_grid_pdf
                else None,
                "multi_source_footprints_dir": str(multi_source_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "multi_source_footprints_summary_json": str(multi_source_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "merged_union_by_unit_dir": str(merged_union_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "merged_union_summary_json": str(merged_union_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "summary_json": str(summary_json),
            },
        )

        return FootprintingOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            footprinting_out_dir=footprinting_out_dir,
            concat_footprints_grid_pdf=(concat_grid_pdf if inputs.plot_concat_footprints_grid_pdf else None),
            concat_footprints_grid_curated_pdf=(
                concat_grid_curated_pdf if inputs.plot_concat_footprints_grid_pdf else None
            ),
            multi_source_footprints_dir=(multi_source_dir if inputs.plot_multi_source_footprints_pdf else None),
            multi_source_footprints_summary_json=(multi_source_summary_json if inputs.plot_multi_source_footprints_pdf else None),
            footprinting_summary_json=summary_json,
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage=ProcessingStage.ANALYZER.name,
            error=exception_to_error_dict(e),
            extra_fields={"footprinting_out_dir": str(footprinting_out_dir)},
        )
        raise
