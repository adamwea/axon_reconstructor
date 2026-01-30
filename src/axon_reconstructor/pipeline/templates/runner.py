from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .plotting import _write_templates_grid_pdf, _write_unit_templates_across_sources_pdf
from .extraction import (
    _choose_grid_source_for_unit,
    _gather_template_sources_for_unit,
    _persist_unit_templates,
)
from .utils import (
    _build_union_template_for_unit,
    _compute_templates_checkpoint_file,
    _jsonable,
    _jsonable_list,
    _jsonable_sequence,
    _read_json,
    _try_get_electrode_ids,
    _write_json,
)

from ..checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from ..pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from ..pipeline_driver import _compute_mea_analysis_output_dir


TEMPLATES_OUTPUTS_DIRNAME = "templates_outputs"


@dataclass(frozen=True)
class TemplateExtractInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    include_concat: bool = True
    include_segments: bool = True

    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    # Optional SpikeInterface auto-merge of units (best-effort).
    run_unit_merging: bool = True
    merge_presets: Optional[list[str]] = None
    merge_recursive: bool = False

    # Plotting
    plot_templates_grid_pdf: bool = True
    plot_multi_source_templates_pdf: bool = True

    # Template overlay plot controls
    top_channels_per_template: int = 8

    n_jobs: int = 8
    force_restart: bool = False


@dataclass(frozen=True)
class TemplateExtractOutputs:
    well_out_dir: Path
    templates_out_dir: Path
    extracted_templates_dir: Path
    merged_union_by_unit_dir: Optional[Path]

    summary_json: Path
    templates_grid_pdf: Optional[Path]
    multi_source_templates_dir: Optional[Path]
    templates_grid_curated_pdf: Optional[Path] = None
    multi_source_templates_dir_uncurated: Optional[Path] = None


def extract_and_merge_templates(*, inputs: TemplateExtractInputs, logger_name_prefix: str = "axon_reconstructor") -> TemplateExtractOutputs:
    """Extract templates from waveforms analyzers, save `.npy`, and produce QC PDFs.

    Mirrors the footprinting step's multi-source logic:
    - loads concat + per-segment waveforms analyzers
    - applies waveforms-stage unit curation (metrics_curated.xlsx) when available
    - handles missing units in some segments by skipping that source
    - builds a per-unit `merged_union` template across sources

        Notes:
        - Templates are sourced from the analyzer `templates` extension.
        - Spike-level exclusions (`wf_exclusions.npz`) are deprecated and are not applied downstream.
            "Curated" vs "uncurated" differs only by unit list selection.
    """

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=inputs.h5_path, stream_id=inputs.stream_id)
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.templates",
        verbose=True,
    )

    templates_out_dir = well_out_dir / TEMPLATES_OUTPUTS_DIRNAME
    extracted_templates_dir = templates_out_dir / "extracted_templates"
    merged_union_by_unit_dir = templates_out_dir / "merged_union_by_unit"

    summary_json = templates_out_dir / "templates_summary.json"
    templates_grid_pdf = templates_out_dir / "templates_grid_uncurated.pdf" if inputs.plot_templates_grid_pdf else None
    templates_grid_curated_pdf = templates_out_dir / "templates_grid_curated.pdf" if inputs.plot_templates_grid_pdf else None
    multi_source_templates_dir = templates_out_dir / "multi_source_by_unit" if inputs.plot_multi_source_templates_pdf else None
    multi_source_templates_dir_uncurated = (
        templates_out_dir / "multi_source_by_unit_uncurated" if inputs.plot_multi_source_templates_pdf else None
    )

    ckpt_file = _compute_templates_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # Resume shortcut.
    # When plotting is enabled, always write BOTH curated+uncurated grids.
    # If waveforms-stage curation exists, "curated" reflects that list; otherwise it's identical to uncurated.
    waveforms_out_dir = well_out_dir / "waveforms_outputs"
    curation_metrics_xlsx_hint = waveforms_out_dir / "metrics_curated.xlsx"
    expect_curated_grid = bool(inputs.plot_templates_grid_pdf)

    resume_ok = (
        (not inputs.force_restart)
        and extracted_templates_dir.exists()
        and summary_json.exists()
        and ((templates_grid_pdf is None) or templates_grid_pdf.exists())
    )
    if expect_curated_grid:
        resume_ok = resume_ok and (templates_grid_curated_pdf is not None) and templates_grid_curated_pdf.exists()

    if resume_ok:
        logger.info("Resuming templates: existing outputs found at %s", templates_out_dir)
        return TemplateExtractOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            extracted_templates_dir=extracted_templates_dir,
            merged_union_by_unit_dir=(merged_union_by_unit_dir if merged_union_by_unit_dir.exists() else None),
            summary_json=summary_json,
            templates_grid_pdf=(templates_grid_pdf if (templates_grid_pdf and templates_grid_pdf.exists()) else None),
            templates_grid_curated_pdf=(
                templates_grid_curated_pdf
                if (templates_grid_curated_pdf and templates_grid_curated_pdf.exists())
                else None
            ),
            multi_source_templates_dir=(
                multi_source_templates_dir if (multi_source_templates_dir and multi_source_templates_dir.exists()) else None
            ),
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={"templates_out_dir": str(templates_out_dir)},
    )

    templates_out_dir.mkdir(parents=True, exist_ok=True)
    extracted_templates_dir.mkdir(parents=True, exist_ok=True)
    merged_union_by_unit_dir.mkdir(parents=True, exist_ok=True)
    if multi_source_templates_dir is not None:
        multi_source_templates_dir.mkdir(parents=True, exist_ok=True)
    if multi_source_templates_dir_uncurated is not None:
        multi_source_templates_dir_uncurated.mkdir(parents=True, exist_ok=True)

    # Load waveforms analyzers (concat + segments).
    import numpy as np  # type: ignore[import-not-found]

    from ..footprinting.utils import (
        _ensure_analyzer_extensions,
        _get_unit_template_from_extension,
        _load_curated_unit_ids_from_waveforms_outputs,
        _load_waveforms_analyzers,
        _normalize_id_for_compare,
        _sparsity_unit_channel_indices,
    )

    analyzers = _load_waveforms_analyzers(
        well_out_dir=well_out_dir,
        include_concat=bool(inputs.include_concat),
        include_segments=bool(inputs.include_segments),
        logger=logger,
    )

    for _, an in analyzers:
        _ensure_analyzer_extensions(analyzer=an, extension_names=["templates"], logger=logger, n_jobs=int(inputs.n_jobs))

    # Determine unit list from concat if present, else from first source.
    unit_ids: list[Any]
    if inputs.unit_ids is not None:
        unit_ids = list(inputs.unit_ids)
    else:
        unit_ids = list(analyzers[0][1].sorting.unit_ids)

    # Preserve an uncurated copy for QC grids.
    unit_ids_all = list(unit_ids)

    # Apply waveforms-stage unit curation if available.
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

    # Time window (for plotting). Best effort: read from waveforms params JSON if present.
    ms_before: Optional[float] = None
    ms_after: Optional[float] = None
    fs_hz: float
    try:
        fs_hz = float(analyzers[0][1].recording.get_sampling_frequency())
    except Exception:
        fs_hz = 10_000.0

    wf_params_json = well_out_dir / "waveforms_outputs" / "waveform_extraction_params.json"
    if wf_params_json.exists():
        try:
            params = _read_json(wf_params_json)
            ms_before = float(params.get("ms_before")) if params.get("ms_before") is not None else None
            ms_after = float(params.get("ms_after")) if params.get("ms_after") is not None else None
        except Exception:
            pass

    unit_grid_entries_uncurated: list[dict[str, Any]] = []
    unit_grid_entries_curated: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "h5_path": str(inputs.h5_path),
        "stream_id": inputs.stream_id,
        "well_out_dir": str(well_out_dir),
        "templates_out_dir": str(templates_out_dir),
        "sources": [name for name, _ in analyzers],
        "templates_grid_pdf": str(templates_grid_pdf) if templates_grid_pdf else None,
        "templates_grid_curated_pdf": str(templates_grid_curated_pdf) if templates_grid_curated_pdf else None,
        "multi_source_templates_dir": str(multi_source_templates_dir) if multi_source_templates_dir else None,
        "multi_source_templates_dir_uncurated": (
            str(multi_source_templates_dir_uncurated) if multi_source_templates_dir_uncurated else None
        ),
        "curation": {
            "metrics_curated_xlsx": str(curation_metrics_xlsx) if curation_metrics_xlsx else None,
            "applied": bool(curated_units_norm is not None),
            "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
        },
        "units": [],
    }

    def _process_unit_list(
        *,
        unit_list: list[Any],
        plot_dir: Optional[Path],
        persist: bool,
    ) -> list[dict[str, Any]]:
        grid_entries: list[dict[str, Any]] = []

        for uid in unit_list:
            sources_for_unit = _gather_template_sources_for_unit(
                uid=uid,
                analyzers=analyzers,
                get_template_from_extension=_get_unit_template_from_extension,
                sparsity_unit_channel_indices=_sparsity_unit_channel_indices,
                try_get_electrode_ids=_try_get_electrode_ids,
            )
            if not sources_for_unit:
                continue

            merged_union = _build_union_template_for_unit(sources_for_unit=sources_for_unit, unit_id=uid, logger=logger)
            sources_for_unit_with_union = list(sources_for_unit) + ([merged_union] if merged_union is not None else [])

            if persist:
                unit_entry = _persist_unit_templates(
                    uid=uid,
                    sources_for_unit_with_union=sources_for_unit_with_union,
                    extracted_templates_dir=extracted_templates_dir,
                    merged_union_by_unit_dir=merged_union_by_unit_dir,
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                    jsonable=_jsonable,
                    jsonable_sequence=_jsonable_sequence,
                    write_json=_write_json,
                    force_restart=bool(inputs.force_restart),
                    logger=logger,
                )
                summary["units"].append(unit_entry)

            chosen = _choose_grid_source_for_unit(sources_for_unit=sources_for_unit)
            if chosen is not None:
                grid_entries.append({"unit_id": uid, "template": chosen["template"]})

            if plot_dir is not None:
                unit_dir = plot_dir / f"unit_{uid}"
                unit_dir.mkdir(parents=True, exist_ok=True)
                unit_pdf = unit_dir / "templates.pdf"
                if (not unit_pdf.exists()) or inputs.force_restart:
                    _write_unit_templates_across_sources_pdf(
                        pdf_path=unit_pdf,
                        unit_id=uid,
                        sources_for_unit=sources_for_unit_with_union,
                        fs_hz=float(fs_hz),
                        ms_before=ms_before,
                        ms_after=ms_after,
                        top_channels=int(inputs.top_channels_per_template),
                    )

        return grid_entries

    # Curated artifacts (unit_ids already curated if waveforms curation exists).
    unit_grid_entries_curated = _process_unit_list(
        unit_list=unit_ids,
        plot_dir=multi_source_templates_dir,
        persist=True,
    )

    # Uncurated artifacts (pre-curation). Note: "uncurated" is now defined by unit list only;
    # spike-level exclusions are not applied downstream.
    if multi_source_templates_dir_uncurated is not None or inputs.plot_templates_grid_pdf:
        unit_grid_entries_uncurated = _process_unit_list(
            unit_list=unit_ids_all,
            plot_dir=multi_source_templates_dir_uncurated,
            persist=False,
        )

    # Write grid PDFs.
    if templates_grid_pdf is not None:
        if (not templates_grid_pdf.exists()) or inputs.force_restart:
            _write_templates_grid_pdf(
                pdf_path=templates_grid_pdf,
                unit_entries=unit_grid_entries_uncurated,
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=int(inputs.top_channels_per_template),
                logger=logger,
            )

    if templates_grid_curated_pdf is not None:
        if (not templates_grid_curated_pdf.exists()) or inputs.force_restart:
            _write_templates_grid_pdf(
                pdf_path=templates_grid_curated_pdf,
                unit_entries=unit_grid_entries_curated,
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=int(inputs.top_channels_per_template),
                logger=logger,
            )

    _write_json(summary_json, summary)

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER_COMPLETE,
        failed_stage=None,
        error=None,
        extra_fields={
            "templates_out_dir": str(templates_out_dir),
            "extracted_templates_dir": str(extracted_templates_dir),
            "merged_union_by_unit_dir": str(merged_union_by_unit_dir),
            "templates_summary_json": str(summary_json),
            "templates_grid_pdf": str(templates_grid_pdf) if templates_grid_pdf else None,
            "templates_grid_curated_pdf": str(templates_grid_curated_pdf) if templates_grid_curated_pdf else None,
        },
    )

    return TemplateExtractOutputs(
        well_out_dir=well_out_dir,
        templates_out_dir=templates_out_dir,
        extracted_templates_dir=extracted_templates_dir,
        merged_union_by_unit_dir=merged_union_by_unit_dir,
        summary_json=summary_json,
        templates_grid_pdf=templates_grid_pdf,
        templates_grid_curated_pdf=templates_grid_curated_pdf,
        multi_source_templates_dir=multi_source_templates_dir,
        multi_source_templates_dir_uncurated=multi_source_templates_dir_uncurated,
    )


__all__ = [
    "TemplateExtractInputs",
    "TemplateExtractOutputs",
    "extract_and_merge_templates",
]
