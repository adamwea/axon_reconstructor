from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .plotting import (
    _write_footprint_ptp_map,
    _write_templates_grid_pdf,
    _write_unit_segment_grids_pdf,
    _write_unit_template_and_footprint_svg,
    _write_unit_segment_footprint_grids_pdf,
    _write_topo_unit_footprint_png,
    _write_unit_propagation_plots_pdf,
)

from .processing import (
    _apply_spikesorting_stage_unit_curation,
    _infer_template_plot_window,
    process_unit_list,
)
from .utils import _compute_templates_checkpoint_file, _jsonable, _jsonable_sequence, _read_json, _write_json

from ..checkpointing import (
    ProcessingStage,
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

    # Templates should only run on curated units derived from spikesorting metrics.
    # If True and `unit_ids` is None, this requires `<well>/spikesorting_outputs/qm_unfiltered.xlsx`.
    require_curated_units: bool = True

    # Optional SpikeInterface auto-merge of units (best-effort).
    run_unit_merging: bool = True
    merge_presets: Optional[list[str]] = None
    merge_recursive: bool = False

    # Plotting
    plot_templates_grid_pdf: bool = True
    plot_multi_source_templates_pdf: bool = True

    # Disabled for now (kept only for backwards compatibility):
    # this output isn't currently desired and was producing confusing artifacts.
    plot_templates_grid_panels_svg: bool = False

    # Footprints
    plot_merged_contributing_footprints_linear_and_log: bool = True

    # Save full-channels templates (zeros on non-contributing channels) for reconstruction.
    # These are persisted on a deterministic “full channels” axis (e.g. Maxwell full chip = 26,400).
    save_full_channels_templates: bool = True

    # 3D topographical footprint plots (PTP amplitude as height).
    plot_topo_unit_footprints: bool = True

    # Propagation plots (ordered extracted waveforms by spike time, concat analyzer).
    plot_propagation_plots: bool = True
    propagation_top_channels: int = 25
    propagation_n_waveforms: int = 12
    propagation_channels_per_panel: int = 25
    propagation_channel_overlap: int = 5

    # If True, label each propagated trace with its electrode id.
    # Implemented by annotating matplotlib output after axon_velocity renders.
    propagation_show_electrode_ids: bool = False

    # Post-render styling knobs for axon_velocity propagation plots.
    # `propagation_trace_gain` scales waveform amplitude around each trace's baseline.
    # `propagation_trace_spacing` scales vertical spacing between traces (lower -> more overlap).
    propagation_trace_gain: float = 1.0
    propagation_trace_spacing: float = 1.0

    # Optional: generate real axon_velocity plot bundle from merged contributing-channels templates.
    # This writes to per-unit `axon_velocity_outputs/` and requires the axon_velocity deps.
    plot_axon_velocity_outputs: bool = False

    # Template overlay plot controls
    top_channels_per_template: int = 8

    n_jobs: int = 8
    force_restart: bool = False


@dataclass(frozen=True)
class TemplateExtractOutputs:
    well_out_dir: Path
    templates_out_dir: Path
    extracted_templates_dir: Path
    merged_units_dir: Optional[Path]
    merged_unit_plots_dir: Optional[Path]

    summary_json: Path
    templates_grid_pdf: Optional[Path]
    multi_source_templates_dir: Optional[Path]


def extract_and_merge_templates(*, inputs: TemplateExtractInputs, logger_name_prefix: str = "axon_reconstructor") -> TemplateExtractOutputs:
    """Extract templates from waveforms analyzers, persist artifacts, and produce QC plots.

        Scientific / data-handling contract:
        - Templates are sourced from the waveforms-stage `SortingAnalyzer` artifacts (`templates` extension).
        - Spike-level exclusions belong to the waveforms stage and are **not** re-applied downstream.
            (The deprecated `wf_exclusions.npz` is intentionally not consumed here.)
        - "Curation" at this stage refers only to selecting which unit ids are processed, by default
            using curated units derived from spikesorting-stage quality metrics.

        Multi-source logic (concat + optional per-segment analyzers):
        - Loads concat + per-segment waveforms analyzers.
        - Handles missing units in some sources by skipping that source for the unit.
        - Builds a per-unit `merged_contributing` template across sources (contributing channels across sources).
        - For overlapping channels across sources (same physical channel), merges the waveform using
            a mean-of-waveforms strategy (best-effort) to avoid keep-first bias.

        QC outputs:
        - Grid PDF (`templates_grid.pdf`) and optional per-unit multi-source overlay PDFs.
        - Per-unit merged_contributing footprint PTP maps (linear + log) and combined SVG panels.
        - Optional per-unit SVG panels for the grid entries (linear + log footprint).
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
    merged_units_dir = templates_out_dir / "merged_units"

    # Plot outputs (plot-type dirs directly under templates_outputs/)
    merged_unit_footprints_dir = templates_out_dir / "footprints"
    merged_unit_footprints_zoomed_dir = templates_out_dir / "footprints_zoomed"
    merged_unit_svgs_dir = templates_out_dir / "svgs"
    merged_unit_full_chip_maps_dir = templates_out_dir / "full_chip_maps"
    axon_velocity_outputs_root_dir = templates_out_dir / "axon_velocity_outputs"

    full_channels_templates_dir = templates_out_dir / "full_channels_templates"
    topo_unit_footprints_dir = templates_out_dir / "topo_unit_footprints"
    propagation_plots_dir = templates_out_dir / "propagation_plots"

    summary_json = templates_out_dir / "templates_summary.json"
    templates_grid_pdf = templates_out_dir / "templates_grid.pdf" if inputs.plot_templates_grid_pdf else None
    # Per-unit concat-vs-segment grids (flattened files).
    unit_segment_grids_dir = templates_out_dir / "unit_segment_grids" if inputs.plot_multi_source_templates_pdf else None

    # Intentionally disabled.
    templates_grid_panels_dir = None

    ckpt_file = _compute_templates_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # Resume shortcut.
    resume_ok = (not inputs.force_restart) and extracted_templates_dir.exists() and summary_json.exists()
    if templates_grid_pdf is not None:
        resume_ok = resume_ok and templates_grid_pdf.exists()

    if resume_ok:
        logger.info("Resuming templates: existing outputs found at %s", templates_out_dir)
        return TemplateExtractOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            extracted_templates_dir=extracted_templates_dir,
            merged_units_dir=(merged_units_dir if merged_units_dir.exists() else None),
            # Plots are now written directly under templates_outputs/* (no merged_unit_plots dir).
            merged_unit_plots_dir=None,
            summary_json=summary_json,
            templates_grid_pdf=templates_grid_pdf,
            multi_source_templates_dir=(unit_segment_grids_dir if unit_segment_grids_dir is not None and unit_segment_grids_dir.exists() else None),
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
    merged_units_dir.mkdir(parents=True, exist_ok=True)
    merged_unit_footprints_dir.mkdir(parents=True, exist_ok=True)
    merged_unit_footprints_zoomed_dir.mkdir(parents=True, exist_ok=True)
    merged_unit_svgs_dir.mkdir(parents=True, exist_ok=True)
    merged_unit_full_chip_maps_dir.mkdir(parents=True, exist_ok=True)
    axon_velocity_outputs_root_dir.mkdir(parents=True, exist_ok=True)
    if bool(inputs.save_full_channels_templates):
        full_channels_templates_dir.mkdir(parents=True, exist_ok=True)
    if bool(inputs.plot_topo_unit_footprints):
        topo_unit_footprints_dir.mkdir(parents=True, exist_ok=True)
    if bool(inputs.plot_propagation_plots):
        propagation_plots_dir.mkdir(parents=True, exist_ok=True)
    if unit_segment_grids_dir is not None:
        unit_segment_grids_dir.mkdir(parents=True, exist_ok=True)

    from .multi_source_utils import (
        _get_unit_template_from_extension,
        _load_waveforms_analyzers,
        _normalize_id_for_compare,
        _sparsity_unit_channel_indices,
        _try_get_electrode_ids,
    )

    analyzers = _load_waveforms_analyzers(
        well_out_dir=well_out_dir,
        include_concat=bool(inputs.include_concat),
        include_segments=bool(inputs.include_segments),
        logger=logger,
    )

    # Unit list:
    # - If the user passes `unit_ids`, we treat that list as the curated set.
    # - Otherwise, we require spikesorting-stage curation (default).
    curation_qm_xlsx: Optional[Path] = None
    curated_units_norm: Optional[list[Any]] = None
    if inputs.unit_ids is not None:
        unit_ids = list(inputs.unit_ids)
    else:
        # Start from concat (or first analyzer) as the universe, then strictly filter.
        unit_ids = list(analyzers[0][1].sorting.unit_ids)
        unit_ids, curated_units_norm, curation_qm_xlsx = _apply_spikesorting_stage_unit_curation(
            unit_ids=unit_ids,
            well_out_dir=well_out_dir,
            normalize_id_for_compare=_normalize_id_for_compare,
            logger=logger,
        )
        if bool(inputs.require_curated_units) and curated_units_norm is None:
            raise RuntimeError(
                "Templates stage is configured to require curated units from spikesorting metrics, "
                "but curated units could not be derived (expected <well>/spikesorting_outputs/qm_unfiltered.xlsx). "
                "Run spikesorting first, or pass TemplateExtractInputs(unit_ids=[...]) explicitly, "
                "or set require_curated_units=False."
            )

    if inputs.unit_limit is not None:
        unit_ids = unit_ids[: int(inputs.unit_limit)]

    fs_hz, ms_before, ms_after = _infer_template_plot_window(
        well_out_dir=well_out_dir,
        analyzers=analyzers,
        read_json=_read_json,
    )

    waveforms_out_dir = well_out_dir / "waveforms_outputs"

    summary: dict[str, Any] = {
        "h5_path": str(inputs.h5_path),
        "stream_id": inputs.stream_id,
        "well_out_dir": str(well_out_dir),
        "templates_out_dir": str(templates_out_dir),
        "sources": [name for name, _ in analyzers],
        "templates_grid_pdf": str(templates_grid_pdf) if templates_grid_pdf else None,
        "templates_grid_panels_dir": str(templates_grid_panels_dir) if templates_grid_panels_dir else None,
        "multi_source_templates_dir": str(unit_segment_grids_dir) if unit_segment_grids_dir else None,
        "full_channels_templates_dir": str(full_channels_templates_dir) if bool(inputs.save_full_channels_templates) else None,
        "topo_unit_footprints_dir": str(topo_unit_footprints_dir) if bool(inputs.plot_topo_unit_footprints) else None,
        "propagation_plots_dir": str(propagation_plots_dir) if bool(inputs.plot_propagation_plots) else None,
        "merged_unit_footprints_dir": str(merged_unit_footprints_dir),
        "merged_unit_footprints_zoomed_dir": str(merged_unit_footprints_zoomed_dir),
        "merged_unit_svgs_dir": str(merged_unit_svgs_dir),
        "merged_unit_full_chip_maps_dir": str(merged_unit_full_chip_maps_dir),
        "axon_velocity_outputs_root_dir": str(axon_velocity_outputs_root_dir),
        "curation": {
            "qm_unfiltered_xlsx": str(curation_qm_xlsx) if curation_qm_xlsx else None,
            "applied": bool(curated_units_norm is not None),
            "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
        },
        "waveforms_best_channel_sources_xlsx": (
            str(waveforms_out_dir / "best_channel_sources.xlsx")
            if (waveforms_out_dir / "best_channel_sources.xlsx").exists()
            else None
        ),
        "units": [],
    }

    unit_grid_entries = process_unit_list(
        unit_list=unit_ids,
        analyzers=analyzers,
        extracted_templates_dir=extracted_templates_dir,
        merged_units_dir=merged_units_dir,
        merged_unit_footprints_dir=merged_unit_footprints_dir,
        merged_unit_footprints_zoomed_dir=merged_unit_footprints_zoomed_dir,
        merged_unit_svgs_dir=merged_unit_svgs_dir,
        merged_unit_full_chip_maps_dir=merged_unit_full_chip_maps_dir,
        axon_velocity_outputs_root_dir=axon_velocity_outputs_root_dir,
        unit_segment_grids_dir=unit_segment_grids_dir,
        full_channels_templates_dir=(full_channels_templates_dir if bool(inputs.save_full_channels_templates) else None),
        topo_unit_footprints_dir=(topo_unit_footprints_dir if bool(inputs.plot_topo_unit_footprints) else None),
        propagation_plots_dir=(propagation_plots_dir if bool(inputs.plot_propagation_plots) else None),
        fs_hz=float(fs_hz),
        ms_before=ms_before,
        ms_after=ms_after,
        top_channels_per_template=int(inputs.top_channels_per_template),
        write_footprint_ptp_map=_write_footprint_ptp_map,
        write_unit_template_and_footprint_svg=_write_unit_template_and_footprint_svg,
        write_topo_unit_footprint_png=_write_topo_unit_footprint_png,
        make_merged_contributing_footprint_plots=bool(inputs.plot_merged_contributing_footprints_linear_and_log),
        make_axon_velocity_plots=bool(inputs.plot_axon_velocity_outputs),
        force_restart=bool(inputs.force_restart),
        n_jobs=int(inputs.n_jobs),
        get_template_from_extension=_get_unit_template_from_extension,
        sparsity_unit_channel_indices=_sparsity_unit_channel_indices,
        try_get_electrode_ids=_try_get_electrode_ids,
        jsonable=_jsonable,
        jsonable_sequence=_jsonable_sequence,
        write_json=_write_json,
        write_unit_segment_grids_pdf=_write_unit_segment_grids_pdf,
        write_unit_segment_footprint_grids_pdf=_write_unit_segment_footprint_grids_pdf,
        write_unit_propagation_plots_pdf=_write_unit_propagation_plots_pdf,
        propagation_top_channels=int(inputs.propagation_top_channels),
        propagation_n_waveforms=int(inputs.propagation_n_waveforms),
        propagation_channels_per_panel=int(inputs.propagation_channels_per_panel),
        propagation_channel_overlap=int(inputs.propagation_channel_overlap),
        propagation_show_electrode_ids=bool(inputs.propagation_show_electrode_ids),
        propagation_trace_gain=float(inputs.propagation_trace_gain),
        propagation_trace_spacing=float(inputs.propagation_trace_spacing),
        logger=logger,
        persist=True,
        summary=summary,
    )

    # Write grid PDFs.
    if templates_grid_pdf is not None:
        if (not templates_grid_pdf.exists()) or inputs.force_restart:
            _write_templates_grid_pdf(
                pdf_path=templates_grid_pdf,
                unit_entries=unit_grid_entries,
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
            "merged_units_dir": str(merged_units_dir),
            # Deprecated (kept for older checkpoint readers).
            "merged_unit_plots_dir": None,
            "templates_summary_json": str(summary_json),
            "templates_grid_pdf": str(templates_grid_pdf) if templates_grid_pdf else None,
        },
    )

    return TemplateExtractOutputs(
        well_out_dir=well_out_dir,
        templates_out_dir=templates_out_dir,
        extracted_templates_dir=extracted_templates_dir,
        merged_units_dir=merged_units_dir,
        # Plots are now written under templates_outputs/* (no dedicated merged_unit_plots dir).
        merged_unit_plots_dir=None,
        summary_json=summary_json,
        templates_grid_pdf=templates_grid_pdf,
        multi_source_templates_dir=unit_segment_grids_dir,
    )


__all__ = [
    "TemplateExtractInputs",
    "TemplateExtractOutputs",
    "extract_and_merge_templates",
]
