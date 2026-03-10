from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Optional

from .plotting import (
    _write_footprint_ptp_map,
    _write_zoomed_footprints_grid_pdf,
    _write_templates_grid_pdf,
    _write_unit_segment_grids_pdf,
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
)
from ..output_paths import compute_mea_analysis_output_dir
from ..pipeline_logging import build_stage_logger, log_stage_complete, log_stage_failure, log_stage_start
from ..checkpointing import save_stage_completed, save_stage_failed, save_stage_started


TEMPLATES_OUTPUTS_DIRNAME = "stg4_templates_outputs"


def _parse_unit_id_from_footprint_name(path: Path) -> Optional[int]:
    m = re.match(r"unit_(\d+)_", str(path.name))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _coerce_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _sort_footprint_images_by_spike_count(
    *,
    image_paths: list[Path],
    unit_spike_counts: dict[int, int],
) -> list[Path]:
    def _key(p: Path):
        uid = _parse_unit_id_from_footprint_name(p)
        count = unit_spike_counts.get(int(uid), -1) if uid is not None else -1
        uid_sort = int(uid) if uid is not None else 10**12
        return (-int(count), uid_sort, str(p.name))

    return sorted(image_paths, key=_key)


def _build_footprint_grid_entries_from_summary(summary_obj: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np  # type: ignore[import-not-found]

    out: list[dict[str, Any]] = []
    for u in list(summary_obj.get("units", []) or []):
        if not isinstance(u, dict):
            continue
        uid = _coerce_int_or_none(u.get("unit_id"))
        if uid is None:
            continue
        n_wf = _coerce_int_or_none(u.get("n_waveforms_sum"))

        merged_src = None
        for s in list(u.get("sources", []) or []):
            if isinstance(s, dict) and str(s.get("name")) == "merged_contributing":
                merged_src = s
                break
        if not isinstance(merged_src, dict):
            continue

        try:
            meta_json = merged_src.get("meta_json")
            if not meta_json:
                continue
            meta = _read_json(Path(str(meta_json)))
            if not isinstance(meta, dict):
                continue

            locs_path = meta.get("channel_locations_npy")
            fp_path = meta.get("footprint_ptp_npy")
            template_path = meta.get("template_npy")
            eids_path = meta.get("electrode_ids_npy")
            if not locs_path:
                continue

            locs = np.load(str(locs_path), allow_pickle=True)
            amp = None
            if fp_path:
                amp = np.load(str(fp_path), allow_pickle=True)
            elif template_path:
                tmpl = np.load(str(template_path), allow_pickle=True)
                amp = np.ptp(np.asarray(tmpl, dtype=float), axis=0)

            if amp is None:
                continue

            eids = None
            if eids_path:
                try:
                    eids = np.load(str(eids_path), allow_pickle=True)
                except Exception:
                    eids = None

            contrib_locs_xy = np.asarray(locs, dtype=float)[:, :2]
            contrib_keys = {
                (round(float(x), 3), round(float(y), 3))
                for x, y in contrib_locs_xy.tolist()
            }
            non_contrib_keys: set[tuple[float, float]] = set()

            for src in list(u.get("sources", []) or []):
                if not isinstance(src, dict):
                    continue
                if str(src.get("name")) == "merged_contributing":
                    continue
                try:
                    src_meta_json = src.get("meta_json")
                    if not src_meta_json:
                        continue
                    src_meta = _read_json(Path(str(src_meta_json)))
                    if not isinstance(src_meta, dict):
                        continue
                    src_locs = src_meta.get("channel_locations")
                    if src_locs is None:
                        continue
                    src_locs_xy = np.asarray(src_locs, dtype=float)
                    if src_locs_xy.ndim != 2 or int(src_locs_xy.shape[1]) < 2:
                        continue
                    for x, y in src_locs_xy[:, :2].tolist():
                        k = (round(float(x), 3), round(float(y), 3))
                        if k in contrib_keys:
                            continue
                        non_contrib_keys.add(k)
                except Exception:
                    continue

            non_contrib_locs_xy = [[float(x), float(y)] for (x, y) in sorted(non_contrib_keys)]

            out.append(
                {
                    "unit_id": int(uid),
                    "n_waveforms_sum": (int(n_wf) if n_wf is not None else None),
                    "channel_locations_xy": contrib_locs_xy,
                    "footprint_ptp": np.asarray(amp, dtype=float),
                    "electrode_ids": (eids.tolist() if eids is not None else None),
                    "non_contributing_locations_xy": non_contrib_locs_xy,
                }
            )
        except Exception:
            continue

    return out


@dataclass(frozen=True)
class TemplateExtractInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    include_concat: bool = True
    include_segments: bool = True

    # Optional input/output variant routing.
    # - If `waveforms_variant_name` is set (e.g. "merged4x4"), templates reads
    #   from `<well>/stg3_waveforms_outputs_<variant>`.
    # - If `templates_variant_name` is set, templates writes to
    #   `<well>/stg4_templates_outputs_<variant>` and uses a variant checkpoint.
    # - If `templates_variant_name` is omitted, it defaults to
    #   `waveforms_variant_name` when provided.
    waveforms_variant_name: Optional[str] = None
    templates_variant_name: Optional[str] = None

    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    # Templates should only run on curated units derived from spikesorting metrics.
    # If True and `unit_ids` is None, this requires `<well>/stg2_spikesorting_outputs/qm_unfiltered.xlsx`.
    # Note: for merged-variant runs (e.g. merged4x4), curation is auto-disabled
    # unless `require_curated_units` is explicitly set to True via stage kwargs.
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
    # If True, rewrite zoomed merged-contributing footprints with shared global
    # color limits across all processed units in this templates run.
    zoomed_footprints_global_color_scale: bool = False

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

    # Optional post-spikesort upsampling of persisted templates (time axis only).
    #
    # Scientific rationale:
    # - Spikesorting is performed at the native sampling rate.
    # - Templates (mean waveforms) can optionally be upsampled after spikesorting to
    #   improve peak-timing resolution for axon_velocity without re-running sorting.
    #
    # Contract:
    # - When enabled, the persisted template arrays and their metadata (`sampling_frequency_hz`, `n_samples`)
    #   reflect the *upsampled* timebase.
    # - Downstream reconstruction consumes that metadata and stays unchanged.
    template_time_upsample_factor: int = 1
    template_time_upsample_method: str = "sinc"

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

    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    logger = build_stage_logger(
        well_out_dir=well_out_dir,
        data_file=inputs.h5_path,
        stream_id=inputs.stream_id,
        stage_name="templates",
        logger_name_prefix=logger_name_prefix,
        verbose=True,
    )

    waveforms_variant = str(inputs.waveforms_variant_name).strip() if inputs.waveforms_variant_name else ""
    templates_variant = str(inputs.templates_variant_name).strip() if inputs.templates_variant_name else ""
    if not templates_variant:
        templates_variant = waveforms_variant

    waveforms_dirname = "stg3_waveforms_outputs" + (f"_{waveforms_variant}" if waveforms_variant else "")
    templates_outputs_dirname = TEMPLATES_OUTPUTS_DIRNAME + (f"_{templates_variant}" if templates_variant else "")
    templates_stage_name = "templates" + (f"_{templates_variant}" if templates_variant else "")

    templates_out_dir = well_out_dir / templates_outputs_dirname
    templates_dir = templates_out_dir / "templates"
    extracted_templates_dir = templates_dir / "sources"
    merged_units_dir = templates_dir / "merged"

    # Plot outputs
    footprints_root_dir = templates_out_dir / "footprints"
    merged_unit_footprints_dir = footprints_root_dir / "full"
    merged_unit_footprints_zoomed_dir = footprints_root_dir / "zoomed"
    merged_unit_full_chip_maps_dir = footprints_root_dir / "full_chip_maps"
    footprint_grids_dir = footprints_root_dir / "grids"
    merged_unit_footprints_zoomed_linear_grid_pdf = footprint_grids_dir / "merged_contributing_footprints_linear_zoom_grid.pdf"
    merged_unit_footprints_zoomed_log_grid_pdf = footprint_grids_dir / "merged_contributing_footprints_log_zoom_grid.pdf"
    merged_unit_footprints_zoomed_linear_grid_pages_dir = (
        footprint_grids_dir / "merged_contributing_footprints_linear_zoom_grid_pages"
    )
    merged_unit_footprints_zoomed_log_grid_pages_dir = (
        footprint_grids_dir / "merged_contributing_footprints_log_zoom_grid_pages"
    )

    # Optional per-unit axon_velocity plot bundle (only created when enabled).
    axon_velocity_outputs_root_dir = templates_out_dir / "axon_velocity_outputs"

    full_channels_templates_dir = templates_dir / "full"
    topo_unit_footprints_dir = footprints_root_dir / "3D"
    propagation_plots_dir = templates_out_dir / "propagation_plots"

    summary_json = templates_out_dir / "templates_summary.json"
    templates_grid_pdf = templates_out_dir / "templates_grid.pdf" if inputs.plot_templates_grid_pdf else None
    # Per-unit concat-vs-segment grids (flattened files).
    unit_segment_grids_dir = templates_out_dir / "segment_grids" if inputs.plot_multi_source_templates_pdf else None

    # Intentionally disabled.
    templates_grid_panels_dir = None

    ckpt_file = _compute_templates_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=inputs.h5_path,
        stream_id=inputs.stream_id,
        stage_name=templates_stage_name,
    )
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
        summary_existing = None
        footprint_grid_entries: list[dict[str, Any]] = []
        lin_vmin = None
        lin_vmax = None
        log_vmin = None
        log_vmax = None
        if summary_json.exists():
            try:
                summary_existing = _read_json(summary_json)
                if isinstance(summary_existing, dict):
                    footprint_grid_entries = _build_footprint_grid_entries_from_summary(summary_existing)
                    try:
                        zcs = ((summary_existing.get("footprints") or {}).get("zoomed_global_color_scale") or {})
                        lin_vmin = zcs.get("linear_vmin")
                        lin_vmax = zcs.get("linear_vmax")
                        log_vmin = zcs.get("log_vmin")
                        log_vmax = zcs.get("log_vmax")
                    except Exception:
                        pass
            except Exception:
                summary_existing = None

        footprint_grid_entries = sorted(
            footprint_grid_entries,
            key=lambda x: (
                -int(_coerce_int_or_none(x.get("n_waveforms_sum")) or -1),
                int(_coerce_int_or_none(x.get("unit_id")) or 10**12),
            ),
        )

        try:
            merged_unit_footprints_dir.mkdir(parents=True, exist_ok=True)
            merged_unit_footprints_zoomed_dir.mkdir(parents=True, exist_ok=True)
            for e in footprint_grid_entries:
                try:
                    uid = e.get("unit_id")
                    locs = e.get("channel_locations_xy")
                    amp = e.get("footprint_ptp")
                    eids = e.get("electrode_ids")
                    _write_footprint_ptp_map(
                        out_path=merged_unit_footprints_dir / f"unit_{uid}_merged_contributing_footprint_ptp_linear.png",
                        channel_locations_xy=locs,
                        footprint_ptp=amp,
                        title=f"Unit {uid} contributing-channels footprint (PTP)",
                        log_scale=False,
                        electrode_ids=eids,
                        all_recorded_electrode_ids=None,
                    )
                    _write_footprint_ptp_map(
                        out_path=merged_unit_footprints_dir / f"unit_{uid}_merged_contributing_footprint_ptp_log.png",
                        channel_locations_xy=locs,
                        footprint_ptp=amp,
                        title=f"Unit {uid} contributing-channels footprint (PTP, log)",
                        log_scale=True,
                        electrode_ids=eids,
                        all_recorded_electrode_ids=None,
                    )
                    _write_footprint_ptp_map(
                        out_path=merged_unit_footprints_zoomed_dir / f"unit_{uid}_merged_contributing_footprint_ptp_linear_zoom.png",
                        channel_locations_xy=locs,
                        footprint_ptp=amp,
                        title="",
                        log_scale=False,
                        electrode_ids=eids,
                        all_recorded_electrode_ids=None,
                        zoom=True,
                    )
                    _write_footprint_ptp_map(
                        out_path=merged_unit_footprints_zoomed_dir / f"unit_{uid}_merged_contributing_footprint_ptp_log_zoom.png",
                        channel_locations_xy=locs,
                        footprint_ptp=amp,
                        title="",
                        log_scale=True,
                        electrode_ids=eids,
                        all_recorded_electrode_ids=None,
                        zoom=True,
                    )
                except Exception:
                    continue

            footprint_grids_dir.mkdir(parents=True, exist_ok=True)
            if footprint_grid_entries:
                _write_zoomed_footprints_grid_pdf(
                    pdf_path=merged_unit_footprints_zoomed_linear_grid_pdf,
                    entries=footprint_grid_entries,
                    title="Merged contributing footprints (linear, zoomed)",
                    log_scale=False,
                    fixed_vmin=(float(lin_vmin) if lin_vmin is not None else None),
                    fixed_vmax=(float(lin_vmax) if lin_vmax is not None else None),
                    write_page_pngs=True,
                    page_pngs_dir=merged_unit_footprints_zoomed_linear_grid_pages_dir,
                )
                _write_zoomed_footprints_grid_pdf(
                    pdf_path=merged_unit_footprints_zoomed_log_grid_pdf,
                    entries=footprint_grid_entries,
                    title="Merged contributing footprints (log, zoomed)",
                    log_scale=True,
                    fixed_vmin=(float(log_vmin) if log_vmin is not None else None),
                    fixed_vmax=(float(log_vmax) if log_vmax is not None else None),
                    write_page_pngs=True,
                    page_pngs_dir=merged_unit_footprints_zoomed_log_grid_pages_dir,
                )
        except Exception:
            pass

        if isinstance(summary_existing, dict):
            try:
                summary_existing["merged_unit_footprints_zoomed_linear_grid_pdf"] = (
                    str(merged_unit_footprints_zoomed_linear_grid_pdf)
                    if merged_unit_footprints_zoomed_linear_grid_pdf.exists()
                    else None
                )
                summary_existing["merged_unit_footprints_zoomed_log_grid_pdf"] = (
                    str(merged_unit_footprints_zoomed_log_grid_pdf)
                    if merged_unit_footprints_zoomed_log_grid_pdf.exists()
                    else None
                )
                summary_existing["merged_unit_footprints_zoomed_linear_grid_pages_dir"] = (
                    str(merged_unit_footprints_zoomed_linear_grid_pages_dir)
                    if merged_unit_footprints_zoomed_linear_grid_pages_dir.exists()
                    else None
                )
                summary_existing["merged_unit_footprints_zoomed_log_grid_pages_dir"] = (
                    str(merged_unit_footprints_zoomed_log_grid_pages_dir)
                    if merged_unit_footprints_zoomed_log_grid_pages_dir.exists()
                    else None
                )
                summary_existing["footprint_grid_sort"] = "n_waveforms_sum_desc"
                _write_json(summary_json, summary_existing)
            except Exception:
                pass

        logger.info("Resuming templates: existing outputs found at %s", templates_out_dir)
        return TemplateExtractOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            extracted_templates_dir=extracted_templates_dir,
            merged_units_dir=(merged_units_dir if merged_units_dir.exists() else None),
            # Plots are now written directly under stg4_templates_outputs/* (no merged_unit_plots dir).
            merged_unit_plots_dir=None,
            summary_json=summary_json,
            templates_grid_pdf=templates_grid_pdf,
            multi_source_templates_dir=(unit_segment_grids_dir if unit_segment_grids_dir is not None and unit_segment_grids_dir.exists() else None),
        )

    ckpt = save_stage_started(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        extra_fields={"templates_out_dir": str(templates_out_dir)},
    )
    log_stage_start(logger=logger, stage="templates", checkpoint_file=ckpt_file)

    try:
        templates_out_dir.mkdir(parents=True, exist_ok=True)
        templates_dir.mkdir(parents=True, exist_ok=True)
        extracted_templates_dir.mkdir(parents=True, exist_ok=True)
        merged_units_dir.mkdir(parents=True, exist_ok=True)
        footprints_root_dir.mkdir(parents=True, exist_ok=True)
        merged_unit_footprints_dir.mkdir(parents=True, exist_ok=True)
        merged_unit_footprints_zoomed_dir.mkdir(parents=True, exist_ok=True)
        merged_unit_full_chip_maps_dir.mkdir(parents=True, exist_ok=True)
        if bool(inputs.save_full_channels_templates):
            full_channels_templates_dir.mkdir(parents=True, exist_ok=True)
        if bool(inputs.plot_topo_unit_footprints):
            topo_unit_footprints_dir.mkdir(parents=True, exist_ok=True)
        if bool(inputs.plot_propagation_plots):
            propagation_plots_dir.mkdir(parents=True, exist_ok=True)
        if unit_segment_grids_dir is not None:
            unit_segment_grids_dir.mkdir(parents=True, exist_ok=True)

        if bool(inputs.plot_axon_velocity_outputs):
            axon_velocity_outputs_root_dir.mkdir(parents=True, exist_ok=True)

        from .multi_source_utils import (
            _get_unit_template_from_extension,
            _load_waveforms_analyzers,
            _normalize_id_for_compare,
            _sparsity_unit_channel_indices,
            _try_get_electrode_ids,
        )

        analyzers = _load_waveforms_analyzers(
            well_out_dir=well_out_dir,
            waveforms_dirname=waveforms_dirname,
            include_concat=bool(inputs.include_concat),
            include_segments=bool(inputs.include_segments),
            logger=logger,
        )
    except Exception as e:
        save_stage_failed(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage="TEMPLATES",
            error=exception_to_error_dict(e),
            extra_fields={"templates_out_dir": str(templates_out_dir)},
        )
        log_stage_failure(
            logger=logger,
            stage="templates",
            checkpoint_file=ckpt_file,
            error=e,
        )
        raise

    # Unit list:
    # - If the user passes `unit_ids`, we treat that list as the curated set.
    # - Otherwise, we require spikesorting-stage curation (default).
    curation_qm_xlsx: Optional[Path] = None
    curated_units_norm: Optional[list[Any]] = None
    merged_variant_mode = (
        bool(waveforms_variant) and ("merged" in str(waveforms_variant).lower())
    ) or (
        bool(templates_variant) and ("merged" in str(templates_variant).lower())
    )
    require_curated_units_effective = bool(inputs.require_curated_units)
    if merged_variant_mode and bool(inputs.require_curated_units):
        require_curated_units_effective = False
        logger.info(
            "Merged-variant templates run detected; disabling spikesorting curation and processing all units"
        )

    if inputs.unit_ids is not None:
        unit_ids = list(inputs.unit_ids)
    else:
        # Start from concat (or first analyzer) as the universe, then strictly filter.
        unit_ids = list(analyzers[0][1].sorting.unit_ids)
        if require_curated_units_effective:
            unit_ids, curated_units_norm, curation_qm_xlsx = _apply_spikesorting_stage_unit_curation(
                unit_ids=unit_ids,
                well_out_dir=well_out_dir,
                normalize_id_for_compare=_normalize_id_for_compare,
                logger=logger,
            )
        if bool(require_curated_units_effective) and curated_units_norm is None:
            raise RuntimeError(
                "Templates stage is configured to require curated units from spikesorting metrics, "
                "but curated units could not be derived (expected <well>/stg2_spikesorting_outputs/qm_unfiltered.xlsx). "
                "Run spikesorting first, or pass TemplateExtractInputs(unit_ids=[...]) explicitly, "
                "or set require_curated_units=False."
            )

    if inputs.unit_limit is not None:
        unit_ids = unit_ids[: int(inputs.unit_limit)]

    fs_hz, ms_before, ms_after = _infer_template_plot_window(
        well_out_dir=well_out_dir,
        waveforms_out_dir=(well_out_dir / waveforms_dirname),
        analyzers=analyzers,
        read_json=_read_json,
    )

    time_upsample_factor = int(getattr(inputs, "template_time_upsample_factor", 1) or 1)
    time_upsample_method = str(getattr(inputs, "template_time_upsample_method", "sinc") or "sinc")
    fs_hz_effective = float(fs_hz) * float(time_upsample_factor) if time_upsample_factor > 1 else float(fs_hz)

    waveforms_out_dir = well_out_dir / waveforms_dirname

    summary: dict[str, Any] = {
        "h5_path": str(inputs.h5_path),
        "stream_id": inputs.stream_id,
        "well_out_dir": str(well_out_dir),
        "waveforms_variant_name": (waveforms_variant if waveforms_variant else None),
        "templates_variant_name": (templates_variant if templates_variant else None),
        "waveforms_out_dir": str(waveforms_out_dir),
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
        "merged_unit_footprints_zoomed_linear_grid_pdf": None,
        "merged_unit_footprints_zoomed_log_grid_pdf": None,
        "merged_unit_footprints_zoomed_linear_grid_pages_dir": None,
        "merged_unit_footprints_zoomed_log_grid_pages_dir": None,
        "merged_unit_full_chip_maps_dir": str(merged_unit_full_chip_maps_dir),
        "axon_velocity_outputs_root_dir": str(axon_velocity_outputs_root_dir) if bool(inputs.plot_axon_velocity_outputs) else None,
        "curation": {
            "qm_unfiltered_xlsx": str(curation_qm_xlsx) if curation_qm_xlsx else None,
            "applied": bool(curated_units_norm is not None),
            "require_curated_units_effective": bool(require_curated_units_effective),
            "n_curated_units": int(len(curated_units_norm)) if curated_units_norm is not None else None,
        },
        "waveforms_best_channel_sources_xlsx": (
            str(waveforms_out_dir / "best_channel_sources.xlsx")
            if (waveforms_out_dir / "best_channel_sources.xlsx").exists()
            else None
        ),
        "units": [],
        "template_time_upsampling": {
            "enabled": bool(time_upsample_factor > 1),
            "factor": int(time_upsample_factor),
            "method": str(time_upsample_method),
            "native_sampling_frequency_hz": float(fs_hz),
            "effective_sampling_frequency_hz": float(fs_hz_effective),
        },
    }

    unit_grid_entries = process_unit_list(
        unit_list=unit_ids,
        analyzers=analyzers,
        extracted_templates_dir=extracted_templates_dir,
        merged_units_dir=merged_units_dir,
        merged_unit_footprints_dir=merged_unit_footprints_dir,
        merged_unit_footprints_zoomed_dir=merged_unit_footprints_zoomed_dir,
        merged_unit_full_chip_maps_dir=merged_unit_full_chip_maps_dir,
        axon_velocity_outputs_root_dir=(axon_velocity_outputs_root_dir if bool(inputs.plot_axon_velocity_outputs) else None),
        unit_segment_grids_dir=unit_segment_grids_dir,
        full_channels_templates_dir=(full_channels_templates_dir if bool(inputs.save_full_channels_templates) else None),
        topo_unit_footprints_dir=(topo_unit_footprints_dir if bool(inputs.plot_topo_unit_footprints) else None),
        propagation_plots_dir=(propagation_plots_dir if bool(inputs.plot_propagation_plots) else None),
        fs_hz=float(fs_hz),
        template_time_upsample_factor=int(time_upsample_factor),
        template_time_upsample_method=str(time_upsample_method),
        ms_before=ms_before,
        ms_after=ms_after,
        top_channels_per_template=int(inputs.top_channels_per_template),
        write_footprint_ptp_map=_write_footprint_ptp_map,
        write_topo_unit_footprint_png=_write_topo_unit_footprint_png,
        make_merged_contributing_footprint_plots=bool(inputs.plot_merged_contributing_footprints_linear_and_log),
        zoomed_footprints_global_color_scale=bool(inputs.zoomed_footprints_global_color_scale),
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

    unit_grid_entries = sorted(
        unit_grid_entries,
        key=lambda x: (
            -int(_coerce_int_or_none(x.get("n_waveforms_sum")) or -1),
            int(_coerce_int_or_none(x.get("unit_id")) or 10**12),
        ),
    )

    footprint_grid_entries = _build_footprint_grid_entries_from_summary(summary)
    footprint_grid_entries = sorted(
        footprint_grid_entries,
        key=lambda x: (
            -int(_coerce_int_or_none(x.get("n_waveforms_sum")) or -1),
            int(_coerce_int_or_none(x.get("unit_id")) or 10**12),
        ),
    )

    # Write grid PDFs.
    if templates_grid_pdf is not None:
        if (not templates_grid_pdf.exists()) or inputs.force_restart:
            _write_templates_grid_pdf(
                pdf_path=templates_grid_pdf,
                unit_entries=unit_grid_entries,
                fs_hz=float(fs_hz_effective),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=int(inputs.top_channels_per_template),
                logger=logger,
            )

    try:
        footprint_grids_dir.mkdir(parents=True, exist_ok=True)
        zcs = ((summary.get("footprints") or {}).get("zoomed_global_color_scale") or {})
        lin_vmin = zcs.get("linear_vmin")
        lin_vmax = zcs.get("linear_vmax")
        log_vmin = zcs.get("log_vmin")
        log_vmax = zcs.get("log_vmax")

        if footprint_grid_entries:
            _write_zoomed_footprints_grid_pdf(
                pdf_path=merged_unit_footprints_zoomed_linear_grid_pdf,
                entries=footprint_grid_entries,
                title="Merged contributing footprints (linear, zoomed)",
                log_scale=False,
                fixed_vmin=(float(lin_vmin) if lin_vmin is not None else None),
                fixed_vmax=(float(lin_vmax) if lin_vmax is not None else None),
                write_page_pngs=True,
                page_pngs_dir=merged_unit_footprints_zoomed_linear_grid_pages_dir,
            )
            _write_zoomed_footprints_grid_pdf(
                pdf_path=merged_unit_footprints_zoomed_log_grid_pdf,
                entries=footprint_grid_entries,
                title="Merged contributing footprints (log, zoomed)",
                log_scale=True,
                fixed_vmin=(float(log_vmin) if log_vmin is not None else None),
                fixed_vmax=(float(log_vmax) if log_vmax is not None else None),
                write_page_pngs=True,
                page_pngs_dir=merged_unit_footprints_zoomed_log_grid_pages_dir,
            )
    except Exception:
        pass

    summary["merged_unit_footprints_zoomed_linear_grid_pdf"] = (
        str(merged_unit_footprints_zoomed_linear_grid_pdf)
        if merged_unit_footprints_zoomed_linear_grid_pdf.exists()
        else None
    )
    summary["merged_unit_footprints_zoomed_log_grid_pdf"] = (
        str(merged_unit_footprints_zoomed_log_grid_pdf)
        if merged_unit_footprints_zoomed_log_grid_pdf.exists()
        else None
    )
    summary["merged_unit_footprints_zoomed_linear_grid_pages_dir"] = (
        str(merged_unit_footprints_zoomed_linear_grid_pages_dir)
        if merged_unit_footprints_zoomed_linear_grid_pages_dir.exists()
        else None
    )
    summary["merged_unit_footprints_zoomed_log_grid_pages_dir"] = (
        str(merged_unit_footprints_zoomed_log_grid_pages_dir)
        if merged_unit_footprints_zoomed_log_grid_pages_dir.exists()
        else None
    )
    summary["footprint_grid_sort"] = "n_waveforms_sum_desc"

    _write_json(summary_json, summary)

    ckpt = save_stage_completed(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER_COMPLETE,
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
    log_stage_complete(
        logger=logger,
        stage="templates",
        checkpoint_file=ckpt_file,
        extra={"templates_summary_json": summary_json},
    )

    return TemplateExtractOutputs(
        well_out_dir=well_out_dir,
        templates_out_dir=templates_out_dir,
        extracted_templates_dir=extracted_templates_dir,
        merged_units_dir=merged_units_dir,
        # Plots are now written under stg4_templates_outputs/* (no dedicated merged_unit_plots dir).
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
