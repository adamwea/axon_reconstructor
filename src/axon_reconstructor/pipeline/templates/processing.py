from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .extraction import _choose_grid_source_for_unit, _gather_template_sources_for_unit, _persist_unit_templates
from .utils import _build_merged_contributing_template_for_unit


def _format_overlap_resolved_line(*, merged_units_dir: Path, unit_id: Any, max_show: int = 12) -> str:
    """Format a short overlap-resolution summary from persisted merged template metadata.

    Returned string is intended to be appended as a single extra line in plot info blocks.
    """

    meta_path = Path(merged_units_dir) / f"unit_{unit_id}" / "merged_contributing_template_meta.json"
    if not meta_path.exists():
        return "overlap-resolved: (meta missing)"

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return "overlap-resolved: (meta unreadable)"

    if not isinstance(meta, dict):
        return "overlap-resolved: (meta invalid)"

    overlap = meta.get("overlap")
    if not isinstance(overlap, dict):
        return "overlap-resolved: none"

    chans = overlap.get("channels")
    if not isinstance(chans, list) or (not chans):
        return "overlap-resolved: none"

    entries: list[str] = []
    for ch in chans:
        if not isinstance(ch, dict):
            continue

        key = ch.get("key")
        idx = ch.get("channel_index")

        kind = None
        val = None
        try:
            if isinstance(key, (list, tuple)) and len(key) >= 2:
                kind = key[0]
                val = key[1]
        except Exception:
            kind = None
            val = None

        if kind == "electrode" and val is not None:
            s = f"e{val}"
            if idx is not None:
                s += f"(idx{idx})"
        elif kind == "channel" and val is not None:
            s = f"{val}"
            if idx is not None:
                s += f"(idx{idx})"
        else:
            s = f"idx{idx}" if idx is not None else "(unknown)"

        entries.append(str(s))

    # De-dup while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for s in entries:
        if s not in seen:
            uniq.append(s)
            seen.add(s)

    if not uniq:
        return "overlap-resolved: none"

    shown = uniq[: int(max_show)]
    more = len(uniq) - len(shown)
    suffix = f" +{more} more" if more > 0 else ""
    return "overlap-resolved: " + ", ".join(shown) + suffix


def _infer_template_plot_window(*, well_out_dir: Path, analyzers: list[tuple[str, Any]], read_json) -> tuple[float, Optional[float], Optional[float]]:
    """Infer sampling frequency + ms window for template plotting.

    Prefers reading waveforms-stage params JSON when available.
    """

    ms_before: Optional[float] = None
    ms_after: Optional[float] = None

    try:
        fs_hz = float(analyzers[0][1].recording.get_sampling_frequency())
    except Exception:
        fs_hz = 10_000.0

    wf_params_json = well_out_dir / "waveforms_outputs" / "waveform_extraction_params.json"
    if wf_params_json.exists():
        try:
            params = read_json(wf_params_json)
            ms_before = float(params.get("ms_before")) if params.get("ms_before") is not None else None
            ms_after = float(params.get("ms_after")) if params.get("ms_after") is not None else None
        except Exception:
            pass

    return float(fs_hz), ms_before, ms_after


def _apply_spikesorting_stage_unit_curation(
    *,
    unit_ids: list[Any],
    well_out_dir: Path,
    normalize_id_for_compare,
    logger,
) -> tuple[list[Any], Optional[list[Any]], Optional[Path]]:
    """Filter unit_ids using spikesorting-stage quality metrics (MEA_Analysis).

    Waveforms stage no longer computes/owns quality metrics; the authoritative metrics
    live at `<well>/spikesorting_outputs/qm_unfiltered.xlsx`.

    This helper applies the same curation logic used elsewhere in this repo
    (`waveforms.curation.apply_mea_analysis_curation`) and then filters `unit_ids`
    to the curated set.
    """

    qm_xlsx = well_out_dir / "spikesorting_outputs" / "qm_unfiltered.xlsx"
    if not qm_xlsx.exists():
        return unit_ids, None, None

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        logger.warning("Found %s but pandas is unavailable; cannot apply curation", qm_xlsx)
        return unit_ids, None, qm_xlsx

    try:
        qm = pd.read_excel(qm_xlsx, index_col=0)
    except Exception as e:
        logger.warning("Failed reading %s: %s", qm_xlsx, e)
        return unit_ids, None, qm_xlsx

    try:
        if (getattr(qm, "index", None) is not None) and (str(qm.index.name) != "unit_id"):
            if "unit_id" in getattr(qm, "columns", []):
                qm = qm.set_index("unit_id", drop=True)
    except Exception:
        pass

    try:
        from ..waveforms.curation import apply_mea_analysis_curation

        clean_metrics, _rej = apply_mea_analysis_curation(q_metrics=qm, user_thresholds=None)
        curated_units_norm = [normalize_id_for_compare(x) for x in list(clean_metrics.index.values)]
    except Exception as e:
        logger.warning("Failed applying curation logic to %s: %s", qm_xlsx, e)
        return unit_ids, None, qm_xlsx

    curated_set = set(curated_units_norm)
    before = len(unit_ids)
    unit_ids = [uid for uid in unit_ids if normalize_id_for_compare(uid) in curated_set]
    logger.info(
        "Applying spikesorting curation: %d -> %d units (from %s)",
        before,
        len(unit_ids),
        qm_xlsx,
    )

    return unit_ids, curated_units_norm, qm_xlsx


def process_unit_list(
    *,
    unit_list: list[Any],
    analyzers: list[tuple[str, Any]],
    extracted_templates_dir: Path,
    merged_units_dir: Path,
    merged_unit_footprints_dir: Path,
    merged_unit_footprints_zoomed_dir: Optional[Path] = None,
    merged_unit_svgs_dir: Path,
    merged_unit_full_chip_maps_dir: Path,
    axon_velocity_outputs_root_dir: Path,
    unit_segment_grids_dir: Optional[Path],
    full_channels_templates_dir: Optional[Path] = None,
    topo_unit_footprints_dir: Optional[Path] = None,
    propagation_plots_dir: Optional[Path] = None,
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels_per_template: int,
    write_footprint_ptp_map=None,
    write_unit_template_and_footprint_svg=None,
    write_topo_unit_footprint_png=None,
    make_merged_contributing_footprint_plots: bool = True,
    make_axon_velocity_plots: bool = False,
    force_restart: bool,
    n_jobs: int,
    get_template_from_extension,
    sparsity_unit_channel_indices,
    try_get_electrode_ids,
    jsonable,
    jsonable_sequence,
    write_json,
    write_unit_segment_grids_pdf,
    write_unit_segment_footprint_grids_pdf=None,
    write_unit_propagation_plots_pdf=None,
    propagation_top_channels: int = 10,
    propagation_n_waveforms: int = 12,
    propagation_channels_per_panel: int = 25,
    propagation_channel_overlap: int = 5,
    logger,
    persist: bool,
    summary: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Process units: gather per-source templates, build merged_contributing, persist, and plot."""

    # Electrode ids present in at least one recording object (across sources).
    # Note: waveforms-stage analyzers may be sparse. When we detect Maxwell electrode id
    # scheme, we can still generate a *full-chip geometry* for full templates, but we
    # must NOT promote the observed recording electrode universe to the full chip.
    #
    # Rationale: downstream plots distinguish
    # - quiet electrodes: not present in any recording/analyzer
    # - non-contributing electrodes: present in at least one recording but not contributing for this unit
    #
    # That distinction requires preserving the observed recording electrode set.
    all_recorded_electrode_ids: Optional[list[Any]] = None
    try:
        all_electrode_ids: list[Any] = []
        for _, an in analyzers:
            try:
                el_ids = try_get_electrode_ids(an.recording)
            except Exception:
                el_ids = None
            if el_ids:
                all_electrode_ids.extend(list(el_ids))
        all_recorded_electrode_ids = all_electrode_ids if all_electrode_ids else None
    except Exception:
        all_recorded_electrode_ids = None

    # Note: do NOT replace all_recorded_electrode_ids with the full chip here.
    # Full-chip geometry is handled separately below when building full_channels_templates.

    # Ensure templates extension exists (best effort).
    for _, an in analyzers:
        try:
            if not an.has_extension("templates"):
                an.compute(["templates"], verbose=False, n_jobs=max(1, int(n_jobs)))
        except Exception:
            pass

    # Choose a reference analyzer for full-channel metadata (prefer concat).
    reference_analyzer = analyzers[0][1] if analyzers else None
    for name, an in analyzers:
        if str(name) == "concat":
            reference_analyzer = an
            break

    full_channel_locations_xy = None
    full_channel_ids = None
    full_electrode_ids = None
    if full_channels_templates_dir is not None:
        # Prefer a deterministic full-chip geometry when we can detect Maxwell electrode ids.
        # This avoids accidentally building a "full" template over only the sparse waveforms channels.
        try:
            from .utils import _looks_like_maxwell_full_chip_electrode_ids, _maxwell_full_chip_channel_metadata

            if _looks_like_maxwell_full_chip_electrode_ids(all_recorded_electrode_ids):
                locs_xy, ch_ids, el_ids = _maxwell_full_chip_channel_metadata()
                full_channel_locations_xy = locs_xy
                full_channel_ids = list(ch_ids.tolist())
                full_electrode_ids = list(el_ids.tolist())
        except Exception:
            pass

        # Fallback: use the reference analyzer's recording geometry.
        if full_channel_locations_xy is None and reference_analyzer is not None:
            try:
                import numpy as np  # type: ignore[import-not-found]

                full_channel_locations_xy = np.asarray(reference_analyzer.recording.get_channel_locations(), dtype=float)[:, :2]
                try:
                    full_channel_ids = list(reference_analyzer.recording.get_channel_ids())
                except Exception:
                    full_channel_ids = None
                try:
                    full_electrode_ids = try_get_electrode_ids(reference_analyzer.recording)
                except Exception:
                    full_electrode_ids = None
            except Exception:
                full_channel_locations_xy = None
                full_channel_ids = None
                full_electrode_ids = None

    grid_entries: list[dict[str, Any]] = []

    for uid in unit_list:
        sources_for_unit = _gather_template_sources_for_unit(
            uid=uid,
            analyzers=analyzers,
            get_template_from_extension=get_template_from_extension,
            sparsity_unit_channel_indices=sparsity_unit_channel_indices,
            try_get_electrode_ids=try_get_electrode_ids,
        )
        if not sources_for_unit:
            continue

        merged_contributing = _build_merged_contributing_template_for_unit(
            sources_for_unit=sources_for_unit,
            unit_id=uid,
            logger=logger,
        )
        sources_for_unit_with_merged = list(sources_for_unit) + (
            [merged_contributing] if merged_contributing is not None else []
        )

        # Best-effort merged_contributing quick-look plots.
        if (
            bool(make_merged_contributing_footprint_plots)
            and merged_contributing is not None
            and write_footprint_ptp_map is not None
            and write_unit_template_and_footprint_svg is not None
        ):
            try:
                import numpy as np  # type: ignore[import-not-found]

                merged_contributing_electrode_ids = merged_contributing.get("electrode_ids")

                merged_unit_footprints_dir.mkdir(parents=True, exist_ok=True)
                merged_unit_svgs_dir.mkdir(parents=True, exist_ok=True)

                tmpl = np.asarray(merged_contributing["template"], dtype=float)
                locs = np.asarray(merged_contributing["channel_locations"], dtype=float)
                amp = np.ptp(tmpl, axis=0)

                write_footprint_ptp_map(
                    out_path=merged_unit_footprints_dir / f"unit_{uid}_merged_contributing_footprint_ptp_linear.png",
                    channel_locations_xy=locs[:, :2],
                    footprint_ptp=amp,
                    title=f"Unit {uid} contributing-channels footprint (PTP)",
                    log_scale=False,
                    electrode_ids=merged_contributing_electrode_ids,
                    all_recorded_electrode_ids=all_recorded_electrode_ids,
                )
                write_footprint_ptp_map(
                    out_path=merged_unit_footprints_dir / f"unit_{uid}_merged_contributing_footprint_ptp_log.png",
                    channel_locations_xy=locs[:, :2],
                    footprint_ptp=amp,
                    title=f"Unit {uid} contributing-channels footprint (PTP, log)",
                    log_scale=True,
                    electrode_ids=merged_contributing_electrode_ids,
                    all_recorded_electrode_ids=all_recorded_electrode_ids,
                )

                # Zoomed-in versions (no title), for summary grids/presentations.
                if merged_unit_footprints_zoomed_dir is not None:
                    merged_unit_footprints_zoomed_dir.mkdir(parents=True, exist_ok=True)
                    write_footprint_ptp_map(
                        out_path=merged_unit_footprints_zoomed_dir
                        / f"unit_{uid}_merged_contributing_footprint_ptp_linear_zoom.png",
                        channel_locations_xy=locs[:, :2],
                        footprint_ptp=amp,
                        title="",
                        log_scale=False,
                        electrode_ids=merged_contributing_electrode_ids,
                        all_recorded_electrode_ids=all_recorded_electrode_ids,
                        zoom=True,
                    )
                    write_footprint_ptp_map(
                        out_path=merged_unit_footprints_zoomed_dir
                        / f"unit_{uid}_merged_contributing_footprint_ptp_log_zoom.png",
                        channel_locations_xy=locs[:, :2],
                        footprint_ptp=amp,
                        title="",
                        log_scale=True,
                        electrode_ids=merged_contributing_electrode_ids,
                        all_recorded_electrode_ids=all_recorded_electrode_ids,
                        zoom=True,
                    )

                write_unit_template_and_footprint_svg(
                    out_path=merged_unit_svgs_dir / f"unit_{uid}_merged_contributing_template_footprint_linear.svg",
                    unit_id=uid,
                    template=tmpl,
                    channel_locations_xy=locs[:, :2],
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                    top_channels=int(top_channels_per_template),
                    log_footprint=False,
                    electrode_ids=merged_contributing_electrode_ids,
                    all_recorded_electrode_ids=all_recorded_electrode_ids,
                )
                write_unit_template_and_footprint_svg(
                    out_path=merged_unit_svgs_dir / f"unit_{uid}_merged_contributing_template_footprint_log.svg",
                    unit_id=uid,
                    template=tmpl,
                    channel_locations_xy=locs[:, :2],
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                    top_channels=int(top_channels_per_template),
                    log_footprint=True,
                    electrode_ids=merged_contributing_electrode_ids,
                    all_recorded_electrode_ids=all_recorded_electrode_ids,
                )
            except Exception:
                pass

        if persist:
            unit_entry = _persist_unit_templates(
                uid=uid,
                sources_for_unit_with_merged=sources_for_unit_with_merged,
                extracted_templates_dir=extracted_templates_dir,
                merged_units_dir=merged_units_dir,
                merged_unit_full_chip_maps_dir=merged_unit_full_chip_maps_dir,
                axon_velocity_outputs_root_dir=axon_velocity_outputs_root_dir,
                full_channels_templates_dir=full_channels_templates_dir,
                full_channel_locations_xy=full_channel_locations_xy,
                full_channel_ids=full_channel_ids,
                full_electrode_ids=full_electrode_ids,
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
                make_axon_velocity_plots=bool(make_axon_velocity_plots),
                jsonable=jsonable,
                jsonable_sequence=jsonable_sequence,
                write_json=write_json,
                force_restart=bool(force_restart),
                logger=logger,
            )
            if summary is not None:
                summary.setdefault("units", []).append(unit_entry)

        # Topographical footprint plot from full-channel template (per-unit).
        if topo_unit_footprints_dir is not None and write_topo_unit_footprint_png is not None and full_channels_templates_dir is not None:
            try:
                out_png = topo_unit_footprints_dir / f"unit_{uid}.png"
                if (not out_png.exists()) or force_restart:
                    unit_full_dir = full_channels_templates_dir / f"unit_{uid}"
                    full_template_npy = unit_full_dir / "full_template.npy"
                    full_electrode_ids_npy = unit_full_dir / "full_electrode_ids.npy"

                    if full_template_npy.exists():
                        import numpy as np  # type: ignore[import-not-found]

                        full_tmpl = np.load(full_template_npy)
                        try:
                            full_eids = (
                                np.load(full_electrode_ids_npy, allow_pickle=True)
                                if full_electrode_ids_npy.exists()
                                else None
                            )
                        except Exception:
                            full_eids = None

                        write_topo_unit_footprint_png(
                            out_path=out_png,
                            unit_id=uid,
                            full_template=full_tmpl,
                            full_electrode_ids=(full_eids.tolist() if full_eids is not None else None),
                            all_recorded_electrode_ids=all_recorded_electrode_ids,
                            title=f"Unit {uid} full-template topo footprint (PTP)",
                            overlap_resolved_line=_format_overlap_resolved_line(
                                merged_units_dir=merged_units_dir,
                                unit_id=uid,
                            ),
                        )
            except Exception:
                pass

        # Templates grid should show the merged contributing-channels template.
        chosen = merged_contributing
        if chosen is not None:
            grid_entries.append(
                {
                    "unit_id": uid,
                    "template": chosen["template"],
                    "channel_locations": chosen.get("channel_locations"),
                    "n_channels": int(chosen["template"].shape[1]) if chosen.get("template") is not None else None,
                    # Best-effort counts for waveforms contributing (computed in sources, if available).
                    "n_waveforms_sum": chosen.get("n_waveforms_sum"),
                    "n_contributing_channels": chosen.get("n_contributing_channels"),
                }
            )

        if unit_segment_grids_dir is not None:
            unit_segment_grids_dir.mkdir(parents=True, exist_ok=True)
            unit_templates_pdf = unit_segment_grids_dir / f"unit_{uid}_templates.pdf"
            if (not unit_templates_pdf.exists()) or force_restart:
                write_unit_segment_grids_pdf(
                    pdf_path=unit_templates_pdf,
                    unit_id=uid,
                    sources_for_unit=sources_for_unit,
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                )

            if write_unit_segment_footprint_grids_pdf is not None:
                unit_footprints_pdf = unit_segment_grids_dir / f"unit_{uid}_footprints.pdf"
                if (not unit_footprints_pdf.exists()) or force_restart:
                    write_unit_segment_footprint_grids_pdf(
                        pdf_path=unit_footprints_pdf,
                        unit_id=uid,
                        sources_for_unit=sources_for_unit,
                        all_recorded_electrode_ids=all_recorded_electrode_ids,
                    )

        # Propagation plots (best-effort).
        # Prefer PNG outputs (faster iteration than PDFs).
        if propagation_plots_dir is not None and write_unit_propagation_plots_pdf is not None and merged_contributing is not None:
            try:
                out_png = propagation_plots_dir / f"unit_{uid}.png"
                if (not out_png.exists()) or force_restart:
                    # `write_unit_propagation_plots_pdf` is a legacy name; current plotting writes PNG(s)
                    # into the provided directory.
                    write_unit_propagation_plots_pdf(
                        pdf_path=(propagation_plots_dir / f"unit_{uid}.pdf"),
                        unit_id=uid,
                        merged_contributing=merged_contributing,
                        fs_hz=float(fs_hz),
                        ms_before=ms_before,
                        ms_after=ms_after,
                        top_channels=int(propagation_top_channels),
                        n_waveforms=int(propagation_n_waveforms),
                        channels_per_panel=int(propagation_channels_per_panel),
                        channel_overlap=int(propagation_channel_overlap),
                        ap_timings_json_path=(merged_units_dir / f"unit_{uid}" / "ap_timings.json"),
                        logger=logger,
                    )
            except Exception:
                pass

    return grid_entries
