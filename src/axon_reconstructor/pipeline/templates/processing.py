from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .extraction import _choose_grid_source_for_unit, _gather_template_sources_for_unit, _persist_unit_templates
from .utils import _build_union_template_for_unit


def _pick_existing(*, candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


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


def _apply_waveforms_stage_unit_curation(
    *,
    unit_ids: list[Any],
    well_out_dir: Path,
    load_curated_unit_ids_from_waveforms_outputs,
    normalize_id_for_compare,
    logger,
) -> tuple[list[Any], Optional[list[Any]], Optional[Path]]:
    """Filter unit_ids using waveforms-stage curation list if present."""

    curated_units_norm = None
    curation_metrics_xlsx = None

    curated_units_norm, curation_metrics_xlsx = load_curated_unit_ids_from_waveforms_outputs(
        well_out_dir=well_out_dir,
        logger=logger,
    )

    if curated_units_norm is not None:
        curated_set = set(curated_units_norm)
        before = len(unit_ids)
        unit_ids = [uid for uid in unit_ids if normalize_id_for_compare(uid) in curated_set]
        logger.info(
            "Applying waveforms curation: %d -> %d units (from %s)",
            before,
            len(unit_ids),
            curation_metrics_xlsx,
        )

    return unit_ids, curated_units_norm, curation_metrics_xlsx


def process_unit_list(
    *,
    unit_list: list[Any],
    analyzers: list[tuple[str, Any]],
    extracted_templates_dir: Path,
    merged_union_by_unit_dir: Path,
    plot_dir: Optional[Path],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels_per_template: int,
    force_restart: bool,
    n_jobs: int,
    get_template_from_extension,
    sparsity_unit_channel_indices,
    try_get_electrode_ids,
    jsonable,
    jsonable_sequence,
    write_json,
    write_unit_templates_across_sources_pdf,
    logger,
    persist: bool,
    summary: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Process units: gather per-source templates, build merged_union, persist, and plot."""

    # Ensure templates extension exists (best effort).
    for _, an in analyzers:
        try:
            if not an.has_extension("templates"):
                an.compute(["templates"], verbose=False, n_jobs=max(1, int(n_jobs)))
        except Exception:
            pass

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
                jsonable=jsonable,
                jsonable_sequence=jsonable_sequence,
                write_json=write_json,
                force_restart=bool(force_restart),
                logger=logger,
            )
            if summary is not None:
                summary.setdefault("units", []).append(unit_entry)

        chosen = _choose_grid_source_for_unit(sources_for_unit=sources_for_unit)
        if chosen is not None:
            grid_entries.append({"unit_id": uid, "template": chosen["template"]})

        if plot_dir is not None:
            unit_dir = plot_dir / f"unit_{uid}"
            unit_dir.mkdir(parents=True, exist_ok=True)
            unit_pdf = unit_dir / "templates.pdf"
            if (not unit_pdf.exists()) or force_restart:
                write_unit_templates_across_sources_pdf(
                    pdf_path=unit_pdf,
                    unit_id=uid,
                    sources_for_unit=sources_for_unit_with_union,
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                    top_channels=int(top_channels_per_template),
                )

    return grid_entries
