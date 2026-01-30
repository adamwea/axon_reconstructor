from __future__ import annotations

from pathlib import Path
from typing import Any

from .plotting import _write_unit_footprints_across_sources_pdf
from .utils import (
    _build_union_source_for_unit,
    _get_unit_template_from_extension,
    _sparsity_unit_channel_indices,
    _try_get_electrode_ids,
)


def _gather_sources_for_unit(*, uid: Any, analyzers: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    """Gather per-source footprint inputs (template-derived amplitudes + channel metadata) for a unit."""

    import numpy as np  # type: ignore[import-not-found]

    sources_for_unit: list[dict[str, Any]] = []

    for name, an in analyzers:
        tmpl_src = None
        try:
            t_ext = an.get_extension("templates") if an.has_extension("templates") else None
            if t_ext is not None:
                tmpl_src = _get_unit_template_from_extension(analyzer=an, templates_ext=t_ext, unit_id=uid)
        except Exception:
            tmpl_src = None
        if tmpl_src is None:
            continue

        tmpl_src = np.asarray(tmpl_src)
        if tmpl_src.ndim != 2 or tmpl_src.size == 0:
            continue

        locs_src = np.asarray(an.recording.get_channel_locations())[:, :2]
        ch_ids_src = None
        try:
            ch_ids_src = np.asarray(an.recording.get_channel_ids())
        except Exception:
            ch_ids_src = None

        el_ids_src = _try_get_electrode_ids(an.recording)
        if el_ids_src is not None:
            try:
                el_ids_src = np.asarray(el_ids_src)
            except Exception:
                el_ids_src = None

        # Support sparse templates by subsetting locations/ids according to sparsity.
        if tmpl_src.shape[1] != locs_src.shape[0]:
            try:
                sp = getattr(an, "sparsity", None)
                if sp is None and an.has_extension("waveforms"):
                    sp = getattr(an.get_extension("waveforms"), "sparsity", None)
                if sp is not None:
                    ch_inds = _sparsity_unit_channel_indices(sparsity=sp, unit_id=uid)
                    ch_inds = np.asarray(ch_inds, dtype=int)
                    if int(ch_inds.size) == int(tmpl_src.shape[1]):
                        locs_src = locs_src[ch_inds, :]
                        if ch_ids_src is not None:
                            ch_ids_src = np.asarray(ch_ids_src)[ch_inds]
                        if el_ids_src is not None:
                            el_ids_src = np.asarray(el_ids_src)[ch_inds]
            except Exception:
                pass

        if tmpl_src.shape[1] != locs_src.shape[0]:
            continue

        amp = np.ptp(tmpl_src, axis=0)
        best_ch = int(np.argmax(amp))
        sources_for_unit.append(
            {
                "name": name,
                "channel_locations": locs_src,
                "amp": amp,
                "best_ch": best_ch,
                "n_channels": int(locs_src.shape[0]),
                "channel_ids": ch_ids_src,
                "electrode_ids": el_ids_src,
            }
        )

    return sources_for_unit


def emit_multi_source_outputs(
    *,
    unit_list: list[Any],
    analyzers: list[tuple[str, Any]],
    out_dir: Path,
    out_summary: dict[str, Any],
    merged_dir: Path,
    merged_summary: dict[str, Any],
    logger,
    layout_locs,
    layout_key_to_index,
    layout_tol: float,
    unit_limit: int | None,
    plot_pdfs: bool,
) -> None:
    """Write per-unit multi-source footprint PDFs and populate the provided summary dicts."""

    processed = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    for uid in unit_list:
        sources_for_unit = _gather_sources_for_unit(uid=uid, analyzers=analyzers)

        merged_union_src = _build_union_source_for_unit(sources=sources_for_unit, unit_id=uid, logger=logger)
        merged_sources_for_unit = ([merged_union_src] + sources_for_unit) if merged_union_src is not None else sources_for_unit

        pdf_path = out_dir / f"unit_{uid}_footprints.pdf"

        unit_entry: dict[str, Any] = {
            "unit_id": int(uid) if str(uid).isdigit() else str(uid),
            "num_sources": int(len(sources_for_unit)),
            "sources": [s["name"] for s in sources_for_unit],
            "merged_union": (merged_union_src.get("merge") if merged_union_src is not None else None),
            "pdf_path": str(pdf_path) if merged_sources_for_unit else None,
            "merged_union_pdf_path": None,
            "error": None,
        }

        # Write merged-union-only PDF per unit.
        if plot_pdfs and merged_union_src is not None:
            merged_pdf_path = merged_dir / f"unit_{uid}_merged_union.pdf"
            try:
                _write_unit_footprints_across_sources_pdf(
                    sources=[merged_union_src],
                    unit_id=uid,
                    pdf_path=merged_pdf_path,
                    logger=logger,
                    layout_locs=layout_locs,
                    layout_key_to_index=layout_key_to_index,
                    layout_tol=float(layout_tol),
                    show_non_contributing_channels=True,
                )
                unit_entry["merged_union_pdf_path"] = str(merged_pdf_path)
                merged_summary["units"].append(
                    {
                        "unit_id": int(uid) if str(uid).isdigit() else str(uid),
                        "pdf_path": str(merged_pdf_path),
                        "merge": merged_union_src.get("merge"),
                        "n_channels": int(merged_union_src.get("n_channels", 0)),
                    }
                )
            except Exception as e:
                logger.warning("Failed writing merged_union PDF for unit %s: %s", uid, e)

        if plot_pdfs and merged_sources_for_unit:
            try:
                _write_unit_footprints_across_sources_pdf(
                    sources=merged_sources_for_unit,
                    unit_id=uid,
                    pdf_path=pdf_path,
                    logger=logger,
                    layout_locs=layout_locs,
                    layout_key_to_index=layout_key_to_index,
                    layout_tol=float(layout_tol),
                    show_non_contributing_channels=True,
                )
            except Exception as e:
                unit_entry["error"] = str(e)
                unit_entry["pdf_path"] = None

        out_summary["units"].append(unit_entry)

        processed += 1
        if unit_limit is not None and processed >= int(unit_limit):
            break
