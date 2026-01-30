from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _gather_template_sources_for_unit(
    *,
    uid: Any,
    analyzers: list[tuple[str, Any]],
    get_template_from_extension,
    sparsity_unit_channel_indices,
    try_get_electrode_ids,
) -> list[dict[str, Any]]:
    """Gather per-source template + channel metadata for one unit.

    Uses the analyzer's `templates` extension (preferred) and supports sparse templates by
    subsetting channel locations/ids according to the analyzer's sparsity.

    Args:
        uid: Unit id.
        analyzers: List of (source_name, SortingAnalyzer).
        get_template_from_extension: Callable compatible with footprinting/templates helpers.
        sparsity_unit_channel_indices: Callable that extracts per-unit channel indices from a sparsity object.
        try_get_electrode_ids: Callable that best-effort extracts electrode ids.

    Returns:
        List of dicts with keys: name, template, channel_locations, channel_ids, electrode_ids.
    """

    import numpy as np  # type: ignore[import-not-found]

    sources_for_unit: list[dict[str, Any]] = []

    for name, an in analyzers:
        tmpl = None
        try:
            t_ext = an.get_extension("templates") if an.has_extension("templates") else None
            if t_ext is not None:
                tmpl = get_template_from_extension(analyzer=an, templates_ext=t_ext, unit_id=uid)
        except Exception:
            tmpl = None

        if tmpl is None:
            continue

        tmpl = np.asarray(tmpl)
        if tmpl.ndim != 2 or tmpl.size == 0:
            continue

        locs = np.asarray(an.recording.get_channel_locations())
        try:
            ch_ids = list(an.recording.get_channel_ids())
        except Exception:
            ch_ids = None
        el_ids = try_get_electrode_ids(an.recording)

        # Support sparse templates by subsetting locations/ids according to sparsity.
        if tmpl.shape[1] != locs.shape[0]:
            try:
                sp = getattr(an, "sparsity", None)
                if sp is None and an.has_extension("waveforms"):
                    sp = getattr(an.get_extension("waveforms"), "sparsity", None)
                if sp is not None:
                    ch_inds = sparsity_unit_channel_indices(sparsity=sp, unit_id=uid)
                    ch_inds = np.asarray(ch_inds, dtype=int)
                    if int(ch_inds.size) == int(tmpl.shape[1]):
                        locs = locs[ch_inds, :]
                        if ch_ids is not None:
                            ch_ids = list(np.asarray(ch_ids, dtype=object)[ch_inds])
                        if el_ids is not None:
                            try:
                                el_ids = list(np.asarray(el_ids, dtype=object)[ch_inds])
                            except Exception:
                                pass
            except Exception:
                pass

        if tmpl.shape[1] != locs.shape[0]:
            continue

        sources_for_unit.append(
            {
                "name": str(name),
                "template": tmpl,
                "channel_locations": locs,
                "channel_ids": ch_ids,
                "electrode_ids": el_ids,
            }
        )

    return sources_for_unit


def _choose_grid_source_for_unit(*, sources_for_unit: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Choose a representative source for grid plotting (prefer concat)."""

    if not sources_for_unit:
        return None
    for s in sources_for_unit:
        if str(s.get("name")) == "concat":
            return s
    return sources_for_unit[0]


def _persist_unit_templates(
    *,
    uid: Any,
    sources_for_unit_with_union: list[dict[str, Any]],
    extracted_templates_dir: Path,
    merged_union_by_unit_dir: Path,
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    jsonable,
    jsonable_sequence,
    write_json,
    force_restart: bool,
    logger,
) -> dict[str, Any]:
    """Persist per-unit templates to disk and return a JSON-able unit summary entry."""

    import numpy as np  # type: ignore[import-not-found]

    unit_entry: dict[str, Any] = {"unit_id": jsonable(uid), "sources": []}

    for src in sources_for_unit_with_union:
        src_name = str(src["name"])
        tmpl = np.asarray(src["template"], dtype=float)
        locs = np.asarray(src["channel_locations"], dtype=float)

        if src_name == "merged_union":
            out_dir = merged_union_by_unit_dir / f"unit_{uid}"
            out_dir.mkdir(parents=True, exist_ok=True)
            npy_path = out_dir / "merged_union_template.npy"
            locs_npy = out_dir / "merged_union_channel_locations.npy"
            ch_ids_npy = out_dir / "merged_union_channel_ids.npy"
            el_ids_npy = out_dir / "merged_union_electrode_ids.npy"
            meta_path = out_dir / "merged_union_template_meta.json"
        else:
            out_dir = extracted_templates_dir / src_name
            out_dir.mkdir(parents=True, exist_ok=True)
            npy_path = out_dir / f"unit_{uid}.npy"
            meta_path = out_dir / f"unit_{uid}_meta.json"
            locs_npy = None
            ch_ids_npy = None
            el_ids_npy = None

        if (not npy_path.exists()) or force_restart:
            np.save(npy_path, tmpl)

        # For merged_union, also persist the merged channel identifiers/locations as arrays
        # (these are the primary inputs needed by downstream reconstruction).
        if src_name == "merged_union":
            try:
                assert locs_npy is not None and ch_ids_npy is not None and el_ids_npy is not None

                if (not locs_npy.exists()) or force_restart:
                    np.save(locs_npy, np.asarray(locs[:, :2], dtype=float))

                ch_ids_seq = jsonable_sequence(src.get("channel_ids"))
                if (not ch_ids_npy.exists()) or force_restart:
                    np.save(ch_ids_npy, np.asarray(ch_ids_seq, dtype=object))

                el_ids_seq = jsonable_sequence(src.get("electrode_ids"))
                if (not el_ids_npy.exists()) or force_restart:
                    np.save(el_ids_npy, np.asarray(el_ids_seq, dtype=object))
            except Exception as e:
                logger.warning("Failed writing merged_union aux arrays for unit %s: %s", uid, e)

        if (not meta_path.exists()) or force_restart:
            meta = {
                "unit_id": jsonable(uid),
                "source_name": src_name,
                "template_npy": str(npy_path),
                "channel_locations_npy": (str(locs_npy) if src_name == "merged_union" and locs_npy is not None else None),
                "channel_ids_npy": (str(ch_ids_npy) if src_name == "merged_union" and ch_ids_npy is not None else None),
                "electrode_ids_npy": (str(el_ids_npy) if src_name == "merged_union" and el_ids_npy is not None else None),
                "sampling_frequency_hz": float(fs_hz),
                "ms_before": ms_before,
                "ms_after": ms_after,
                "n_samples": int(tmpl.shape[0]),
                "n_channels": int(tmpl.shape[1]),
                "channel_ids": jsonable_sequence(src.get("channel_ids")),
                "electrode_ids": jsonable_sequence(src.get("electrode_ids")),
                "channel_locations": locs[:, :2].tolist(),
            }
            write_json(meta_path, meta)

        unit_entry["sources"].append(
            {
                "name": src_name,
                "template_npy": str(npy_path),
                "meta_json": str(meta_path),
                "n_channels": int(tmpl.shape[1]),
                "channel_locations_npy": (str(locs_npy) if src_name == "merged_union" and locs_npy is not None else None),
                "channel_ids_npy": (str(ch_ids_npy) if src_name == "merged_union" and ch_ids_npy is not None else None),
            }
        )

    return unit_entry
