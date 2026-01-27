from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from .pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from .pipeline_driver import _compute_mea_analysis_output_dir


FOOTPRINTING_OUTPUTS_DIRNAME = "footprinting_outputs"


def _infer_location_tolerance(locations: "Any") -> float:
    """Infer a reasonable coordinate-space tolerance for location matching.

    Channel locations are typically in µm for MEA data, but can be mm or meters.
    We use a 0.5 µm tolerance (scaled to the inferred units).
    """

    import numpy as np  # type: ignore[import-not-found]

    locs = np.asarray(locations)
    if locs.size == 0:
        return 0.0
    max_coord = float(np.nanmax(np.abs(locs)))
    # Heuristic matching the one used for electrode square sizing.
    if max_coord > 100.0:
        return 0.5
    if max_coord > 1.0:
        return 0.5 / 1000.0
    return 0.5e-6


def _try_get_recording_property(recording: Any, key: str):
    try:
        if hasattr(recording, "get_property_keys"):
            keys = set(recording.get_property_keys())
            if key not in keys:
                return None
        if hasattr(recording, "get_property"):
            return recording.get_property(key)
    except Exception:
        return None
    return None


def _try_get_electrode_ids(recording: Any):
    """Best-effort extraction of an 'electrode id' per channel.

    SpikeInterface recordings can carry various per-channel properties.
    We check a few common candidates; if none exist, returns None.
    """

    for key in (
        "electrode_id",
        "electrode",
        "contact_id",
        "contact_ids",
        "contact",
        "site_id",
        "site",
    ):
        vals = _try_get_recording_property(recording, key)
        if vals is not None:
            return vals
    return None


def _layout_loc_key(x: float, y: float, *, tol: float) -> tuple[int, int]:
    if tol <= 0:
        return (int(round(x * 1e6)), int(round(y * 1e6)))
    return (int(round(x / tol)), int(round(y / tol)))


def _layout_channel_key(*, x: float, y: float, tol: float, channel_id: Any = None, electrode_id: Any = None) -> Any:
    """Stable key for aligning channels across sources.

    Preference order:
    1) electrode_id (when present)
    2) channel_id
    3) binned location
    """

    if electrode_id is not None:
        try:
            return ("electrode", int(electrode_id))
        except Exception:
            return ("electrode", str(electrode_id))
    if channel_id is not None:
        return ("channel", str(channel_id))
    return ("loc", _layout_loc_key(float(x), float(y), tol=float(tol)))


def _build_global_channel_layout(*, analyzers: list[tuple[str, Any]]):
    """Build a union electrode layout across analyzers.

    Returns:
        layout_locs: (N, 2) array
        key_to_index: mapping key->index into layout_locs
        tol: location tolerance used for loc bucketing
    """

    import numpy as np  # type: ignore[import-not-found]

    all_locs = []
    for _, an in analyzers:
        try:
            all_locs.append(np.asarray(an.recording.get_channel_locations())[:, :2])
        except Exception:
            continue
    if not all_locs:
        return np.zeros((0, 2), dtype=float), {}, 0.0

    stacked = np.concatenate(all_locs, axis=0)
    tol = float(_infer_location_tolerance(stacked))

    def _add_from_recording(recording, *, prefer_existing_order: bool, key_to_index: dict[Any, int], locs_list: list[list[float]]):
        try:
            locs = np.asarray(recording.get_channel_locations())[:, :2]
        except Exception:
            return

        try:
            ch_ids = list(recording.get_channel_ids())
        except Exception:
            ch_ids = None

        el_ids = _try_get_electrode_ids(recording)
        if el_ids is not None:
            try:
                el_ids = list(el_ids)
            except Exception:
                el_ids = None

        # Determine insertion order: keep concat ordering first, then append new channels.
        for i in range(locs.shape[0]):
            x = float(locs[i, 0])
            y = float(locs[i, 1])
            cid = None
            if ch_ids is not None:
                try:
                    cid = ch_ids[i]
                except Exception:
                    cid = None
            eid = None
            if el_ids is not None:
                try:
                    eid = el_ids[i]
                except Exception:
                    eid = None
            key = _layout_channel_key(x=x, y=y, tol=float(tol), channel_id=cid, electrode_id=eid)
            if key in key_to_index:
                continue
            key_to_index[key] = len(locs_list)
            locs_list.append([x, y])

    # Start with concat (if present) for stable base ordering.
    key_to_index: dict[Any, int] = {}
    locs_list: list[list[float]] = []

    concat = None
    for name, an in analyzers:
        if str(name) == "concat":
            concat = an
            break
    if concat is not None:
        _add_from_recording(concat.recording, prefer_existing_order=True, key_to_index=key_to_index, locs_list=locs_list)

    for _, an in analyzers:
        _add_from_recording(an.recording, prefer_existing_order=False, key_to_index=key_to_index, locs_list=locs_list)

    return np.asarray(locs_list, dtype=float), key_to_index, tol


def _build_union_source_for_unit(
    *,
    sources: list[dict[str, Any]],
    unit_id: Any,
    logger,
) -> Optional[dict[str, Any]]:
    """Union channels across per-source footprints for a unit.

    If overlaps are detected (by channel_id, electrode_id, or location within tolerance),
    we warn and keep the first occurrence.
    """

    import numpy as np  # type: ignore[import-not-found]

    if not sources:
        return None

    # Determine location tolerance from all locations.
    all_locs = [np.asarray(s.get("channel_locations")) for s in sources if s.get("channel_locations") is not None]
    if not all_locs:
        return None
    stacked = np.concatenate(all_locs, axis=0)
    tol = float(_infer_location_tolerance(stacked))
    if tol <= 0:
        tol = 0.0

    def loc_key(x: float, y: float) -> tuple[int, int]:
        if tol <= 0:
            return (int(round(x * 1e6)), int(round(y * 1e6)))
        return (int(round(x / tol)), int(round(y / tol)))

    union_locs: list[list[float]] = []
    union_amp: list[float] = []
    union_channel_ids: list[Any] = []
    union_electrode_ids: list[Any] = []

    seen_channel_ids: set[Any] = set()
    seen_electrode_ids: set[Any] = set()
    seen_loc_keys: set[tuple[int, int]] = set()

    overlap_counts = {"channel_id": 0, "electrode_id": 0, "location": 0}

    for src in sources:
        locs = np.asarray(src.get("channel_locations"))
        amp = np.asarray(src.get("amp"))
        ch_ids = src.get("channel_ids")
        el_ids = src.get("electrode_ids")

        if locs.size == 0 or amp.size == 0 or locs.shape[0] != amp.shape[0]:
            continue

        # Optional: warn about duplicates within source.
        try:
            if ch_ids is not None:
                ch_list = list(ch_ids)
                if len(set(ch_list)) != len(ch_list):
                    logger.warning("Duplicate channel_ids within source %s (unit %s)", src.get("name"), unit_id)
        except Exception:
            pass

        for i in range(locs.shape[0]):
            x = float(locs[i, 0])
            y = float(locs[i, 1])
            lk = loc_key(x, y)

            cid = None
            if ch_ids is not None:
                try:
                    cid = ch_ids[i]
                except Exception:
                    cid = None

            eid = None
            if el_ids is not None:
                try:
                    eid = el_ids[i]
                except Exception:
                    eid = None

            # Overlap checks: if any overlap, keep the first and skip.
            if cid is not None and cid in seen_channel_ids:
                overlap_counts["channel_id"] += 1
                continue
            if eid is not None and eid in seen_electrode_ids:
                overlap_counts["electrode_id"] += 1
                continue
            if lk in seen_loc_keys:
                overlap_counts["location"] += 1
                continue

            seen_loc_keys.add(lk)
            if cid is not None:
                seen_channel_ids.add(cid)
            if eid is not None:
                seen_electrode_ids.add(eid)

            union_locs.append([x, y])
            union_amp.append(float(amp[i]))
            union_channel_ids.append(cid)
            union_electrode_ids.append(eid)

    total_overlaps = sum(overlap_counts.values())
    if total_overlaps:
        logger.warning(
            "Merged footprint overlaps for unit %s: %s (kept first, skipped the rest)",
            unit_id,
            overlap_counts,
        )

    if not union_locs:
        return None

    union_locs_arr = np.asarray(union_locs)
    union_amp_arr = np.asarray(union_amp)

    return {
        "name": "merged_union",
        "channel_locations": union_locs_arr,
        "amp": union_amp_arr,
        "best_ch": int(np.argmax(union_amp_arr)) if union_amp_arr.size else 0,
        "n_channels": int(union_locs_arr.shape[0]),
        "channel_ids": union_channel_ids,
        "electrode_ids": union_electrode_ids,
        "merge": {
            "location_tolerance": tol,
            "overlap_counts": overlap_counts,
        },
    }


def _normalize_id_for_compare(x: Any) -> Any:
    """Normalize ids so np scalars / floats round-trip consistently."""

    try:
        # numpy scalar -> python scalar
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass

    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return str(x)


def _load_curated_unit_ids_from_waveforms_outputs(*, well_out_dir: Path, logger) -> tuple[Optional[list[Any]], Optional[Path]]:
    """Load curated (kept) unit ids from waveforms outputs, if available.

    Waveforms stage writes MEA_Analysis-style curation artifacts, including:
      <well>/waveforms_outputs/metrics_curated.xlsx

    We treat the index of that spreadsheet as the curated/kept unit ids.
    """

    metrics_curated_xlsx = well_out_dir / "waveforms_outputs" / "metrics_curated.xlsx"
    if not metrics_curated_xlsx.exists():
        return None, None

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        logger.warning("Found %s but pandas is unavailable; cannot apply unit curation", metrics_curated_xlsx)
        return None, metrics_curated_xlsx

    try:
        df = pd.read_excel(metrics_curated_xlsx, index_col=0)
        curated = [_normalize_id_for_compare(x) for x in list(df.index.values)]
        # Preserve order but de-dupe.
        seen: set[Any] = set()
        curated_unique: list[Any] = []
        for u in curated:
            if u in seen:
                continue
            seen.add(u)
            curated_unique.append(u)
        return curated_unique, metrics_curated_xlsx
    except Exception as e:
        logger.warning("Failed reading curated unit list from %s: %s", metrics_curated_xlsx, e)
        return None, metrics_curated_xlsx

def _load_wf_rejection_log_summary(*, well_out_dir: Path, logger) -> tuple[Optional[dict[str, Any]], Optional[Path]]:
    """Load waveforms-stage per-spike rejection log summary (best effort).

    The detailed sheets can be extremely large; footprinting only consumes the
    lightweight `summary` sheet so we can carry the metadata forward without
    incurring a heavy read.
    """

    wf_rej_xlsx = well_out_dir / "waveforms_outputs" / "wf_rejection_log.xlsx"
    if not wf_rej_xlsx.exists():
        return None, None

    try:
        import pandas as pd  # type: ignore[import-not-found]

        df = pd.read_excel(wf_rej_xlsx, sheet_name="summary")
        summary: dict[str, Any] = {"wf_rejection_log_xlsx": str(wf_rej_xlsx)}
        if not df.empty and {"metric", "value"}.issubset(set(df.columns)):
            for _, row in df.iterrows():
                try:
                    summary[str(row["metric"])] = row["value"]
                except Exception:
                    continue
        return summary, wf_rej_xlsx
    except Exception as e:
        logger.warning("Failed to read wf_rejection_log.xlsx summary: %s", e)
        return {"wf_rejection_log_xlsx": str(wf_rej_xlsx), "error": str(e)}, wf_rej_xlsx


def _ensure_analyzer_extensions(*, analyzer, extension_names: list[str], logger, n_jobs: int) -> None:
    missing = [name for name in extension_names if not analyzer.has_extension(name)]
    if not missing:
        return
    logger.info("Computing extensions: %s", ", ".join(missing))
    analyzer.compute(missing, verbose=False, n_jobs=max(1, int(n_jobs)))


def _sparsity_unit_channel_indices(*, sparsity, unit_id: Any):
    """Return per-unit channel indices from a `ChannelSparsity` object.

    SpikeInterface API differs across versions:
    - Some versions expose `unit_id_to_channel_indices` as a callable.
    - Others (e.g. SI 0.103.x) expose it as a dict.
    """

    if sparsity is None:
        return None

    try:
        mapping = getattr(sparsity, "unit_id_to_channel_indices", None)
        if callable(mapping):
            return mapping(unit_id)
        if isinstance(mapping, dict):
            if unit_id in mapping:
                return mapping[unit_id]
            # Best-effort normalized match.
            try:
                from .waveform_exclusions import normalize_unit_id

                uid_norm = normalize_unit_id(unit_id)
                for k, v in mapping.items():
                    if normalize_unit_id(k) == uid_norm:
                        return v
            except Exception:
                pass
    except Exception:
        pass

    # Additional fallbacks across versions.
    for attr in ("get_channel_indices", "get_channel_indices_for_unit"):
        if hasattr(sparsity, attr):
            fn = getattr(sparsity, attr)
            if callable(fn):
                try:
                    return fn(unit_id)
                except Exception:
                    pass

    return None


def _get_unit_template_from_waveforms_with_exclusions(
    *,
    analyzer,
    unit_id: Any,
    excluded_spike_samples: Optional[set[int]],
    logger,
    return_result: bool = False,
):
    """Compute a unit template from waveforms, applying spike-level exclusions."""

    try:
        from .waveform_exclusions import compute_unit_template_from_waveforms

        res = compute_unit_template_from_waveforms(
            analyzer=analyzer,
            unit_id=unit_id,
            excluded_spike_samples=excluded_spike_samples,
            logger=logger,
        )
        if return_result:
            return res
        return None if res is None else res.template
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_footprinting_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_footprinting_checkpoint.json"
    else:
        name = main_ckpt.stem + "_footprinting_checkpoint.json"
    return main_ckpt.with_name(name)


def _load_waveforms_analyzers(
    *,
    well_out_dir: Path,
    include_concat: bool,
    include_segments: bool,
    logger,
):
    """Load analyzers produced by the waveforms stage.

    Returns a list of (source_name, analyzer).
    """

    import spikeinterface.full as si  # type: ignore[import-not-found]

    waveforms_out_dir = well_out_dir / "waveforms_outputs"
    concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"
    segment_waveforms_dir = waveforms_out_dir / "segment_waveforms"

    analyzers: list[tuple[str, Any]] = []

    if include_concat:
        if not concat_waveforms_dir.exists():
            raise FileNotFoundError(f"Missing concat waveforms analyzer at {concat_waveforms_dir}")
        logger.info("Loading concat analyzer: %s", concat_waveforms_dir)
        analyzers.append(("concat", si.load_sorting_analyzer(concat_waveforms_dir)))

    if include_segments and segment_waveforms_dir.exists():
        seg_dirs = sorted([p for p in segment_waveforms_dir.iterdir() if p.is_dir()])
        logger.info("Found %d segment analyzers", len(seg_dirs))
        for p in seg_dirs:
            try:
                analyzers.append((p.name, si.load_sorting_analyzer(p)))
            except Exception:
                logger.warning("Skipping unreadable segment analyzer: %s", p)

    if not analyzers:
        raise RuntimeError("No analyzers available for footprinting")

    return analyzers


def _get_unit_template_from_extension(*, analyzer, templates_ext, unit_id: Any):
    """Compatibility helper for SpikeInterface templates extension."""

    # When comparing across sources (concat vs segments), some units may be
    # absent in a particular analyzer. SpikeInterface will raise in that case;
    # treat it as a missing template instead.
    try:
        if hasattr(analyzer, "sorting") and hasattr(analyzer.sorting, "id_to_index"):
            analyzer.sorting.id_to_index(unit_id)
    except Exception:
        return None

    if hasattr(templates_ext, "get_unit_template"):
        try:
            return templates_ext.get_unit_template(unit_id=unit_id)
        except Exception:
            return None

    if hasattr(templates_ext, "get_templates"):
        try:
            all_templates = templates_ext.get_templates()
            unit_index = list(analyzer.sorting.unit_ids).index(unit_id)
            return all_templates[unit_index]
        except Exception:
            return None

    return None


def _write_merged_union_footprints_grid_pdf(
    *,
    analyzers: list[tuple[str, Any]],
    pdf_path: Path,
    unit_ids: list[Any],
    exclusions_by_source: dict[str, dict[Any, set[int]]],
    apply_exclusions: bool,
    layout_locs: "Any",
    layout_key_to_index: dict[Any, int],
    layout_tol: float,
    logger,
    wf_excl_report: Optional[dict[str, Any]] = None,
    wf_excl_scope: str = "merged_union_grid",
) -> None:
    """Write a multi-page PDF of per-unit merged_union footprints on a global layout.

    Uncurated semantics: apply_exclusions=False (templates average all stored waveforms)
    Curated semantics: apply_exclusions=True (templates drop excluded spikes)

    Non-contributing channels are always shown in gray.
    """

    try:
        import logging

        import numpy as np  # type: ignore[import-not-found]

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as pdf
        from matplotlib.colors import LogNorm
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting footprints requires numpy/matplotlib") from e

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    layout_locs = np.asarray(layout_locs)
    if layout_locs.ndim != 2 or layout_locs.shape[0] == 0:
        return

    # Map of normalized unit id for exclusions lookup.
    try:
        from .waveform_exclusions import normalize_unit_id

        _norm_unit_id = normalize_unit_id
    except Exception:
        _norm_unit_id = lambda x: x

    # Compute union amps for each unit (two-pass so we can build a shared LogNorm).
    union_by_unit: list[tuple[Any, "np.ndarray"]] = []
    vmin_pos = None
    vmax = 0.0

    for uid in unit_ids:
        amp_union = np.full((layout_locs.shape[0],), np.nan, dtype=float)

        for name, an in analyzers:
            excluded = set()
            if apply_exclusions:
                try:
                    excluded = exclusions_by_source.get(str(name), {}).get(_norm_unit_id(uid), set())
                except Exception:
                    excluded = set()

            res = _get_unit_template_from_waveforms_with_exclusions(
                analyzer=an,
                unit_id=uid,
                excluded_spike_samples=(excluded if apply_exclusions else set()),
                logger=logger,
                return_result=True,
            )

            if wf_excl_report is not None:
                try:
                    from .waveform_exclusions import update_wf_exclusion_report

                    update_wf_exclusion_report(
                        wf_excl_report,
                        scope=str(wf_excl_scope),
                        source_name=str(name),
                        unit_id=uid,
                        excluded_spike_samples=(excluded if apply_exclusions else set()),
                        result=res,
                    )
                except Exception:
                    pass

            tmpl = (None if res is None else getattr(res, "template", None))
            if tmpl is None:
                continue
            tmpl = np.asarray(tmpl)
            if tmpl.ndim != 2 or tmpl.size == 0:
                continue

            locs_src = np.asarray(an.recording.get_channel_locations())[:, :2]
            try:
                ch_ids_src = list(an.recording.get_channel_ids())
            except Exception:
                ch_ids_src = None
            el_ids_src = _try_get_electrode_ids(an.recording)
            if el_ids_src is not None:
                try:
                    el_ids_src = list(el_ids_src)
                except Exception:
                    el_ids_src = None

            # Support sparse templates by subsetting locations/ids according to sparsity.
            if tmpl.shape[1] != locs_src.shape[0]:
                try:
                    sp = getattr(an, "sparsity", None)
                    if sp is None and an.has_extension("waveforms"):
                        sp = getattr(an.get_extension("waveforms"), "sparsity", None)
                    if sp is not None:
                        ch_inds = _sparsity_unit_channel_indices(sparsity=sp, unit_id=uid)
                        ch_inds = np.asarray(ch_inds, dtype=int)
                        if int(ch_inds.size) == int(tmpl.shape[1]):
                            locs_src = locs_src[ch_inds, :]
                            if ch_ids_src is not None:
                                ch_ids_src = list(np.asarray(ch_ids_src, dtype=object)[ch_inds])
                            if el_ids_src is not None:
                                try:
                                    el_ids_src = list(np.asarray(el_ids_src, dtype=object)[ch_inds])
                                except Exception:
                                    pass
                except Exception:
                    pass

            if tmpl.shape[1] != locs_src.shape[0]:
                continue

            amp = np.ptp(tmpl, axis=0)
            for i in range(locs_src.shape[0]):
                cid = None
                if ch_ids_src is not None:
                    try:
                        cid = ch_ids_src[i]
                    except Exception:
                        cid = None
                eid = None
                if el_ids_src is not None:
                    try:
                        eid = el_ids_src[i]
                    except Exception:
                        eid = None

                key = _layout_channel_key(
                    x=float(locs_src[i, 0]),
                    y=float(locs_src[i, 1]),
                    tol=float(layout_tol),
                    channel_id=cid,
                    electrode_id=eid,
                )
                idx = layout_key_to_index.get(key)
                if idx is None:
                    continue
                if np.isfinite(amp_union[int(idx)]):
                    continue
                amp_union[int(idx)] = float(amp[i])

        union_by_unit.append((uid, amp_union))
        if np.any(np.isfinite(amp_union)):
            vmax = max(vmax, float(np.nanmax(amp_union)))
            pos = amp_union[np.isfinite(amp_union) & (amp_union > 0)]
            if pos.size:
                v = float(np.nanmin(pos))
                vmin_pos = v if vmin_pos is None else min(vmin_pos, v)

    # Shared LogNorm across the whole PDF.
    norm = None
    norm_vmin_for_zeros = None
    try:
        if vmin_pos is not None and vmax > 0:
            vmin = max(1.0, float(vmin_pos))
            if vmin >= vmax:
                vmin = vmax / 10.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
            norm_vmin_for_zeros = float(vmin)
    except Exception:
        norm = None
        norm_vmin_for_zeros = None

    xs_all = layout_locs[:, 0]
    ys_all = layout_locs[:, 1]
    pad = 20.0
    xlim = (float(np.min(xs_all)) - pad, float(np.max(xs_all)) + pad)
    ylim = (float(np.min(ys_all)) - pad, float(np.max(ys_all)) + pad)

    def _electrode_square_side_in_data_units(channel_locations: "np.ndarray", *, side_um: float = 17.5) -> float:
        max_coord = float(np.nanmax(np.abs(channel_locations)))
        if max_coord > 100.0:
            return float(side_um)
        if max_coord > 1.0:
            return float(side_um) / 1000.0
        return float(side_um) * 1e-6

    def _square_marker_area_points2(ax, *, side_len: float) -> float:
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((float(side_len), 0.0))
        dx_pixels = abs(float(p1[0]) - float(p0[0]))
        side_points = dx_pixels * 72.0 / float(ax.figure.dpi)
        return float(side_points * side_points)

    n_per_page = 12
    n_rows = 4
    n_cols = 3
    fig_size = (10, 12)
    dark_bg = "#0b0b0b"
    cmap_name = "turbo"
    base_gray = "#6b6b6b"

    square_side = _electrode_square_side_in_data_units(layout_locs, side_um=17.5)

    with pdf.PdfPages(pdf_path) as pdf_doc:
        for i in range(0, len(union_by_unit), n_per_page):
            batch = union_by_unit[i : i + n_per_page]
            fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
            axes = np.asarray(axes).flatten()
            fig.patch.set_facecolor("white")
            fig.subplots_adjust(left=0.04, right=0.88, bottom=0.04, top=0.92, wspace=0.05, hspace=0.12)

            last_mappable = None
            marker_area = None

            for ax, (uid, amp_union) in zip(axes, batch, strict=False):
                ax.set_facecolor(dark_bg)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")

                if marker_area is None:
                    marker_area = _square_marker_area_points2(ax, side_len=square_side)

                # Base gray for all channels.
                ax.scatter(
                    layout_locs[:, 0],
                    layout_locs[:, 1],
                    c=base_gray,
                    s=float(marker_area or 1.0),
                    marker="s",
                    linewidths=0,
                    edgecolors="none",
                    alpha=1.0,
                    rasterized=True,
                )

                keep = np.isfinite(amp_union)
                if not np.any(keep):
                    ax.set_title(f"Unit {uid} (no template)", fontsize=10, color="black")
                    continue

                amp_for_color = amp_union[keep]
                if norm is not None and norm_vmin_for_zeros is not None:
                    amp_for_color = np.where(amp_for_color <= 0, norm_vmin_for_zeros, amp_for_color)

                last_mappable = ax.scatter(
                    layout_locs[keep, 0],
                    layout_locs[keep, 1],
                    c=amp_for_color,
                    s=float(marker_area or 1.0),
                    marker="s",
                    cmap=cmap_name,
                    norm=norm,
                    linewidths=0,
                    edgecolors="none",
                    alpha=1.0,
                )
                last_mappable.set_rasterized(True)
                ax.set_title(f"Unit {uid}", fontsize=10, color="black")

            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            fig.suptitle("Footprints merged_union (template PTP)", fontsize=12, color="black")

            if last_mappable is not None:
                try:
                    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
                    cax.set_facecolor("white")
                    cbar = fig.colorbar(last_mappable, cax=cax)
                    cbar.set_label("Template PTP (µV)", fontsize=9, color="black")
                    cbar.ax.tick_params(labelsize=8, colors="black")
                    try:
                        cbar.outline.set_edgecolor("black")
                    except Exception:
                        pass
                except Exception:
                    pass

            pdf_doc.savefig(fig, dpi=300)
            plt.close(fig)

    logger.info("Wrote merged_union footprints grid PDF: %s", pdf_path)


def _write_unit_footprints_across_sources_pdf(
    *,
    sources: list[dict[str, Any]],
    unit_id: Any,
    pdf_path: Path,
    logger,
    layout_locs: Optional["Any"] = None,
    layout_key_to_index: Optional[dict[Any, int]] = None,
    layout_tol: Optional[float] = None,
    show_non_contributing_channels: bool = True,
    non_contributing_color: str = "#6b6b6b",
) -> None:
    """Write a per-unit multi-page PDF showing footprints across sources."""

    try:
        import logging

        import numpy as np  # type: ignore[import-not-found]

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as pdf
        from matplotlib.colors import LogNorm

        # Intentionally no scalebar here (see _write_footprints_grid_pdf).
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting multi-source footprints requires numpy/matplotlib") from e

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    all_locs = [np.asarray(s["channel_locations"]) for s in sources if s.get("channel_locations") is not None]
    if not all_locs:
        return

    if layout_locs is None:
        stacked = np.concatenate(all_locs, axis=0)
        layout_locs = stacked
    layout_locs = np.asarray(layout_locs)
    if layout_locs.size == 0:
        return

    if layout_tol is None:
        try:
            layout_tol = float(_infer_location_tolerance(layout_locs))
        except Exception:
            layout_tol = 0.0

    xs_all = layout_locs[:, 0]
    ys_all = layout_locs[:, 1]

    pad = 20.0
    xlim = (float(np.min(xs_all)) - pad, float(np.max(xs_all)) + pad)
    ylim = (float(np.min(ys_all)) - pad, float(np.max(ys_all)) + pad)

    def _electrode_square_side_in_data_units(channel_locations: "np.ndarray", *, side_um: float = 17.5) -> float:
        max_coord = float(np.nanmax(np.abs(channel_locations)))
        if max_coord > 100.0:
            return float(side_um)
        if max_coord > 1.0:
            return float(side_um) / 1000.0
        return float(side_um) * 1e-6

    def _square_marker_area_points2(ax, *, side_len: float) -> float:
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((float(side_len), 0.0))
        dx_pixels = abs(float(p1[0]) - float(p0[0]))
        side_points = dx_pixels * 72.0 / float(ax.figure.dpi)
        return float(side_points * side_points)

    # Simple shared log scaling across sources for this unit.
    norm = None
    norm_vmin_for_zeros = None
    try:
        amp_all = np.concatenate([np.asarray(s["amp"]) for s in sources if s.get("amp") is not None])
        vmax = float(np.nanmax(amp_all))
        pos = amp_all[amp_all > 0]
        if pos.size and vmax > 0:
            vmin = float(np.nanmin(pos))
            vmin = max(1.0, vmin)
            if vmin >= vmax:
                vmin = vmax / 10.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
            norm_vmin_for_zeros = float(vmin)
    except Exception:
        norm = None
        norm_vmin_for_zeros = None

    if len(sources) == 1:
        # For merged-union-only PDFs, use the whole page (no empty grid).
        panels_per_page = 1
        n_rows = 1
        n_cols = 1
        fig_size = (9, 9)
        top = 0.92
        right = 0.88
    else:
        panels_per_page = 12
        n_rows = 4
        n_cols = 3
        fig_size = (10, 12)
        top = 0.91
        right = 0.88
    square_side = _electrode_square_side_in_data_units(layout_locs, side_um=17.5)

    def _src_keys_and_values(src: dict[str, Any]):
        locs = np.asarray(src.get("channel_locations"))
        amp = np.asarray(src.get("amp"))
        ch_ids = src.get("channel_ids")
        el_ids = src.get("electrode_ids")
        if el_ids is not None:
            try:
                el_ids = list(el_ids)
            except Exception:
                el_ids = None
        if ch_ids is not None:
            try:
                ch_ids = list(ch_ids)
            except Exception:
                ch_ids = None

        keys = []
        for i in range(locs.shape[0]):
            cid = None
            if ch_ids is not None:
                try:
                    cid = ch_ids[i]
                except Exception:
                    cid = None
            eid = None
            if el_ids is not None:
                try:
                    eid = el_ids[i]
                except Exception:
                    eid = None

            keys.append(
                _layout_channel_key(
                    x=float(locs[i, 0]),
                    y=float(locs[i, 1]),
                    tol=float(layout_tol or 0.0),
                    channel_id=cid,
                    electrode_id=eid,
                )
            )
        return keys, amp
    with pdf.PdfPages(pdf_path) as pdf_doc:
        for i in range(0, len(sources), panels_per_page):
            batch = sources[i : i + panels_per_page]
            fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
            if isinstance(axes, (list, tuple)):
                axes = np.asarray(axes)
            axes = np.atleast_1d(axes).flatten()

            dark_bg = "#0b0b0b"
            cmap_name = "turbo"
            fig.patch.set_facecolor("white")

            fig.subplots_adjust(left=0.04, right=right, bottom=0.04, top=top, wspace=0.05, hspace=0.12)

            last_mappable = None
            marker_area = None
            for ax, src in zip(axes, batch, strict=False):
                ax.set_facecolor(dark_bg)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")

                if marker_area is None:
                    marker_area = _square_marker_area_points2(ax, side_len=square_side)

                locs = np.asarray(src["channel_locations"])
                amp = np.asarray(src["amp"])

                # Optionally plot non-contributing channels as a gray MEA backdrop.
                if show_non_contributing_channels and layout_key_to_index is not None:
                    try:
                        ax.scatter(
                            layout_locs[:, 0],
                            layout_locs[:, 1],
                            c=non_contributing_color,
                            s=float(marker_area or 1.0),
                            marker="s",
                            linewidths=0,
                            edgecolors="none",
                            alpha=1.0,
                            rasterized=True,
                        )
                    except Exception:
                        pass

                # Map this source onto the global layout.
                amp_full = None
                if layout_key_to_index is not None:
                    try:
                        keys, amp_vals = _src_keys_and_values(src)
                        amp_full = np.full((layout_locs.shape[0],), np.nan, dtype=float)
                        for k, v in zip(keys, amp_vals, strict=False):
                            idx = layout_key_to_index.get(k)
                            if idx is None:
                                continue
                            amp_full[int(idx)] = float(v)
                    except Exception:
                        amp_full = None

                # Fall back to local-only plotting if mapping fails.
                plot_locs = layout_locs if amp_full is not None else locs
                plot_amp = amp_full if amp_full is not None else amp

                amp_for_color = plot_amp
                if norm is not None and norm_vmin_for_zeros is not None:
                    amp_for_color = np.where(plot_amp <= 0, norm_vmin_for_zeros, plot_amp)

                # Only color contributing channels; leave NaNs as backdrop gray.
                if amp_full is not None:
                    keep = np.isfinite(plot_amp)
                    plot_locs = plot_locs[keep, :]
                    amp_for_color = amp_for_color[keep]

                last_mappable = ax.scatter(
                    plot_locs[:, 0],
                    plot_locs[:, 1],
                    c=amp_for_color,
                    s=float(marker_area or 1.0),
                    marker="s",
                    cmap=cmap_name,
                    norm=norm,
                    linewidths=0,
                    edgecolors="none",
                    alpha=1.0,
                )
                last_mappable.set_rasterized(True)

                if len(sources) != 1:
                    ax.set_title(
                        f"{src['name']} | n={int(src.get('n_channels', locs.shape[0]))}",
                        fontsize=10,
                        color="black",
                    )



            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            title = (
                f"Unit {unit_id} | {sources[0].get('name', 'source')} footprint (log color scale)"
                if len(sources) == 1
                else f"Unit {unit_id} | Footprints across sources (log color scale)"
            )
            fig.suptitle(title, fontsize=12, color="black")

            if last_mappable is not None:
                try:
                    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
                    cax.set_facecolor("white")
                    cbar = fig.colorbar(last_mappable, cax=cax)
                    cbar.set_label("Template PTP (µV)", fontsize=9, color="black")
                    cbar.ax.tick_params(labelsize=8, colors="black")
                    try:
                        cbar.outline.set_edgecolor("black")
                    except Exception:
                        pass
                except Exception:
                    pass

            pdf_doc.savefig(fig, dpi=300)
            plt.close(fig)

    logger.info("Wrote multi-source footprints PDF: %s", pdf_path)


@dataclass(frozen=True)
class FootprintingInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    # Sources
    include_concat: bool = True
    include_segments: bool = True

    # Plotting
    plot_concat_footprints_grid_pdf: bool = True
    plot_multi_source_footprints_pdf: bool = True

    # Controls
    n_jobs: int = 8
    unit_limit: Optional[int] = None
    unit_ids: Optional[list[Any]] = None
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

        # Spike-level exclusions produced by the waveforms stage (fast .npz).
        try:
            from .waveform_exclusions import (
                init_wf_exclusion_report,
                load_wf_exclusions_by_source,
                normalize_unit_id,
            )

            exclusions_by_source = load_wf_exclusions_by_source(well_out_dir=well_out_dir, logger=logger)
            _norm_unit_id = normalize_unit_id
        except Exception:
            exclusions_by_source = {}
            _norm_unit_id = lambda x: x

        wf_exclusions_applied_report_json = footprinting_out_dir / "wf_exclusions_applied_report.json"
        try:
            wf_excl_report = init_wf_exclusion_report(stage="footprinting", exclusions_by_source=exclusions_by_source)
        except Exception:
            wf_excl_report = {
                "stage": "footprinting",
                "loaded_exclusions": {"n_sources": 0, "n_units": 0, "n_spikes": 0, "by_source": {}},
                "application": {"scopes": {}},
            }

        for _, an in analyzers:
            _ensure_analyzer_extensions(
                analyzer=an,
                extension_names=["random_spikes", "waveforms"],
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

        def _gather_sources_for_unit(*, uid: Any, apply_exclusions: bool) -> list[dict[str, Any]]:
            sources_for_unit: list[dict[str, Any]] = []
            for name, an in analyzers:
                excluded = set()
                if apply_exclusions:
                    try:
                        excluded = exclusions_by_source.get(str(name), {}).get(_norm_unit_id(uid), set())
                    except Exception:
                        excluded = set()

                res = _get_unit_template_from_waveforms_with_exclusions(
                    analyzer=an,
                    unit_id=uid,
                    excluded_spike_samples=(excluded if apply_exclusions else set()),
                    logger=logger,
                    return_result=True,
                )

                try:
                    from .waveform_exclusions import update_wf_exclusion_report

                    update_wf_exclusion_report(
                        wf_excl_report,
                        scope=("multi_source_curated" if apply_exclusions else "multi_source_uncurated"),
                        source_name=str(name),
                        unit_id=uid,
                        excluded_spike_samples=(excluded if apply_exclusions else set()),
                        result=res,
                    )
                except Exception:
                    pass

                tmpl_src = (None if res is None else getattr(res, "template", None))
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

        def _emit_multi_source_outputs(
            *,
            unit_list: list[Any],
            apply_exclusions: bool,
            out_dir: Path,
            out_summary: dict[str, Any],
            merged_dir: Path,
            merged_summary: dict[str, Any],
        ) -> None:
            processed = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            merged_dir.mkdir(parents=True, exist_ok=True)

            for uid in unit_list:
                sources_for_unit = _gather_sources_for_unit(uid=uid, apply_exclusions=apply_exclusions)

                merged_union_src = _build_union_source_for_unit(sources=sources_for_unit, unit_id=uid, logger=logger)
                merged_sources_for_unit = (
                    ([merged_union_src] + sources_for_unit) if merged_union_src is not None else sources_for_unit
                )

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
                if inputs.plot_multi_source_footprints_pdf and merged_union_src is not None:
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

                if inputs.plot_multi_source_footprints_pdf and merged_sources_for_unit:
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
                if inputs.unit_limit is not None and processed >= int(inputs.unit_limit):
                    break

        if inputs.plot_multi_source_footprints_pdf:
            # Uncurated: all units, pre-exclusion.
            _emit_multi_source_outputs(
                unit_list=unit_ids_all,
                apply_exclusions=False,
                out_dir=multi_source_dir_uncurated,
                out_summary=multi_source_summary_uncurated,
                merged_dir=merged_union_dir_uncurated,
                merged_summary=merged_union_summary_uncurated,
            )

            # Curated: curated units, exclusions applied.
            _emit_multi_source_outputs(
                unit_list=unit_ids,
                apply_exclusions=True,
                out_dir=multi_source_dir,
                out_summary=multi_source_summary,
                merged_dir=merged_union_dir,
                merged_summary=merged_union_summary,
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
                wf_excl_report=wf_excl_report,
                wf_excl_scope="concat_grid_uncurated",
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
                wf_excl_report=wf_excl_report,
                wf_excl_scope="concat_grid_curated",
            )

        # Persist a compact report so it's easy to audit whether exclusions were applied.
        try:
            _write_json(wf_exclusions_applied_report_json, wf_excl_report)
            scopes = (wf_excl_report.get("application") or {}).get("scopes") or {}
            for scope_name, scope_entry in scopes.items():
                by_source = (scope_entry or {}).get("by_source") or {}
                for src, src_entry in by_source.items():
                    n_req_units = int((src_entry or {}).get("n_units_with_exclusions_requested", 0) or 0)
                    n_matched_units = int((src_entry or {}).get("n_units_with_exclusions_matched", 0) or 0)
                    n_excl = int((src_entry or {}).get("waveforms_excluded_matched", 0) or 0)
                    if (n_req_units > 0) or (n_excl > 0):
                        logger.info(
                            "wf_exclusions applied (%s/%s): requested_units=%d matched_units=%d matched_excluded=%d",
                            str(scope_name),
                            str(src),
                            n_req_units,
                            n_matched_units,
                            n_excl,
                        )
        except Exception as e:
            logger.warning("Failed writing wf_exclusions applied report: %s", e)

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
                "wf_exclusions_applied_report_json": str(wf_exclusions_applied_report_json),
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
                "wf_exclusions_applied_report_json": str(wf_exclusions_applied_report_json),
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
