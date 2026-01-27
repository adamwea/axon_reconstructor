"""Footprinting plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .utils import (
    _get_unit_template_from_waveforms_with_exclusions,
    _infer_location_tolerance,
    _sparsity_unit_channel_indices,
    _try_get_electrode_ids,
)


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
        from ..waveforms.exclusions import normalize_unit_id

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
                    from ..waveforms.exclusions import update_wf_exclusion_report

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


__all__ = [
    "_layout_channel_key",
    "_build_global_channel_layout",
    "_write_merged_union_footprints_grid_pdf",
    "_write_unit_footprints_across_sources_pdf",
]
