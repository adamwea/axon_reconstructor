"""Templates plotting helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional


def _try_get_add_scalebar():
    """Best-effort import of the MEA_Analysis-style scalebar helper.

    Mirrors the waveforms stage behavior: try `scalebury`, then MEA_Analysis import,
    then fall back to a small inline implementation.
    """

    try:
        from scalebury import add_scalebar  # type: ignore[import-not-found]

        return add_scalebar
    except Exception:
        pass

    try:
        from MEA_Analysis.IPNAnalysis.scalebury import add_scalebar  # type: ignore[import-not-found]

        return add_scalebar
    except Exception:
        pass

    try:
        from matplotlib.offsetbox import AnchoredOffsetbox

        class AnchoredScaleBar(AnchoredOffsetbox):
            def __init__(
                self,
                transform,
                sizex=0,
                sizey=0,
                labelx=None,
                labely=None,
                loc=4,
                pad=0.1,
                borderpad=0.1,
                sep=2,
                prop=None,
                barcolor="black",
                barwidth=None,
                **kwargs,
            ):
                from matplotlib.patches import Rectangle
                from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea

                bars = AuxTransformBox(transform)
                if sizex:
                    bars.add_artist(Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
                if sizey:
                    bars.add_artist(Rectangle((0, 0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"))

                if sizex and labelx:
                    xlabel = TextArea(labelx)
                    bars = VPacker(children=[bars, xlabel], align="center", pad=0, sep=sep)
                if sizey and labely:
                    ylabel = TextArea(labely)
                    bars = HPacker(children=[ylabel, bars], align="center", pad=0, sep=sep)

                super().__init__(
                    loc,
                    pad=pad,
                    borderpad=borderpad,
                    child=bars,
                    prop=prop,
                    frameon=False,
                    **kwargs,
                )

        def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
            def f(axis):
                l = axis.get_majorticklocs()
                return len(l) > 1 and (l[1] - l[0])

            if matchx:
                kwargs["sizex"] = f(ax.xaxis)
                kwargs["labelx"] = str(kwargs["sizex"])
            if matchy:
                kwargs["sizey"] = f(ax.yaxis)
                kwargs["labely"] = str(kwargs["sizey"])

            sb = AnchoredScaleBar(ax.transData, **kwargs)
            ax.add_artist(sb)

            if hidex:
                ax.xaxis.set_visible(False)
            if hidey:
                ax.yaxis.set_visible(False)
            if hidex and hidey:
                ax.set_frame_on(False)
            return sb

        return add_scalebar
    except Exception:
        return None


# Maxwell MEA 3.85 x 2.10 mm active area (26400 electrodes) with 17.5 µm pitch.
# 3.85 mm / 17.5 µm = 220 columns; 2.10 mm / 17.5 µm = 120 rows.
CHIP_COLS = 220
CHIP_ROWS = 120
CHIP_PITCH_UM = 17.5


def _try_int_array(x: Any):
    try:
        import numpy as np  # type: ignore[import-not-found]

        if x is None:
            return None
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return arr.astype(int)
    except Exception:
        return None


def _electrode_ids_to_rowcol(*, electrode_ids: Any):
    import numpy as np  # type: ignore[import-not-found]

    eids = _try_int_array(electrode_ids)
    if eids is None:
        return None
    if eids.ndim != 1:
        eids = eids.ravel()
    if eids.size == 0:
        return None
    if int(np.min(eids)) < 0:
        return None
    if int(np.max(eids)) >= int(CHIP_COLS * CHIP_ROWS):
        return None
    row = (eids // int(CHIP_COLS)).astype(int)
    col = (eids % int(CHIP_COLS)).astype(int)
    return eids, row, col


def _render_full_chip_value_map(
    *,
    ax,
    values_by_electrode_id: dict[int, float],
    all_recorded_electrode_ids: Optional[Any],
    title: str,
    cmap: str,
    norm,
    quiet_rgba=(0.92, 0.92, 0.92, 1.0),
    noncontrib_rgba=(0.55, 0.55, 0.55, 1.0),
    show_axes: bool = False,
    cbar_label: Optional[str] = None,
    cbar_shrink: float = 0.85,
):
    """Render a full-chip (rows x cols) map with quiet/non-contrib/contrib coloring."""

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    rgba = np.empty((int(CHIP_ROWS), int(CHIP_COLS), 4), dtype=float)
    rgba[:] = np.asarray(quiet_rgba, dtype=float)

    rec_eids = _try_int_array(all_recorded_electrode_ids)
    if rec_eids is not None:
        rec_eids = rec_eids.ravel()
        ok = (rec_eids >= 0) & (rec_eids < int(CHIP_ROWS * CHIP_COLS))
        rec_eids = rec_eids[ok]
        rr = (rec_eids // int(CHIP_COLS)).astype(int)
        cc = (rec_eids % int(CHIP_COLS)).astype(int)
        rgba[rr, cc, :] = np.asarray(noncontrib_rgba, dtype=float)

    if values_by_electrode_id:
        eids = np.fromiter(values_by_electrode_id.keys(), dtype=int)
        vals = np.fromiter(values_by_electrode_id.values(), dtype=float)
        ok = (eids >= 0) & (eids < int(CHIP_ROWS * CHIP_COLS)) & np.isfinite(vals)
        eids = eids[ok]
        vals = vals[ok]
        if eids.size:
            rr = (eids // int(CHIP_COLS)).astype(int)
            cc = (eids % int(CHIP_COLS)).astype(int)
            cm = plt.get_cmap(cmap)
            rgba[rr, cc, :] = cm(norm(vals))

    extent = [
        -0.5 * float(CHIP_PITCH_UM),
        (float(CHIP_COLS) - 0.5) * float(CHIP_PITCH_UM),
        -0.5 * float(CHIP_PITCH_UM),
        (float(CHIP_ROWS) - 0.5) * float(CHIP_PITCH_UM),
    ]

    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    if show_axes:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    else:
        ax.set_axis_off()

    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=float(cbar_shrink))
    if cbar_label:
        cbar.set_label(cbar_label)
    return cbar


def _write_template_overlay(
    *,
    ax,
    template: Any,
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int,
    title: str,
    annotation_text: Optional[str] = None,
    line_color: str = "gray",
    add_scalebar=None,
) -> None:
    import numpy as np  # type: ignore[import-not-found]

    tmpl = np.asarray(template)
    if tmpl.ndim != 2 or tmpl.size == 0:
        ax.set_axis_off()
        return

    n_samples = tmpl.shape[0]

    # Time axis in ms (best effort).
    if ms_before is not None and ms_after is not None:
        # Use linspace so the full window maps nicely.
        t_ms = np.linspace(-float(ms_before), float(ms_after), n_samples, endpoint=False)
    else:
        t_ms = (np.arange(n_samples, dtype=float) / float(fs_hz)) * 1000.0

    ptp = np.ptp(tmpl, axis=0)
    order = np.argsort(ptp)[::-1]
    # If top_channels <= 0, plot all channels.
    if int(top_channels) <= 0:
        sel = order
    else:
        k = int(min(max(1, int(top_channels)), len(order)))
        sel = order[:k]

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for j in sel:
        y = tmpl[:, int(j)]
        ax.plot(t_ms, y, lw=0.6, alpha=0.35, c=str(line_color))

    ax.set_title(title, fontsize=10)

    if annotation_text:
        try:
            ax.text(
                0.98,
                0.14,
                annotation_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
            )
        except Exception:
            pass

    if add_scalebar is not None:
        try:
            add_scalebar(
                ax,
                matchx=False,
                matchy=False,
                sizex=1.0,
                labelx="1 ms",
                sizey=50,
                labely="50 µV",
                loc=4,
                hidex=True,
                hidey=True,
            )
        except Exception:
            pass


def _write_templates_grid_pdf(
    *,
    pdf_path: Path,
    unit_entries: list[dict[str, Any]],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int,
    logger,
) -> None:
    """Write a grid PDF of templates (one subplot per unit)."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf

    ncols = 3
    nrows = 4
    per_page = ncols * nrows

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf.PdfPages(pdf_path) as out:
        for i0 in range(0, len(unit_entries), per_page):
            chunk = unit_entries[i0 : i0 + per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5), constrained_layout=False)
            axes = axes.ravel().tolist()

            add_scalebar = _try_get_add_scalebar()

            for ax, entry in zip(axes, chunk):
                uid = entry.get("unit_id")
                title = f"unit {uid}"

                n_ch = entry.get("n_channels")
                if n_ch is None:
                    try:
                        n_ch = int(entry["template"].shape[1])
                    except Exception:
                        n_ch = None
                n_wf = entry.get("n_waveforms_sum")
                annotation = None
                if n_ch is not None or n_wf is not None:
                    parts = []
                    if n_ch is not None:
                        parts.append(f"nCh={int(n_ch)}")
                    if n_wf is not None:
                        parts.append(f"nWfSum={int(n_wf)}")
                    annotation = "\n".join(parts)

                _write_template_overlay(
                    ax=ax,
                    template=entry["template"],
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                    top_channels=int(top_channels),
                    title=title,
                    annotation_text=annotation,
                    line_color="gray",
                    add_scalebar=add_scalebar,
                )

            for j in range(len(chunk), len(axes)):
                axes[j].set_axis_off()

            try:
                fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.94, wspace=0.10, hspace=0.28)
            except Exception:
                pass
            out.savefig(fig, dpi=150)
            plt.close(fig)

    logger.info("Wrote templates grid PDF -> %s", pdf_path)


def _write_unit_segment_grids_pdf(
    *,
    pdf_path: Path,
    unit_id: Any,
    sources_for_unit: list[dict[str, Any]],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
) -> None:
    """Write per-unit concat-vs-segment waveform panels.

    This plots raw waveforms (no mean), using a single representative channel
    chosen across sources, to mimic waveforms-step QC formatting.
    """

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf

    if not sources_for_unit:
        return

    add_scalebar = _try_get_add_scalebar()

    wf_sources: list[tuple[str, Any, list[Any]]] = []
    for src in sources_for_unit:
        name = str(src.get("name"))
        an = src.get("_analyzer")
        if an is None:
            continue

        try:
            wf_ext = an.get_extension("waveforms") if an.has_extension("waveforms") else None
        except Exception:
            wf_ext = None
        if wf_ext is None:
            continue

        try:
            wf = wf_ext.get_waveforms_one_unit(unit_id=unit_id)
        except Exception:
            wf = None
        if wf is None or getattr(wf, "shape", (0,))[0] == 0:
            continue

        ch_ids = None
        try:
            sp = getattr(an, "sparsity", None)
            if sp is None:
                sp = getattr(wf_ext, "sparsity", None)
            if sp is not None:
                ch_ids = list(sp.unit_id_to_channel_ids.get(unit_id))
        except Exception:
            ch_ids = None
        if ch_ids is None:
            try:
                ch_ids = list(an.recording.get_channel_ids())
            except Exception:
                ch_ids = list(range(int(wf.shape[2]) if wf.ndim == 3 else 0))

        wf_sources.append((name, wf, ch_ids))

    if not wf_sources:
        return

    # Normalize channel id lists to match waveform channel dimension.
    norm_wf_sources: list[tuple[str, Any, list[Any]]] = []
    for name, wf, ch_ids in wf_sources:
        try:
            n_ch = int(wf.shape[2]) if getattr(wf, "ndim", 0) == 3 else 0
        except Exception:
            n_ch = 0
        if n_ch <= 0:
            continue

        if (ch_ids is None) or (len(ch_ids) != n_ch):
            ch_ids = list(range(n_ch))
        norm_wf_sources.append((name, wf, list(ch_ids)))

    if not norm_wf_sources:
        return

    # Choose a global channel that exists across the most sources.
    # Tie-breaker: strongest (most negative) mean peak across sources.
    channel_presence: dict[Any, int] = {}
    channel_best_score: dict[Any, float] = {}
    per_source_best: dict[str, Any] = {}

    for name, wf, ch_ids in norm_wf_sources:
        try:
            mean_wf = np.mean(wf, axis=0)  # (n_samples, n_channels)
        except Exception:
            continue

        seen = set(ch_ids)
        for ch in seen:
            channel_presence[ch] = channel_presence.get(ch, 0) + 1

        best_ch = None
        best_score = None
        for ch_idx, ch_id in enumerate(ch_ids):
            try:
                score = float(np.min(mean_wf[:, int(ch_idx)]))
            except Exception:
                continue

            # Track per-source best (for fallback).
            if best_score is None or score < best_score:
                best_score = score
                best_ch = ch_id

            # Track global best score per channel.
            prev = channel_best_score.get(ch_id)
            if prev is None or score < prev:
                channel_best_score[ch_id] = score

        if best_ch is not None:
            per_source_best[str(name)] = best_ch

    best_channel_id: Any | None = None
    best_presence: int = -1
    best_score: float | None = None
    for ch_id, presence in channel_presence.items():
        score = channel_best_score.get(ch_id)
        if score is None:
            continue
        if (presence > best_presence) or (presence == best_presence and (best_score is None or score < best_score)):
            best_channel_id = ch_id
            best_presence = int(presence)
            best_score = float(score)

    if best_channel_id is None:
        # Fall back to the first source's best.
        best_channel_id = per_source_best.get(str(norm_wf_sources[0][0]), norm_wf_sources[0][2][0])

    n = len(norm_wf_sources)
    if n == 1:
        ncols, nrows = 1, 1
        figsize = (11, 8.5)
    else:
        ncols = 3
        nrows = int(math.ceil(n / ncols))
        figsize = (11, 3.0 * nrows)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf.PdfPages(pdf_path) as out:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=False)
        axes_list = list(np.asarray(axes).ravel())

        for ax, (name, wf, ch_ids) in zip(axes_list, norm_wf_sources, strict=False):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            ch_to_idx = {cid: idx for idx, cid in enumerate(ch_ids)}
            ch_id_to_plot = best_channel_id
            if ch_id_to_plot not in ch_to_idx:
                # Fallback: still show something for this source.
                ch_id_to_plot = per_source_best.get(str(name), None)
            if ch_id_to_plot not in ch_to_idx:
                ax.set_axis_off()
                continue
            ch_idx = int(ch_to_idx[ch_id_to_plot])
            if wf.ndim != 3 or ch_idx >= wf.shape[2]:
                ax.set_axis_off()
                continue

            w2d = wf[:, :, ch_idx]
            time_ms = (np.arange(w2d.shape[1], dtype=float) / float(fs_hz)) * 1000.0

            ax.plot(time_ms, w2d.T, c="gray", lw=0.5, alpha=0.3)

            try:
                n_ch = len(set(ch_ids))
                n_wf = int(wf.shape[0])
                annotation_text = f"nCh={n_ch}\nnWf={n_wf}"
                ax.text(
                    0.98,
                    0.14,
                    annotation_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
                )
            except Exception:
                pass

            ax.set_title(f"{name} | unit {unit_id} | ch {ch_id_to_plot}", fontsize=10)

            if add_scalebar is not None:
                try:
                    add_scalebar(
                        ax,
                        matchx=False,
                        matchy=False,
                        sizex=1.0,
                        labelx="1 ms",
                        sizey=50,
                        labely="50 µV",
                        loc=4,
                        hidex=True,
                        hidey=True,
                    )
                except Exception:
                    pass

        for j in range(len(wf_sources), len(axes_list)):
            axes_list[j].set_axis_off()

        fig.suptitle(f"Unit segment grids (unit {unit_id})", fontsize=12)
        try:
            fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.93, wspace=0.12, hspace=0.40)
        except Exception:
            pass
        out.savefig(fig, dpi=150)
        plt.close(fig)


def _write_unit_templates_across_sources_pdf(
    *,
    pdf_path: Path,
    unit_id: Any,
    sources_for_unit: list[dict[str, Any]],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf

    n = len(sources_for_unit)
    if n <= 0:
        return

    if n == 1:
        ncols, nrows = 1, 1
        figsize = (11, 8.5)
    else:
        ncols = 3
        nrows = int(math.ceil(n / ncols))
        figsize = (11, 3.0 * nrows)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf.PdfPages(pdf_path) as out:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        import numpy as np  # type: ignore[import-not-found]

        axes_list = list(np.asarray(axes).ravel())

        for ax, src in zip(axes_list, sources_for_unit):
            _write_template_overlay(
                ax=ax,
                template=src["template"],
                fs_hz=float(fs_hz),
                ms_before=ms_before,
                ms_after=ms_after,
                top_channels=int(top_channels),
                title=str(src["name"]),
            )

        for j in range(len(sources_for_unit), len(axes_list)):
            axes_list[j].set_axis_off()

        fig.suptitle(f"Templates overlay (unit {unit_id})", fontsize=12)
        out.savefig(fig, dpi=150)
        plt.close(fig)


def _render_full_chip_value_map_no_colorbar(
    *,
    ax,
    values_by_electrode_id: dict[int, float],
    all_recorded_electrode_ids: Optional[Any],
    title: str,
    cmap: str,
    norm,
    quiet_rgba=(0.92, 0.92, 0.92, 1.0),
    noncontrib_rgba=(0.55, 0.55, 0.55, 1.0),
    show_axes: bool = False,
):
    """Render a full-chip map without creating a per-axis colorbar."""

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt

    rgba = np.empty((int(CHIP_ROWS), int(CHIP_COLS), 4), dtype=float)
    rgba[:] = np.asarray(quiet_rgba, dtype=float)

    rec_eids = _try_int_array(all_recorded_electrode_ids)
    if rec_eids is not None:
        rec_eids = rec_eids.ravel()
        ok = (rec_eids >= 0) & (rec_eids < int(CHIP_ROWS * CHIP_COLS))
        rec_eids = rec_eids[ok]
        rr = (rec_eids // int(CHIP_COLS)).astype(int)
        cc = (rec_eids % int(CHIP_COLS)).astype(int)
        rgba[rr, cc, :] = np.asarray(noncontrib_rgba, dtype=float)

    if values_by_electrode_id:
        eids = np.fromiter(values_by_electrode_id.keys(), dtype=int)
        vals = np.fromiter(values_by_electrode_id.values(), dtype=float)
        ok = (eids >= 0) & (eids < int(CHIP_ROWS * CHIP_COLS)) & np.isfinite(vals)
        eids = eids[ok]
        vals = vals[ok]
        if eids.size:
            rr = (eids // int(CHIP_COLS)).astype(int)
            cc = (eids % int(CHIP_COLS)).astype(int)
            cm = plt.get_cmap(cmap)
            rgba[rr, cc, :] = cm(norm(vals))

    extent = [
        -0.5 * float(CHIP_PITCH_UM),
        (float(CHIP_COLS) - 0.5) * float(CHIP_PITCH_UM),
        -0.5 * float(CHIP_PITCH_UM),
        (float(CHIP_ROWS) - 0.5) * float(CHIP_PITCH_UM),
    ]

    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    if show_axes:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    else:
        ax.set_axis_off()


def _write_unit_segment_footprint_grids_pdf(
    *,
    pdf_path: Path,
    unit_id: Any,
    sources_for_unit: list[dict[str, Any]],
    all_recorded_electrode_ids: Any = None,
    cmap: str = "viridis",
) -> None:
    """Write per-unit concat-vs-segment footprint PTP maps as a grid PDF."""

    import math
    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf

    if not sources_for_unit:
        return

    amps: list[np.ndarray] = []
    eids_list: list[Any] = []
    locs_list: list[Any] = []
    names: list[str] = []

    for src in sources_for_unit:
        try:
            tmpl = np.asarray(src.get("template"), dtype=float)
            if tmpl.ndim != 2 or tmpl.size == 0:
                continue
            amp = np.ptp(tmpl, axis=0).astype(float)
            amps.append(amp)
            eids_list.append(src.get("electrode_ids"))
            locs_list.append(src.get("channel_locations"))
            names.append(str(src.get("name")))
        except Exception:
            continue

    if not amps:
        return

    finite = np.concatenate([a[np.isfinite(a)] for a in amps if a is not None and np.asarray(a).size], axis=0)
    if finite.size:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if not (vmax > vmin):
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    import matplotlib.colors

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    n = len(amps)
    if n == 1:
        ncols, nrows = 1, 1
        figsize = (11, 8.5)
    else:
        ncols = 3
        nrows = int(math.ceil(n / ncols))
        figsize = (11, 3.2 * nrows)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf.PdfPages(pdf_path) as out:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_list = list(np.asarray(axes).ravel())

        for ax, name, amp, eids, locs in zip(axes_list, names, amps, eids_list, locs_list):
            e_trip = _electrode_ids_to_rowcol(electrode_ids=eids)
            if e_trip is not None and amp.ndim == 1 and int(amp.size) == int(e_trip[0].size):
                values = {int(e): float(v) for e, v in zip(e_trip[0].tolist(), amp.tolist()) if np.isfinite(v)}
                _render_full_chip_value_map_no_colorbar(
                    ax=ax,
                    values_by_electrode_id=values,
                    all_recorded_electrode_ids=all_recorded_electrode_ids,
                    title=str(name),
                    cmap=cmap,
                    norm=norm,
                )
            else:
                # Fallback scatter in local coordinates.
                try:
                    locs_xy = np.asarray(locs, dtype=float)
                    if locs_xy.ndim == 2 and locs_xy.shape[1] >= 2 and int(locs_xy.shape[0]) == int(amp.shape[0]):
                        ax.set_title(str(name), fontsize=10)
                        ax.set_aspect("equal", adjustable="box")
                        ax.set_axis_off()
                        sc = ax.scatter(
                            locs_xy[:, 0],
                            locs_xy[:, 1],
                            c=np.asarray(amp, dtype=float),
                            s=18,
                            cmap=cmap,
                            norm=norm,
                            marker="s",
                            linewidths=0,
                        )
                except Exception:
                    ax.set_axis_off()

        for j in range(len(amps), len(axes_list)):
            axes_list[j].set_axis_off()

        fig.suptitle(f"Segment footprints (unit {unit_id})", fontsize=12)
        try:
            fig.subplots_adjust(left=0.03, right=0.93, bottom=0.03, top=0.92, wspace=0.08, hspace=0.22)
        except Exception:
            pass

        # Shared colorbar.
        try:
            import matplotlib.cm as cm
            from matplotlib.cm import ScalarMappable

            sm = ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap))
            sm.set_array([])
            cax = fig.add_axes([0.94, 0.12, 0.015, 0.70])
            cb = fig.colorbar(sm, cax=cax)
            cb.set_label("PTP (µV)")
        except Exception:
            pass

        out.savefig(fig, dpi=150)
        plt.close(fig)


def _write_topo_unit_footprint_png(
    *,
    out_path: Path,
    unit_id: Any,
    full_template: Any,
    full_electrode_ids: Any,
    all_recorded_electrode_ids: Any,
    title: str,
    overlap_resolved_line: Optional[str] = None,
    zoom_electrode_ids: Optional[Any] = None,
    zoom_pad_um: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    """Write a 3D full-chip topographical footprint plot.

    - Z is linear PTP amplitude.
    - Colors are log-scaled to emphasize subtle differences.
    - X/Y aspect reflects physical chip dimensions.
    """

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return

    tmpl = np.asarray(full_template, dtype=float)
    if tmpl.ndim != 2 or tmpl.size == 0:
        return
    amp = np.ptp(tmpl, axis=0).astype(float)

    e_trip = _electrode_ids_to_rowcol(electrode_ids=full_electrode_ids)
    if e_trip is None or amp.ndim != 1 or int(amp.size) != int(e_trip[0].size):
        return

    quiet_rgba = (0.92, 0.92, 0.92, 1.0)
    noncontrib_rgba = (0.55, 0.55, 0.55, 1.0)
    # Give non-contributing recording electrodes a small height so their "bars" are visible.
    noncontrib_height_uv = 1.0

    # Build a full-chip Z grid (rows x cols) using electrode ids.
    # - quiet electrodes (not in recording) are NaN
    # - non-contributing recording electrodes are set to `noncontrib_height_uv`
    # - contributing electrodes are set to their linear PTP amplitude
    Z = np.full((int(CHIP_ROWS), int(CHIP_COLS)), np.nan, dtype=float)

    # Recording mask.
    rec_eids = _try_int_array(all_recorded_electrode_ids)
    rec_mask = np.zeros((int(CHIP_ROWS), int(CHIP_COLS)), dtype=bool)
    if rec_eids is not None:
        rec_eids = rec_eids.ravel()
        ok_rec = (rec_eids >= 0) & (rec_eids < int(CHIP_ROWS * CHIP_COLS))
        rec_eids = rec_eids[ok_rec]
        rr_rec = (rec_eids // int(CHIP_COLS)).astype(int)
        cc_rec = (rec_eids % int(CHIP_COLS)).astype(int)
        rec_mask[rr_rec, cc_rec] = True

    # Default recording electrodes to non-contributing height.
    Z[rec_mask] = float(noncontrib_height_uv)

    # Fill contributing electrodes with their PTP values.
    eids = e_trip[0]
    rr = e_trip[1]
    cc = e_trip[2]
    ok_amp = np.isfinite(amp) & (amp > 0)
    try:
        Z[rr[ok_amp], cc[ok_amp]] = amp[ok_amp]
    except Exception:
        for r, c, v in zip(rr.tolist(), cc.tolist(), amp.tolist()):
            try:
                if np.isfinite(v) and float(v) > 0:
                    Z[int(r), int(c)] = float(v)
            except Exception:
                pass

    # 3D surface coordinates in microns.
    xs = (np.arange(int(CHIP_COLS), dtype=float) * float(CHIP_PITCH_UM))
    ys = (np.arange(int(CHIP_ROWS), dtype=float) * float(CHIP_PITCH_UM))
    X, Y = np.meshgrid(xs, ys)

    # Facecolors use log scale on positive values.
    import matplotlib.colors
    from matplotlib.cm import ScalarMappable

    pos = Z[np.isfinite(Z) & (Z > float(noncontrib_height_uv))]
    if pos.size:
        vmin = max(float(np.nanmin(pos)), 1e-9)
        vmax = float(np.nanmax(pos))
        if not (vmax > vmin):
            vmax = vmin * 10.0
    else:
        vmin, vmax = 1e-9, 1.0
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)

    # plot_surface expects facecolors shape (rows-1, cols-1, 4)
    Zc = 0.25 * (Z[:-1, :-1] + Z[1:, :-1] + Z[:-1, 1:] + Z[1:, 1:])
    facecolors = np.empty((Zc.shape[0], Zc.shape[1], 4), dtype=float)
    facecolors[:] = np.array(quiet_rgba, dtype=float)  # quiet

    # Cell masks based on corner membership.
    rec_cell = (rec_mask[:-1, :-1] | rec_mask[1:, :-1] | rec_mask[:-1, 1:] | rec_mask[1:, 1:])
    # Contributing if any corner has amplitude > noncontrib floor.
    contrib_corner = np.isfinite(Z) & (Z > float(noncontrib_height_uv))
    contrib_cell = (
        contrib_corner[:-1, :-1]
        | contrib_corner[1:, :-1]
        | contrib_corner[:-1, 1:]
        | contrib_corner[1:, 1:]
    )
    noncontrib_cell = rec_cell & (~contrib_cell)

    # Noncontributing: darker gray.
    facecolors[noncontrib_cell] = np.array(noncontrib_rgba, dtype=float)

    # Contributing: colormap on log-scaled colors.
    okc = contrib_cell & np.isfinite(Zc) & (Zc > float(noncontrib_height_uv))
    try:
        facecolors[okc] = cm(norm(Zc[okc]))
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        np.nan_to_num(Z, nan=0.0),
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("PTP (µV)")
    try:
        ax.view_init(elev=32, azim=-55)
    except Exception:
        pass

    # Optional zoom: constrain x/y limits to the contributing electrode region.
    # This is intentionally done at *render time* (templates stage), so analysis only
    # loads a pre-rendered zoomed topo instead of post-hoc cropping.
    try:
        if zoom_electrode_ids is not None:
            z_trip = _electrode_ids_to_rowcol(electrode_ids=zoom_electrode_ids)
            if z_trip is not None:
                z_rr = z_trip[1]
                z_cc = z_trip[2]
                if int(z_rr.size) > 0 and int(z_cc.size) > 0:
                    pad_um = float(zoom_pad_um) if zoom_pad_um is not None else 4.0 * float(CHIP_PITCH_UM)

                    x_full_max = float((int(CHIP_COLS) - 1) * float(CHIP_PITCH_UM))
                    y_full_max = float((int(CHIP_ROWS) - 1) * float(CHIP_PITCH_UM))

                    x0 = float(np.min(z_cc)) * float(CHIP_PITCH_UM)
                    x1 = float(np.max(z_cc)) * float(CHIP_PITCH_UM)
                    y0 = float(np.min(z_rr)) * float(CHIP_PITCH_UM)
                    y1 = float(np.max(z_rr)) * float(CHIP_PITCH_UM)

                    ax.set_xlim(max(0.0, x0 - pad_um), min(x_full_max, x1 + pad_um))
                    ax.set_ylim(max(0.0, y0 - pad_um), min(y_full_max, y1 + pad_um))
    except Exception:
        pass

    # Add contributing/noncontrib counts.
    try:
        n_recording = int(np.sum(rec_mask)) if np.any(rec_mask) else int(CHIP_ROWS) * int(CHIP_COLS)
        n_contrib = int(np.sum(np.isfinite(Z) & (Z > float(noncontrib_height_uv))))
        n_noncontrib = int(max(0, n_recording - n_contrib))

        overlap_line = str(overlap_resolved_line).strip() if overlap_resolved_line is not None else ""
        if not overlap_line:
            overlap_line = "overlap-resolved: none"

        ax.text2D(
            0.02,
            0.98,
            "\n".join(
                [
                    f"recording={n_recording}",
                    f"contrib={n_contrib}",
                    f"noncontrib={n_noncontrib}",
                    f"noncontrib floor={noncontrib_height_uv:g} µV",
                    overlap_line,
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
        )
    except Exception:
        pass

    # Respect physical aspect on x/y; make z visually taller so structure is visible.
    try:
        x0_lim, x1_lim = ax.get_xlim3d()
        y0_lim, y1_lim = ax.get_ylim3d()
        x_range = float(abs(x1_lim - x0_lim))
        y_range = float(abs(y1_lim - y0_lim))
        z_max = float(np.nanmax(Z)) if np.isfinite(np.nanmax(Z)) else 1.0
        z_max = max(z_max, 1.0)
        ax.set_zlim(0.0, z_max * 1.05)
        ax.set_box_aspect((x_range, y_range, 0.35 * max(x_range, y_range)))
    except Exception:
        pass

    # Shared colorbar in log-scale.
    try:
        sm = ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.08)
        cb.set_label("PTP (µV) [log colors]")
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_multichannel_overlay(
    *,
    ax,
    waveform: Any,
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    channel_labels: Optional[list[str]] = None,
    line_color: str = "gray",
    add_scalebar=None,
):
    import numpy as np  # type: ignore[import-not-found]

    wf = np.asarray(waveform, dtype=float)
    if wf.ndim != 2 or wf.size == 0:
        return

    n_samples = int(wf.shape[0])
    n_ch = int(wf.shape[1])

    # Time axis (ms).
    if ms_before is not None and ms_after is not None and (ms_before + ms_after) > 0 and n_samples > 1:
        t_ms = np.linspace(-float(ms_before), float(ms_after), n_samples)
    else:
        t_ms = (np.arange(n_samples, dtype=float) / float(fs_hz)) * 1000.0

    # Vertical offsets.
    try:
        scale = float(np.nanmax(np.ptp(wf, axis=0)))
        if not (scale > 0):
            scale = 1.0
    except Exception:
        scale = 1.0
    spacing = 1.4 * scale

    for j in range(n_ch):
        y = wf[:, j] + float(j) * spacing
        ax.plot(t_ms, y, color=line_color, linewidth=0.8)

    ax.set_axis_off()

    if channel_labels is not None and len(channel_labels) == n_ch:
        try:
            ax.text(
                0.02,
                0.98,
                "\n".join([str(x) for x in channel_labels]),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                color="black",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.65, linewidth=0.0),
            )
        except Exception:
            pass

    if add_scalebar is not None:
        try:
            add_scalebar(ax=ax, units="µV", fontsize=7)
        except Exception:
            pass


def _write_unit_propagation_plots_png(
    *,
    out_dir: Path,
    unit_id: Any,
    merged_contributing: dict[str, Any],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int = 10,
    n_waveforms: int = 12,
    channels_per_panel: int = 25,
    channel_overlap: int = 5,
    show_electrode_ids: bool = False,
    trace_gain: float = 1.0,
    trace_spacing: float = 1.0,
    ap_timings_json_path: Optional[Path] = None,
    logger=None,
) -> None:
    """Write a propagation plot PNG via `axon_velocity`.

    No legacy fallback: if axon_velocity is unavailable or inputs are invalid, return.

    Optional cosmetics (implemented post-render, without editing axon_velocity):
    - `trace_gain`: scale waveform amplitude about each trace baseline
    - `trace_spacing`: scale vertical spacing between traces (smaller -> more overlap)
    - `show_electrode_ids`: annotate electrode ids left of each trace
    """

    _ = n_waveforms
    _ = channels_per_panel
    _ = channel_overlap
    _ = ap_timings_json_path

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    try:
        import axon_velocity.plotting as av_plot  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        tmpl = np.asarray(merged_contributing.get("template"), dtype=float)
    except Exception:
        return
    if tmpl.ndim != 2 or tmpl.size == 0:
        return

    n_samp = int(tmpl.shape[0])
    n_ch_total = int(tmpl.shape[1])
    if n_samp <= 1 or n_ch_total <= 0:
        return

    locs_xy = merged_contributing.get("channel_locations")
    if locs_xy is None:
        return
    try:
        locs_xy = np.asarray(locs_xy, dtype=float)
    except Exception:
        return
    if locs_xy.ndim != 2 or int(locs_xy.shape[0]) != int(n_ch_total) or int(locs_xy.shape[1]) < 2:
        return

    try:
        amp = np.ptp(tmpl, axis=0).astype(float)
    except Exception:
        return
    if np.asarray(amp).ndim != 1:
        return

    top_n = int(max(1, min(int(top_channels), int(n_ch_total))))
    by_amp = np.argsort(-np.asarray(amp, dtype=float))
    selected = [int(i) for i in by_amp[:top_n].tolist()]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"unit_{unit_id}.png"

    out_svg = out_dir / f"unit_{unit_id}.svg"
    fig_w = 5.0
    fig_h = float(min(26.0, max(8.0, 0.60 * float(len(selected)) + 3.0)))
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)

    av_plot.plot_template_propagation(
        tmpl.T,
        locs_xy[:, :2],
        selected,
        sort_templates=True,
        color="black",
        color_marker="black",
        ax=ax,
    )

    trace_gain_f = float(trace_gain) if trace_gain is not None else 1.0
    trace_spacing_f = float(trace_spacing) if trace_spacing is not None else 1.0
    if not (trace_gain_f > 0):
        trace_gain_f = 1.0
    if not (trace_spacing_f > 0):
        trace_spacing_f = 1.0

    selected_sorted: list[int] = []
    new_spacing: Optional[float] = None
    old_spacing: Optional[float] = None

    # Move both wave lines and peak-dot lines after applying gain/spacing.
    try:
        template_selected = np.asarray(tmpl.T, dtype=float)[np.asarray(selected, dtype=int), :]
        peaks = np.argmin(template_selected, axis=1)
        sort_idx = np.argsort(peaks)
        selected_sorted = [int(selected[int(i)]) for i in sort_idx.tolist()]

        template_sorted = template_selected[sort_idx]
        ptp_glob = float(np.max(np.ptp(template_sorted, axis=1))) if template_sorted.size else 1.0
        if not (ptp_glob > 0):
            ptp_glob = 1.0

        old_spacing = 1.5 * float(ptp_glob)
        new_spacing = float(old_spacing) * float(trace_spacing_f)

        lines = list(getattr(ax, "lines", []))
        n_traces = int(template_sorted.shape[0])
        expected = 2 * n_traces
        ok_pairing = (n_traces > 0) and (len(lines) >= expected)
        if ok_pairing:
            for i in range(n_traces):
                if str(lines[2 * i + 1].get_marker()) != "o":
                    ok_pairing = False
                    break

        if ok_pairing:
            for i in range(n_traces):
                base_old = float(i) * float(old_spacing)
                base_new = float(i) * float(new_spacing)
                for ln in (lines[2 * i], lines[2 * i + 1]):
                    yd = np.asarray(ln.get_ydata(orig=False), dtype=float)
                    if yd.size:
                        ln.set_ydata((yd - base_old) * float(trace_gain_f) + base_new)

        # Expand y-limits so large gain doesn't clip the bottom/top.
        ymins: list[float] = []
        ymaxs: list[float] = []
        for ln in list(getattr(ax, "lines", [])):
            yd = np.asarray(ln.get_ydata(orig=False), dtype=float)
            if yd.size == 0 or not np.isfinite(yd).any():
                continue
            ymins.append(float(np.nanmin(yd)))
            ymaxs.append(float(np.nanmax(yd)))
        if ymins and ymaxs and old_spacing is not None:
            ymin = float(min(ymins))
            ymax = float(max(ymaxs))
            pad = max(0.15 * float(old_spacing) * float(trace_gain_f), 0.05 * (ymax - ymin))
            if np.isfinite(ymin) and np.isfinite(ymax) and (ymax > ymin):
                ax.set_ylim(ymin - pad, ymax + pad)
    except Exception:
        selected_sorted = []
        new_spacing = None

    if bool(show_electrode_ids):
        try:
            from matplotlib.transforms import blended_transform_factory

            electrode_ids = merged_contributing.get("electrode_ids")
            channel_ids = merged_contributing.get("channel_ids")

            labels_all: list[str] = []
            for ch_idx in range(n_ch_total):
                lab = None
                if electrode_ids is not None:
                    try:
                        e = list(electrode_ids)[int(ch_idx)]
                        if e is not None:
                            lab = f"e{int(e)}"
                    except Exception:
                        lab = None
                if lab is None and channel_ids is not None:
                    try:
                        c = list(channel_ids)[int(ch_idx)]
                        if c is not None:
                            lab = str(c)
                    except Exception:
                        lab = None
                if lab is None:
                    lab = str(int(ch_idx))
                labels_all.append(str(lab))

            if not selected_sorted:
                template_selected = np.asarray(tmpl.T, dtype=float)[np.asarray(selected, dtype=int), :]
                peaks = np.argmin(template_selected, axis=1)
                sort_idx = np.argsort(peaks)
                selected_sorted = [int(selected[int(i)]) for i in sort_idx.tolist()]

            if new_spacing is None:
                template_selected = np.asarray(tmpl.T, dtype=float)[np.asarray(selected, dtype=int), :]
                template_sorted = template_selected[np.argsort(np.argmin(template_selected, axis=1))]
                ptp_glob = float(np.max(np.ptp(template_sorted, axis=1))) if template_sorted.size else 1.0
                if not (ptp_glob > 0):
                    ptp_glob = 1.0
                new_spacing = 1.5 * float(ptp_glob) * float(trace_spacing_f)

            trans = blended_transform_factory(ax.transAxes, ax.transData)
            x_ax = -0.01
            for i, ch_idx in enumerate(selected_sorted):
                ax.text(
                    x_ax,
                    float(i) * float(new_spacing),
                    labels_all[int(ch_idx)],
                    transform=trans,
                    ha="right",
                    va="center",
                    fontsize=12,
                    color="black",
                    clip_on=False,
                )
        except Exception:
            pass

    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.02)
    try:
        fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.02)
    except Exception:
        pass
    plt.close(fig)
    return


def _write_unit_propagation_plots_pdf(
    *,
    pdf_path: Path,
    unit_id: Any,
    merged_contributing: dict[str, Any],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int = 10,
    n_waveforms: int = 12,
    channels_per_panel: int = 25,
    channel_overlap: int = 5,
    show_electrode_ids: bool = False,
    trace_gain: float = 1.0,
    trace_spacing: float = 1.0,
    ap_timings_json_path: Optional[Path] = None,
    logger=None,
) -> None:
    """Backward-compat wrapper. Prefer `_write_unit_propagation_plots_png` for iteration."""

    _ = pdf_path
    _ = n_waveforms
    out_dir = Path(pdf_path).parent
    _write_unit_propagation_plots_png(
        out_dir=out_dir,
        unit_id=unit_id,
        merged_contributing=merged_contributing,
        fs_hz=fs_hz,
        ms_before=ms_before,
        ms_after=ms_after,
        top_channels=top_channels,
        n_waveforms=n_waveforms,
        channels_per_panel=channels_per_panel,
        channel_overlap=channel_overlap,
        show_electrode_ids=bool(show_electrode_ids),
        trace_gain=float(trace_gain),
        trace_spacing=float(trace_spacing),
        ap_timings_json_path=ap_timings_json_path,
        logger=logger,
    )


def _write_footprint_ptp_map(
    *,
    out_path: Path,
    channel_locations_xy: Any,
    footprint_ptp: Any,
    title: str = "",
    log_scale: bool,
    cmap: str = "viridis",
    electrode_ids: Any = None,
    all_recorded_electrode_ids: Any = None,
    zoom: bool = False,
    zoom_pad_um: float = 200.0,
    zoom_pad_frac: float = 0.10,
) -> None:
    """Write a footprint PTP amplitude map as a single image.

    This is intentionally lightweight and does not depend on axon_velocity.
    """

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm

    amp = np.asarray(footprint_ptp, dtype=float)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer full-chip rendering when electrode ids are available.
    e_trip = _electrode_ids_to_rowcol(electrode_ids=electrode_ids)
    if e_trip is not None and amp.ndim == 1 and int(amp.size) == int(e_trip[0].size):
        fig = plt.figure(figsize=(6.2, 3.9))
        ax = fig.add_subplot(111)

        # Norm for colormap.
        norm = None
        if bool(log_scale):
            try:
                pos = amp[np.isfinite(amp) & (amp > 0)]
                if pos.size:
                    vmin = float(np.nanmin(pos))
                    vmax = float(np.nanmax(pos))
                    vmin = max(vmin, 1e-9)
                    if vmax > vmin:
                        norm = LogNorm(vmin=vmin, vmax=vmax)
            except Exception:
                norm = None
        if norm is None:
            try:
                finite = amp[np.isfinite(amp)]
                vmin = float(np.nanmin(finite)) if finite.size else 0.0
                vmax = float(np.nanmax(finite)) if finite.size else 1.0
                if not (vmax > vmin):
                    vmax = vmin + 1.0
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            except Exception:
                norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

        values = {int(e): float(v) for e, v in zip(e_trip[0].tolist(), amp.tolist()) if np.isfinite(v)}
        _render_full_chip_value_map(
            ax=ax,
            values_by_electrode_id=values,
            all_recorded_electrode_ids=all_recorded_electrode_ids,
            title=(title or ""),
            cmap=cmap,
            norm=norm,
            cbar_label="PTP (µV)",
        )

        if bool(zoom):
            try:
                # Convert contributing electrode ids to µm coordinates.
                cols = e_trip[2].astype(float)
                rows = e_trip[1].astype(float)
                xs = cols * float(CHIP_PITCH_UM)
                ys = rows * float(CHIP_PITCH_UM)

                xmin, xmax = float(np.min(xs)), float(np.max(xs))
                ymin, ymax = float(np.min(ys)), float(np.max(ys))
                dx = max(xmax - xmin, 0.0)
                dy = max(ymax - ymin, 0.0)
                pad_x = max(float(zoom_pad_um), float(zoom_pad_frac) * dx)
                pad_y = max(float(zoom_pad_um), float(zoom_pad_frac) * dy)
                ax.set_xlim(xmin - pad_x, xmax + pad_x)
                ax.set_ylim(ymin - pad_y, ymax + pad_y)
            except Exception:
                pass
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        try:
            fig.savefig(out_path.with_suffix(".svg"), format="svg")
        except Exception:
            pass
        plt.close(fig)
        return

    # Fallback scatter (legacy behavior).
    locs = np.asarray(channel_locations_xy, dtype=float)
    if locs.ndim != 2 or locs.shape[1] < 2 or amp.ndim != 1 or locs.shape[0] != amp.shape[0]:
        return

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    if title:
        ax.set_title(str(title), fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    norm = None
    if bool(log_scale):
        try:
            pos = amp[np.isfinite(amp) & (amp > 0)]
            if pos.size:
                vmin = float(np.nanmin(pos))
                vmax = float(np.nanmax(pos))
                vmin = max(vmin, 1e-9)
                if vmax > vmin:
                    norm = LogNorm(vmin=vmin, vmax=vmax)
        except Exception:
            norm = None

    sc = ax.scatter(locs[:, 0], locs[:, 1], c=amp, s=18, cmap=cmap, norm=norm, marker="s", linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="PTP")

    if bool(zoom):
        try:
            good = np.isfinite(amp) & (amp > 0)
            xs = locs[good, 0]
            ys = locs[good, 1]
            if xs.size and ys.size:
                xmin, xmax = float(np.min(xs)), float(np.max(xs))
                ymin, ymax = float(np.min(ys)), float(np.max(ys))
                dx = max(xmax - xmin, 0.0)
                dy = max(ymax - ymin, 0.0)
                pad_x = max(float(zoom_pad_um), float(zoom_pad_frac) * dx)
                pad_y = max(float(zoom_pad_um), float(zoom_pad_frac) * dy)
                ax.set_xlim(xmin - pad_x, xmax + pad_x)
                ax.set_ylim(ymin - pad_y, ymax + pad_y)
        except Exception:
            pass
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    try:
        fig.savefig(out_path.with_suffix(".svg"), format="svg")
    except Exception:
        pass
    plt.close(fig)


def _write_unit_template_and_footprint_svg(
    *,
    out_path: Path,
    unit_id: Any,
    template: Any,
    channel_locations_xy: Any,
    electrode_ids: Any = None,
    all_recorded_electrode_ids: Any = None,
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int,
    log_footprint: bool,
) -> None:
    """Write a per-unit vector panel combining footprint + template overlay."""

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    tmpl = np.asarray(template, dtype=float)
    locs = np.asarray(channel_locations_xy, dtype=float)
    if tmpl.ndim != 2 or locs.ndim != 2 or locs.shape[0] != tmpl.shape[1]:
        return

    amp = np.ptp(tmpl, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    # Footprint map.
    try:
        from matplotlib.colors import LogNorm

        norm = None
        if bool(log_footprint):
            pos = amp[np.isfinite(amp) & (amp > 0)]
            if pos.size:
                vmin = float(np.nanmin(pos))
                vmax = float(np.nanmax(pos))
                vmin = max(vmin, 1e-9)
                if vmax > vmin:
                    norm = LogNorm(vmin=vmin, vmax=vmax)
        if norm is None:
            finite = amp[np.isfinite(amp)]
            vmin = float(np.nanmin(finite)) if finite.size else 0.0
            vmax = float(np.nanmax(finite)) if finite.size else 1.0
            if not (vmax > vmin):
                vmax = vmin + 1.0
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        e_trip = _electrode_ids_to_rowcol(electrode_ids=electrode_ids)
        if e_trip is not None and int(e_trip[0].size) == int(amp.size):
            values = {int(e): float(v) for e, v in zip(e_trip[0].tolist(), amp.tolist()) if np.isfinite(v)}
            _render_full_chip_value_map(
                ax=ax0,
                values_by_electrode_id=values,
                all_recorded_electrode_ids=all_recorded_electrode_ids,
                title="Footprint (PTP)" + (" [log]" if log_footprint else ""),
                cmap="viridis",
                norm=norm,
                cbar_label="PTP (µV)",
            )
        else:
            sc = ax0.scatter(locs[:, 0], locs[:, 1], c=amp, s=18, cmap="viridis", norm=norm, marker="s", linewidths=0)
            ax0.set_title("Footprint (PTP)" + (" [log]" if log_footprint else ""), fontsize=10)
            ax0.set_aspect("equal", adjustable="box")
            ax0.set_xlabel("x")
            ax0.set_ylabel("y")
            fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.04)
    except Exception:
        ax0.set_axis_off()

    # Template overlay.
    _write_template_overlay(
        ax=ax1,
        template=tmpl,
        fs_hz=float(fs_hz),
        ms_before=ms_before,
        ms_after=ms_after,
        top_channels=int(top_channels),
        title=f"Unit {unit_id} template (top {top_channels})",
    )

    fig.suptitle(f"Unit {unit_id}", fontsize=12)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _write_full_chip_template_amplitude_map_png(
    *,
    out_path: Path,
    template_ch_by_t: Any,
    electrode_ids: Any,
    all_recorded_electrode_ids: Any,
    title: str,
    cmap: str = "viridis",
) -> None:
    """Template amplitude (max |uV|) map rendered over the full chip grid."""

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    tmp = np.asarray(template_ch_by_t, dtype=float)
    if tmp.ndim != 2 or tmp.size == 0:
        return

    e_trip = _electrode_ids_to_rowcol(electrode_ids=electrode_ids)
    if e_trip is None or int(e_trip[0].size) != int(tmp.shape[0]):
        return

    amp = np.max(np.abs(tmp), axis=1).astype(float)
    finite = amp[np.isfinite(amp)]
    if finite.size:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if not (vmax > vmin):
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    values = {int(e): float(v) for e, v in zip(e_trip[0].tolist(), amp.tolist()) if np.isfinite(v)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.2, 3.9))
    ax = fig.add_subplot(111)
    _render_full_chip_value_map(
        ax=ax,
        values_by_electrode_id=values,
        all_recorded_electrode_ids=all_recorded_electrode_ids,
        title=title,
        cmap=cmap,
        norm=norm,
        cbar_label="Amplitude (µV)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_full_chip_template_peak_latency_map_png(
    *,
    out_path: Path,
    template_ch_by_t: Any,
    electrode_ids: Any,
    all_recorded_electrode_ids: Any,
    sampling_frequency_hz: float,
    title: str,
    cmap: str = "viridis",
) -> None:
    """Template peak-latency map rendered over the full chip grid."""

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    tmp = np.asarray(template_ch_by_t, dtype=float)
    if tmp.ndim != 2 or tmp.size == 0:
        return

    e_trip = _electrode_ids_to_rowcol(electrode_ids=electrode_ids)
    if e_trip is None or int(e_trip[0].size) != int(tmp.shape[0]):
        return

    fs = float(sampling_frequency_hz)
    if not (fs > 0):
        return

    # Peak latency definition: argmin index -> ms.
    lat_ms = (np.argmin(tmp, axis=1).astype(float) / fs) * 1000.0
    finite = lat_ms[np.isfinite(lat_ms)]
    if finite.size:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if not (vmax > vmin):
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    values = {int(e): float(v) for e, v in zip(e_trip[0].tolist(), lat_ms.tolist()) if np.isfinite(v)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.2, 3.9))
    ax = fig.add_subplot(111)
    _render_full_chip_value_map(
        ax=ax,
        values_by_electrode_id=values,
        all_recorded_electrode_ids=all_recorded_electrode_ids,
        title=title,
        cmap=cmap,
        norm=norm,
        cbar_label="Latency (ms)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


__all__ = [
    "_write_templates_grid_pdf",
    "_write_unit_segment_grids_pdf",
    "_write_unit_segment_footprint_grids_pdf",
    "_write_footprint_ptp_map",
    "_write_unit_template_and_footprint_svg",
    "_write_topo_unit_footprint_png",
    "_write_unit_propagation_plots_pdf",
    "_write_full_chip_template_amplitude_map_png",
    "_write_full_chip_template_peak_latency_map_png",
]
