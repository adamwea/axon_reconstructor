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
    recording_electrode_ids: Optional[Any],
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

    rec_eids = _try_int_array(recording_electrode_ids)
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
    recording_electrode_ids: Optional[Any],
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

    rec_eids = _try_int_array(recording_electrode_ids)
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
    recording_electrode_ids: Any = None,
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
                    recording_electrode_ids=recording_electrode_ids,
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
    recording_electrode_ids: Any,
    title: str,
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
    rec_eids = _try_int_array(recording_electrode_ids)
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

    # Add contributing/noncontrib counts.
    try:
        n_recording = int(np.sum(rec_mask)) if np.any(rec_mask) else int(CHIP_ROWS) * int(CHIP_COLS)
        n_contrib = int(np.sum(np.isfinite(Z) & (Z > float(noncontrib_height_uv))))
        n_noncontrib = int(max(0, n_recording - n_contrib))
        ax.text2D(
            0.02,
            0.98,
            f"recording={n_recording}\ncontrib={n_contrib}\nnoncontrib={n_noncontrib}\nnoncontrib floor={noncontrib_height_uv:g} µV",
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
        x_range = float((int(CHIP_COLS) - 1) * float(CHIP_PITCH_UM))
        y_range = float((int(CHIP_ROWS) - 1) * float(CHIP_PITCH_UM))
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
    merged_union: dict[str, Any],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int = 10,
    n_waveforms: int = 12,
    channels_per_panel: int = 25,
    channel_overlap: int = 5,
    ap_timings_json_path: Optional[Path] = None,
    logger=None,
) -> None:
    """Write propagation plots derived from the merged_union template as PNG(s).

    This is intentionally *template-based* (one trace per channel) so we don't need to
    re-extract or load waveforms/spike trains for plotting.

    The plot shows the top-N channels by PTP (from the merged_union template), ordered by
    best-effort timing estimates. A small triangle marks the negative-peak time for each channel.
    """

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    # PNG output only (faster iteration than multipage PDFs).

    _ = n_waveforms  # deprecated: kept for API/backwards-compatibility

    try:
        tmpl = np.asarray(merged_union.get("template"), dtype=float)
    except Exception:
        tmpl = None
    if tmpl is None or getattr(tmpl, "ndim", 0) != 2 or tmpl.size == 0:
        return

    n_samp = int(tmpl.shape[0])
    n_ch_total = int(tmpl.shape[1])
    if n_samp <= 1 or n_ch_total <= 0:
        return

    # Time axis (ms) for plotting.
    if ms_before is not None and ms_after is not None and (ms_before + ms_after) > 0 and n_samp > 1:
        t_ms = np.linspace(-float(ms_before), float(ms_after), n_samp)
    else:
        t_ms = (np.arange(n_samp, dtype=float) / float(fs_hz)) * 1000.0

    # Channel IDs for labeling (prefer electrode ids).
    electrode_ids = merged_union.get("electrode_ids")
    channel_ids = merged_union.get("channel_ids")
    channel_labels: list[str] = []
    for j in range(n_ch_total):
        lab = None
        if electrode_ids is not None:
            try:
                e = list(electrode_ids)[int(j)]
                if e is not None:
                    lab = f"e{int(e)}"
            except Exception:
                lab = None
        if lab is None and channel_ids is not None:
            try:
                c = list(channel_ids)[int(j)]
                if c is not None:
                    lab = str(c)
            except Exception:
                lab = None
        if lab is None:
            lab = str(j)
        channel_labels.append(lab)

    # Select channels for plotting: hybridize amplitude + chronology.
    try:
        amp = np.ptp(tmpl, axis=0).astype(float)
    except Exception:
        amp = None
    if amp is None or np.asarray(amp).ndim != 1:
        return

    def _best_effort_ap_timings(*, template_t_by_ch: Any, t_ms: Any) -> dict[str, Any]:
        """Best-effort AP timing indices for each channel.

        Assumes an extracellular waveform shape:
        - a large negative deflection (often the dominant feature)
        - often a smaller positive deflection before and/or after the negative peak

                This implementation enforces a strict chronology of timepoints per channel:
                    ap_start -> pre_pos -> neg_peak -> post_pos -> ap_end

                Method:
                - Anchor on the negative peak.
                - Define a symmetric "noise band" around a robust baseline using a robust noise estimate.
                - Find AP start/end as the boundaries of the contiguous (best-effort) region around the negative
                    peak where the waveform is outside that noise band.
                - Within the AP window, pick the strongest positive deflection before/after the negative peak.
        """

        import numpy as np  # type: ignore[import-not-found]

        tmpl2 = np.asarray(template_t_by_ch, dtype=float)
        t_ms_arr = np.asarray(t_ms, dtype=float)
        n_samp_local = int(tmpl2.shape[0])
        n_ch_local = int(tmpl2.shape[1])

        n0 = max(3, int(round(0.15 * n_samp_local)))
        baseline = np.nanmedian(tmpl2[:n0, :], axis=0)

        # Noise estimate (MAD) from early+late windows, excluding the central region.
        try:
            early = tmpl2[:n0, :]
            late = tmpl2[max(0, n_samp_local - n0) :, :]
            noise_samples = np.concatenate([early, late], axis=0)
            mad = np.nanmedian(np.abs(noise_samples - baseline[None, :]), axis=0)
            noise = 1.4826 * mad
        except Exception:
            noise = np.zeros(n_ch_local, dtype=float)

        neg_i = np.argmin(tmpl2, axis=0).astype(int)
        pos_i = np.argmax(tmpl2, axis=0).astype(int)

        ap_start_i = np.zeros(n_ch_local, dtype=int)
        ap_end_i = (n_samp_local - 1) * np.ones(n_ch_local, dtype=int)
        pre_pos_i = np.zeros(n_ch_local, dtype=int)
        post_pos_i = np.zeros(n_ch_local, dtype=int)

        hold = 3  # require a short run inside the noise band to declare start/end
        k_sigma = 3.0

        for ch in range(n_ch_local):
            w = np.asarray(tmpl2[:, ch], dtype=float)
            b = float(baseline[ch])
            ni = int(neg_i[ch])
            if ni <= 1:
                start_i = int(max(0, ni - 1))
                pre_i = int(start_i)
                post_i = int(min(n_samp_local - 1, ni + 1))
                end_i = int(max(ni, post_i))

                ap_start_i[ch] = int(start_i)
                ap_end_i[ch] = int(end_i)
                pre_pos_i[ch] = int(pre_i)
                post_pos_i[ch] = int(post_i)
                continue

            # Symmetric noise band around baseline.
            ns = float(noise[ch]) if np.isfinite(noise[ch]) else 0.0
            ptp_ch = float(np.nanmax(w) - np.nanmin(w)) if np.all(np.isfinite(w)) else float(np.ptp(np.nan_to_num(w)))
            band = max(k_sigma * ns, 0.02 * max(ptp_ch, 0.0), 1e-9)
            pos_thr = b + band
            neg_thr = b - band

            active = (w > pos_thr) | (w < neg_thr)

            # Find AP start/end as boundaries of the "active" region around the negative peak.
            start_i = 0
            for j in range(int(ni) - int(hold), -1, -1):
                try:
                    if np.all(~active[j : j + hold]) and np.any(active[j + hold : ni + 1]):
                        start_i = int(j + hold)
                        break
                except Exception:
                    continue

            end_i = int(n_samp_local - 1)
            for j in range(int(ni) + 1, int(n_samp_local) - int(hold) + 1):
                try:
                    if np.all(~active[j : j + hold]) and np.any(active[ni:j]):
                        end_i = int(j - 1)
                        break
                except Exception:
                    continue

            start_i = int(np.clip(start_i, 0, ni))
            end_i = int(np.clip(end_i, ni, n_samp_local - 1))

            # Pick strongest pre/post positive deflections within the AP window.
            if start_i < ni:
                pre_slice = w[start_i:ni]
                ppi = int(start_i + int(np.argmax(pre_slice))) if pre_slice.size else int(start_i)
            else:
                ppi = int(start_i)

            if ni < end_i:
                post_slice = w[ni : end_i + 1]
                psti = int(ni + int(np.argmax(post_slice))) if post_slice.size else int(end_i)
            else:
                psti = int(end_i)

            # Enforce strict chronology and monotonicity.
            start_i = int(min(int(start_i), int(ppi)))
            end_i = int(max(int(end_i), int(psti)))
            start_i = int(np.clip(start_i, 0, ni))
            end_i = int(np.clip(end_i, ni, n_samp_local - 1))
            ppi = int(np.clip(ppi, start_i, ni))
            psti = int(np.clip(psti, ni, end_i))

            ap_start_i[ch] = int(start_i)
            ap_end_i[ch] = int(end_i)
            pre_pos_i[ch] = int(ppi)
            post_pos_i[ch] = int(psti)

        return {
            "baseline": baseline,
            "noise": noise,
            "neg_peak_i": neg_i,
            "pos_peak_i": pos_i,
            "pre_pos_i": pre_pos_i,
            "post_pos_i": post_pos_i,
            "ap_start_i": ap_start_i,
            "ap_end_i": ap_end_i,
            "neg_peak_ms": t_ms_arr[np.clip(neg_i, 0, n_samp_local - 1)],
            "pos_peak_ms": t_ms_arr[np.clip(pos_i, 0, n_samp_local - 1)],
            "ap_start_ms": t_ms_arr[np.clip(ap_start_i, 0, n_samp_local - 1)],
            "ap_end_ms": t_ms_arr[np.clip(ap_end_i, 0, n_samp_local - 1)],
            "pre_pos_ms": t_ms_arr[np.clip(pre_pos_i, 0, n_samp_local - 1)],
            "post_pos_ms": t_ms_arr[np.clip(post_pos_i, 0, n_samp_local - 1)],
        }

    timings = _best_effort_ap_timings(template_t_by_ch=tmpl, t_ms=t_ms)

    # Persist timing analysis next to merged_unit outputs when requested.
    if ap_timings_json_path is not None:
        try:
            import json
            import numpy as np  # type: ignore[import-not-found]

            ap_timings_json_path = Path(ap_timings_json_path)
            ap_timings_json_path.parent.mkdir(parents=True, exist_ok=True)

            payload: dict[str, Any] = {
                "unit_id": unit_id,
                "sampling_frequency_hz": float(fs_hz),
                "ms_before": ms_before,
                "ms_after": ms_after,
                "n_samples": int(n_samp),
                "n_channels": int(n_ch_total),
                "channel_labels": list(channel_labels),
                "ptp_uv": np.asarray(amp, dtype=float).tolist(),
                "ap_start_ms": np.asarray(timings.get("ap_start_ms"), dtype=float).tolist(),
                "pre_pos_ms": np.asarray(timings.get("pre_pos_ms"), dtype=float).tolist(),
                "pos_peak_ms": np.asarray(timings.get("pos_peak_ms"), dtype=float).tolist(),
                "neg_peak_ms": np.asarray(timings.get("neg_peak_ms"), dtype=float).tolist(),
                "post_pos_ms": np.asarray(timings.get("post_pos_ms"), dtype=float).tolist(),
                "ap_end_ms": np.asarray(timings.get("ap_end_ms"), dtype=float).tolist(),
            }
            ap_timings_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    top_n = max(1, int(min(int(top_channels), int(n_ch_total))))
    order_by_amp = np.argsort(-np.asarray(amp))

    def _select_hybrid_chain(
        *,
        neg_peak_i_all: Any,
        amp_all: Any,
        top_n: int,
        max_candidates: int,
        eps_amp: float = 1e-9,
    ) -> list[int]:
        """Pick a subset that is strictly chronological and strictly decreasing in amplitude.

        We model each channel as a point (t_peak, amp). We want an ordered sequence where:
        - t_peak is strictly increasing
        - amp is strictly decreasing

        This enforces: each channel peaks after the previous AND has lower amplitude.
        """

        import numpy as np  # type: ignore[import-not-found]

        neg_i = np.asarray(neg_peak_i_all)
        amp_v = np.asarray(amp_all, dtype=float)
        if neg_i.ndim != 1 or amp_v.ndim != 1:
            return []

        valid = np.isfinite(amp_v) & np.isfinite(neg_i.astype(float))
        cand = np.where(valid)[0].astype(int)
        if cand.size == 0:
            return []

        # Restrict to high-amplitude candidates to avoid filling the chain with noise.
        cand = cand[np.argsort(-amp_v[cand])]
        cand = cand[: int(max(1, min(int(max_candidates), int(cand.size))))]

        # Sort candidates by peak time (then by amplitude descending for stability).
        cand = sorted(cand.tolist(), key=lambda i: (int(neg_i[int(i)]), -float(amp_v[int(i)])))

        m = len(cand)
        if m == 0:
            return []

        t = np.asarray([int(neg_i[int(i)]) for i in cand], dtype=int)
        a = np.asarray([float(amp_v[int(i)]) for i in cand], dtype=float)

        # DP over lengths up to top_n: maximize total amplitude for each length.
        top_n2 = int(min(int(top_n), int(m)))
        score = np.full((top_n2 + 1, m), -np.inf, dtype=float)
        prev = np.full((top_n2 + 1, m), -1, dtype=int)

        for i in range(m):
            score[1, i] = a[i]

        for L in range(2, top_n2 + 1):
            for i in range(m):
                best_s = -np.inf
                best_j = -1
                ti = int(t[i])
                ai = float(a[i])
                for j in range(0, i):
                    if int(t[j]) >= ti:
                        continue
                    if float(a[j]) <= ai + float(eps_amp):
                        continue
                    s = float(score[L - 1, j]) + ai
                    if s > best_s:
                        best_s = s
                        best_j = int(j)
                if best_j >= 0:
                    score[L, i] = best_s
                    prev[L, i] = best_j

        # Prefer the longest feasible chain (up to top_n), then max sum amplitude.
        best_L = 1
        best_end = int(np.nanargmax(score[1, :]))
        best_sum = float(score[1, best_end])
        for L in range(2, top_n2 + 1):
            if not np.any(np.isfinite(score[L, :])):
                continue
            end = int(np.nanargmax(score[L, :]))
            s = float(score[L, end])
            if (L > best_L) or (L == best_L and s > best_sum):
                best_L = int(L)
                best_end = int(end)
                best_sum = float(s)

        # Reconstruct chain.
        chain_local: list[int] = []
        L = int(best_L)
        i = int(best_end)
        while L >= 1 and i >= 0:
            chain_local.append(int(cand[i]))
            pj = int(prev[L, i])
            i = pj
            L -= 1
        chain_local.reverse()
        return [int(x) for x in chain_local]

    # Build a hybrid chain from high-amplitude candidates.
    try:
        neg_peak_i_all = timings.get("neg_peak_i")
        max_candidates = int(n_ch_total)
        picked = _select_hybrid_chain(
            neg_peak_i_all=neg_peak_i_all,
            amp_all=amp,
            top_n=top_n,
            max_candidates=max_candidates,
        )
        if not picked:
            raise RuntimeError("empty hybrid selection")
    except Exception:
        # Fallback: take top channels by amplitude.
        picked = [int(i) for i in order_by_amp[:top_n].tolist()]

    def _warn(fmt: str, *args) -> None:
        try:
            msg = fmt % args if args else fmt
        except Exception:
            msg = fmt
        if logger is not None:
            try:
                logger.warning(msg)
                return
            except Exception:
                pass
        try:
            print(f"[WARN] {msg}")
        except Exception:
            pass

    if len(picked) < int(top_n):
        _warn(
            "Propagation: strict amp+chron constraints selected %d/%d channels for unit %s",
            int(len(picked)),
            int(top_n),
            unit_id,
        )

    # Warnings about "best channel" selection.
    try:
        best_ch = int(order_by_amp[0])
        if best_ch not in picked:
            _warn("Propagation: best channel (max PTP) not included in picked set for unit %s", unit_id)
    except Exception:
        best_ch = None

    # Best channel by negative deflection magnitude (extracellular AP heuristic).
    try:
        baseline_all = np.asarray(timings.get("baseline"), dtype=float)
        neg_amp_all = baseline_all - np.asarray(np.min(tmpl, axis=0), dtype=float)
        best_neg_ch = int(np.nanargmax(neg_amp_all))
    except Exception:
        best_neg_ch = None

    # Ensure final ordering is chronological by negative peak.
    try:
        neg_peak_i_all = timings.get("neg_peak_i")
        neg_peak_i_sel = [int(np.asarray(neg_peak_i_all)[int(i)]) for i in picked]
        picked = [i for _, i in sorted(zip(neg_peak_i_sel, picked), key=lambda x: int(x[0]))]
    except Exception:
        pass

    # Build channel panels with overlap.
    cpp = max(1, int(channels_per_panel))
    ov = max(0, int(channel_overlap))
    step = max(1, cpp - ov)
    panels: list[list[int]] = []
    for start in range(0, len(picked), step):
        sl = picked[start : start + cpp]
        if not sl:
            continue
        panels.append(list(sl))
        if start + cpp >= len(picked):
            break
    if not panels:
        panels = [picked]

    # Robust vertical spacing scale. Use smaller spacing so traces visually pop more.
    try:
        ptp_sel = np.asarray(amp, dtype=float)[picked]
        scale = float(np.nanpercentile(ptp_sel[np.isfinite(ptp_sel)], 90)) if np.any(np.isfinite(ptp_sel)) else 1.0
        if not (scale > 0):
            scale = 1.0
        spacing = 0.55 * scale
    except Exception:
        spacing = 1.0

    add_scalebar = _try_get_add_scalebar()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Larger height than PPT to emphasize amplitudes during iteration.
    png_figsize = (13.333, 10.0)

    neg_peak_i_all = timings.get("neg_peak_i")
    neg_peak_i_all = None if neg_peak_i_all is None else list(neg_peak_i_all)

    for panel_idx, panel in enumerate(panels):
        # Output name: single panel gets unit_<id>.png; multi-panels get unit_<id>_panel_XX.png
        out_path = (
            out_dir / f"unit_{unit_id}.png"
            if len(panels) == 1
            else out_dir / f"unit_{unit_id}_panel_{panel_idx + 1:02d}.png"
        )

        fig = plt.figure(figsize=png_figsize)
        ax = fig.add_subplot(111)

        # Chronology should read top-down: earliest (smallest negative-peak time) on top.
        n_in_panel = int(len(panel))

        # Warn if the first plotted channel isn't the max-PTP channel.
        try:
            first_plotted = int(panel[0]) if n_in_panel > 0 else None
            if best_ch is not None and first_plotted is not None and int(first_plotted) != int(best_ch):
                _warn(
                    "Propagation: first plotted channel %s != best channel %s (max PTP) for unit %s",
                    channel_labels[int(first_plotted)],
                    channel_labels[int(best_ch)],
                    unit_id,
                )
            if best_neg_ch is not None and first_plotted is not None and int(first_plotted) != int(best_neg_ch):
                _warn(
                    "Propagation: first plotted channel %s != largest-negative-deflection channel %s for unit %s",
                    channel_labels[int(first_plotted)],
                    channel_labels[int(best_neg_ch)],
                    unit_id,
                )
        except Exception:
            pass

        # Plot one line per channel (single color).
        for j, ch in enumerate(panel):
            ch = int(ch)
            # Top-down ordering.
            y0 = float(n_in_panel - 1 - int(j)) * float(spacing)
            w = np.asarray(tmpl[:, ch], dtype=float)
            ax.plot(t_ms, w + y0, color="black", alpha=0.92, linewidth=1.2)

            # Negative-peak marker (triangle).
            try:
                if neg_peak_i_all is not None:
                    si = int(neg_peak_i_all[ch])
                else:
                    si = int(np.argmin(w))
                si = int(max(0, min(n_samp - 1, si)))
                ax.scatter([float(t_ms[si])], [float(w[si] + y0)], s=18, color="black", marker="v", zorder=10)
            except Exception:
                pass

            # Channel label aligned to this trace.
            try:
                ax.text(
                    float(t_ms[0]),
                    y0,
                    str(channel_labels[ch]),
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            except Exception:
                pass

        # Minimal styling.
        try:
            ax.set_yticks([])
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        except Exception:
            pass

        if add_scalebar is not None:
            try:
                add_scalebar(ax=ax, units="µV", fontsize=8)
            except Exception:
                pass

        title_text = f"Propagation unit {unit_id} — {len(panel)} ch (hybrid amp+chron), ordered by negative peak (v = neg peak)"
        fig.suptitle(title_text, fontsize=12)
        try:
            fig.subplots_adjust(left=0.10, right=0.99, bottom=0.03, top=0.92)
        except Exception:
            pass

        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def _write_unit_propagation_plots_pdf(
    *,
    pdf_path: Path,
    unit_id: Any,
    merged_union: dict[str, Any],
    fs_hz: float,
    ms_before: Optional[float],
    ms_after: Optional[float],
    top_channels: int = 10,
    n_waveforms: int = 12,
    channels_per_panel: int = 25,
    channel_overlap: int = 5,
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
        merged_union=merged_union,
        fs_hz=fs_hz,
        ms_before=ms_before,
        ms_after=ms_after,
        top_channels=top_channels,
        n_waveforms=n_waveforms,
        channels_per_panel=channels_per_panel,
        channel_overlap=channel_overlap,
        ap_timings_json_path=ap_timings_json_path,
        logger=logger,
    )


def _write_footprint_ptp_map(
    *,
    out_path: Path,
    channel_locations_xy: Any,
    footprint_ptp: Any,
    title: str,
    log_scale: bool,
    cmap: str = "viridis",
    electrode_ids: Any = None,
    recording_electrode_ids: Any = None,
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
            recording_electrode_ids=recording_electrode_ids,
            title=title,
            cmap=cmap,
            norm=norm,
            cbar_label="PTP (µV)",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    # Fallback scatter (legacy behavior).
    locs = np.asarray(channel_locations_xy, dtype=float)
    if locs.ndim != 2 or locs.shape[1] < 2 or amp.ndim != 1 or locs.shape[0] != amp.shape[0]:
        return

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=10)
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_unit_template_and_footprint_svg(
    *,
    out_path: Path,
    unit_id: Any,
    template: Any,
    channel_locations_xy: Any,
    electrode_ids: Any = None,
    recording_electrode_ids: Any = None,
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
                recording_electrode_ids=recording_electrode_ids,
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
    recording_electrode_ids: Any,
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
        recording_electrode_ids=recording_electrode_ids,
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
    recording_electrode_ids: Any,
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
        recording_electrode_ids=recording_electrode_ids,
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
