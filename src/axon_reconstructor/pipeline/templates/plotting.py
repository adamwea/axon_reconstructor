"""Templates plotting helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional


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
    noncontrib_rgba=(0.70, 0.70, 0.70, 1.0),
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

    # Overlay selected channels.
    for j in sel:
        y = tmpl[:, int(j)]
        ax.plot(t_ms, y, lw=0.8, alpha=0.85)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("uV", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)


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
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5), constrained_layout=True)
            axes = axes.ravel().tolist()

            for ax, entry in zip(axes, chunk):
                uid = entry.get("unit_id")
                title = f"unit {uid}"
                _write_template_overlay(
                    ax=ax,
                    template=entry["template"],
                    fs_hz=float(fs_hz),
                    ms_before=ms_before,
                    ms_after=ms_after,
                    top_channels=int(top_channels),
                    title=title,
                )

            for j in range(len(chunk), len(axes)):
                axes[j].set_axis_off()

            out.savefig(fig, dpi=150)
            plt.close(fig)

    logger.info("Wrote templates grid PDF -> %s", pdf_path)


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


def _write_full_chip_amplitude_map_png(
    *,
    out_path: Path,
    template_ch_by_t: Any,
    electrode_ids: Any,
    recording_electrode_ids: Any,
    title: str,
    cmap: str = "viridis",
) -> None:
    """axon_velocity-like amplitude map, but rendered over the full chip grid."""

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


def _write_full_chip_peak_latency_map_png(
    *,
    out_path: Path,
    template_ch_by_t: Any,
    electrode_ids: Any,
    recording_electrode_ids: Any,
    sampling_frequency_hz: float,
    title: str,
    cmap: str = "viridis",
) -> None:
    """axon_velocity-like peak latency map, rendered over the full chip grid."""

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

    # Match axon_velocity.plotting.plot_peak_latency_map behavior (argmin index -> ms).
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
    "_write_unit_templates_across_sources_pdf",
    "_write_footprint_ptp_map",
    "_write_unit_template_and_footprint_svg",
    "_write_full_chip_amplitude_map_png",
    "_write_full_chip_peak_latency_map_png",
]
