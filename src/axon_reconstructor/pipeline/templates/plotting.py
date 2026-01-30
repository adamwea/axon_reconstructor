"""Templates plotting helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional


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
) -> None:
    """Write a footprint PTP amplitude map as a single image.

    This is intentionally lightweight and does not depend on axon_velocity.
    """

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm

    locs = np.asarray(channel_locations_xy, dtype=float)
    amp = np.asarray(footprint_ptp, dtype=float)
    if locs.ndim != 2 or locs.shape[1] < 2 or amp.ndim != 1 or locs.shape[0] != amp.shape[0]:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

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


__all__ = [
    "_write_templates_grid_pdf",
    "_write_unit_templates_across_sources_pdf",
    "_write_footprint_ptp_map",
    "_write_unit_template_and_footprint_svg",
]
