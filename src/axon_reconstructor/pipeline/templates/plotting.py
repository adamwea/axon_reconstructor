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


__all__ = [
    "_write_templates_grid_pdf",
    "_write_unit_templates_across_sources_pdf",
]
