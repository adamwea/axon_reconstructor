"""All-units overview plotting helper (internal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .plotting_core import _save_fig_pdf_and_png, _white_bg_rc_params, _with_suffix

def write_all_units_overview_pdf(
    *,
    all_units_overview_pdf: Path,
    all_locations: list[Any],
    all_unit_polylines: list[dict[str, Any]],
    stream_id: str,
    force_restart: bool,
    logger: Any,
) -> bool:
    """Write the all-units overview morphology PDF."""

    try:
        import numpy as np  # type: ignore[import-not-found]
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        logger.warning("Plotting dependencies unavailable: %s", e)
        return False

    all_units_overview_png = _with_suffix(all_units_overview_pdf, ".png")

    if (not all_units_overview_pdf.exists()) or force_restart:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)

        if all_locations:
            locs_all = np.concatenate([np.asarray(x)[:, :2] for x in all_locations if np.asarray(x).size], axis=0)
            ax.plot(locs_all[:, 0], locs_all[:, 1], marker=".", ls="", color="0.8", alpha=0.15, ms=2)

        cm = plt.get_cmap("tab20")
        for i, poly in enumerate(all_unit_polylines):
            xy = poly.get("polyline_xy")
            if not xy:
                continue
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            ax.plot(xs, ys, lw=1.2, alpha=0.9, color=cm(i % 20))

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"All units morphology (stream={stream_id})")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")

        _save_fig_pdf_and_png(fig=fig, pdf_path=all_units_overview_pdf, png_path=all_units_overview_png, dpi=150)
        plt.close(fig)

    return all_units_overview_pdf.exists()

__all__ = [
    "write_all_units_overview_pdf",
    "write_unit_reconstruction_pdfs",
]
