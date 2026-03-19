"""Core reconstruction plotting helpers (internal)."""

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Any, Optional


DPI_STD = 180
DPI_HI = 350


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_unit_output_layout(
    *,
    out_unit_dir: Path,
    branches_root_relpath: str = "branches",
    branches_clean_relpath: str = "branches/clean",
    branches_raw_relpath: str = "branches/raw",
    morphology_relpath: str = "morphology",
    heuristics_relpath: str = "heuristics",
    maps_relpath: str = "maps",
) -> dict[str, Path]:
    out_unit_dir = Path(out_unit_dir)

    def _resolve(p: str) -> Path:
        rel = Path(str(p)).expanduser()
        return rel if rel.is_absolute() else (out_unit_dir / rel)

    branches_root = _resolve(branches_root_relpath)
    return {
        "branches_root": branches_root,
        "branches_clean": _resolve(branches_clean_relpath),
        "branches_raw": _resolve(branches_raw_relpath),
        "morphology": _resolve(morphology_relpath),
        "heuristics": _resolve(heuristics_relpath),
        "maps": _resolve(maps_relpath),
    }


def _ensure_unit_output_layout(layout: dict[str, Path]) -> None:
    for k in ["branches_root", "branches_clean", "branches_raw", "morphology", "heuristics", "maps"]:
        Path(layout[k]).mkdir(parents=True, exist_ok=True)


def _maybe_migrate_legacy_unit_outputs(*, out_unit_dir: Path, layout: dict[str, Path]) -> None:
    """Best-effort migration from legacy flat layout into the new subdirs.

    Intentionally does not move JSON files.
    """

    out_unit_dir = Path(out_unit_dir)

    def _move_if_exists(src: Path, dst: Path) -> None:
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            return
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    # Legacy: branches/ contained per-branch velocity plots directly.
    legacy_branches_dir = out_unit_dir / "branches"
    if legacy_branches_dir.exists() and legacy_branches_dir.is_dir():
        clean_dir = Path(layout["branches_clean"])
        clean_dir.mkdir(parents=True, exist_ok=True)
        for p in list(legacy_branches_dir.glob("*.png")) + list(legacy_branches_dir.glob("*.pdf")):
            if p.is_file():
                _move_if_exists(p, clean_dir / p.name)

    branches_clean_dir = Path(layout["branches_clean"])
    branches_raw_dir = Path(layout["branches_raw"])
    morphology_dir = Path(layout["morphology"])
    heuristics_dir = Path(layout["heuristics"])

    for stem in ["branches_clean", "branches_clean_zoom", "branch_velocities", "branch_velocities_overlay"]:
        _move_if_exists(out_unit_dir / f"{stem}.pdf", branches_clean_dir / f"{stem}.pdf")
        _move_if_exists(out_unit_dir / f"{stem}.png", branches_clean_dir / f"{stem}.png")

    for stem in ["branches_raw", "branches_raw_zoom", "branches_raw_clean", "branch_velocities_raw", "branch_velocities_raw_overlay"]:
        _move_if_exists(out_unit_dir / f"{stem}.pdf", branches_raw_dir / f"{stem}.pdf")
        _move_if_exists(out_unit_dir / f"{stem}.png", branches_raw_dir / f"{stem}.png")

    for stem in ["morphology", "morphology_zoom"]:
        _move_if_exists(out_unit_dir / f"{stem}.pdf", morphology_dir / f"{stem}.pdf")
        _move_if_exists(out_unit_dir / f"{stem}.png", morphology_dir / f"{stem}.png")

    for stem in ["heuristics", "graph_heuristics"]:
        _move_if_exists(out_unit_dir / f"{stem}.pdf", heuristics_dir / f"{stem}.pdf")
        _move_if_exists(out_unit_dir / f"{stem}.png", heuristics_dir / f"{stem}.png")


def _with_suffix(path: Path, suffix: str) -> Path:
    path = Path(path)
    if not suffix.startswith("."):
        suffix = "." + suffix
    return path.with_suffix(suffix)


def _save_fig_pdf_and_png(
    *,
    fig: Any,
    pdf_path: Path,
    png_path: Path,
    dpi: int = 150,
    write_png: bool = True,
    write_svg: bool = False,
    svg_path: Path | None = None,
) -> None:
    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if bool(write_png):
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if bool(write_svg):
        try:
            svg_out = Path(svg_path) if svg_path is not None else pdf_path.with_suffix(".svg")
            fig.savefig(svg_out, format="svg", bbox_inches="tight", facecolor="white")
        except Exception:
            pass


def _save_fig_png(
    *,
    fig: Any,
    png_path: Path,
    dpi: int,
    write_png: bool = True,
    write_svg: bool = False,
    svg_path: Path | None = None,
) -> None:
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(write_png):
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if bool(write_svg):
        try:
            svg_out = Path(svg_path) if svg_path is not None else png_path.with_suffix(".svg")
            fig.savefig(svg_out, format="svg", bbox_inches="tight", facecolor="white")
        except Exception:
            pass


def _minimal_axes(ax: Any) -> None:
    """Make axes minimalist: hide top/right spines, keep left/bottom."""

    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass
    try:
        ax.tick_params(direction="out", length=3, width=0.8)
    except Exception:
        pass


def _thin_lines_and_markers(ax: Any, *, lw: float = 0.55, ms: float = 2.0, alpha: float = 0.9) -> None:
    for ln in getattr(ax, "lines", []) or []:
        try:
            ln.set_linewidth(lw)
        except Exception:
            pass
        try:
            ln.set_markersize(ms)
        except Exception:
            pass
        try:
            ln.set_alpha(alpha)
        except Exception:
            pass


def _white_bg_rc_params() -> dict[str, Any]:
    """Matplotlib rcParams tuned for white backgrounds (high contrast)."""

    try:
        from cycler import cycler  # type: ignore[import-not-found]

        prop_cycle = cycler(
            "color",
            [
                "#0072B2",  # blue
                "#D55E00",  # vermillion
                "#009E73",  # green
                "#CC79A7",  # purple
                "#E69F00",  # orange
                "#56B4E9",  # sky blue
                "#000000",  # black
                "#F0E442",  # yellow
            ],
        )
    except Exception:
        prop_cycle = None

    rc: dict[str, Any] = {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#222222",
        "grid.color": "#DDDDDD",
        "axes.grid": False,
    }
    if prop_cycle is not None:
        rc["axes.prop_cycle"] = prop_cycle
    return rc


def _force_white_background(fig: Any) -> None:
    """Best-effort: make figure/axes readable on white slides."""

    try:
        fig.patch.set_facecolor("white")
    except Exception:
        pass
    for ax in getattr(fig, "axes", []) or []:
        try:
            ax.set_facecolor("white")
        except Exception:
            pass
        try:
            ax.tick_params(colors="#222222")
        except Exception:
            pass
        for spine in getattr(ax, "spines", {}).values() if hasattr(ax, "spines") else []:
            try:
                spine.set_color("#333333")
            except Exception:
                pass


def _okabe_ito_palette() -> list[str]:
    # Colorblind-safe, high-contrast palette.
    return [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # purple
        "#E69F00",  # orange
        "#56B4E9",  # sky
        "#000000",  # black
        "#F0E442",  # yellow (use sparingly)
    ]


def _recolor_noncolormapped_artists(fig: Any) -> None:
    """Best-effort recolor for axon_velocity figures.

    We only recolor:
      - Line2D objects
      - PathCollections that are NOT colormapped (i.e., get_array() is None)

    This avoids breaking template heatmaps / imshow / colormap-based electrode plots.
    """

    try:
        palette = _okabe_ito_palette()
        pi = 0

        def next_color() -> str:
            nonlocal pi
            c = palette[pi % len(palette)]
            pi += 1
            return c

        for ax in getattr(fig, "axes", []) or []:
            # Lines (often branches / fits)
            try:
                for line in ax.get_lines() or []:
                    c = next_color()
                    try:
                        line.set_color(c)
                    except Exception:
                        pass
                    try:
                        line.set_alpha(0.95)
                    except Exception:
                        pass
                    try:
                        lw = float(line.get_linewidth() or 1.0)
                        line.set_linewidth(max(lw, 2.0))
                    except Exception:
                        pass
            except Exception:
                pass

            # Scatter collections (often per-branch points); skip colormapped ones.
            try:
                cols = getattr(ax, "collections", []) or []
                for coll in cols:
                    try:
                        arr = coll.get_array()
                    except Exception:
                        arr = None
                    if arr is not None:
                        continue
                    c = next_color()
                    try:
                        coll.set_facecolor(c)
                    except Exception:
                        pass
                    try:
                        coll.set_edgecolor("none")
                    except Exception:
                        pass
                    try:
                        coll.set_alpha(0.85)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        return


def _apply_raw_branch_colors(*, ax: Any, n_raw_paths: int) -> None:
    """Make raw-branches and velocities use the same color per raw path.

    `axon_velocity.Tracking.plot_raw_branches()` uses a colormap. We instead enforce the
    same stable Okabe–Ito palette mapping used by `_plot_raw_branch_velocities()`:

      raw_idx -> palette[raw_idx % len(palette)]

    We only recolor the actual branch lines (marker 'o', linestyle '-') and leave the
    background electrode dots (marker '.', linestyle '') alone.
    """

    try:
        palette = _okabe_ito_palette()

        lines = []
        for line in (ax.get_lines() or []):
            try:
                if (line.get_marker() == "o") and (line.get_linestyle() == "-"):
                    lines.append(line)
            except Exception:
                continue

        for raw_idx in range(min(int(n_raw_paths), len(lines))):
            c = palette[int(raw_idx) % len(palette)]
            line = lines[raw_idx]
            try:
                line.set_color(c)
            except Exception:
                pass
            try:
                line.set_alpha(0.95)
            except Exception:
                pass
            try:
                lw = float(line.get_linewidth() or 1.0)
                line.set_linewidth(max(lw, 2.0))
            except Exception:
                pass
            try:
                line.set_markeredgecolor("k")
            except Exception:
                pass
    except Exception:
        return


def _extract_raw_branch_colors_from_plot(*, gtr: Any) -> list[Any]:
    """Best-effort: read raw branch line colors from gtr.plot_raw_branches().

    Returns colors in the same raw-branch plotting order used by axon_velocity.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        plot_fn = getattr(gtr, "plot_raw_branches", None)
        if not callable(plot_fn):
            return []

        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        try:
            with plt.rc_context(_white_bg_rc_params()):
                _ = plot_fn(plot_full_template=True, ax=ax)

            colors: list[Any] = []
            for line in (ax.get_lines() or []):
                try:
                    if (line.get_marker() == "o") and (line.get_linestyle() == "-"):
                        colors.append(line.get_color())
                except Exception:
                    continue
            return colors
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass
    except Exception:
        return []


def _plot_raw_branch_velocities(
    *,
    uid: Any,
    gtr: Any,
    ax: Any,
    logger: Any,
) -> None:
    """Plot per-raw-branch velocity fits with a stable palette.

    axon_velocity's built-in `plot_velocities()` iterates `self.branches` (cleaned/accepted
    branches), which can be fewer than the discovered raw paths. For QC we want to see
    *all* raw paths here.
    """

    import numpy as np  # type: ignore[import-not-found]

    paths_raw = getattr(gtr, "_paths_raw", None)
    if not paths_raw:
        ax.text(0.5, 0.5, "no raw paths", ha="center", va="center", fontsize=10)
        return

    logger.info("Unit %s: plotting raw-branch velocities for %d raw paths", uid, len(paths_raw))

    # Reuse the exact branch colors emitted by plot_raw_branches(), so overlay
    # velocity colors match branches_raw / branches_raw_zoom.
    raw_branch_colors = _extract_raw_branch_colors_from_plot(gtr=gtr)

    handles = []
    labels = []

    # Keep these readable in the analysis montage.
    # (User request: increase fonts by ~100%.)
    label_fs = 22
    tick_fs = 18
    legend_fs = 16

    for raw_idx, raw_path in enumerate(paths_raw):
        try:
            # Match axon_velocity's convention.
            path = list(raw_path)[::-1][1:]

            est = getattr(gtr, "_estimate_peaks_and_dists", None)
            rve = getattr(gtr, "robust_velocity_estimator", None)
            if (not callable(est)) or (not callable(rve)):
                continue

            peaks, dists = est(path)
            peaks = np.asarray(peaks, dtype=float)
            dists = np.asarray(dists, dtype=float)
            if (peaks.size < 2) or (dists.size != peaks.size):
                continue

            # axon_velocity returns inliers/outliers when last arg True.
            (
                _path_clean,
                velocity,
                offset,
                r2,
                _p_value,
                _dists_clean,
                _peaks_clean,
                inlier_mask,
            ) = rve(path, peaks, dists, True)

            try:
                inlier_mask = np.asarray(inlier_mask, dtype=bool)
            except Exception:
                inlier_mask = np.ones_like(peaks, dtype=bool)

            # Fit line
            color = None
            try:
                if raw_idx < len(raw_branch_colors):
                    color = raw_branch_colors[raw_idx]
            except Exception:
                color = None

            try:
                v = float(velocity)
                b = float(offset)
                xs = np.linspace(float(np.min(peaks)), float(np.max(peaks)), 50)
                ys = v * xs + b
                (ln,) = ax.plot(xs, ys, lw=2.5, alpha=0.95, color=color, linestyle=":")
            except Exception:
                (ln,) = ax.plot([], [], lw=2.5, alpha=0.95, color=color, linestyle=":")

            color = ln.get_color()

            # Markers (match line color selected by matplotlib cycle)
            ax.scatter(
                peaks[inlier_mask],
                dists[inlier_mask],
                s=72,
                color=color,
                alpha=0.75,
                edgecolors="k",
                linewidths=0.3,
            )
            out = ~inlier_mask
            if np.any(out):
                ax.scatter(
                    peaks[out],
                    dists[out],
                    s=104,
                    marker="d",
                    color=color,
                    alpha=0.85,
                    edgecolors="k",
                    linewidths=0.3,
                )

            handles.append(ln)
            vel_label = "NA"
            r2_label = "NA"
            try:
                vel_label = f"{float(velocity):.2f}"
            except Exception:
                pass
            try:
                r2_label = f"{float(r2):.2f}"
            except Exception:
                pass
            labels.append(f"Raw {raw_idx}  vel: {vel_label} mm/s  r2: {r2_label}")
        except Exception:
            continue

    ax.set_xlabel("Peak time (ms)", fontsize=label_fs)
    ax.set_ylabel("Distance (um)", fontsize=label_fs)
    try:
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    except Exception:
        pass
    try:
        _minimal_axes(ax)
        ax.tick_params(top=False, right=False)
    except Exception:
        pass
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower right",
            bbox_to_anchor=(2.10, 0.02),
            fontsize=legend_fs,
            frameon=False,
            ncol=1,
            borderaxespad=0.0,
        )


def _strip_axes_titles(fig: Any, *, titles_to_remove: set[str]) -> None:
    """Best-effort removal of specific axis/figure titles.

    This is intentionally conservative: we only remove exact matches (case-insensitive)
    for titles in `titles_to_remove`.
    """

    try:
        norm = {t.strip().lower() for t in titles_to_remove}
        for ax in getattr(fig, "axes", []) or []:
            try:
                t = (ax.get_title() or "").strip().lower()
                if t in norm:
                    ax.set_title("")
            except Exception:
                continue

        # Handle suptitle (matplotlib keeps this in fig._suptitle)
        try:
            st = getattr(fig, "_suptitle", None)
            if st is not None:
                t = (st.get_text() or "").strip().lower()
                if t in norm:
                    st.set_text("")
        except Exception:
            pass
    except Exception:
        return

