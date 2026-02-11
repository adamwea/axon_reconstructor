"""Reconstruction plotting helpers.

Keep plotting out of `reconstruction/main.py` so the step entrypoint stays focused
on orchestration and IO.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


DPI_STD = 180
DPI_HI = 350


def _compute_unit_output_layout(*, out_unit_dir: Path) -> dict[str, Path]:
    out_unit_dir = Path(out_unit_dir)
    branches_root = out_unit_dir / "branches"
    return {
        "branches_root": branches_root,
        "branches_clean": branches_root / "clean",
        "branches_raw": branches_root / "raw",
        "morphology": out_unit_dir / "morphology",
        "heuristics": out_unit_dir / "heuristics",
        "maps": out_unit_dir / "maps",
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


def _save_fig_pdf_and_png(*, fig: Any, pdf_path: Path, png_path: Path, dpi: int = 150) -> None:
    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    # Optional vector output for downstream compositing.
    try:
        fig.savefig(pdf_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    except Exception:
        pass


def _save_fig_png(*, fig: Any, png_path: Path, dpi: int) -> None:
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    # Optional vector output for downstream compositing.
    try:
        fig.savefig(png_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
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

    palette = _okabe_ito_palette()
    paths_raw = getattr(gtr, "_paths_raw", None)
    if not paths_raw:
        ax.text(0.5, 0.5, "no raw paths", ha="center", va="center", fontsize=10)
        return

    logger.info("Unit %s: plotting raw-branch velocities for %d raw paths", uid, len(paths_raw))

    handles = []
    labels = []

    # Keep these readable in the analysis montage.
    # (User request: increase fonts by ~100%.)
    label_fs = 22
    tick_fs = 18
    legend_fs = 16

    for raw_idx, raw_path in enumerate(paths_raw):
        try:
            color = palette[int(raw_idx) % len(palette)]

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

            # Markers
            ax.scatter(
                peaks[inlier_mask],
                dists[inlier_mask],
                s=18,
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
                    s=26,
                    marker="d",
                    color=color,
                    alpha=0.85,
                    edgecolors="k",
                    linewidths=0.3,
                )

            # Fit line
            try:
                v = float(velocity)
                b = float(offset)
                xs = np.linspace(float(np.min(peaks)), float(np.max(peaks)), 50)
                ys = v * xs + b
                (ln,) = ax.plot(xs, ys, color=color, lw=2.5, alpha=0.95)
            except Exception:
                (ln,) = ax.plot([], [], color=color, lw=2.5, alpha=0.95)

            handles.append(ln)
            labels.append(f"Raw {raw_idx}  r2={float(r2):.2f}" if r2 is not None else f"Raw {raw_idx}")
        except Exception:
            continue

    ax.set_xlabel("peak_time", fontsize=label_fs)
    ax.set_ylabel("distance", fontsize=label_fs)
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
        ax.legend(handles, labels, loc="best", fontsize=legend_fs, frameon=False, ncol=1)


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


def _compute_zoom_limits_from_xy(
    xy_points: list[list[float]],
    *,
    pad_frac: float = 0.08,
    pad_abs: float = 20.0,
) -> tuple[float, float, float, float]:
    """Compute (xmin, xmax, ymin, ymax) with padding for a set of [x,y] points."""

    xs = [p[0] for p in xy_points if (p is not None and len(p) >= 2)]
    ys = [p[1] for p in xy_points if (p is not None and len(p) >= 2)]
    if not xs or not ys:
        raise ValueError("No XY points")

    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))

    dx = max(xmax - xmin, 0.0)
    dy = max(ymax - ymin, 0.0)
    pad_x = max(pad_abs, pad_frac * dx)
    pad_y = max(pad_abs, pad_frac * dy)
    if dx == 0.0:
        pad_x = max(pad_x, pad_abs)
    if dy == 0.0:
        pad_y = max(pad_y, pad_abs)

    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (str, bytes)):
        return [x]
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, np.ndarray):
            return x.ravel().tolist()
    except Exception:
        pass
    try:
        return list(x)
    except Exception:
        return [x]


def _as_float_list(x: Any) -> list[float]:
    out: list[float] = []
    for v in _as_list(x):
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _as_int_list(x: Any) -> list[int]:
    out: list[int] = []
    for v in _as_list(x):
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


"""NOTE: summary + template movie are rendered via axon_velocity."""


def write_unit_reconstruction_pdfs(
    *,
    uid: Any,
    gtr: Any,
    locs_xy: Any,
    out_unit_dir: Path,
    force_restart: bool,
    logger: Any,
) -> dict[str, str]:
    """Write per-unit reconstruction PDFs.

    Returns a dict of output paths suitable for merging into a JSON summary.
    """

    outputs: dict[str, str] = {}

    layout = _compute_unit_output_layout(out_unit_dir=out_unit_dir)
    _ensure_unit_output_layout(layout)
    _maybe_migrate_legacy_unit_outputs(out_unit_dir=out_unit_dir, layout=layout)

    try:
        import numpy as np  # type: ignore[import-not-found]
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        logger.warning("Plotting dependencies unavailable: %s", e)
        return outputs

    # Extra plots requested: template + summary (axon_velocity plotting).
    template = getattr(gtr, "template", None)
    fs = getattr(gtr, "fs", None)

    template_png = out_unit_dir / "template.png"
    template_zoom_png = out_unit_dir / "template_zoom.png"
    summary_png = out_unit_dir / "summary.png"
    template_movie_gif = out_unit_dir / "template_movie.gif"

    write_template_movie_gif = str(os.getenv("AXON_RECON_RECON_WRITE_TEMPLATE_MOVIE_GIF", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }

    # Zoom region used for template/maps.
    branch_xy_points: list[list[float]] = []
    for br in _as_list(getattr(gtr, "branches", None)):
        chans: list[int] = []
        if isinstance(br, dict):
            chans = _as_int_list(br.get("channels"))
        else:
            try:
                chans = _as_int_list(getattr(br, "channels", None))
            except Exception:
                chans = []
        for ch in chans:
            if 0 <= ch < locs_xy.shape[0]:
                branch_xy_points.append([float(locs_xy[ch, 0]), float(locs_xy[ch, 1])])

    if ((not template_png.exists()) or force_restart) and (template is not None):
        try:
            from axon_velocity.plotting import plot_template as av_plot_template  # type: ignore[import-not-found]

            fig = plt.figure(figsize=(13, 10))
            ax = fig.add_subplot(111)
            with plt.rc_context(_white_bg_rc_params()):
                _ = av_plot_template(template=template, locations=locs_xy, ax=ax)
            _thin_lines_and_markers(ax, lw=0.45, ms=1.5, alpha=0.9)
            _force_white_background(fig)
            _save_fig_png(fig=fig, png_path=template_png, dpi=DPI_HI)
            plt.close(fig)
        except Exception as e:
            logger.warning("Template plotting failed for unit %s: %s", uid, e)

    if ((not template_zoom_png.exists()) or force_restart) and (template is not None) and branch_xy_points:
        try:
            from axon_velocity.plotting import plot_template as av_plot_template  # type: ignore[import-not-found]

            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_subplot(111)
            with plt.rc_context(_white_bg_rc_params()):
                _ = av_plot_template(template=template, locations=locs_xy, ax=ax)
            _thin_lines_and_markers(ax, lw=0.45, ms=1.5, alpha=0.9)
            xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(branch_xy_points)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            _force_white_background(fig)
            _save_fig_png(fig=fig, png_path=template_zoom_png, dpi=DPI_HI)
            plt.close(fig)
        except Exception as e:
            logger.warning("Template zoom plotting failed for unit %s: %s", uid, e)

    # Summary + template animation are generated via axon_velocity.

    if template_png.exists():
        outputs["template_png"] = str(template_png)
    if template_zoom_png.exists():
        outputs["template_zoom_png"] = str(template_zoom_png)
    # summary_png + template_movie_gif are written later.

    # Maps into <unit>/maps/
    maps_dir = Path(layout["maps"])
    if (template is not None) and (fs is not None):
        try:
            from axon_velocity.plotting import (  # type: ignore[import-not-found]
                plot_amplitude_map as av_plot_amplitude_map,
                plot_peak_latency_map as av_plot_peak_latency_map,
                plot_peak_std_map as av_plot_peak_std_map,
            )

            def _write_map(fn: Any, out_png: Path, out_zoom_png: Path) -> None:
                if (out_png.exists() and (not force_restart)) and (out_zoom_png.exists() or (not branch_xy_points)):
                    return
                fig = plt.figure(figsize=(8.5, 7.5))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = fn(ax=ax)
                _force_white_background(fig)
                _save_fig_png(fig=fig, png_path=out_png, dpi=DPI_HI)
                if branch_xy_points:
                    xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(branch_xy_points)
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    ax.set_aspect("equal", adjustable="box")
                    _save_fig_png(fig=fig, png_path=out_zoom_png, dpi=DPI_HI)
                plt.close(fig)

            amp_png = maps_dir / "amplitude_map.png"
            amp_zoom_png = maps_dir / "amplitude_map_zoom.png"
            _write_map(lambda ax: av_plot_amplitude_map(template, locs_xy, log=True, ax=ax), amp_png, amp_zoom_png)

            lat_png = maps_dir / "peak_latency_map.png"
            lat_zoom_png = maps_dir / "peak_latency_map_zoom.png"
            _write_map(lambda ax: av_plot_peak_latency_map(template, locs_xy, float(fs), ax=ax), lat_png, lat_zoom_png)

            std_png = maps_dir / "peak_std_map.png"
            std_zoom_png = maps_dir / "peak_std_map_zoom.png"
            _write_map(lambda ax: av_plot_peak_std_map(template, locs_xy, float(fs), ax=ax), std_png, std_zoom_png)

            for p, k in [
                (amp_png, "amplitude_map_png"),
                (amp_zoom_png, "amplitude_map_zoom_png"),
                (lat_png, "peak_latency_map_png"),
                (lat_zoom_png, "peak_latency_map_zoom_png"),
                (std_png, "peak_std_map_png"),
                (std_zoom_png, "peak_std_map_zoom_png"),
            ]:
                if p.exists():
                    outputs[k] = str(p)
        except Exception as e:
            logger.warning("Map plotting failed for unit %s: %s", uid, e)

    # Channel selection maps (Detection/Kurtosis/Delay/All) into maps/.
    try:
        chan_sets = {
            "detect": getattr(gtr, "_selected_channels_detect", None),
            "kurt": getattr(gtr, "_selected_channels_kurt", None),
            "delay": getattr(gtr, "_selected_channels_init", None),
            "all": getattr(gtr, "selected_channels", None),
        }

        def _as_ch_list(v: Any) -> list[int]:
            if v is None:
                return []
            try:
                return [int(x) for x in list(v)]
            except Exception:
                return []

        for name, raw in chan_sets.items():
            sel = _as_ch_list(raw)
            if not sel:
                continue
            out_png = maps_dir / f"channel_selection_{name}.png"
            if out_png.exists() and (not force_restart):
                continue
            fig = plt.figure(figsize=(8.5, 7.5))
            ax = fig.add_subplot(111)
            with plt.rc_context(_white_bg_rc_params()):
                ax.plot(locs_xy[:, 0], locs_xy[:, 1], marker=".", color="0.65", ls="", alpha=0.15)
                ax.plot(locs_xy[sel, 0], locs_xy[sel, 1], marker=".", color="k", ls="", alpha=0.75)
                try:
                    init_ch = int(getattr(gtr, "init_channel"))
                    ax.plot(locs_xy[init_ch, 0], locs_xy[init_ch, 1], marker="o", color="r", ms=4, ls="")
                except Exception:
                    pass
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                ax.set_title(f"Channel selection: {name}")
            _force_white_background(fig)
            _save_fig_png(fig=fig, png_path=out_png, dpi=DPI_HI)
            plt.close(fig)
            outputs[f"channel_selection_{name}_png"] = str(out_png)
    except Exception as e:
        logger.warning("Channel selection map plotting failed for unit %s: %s", uid, e)

    # Graph: nodes + edges as separate PNGs into maps/, plus a combined overview into heuristics/ for analysis.
    try:
        import matplotlib as mpl

        graph_nodes_png = maps_dir / "graph_nodes.png"
        graph_edges_png = maps_dir / "graph_edges.png"
        graph_combined_png = Path(layout["heuristics"]) / "graph_heuristics.png"

        if ((not graph_nodes_png.exists()) or force_restart) and hasattr(gtr, "_plot_nodes"):
            fig = plt.figure(figsize=(8.5, 7.5))
            ax = fig.add_subplot(111)
            with plt.rc_context(_white_bg_rc_params()):
                _ = getattr(gtr, "_plot_nodes")(ax=ax)
            _force_white_background(fig)
            try:
                import numpy as np  # type: ignore[import-not-found]

                node_h = getattr(gtr, "_node_heuristic", None)
                if node_h is not None:
                    node_h = np.asarray(node_h)
                    if node_h.size > 0:
                        norm = mpl.colors.Normalize(vmin=float(np.min(node_h)), vmax=float(np.max(node_h)))
                        sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("viridis"))
                        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="node heuristic")
            except Exception:
                pass
            _save_fig_png(fig=fig, png_path=graph_nodes_png, dpi=DPI_HI)
            plt.close(fig)

        if ((not graph_edges_png.exists()) or force_restart) and hasattr(gtr, "_plot_edges"):
            fig = plt.figure(figsize=(8.5, 7.5))
            ax = fig.add_subplot(111)
            with plt.rc_context(_white_bg_rc_params()):
                _ = getattr(gtr, "_plot_edges")(ax=ax)
            _force_white_background(fig)
            try:
                import numpy as np  # type: ignore[import-not-found]

                heuristics = []
                for _n1, _n2, d in getattr(gtr, "graph").edges.data():
                    heuristics.append(d.get("heur"))
                heur = np.asarray([h for h in heuristics if h is not None], dtype=float)
                if heur.size > 0:
                    norm = mpl.colors.Normalize(vmin=float(np.min(heur)), vmax=float(np.max(heur)))
                    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("rainbow"))
                    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="edge heuristic")
            except Exception:
                pass
            _save_fig_png(fig=fig, png_path=graph_edges_png, dpi=DPI_HI)
            plt.close(fig)

        if (not graph_combined_png.exists()) or force_restart:
            try:
                fig = plt.figure(figsize=(16, 7.5))
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                with plt.rc_context(_white_bg_rc_params()):
                    if hasattr(gtr, "_plot_nodes"):
                        _ = getattr(gtr, "_plot_nodes")(ax=ax1)
                    if hasattr(gtr, "_plot_edges"):
                        _ = getattr(gtr, "_plot_edges")(ax=ax2)
                ax1.set_title("Graph nodes")
                ax2.set_title("Graph edges")
                _force_white_background(fig)
                _save_fig_png(fig=fig, png_path=graph_combined_png, dpi=DPI_HI)
                plt.close(fig)
            except Exception:
                pass

        if graph_nodes_png.exists():
            outputs["graph_nodes_png"] = str(graph_nodes_png)
        if graph_edges_png.exists():
            outputs["graph_edges_png"] = str(graph_edges_png)
        if graph_combined_png.exists():
            outputs["graph_heuristics_png"] = str(graph_combined_png)
    except Exception as e:
        logger.warning("Graph plotting failed for unit %s: %s", uid, e)

    # Always write a simple morphology PDF.
    morphology_dir = Path(layout["morphology"])
    morphology_pdf = morphology_dir / "morphology.pdf"
    morphology_png = _with_suffix(morphology_pdf, ".png")
    if (not morphology_pdf.exists()) or force_restart:
        try:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.plot(locs_xy[:, 0], locs_xy[:, 1], marker=".", ls="", color="0.8", alpha=0.6, ms=3)

            branches_for_plot = _as_list(getattr(gtr, "branches", None))
            cm = plt.get_cmap("tab10")
            for bi, br in enumerate(branches_for_plot):
                chans = _as_int_list(br.get("channels"))
                if not chans:
                    continue
                xy = np.asarray([[locs_xy[ch, 0], locs_xy[ch, 1]] for ch in chans if 0 <= ch < locs_xy.shape[0]])
                if xy.size == 0:
                    continue
                color = cm(bi % 10)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2, alpha=0.9)
                ax.plot(xy[:, 0], xy[:, 1], marker="o", ls="", color=color, ms=3, alpha=0.9)

            ax.set_title(f"unit {uid} morphology")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            _save_fig_pdf_and_png(fig=fig, pdf_path=morphology_pdf, png_path=morphology_png, dpi=DPI_STD)
            plt.close(fig)
        except Exception as e:
            logger.warning("Morphology plotting failed for unit %s: %s", uid, e)

    if morphology_pdf.exists():
        outputs["morphology_pdf"] = str(morphology_pdf)
    if morphology_png.exists():
        outputs["morphology_png"] = str(morphology_png)

    # Zoomed-in morphology around the reconstruction.
    morphology_zoom_pdf = morphology_dir / "morphology_zoom.pdf"
    morphology_zoom_png = _with_suffix(morphology_zoom_pdf, ".png")
    if (not morphology_zoom_pdf.exists()) or force_restart:
        try:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            branches_for_plot = _as_list(getattr(gtr, "branches", None))
            cm = plt.get_cmap("tab10")
            branch_xy_points: list[list[float]] = []
            for bi, br in enumerate(branches_for_plot):
                chans = _as_int_list(br.get("channels"))
                if not chans:
                    continue
                xy = np.asarray([[locs_xy[ch, 0], locs_xy[ch, 1]] for ch in chans if 0 <= ch < locs_xy.shape[0]])
                if xy.size == 0:
                    continue
                color = cm(bi % 10)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2, alpha=0.95)
                ax.plot(xy[:, 0], xy[:, 1], marker="o", ls="", color=color, ms=3, alpha=0.95)
                for row in xy.tolist():
                    branch_xy_points.append([float(row[0]), float(row[1])])

            xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(branch_xy_points)
            in_view = (locs_xy[:, 0] >= xmin) & (locs_xy[:, 0] <= xmax) & (locs_xy[:, 1] >= ymin) & (locs_xy[:, 1] <= ymax)
            if np.any(in_view):
                ax.plot(locs_xy[in_view, 0], locs_xy[in_view, 1], marker=".", ls="", color="0.85", alpha=0.6, ms=3)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"unit {uid} morphology (zoom)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            _save_fig_pdf_and_png(fig=fig, pdf_path=morphology_zoom_pdf, png_path=morphology_zoom_png, dpi=DPI_STD)
            plt.close(fig)
        except Exception as e:
            logger.warning("Zoom morphology plotting failed for unit %s: %s", uid, e)

    if morphology_zoom_pdf.exists():
        outputs["morphology_zoom_pdf"] = str(morphology_zoom_pdf)
    if morphology_zoom_png.exists():
        outputs["morphology_zoom_png"] = str(morphology_zoom_png)

    heuristics_dir = Path(layout["heuristics"])

    # Raw + clean branches (axon_velocity built-in). This explicitly shows pre/post clean_paths.
    branches_raw_dir = Path(layout["branches_raw"])
    branches_clean_dir = Path(layout["branches_clean"])

    branches_pdf = branches_raw_dir / "branches_raw_clean.pdf"
    branches_png = _with_suffix(branches_pdf, ".png")
    if (not branches_pdf.exists()) or force_restart:
        try:
            # Custom two-panel plot with zoom for visibility.
            fig = plt.figure(figsize=(14, 6.5))
            ax_raw = fig.add_subplot(1, 2, 1)
            ax_clean = fig.add_subplot(1, 2, 2)
            plot_raw = getattr(gtr, "plot_raw_branches", None)
            plot_clean = getattr(gtr, "plot_clean_branches", None)
            with plt.rc_context(_white_bg_rc_params()):
                if callable(plot_raw):
                    _ = plot_raw(plot_full_template=True, ax=ax_raw)
                if callable(plot_clean):
                    _ = plot_clean(plot_full_template=True, ax=ax_clean)
            ax_raw.set_title("Raw branches")
            ax_clean.set_title("Clean branches")
            raw_xy_points: list[list[float]] = []
            for path in _as_list(getattr(gtr, "_paths_raw", None)):
                for ch in _as_int_list(path):
                    if 0 <= ch < locs_xy.shape[0]:
                        raw_xy_points.append([float(locs_xy[ch, 0]), float(locs_xy[ch, 1])])
            all_xy = raw_xy_points + branch_xy_points
            if all_xy:
                xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(all_xy)
                for ax in [ax_raw, ax_clean]:
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    ax.set_aspect("equal", adjustable="box")
            _force_white_background(fig)
            _save_fig_pdf_and_png(fig=fig, pdf_path=branches_pdf, png_path=branches_png, dpi=DPI_STD)
            plt.close(fig)
        except Exception as e:
            logger.warning("Branches (raw+clean) plotting failed for unit %s: %s", uid, e)

    if branches_pdf.exists():
        outputs["branches_raw_clean_pdf"] = str(branches_pdf)
    if branches_png.exists():
        outputs["branches_raw_clean_png"] = str(branches_png)

    # Raw branches only (axon_velocity built-in). This is useful when you want to see
    # everything before clean_paths duplicate-removal.
    raw_branches_pdf = branches_raw_dir / "branches_raw.pdf"
    raw_branches_png = _with_suffix(raw_branches_pdf, ".png")
    if (not raw_branches_pdf.exists()) or force_restart:
        try:
            plot_fn = getattr(gtr, "plot_raw_branches", None)
            paths_raw = getattr(gtr, "_paths_raw", None)
            if callable(plot_fn):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = plot_fn(plot_full_template=True, ax=ax)
                _minimal_axes(ax)
                _force_white_background(fig)
                _apply_raw_branch_colors(ax=ax, n_raw_paths=len(paths_raw) if paths_raw is not None else 0)
                _save_fig_pdf_and_png(fig=fig, pdf_path=raw_branches_pdf, png_path=raw_branches_png, dpi=DPI_STD)
                plt.close(fig)
        except Exception as e:
            logger.warning("Raw branches plotting failed for unit %s: %s", uid, e)

    if raw_branches_pdf.exists():
        outputs["branches_raw_pdf"] = str(raw_branches_pdf)
    if raw_branches_png.exists():
        outputs["branches_raw_png"] = str(raw_branches_png)

    # Zoomed raw branches (use raw path channels to compute limits).
    raw_branches_zoom_pdf = branches_raw_dir / "branches_raw_zoom.pdf"
    raw_branches_zoom_png = _with_suffix(raw_branches_zoom_pdf, ".png")
    if (not raw_branches_zoom_pdf.exists()) or force_restart:
        try:
            plot_fn = getattr(gtr, "plot_raw_branches", None)
            paths_raw = getattr(gtr, "_paths_raw", None)
            raw_xy_points: list[list[float]] = []
            for path in _as_list(paths_raw):
                for ch in _as_int_list(path):
                    if 0 <= ch < locs_xy.shape[0]:
                        raw_xy_points.append([float(locs_xy[ch, 0]), float(locs_xy[ch, 1])])

            if callable(plot_fn) and raw_xy_points:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = plot_fn(plot_full_template=True, ax=ax)
                xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(raw_xy_points)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                _minimal_axes(ax)
                _force_white_background(fig)
                _apply_raw_branch_colors(ax=ax, n_raw_paths=len(paths_raw) if paths_raw is not None else 0)
                _save_fig_pdf_and_png(fig=fig, pdf_path=raw_branches_zoom_pdf, png_path=raw_branches_zoom_png, dpi=DPI_STD)
                plt.close(fig)
        except Exception as e:
            logger.warning("Raw branches zoom plotting failed for unit %s: %s", uid, e)

    if raw_branches_zoom_pdf.exists():
        outputs["branches_raw_zoom_pdf"] = str(raw_branches_zoom_pdf)
    if raw_branches_zoom_png.exists():
        outputs["branches_raw_zoom_png"] = str(raw_branches_zoom_png)

    # Clean branches (axon_velocity built-in). This shows post-clean_paths results.
    clean_branches_pdf = branches_clean_dir / "branches_clean.pdf"
    clean_branches_png = _with_suffix(clean_branches_pdf, ".png")
    if (not clean_branches_pdf.exists()) or force_restart:
        try:
            plot_fn = getattr(gtr, "plot_clean_branches", None)
            if callable(plot_fn):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = plot_fn(plot_full_template=True, ax=ax)
                _force_white_background(fig)
                _save_fig_pdf_and_png(fig=fig, pdf_path=clean_branches_pdf, png_path=clean_branches_png, dpi=DPI_STD)
                plt.close(fig)
        except Exception as e:
            logger.warning("Clean branches plotting failed for unit %s: %s", uid, e)

    if clean_branches_pdf.exists():
        outputs["branches_clean_pdf"] = str(clean_branches_pdf)
    if clean_branches_png.exists():
        outputs["branches_clean_png"] = str(clean_branches_png)

    # Zoomed clean branches (use clean branch channel lists to compute limits).
    clean_branches_zoom_pdf = branches_clean_dir / "branches_clean_zoom.pdf"
    clean_branches_zoom_png = _with_suffix(clean_branches_zoom_pdf, ".png")
    if (not clean_branches_zoom_pdf.exists()) or force_restart:
        try:
            plot_fn = getattr(gtr, "plot_clean_branches", None)
            clean_xy_points: list[list[float]] = []
            for br in _as_list(getattr(gtr, "branches", None)):
                try:
                    chans = _as_int_list(br.get("channels"))
                except Exception:
                    chans = []
                for ch in chans:
                    if 0 <= ch < locs_xy.shape[0]:
                        clean_xy_points.append([float(locs_xy[ch, 0]), float(locs_xy[ch, 1])])

            if callable(plot_fn) and clean_xy_points:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = plot_fn(plot_full_template=True, ax=ax)
                xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(clean_xy_points)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                _force_white_background(fig)
                _save_fig_pdf_and_png(fig=fig, pdf_path=clean_branches_zoom_pdf, png_path=clean_branches_zoom_png, dpi=DPI_STD)
                plt.close(fig)
        except Exception as e:
            logger.warning("Clean branches zoom plotting failed for unit %s: %s", uid, e)

    if clean_branches_zoom_pdf.exists():
        outputs["branches_clean_zoom_pdf"] = str(clean_branches_zoom_pdf)
    if clean_branches_zoom_png.exists():
        outputs["branches_clean_zoom_png"] = str(clean_branches_zoom_png)

    # axon_velocity built-in branch velocities plot.
    # Clean branch velocities (for analysis panel).
    velocities_pdf = branches_clean_dir / "branch_velocities.pdf"
    velocities_png = _with_suffix(velocities_pdf, ".png")
    if (not velocities_pdf.exists()) or force_restart:
        try:
            plot_fn = getattr(gtr, "plot_velocities", None)
            if callable(plot_fn):
                with plt.rc_context(_white_bg_rc_params()):
                    fig = plot_fn()
                _force_white_background(fig)
                _recolor_noncolormapped_artists(fig)
                _save_fig_pdf_and_png(fig=fig, pdf_path=velocities_pdf, png_path=velocities_png, dpi=DPI_STD)
                plt.close(fig)
        except Exception as e:
            logger.warning("Branch velocities plotting failed for unit %s: %s", uid, e)

    if velocities_pdf.exists():
        outputs["branch_velocities_pdf"] = str(velocities_pdf)
    if velocities_png.exists():
        outputs["branch_velocities_png"] = str(velocities_png)

    # Raw velocity plot (separate) under branches/raw.
    raw_vel_pdf = branches_raw_dir / "branch_velocities_overlay.pdf"
    raw_vel_png = _with_suffix(raw_vel_pdf, ".png")
    if (not raw_vel_pdf.exists()) or force_restart:
        try:
            with plt.rc_context(_white_bg_rc_params()):
                # Thinner + taller so it fills the narrow analysis slot.
                fig = plt.figure(figsize=(4.4, 14.4))
                ax = fig.add_subplot(111)
                _plot_raw_branch_velocities(uid=uid, gtr=gtr, ax=ax, logger=logger)
                _force_white_background(fig)
                _save_fig_pdf_and_png(fig=fig, pdf_path=raw_vel_pdf, png_path=raw_vel_png, dpi=DPI_STD)
                plt.close(fig)
        except Exception as e:
            logger.warning("Raw branch velocities plotting failed for unit %s: %s", uid, e)

    if raw_vel_pdf.exists():
        outputs["branch_velocities_raw_overlay_pdf"] = str(raw_vel_pdf)
    if raw_vel_png.exists():
        outputs["branch_velocities_raw_overlay_png"] = str(raw_vel_png)

    per_branch_dir = branches_clean_dir
    per_branch_dir.mkdir(parents=True, exist_ok=True)

    overlay_pdf = per_branch_dir / "branch_velocities_overlay.pdf"
    overlay_png = _with_suffix(overlay_pdf, ".png")
    if (not overlay_pdf.exists()) or force_restart:
        try:
            branches_for_plot = _as_list(getattr(gtr, "branches", None))
            if branches_for_plot:
                # Thinner + taller so it fills the narrow analysis slot.
                fig = plt.figure(figsize=(4.4, 14.4))
                ax = fig.add_subplot(111)
                cm = plt.get_cmap("tab10")
                any_plotted = False
                for bi, br in enumerate(branches_for_plot):
                    peak_times = _as_float_list(br.get("peak_times"))
                    distances = _as_float_list(br.get("distances"))
                    if (len(peak_times) < 2) or (len(distances) != len(peak_times)):
                        continue
                    color = cm(bi % 10)
                    ax.scatter(peak_times, distances, s=12, alpha=0.7, color=color, label=f"b{bi}")
                    any_plotted = True
                    try:
                        velocity = br.get("velocity")
                        offset = br.get("offset")
                        if (velocity is not None) and (offset is not None):
                            v = float(velocity)
                            b = float(offset)
                            xs = np.linspace(min(peak_times), max(peak_times), 50)
                            ys = v * xs + b
                            ax.plot(xs, ys, lw=2, alpha=0.8, color=color)
                    except Exception:
                        pass

                if any_plotted:
                    ax.set_title(f"unit {uid} branch velocities (overlay)", fontsize=11)
                    ax.set_xlabel("peak_time", fontsize=11)
                    ax.set_ylabel("distance", fontsize=11)
                    try:
                        ax.tick_params(axis="both", which="major", labelsize=9)
                    except Exception:
                        pass
                    try:
                        _minimal_axes(ax)
                        ax.tick_params(top=False, right=False)
                    except Exception:
                        pass
                    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
                    _save_fig_pdf_and_png(fig=fig, pdf_path=overlay_pdf, png_path=overlay_png, dpi=DPI_STD)
                plt.close(fig)
        except Exception as e:
            logger.warning("Overlay velocity plotting failed for unit %s: %s", uid, e)

    if overlay_pdf.exists():
        outputs["branch_velocities_overlay_pdf"] = str(overlay_pdf)
    if overlay_png.exists():
        outputs["branch_velocities_overlay_png"] = str(overlay_png)

    for bi, br in enumerate(_as_list(getattr(gtr, "branches", None))):
        br_pdf = per_branch_dir / f"branch_{bi:02d}_velocity.pdf"
        br_png = _with_suffix(br_pdf, ".png")
        if br_pdf.exists() and (not force_restart):
            continue
        try:
            peak_times = _as_float_list(br.get("peak_times"))
            distances = _as_float_list(br.get("distances"))
            velocity = br.get("velocity")
            offset = br.get("offset")
            r2 = br.get("r2")

            if (len(peak_times) < 2) or (len(distances) != len(peak_times)):
                continue

            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.scatter(peak_times, distances, s=12, alpha=0.8)
            ax.set_xlabel("peak_time")
            ax.set_ylabel("distance")

            try:
                if (velocity is not None) and (offset is not None):
                    v = float(velocity)
                    b = float(offset)
                    xs = np.linspace(min(peak_times), max(peak_times), 50)
                    ys = v * xs + b
                    ax.plot(xs, ys, lw=2, alpha=0.8)
            except Exception:
                pass

            title_bits = [f"unit {uid}", f"branch {bi}"]
            try:
                if velocity is not None:
                    title_bits.append(f"v={float(velocity):.3g}")
            except Exception:
                pass
            try:
                if r2 is not None:
                    title_bits.append(f"r2={float(r2):.3g}")
            except Exception:
                pass
            ax.set_title("  ".join(title_bits))

            _save_fig_pdf_and_png(fig=fig, pdf_path=br_pdf, png_path=br_png, dpi=DPI_STD)
            plt.close(fig)
        except Exception:
            continue

        if br_pdf.exists():
            outputs[f"branch_{bi:02d}_velocity_pdf"] = str(br_pdf)
        if br_png.exists():
            outputs[f"branch_{bi:02d}_velocity_png"] = str(br_png)

    if write_template_movie_gif and ((not template_movie_gif.exists()) or force_restart) and (template is not None):
        try:
            from axon_velocity.plotting import play_template_map as av_play_template_map  # type: ignore[import-not-found]
            from matplotlib.animation import PillowWriter

            fig = plt.figure(figsize=(7.2, 6.2))
            ax = fig.add_subplot(111)
            with plt.rc_context(_white_bg_rc_params()):
                ani = av_play_template_map(
                    template,
                    locs_xy,
                    gtr=gtr,
                    ax=ax,
                    cmap="seismic",
                    log=False,
                    skip_frames=2,
                    interval=40,
                )
            _force_white_background(fig)
            template_movie_gif.parent.mkdir(parents=True, exist_ok=True)
            ani.save(str(template_movie_gif), writer=PillowWriter(fps=12), dpi=DPI_STD)
            plt.close(fig)
        except Exception as e:
            logger.warning("Template animation failed for unit %s: %s", uid, e)

    if template_movie_gif.exists():
        outputs["template_movie_gif"] = str(template_movie_gif)

    if ((not summary_png.exists()) or force_restart):
        try:
            from axon_velocity.plotting import plot_axon_summary as av_plot_axon_summary  # type: ignore[import-not-found]

            fig = plt.figure(figsize=(12, 9))
            with plt.rc_context(_white_bg_rc_params()):
                _ = av_plot_axon_summary(gtr, fig=fig)
            _force_white_background(fig)
            _save_fig_png(fig=fig, png_path=summary_png, dpi=DPI_HI)
            plt.close(fig)
        except Exception as e:
            logger.warning("Summary plotting failed for unit %s: %s", uid, e)

    if summary_png.exists():
        outputs["summary_png"] = str(summary_png)

    return outputs


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
