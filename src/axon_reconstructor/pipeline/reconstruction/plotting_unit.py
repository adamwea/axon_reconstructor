"""Per-unit reconstruction plotting outputs (internal)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .plotting_core import (
    DPI_HI,
    DPI_STD,
    _compute_unit_output_layout,
    _ensure_unit_output_layout,
    _force_white_background,
    _maybe_migrate_legacy_unit_outputs,
    _minimal_axes,
    _plot_raw_branch_velocities,
    _read_json,
    _recolor_noncolormapped_artists,
    _save_fig_pdf_and_png,
    _save_fig_png,
    _white_bg_rc_params,
    _with_suffix,
)
from .plotting_summary import _plot_summary_from_parts, compute_raw_branches_for_summary


def _thin_lines_and_markers(ax: Any, *, lw: float = 0.45, ms: float = 1.5, alpha: float = 0.9) -> None:
    try:
        for line in getattr(ax, "lines", []):
            try:
                line.set_linewidth(lw)
            except Exception:
                pass
            try:
                line.set_markersize(ms)
            except Exception:
                pass
            try:
                line.set_alpha(alpha)
            except Exception:
                pass
    except Exception:
        pass

    try:
        for coll in getattr(ax, "collections", []):
            try:
                coll.set_alpha(alpha)
            except Exception:
                pass
            try:
                coll.set_linewidths(lw)
            except Exception:
                pass
    except Exception:
        pass

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
    summary_clean_png = out_unit_dir / "summary_clean.png"
    summary_raw_png = out_unit_dir / "summary_raw.png"
    template_movie_gif = out_unit_dir / "template_movie.gif"

    write_template_movie_gif = str(os.getenv("AXON_RECON_RECON_WRITE_TEMPLATE_MOVIE_GIF", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }

    crop_template_movie_gif = (
        str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_CROP", "1")).strip().lower()
        not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
    )

    template_movie_gif_cmap = str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_CMAP", "coolwarm")).strip() or "coolwarm"
    try:
        template_movie_gif_clip_quantile = float(
            str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_CLIP_QUANTILE", "0.995")).strip()
        )
    except Exception:
        template_movie_gif_clip_quantile = 0.995

    write_template_movie_gif_colorbar = (
        str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_COLORBAR", "1")).strip().lower()
        not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
    )
    template_movie_gif_colorbar_label = str(
        os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_COLORBAR_LABEL", "")
    ).strip()

    write_template_movie_gif_time_counter = (
        str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_TIME_COUNTER", "1")).strip().lower()
        not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
    )

    zoom_template_movie_gif = (
        str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_ZOOM", "1")).strip().lower()
        not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
    )
    try:
        template_movie_zoom_pad_frac = float(
            str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_ZOOM_PAD_FRAC", "0.08")).strip()
        )
    except Exception:
        template_movie_zoom_pad_frac = 0.08
    try:
        template_movie_zoom_pad_abs = float(
            str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_ZOOM_PAD_ABS", "20.0")).strip()
        )
    except Exception:
        template_movie_zoom_pad_abs = 20.0

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

    contributing_channels: list[int] = []
    try:
        contrib_set: set[int] = set()
        for br in _as_list(getattr(gtr, "branches", None)):
            if isinstance(br, dict):
                chans = _as_int_list(br.get("channels"))
            else:
                try:
                    chans = _as_int_list(getattr(br, "channels", None))
                except Exception:
                    chans = []
            for ch in chans:
                if 0 <= ch < locs_xy.shape[0]:
                    contrib_set.add(int(ch))
        contributing_channels = sorted(contrib_set)
    except Exception:
        contributing_channels = []

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
            if callable(plot_fn):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = plot_fn(plot_full_template=True, ax=ax)
                _minimal_axes(ax)
                _force_white_background(fig)
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
            from types import SimpleNamespace

            fig = plt.figure(figsize=(7.2, 6.2))
            ax = fig.add_subplot(111)

            template_movie_skip_frames = 2

            template_for_movie = template
            locs_xy_for_movie = locs_xy
            gtr_for_movie: Any | None = gtr

            # Prefer raw branches for the template movie overlay (matches summary_raw.png).
            branches_for_movie: list[dict[str, Any]] = []
            try:
                branches_for_movie = compute_raw_branches_for_summary(uid=uid, gtr=gtr)
            except Exception:
                branches_for_movie = []

            contributing_channels_for_movie: list[int] = []
            try:
                contrib: set[int] = set()
                for br in _as_list(branches_for_movie):
                    if not isinstance(br, dict):
                        continue
                    for ch in _as_int_list(br.get("channels")):
                        contrib.add(int(ch))
                contributing_channels_for_movie = sorted(contrib)
            except Exception:
                contributing_channels_for_movie = []

            branch_xy_points_for_movie: list[list[float]] = []
            try:
                for br in _as_list(branches_for_movie):
                    if not isinstance(br, dict):
                        continue
                    for ch in _as_int_list(br.get("channels")):
                        if 0 <= ch < locs_xy.shape[0]:
                            branch_xy_points_for_movie.append([float(locs_xy[ch, 0]), float(locs_xy[ch, 1])])
            except Exception:
                branch_xy_points_for_movie = []

            # Performance: crop template+locations to a square ROI around the contributing channels.
            # This reduces the probe size passed into probe.to_image() for each animation frame.
            if crop_template_movie_gif and contributing_channels_for_movie:
                try:
                    xy_contrib = [[float(locs_xy[ch, 0]), float(locs_xy[ch, 1])] for ch in contributing_channels_for_movie]
                    xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(
                        xy_contrib,
                        pad_frac=template_movie_zoom_pad_frac,
                        pad_abs=template_movie_zoom_pad_abs,
                    )
                    # Make it square in XY.
                    cx = 0.5 * (xmin + xmax)
                    cy = 0.5 * (ymin + ymax)
                    w = float(max(xmax - xmin, ymax - ymin))
                    xmin, xmax = cx - 0.5 * w, cx + 0.5 * w
                    ymin, ymax = cy - 0.5 * w, cy + 0.5 * w

                    xs = locs_xy[:, 0]
                    ys = locs_xy[:, 1]
                    mask = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
                    crop_inds = np.where(mask)[0].astype(int).tolist()

                    # If the ROI got too small for any reason, fall back to just the contributing channels.
                    if len(crop_inds) < max(4, min(16, len(contributing_channels_for_movie))):
                        crop_inds = list(contributing_channels_for_movie)

                    if crop_inds:
                        locs_xy_for_movie = locs_xy[crop_inds, :]
                        template_for_movie = template[crop_inds, :]

                        # Remap branches into the cropped index space so axon_velocity can draw them.
                        remap = {int(old): int(new) for new, old in enumerate(crop_inds)}
                        branches_cropped: list[dict[str, Any]] = []
                        for br in _as_list(branches_for_movie):
                            if not isinstance(br, dict):
                                continue
                            br_ch = [int(c) for c in _as_int_list(br.get("channels")) if int(c) in remap]
                            if len(br_ch) < 2:
                                continue
                            branches_cropped.append({"channels": [remap[c] for c in br_ch]})
                        gtr_for_movie = (
                            SimpleNamespace(branches=branches_cropped, locations=locs_xy_for_movie)
                            if branches_cropped
                            else None
                        )
                except Exception:
                    template_for_movie = template
                    locs_xy_for_movie = locs_xy
                    gtr_for_movie = gtr

            # Reduce saturated colors by clipping extreme amplitudes (keeps sign).
            try:
                q = float(template_movie_gif_clip_quantile)
                if 0.0 < q < 1.0:
                    vals = np.asarray(template_for_movie)
                    vmax = float(np.quantile(np.abs(vals), q))
                    if vmax > 0.0:
                        template_for_movie = np.clip(vals, -vmax, vmax)
            except Exception:
                pass

            with plt.rc_context(_white_bg_rc_params()):
                ani = av_play_template_map(
                    template_for_movie,
                    locs_xy_for_movie,
                    gtr=gtr_for_movie,
                    ax=ax,
                    cmap=template_movie_gif_cmap,
                    log=False,
                    skip_frames=template_movie_skip_frames,
                    interval=40,
                )

            # Add a colorbar to interpret color intensity (use any of the images; they share vmin/vmax).
            if write_template_movie_gif_colorbar:
                try:
                    images = getattr(ax, "images", None)
                    if images:
                        mappable = images[0]
                        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
                        try:
                            cbar.ax.tick_params(colors="#222222")
                        except Exception:
                            pass
                        if template_movie_gif_colorbar_label:
                            try:
                                cbar.set_label(template_movie_gif_colorbar_label, color="#222222")
                            except Exception:
                                pass
                except Exception:
                    pass

            # Add a time counter overlay in the corner.
            if write_template_movie_gif_time_counter:
                try:
                    fs_hz = float(getattr(gtr, "fs", None) or 0.0)
                except Exception:
                    fs_hz = 0.0
                try:
                    framedata = getattr(ani, "_framedata", None)
                except Exception:
                    framedata = None
                if fs_hz > 0.0 and framedata:
                    for fi, artists in enumerate(framedata):
                        t_sec = (float(fi) * float(template_movie_skip_frames)) / fs_hz
                        if t_sec < 1.0:
                            t_val = t_sec * 1000.0
                            label = f"t={t_val:.0f} ms" if t_val >= 10.0 else f"t={t_val:.1f} ms"
                        else:
                            label = f"t={t_sec:.2f} s"
                        try:
                            txt = ax.text(
                                0.02,
                                0.98,
                                label,
                                transform=ax.transAxes,
                                ha="left",
                                va="top",
                                fontsize=10,
                                color="#222222",
                                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
                            )
                            # Each frame is a list of artists; append our label for blitting.
                            try:
                                artists.append(txt)
                            except Exception:
                                pass
                        except Exception:
                            continue

            # axon_velocity draws morphology branches as black lines; soften them.
            try:
                for ln in (ax.get_lines() or []):
                    try:
                        ln.set_color("#666666")
                    except Exception:
                        pass
                    try:
                        ln.set_alpha(0.55)
                    except Exception:
                        pass
                    try:
                        ln.set_linewidth(1.0)
                    except Exception:
                        pass
            except Exception:
                pass

            # If we didn't crop, still optionally zoom the viewport.
            if (not crop_template_movie_gif) and zoom_template_movie_gif and branch_xy_points_for_movie:
                xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(
                    branch_xy_points_for_movie,
                    pad_frac=template_movie_zoom_pad_frac,
                    pad_abs=template_movie_zoom_pad_abs,
                )
                # Keep it square (matches crop behavior).
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                w = float(max(xmax - xmin, ymax - ymin))
                ax.set_xlim(cx - 0.5 * w, cx + 0.5 * w)
                ax.set_ylim(cy - 0.5 * w, cy + 0.5 * w)
                ax.set_aspect("equal", adjustable="box")
            _force_white_background(fig)
            template_movie_gif.parent.mkdir(parents=True, exist_ok=True)
            ani.save(
                str(template_movie_gif),
                writer=PillowWriter(fps=12),
                dpi=DPI_STD,
                savefig_kwargs={"facecolor": "white"},
            )
            plt.close(fig)
        except Exception as e:
            logger.warning("Template animation failed for unit %s: %s", uid, e)

    if template_movie_gif.exists():
        outputs["template_movie_gif"] = str(template_movie_gif)

    # Summary plots: one for clean branches + one for raw paths.
    # We avoid calling axon_velocity.plot_axon_summary directly so we can swap
    # the branch list (raw vs clean) while still using axon_velocity plotting helpers.
    try:
        fs_hz = float(getattr(gtr, "fs", 10_000.0))
    except Exception:
        fs_hz = 10_000.0

    branches_clean = []
    try:
        branches_clean = [dict(b) for b in _as_list(getattr(gtr, "branches", None)) if isinstance(b, dict)]
    except Exception:
        branches_clean = []

    branches_raw = []
    try:
        branches_raw = compute_raw_branches_for_summary(uid=uid, gtr=gtr)
    except Exception:
        branches_raw = []

    if (force_restart or (not summary_clean_png.exists())) and branches_clean:
        try:
            with plt.rc_context(_white_bg_rc_params()):
                fig = _plot_summary_from_parts(
                    template_ch_by_t=np.asarray(template),
                    locs_xy=locs_xy,
                    fs_hz=fs_hz,
                    init_channel=int(getattr(gtr, "init_channel", 0)),
                    branches=branches_clean,
                    title_suffix=" (clean)",
                )
            _force_white_background(fig)
            _save_fig_png(fig=fig, png_path=summary_clean_png, dpi=DPI_HI)
            plt.close(fig)
        except Exception as e:
            logger.warning("Clean summary plotting failed for unit %s: %s", uid, e)

    if (force_restart or (not summary_raw_png.exists())):
        try:
            # Do not fall back: only write raw summary if raw branches exist.
            if not branches_raw:
                logger.warning(
                    "Skipping raw summary for unit %s because branches_raw are missing/empty; run branches_raw-only mode first",
                    uid,
                )
            else:
                with plt.rc_context(_white_bg_rc_params()):
                    fig = _plot_summary_from_parts(
                        template_ch_by_t=np.asarray(template),
                        locs_xy=locs_xy,
                        fs_hz=fs_hz,
                        init_channel=int(getattr(gtr, "init_channel", 0)),
                        branches=branches_raw,
                        title_suffix=" (raw)",
                    )
                _force_white_background(fig)
                _save_fig_png(fig=fig, png_path=summary_raw_png, dpi=DPI_HI)
                plt.close(fig)
        except Exception as e:
            logger.warning("Raw summary plotting failed for unit %s: %s", uid, e)

    # Back-compat: keep summary.png as clean summary.
    try:
        if summary_clean_png.exists() and (force_restart or (not summary_png.exists())):
            import shutil

            shutil.copyfile(summary_clean_png, summary_png)
    except Exception:
        pass

    for p, k in [
        (summary_clean_png, "summary_clean_png"),
        (summary_raw_png, "summary_raw_png"),
        (summary_png, "summary_png"),
    ]:
        if p.exists():
            outputs[k] = str(p)
            svg = p.with_suffix(".svg")
            if svg.exists():
                outputs[k.replace("_png", "_svg")] = str(svg)

    return outputs


