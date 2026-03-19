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


def _normalize_template_channel_scope(raw: Any) -> str:
    """Normalize template channel-scope option for full-template plotting.

    Supported values:
    - contributing_channels (aliases: contributing, branches)
    - recorded_channels (aliases: recorded, recorded channel(s))
    - all_channels (aliases: all, all channel(s))
    """

    v = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if v in {"contributing", "contributing_channel", "contributing_channels", "branches"}:
        return "contributing_channels"
    if v in {"recorded", "recorded_channel", "recorded_channels"}:
        return "recorded_channels"
    if v in {"all", "all_channel", "all_channels"}:
        return "all_channels"
    return "all_channels"


def _make_square_limits(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    center_x: float | None = None,
    center_y: float | None = None,
) -> tuple[float, float, float, float]:
    """Expand the short side so (x, y) span forms a square."""

    w = float(xmax - xmin)
    h = float(ymax - ymin)
    side = max(w, h)

    cx = float((xmin + xmax) / 2.0) if center_x is None else float(center_x)
    cy = float((ymin + ymax) / 2.0) if center_y is None else float(center_y)
    half = float(side / 2.0)

    return cx - half, cx + half, cy - half, cy + half


def _apply_template_style(
    *,
    fig: Any,
    ax: Any,
    background: str,
    signal_color: str,
) -> None:
    """Apply template plot styling for background and signal traces."""

    bg = str(background or "").strip().lower()
    sig = str(signal_color or "").strip() or "white"

    if bg == "black":
        try:
            fig.patch.set_facecolor("black")
        except Exception:
            pass
        try:
            ax.set_facecolor("black")
        except Exception:
            pass

        try:
            ax.tick_params(colors="white")
        except Exception:
            pass
        try:
            for spine in ax.spines.values():
                spine.set_color("white")
        except Exception:
            pass
        try:
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
        except Exception:
            pass
    else:
        _force_white_background(fig)

    try:
        _recolor_noncolormapped_artists(ax, color=sig)
    except Exception:
        pass
    # Fallback recoloring for artists not handled by helper.
    try:
        for ln in getattr(ax, "lines", []) or []:
            try:
                ln.set_color(sig)
            except Exception:
                pass
    except Exception:
        pass
    try:
        for coll in getattr(ax, "collections", []) or []:
            try:
                coll.set_color(sig)
            except Exception:
                pass
            try:
                coll.set_edgecolor(sig)
            except Exception:
                pass
            try:
                coll.set_facecolor(sig)
            except Exception:
                pass
    except Exception:
        pass


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
    write_template_movie_gif: bool | None = None,
    branches_root_relpath: str = "branches",
    branches_clean_dir_relpath: str = "branches/clean",
    branches_raw_dir_relpath: str = "branches/raw",
    morphology_dir_relpath: str = "morphology",
    heuristics_dir_relpath: str = "heuristics",
    maps_dir_relpath: str = "maps",
    template_relpath: str = "template.png",
    template_zoom_relpath: str = "template_zoom.png",
    template_movie_relpath: str = "template_movie.gif",
    summary_relpath: str = "summary.png",
    summary_clean_relpath: str = "summary_clean.png",
    summary_raw_relpath: str = "summary_raw.png",
    amplitude_map_relpath: str = "maps/amplitude_map.png",
    amplitude_map_zoom_relpath: str = "maps/amplitude_map_zoom.png",
    peak_latency_map_relpath: str = "maps/peak_latency_map.png",
    peak_latency_map_zoom_relpath: str = "maps/peak_latency_map_zoom.png",
    peak_std_map_relpath: str = "maps/peak_std_map.png",
    peak_std_map_zoom_relpath: str = "maps/peak_std_map_zoom.png",
    channel_selection_detect_relpath: str = "maps/channel_selection_detect.png",
    channel_selection_kurt_relpath: str = "maps/channel_selection_kurt.png",
    channel_selection_delay_relpath: str = "maps/channel_selection_delay.png",
    channel_selection_all_relpath: str = "maps/channel_selection_all.png",
    graph_nodes_relpath: str = "maps/graph_nodes.png",
    graph_edges_relpath: str = "maps/graph_edges.png",
    graph_heuristics_relpath: str = "heuristics/graph_heuristics.png",
    morphology_pdf_relpath: str = "morphology/morphology.pdf",
    morphology_zoom_pdf_relpath: str = "morphology/morphology_zoom.pdf",
    branches_raw_clean_pdf_relpath: str = "branches/raw/branches_raw_clean.pdf",
    branches_raw_pdf_relpath: str = "branches/raw/branches_raw.pdf",
    branches_raw_zoom_pdf_relpath: str = "branches/raw/branches_raw_zoom.pdf",
    branches_clean_pdf_relpath: str = "branches/clean/branches_clean.pdf",
    branches_clean_zoom_pdf_relpath: str = "branches/clean/branches_clean_zoom.pdf",
    branch_velocities_pdf_relpath: str = "branches/clean/branch_velocities.pdf",
    branch_velocities_raw_overlay_pdf_relpath: str = "branches/raw/branch_velocities_overlay.pdf",
    branch_velocities_overlay_pdf_relpath: str = "branches/clean/branch_velocities_overlay.pdf",
    branch_velocity_template_relpath: str = "branches/clean/branch_{index:02d}_velocity",
    write_template: bool = True,
    write_template_zoom: bool = True,
    write_template_movie: bool = True,
    write_summary: bool = True,
    write_summary_clean: bool = True,
    write_summary_raw: bool = True,
    write_amplitude_map: bool = True,
    write_amplitude_map_zoom: bool = True,
    write_peak_latency_map: bool = True,
    write_peak_latency_map_zoom: bool = True,
    write_peak_std_map: bool = True,
    write_peak_std_map_zoom: bool = True,
    write_channel_selection_detect: bool = True,
    write_channel_selection_kurt: bool = True,
    write_channel_selection_delay: bool = True,
    write_channel_selection_all: bool = True,
    write_graph_nodes: bool = True,
    write_graph_edges: bool = True,
    write_graph_heuristics: bool = True,
    write_morphology_pdf: bool = True,
    write_morphology_zoom_pdf: bool = True,
    write_branches_raw_clean_pdf: bool = True,
    write_branches_raw_pdf: bool = True,
    write_branches_raw_zoom_pdf: bool = True,
    write_branches_clean_pdf: bool = True,
    write_branches_clean_zoom_pdf: bool = True,
    write_branch_velocities_pdf: bool = True,
    write_branch_velocities_raw_overlay_pdf: bool = True,
    write_branch_velocities_overlay_pdf: bool = True,
    write_branch_velocity_template: bool = True,
    per_unit_outputs_schema: dict[str, Any] | None = None,
    logger: Any,
) -> dict[str, str]:
    """Write per-unit reconstruction PDFs.

    Returns a dict of output paths suitable for merging into a JSON summary.
    """

    outputs: dict[str, str] = {}

    def _unit_path(relpath: str) -> Path:
        return Path(out_unit_dir) / Path(str(relpath)).expanduser()

    def _unit_output_path(relpath: str) -> Path:
        p = _unit_path(relpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    layout = _compute_unit_output_layout(
        out_unit_dir=out_unit_dir,
        branches_root_relpath=branches_root_relpath,
        branches_clean_relpath=branches_clean_dir_relpath,
        branches_raw_relpath=branches_raw_dir_relpath,
        morphology_relpath=morphology_dir_relpath,
        heuristics_relpath=heuristics_dir_relpath,
        maps_relpath=maps_dir_relpath,
    )
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

    schema = per_unit_outputs_schema if isinstance(per_unit_outputs_schema, dict) else {}

    def _schema_get(path: tuple[str, ...], default: Any = None) -> Any:
        cur: Any = schema
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _normalize_relpath(relpath: str, *, ext: str) -> str:
        raw = str(relpath).strip()
        if not raw:
            return raw
        p = Path(raw)
        if p.suffix == "":
            p = p.with_suffix(ext)
        return str(p)

    def _resolve_png_spec(
        *,
        cfg_path: tuple[str, ...],
        default_relpath: str,
        default_write_png: bool,
        default_write_svg: bool = False,
    ) -> tuple[bool, bool, str, str]:
        cfg = _schema_get(cfg_path, default={})
        if isinstance(cfg, dict):
            rel = _normalize_relpath(str(cfg.get("relpath", default_relpath)), ext=".png")
            w_png = bool(cfg.get("write_png", default_write_png))
            w_svg = bool(cfg.get("write_svg", default_write_svg))
        else:
            rel = _normalize_relpath(default_relpath, ext=".png")
            w_png = bool(default_write_png)
            w_svg = bool(default_write_svg)
        svg_rel = str(Path(rel).with_suffix(".svg"))
        return w_png, w_svg, rel, svg_rel

    def _resolve_pdf_spec(
        *,
        cfg_path: tuple[str, ...],
        default_relpath: str,
        default_write_pdf: bool,
        default_write_png: bool = True,
        default_write_svg: bool = False,
    ) -> tuple[bool, bool, bool, str, str, str]:
        cfg = _schema_get(cfg_path, default={})
        if isinstance(cfg, dict):
            rel = _normalize_relpath(str(cfg.get("relpath", default_relpath)), ext=".pdf")
            w_pdf = bool(cfg.get("write_pdf", default_write_pdf))
            w_png = bool(cfg.get("write_png", default_write_png))
            w_svg = bool(cfg.get("write_svg", default_write_svg))
        else:
            rel = _normalize_relpath(default_relpath, ext=".pdf")
            w_pdf = bool(default_write_pdf)
            w_png = bool(default_write_png)
            w_svg = bool(default_write_svg)
        png_rel = str(Path(rel).with_suffix(".png"))
        svg_rel = str(Path(rel).with_suffix(".svg"))
        return w_pdf, w_png, w_svg, rel, png_rel, svg_rel

    def _resolve_gif_spec(*, cfg_path: tuple[str, ...], default_relpath: str, default_write_gif: bool) -> tuple[bool, str]:
        cfg = _schema_get(cfg_path, default={})
        if isinstance(cfg, dict):
            rel = _normalize_relpath(str(cfg.get("relpath", default_relpath)), ext=".gif")
            w_gif = bool(cfg.get("write_gif", default_write_gif))
        else:
            rel = _normalize_relpath(default_relpath, ext=".gif")
            w_gif = bool(default_write_gif)
        return w_gif, rel

    # Support both nested and flat template schema shapes:
    # nested: template.full.{write_png,write_svg,relpath,channel_scope}
    # flat:   template.{write_png,write_svg,relpath,channel_scope}
    flat_template_cfg = _schema_get(("template",), default={})
    if not isinstance(flat_template_cfg, dict):
        flat_template_cfg = {}

    use_flat_template_cfg = False
    if isinstance(flat_template_cfg, dict) and flat_template_cfg:
        has_nested_full = isinstance(_schema_get(("template", "full"), default=None), dict)
        flat_has_direct_fields = any(
            k in flat_template_cfg
            for k in (
                "write_png",
                "write_svg",
                "relpath",
                "channel_scope",
                "background",
                "signal_color",
                "zoom_priority",
            )
        )
        use_flat_template_cfg = bool(flat_has_direct_fields and (not has_nested_full))

    tpl_cfg_path = ("template",) if use_flat_template_cfg else ("template", "full")
    tpl_png, tpl_svg, template_relpath, template_svg_relpath = _resolve_png_spec(
        cfg_path=tpl_cfg_path,
        default_relpath=template_relpath,
        default_write_png=bool(write_template),
    )
    write_template = bool(tpl_png or tpl_svg)
    write_template_png = bool(tpl_png)
    write_template_svg = bool(tpl_svg)

    template_channel_scope = _normalize_template_channel_scope(
        _schema_get(("template", "full", "channel_scope"), None)
    )
    template_force_center_soma = bool(_schema_get(("template", "full", "force_center_soma"), False))
    template_force_square_aspect = bool(_schema_get(("template", "full", "force_square_aspect"), False))
    template_background = str(_schema_get(("template", "full", "background"), "white") or "white")
    template_signal_color = str(_schema_get(("template", "full", "signal_color"), "black") or "black")
    if use_flat_template_cfg:
        template_channel_scope = _normalize_template_channel_scope(
            flat_template_cfg.get("channel_scope", template_channel_scope)
        )
        template_force_center_soma = bool(
            flat_template_cfg.get("force_center_soma", template_force_center_soma)
        )
        template_force_square_aspect = bool(
            flat_template_cfg.get("force_square_aspect", template_force_square_aspect)
        )
        template_background = str(flat_template_cfg.get("background", template_background) or template_background)
        template_signal_color = str(flat_template_cfg.get("signal_color", template_signal_color) or template_signal_color)

    tplz_png, tplz_svg, template_zoom_relpath, template_zoom_svg_relpath = _resolve_png_spec(
        cfg_path=("template", "zoom"),
        default_relpath=template_zoom_relpath,
        default_write_png=bool(write_template_zoom),
    )
    write_template_zoom = bool(tplz_png or tplz_svg)
    write_template_zoom_png = bool(tplz_png)
    write_template_zoom_svg = bool(tplz_svg)

    mov_gif, template_movie_relpath = _resolve_gif_spec(
        cfg_path=("movie", "template"),
        default_relpath=template_movie_relpath,
        default_write_gif=bool(write_template_movie),
    )
    write_template_movie = bool(mov_gif)

    sum_png, sum_svg, summary_relpath, summary_svg_relpath = _resolve_png_spec(
        cfg_path=("summary", "full"),
        default_relpath=summary_relpath,
        default_write_png=bool(write_summary),
    )
    write_summary = bool(sum_png or sum_svg)
    write_summary_png = bool(sum_png)
    write_summary_svg = bool(sum_svg)

    sumc_png, sumc_svg, summary_clean_relpath, summary_clean_svg_relpath = _resolve_png_spec(
        cfg_path=("summary", "clean"),
        default_relpath=summary_clean_relpath,
        default_write_png=bool(write_summary_clean),
    )
    write_summary_clean = bool(sumc_png or sumc_svg)
    write_summary_clean_png = bool(sumc_png)
    write_summary_clean_svg = bool(sumc_svg)

    sumr_png, sumr_svg, summary_raw_relpath, summary_raw_svg_relpath = _resolve_png_spec(
        cfg_path=("summary", "raw"),
        default_relpath=summary_raw_relpath,
        default_write_png=bool(write_summary_raw),
    )
    write_summary_raw = bool(sumr_png or sumr_svg)
    write_summary_raw_png = bool(sumr_png)
    write_summary_raw_svg = bool(sumr_svg)

    amp_png, amp_svg, amplitude_map_relpath, amplitude_map_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "amplitude", "full"),
        default_relpath=amplitude_map_relpath,
        default_write_png=bool(write_amplitude_map),
    )
    write_amplitude_map = bool(amp_png or amp_svg)
    write_amplitude_map_png = bool(amp_png)
    write_amplitude_map_svg = bool(amp_svg)

    ampz_png, ampz_svg, amplitude_map_zoom_relpath, amplitude_map_zoom_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "amplitude", "zoom"),
        default_relpath=amplitude_map_zoom_relpath,
        default_write_png=bool(write_amplitude_map_zoom),
    )
    write_amplitude_map_zoom = bool(ampz_png or ampz_svg)
    write_amplitude_map_zoom_png = bool(ampz_png)
    write_amplitude_map_zoom_svg = bool(ampz_svg)

    lat_png, lat_svg, peak_latency_map_relpath, peak_latency_map_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "peak_latency", "full"),
        default_relpath=peak_latency_map_relpath,
        default_write_png=bool(write_peak_latency_map),
    )
    write_peak_latency_map = bool(lat_png or lat_svg)
    write_peak_latency_map_png = bool(lat_png)
    write_peak_latency_map_svg = bool(lat_svg)

    latz_png, latz_svg, peak_latency_map_zoom_relpath, peak_latency_map_zoom_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "peak_latency", "zoom"),
        default_relpath=peak_latency_map_zoom_relpath,
        default_write_png=bool(write_peak_latency_map_zoom),
    )
    write_peak_latency_map_zoom = bool(latz_png or latz_svg)
    write_peak_latency_map_zoom_png = bool(latz_png)
    write_peak_latency_map_zoom_svg = bool(latz_svg)

    std_png, std_svg, peak_std_map_relpath, peak_std_map_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "peak_std", "full"),
        default_relpath=peak_std_map_relpath,
        default_write_png=bool(write_peak_std_map),
    )
    write_peak_std_map = bool(std_png or std_svg)
    write_peak_std_map_png = bool(std_png)
    write_peak_std_map_svg = bool(std_svg)

    stdz_png, stdz_svg, peak_std_map_zoom_relpath, peak_std_map_zoom_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "peak_std", "zoom"),
        default_relpath=peak_std_map_zoom_relpath,
        default_write_png=bool(write_peak_std_map_zoom),
    )
    write_peak_std_map_zoom = bool(stdz_png or stdz_svg)
    write_peak_std_map_zoom_png = bool(stdz_png)
    write_peak_std_map_zoom_svg = bool(stdz_svg)

    cdet_png, cdet_svg, channel_selection_detect_relpath, channel_selection_detect_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "channel_selection", "detect"),
        default_relpath=channel_selection_detect_relpath,
        default_write_png=bool(write_channel_selection_detect),
    )
    write_channel_selection_detect = bool(cdet_png or cdet_svg)
    write_channel_selection_detect_png = bool(cdet_png)
    write_channel_selection_detect_svg = bool(cdet_svg)

    ckurt_png, ckurt_svg, channel_selection_kurt_relpath, channel_selection_kurt_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "channel_selection", "kurt"),
        default_relpath=channel_selection_kurt_relpath,
        default_write_png=bool(write_channel_selection_kurt),
    )
    write_channel_selection_kurt = bool(ckurt_png or ckurt_svg)
    write_channel_selection_kurt_png = bool(ckurt_png)
    write_channel_selection_kurt_svg = bool(ckurt_svg)

    cdel_png, cdel_svg, channel_selection_delay_relpath, channel_selection_delay_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "channel_selection", "delay"),
        default_relpath=channel_selection_delay_relpath,
        default_write_png=bool(write_channel_selection_delay),
    )
    write_channel_selection_delay = bool(cdel_png or cdel_svg)
    write_channel_selection_delay_png = bool(cdel_png)
    write_channel_selection_delay_svg = bool(cdel_svg)

    call_png, call_svg, channel_selection_all_relpath, channel_selection_all_svg_relpath = _resolve_png_spec(
        cfg_path=("maps", "channel_selection", "all"),
        default_relpath=channel_selection_all_relpath,
        default_write_png=bool(write_channel_selection_all),
    )
    write_channel_selection_all = bool(call_png or call_svg)
    write_channel_selection_all_png = bool(call_png)
    write_channel_selection_all_svg = bool(call_svg)

    gn_png, gn_svg, graph_nodes_relpath, graph_nodes_svg_relpath = _resolve_png_spec(
        cfg_path=("graph", "nodes"),
        default_relpath=graph_nodes_relpath,
        default_write_png=bool(write_graph_nodes),
    )
    write_graph_nodes = bool(gn_png or gn_svg)
    write_graph_nodes_png = bool(gn_png)
    write_graph_nodes_svg = bool(gn_svg)

    ge_png, ge_svg, graph_edges_relpath, graph_edges_svg_relpath = _resolve_png_spec(
        cfg_path=("graph", "edges"),
        default_relpath=graph_edges_relpath,
        default_write_png=bool(write_graph_edges),
    )
    write_graph_edges = bool(ge_png or ge_svg)
    write_graph_edges_png = bool(ge_png)
    write_graph_edges_svg = bool(ge_svg)

    gh_png, gh_svg, graph_heuristics_relpath, graph_heuristics_svg_relpath = _resolve_png_spec(
        cfg_path=("graph", "heuristics"),
        default_relpath=graph_heuristics_relpath,
        default_write_png=bool(write_graph_heuristics),
    )
    write_graph_heuristics = bool(gh_png or gh_svg)
    write_graph_heuristics_png = bool(gh_png)
    write_graph_heuristics_svg = bool(gh_svg)

    (
        write_morphology_pdf,
        write_morphology_png,
        write_morphology_svg,
        morphology_pdf_relpath,
        morphology_png_relpath,
        morphology_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("morphology", "full"),
        default_relpath=morphology_pdf_relpath,
        default_write_pdf=bool(write_morphology_pdf),
    )

    (
        write_morphology_zoom_pdf,
        write_morphology_zoom_png,
        write_morphology_zoom_svg,
        morphology_zoom_pdf_relpath,
        morphology_zoom_png_relpath,
        morphology_zoom_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("morphology", "zoom"),
        default_relpath=morphology_zoom_pdf_relpath,
        default_write_pdf=bool(write_morphology_zoom_pdf),
    )

    (
        write_branches_raw_clean_pdf,
        write_branches_raw_clean_png,
        write_branches_raw_clean_svg,
        branches_raw_clean_pdf_relpath,
        branches_raw_clean_png_relpath,
        branches_raw_clean_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("branches", "raw_clean"),
        default_relpath=branches_raw_clean_pdf_relpath,
        default_write_pdf=bool(write_branches_raw_clean_pdf),
    )

    (
        write_branches_raw_pdf,
        write_branches_raw_png,
        write_branches_raw_svg,
        branches_raw_pdf_relpath,
        branches_raw_png_relpath,
        branches_raw_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("branches", "raw"),
        default_relpath=branches_raw_pdf_relpath,
        default_write_pdf=bool(write_branches_raw_pdf),
    )

    (
        write_branches_raw_zoom_pdf,
        write_branches_raw_zoom_png,
        write_branches_raw_zoom_svg,
        branches_raw_zoom_pdf_relpath,
        branches_raw_zoom_png_relpath,
        branches_raw_zoom_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("branches", "raw_zoom"),
        default_relpath=branches_raw_zoom_pdf_relpath,
        default_write_pdf=bool(write_branches_raw_zoom_pdf),
    )

    (
        write_branches_clean_pdf,
        write_branches_clean_png,
        write_branches_clean_svg,
        branches_clean_pdf_relpath,
        branches_clean_png_relpath,
        branches_clean_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("branches", "clean"),
        default_relpath=branches_clean_pdf_relpath,
        default_write_pdf=bool(write_branches_clean_pdf),
    )

    (
        write_branches_clean_zoom_pdf,
        write_branches_clean_zoom_png,
        write_branches_clean_zoom_svg,
        branches_clean_zoom_pdf_relpath,
        branches_clean_zoom_png_relpath,
        branches_clean_zoom_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("branches", "clean_zoom"),
        default_relpath=branches_clean_zoom_pdf_relpath,
        default_write_pdf=bool(write_branches_clean_zoom_pdf),
    )

    (
        write_branch_velocities_pdf,
        write_branch_velocities_png,
        write_branch_velocities_svg,
        branch_velocities_pdf_relpath,
        branch_velocities_png_relpath,
        branch_velocities_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("branches", "velocities"),
        default_relpath=branch_velocities_pdf_relpath,
        default_write_pdf=bool(write_branch_velocities_pdf),
    )

    (
        write_branch_velocities_raw_overlay_pdf,
        write_branch_velocities_raw_overlay_png,
        write_branch_velocities_raw_overlay_svg,
        branch_velocities_raw_overlay_pdf_relpath,
        branch_velocities_raw_overlay_png_relpath,
        branch_velocities_raw_overlay_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("branches", "velocities_raw_overlay"),
        default_relpath=branch_velocities_raw_overlay_pdf_relpath,
        default_write_pdf=bool(write_branch_velocities_raw_overlay_pdf),
    )

    (
        write_branch_velocities_overlay_pdf,
        write_branch_velocities_overlay_png,
        write_branch_velocities_overlay_svg,
        branch_velocities_overlay_pdf_relpath,
        branch_velocities_overlay_png_relpath,
        branch_velocities_overlay_svg_relpath,
    ) = _resolve_pdf_spec(
        cfg_path=("branches", "velocities_overlay"),
        default_relpath=branch_velocities_overlay_pdf_relpath,
        default_write_pdf=bool(write_branch_velocities_overlay_pdf),
    )

    branch_template_cfg = _schema_get(("branches", "velocity_template"), default={})
    write_branch_velocity_template_pdf = bool(write_branch_velocity_template)
    write_branch_velocity_template_png = True
    write_branch_velocity_template_svg = False
    if isinstance(branch_template_cfg, dict):
        branch_velocity_template_relpath = str(
            branch_template_cfg.get("relpath", branch_velocity_template_relpath)
        )
        write_branch_velocity_template_pdf = bool(
            branch_template_cfg.get("write_pdf", write_branch_velocity_template_pdf)
        )
        write_branch_velocity_template_png = bool(branch_template_cfg.get("write_png", True))
        write_branch_velocity_template_svg = bool(branch_template_cfg.get("write_svg", False))
    write_branch_velocity_template = bool(
        write_branch_velocity_template_pdf or write_branch_velocity_template_png or write_branch_velocity_template_svg
    )

    template_png = _unit_path(template_relpath)
    template_svg = _unit_path(template_svg_relpath)
    template_zoom_png = _unit_path(template_zoom_relpath)
    template_zoom_svg = _unit_path(template_zoom_svg_relpath)
    summary_png = _unit_path(summary_relpath)
    summary_svg = _unit_path(summary_svg_relpath)
    summary_clean_png = _unit_path(summary_clean_relpath)
    summary_clean_svg = _unit_path(summary_clean_svg_relpath)
    summary_raw_png = _unit_path(summary_raw_relpath)
    summary_raw_svg = _unit_path(summary_raw_svg_relpath)
    template_movie_gif = _unit_path(template_movie_relpath)

    if write_template_movie_gif is None:
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
        # Preferred definition: channels from stage-4 merged contributing template.
        # Resolve from the current well output dir and this unit id.
        well_out_dir = Path(out_unit_dir).resolve().parents[2]
        merged_glob = list(
            well_out_dir.glob(
                f"stg4_templates_outputs*/templates/merged/unit_{int(uid)}/merged_contributing_channel_locations.npy"
            )
        )
        merged_loc_path = merged_glob[0] if merged_glob else None

        if merged_loc_path is not None and merged_loc_path.exists():
            merged_locs = np.asarray(np.load(merged_loc_path))
            if merged_locs.ndim == 2 and merged_locs.shape[1] >= 2:
                loc_map: dict[tuple[float, float], int] = {}
                for i in range(int(locs_xy.shape[0])):
                    key = (round(float(locs_xy[i, 0]), 6), round(float(locs_xy[i, 1]), 6))
                    if key not in loc_map:
                        loc_map[key] = int(i)

                contrib_idx: list[int] = []
                for row in merged_locs:
                    x = float(row[0])
                    y = float(row[1])
                    key = (round(x, 6), round(y, 6))
                    idx = loc_map.get(key)
                    if idx is None:
                        # Fallback: nearest-neighbor match in full-channel coordinates.
                        d2 = (locs_xy[:, 0] - x) ** 2 + (locs_xy[:, 1] - y) ** 2
                        idx = int(np.argmin(d2))
                    if 0 <= int(idx) < int(locs_xy.shape[0]):
                        contrib_idx.append(int(idx))
                contributing_channels = sorted(set(contrib_idx))

        # Backward fallback: branch channels when merged channel map is unavailable.
        if not contributing_channels:
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

    soma_channel: int | None = None
    soma_xy: tuple[float, float] | None = None
    try:
        init_ch = int(getattr(gtr, "init_channel", -1))
        if 0 <= init_ch < int(locs_xy.shape[0]):
            soma_channel = int(init_ch)
            soma_xy = (float(locs_xy[init_ch, 0]), float(locs_xy[init_ch, 1]))
    except Exception:
        soma_channel = None
        soma_xy = None

    # Prefer stage-4 templates as plotting source so channel-scope options map
    # directly to full/merged template definitions on disk.
    template_plot = template
    template_locs = np.asarray(locs_xy)
    try:
        well_out_dir = Path(out_unit_dir).resolve().parents[2]
        stage4_roots = sorted(list(well_out_dir.glob("stg4_templates_outputs*")))
        stage4_root = stage4_roots[0] if stage4_roots else None
        if stage4_root is not None and stage4_root.exists():
            full_template_path = stage4_root / "templates" / "full" / f"unit_{int(uid)}" / "full_template.npy"
            full_locs_path = stage4_root / "templates" / "full" / f"unit_{int(uid)}" / "full_channel_locations_xy.npy"
            merged_template_path = (
                stage4_root / "templates" / "merged" / f"unit_{int(uid)}" / "merged_contributing_template.npy"
            )
            merged_locs_path = (
                stage4_root / "templates" / "merged" / f"unit_{int(uid)}" / "merged_contributing_channel_locations.npy"
            )

            if (
                template_channel_scope == "contributing_channels"
                and merged_template_path.exists()
                and merged_locs_path.exists()
            ):
                merged_template = np.asarray(np.load(merged_template_path))
                merged_locs = np.asarray(np.load(merged_locs_path))[:, :2]
                if merged_template.ndim == 2:
                    if merged_template.shape[0] == int(merged_locs.shape[0]):
                        template_plot = merged_template
                    elif merged_template.shape[1] == int(merged_locs.shape[0]):
                        template_plot = merged_template.T
                    else:
                        template_plot = merged_template
                else:
                    template_plot = merged_template
                template_locs = merged_locs
            elif full_template_path.exists() and full_locs_path.exists():
                full_template = np.asarray(np.load(full_template_path))
                full_locs = np.asarray(np.load(full_locs_path))[:, :2]

                full_template_cf = full_template
                if full_template.ndim == 2:
                    if full_template.shape[0] == int(full_locs.shape[0]):
                        full_template_cf = full_template
                    elif full_template.shape[1] == int(full_locs.shape[0]):
                        full_template_cf = full_template.T

                if template_channel_scope == "recorded_channels":
                    keep_idx: list[int] = []
                    try:
                        if full_template_cf.ndim == 2:
                            if full_template_cf.shape[0] == int(full_locs.shape[0]):
                                per_ch_max = np.nanmax(np.abs(full_template_cf), axis=1)
                            else:
                                per_ch_max = np.array([], dtype=float)
                            eps = float(np.finfo(float).eps)
                            keep_idx = [int(i) for i, v in enumerate(per_ch_max.tolist()) if float(v) > eps]
                    except Exception:
                        keep_idx = []

                    if keep_idx:
                        template_plot = full_template_cf[keep_idx, :]
                        template_locs = full_locs[keep_idx, :]
                    else:
                        template_plot = full_template_cf
                        template_locs = full_locs
                else:
                    # all_channels (or fallback)
                    template_plot = full_template_cf
                    template_locs = full_locs
    except Exception:
        template_plot = template
        template_locs = np.asarray(locs_xy)

    contributing_scope_points: list[list[float]] = [
        [float(x), float(y)] for x, y in np.asarray(template_locs)[:, :2].tolist()
    ] if template_locs is not None else []

    if write_template and ((not template_png.exists()) or force_restart) and (template_plot is not None):
        try:
            from axon_velocity.plotting import plot_template as av_plot_template  # type: ignore[import-not-found]

            fig = plt.figure(figsize=(13, 10))
            ax = fig.add_subplot(111)
            with plt.rc_context(_white_bg_rc_params()):
                _ = av_plot_template(template=template_plot, locations=template_locs, ax=ax)
            _thin_lines_and_markers(ax, lw=0.45, ms=1.5, alpha=0.9)

            if contributing_scope_points:
                xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(
                    contributing_scope_points,
                    pad_frac=0.03,
                    pad_abs=10.0,
                )
                if template_force_square_aspect:
                    cx = soma_xy[0] if (template_force_center_soma and soma_xy is not None) else None
                    cy = soma_xy[1] if (template_force_center_soma and soma_xy is not None) else None
                    xmin, xmax, ymin, ymax = _make_square_limits(
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        center_x=cx,
                        center_y=cy,
                    )
                elif template_force_center_soma and soma_xy is not None:
                    w = float(xmax - xmin)
                    h = float(ymax - ymin)
                    xmin = float(soma_xy[0] - (w / 2.0))
                    xmax = float(soma_xy[0] + (w / 2.0))
                    ymin = float(soma_xy[1] - (h / 2.0))
                    ymax = float(soma_xy[1] + (h / 2.0))
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal", adjustable="box")

            _apply_template_style(
                fig=fig,
                ax=ax,
                background=template_background,
                signal_color=template_signal_color,
            )
            _save_fig_png(
                fig=fig,
                png_path=template_png,
                dpi=DPI_HI,
                write_png=bool(write_template_png),
                write_svg=bool(write_template_svg),
                svg_path=template_svg,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning("Template plotting failed for unit %s: %s", uid, e)

    if write_template_zoom and ((not template_zoom_png.exists()) or force_restart) and (template_plot is not None):
        try:
            from axon_velocity.plotting import plot_template as av_plot_template  # type: ignore[import-not-found]

            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_subplot(111)
            with plt.rc_context(_white_bg_rc_params()):
                _ = av_plot_template(template=template_plot, locations=template_locs, ax=ax)
            _thin_lines_and_markers(ax, lw=0.45, ms=1.5, alpha=0.9)
            zoom_points = contributing_scope_points if contributing_scope_points else branch_xy_points
            if not zoom_points:
                raise ValueError("No points available for template zoom")
            xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(zoom_points)
            if template_force_square_aspect:
                cx = soma_xy[0] if (template_force_center_soma and soma_xy is not None) else None
                cy = soma_xy[1] if (template_force_center_soma and soma_xy is not None) else None
                xmin, xmax, ymin, ymax = _make_square_limits(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    center_x=cx,
                    center_y=cy,
                )
            elif template_force_center_soma and soma_xy is not None:
                w = float(xmax - xmin)
                h = float(ymax - ymin)
                xmin = float(soma_xy[0] - (w / 2.0))
                xmax = float(soma_xy[0] + (w / 2.0))
                ymin = float(soma_xy[1] - (h / 2.0))
                ymax = float(soma_xy[1] + (h / 2.0))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            _apply_template_style(
                fig=fig,
                ax=ax,
                background=template_background,
                signal_color=template_signal_color,
            )
            _save_fig_png(
                fig=fig,
                png_path=template_zoom_png,
                dpi=DPI_HI,
                write_png=bool(write_template_zoom_png),
                write_svg=bool(write_template_zoom_svg),
                svg_path=template_zoom_svg,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning("Template zoom plotting failed for unit %s: %s", uid, e)

    # Summary + template animation are generated via axon_velocity.

    if write_template_png and template_png.exists():
        outputs["template_png"] = str(template_png)
    if write_template_svg and template_svg.exists():
        outputs["template_svg"] = str(template_svg)
    if write_template_zoom_png and template_zoom_png.exists():
        outputs["template_zoom_png"] = str(template_zoom_png)
    if write_template_zoom_svg and template_zoom_svg.exists():
        outputs["template_zoom_svg"] = str(template_zoom_svg)
    # summary_png + template_movie_gif are written later.

    if (
        write_amplitude_map
        or write_amplitude_map_zoom
        or write_peak_latency_map
        or write_peak_latency_map_zoom
        or write_peak_std_map
        or write_peak_std_map_zoom
    ) and (template is not None) and (fs is not None):
        try:
            from axon_velocity.plotting import (  # type: ignore[import-not-found]
                plot_amplitude_map as av_plot_amplitude_map,
                plot_peak_latency_map as av_plot_peak_latency_map,
                plot_peak_std_map as av_plot_peak_std_map,
            )

            def _write_map(
                fn: Any,
                out_png: Path,
                out_zoom_png: Path,
                out_svg: Path,
                out_zoom_svg: Path,
                *,
                write_png: bool,
                write_svg: bool,
                write_zoom_png: bool,
                write_zoom_svg: bool,
            ) -> None:
                if not (write_png or write_svg or write_zoom_png or write_zoom_svg):
                    return
                if (not force_restart):
                    png_ready = (not write_png) or out_png.exists()
                    svg_ready = (not write_svg) or out_svg.exists()
                    zoom_ready = (not write_zoom_png) or ((not branch_xy_points) or out_zoom_png.exists())
                    zoom_svg_ready = (not write_zoom_svg) or ((not branch_xy_points) or out_zoom_svg.exists())
                    if png_ready and svg_ready and zoom_ready and zoom_svg_ready:
                        return
                fig = plt.figure(figsize=(8.5, 7.5))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = fn(ax=ax)
                _force_white_background(fig)
                if write_png or write_svg:
                    _save_fig_png(
                        fig=fig,
                        png_path=out_png,
                        dpi=DPI_HI,
                        write_png=bool(write_png),
                        write_svg=bool(write_svg),
                        svg_path=out_svg,
                    )
                if (write_zoom_png or write_zoom_svg) and branch_xy_points:
                    xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(branch_xy_points)
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    ax.set_aspect("equal", adjustable="box")
                    _save_fig_png(
                        fig=fig,
                        png_path=out_zoom_png,
                        dpi=DPI_HI,
                        write_png=bool(write_zoom_png),
                        write_svg=bool(write_zoom_svg),
                        svg_path=out_zoom_svg,
                    )
                plt.close(fig)

            amp_png = _unit_output_path(amplitude_map_relpath)
            amp_zoom_png = _unit_output_path(amplitude_map_zoom_relpath)
            amp_svg = _unit_output_path(amplitude_map_svg_relpath)
            amp_zoom_svg = _unit_output_path(amplitude_map_zoom_svg_relpath)
            _write_map(
                lambda ax: av_plot_amplitude_map(template, locs_xy, log=True, ax=ax),
                amp_png,
                amp_zoom_png,
                amp_svg,
                amp_zoom_svg,
                write_png=bool(write_amplitude_map),
                write_svg=bool(write_amplitude_map_svg),
                write_zoom_png=bool(write_amplitude_map_zoom),
                write_zoom_svg=bool(write_amplitude_map_zoom_svg),
            )

            lat_png = _unit_output_path(peak_latency_map_relpath)
            lat_zoom_png = _unit_output_path(peak_latency_map_zoom_relpath)
            lat_svg = _unit_output_path(peak_latency_map_svg_relpath)
            lat_zoom_svg = _unit_output_path(peak_latency_map_zoom_svg_relpath)
            _write_map(
                lambda ax: av_plot_peak_latency_map(template, locs_xy, float(fs), ax=ax),
                lat_png,
                lat_zoom_png,
                lat_svg,
                lat_zoom_svg,
                write_png=bool(write_peak_latency_map),
                write_svg=bool(write_peak_latency_map_svg),
                write_zoom_png=bool(write_peak_latency_map_zoom),
                write_zoom_svg=bool(write_peak_latency_map_zoom_svg),
            )

            std_png = _unit_output_path(peak_std_map_relpath)
            std_zoom_png = _unit_output_path(peak_std_map_zoom_relpath)
            std_svg = _unit_output_path(peak_std_map_svg_relpath)
            std_zoom_svg = _unit_output_path(peak_std_map_zoom_svg_relpath)
            _write_map(
                lambda ax: av_plot_peak_std_map(template, locs_xy, float(fs), ax=ax),
                std_png,
                std_zoom_png,
                std_svg,
                std_zoom_svg,
                write_png=bool(write_peak_std_map),
                write_svg=bool(write_peak_std_map_svg),
                write_zoom_png=bool(write_peak_std_map_zoom),
                write_zoom_svg=bool(write_peak_std_map_zoom_svg),
            )

            for p, k, enabled in [
                (amp_png, "amplitude_map_png", bool(write_amplitude_map)),
                (amp_zoom_png, "amplitude_map_zoom_png", bool(write_amplitude_map_zoom)),
                (lat_png, "peak_latency_map_png", bool(write_peak_latency_map)),
                (lat_zoom_png, "peak_latency_map_zoom_png", bool(write_peak_latency_map_zoom)),
                (std_png, "peak_std_map_png", bool(write_peak_std_map)),
                (std_zoom_png, "peak_std_map_zoom_png", bool(write_peak_std_map_zoom)),
            ]:
                if not enabled:
                    continue
                if p.exists():
                    outputs[k] = str(p)
            for p, k, enabled in [
                (amp_svg, "amplitude_map_svg", bool(write_amplitude_map_svg)),
                (amp_zoom_svg, "amplitude_map_zoom_svg", bool(write_amplitude_map_zoom_svg)),
                (lat_svg, "peak_latency_map_svg", bool(write_peak_latency_map_svg)),
                (lat_zoom_svg, "peak_latency_map_zoom_svg", bool(write_peak_latency_map_zoom_svg)),
                (std_svg, "peak_std_map_svg", bool(write_peak_std_map_svg)),
                (std_zoom_svg, "peak_std_map_zoom_svg", bool(write_peak_std_map_zoom_svg)),
            ]:
                if not enabled:
                    continue
                if p.exists():
                    outputs[k] = str(p)
        except Exception as e:
            logger.warning("Map plotting failed for unit %s: %s", uid, e)

    # Channel selection maps (Detection/Kurtosis/Delay/All) into maps/.
    if (
        write_channel_selection_detect
        or write_channel_selection_kurt
        or write_channel_selection_delay
        or write_channel_selection_all
    ):
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

            ch_sel_paths = {
                "detect": (
                    _unit_output_path(channel_selection_detect_relpath),
                    _unit_output_path(channel_selection_detect_svg_relpath),
                    bool(write_channel_selection_detect_png),
                    bool(write_channel_selection_detect_svg),
                ),
                "kurt": (
                    _unit_output_path(channel_selection_kurt_relpath),
                    _unit_output_path(channel_selection_kurt_svg_relpath),
                    bool(write_channel_selection_kurt_png),
                    bool(write_channel_selection_kurt_svg),
                ),
                "delay": (
                    _unit_output_path(channel_selection_delay_relpath),
                    _unit_output_path(channel_selection_delay_svg_relpath),
                    bool(write_channel_selection_delay_png),
                    bool(write_channel_selection_delay_svg),
                ),
                "all": (
                    _unit_output_path(channel_selection_all_relpath),
                    _unit_output_path(channel_selection_all_svg_relpath),
                    bool(write_channel_selection_all_png),
                    bool(write_channel_selection_all_svg),
                ),
            }

            for name, raw in chan_sets.items():
                out_info = ch_sel_paths.get(name)
                if out_info is not None and (not out_info[2]) and (not out_info[3]):
                    continue
                sel = _as_ch_list(raw)
                if not sel:
                    continue
                out_png = (out_info[0] if out_info is not None else _unit_output_path(f"maps/channel_selection_{name}.png"))
                out_svg = (
                    out_info[1]
                    if out_info is not None
                    else _unit_output_path(f"maps/channel_selection_{name}.svg")
                )
                write_png_local = bool(out_info[2]) if out_info is not None else True
                write_svg_local = bool(out_info[3]) if out_info is not None else False
                if out_png.exists() and (not force_restart) and (not write_svg_local or out_svg.exists()):
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
                _save_fig_png(
                    fig=fig,
                    png_path=out_png,
                    dpi=DPI_HI,
                    write_png=write_png_local,
                    write_svg=write_svg_local,
                    svg_path=out_svg,
                )
                plt.close(fig)
                if write_png_local and out_png.exists():
                    outputs[f"channel_selection_{name}_png"] = str(out_png)
                if write_svg_local and out_svg.exists():
                    outputs[f"channel_selection_{name}_svg"] = str(out_svg)
        except Exception as e:
            logger.warning("Channel selection map plotting failed for unit %s: %s", uid, e)

    # Graph: nodes + edges as separate PNGs into maps/, plus a combined overview into heuristics/ for analysis.
    if write_graph_nodes or write_graph_edges or write_graph_heuristics:
        try:
            import matplotlib as mpl

            graph_nodes_png = _unit_output_path(graph_nodes_relpath)
            graph_nodes_svg = _unit_output_path(graph_nodes_svg_relpath)
            graph_edges_png = _unit_output_path(graph_edges_relpath)
            graph_edges_svg = _unit_output_path(graph_edges_svg_relpath)
            graph_combined_png = _unit_output_path(graph_heuristics_relpath)
            graph_combined_svg = _unit_output_path(graph_heuristics_svg_relpath)

            if write_graph_nodes and ((not graph_nodes_png.exists()) or force_restart) and hasattr(gtr, "_plot_nodes"):
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
                _save_fig_png(
                    fig=fig,
                    png_path=graph_nodes_png,
                    dpi=DPI_HI,
                    write_png=bool(write_graph_nodes_png),
                    write_svg=bool(write_graph_nodes_svg),
                    svg_path=graph_nodes_svg,
                )
                plt.close(fig)

            if write_graph_edges and ((not graph_edges_png.exists()) or force_restart) and hasattr(gtr, "_plot_edges"):
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
                _save_fig_png(
                    fig=fig,
                    png_path=graph_edges_png,
                    dpi=DPI_HI,
                    write_png=bool(write_graph_edges_png),
                    write_svg=bool(write_graph_edges_svg),
                    svg_path=graph_edges_svg,
                )
                plt.close(fig)

            if write_graph_heuristics and ((not graph_combined_png.exists()) or force_restart):
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
                    _save_fig_png(
                        fig=fig,
                        png_path=graph_combined_png,
                        dpi=DPI_HI,
                        write_png=bool(write_graph_heuristics_png),
                        write_svg=bool(write_graph_heuristics_svg),
                        svg_path=graph_combined_svg,
                    )
                    plt.close(fig)
                except Exception:
                    pass

            if write_graph_nodes_png and graph_nodes_png.exists():
                outputs["graph_nodes_png"] = str(graph_nodes_png)
            if write_graph_nodes_svg and graph_nodes_svg.exists():
                outputs["graph_nodes_svg"] = str(graph_nodes_svg)
            if write_graph_edges_png and graph_edges_png.exists():
                outputs["graph_edges_png"] = str(graph_edges_png)
            if write_graph_edges_svg and graph_edges_svg.exists():
                outputs["graph_edges_svg"] = str(graph_edges_svg)
            if write_graph_heuristics_png and graph_combined_png.exists():
                outputs["graph_heuristics_png"] = str(graph_combined_png)
            if write_graph_heuristics_svg and graph_combined_svg.exists():
                outputs["graph_heuristics_svg"] = str(graph_combined_svg)
        except Exception as e:
            logger.warning("Graph plotting failed for unit %s: %s", uid, e)

    # Always write a simple morphology PDF.
    morphology_pdf = _unit_output_path(morphology_pdf_relpath)
    morphology_png = _with_suffix(morphology_pdf, ".png")
    morphology_svg = _with_suffix(morphology_pdf, ".svg")
    if write_morphology_pdf and ((not morphology_pdf.exists()) or force_restart):
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
            _save_fig_pdf_and_png(
                fig=fig,
                pdf_path=morphology_pdf,
                png_path=morphology_png,
                dpi=DPI_STD,
                write_png=bool(write_morphology_png),
                write_svg=bool(write_morphology_svg),
                svg_path=morphology_svg,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning("Morphology plotting failed for unit %s: %s", uid, e)

    if write_morphology_pdf and morphology_pdf.exists():
        outputs["morphology_pdf"] = str(morphology_pdf)
    if write_morphology_png and morphology_png.exists():
        outputs["morphology_png"] = str(morphology_png)
    if write_morphology_svg and morphology_svg.exists():
        outputs["morphology_svg"] = str(morphology_svg)

    # Zoomed-in morphology around the reconstruction.
    morphology_zoom_pdf = _unit_output_path(morphology_zoom_pdf_relpath)
    morphology_zoom_png = _with_suffix(morphology_zoom_pdf, ".png")
    morphology_zoom_svg = _with_suffix(morphology_zoom_pdf, ".svg")
    if write_morphology_zoom_pdf and ((not morphology_zoom_pdf.exists()) or force_restart):
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
            _save_fig_pdf_and_png(
                fig=fig,
                pdf_path=morphology_zoom_pdf,
                png_path=morphology_zoom_png,
                dpi=DPI_STD,
                write_png=bool(write_morphology_zoom_png),
                write_svg=bool(write_morphology_zoom_svg),
                svg_path=morphology_zoom_svg,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning("Zoom morphology plotting failed for unit %s: %s", uid, e)

    if write_morphology_zoom_pdf and morphology_zoom_pdf.exists():
        outputs["morphology_zoom_pdf"] = str(morphology_zoom_pdf)
    if write_morphology_zoom_png and morphology_zoom_png.exists():
        outputs["morphology_zoom_png"] = str(morphology_zoom_png)
    if write_morphology_zoom_svg and morphology_zoom_svg.exists():
        outputs["morphology_zoom_svg"] = str(morphology_zoom_svg)

    # Raw + clean branches (axon_velocity built-in). This explicitly shows pre/post clean_paths.
    branches_pdf = _unit_output_path(branches_raw_clean_pdf_relpath)
    branches_png = _with_suffix(branches_pdf, ".png")
    branches_svg = _with_suffix(branches_pdf, ".svg")
    if write_branches_raw_clean_pdf and ((not branches_pdf.exists()) or force_restart):
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
            _save_fig_pdf_and_png(
                fig=fig,
                pdf_path=branches_pdf,
                png_path=branches_png,
                dpi=DPI_STD,
                write_png=bool(write_branches_raw_clean_png),
                write_svg=bool(write_branches_raw_clean_svg),
                svg_path=branches_svg,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning("Branches (raw+clean) plotting failed for unit %s: %s", uid, e)

    if write_branches_raw_clean_pdf and branches_pdf.exists():
        outputs["branches_raw_clean_pdf"] = str(branches_pdf)
    if write_branches_raw_clean_png and branches_png.exists():
        outputs["branches_raw_clean_png"] = str(branches_png)
    if write_branches_raw_clean_svg and branches_svg.exists():
        outputs["branches_raw_clean_svg"] = str(branches_svg)

    # Raw branches only (axon_velocity built-in). This is useful when you want to see
    # everything before clean_paths duplicate-removal.
    raw_branches_pdf = _unit_output_path(branches_raw_pdf_relpath)
    raw_branches_png = _with_suffix(raw_branches_pdf, ".png")
    raw_branches_svg = _with_suffix(raw_branches_pdf, ".svg")
    if write_branches_raw_pdf and ((not raw_branches_pdf.exists()) or force_restart):
        try:
            plot_fn = getattr(gtr, "plot_raw_branches", None)
            if callable(plot_fn):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = plot_fn(plot_full_template=True, ax=ax)
                _minimal_axes(ax)
                _force_white_background(fig)
                _save_fig_pdf_and_png(
                    fig=fig,
                    pdf_path=raw_branches_pdf,
                    png_path=raw_branches_png,
                    dpi=DPI_STD,
                    write_png=bool(write_branches_raw_png),
                    write_svg=bool(write_branches_raw_svg),
                    svg_path=raw_branches_svg,
                )
                plt.close(fig)
        except Exception as e:
            logger.warning("Raw branches plotting failed for unit %s: %s", uid, e)

    if write_branches_raw_pdf and raw_branches_pdf.exists():
        outputs["branches_raw_pdf"] = str(raw_branches_pdf)
    if write_branches_raw_png and raw_branches_png.exists():
        outputs["branches_raw_png"] = str(raw_branches_png)
    if write_branches_raw_svg and raw_branches_svg.exists():
        outputs["branches_raw_svg"] = str(raw_branches_svg)

    # Zoomed raw branches (use raw path channels to compute limits).
    raw_branches_zoom_pdf = _unit_output_path(branches_raw_zoom_pdf_relpath)
    raw_branches_zoom_png = _with_suffix(raw_branches_zoom_pdf, ".png")
    raw_branches_zoom_svg = _with_suffix(raw_branches_zoom_pdf, ".svg")
    if write_branches_raw_zoom_pdf and ((not raw_branches_zoom_pdf.exists()) or force_restart):
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
                _save_fig_pdf_and_png(
                    fig=fig,
                    pdf_path=raw_branches_zoom_pdf,
                    png_path=raw_branches_zoom_png,
                    dpi=DPI_STD,
                    write_png=bool(write_branches_raw_zoom_png),
                    write_svg=bool(write_branches_raw_zoom_svg),
                    svg_path=raw_branches_zoom_svg,
                )
                plt.close(fig)
        except Exception as e:
            logger.warning("Raw branches zoom plotting failed for unit %s: %s", uid, e)

    if write_branches_raw_zoom_pdf and raw_branches_zoom_pdf.exists():
        outputs["branches_raw_zoom_pdf"] = str(raw_branches_zoom_pdf)
    if write_branches_raw_zoom_png and raw_branches_zoom_png.exists():
        outputs["branches_raw_zoom_png"] = str(raw_branches_zoom_png)
    if write_branches_raw_zoom_svg and raw_branches_zoom_svg.exists():
        outputs["branches_raw_zoom_svg"] = str(raw_branches_zoom_svg)

    # Clean branches (axon_velocity built-in). This shows post-clean_paths results.
    clean_branches_pdf = _unit_output_path(branches_clean_pdf_relpath)
    clean_branches_png = _with_suffix(clean_branches_pdf, ".png")
    clean_branches_svg = _with_suffix(clean_branches_pdf, ".svg")
    if write_branches_clean_pdf and ((not clean_branches_pdf.exists()) or force_restart):
        try:
            plot_fn = getattr(gtr, "plot_clean_branches", None)
            if callable(plot_fn):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                with plt.rc_context(_white_bg_rc_params()):
                    _ = plot_fn(plot_full_template=True, ax=ax)
                _force_white_background(fig)
                _save_fig_pdf_and_png(
                    fig=fig,
                    pdf_path=clean_branches_pdf,
                    png_path=clean_branches_png,
                    dpi=DPI_STD,
                    write_png=bool(write_branches_clean_png),
                    write_svg=bool(write_branches_clean_svg),
                    svg_path=clean_branches_svg,
                )
                plt.close(fig)
        except Exception as e:
            logger.warning("Clean branches plotting failed for unit %s: %s", uid, e)

    if write_branches_clean_pdf and clean_branches_pdf.exists():
        outputs["branches_clean_pdf"] = str(clean_branches_pdf)
    if write_branches_clean_png and clean_branches_png.exists():
        outputs["branches_clean_png"] = str(clean_branches_png)
    if write_branches_clean_svg and clean_branches_svg.exists():
        outputs["branches_clean_svg"] = str(clean_branches_svg)

    # Zoomed clean branches (use clean branch channel lists to compute limits).
    clean_branches_zoom_pdf = _unit_output_path(branches_clean_zoom_pdf_relpath)
    clean_branches_zoom_png = _with_suffix(clean_branches_zoom_pdf, ".png")
    clean_branches_zoom_svg = _with_suffix(clean_branches_zoom_pdf, ".svg")
    if write_branches_clean_zoom_pdf and ((not clean_branches_zoom_pdf.exists()) or force_restart):
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
                _save_fig_pdf_and_png(
                    fig=fig,
                    pdf_path=clean_branches_zoom_pdf,
                    png_path=clean_branches_zoom_png,
                    dpi=DPI_STD,
                    write_png=bool(write_branches_clean_zoom_png),
                    write_svg=bool(write_branches_clean_zoom_svg),
                    svg_path=clean_branches_zoom_svg,
                )
                plt.close(fig)
        except Exception as e:
            logger.warning("Clean branches zoom plotting failed for unit %s: %s", uid, e)

    if write_branches_clean_zoom_pdf and clean_branches_zoom_pdf.exists():
        outputs["branches_clean_zoom_pdf"] = str(clean_branches_zoom_pdf)
    if write_branches_clean_zoom_png and clean_branches_zoom_png.exists():
        outputs["branches_clean_zoom_png"] = str(clean_branches_zoom_png)
    if write_branches_clean_zoom_svg and clean_branches_zoom_svg.exists():
        outputs["branches_clean_zoom_svg"] = str(clean_branches_zoom_svg)

    # axon_velocity built-in branch velocities plot.
    # Clean branch velocities (for analysis panel).
    velocities_pdf = _unit_output_path(branch_velocities_pdf_relpath)
    velocities_png = _with_suffix(velocities_pdf, ".png")
    velocities_svg = _with_suffix(velocities_pdf, ".svg")
    if write_branch_velocities_pdf and ((not velocities_pdf.exists()) or force_restart):
        try:
            plot_fn = getattr(gtr, "plot_velocities", None)
            if callable(plot_fn):
                with plt.rc_context(_white_bg_rc_params()):
                    fig = plot_fn()
                _force_white_background(fig)
                _recolor_noncolormapped_artists(fig)
                _save_fig_pdf_and_png(
                    fig=fig,
                    pdf_path=velocities_pdf,
                    png_path=velocities_png,
                    dpi=DPI_STD,
                    write_png=bool(write_branch_velocities_png),
                    write_svg=bool(write_branch_velocities_svg),
                    svg_path=velocities_svg,
                )
                plt.close(fig)
        except Exception as e:
            logger.warning("Branch velocities plotting failed for unit %s: %s", uid, e)

    if write_branch_velocities_pdf and velocities_pdf.exists():
        outputs["branch_velocities_pdf"] = str(velocities_pdf)
    if write_branch_velocities_png and velocities_png.exists():
        outputs["branch_velocities_png"] = str(velocities_png)
    if write_branch_velocities_svg and velocities_svg.exists():
        outputs["branch_velocities_svg"] = str(velocities_svg)

    # Raw velocity plot (separate) under branches/raw.
    raw_vel_pdf = _unit_output_path(branch_velocities_raw_overlay_pdf_relpath)
    raw_vel_png = _with_suffix(raw_vel_pdf, ".png")
    raw_vel_svg = _with_suffix(raw_vel_pdf, ".svg")
    if write_branch_velocities_raw_overlay_pdf and ((not raw_vel_pdf.exists()) or force_restart):
        try:
            with plt.rc_context(_white_bg_rc_params()):
                # Thinner + taller so it fills the narrow analysis slot.
                fig = plt.figure(figsize=(4.4, 14.4))
                ax = fig.add_subplot(111)
                _plot_raw_branch_velocities(uid=uid, gtr=gtr, ax=ax, logger=logger)
                _force_white_background(fig)
                _save_fig_pdf_and_png(
                    fig=fig,
                    pdf_path=raw_vel_pdf,
                    png_path=raw_vel_png,
                    dpi=DPI_STD,
                    write_png=bool(write_branch_velocities_raw_overlay_png),
                    write_svg=bool(write_branch_velocities_raw_overlay_svg),
                    svg_path=raw_vel_svg,
                )
                plt.close(fig)
        except Exception as e:
            logger.warning("Raw branch velocities plotting failed for unit %s: %s", uid, e)

    if write_branch_velocities_raw_overlay_pdf and raw_vel_pdf.exists():
        outputs["branch_velocities_raw_overlay_pdf"] = str(raw_vel_pdf)
    if write_branch_velocities_raw_overlay_png and raw_vel_png.exists():
        outputs["branch_velocities_raw_overlay_png"] = str(raw_vel_png)
    if write_branch_velocities_raw_overlay_svg and raw_vel_svg.exists():
        outputs["branch_velocities_raw_overlay_svg"] = str(raw_vel_svg)

    overlay_pdf = _unit_output_path(branch_velocities_overlay_pdf_relpath)
    overlay_png = _with_suffix(overlay_pdf, ".png")
    overlay_svg = _with_suffix(overlay_pdf, ".svg")
    if write_branch_velocities_overlay_pdf and ((not overlay_pdf.exists()) or force_restart):
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
                    _save_fig_pdf_and_png(
                        fig=fig,
                        pdf_path=overlay_pdf,
                        png_path=overlay_png,
                        dpi=DPI_STD,
                        write_png=bool(write_branch_velocities_overlay_png),
                        write_svg=bool(write_branch_velocities_overlay_svg),
                        svg_path=overlay_svg,
                    )
                plt.close(fig)
        except Exception as e:
            logger.warning("Overlay velocity plotting failed for unit %s: %s", uid, e)

    if write_branch_velocities_overlay_pdf and overlay_pdf.exists():
        outputs["branch_velocities_overlay_pdf"] = str(overlay_pdf)
    if write_branch_velocities_overlay_png and overlay_png.exists():
        outputs["branch_velocities_overlay_png"] = str(overlay_png)
    if write_branch_velocities_overlay_svg and overlay_svg.exists():
        outputs["branch_velocities_overlay_svg"] = str(overlay_svg)

    for bi, br in enumerate(_as_list(getattr(gtr, "branches", None))):
        if not write_branch_velocity_template:
            continue
        br_base = _unit_output_path(str(branch_velocity_template_relpath).format(index=int(bi)))
        if br_base.suffix:
            br_base = br_base.with_suffix("")
        br_pdf = br_base.with_suffix(".pdf")
        br_png = _with_suffix(br_pdf, ".png")
        br_svg = _with_suffix(br_pdf, ".svg")
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

            _save_fig_pdf_and_png(
                fig=fig,
                pdf_path=br_pdf,
                png_path=br_png,
                dpi=DPI_STD,
                write_png=bool(write_branch_velocity_template_png),
                write_svg=bool(write_branch_velocity_template_svg),
                svg_path=br_svg,
            )
            plt.close(fig)
        except Exception:
            continue

        if br_pdf.exists():
            outputs[f"branch_{bi:02d}_velocity_pdf"] = str(br_pdf)
        if write_branch_velocity_template_png and br_png.exists():
            outputs[f"branch_{bi:02d}_velocity_png"] = str(br_png)
        if write_branch_velocity_template_svg and br_svg.exists():
            outputs[f"branch_{bi:02d}_velocity_svg"] = str(br_svg)

    if write_template_movie and write_template_movie_gif and ((not template_movie_gif.exists()) or force_restart) and (template is not None):
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

    if write_template_movie and template_movie_gif.exists():
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

    if write_summary_clean and (force_restart or (not summary_clean_png.exists())) and branches_clean:
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
            _save_fig_png(
                fig=fig,
                png_path=summary_clean_png,
                dpi=DPI_HI,
                write_png=bool(write_summary_clean_png),
                write_svg=bool(write_summary_clean_svg),
                svg_path=summary_clean_svg,
            )
            plt.close(fig)
        except Exception as e:
            logger.warning("Clean summary plotting failed for unit %s: %s", uid, e)

    if write_summary_raw and (force_restart or (not summary_raw_png.exists())):
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
                _save_fig_png(
                    fig=fig,
                    png_path=summary_raw_png,
                    dpi=DPI_HI,
                    write_png=bool(write_summary_raw_png),
                    write_svg=bool(write_summary_raw_svg),
                    svg_path=summary_raw_svg,
                )
                plt.close(fig)
        except Exception as e:
            logger.warning("Raw summary plotting failed for unit %s: %s", uid, e)

    # Back-compat: keep summary.png as clean summary.
    try:
        if write_summary and summary_clean_png.exists() and (force_restart or (not summary_png.exists())):
            import shutil

            shutil.copyfile(summary_clean_png, summary_png)
            if write_summary_svg and summary_clean_svg.exists():
                shutil.copyfile(summary_clean_svg, summary_svg)
    except Exception:
        pass

    for p, k, enabled in [
        (summary_clean_png, "summary_clean_png", bool(write_summary_clean_png)),
        (summary_raw_png, "summary_raw_png", bool(write_summary_raw_png)),
        (summary_png, "summary_png", bool(write_summary_png)),
        (summary_clean_svg, "summary_clean_svg", bool(write_summary_clean_svg)),
        (summary_raw_svg, "summary_raw_svg", bool(write_summary_raw_svg)),
        (summary_svg, "summary_svg", bool(write_summary_svg)),
    ]:
        if enabled and p.exists():
            outputs[k] = str(p)

    return outputs


