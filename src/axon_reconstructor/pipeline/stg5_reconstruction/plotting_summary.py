"""Summary plotting helpers for reconstruction (internal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from .plotting_core import (
    _force_white_background,
    _read_json,
    _strip_axes_titles,
    _white_bg_rc_params,
    _with_suffix,
)
from .plotting_shared import add_axes_scalebar as _shared_add_axes_scalebar
from .plotting_shared import colorbar_axes_bounds as _shared_colorbar_axes_bounds
from .plotting_shared import normalize_corner_location as _shared_normalize_corner_location
from .plotting_shared import normalize_minimap_linestyle as _shared_normalize_minimap_linestyle
from .plotting_shared import parse_show_ticks_spec as _shared_parse_show_ticks_spec
from .plotting_shared import resolve_colorbar_ticks as _shared_resolve_colorbar_ticks

def compute_raw_branches_for_summary(*, uid: Any, gtr: Any) -> list[dict[str, Any]]:
    """Compute a raw-branches list compatible with axon_velocity plotting helpers.

    Returns a list of dicts containing keys expected by:
      - axon_velocity.plotting.plot_template_propagation (via 'channels')
      - axon_velocity.plotting.plot_branch_velocities (velocity/offset/r2/distances/peak_times)

    This is best-effort and may return fewer raw branches if velocity estimation fails.
    """

    import numpy as np  # type: ignore[import-not-found]

    raw = getattr(gtr, "_paths_raw", None)
    if not raw:
        return []

    est = getattr(gtr, "_estimate_peaks_and_dists", None)
    rve = getattr(gtr, "robust_velocity_estimator", None)

    out: list[dict[str, Any]] = []
    for raw_idx, raw_path in enumerate(list(raw)):
        try:
            # Keep full raw path channels for plotting overlays.
            raw_chans = [int(x) for x in list(raw_path)[::-1]]
            if len(raw_chans) < 2:
                continue

            # Best-effort velocity metadata. If fitting fails, keep channels anyway.
            velocity = None
            offset = None
            r2 = None
            p_value = None
            dists_clean: list[float] = []
            peaks_clean: list[float] = []

            if callable(est) and callable(rve):
                fit_chans = raw_chans[1:] if len(raw_chans) > 2 else raw_chans
                peaks, dists = est(fit_chans)
                peaks = np.asarray(peaks, dtype=float)
                dists = np.asarray(dists, dtype=float)
                if (peaks.size >= 2) and (dists.size == peaks.size):
                    (
                        _path_clean,
                        velocity,
                        offset,
                        r2,
                        p_value,
                        dists_clean,
                        peaks_clean,
                        _inlier_mask,
                    ) = rve(fit_chans, peaks, dists, True)

            # Keep the same keys axon_velocity expects.
            out.append(
                {
                    "branch_index": int(raw_idx),
                    "channels": [int(x) for x in list(raw_chans)],
                    "velocity": float(velocity) if velocity is not None else None,
                    "offset": float(offset) if offset is not None else None,
                    "r2": float(r2) if r2 is not None else None,
                    "pval": float(p_value) if p_value is not None else None,
                    "distances": [float(x) for x in list(dists_clean)],
                    "peak_times": [float(x) for x in list(peaks_clean)],
                }
            )
        except Exception:
            continue

    return out


def _plot_summary_from_parts(
    *,
    template_ch_by_t: Any,
    locs_xy: Any,
    fs_hz: float,
    init_channel: Optional[int],
    branches: list[dict[str, Any]],
    title_suffix: str,
    figsize: tuple[float, float] = (12, 9),
) -> Any:
    """Plot an axon_velocity-style summary using explicit branch dicts."""

    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    from axon_velocity.plotting import (  # type: ignore[import-not-found]
        plot_amplitude_map as av_plot_amplitude_map,
        plot_branch_velocities as av_plot_branch_velocities,
        plot_peak_latency_map as av_plot_peak_latency_map,
        plot_template_propagation as av_plot_template_propagation,
    )

    template = template_ch_by_t
    locations = locs_xy

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax3.axis("off")
    try:
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
    except Exception:
        pass

    av_plot_amplitude_map(template, locations, log=True, ax=ax1)
    av_plot_peak_latency_map(template, locations, float(fs_hz), ax=ax2)

    if init_channel is not None:
        try:
            init = locations[int(init_channel)]
            ax1.plot(init[0], init[1], marker="o", color="r", markersize=5)
            ax2.plot(init[0], init[1], marker="o", color="r", markersize=5)
        except Exception:
            pass

    ax1.set_title(f"amplitude{title_suffix}", fontsize=14)
    ax2.set_title(f"peak latency{title_suffix}", fontsize=14)

    # Propagation row: one mini-axis per branch.
    n = max(1, len(branches))
    gs = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=ax3.get_subplotspec())
    cm = plt.get_cmap("rainbow")

    for bi, branch in enumerate(branches):
        color = cm(bi / max(1, len(branches)))
        sel_idxs = [int(x) for x in branch.get("channels", [])]
        axpr = fig.add_subplot(gs[0, bi])
        try:
            _ = av_plot_template_propagation(
                template,
                locations,
                sel_idxs,
                color="k",
                sort_templates=False,
                color_marker=color,
                ax=axpr,
            )
        except Exception:
            pass
        try:
            axpr.text(
                0.45,
                -0.05,
                f"br.{bi}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axpr.transAxes,
                fontsize=10,
            )
        except Exception:
            pass

        # Overlay path on the maps.
        try:
            for ci, sel in enumerate(sel_idxs[:-1]):
                nxt = sel_idxs[ci + 1]
                ax1.plot(
                    [locations[sel, 0], locations[nxt, 0]],
                    [locations[sel, 1], locations[nxt, 1]],
                    color=color,
                    lw=1,
                    marker=".",
                    markersize=5,
                    alpha=0.8,
                )
                ax2.plot(
                    [locations[sel, 0], locations[nxt, 0]],
                    [locations[sel, 1], locations[nxt, 1]],
                    color=color,
                    lw=1,
                    marker=".",
                    markersize=5,
                    alpha=0.8,
                )
        except Exception:
            pass

    try:
        av_plot_branch_velocities(branches, ax=ax4, cmap="rainbow", fontsize=10)
    except Exception:
        pass
    ax3.set_title(f"propagation{title_suffix}", fontsize=14)
    ax4.set_title(f"velocity{title_suffix}", fontsize=14)
    fig.subplots_adjust(wspace=0.5, hspace=0.2)
    return fig


def write_unit_summary_plots_from_disk(
    *,
    uid: Any,
    out_unit_dir: Path,
    template_ch_by_t: Any,
    locs_xy: Any,
    fs_hz: Optional[float],
    force_restart: bool,
    branches_relpath: str = "branches.json",
    heuristics_relpath: str = "heuristics.json",
    branches_raw_relpath: str = "branches_raw.json",
    summary_relpath: str = "summary.png",
    summary_clean_relpath: str = "summary_clean.png",
    summary_raw_relpath: str = "summary_raw.png",
    logger: Any,
) -> dict[str, str]:
    """(Re)render summary plots without needing a live GraphAxonTracking object."""

    outputs: dict[str, str] = {}

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        logger.warning("Plotting dependencies unavailable: %s", e)
        return outputs

    branches_json = Path(out_unit_dir) / Path(str(branches_relpath)).expanduser()
    heuristics_json = Path(out_unit_dir) / Path(str(heuristics_relpath)).expanduser()
    raw_json = Path(out_unit_dir) / Path(str(branches_raw_relpath)).expanduser()

    branches_clean: list[dict[str, Any]] = []
    init_channel: Optional[int] = None

    try:
        payload = _read_json(branches_json)
        branches_clean = list(payload.get("branches") or [])
        outputs["branches_json"] = str(branches_json)
    except Exception as e:
        logger.warning("Summary-only: missing/invalid branches.json for unit %s: %s", uid, e)

    try:
        payload = _read_json(heuristics_json)
        init_channel = payload.get("heuristics", {}).get("init_channel")
        try:
            init_channel = int(init_channel) if init_channel is not None else None
        except Exception:
            init_channel = None
        outputs["heuristics_json"] = str(heuristics_json)
    except Exception:
        init_channel = None

    branches_raw: list[dict[str, Any]] = []
    if raw_json.exists():
        try:
            payload = _read_json(raw_json)
            branches_raw = list(payload.get("branches") or [])
            outputs["branches_raw_json"] = str(raw_json)
        except Exception:
            branches_raw = []

    # fs fallback: try to infer from template meta already written in templates stage.
    if fs_hz is None:
        fs_hz = 10_000.0

    summary_clean_png = Path(out_unit_dir) / Path(str(summary_clean_relpath)).expanduser()
    summary_raw_png = Path(out_unit_dir) / Path(str(summary_raw_relpath)).expanduser()
    summary_png = Path(out_unit_dir) / Path(str(summary_relpath)).expanduser()

    if (force_restart or (not summary_clean_png.exists())) and branches_clean:
        try:
            with plt.rc_context(_white_bg_rc_params()):
                fig = _plot_summary_from_parts(
                    template_ch_by_t=template_ch_by_t,
                    locs_xy=locs_xy,
                    fs_hz=float(fs_hz),
                    init_channel=init_channel,
                    branches=branches_clean,
                    title_suffix=" (clean)",
                )
            _force_white_background(fig)
            _save_fig_png(fig=fig, png_path=summary_clean_png, dpi=DPI_HI)
            plt.close(fig)
        except Exception as e:
            logger.warning("Summary-only clean summary failed for unit %s: %s", uid, e)

    if (force_restart or (not summary_raw_png.exists())):
        try:
            # Do not fall back: only write raw summary if raw branches exist.
            if not branches_raw:
                logger.warning(
                    "Summary-only: skipping raw summary for unit %s because %s is missing/empty",
                    uid,
                    raw_json,
                )
            else:
                with plt.rc_context(_white_bg_rc_params()):
                    fig = _plot_summary_from_parts(
                        template_ch_by_t=template_ch_by_t,
                        locs_xy=locs_xy,
                        fs_hz=float(fs_hz),
                        init_channel=init_channel,
                        branches=branches_raw,
                        title_suffix=" (raw)",
                    )
                _force_white_background(fig)
                _save_fig_png(fig=fig, png_path=summary_raw_png, dpi=DPI_HI)
                plt.close(fig)
        except Exception as e:
            logger.warning("Summary-only raw summary failed for unit %s: %s", uid, e)

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


def write_top_density_raw_branch_footprint_grid(
    *,
    templates_out_dir: Path,
    reconstruction_out_dir: Path,
    selected_unit_ids: list[Any],
    top_n: Optional[int],
    output_subdir: str = "grids",
    write_pdf: bool = True,
    write_png: bool = True,
    write_ranking_json: bool = True,
    log_basename: str = "raw_branch_log_footprint_top_density_grid",
    linear_basename: str = "raw_branch_linear_footprint_top_density_grid",
    ranking_filename: str = "raw_branch_log_footprint_top_density_ranking.json",
    ncols: int = 5,
    dpi: int = 220,
    panel_background_color: str = "black",
    branch_color: str = "red",
    branch_outline_color: str = "white",
    node_radius_um: float = 5.0,
    soma_node_radius_um: float = 10.0,
    soma_node_color: str = "yellow",
    sort_by: str = "density",
    zoom_priority: str = "branches",
    zoom_padding_percent: float = 20.0,
    force_soma_centering: bool = False,
    soma_xy_show: bool = False,
    soma_xy_color: str = "white",
    soma_xy_fontsize: float = 5.0,
    soma_xy_location: str = "bottom left",
    show_unit_id_in_plot: bool = True,
    unit_id_fontsize: float = 6.0,
    unit_id_color: str = "white",
    show_minimap: bool = True,
    minimap_position: str = "bottomright",
    minimap_size: float = 0.20,
    minimap_outline_color: str = "white",
    minimap_chip_width_mm: float = 3.85,
    minimap_chip_height_mm: float = 2.10,
    minimap_inner_box_linestyle: str = "dotted",
    minimap_inner_box_linewidth: float = 0.8,
    minimap_include_footprint: bool = False,
    minimap_prevent_occlusions: bool = False,
    minimap_clearance_um: float = 2.0,
    minimap_linewidth_buffer_pt: float = 0.5,
    minimap_occlusion_max_iters: int = 8,
    minimap_occlusion_growth_factor: float = 1.20,
    legend_show: bool = False,
    legend_location: str = "first_empty_panel",
    legend_fontsize: float = 6.0,
    legend_fontcolor: str = "white",
    legend_marker_size: float = 3.0,
    legend_show_nodes_in_legend: bool = True,
    legend_show_footprint_in_legend: bool = True,
    local_color_bars_show: bool = False,
    local_color_bars_location: str = "topright",
    local_color_bars_fontsize: float = 5.0,
    local_color_bars_fontcolor: str = "white",
    local_color_bars_length_fraction: float = 0.26,
    local_color_bars_pad_fraction: float = 0.01,
    local_color_bars_show_ticks: Any = None,
    global_color_bar_show: bool = True,
    global_color_bar_location: str = "topright",
    global_color_bar_fontsize: float = 6.0,
    global_color_bar_fontcolor: str = "white",
    global_color_bar_length_fraction: float = 0.30,
    global_color_bar_pad_fraction: float = 0.02,
    global_color_bar_low_color: str = "blue",
    global_color_bar_mid_color: str = "white",
    global_color_bar_high_color: str = "red",
    global_color_bar_force_low_value: float | None = None,
    global_color_bar_force_high_value: float | None = None,
    global_color_bar_show_ticks: Any = None,
    global_color_bar_percentile_low: float = 5.0,
    global_color_bar_percentile_high_linear: float = 99.0,
    global_color_bar_percentile_high_log: float = 99.5,
    global_color_bar_knot_anchor_values: Sequence[float] = (1.0, 10.0),
    global_color_bar_knot_y1_min: float = 0.02,
    global_color_bar_knot_y1_max: float = 0.90,
    global_color_bar_knot_y2_min: float = 0.07,
    global_color_bar_knot_y2_max: float = 0.98,
    global_color_bar_knot_min_gap: float = 0.05,
    global_color_bar_linear_cap_rounding_mode: str = "ceil_step",
    global_color_bar_linear_cap_rounding_step: float = 10.0,
    global_color_bar_linear_cap_min_vmax: float = 11.0,
    emit_debug_logs: bool = False,
    show_scale_debug_text: bool = False,
    show_global_debug_text: bool = False,
    show_local_debug_text: bool = False,
    draw_zoom_range_box: bool = False,
    logger: Any,
) -> dict[str, Any]:
    """Write a top-N grid of raw-branch morphologies over log-zoom footprints.

        Ranking metric (higher is better):
            - sort_by="density": density = n_waveforms_sum / n_channels / footprint_area_um2
            - sort_by="total_branch_length": total reconstructed branch length in um

        where `footprint_area_um2` is the axis-aligned area of contributing channel
        locations expanded by one inferred pitch on each dimension.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as pdf
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as e:
        logger.warning("Footprint-density grid plotting dependencies unavailable: %s", e)
        return {}

    templates_summary_json = Path(templates_out_dir) / "templates_summary.json"
    if not templates_summary_json.exists():
        logger.warning("Skipping footprint-density grid: missing %s", templates_summary_json)
        return {}

    try:
        templates_summary = _read_json(templates_summary_json)
    except Exception as e:
        logger.warning("Skipping footprint-density grid: unreadable %s (%s)", templates_summary_json, e)
        return {}

    unit_id_set = {str(u) for u in list(selected_unit_ids or [])}

    def _infer_pitch_um(locs_xy: Any) -> float:
        try:
            locs = np.asarray(locs_xy, dtype=float)
            if locs.ndim != 2 or locs.shape[0] < 2:
                return 17.5
            # Nearest-neighbor distance (brute force; n is small for contributing channels)
            dmin: list[float] = []
            for i in range(int(locs.shape[0])):
                d = np.sqrt(np.sum((locs - locs[i]) ** 2, axis=1))
                d = d[d > 0]
                if d.size:
                    dmin.append(float(np.min(d)))
            if not dmin:
                return 17.5
            med = float(np.median(np.asarray(dmin, dtype=float)))
            return med if med > 0 else 17.5
        except Exception:
            return 17.5

    def _compute_density(n_wf: float, n_ch: int, locs_xy: Any) -> tuple[float, float]:
        locs = np.asarray(locs_xy, dtype=float)
        if locs.ndim != 2 or int(locs.shape[0]) <= 0:
            return 0.0, 0.0
        x = locs[:, 0]
        y = locs[:, 1]
        pitch = _infer_pitch_um(locs)
        width = float(max(1e-6, (float(np.max(x)) - float(np.min(x)) + float(pitch))))
        height = float(max(1e-6, (float(np.max(y)) - float(np.min(y)) + float(pitch))))
        area = float(width * height)
        denom = float(max(1, int(n_ch))) * area
        return float(n_wf) / float(max(1e-12, denom)), area

    def _compute_total_branch_length_um(*, uid: Any, locs_xy: Any, branches_raw_json: Path) -> float:
        try:
            raw_payload = _read_json(Path(branches_raw_json))
            raw_branches = list((raw_payload or {}).get("branches", []) or [])
        except Exception:
            return 0.0

        if not raw_branches:
            return 0.0

        locs = np.asarray(locs_xy, dtype=float)
        if locs.ndim != 2 or int(locs.shape[0]) <= 1:
            return 0.0

        full_locs_xy = None
        try:
            full_locs_npy = (
                Path(templates_out_dir)
                / "templates"
                / "full"
                / f"unit_{uid}"
                / "full_channel_locations_xy.npy"
            )
            if full_locs_npy.exists():
                full_locs_xy = np.asarray(np.load(str(full_locs_npy), allow_pickle=True), dtype=float)[:, :2]
        except Exception:
            full_locs_xy = None

        def _map_branch_indices(branch_channels: list[int]) -> list[int]:
            if not branch_channels:
                return []

            merged_n = int(locs.shape[0])
            if max(branch_channels) < merged_n:
                return [int(c) for c in branch_channels if 0 <= int(c) < merged_n]

            if full_locs_xy is None:
                return []

            mapped: list[int] = []
            used: set[int] = set()
            for c in branch_channels:
                ci = int(c)
                if ci < 0 or ci >= int(full_locs_xy.shape[0]):
                    continue
                xy = full_locs_xy[ci, :2]
                d = np.sqrt(np.sum((locs - xy) ** 2, axis=1))
                if d.size == 0:
                    continue
                mi = int(np.argmin(d))
                if float(d[mi]) <= 5.0 and mi not in used:
                    mapped.append(mi)
                    used.add(mi)
            return mapped

        total_len = 0.0
        for br in raw_branches:
            try:
                chs = [int(c) for c in list(br.get("channels", []) or []) if isinstance(c, (int, float))]
                if len(chs) < 2:
                    continue
                chs = _map_branch_indices(chs)
                if len(chs) < 2:
                    continue
                pts = locs[np.asarray(chs, dtype=int), :2]
                if int(pts.shape[0]) < 2:
                    continue
                seg = np.diff(np.asarray(pts, dtype=float), axis=0)
                total_len += float(np.sum(np.sqrt(np.sum(seg * seg, axis=1))))
            except Exception:
                continue

        return float(total_len)

    # Build candidate rows from templates summary.
    # Keep a separate pool for scale stats that does not require reconstruction outputs,
    # so vmax is not biased toward only successfully reconstructed/high-SNR units.
    scale_amp_arrays: list[Any] = []
    candidates: list[dict[str, Any]] = []
    for u in list((templates_summary or {}).get("units", []) or []):
        if not isinstance(u, dict):
            continue
        uid = u.get("unit_id")
        if uid is None:
            continue
        if unit_id_set and (str(uid) not in unit_id_set):
            continue

        n_wf = u.get("n_waveforms_sum")
        if n_wf is None:
            continue

        merged_src = None
        for s in list(u.get("sources", []) or []):
            if isinstance(s, dict) and str(s.get("name")) == "merged_contributing":
                merged_src = s
                break
        if not isinstance(merged_src, dict):
            continue

        meta_json = merged_src.get("meta_json")
        if not meta_json:
            continue
        try:
            meta = _read_json(Path(str(meta_json)))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue

        locs_path = meta.get("channel_locations_npy")
        fp_path = meta.get("footprint_ptp_npy")
        if not locs_path or not fp_path:
            continue

        try:
            locs_xy = np.asarray(np.load(str(locs_path), allow_pickle=True), dtype=float)[:, :2]
            footprint_ptp = np.asarray(np.load(str(fp_path), allow_pickle=True), dtype=float).reshape(-1)
            n_ch = int(len(footprint_ptp))
            if int(locs_xy.shape[0]) != int(n_ch):
                continue
            density, area_um2 = _compute_density(float(n_wf), int(n_ch), locs_xy)
            scale_amp_arrays.append(np.asarray(footprint_ptp, dtype=float).reshape(-1))
        except Exception:
            continue

        branches_raw_json = Path(reconstruction_out_dir) / "by_unit" / f"unit_{uid}" / "branches_raw.json"
        if not branches_raw_json.exists():
            try:
                unit_summary_json = Path(reconstruction_out_dir) / "by_unit" / f"unit_{uid}" / "unit_reconstruction_summary.json"
                if unit_summary_json.exists():
                    unit_payload = _read_json(unit_summary_json)
                    candidate_raw = (unit_payload or {}).get("outputs", {}).get("branches_raw_json")
                    if candidate_raw:
                        candidate_path = Path(str(candidate_raw))
                        if candidate_path.exists():
                            branches_raw_json = candidate_path
            except Exception:
                pass
        if not branches_raw_json.exists():
            continue

        total_branch_length_um = _compute_total_branch_length_um(
            uid=uid,
            locs_xy=locs_xy,
            branches_raw_json=branches_raw_json,
        )

        candidates.append(
            {
                "unit_id": uid,
                "density": float(density),
                "total_branch_length_um": float(total_branch_length_um),
                "n_waveforms_sum": int(n_wf),
                "n_channels": int(n_ch),
                "area_um2": float(area_um2),
                "locs_xy": locs_xy,
                "footprint_ptp": footprint_ptp,
                "branches_raw_json": str(branches_raw_json),
            }
        )

    if not candidates:
        logger.info("No candidates available for raw-branch footprint-density grid")
        return {}

    sort_mode = str(sort_by or "density").strip().lower()
    if sort_mode in {"total_branch_length", "branch_length", "total_branch_length_um"}:
        candidates = sorted(
            candidates,
            key=lambda r: (
                -float(r.get("total_branch_length_um", 0.0)),
                -float(r.get("density", 0.0)),
                -int(r.get("n_waveforms_sum", 0)),
                int(r.get("unit_id", 10**12)),
            ),
        )
        ranking_metric = "total_branch_length_um"
        ranking_sort_by = "total_branch_length"
    else:
        candidates = sorted(
            candidates,
            key=lambda r: (
                -float(r.get("density", 0.0)),
                -int(r.get("n_waveforms_sum", 0)),
                int(r.get("unit_id", 10**12)),
            ),
        )
        ranking_metric = "n_waveforms_sum / n_channels / area_um2"
        ranking_sort_by = "density"
    if top_n is None:
        top = list(candidates)
    else:
        top = candidates[: max(1, int(top_n))]

    # Rendering.
    out_dir = Path(reconstruction_out_dir) / str(output_subdir or "grids")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_pdf_path = out_dir / f"{str(log_basename)}.pdf"
    log_png_path = out_dir / f"{str(log_basename)}.png"
    linear_pdf_path = out_dir / f"{str(linear_basename)}.pdf"
    linear_png_path = out_dir / f"{str(linear_basename)}.png"

    ncols = max(1, int(ncols))
    dpi = max(72, int(dpi))
    nrows = max(1, int(np.ceil(float(len(top)) / float(ncols))))

    if bool(emit_debug_logs):
        logger.info(
            "Top-density grid config: out_dir=%s write_pdf=%s write_png=%s write_ranking_json=%s ncols=%d dpi=%d top_n=%s",
            out_dir,
            bool(write_pdf),
            bool(write_png),
            bool(write_ranking_json),
            int(ncols),
            int(dpi),
            "all" if top_n is None else int(top_n),
        )
        logger.info(
            "Top-density grid candidates: total=%d selected=%d",
            int(len(candidates)),
            int(len(top)),
        )

    # Compute color scaling from all eligible template units (not only reconstructed
    # candidates or displayed top-N) so gradients are stable across method variants.
    if scale_amp_arrays:
        all_amp_raw = np.concatenate(scale_amp_arrays)
    else:
        all_amp_raw = np.concatenate([np.asarray(r["footprint_ptp"], dtype=float).reshape(-1) for r in candidates])
    all_amp_raw = all_amp_raw[np.isfinite(all_amp_raw)]
    all_amp_pos = all_amp_raw[all_amp_raw > 0]

    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    import matplotlib.ticker as mticker

    local_ticks_spec = _shared_parse_show_ticks_spec(local_color_bars_show_ticks)
    global_ticks_spec = _shared_parse_show_ticks_spec(global_color_bar_show_ticks)

    try:
        pct_low = float(global_color_bar_percentile_low)
    except Exception:
        pct_low = 5.0
    try:
        pct_high_linear = float(global_color_bar_percentile_high_linear)
    except Exception:
        pct_high_linear = 99.0
    try:
        pct_high_log = float(global_color_bar_percentile_high_log)
    except Exception:
        pct_high_log = 99.5
    pct_low = float(np.clip(pct_low, 0.0, 100.0))
    pct_high_linear = float(np.clip(pct_high_linear, 0.0, 100.0))
    pct_high_log = float(np.clip(pct_high_log, 0.0, 100.0))

    anchors_raw = list(global_color_bar_knot_anchor_values) if global_color_bar_knot_anchor_values is not None else [1.0, 10.0]
    if len(anchors_raw) < 2:
        anchors_raw = [1.0, 10.0]
    try:
        anchor_1 = float(anchors_raw[0])
        anchor_2 = float(anchors_raw[1])
    except Exception:
        anchor_1, anchor_2 = 1.0, 10.0
    if not np.isfinite(anchor_1):
        anchor_1 = 1.0
    if not np.isfinite(anchor_2):
        anchor_2 = 10.0
    if anchor_1 > anchor_2:
        anchor_1, anchor_2 = anchor_2, anchor_1

    y1_min = float(np.clip(float(global_color_bar_knot_y1_min), 0.0, 1.0))
    y1_max = float(np.clip(float(global_color_bar_knot_y1_max), y1_min, 1.0))
    y2_min = float(np.clip(float(global_color_bar_knot_y2_min), 0.0, 1.0))
    y2_max = float(np.clip(float(global_color_bar_knot_y2_max), y2_min, 1.0))
    y_gap = float(max(0.0, float(global_color_bar_knot_min_gap)))

    round_mode = str(global_color_bar_linear_cap_rounding_mode or "ceil_step").strip().lower()
    try:
        round_step = float(global_color_bar_linear_cap_rounding_step)
    except Exception:
        round_step = 10.0
    try:
        linear_cap_min_vmax = float(global_color_bar_linear_cap_min_vmax)
    except Exception:
        linear_cap_min_vmax = 11.0

    def _round_up_nice(x: float) -> float:
        try:
            if not np.isfinite(x) or x <= 0:
                return 1.0
            exp = np.floor(np.log10(float(x)))
            base = float(x) / (10.0**exp)
            if base <= 1.0:
                nice = 1.0
            elif base <= 2.0:
                nice = 2.0
            elif base <= 5.0:
                nice = 5.0
            else:
                nice = 10.0
            return float(nice * (10.0**exp))
        except Exception:
            return float(max(1.0, x))

    def _round_up_10(x: float) -> float:
        try:
            if not np.isfinite(x):
                return 10.0
            return float(10.0 * np.ceil(float(x) / 10.0))
        except Exception:
            return 10.0

    def _round_up_step(x: float, step: float) -> float:
        try:
            if not np.isfinite(x):
                return float(max(1.0, step))
            s = max(1e-12, float(step))
            return float(s * np.ceil(float(x) / s))
        except Exception:
            return float(max(1.0, step))

    if all_amp_pos.size:
        log_vmin = float(np.percentile(all_amp_pos, pct_low))
        log_vmax = float(np.percentile(all_amp_pos, pct_high_log))
    else:
        log_vmin, log_vmax = 1e-3, 1.0
    if log_vmax <= log_vmin:
        log_vmax = log_vmin * 1.01
    if global_color_bar_force_low_value is not None:
        try:
            log_vmin = max(1e-12, float(global_color_bar_force_low_value))
        except Exception:
            pass
    if global_color_bar_force_high_value is not None:
        try:
            log_vmax = max(log_vmin * 1.01, float(global_color_bar_force_high_value))
        except Exception:
            pass
    log_norm = mcolors.LogNorm(vmin=max(1e-12, float(log_vmin)), vmax=max(float(log_vmax), float(log_vmin) * 1.01))

    # Dynamic piecewise anchored scaling for the comparison panel,
    # with anchors at [vmin, 1, 10, vmax] and knot positions driven by data percentiles.
    detected_amp_max = float(np.max(all_amp_raw)) if all_amp_raw.size else 1.0
    dyn_vmax_raw = float('nan')
    if all_amp_pos.size:
        dyn_vmax_raw = float(np.percentile(all_amp_pos, pct_high_linear))
        dyn_vmin = 0.0
        if dyn_vmax_raw <= dyn_vmin * 1.01:
            dyn_vmax_raw = dyn_vmin * 1.01
        if round_mode in {"none", "off", "disabled"}:
            dyn_vmax_rounded = float(dyn_vmax_raw)
        elif round_mode in {"nice", "nice_1_2_5", "1-2-5"}:
            dyn_vmax_rounded = float(_round_up_nice(dyn_vmax_raw))
        elif round_mode in {"ceil_step", "step", "round_step"}:
            dyn_vmax_rounded = float(_round_up_step(dyn_vmax_raw, round_step))
        else:
            dyn_vmax_rounded = float(_round_up_10(dyn_vmax_raw))
        dyn_vmax = float(max(linear_cap_min_vmax, dyn_vmax_rounded))
    else:
        dyn_vmin = 0.0
        dyn_vmax = 100.0
    if global_color_bar_force_low_value is not None:
        try:
            dyn_vmin = float(global_color_bar_force_low_value)
        except Exception:
            pass
    if global_color_bar_force_high_value is not None:
        try:
            dyn_vmax = float(global_color_bar_force_high_value)
        except Exception:
            pass
    if dyn_vmax <= dyn_vmin:
        dyn_vmax = dyn_vmin + 1.0
    if all_amp_raw.size:
        amp_sorted = np.sort(np.asarray(all_amp_raw, dtype=float))
    else:
        amp_sorted = np.asarray([0.0, 1.0, 10.0, dyn_vmax], dtype=float)

    def _pct_rank(v: float) -> float:
        if amp_sorted.size == 0:
            return 0.5
        idx = int(np.searchsorted(amp_sorted, float(v), side="right"))
        return float(idx) / float(max(1, amp_sorted.size))

    anchor_1 = float(np.clip(anchor_1, dyn_vmin, dyn_vmax))
    anchor_2 = float(np.clip(anchor_2, dyn_vmin, dyn_vmax))
    if anchor_2 <= anchor_1:
        anchor_2 = min(dyn_vmax, anchor_1 + max(1e-6, 0.01 * max(1.0, dyn_vmax - dyn_vmin)))

    y1 = float(np.clip(_pct_rank(anchor_1), y1_min, y1_max))
    y2_lo = max(y2_min, y1 + y_gap)
    y2 = float(np.clip(_pct_rank(anchor_2), y2_lo, y2_max))
    if y2 <= y1:
        y2 = min(1.0, y1 + max(1e-6, y_gap))

    x_knots = np.asarray([float(dyn_vmin), float(anchor_1), float(anchor_2), float(dyn_vmax)], dtype=float)
    y_knots = np.asarray([0.0, y1, y2, 1.0], dtype=float)

    def _piecewise_forward(x: Any) -> Any:
        arr = np.asarray(x, dtype=float)
        arr = np.clip(arr, x_knots[0], x_knots[-1])
        return np.interp(arr, x_knots, y_knots)

    def _piecewise_inverse(y: Any) -> Any:
        arr = np.asarray(y, dtype=float)
        arr = np.clip(arr, y_knots[0], y_knots[-1])
        return np.interp(arr, y_knots, x_knots)

    dynamic_piecewise_norm = mcolors.FuncNorm(
        (_piecewise_forward, _piecewise_inverse),
        vmin=float(dyn_vmin),
        vmax=float(dyn_vmax),
    )

    # Position color stops so 10 µV is above white, in the orange regime.
    n10_ref = float(np.clip(_piecewise_forward(10.0), 0.0, 1.0))
    white_pos = float(np.clip(n10_ref - 0.10, 0.20, 0.85))
    orange_pos = float(np.clip(n10_ref, white_pos + 0.03, 0.95))
    blue_pos = float(np.clip(white_pos * 0.45, 0.05, white_pos - 0.02))

    cmap_log = LinearSegmentedColormap.from_list(
        "black_low_mid_high",
        ["#000000", str(global_color_bar_low_color), str(global_color_bar_mid_color), str(global_color_bar_high_color)],
        N=256,
    )

    cmap_linear = LinearSegmentedColormap.from_list(
        "black_low_mid_high_linear_dynamic",
        [
            (0.0, "#000000"),
            (blue_pos, str(global_color_bar_low_color)),
            (white_pos, str(global_color_bar_mid_color)),
            (1.0, str(global_color_bar_high_color)),
        ],
        N=256,
    )

    def _add_scalebar(
        ax: Any,
        *,
        zoom_x0: float,
        zoom_x1: float,
        zoom_y0: float,
        zoom_y1: float,
    ) -> None:
        try:
            span = max(1.0, float(zoom_x1) - float(zoom_x0))
            bar_um = 100.0 if span >= 180.0 else 50.0
            margin = 0.06 * span

            x1 = float(zoom_x1) - margin
            x0 = x1 - float(bar_um)
            y0 = float(zoom_y0) + margin
            ax.plot([x0, x1], [y0, y0], color="white", lw=1.8, solid_capstyle="butt", zorder=8)
            ax.text(
                (x0 + x1) * 0.5,
                y0 + 0.04 * span,
                f"{int(bar_um)} µm",
                color="white",
                fontsize=6,
                ha="center",
                va="bottom",
                zorder=9,
            )
        except Exception:
            return

    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    def _fmt_tick_plain(v: float) -> str:
        vv = float(v)
        if vv >= 100:
            return f"{vv:.0f}"
        if vv >= 10:
            return f"{vv:.1f}".rstrip("0").rstrip(".")
        return f"{vv:.2g}" if vv >= 1 else f"{vv:.2f}".rstrip("0").rstrip(".")

    def _render_grid(
        *,
        norm: Any,
        cmap_render: Any,
        colorbar_label: str,
        pdf_path: Path,
        png_path: Path,
        use_log_norm: bool,
        use_log_tick_format: bool,
    ) -> None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 2.6 * float(nrows)), constrained_layout=False)
        axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

        color_mappable = mcm.ScalarMappable(norm=norm, cmap=cmap_render)
        color_mappable.set_array([])
        effective_global_debug = bool(show_global_debug_text) or bool(show_scale_debug_text)
        effective_local_debug = bool(show_local_debug_text)
        reserved_debug_axis_index: Optional[int] = None

        for ax, row in zip(axes_list, top):
            uid = row["unit_id"]
            locs_xy = np.asarray(row["locs_xy"], dtype=float)
            amp = np.asarray(row["footprint_ptp"], dtype=float).reshape(-1)
            ax.set_facecolor(str(panel_background_color))
            branch_points_xy: list[Any] = []
            soma_xy: tuple[float, float] | None = None

            # Draw footprint channels as true 17.5um x 17.5um squares in data coordinates.
            # This preserves physical scale against the zoom/scalebar across subplots.
            ch_pitch_um = 17.5
            amp_for_color = np.maximum(amp, 1e-12) if bool(use_log_norm) else np.maximum(amp, 0.0)
            facecolors = cmap_render(norm(amp_for_color))
            for (x, y), fc in zip(locs_xy[:, :2], facecolors):
                ax.add_patch(
                    Rectangle(
                        (float(x) - 0.5 * ch_pitch_um, float(y) - 0.5 * ch_pitch_um),
                        ch_pitch_um,
                        ch_pitch_um,
                        facecolor=fc,
                        edgecolor="none",
                        alpha=0.95,
                        zorder=1,
                    )
                )

            # Overlay raw branches using channel indices on same merged-contributing axis.
            n_branches_plotted = 0
            branch_channel_pts: Any | None = None
            try:
                raw_payload = _read_json(Path(str(row["branches_raw_json"])))
                raw_branches = list((raw_payload or {}).get("branches", []) or [])
                mapped_branch_channels: list[list[int]] = []

                # Branch channel indices are produced against the template source used in
                # reconstruction (often full-channel templates). The footprint grid uses
                # merged-contributing channel coordinates, so map full->merged by location.
                full_locs_xy = None
                full_locs_npy = (
                    Path(templates_out_dir)
                    / "templates"
                    / "full"
                    / f"unit_{uid}"
                    / "full_channel_locations_xy.npy"
                )
                if full_locs_npy.exists():
                    try:
                        full_locs_xy = np.asarray(np.load(str(full_locs_npy), allow_pickle=True), dtype=float)[:, :2]
                    except Exception:
                        full_locs_xy = None

                def _map_branch_indices(branch_channels: list[int]) -> list[int]:
                    if not branch_channels:
                        return []

                    merged_n = int(locs_xy.shape[0])
                    # Direct case: indices already in merged-contributing axis.
                    if max(branch_channels) < merged_n:
                        return [int(c) for c in branch_channels if 0 <= int(c) < merged_n]

                    # Fallback: map from full-channel indices to merged axis by nearest location.
                    if full_locs_xy is None:
                        return []

                    mapped: list[int] = []
                    used: set[int] = set()
                    for c in branch_channels:
                        ci = int(c)
                        if ci < 0 or ci >= int(full_locs_xy.shape[0]):
                            continue
                        xy = full_locs_xy[ci, :2]
                        d = np.sqrt(np.sum((locs_xy - xy) ** 2, axis=1))
                        if d.size == 0:
                            continue
                        mi = int(np.argmin(d))
                        # Maxwell pitch-scale tolerant matching.
                        if float(d[mi]) <= 5.0 and mi not in used:
                            mapped.append(mi)
                            used.add(mi)
                    return mapped

                branch_outline_lw_pt = 2.2 if str(branch_outline_color).strip() else 0.0
                branch_main_lw_pt = 0.9

                for bi, br in enumerate(raw_branches):
                    chs = [int(c) for c in list(br.get("channels", []) or []) if isinstance(c, (int, float))]
                    if len(chs) < 2:
                        continue
                    chs = _map_branch_indices(chs)
                    if len(chs) < 2:
                        continue
                    mapped_branch_channels.append([int(c) for c in chs])
                    pts = locs_xy[np.asarray(chs, dtype=int), :2]
                    # Optional outline under branch traces for visibility on bright footprint regions.
                    if str(branch_outline_color).strip():
                        ax.plot(
                            pts[:, 0],
                            pts[:, 1],
                            color=str(branch_outline_color),
                            lw=float(branch_outline_lw_pt),
                            alpha=0.9,
                            zorder=2,
                        )
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        color=str(branch_color),
                        lw=float(branch_main_lw_pt),
                        alpha=0.95,
                        zorder=3,
                    )
                    branch_points_xy.append(np.asarray(pts, dtype=float))
                    n_branches_plotted += 1

                # Draw node markers with soma candidate highlighted.
                soma_ch: int | None = None
                if mapped_branch_channels:
                    b0 = list(mapped_branch_channels[0])
                    counts: dict[int, int] = {}
                    for br in mapped_branch_channels[1:]:
                        for ch in set(int(c) for c in br):
                            counts[int(ch)] = int(counts.get(int(ch), 0) + 1)
                    if b0:
                        best_score = -1
                        best_idx = 10**9
                        best_ch = int(b0[0])
                        for i, ch in enumerate(b0):
                            score = int(counts.get(int(ch), 0))
                            if score > best_score or (score == best_score and i < best_idx):
                                best_score = score
                                best_idx = i
                                best_ch = int(ch)
                        soma_ch = int(best_ch)

                node_r = max(0.1, float(node_radius_um))
                soma_r = max(node_r, float(soma_node_radius_um))
                outline_color = str(branch_outline_color).strip()
                edge_c = outline_color if outline_color else "none"
                edge_lw = 0.55 if outline_color else 0.0

                all_nodes: set[int] = set()
                for br in mapped_branch_channels:
                    all_nodes.update(int(c) for c in br)

                if all_nodes:
                    try:
                        branch_channel_pts = np.asarray(
                            locs_xy[np.asarray(sorted(int(c) for c in all_nodes), dtype=int), :2],
                            dtype=float,
                        )
                    except Exception:
                        branch_channel_pts = None

                for ch in sorted(all_nodes):
                    if soma_ch is not None and int(ch) == int(soma_ch):
                        continue
                    try:
                        x = float(locs_xy[int(ch), 0])
                        y = float(locs_xy[int(ch), 1])
                        ax.add_patch(
                            Circle(
                                (x, y),
                                radius=float(node_r),
                                facecolor=str(branch_color),
                                edgecolor=edge_c,
                                linewidth=float(edge_lw),
                                zorder=4,
                            )
                        )
                    except Exception:
                        continue

                if soma_ch is not None:
                    try:
                        sx = float(locs_xy[int(soma_ch), 0])
                        sy = float(locs_xy[int(soma_ch), 1])
                        soma_xy = (float(sx), float(sy))
                        ax.add_patch(
                            Circle(
                                (sx, sy),
                                radius=float(soma_r),
                                facecolor=str(soma_node_color),
                                edgecolor=edge_c,
                                linewidth=float(edge_lw),
                                zorder=5,
                            )
                        )
                    except Exception:
                        pass

                node_collision_geom: list[tuple[float, float, float, bool]] = []
                try:
                    for ch in sorted(all_nodes):
                        if soma_ch is not None and int(ch) == int(soma_ch):
                            continue
                        nx = float(locs_xy[int(ch), 0])
                        ny = float(locs_xy[int(ch), 1])
                        node_collision_geom.append((float(nx), float(ny), float(node_r), False))
                    if soma_xy is not None:
                        node_collision_geom.append((float(soma_xy[0]), float(soma_xy[1]), float(soma_r), True))
                except Exception:
                    node_collision_geom = []
            except Exception:
                pass

            # Zoom to local footprint extent with small margin.
            zoom_x0 = None
            zoom_x1 = None
            zoom_y0 = None
            zoom_y1 = None
            try:
                high_amp_pts = None
                try:
                    high_amp_mask = np.asarray(amp, dtype=float) > 10.0
                    if np.any(high_amp_mask):
                        high_amp_pts = np.asarray(locs_xy[np.asarray(high_amp_mask), :2], dtype=float)
                except Exception:
                    high_amp_pts = None

                priority = str(zoom_priority or "branches").strip().lower()
                pad_pct_raw = max(0.0, float(zoom_padding_percent))
                pad_frac = (pad_pct_raw / 100.0) if pad_pct_raw > 1.0 else pad_pct_raw

                if priority == "channels":
                    if branch_channel_pts is not None and int(getattr(branch_channel_pts, "shape", [0])[0]) > 0:
                        pts_for_zoom = branch_channel_pts
                    elif high_amp_pts is not None and int(high_amp_pts.shape[0]) > 0:
                        pts_for_zoom = high_amp_pts
                    else:
                        pts_for_zoom = np.asarray(locs_xy, dtype=float)
                else:
                    # Default: branch-priority zoom
                    if branch_points_xy:
                        pts_for_zoom = np.vstack(branch_points_xy)
                    elif branch_channel_pts is not None and int(getattr(branch_channel_pts, "shape", [0])[0]) > 0:
                        pts_for_zoom = branch_channel_pts
                    elif high_amp_pts is not None and int(high_amp_pts.shape[0]) > 0:
                        pts_for_zoom = high_amp_pts
                    else:
                        pts_for_zoom = np.asarray(locs_xy, dtype=float)

                if int(getattr(pts_for_zoom, "shape", [0])[0]) > 0:
                    xmin, xmax = float(np.min(pts_for_zoom[:, 0])), float(np.max(pts_for_zoom[:, 0]))
                    ymin, ymax = float(np.min(pts_for_zoom[:, 1])), float(np.max(pts_for_zoom[:, 1]))
                else:
                    xmin, xmax = float(np.min(locs_xy[:, 0])), float(np.max(locs_xy[:, 0]))
                    ymin, ymax = float(np.min(locs_xy[:, 1])), float(np.max(locs_xy[:, 1]))

                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                span_x = max(1e-6, (xmax - xmin))
                span_y = max(1e-6, (ymax - ymin))
                span = max(span_x, span_y)

                if bool(force_soma_centering) and (soma_xy is not None):
                    sx, sy = float(soma_xy[0]), float(soma_xy[1])
                    req_half = max(
                        abs(float(xmin) - sx),
                        abs(float(xmax) - sx),
                        abs(float(ymin) - sy),
                        abs(float(ymax) - sy),
                        1e-6,
                    )
                    base_half = max(0.5 * span, req_half)
                    cx, cy = sx, sy
                    half_span = base_half * (1.0 + (2.0 * pad_frac))
                else:
                    half_span = 0.5 * span + (pad_frac * span)

                if bool(show_minimap) and bool(minimap_prevent_occlusions):
                    occ_pts_list: list[Any] = []
                    if branch_points_xy:
                        try:
                            occ_pts_list.append(np.vstack(branch_points_xy))
                        except Exception:
                            pass
                    if branch_channel_pts is not None and int(getattr(branch_channel_pts, "shape", [0])[0]) > 0:
                        occ_pts_list.append(np.asarray(branch_channel_pts, dtype=float))

                    if occ_pts_list:
                        try:
                            occ_pts = np.vstack(occ_pts_list)
                        except Exception:
                            occ_pts = None

                        if occ_pts is not None and int(getattr(occ_pts, "shape", [0])[0]) > 0:
                            size = min(0.45, max(0.08, float(minimap_size)))
                            chip_w_mm = max(1e-9, float(minimap_chip_width_mm))
                            chip_h_mm = max(1e-9, float(minimap_chip_height_mm))
                            chip_aspect = chip_w_mm / chip_h_mm
                            ins_w = float(size)
                            ins_h = float(max(0.06, size / max(1e-9, chip_aspect)))
                            margin = 0.03
                            pos = str(minimap_position or "bottomright").strip().lower()
                            if pos not in {"topleft", "topright", "bottomleft", "bottomright"}:
                                pos = "bottomright"
                            if pos == "topleft":
                                x0_ins, y0_ins = margin, 1.0 - ins_h - margin
                            elif pos == "topright":
                                x0_ins, y0_ins = 1.0 - ins_w - margin, 1.0 - ins_h - margin
                            elif pos == "bottomleft":
                                x0_ins, y0_ins = margin, margin
                            else:
                                x0_ins, y0_ins = 1.0 - ins_w - margin, margin

                            x1_ins = x0_ins + ins_w
                            y1_ins = y0_ins + ins_h

                            def _segments_for_occlusion() -> list[tuple[float, float, float, float]]:
                                segs: list[tuple[float, float, float, float]] = []
                                try:
                                    for bpts in branch_points_xy:
                                        arr = np.asarray(bpts, dtype=float)
                                        if arr.ndim != 2 or arr.shape[0] < 2:
                                            continue
                                        for i in range(int(arr.shape[0]) - 1):
                                            segs.append((float(arr[i, 0]), float(arr[i, 1]), float(arr[i + 1, 0]), float(arr[i + 1, 1])))
                                except Exception:
                                    return []
                                return segs

                            segs_for_occlusion = _segments_for_occlusion()

                            def _point_in_rect(px: float, py: float, rx0: float, ry0: float, rx1: float, ry1: float) -> bool:
                                return (px >= rx0) and (px <= rx1) and (py >= ry0) and (py <= ry1)

                            def _seg_intersects_seg(
                                ax0: float,
                                ay0: float,
                                ax1: float,
                                ay1: float,
                                bx0: float,
                                by0: float,
                                bx1: float,
                                by1: float,
                            ) -> bool:
                                def _orient(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> float:
                                    return (qy - py) * (rx - qx) - (qx - px) * (ry - qy)

                                def _on_seg(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> bool:
                                    return min(px, rx) <= qx <= max(px, rx) and min(py, ry) <= qy <= max(py, ry)

                                o1 = _orient(ax0, ay0, ax1, ay1, bx0, by0)
                                o2 = _orient(ax0, ay0, ax1, ay1, bx1, by1)
                                o3 = _orient(bx0, by0, bx1, by1, ax0, ay0)
                                o4 = _orient(bx0, by0, bx1, by1, ax1, ay1)

                                if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
                                    return True
                                eps = 1e-12
                                if abs(o1) <= eps and _on_seg(ax0, ay0, bx0, by0, ax1, ay1):
                                    return True
                                if abs(o2) <= eps and _on_seg(ax0, ay0, bx1, by1, ax1, ay1):
                                    return True
                                if abs(o3) <= eps and _on_seg(bx0, by0, ax0, ay0, bx1, by1):
                                    return True
                                if abs(o4) <= eps and _on_seg(bx0, by0, ax1, ay1, bx1, by1):
                                    return True
                                return False

                            def _segment_hits_rect_with_buffer(
                                x0s: float,
                                y0s: float,
                                x1s: float,
                                y1s: float,
                                rx0: float,
                                ry0: float,
                                rx1: float,
                                ry1: float,
                                rb: float,
                            ) -> bool:
                                ex0 = rx0 - rb
                                ey0 = ry0 - rb
                                ex1 = rx1 + rb
                                ey1 = ry1 + rb

                                if max(x0s, x1s) < ex0 or min(x0s, x1s) > ex1:
                                    return False
                                if max(y0s, y1s) < ey0 or min(y0s, y1s) > ey1:
                                    return False

                                if _point_in_rect(x0s, y0s, ex0, ey0, ex1, ey1) or _point_in_rect(x1s, y1s, ex0, ey0, ex1, ey1):
                                    return True

                                return (
                                    _seg_intersects_seg(x0s, y0s, x1s, y1s, ex0, ey0, ex1, ey0)
                                    or _seg_intersects_seg(x0s, y0s, x1s, y1s, ex1, ey0, ex1, ey1)
                                    or _seg_intersects_seg(x0s, y0s, x1s, y1s, ex1, ey1, ex0, ey1)
                                    or _seg_intersects_seg(x0s, y0s, x1s, y1s, ex0, ey1, ex0, ey0)
                                )

                            def _pt_to_data_um(test_half_span: float, pt: float) -> float:
                                try:
                                    hs = max(1e-6, float(test_half_span))
                                    span = 2.0 * hs
                                    bbox = ax.get_window_extent()
                                    w_px = max(1.0, float(getattr(bbox, "width", 1.0)))
                                    h_px = max(1.0, float(getattr(bbox, "height", 1.0)))
                                    px_per_pt = float(getattr(fig, "dpi", 100.0)) / 72.0
                                    d_x = span / w_px
                                    d_y = span / h_px
                                    return float(pt) * px_per_pt * max(d_x, d_y)
                                except Exception:
                                    return 0.0

                            def _has_occlusion(test_half_span: float) -> tuple[bool, str]:
                                hs = max(1e-6, float(test_half_span))
                                x0 = float(cx) - hs
                                x1 = float(cx) + hs
                                y0 = float(cy) - hs
                                y1 = float(cy) + hs
                                fx = (np.asarray(occ_pts[:, 0], dtype=float) - x0) / max(1e-12, (x1 - x0))
                                fy = (np.asarray(occ_pts[:, 1], dtype=float) - y0) / max(1e-12, (y1 - y0))
                                in_x = np.logical_and(fx >= x0_ins, fx <= x1_ins)
                                in_y = np.logical_and(fy >= y0_ins, fy <= y1_ins)
                                if bool(np.any(np.logical_and(in_x, in_y))):
                                    return True, "point"

                                # Geometry-aware checks: node circles and branch strokes.
                                clearance_um = max(0.0, float(minimap_clearance_um))
                                stroke_um = _pt_to_data_um(
                                    hs,
                                    max(0.0, 0.5 * max(float(branch_main_lw_pt), float(branch_outline_lw_pt)) + float(minimap_linewidth_buffer_pt)),
                                )

                                # Node circles
                                try:
                                    for nx, ny, nr, is_soma in node_collision_geom:
                                        fnx = (float(nx) - x0) / max(1e-12, (x1 - x0))
                                        fny = (float(ny) - y0) / max(1e-12, (y1 - y0))
                                        rn = max(0.0, (float(nr) + clearance_um) / max(1e-12, (2.0 * hs)))
                                        dx = max(float(x0_ins) - fnx, 0.0, fnx - float(x1_ins))
                                        dy = max(float(y0_ins) - fny, 0.0, fny - float(y1_ins))
                                        if (dx * dx + dy * dy) <= (rn * rn):
                                            return True, ("soma" if bool(is_soma) else "node")
                                except Exception:
                                    pass

                                # Branch segments with line buffer.
                                try:
                                    rb = max(0.0, (clearance_um + stroke_um) / max(1e-12, (2.0 * hs)))
                                    for sx0, sy0, sx1, sy1 in segs_for_occlusion:
                                        fsx0 = (float(sx0) - x0) / max(1e-12, (x1 - x0))
                                        fsy0 = (float(sy0) - y0) / max(1e-12, (y1 - y0))
                                        fsx1 = (float(sx1) - x0) / max(1e-12, (x1 - x0))
                                        fsy1 = (float(sy1) - y0) / max(1e-12, (y1 - y0))
                                        if _segment_hits_rect_with_buffer(
                                            fsx0,
                                            fsy0,
                                            fsx1,
                                            fsy1,
                                            float(x0_ins),
                                            float(y0_ins),
                                            float(x1_ins),
                                            float(y1_ins),
                                            float(rb),
                                        ):
                                            return True, "branch"
                                except Exception:
                                    pass

                                return False, "none"

                            hs = float(half_span)
                            occ_iters = max(1, int(minimap_occlusion_max_iters))
                            growth = max(1.01, float(minimap_occlusion_growth_factor))
                            last_reason = "none"
                            for _ in range(occ_iters):
                                has_occ, reason = _has_occlusion(hs)
                                last_reason = str(reason)
                                if not has_occ:
                                    break
                                hs *= growth
                            if bool(emit_debug_logs):
                                logger.debug(
                                    "Grid occlusion adjust unit=%s initial_half_span=%.3f final_half_span=%.3f iters=%d reason=%s",
                                    str(uid),
                                    float(half_span),
                                    float(hs),
                                    int(occ_iters),
                                    str(last_reason),
                                )
                            half_span = float(hs)

                zoom_x0, zoom_x1 = (cx - half_span), (cx + half_span)
                zoom_y0, zoom_y1 = (cy - half_span), (cy + half_span)
                ax.set_xlim(zoom_x0, zoom_x1)
                ax.set_ylim(zoom_y0, zoom_y1)
            except Exception:
                pass

            if bool(draw_zoom_range_box) and None not in (zoom_x0, zoom_x1, zoom_y0, zoom_y1):
                try:
                    rect = Rectangle(
                        (float(zoom_x0), float(zoom_y0)),
                        float(zoom_x1) - float(zoom_x0),
                        float(zoom_y1) - float(zoom_y0),
                        fill=False,
                        edgecolor="#9a9a9a",
                        linewidth=0.9,
                        linestyle=(0, (1.0, 1.6)),
                        zorder=7,
                    )
                    ax.add_patch(rect)
                except Exception:
                    pass

            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(
                f"d={float(row['density']):.2e}  br={int(n_branches_plotted)}\nWf={int(row['n_waveforms_sum'])} Ch={int(row['n_channels'])}",
                fontsize=7,
                pad=1.5,
                color="white",
            )
            if bool(show_unit_id_in_plot):
                try:
                    ax.text(
                        0.98,
                        0.98,
                        f"u{uid}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=float(unit_id_fontsize),
                        color=str(unit_id_color),
                        zorder=11,
                        bbox={"facecolor": "black", "alpha": 0.30, "pad": 1.5, "edgecolor": "none"},
                    )
                except Exception:
                    pass
            if bool(soma_xy_show) and (soma_xy is not None):
                try:
                    sx, sy = float(soma_xy[0]), float(soma_xy[1])
                    loc_raw = str(soma_xy_location or "bottom left").strip().lower().replace("_", " ")
                    loc_map = {
                        "top left": (0.02, 0.98, "left", "top"),
                        "topleft": (0.02, 0.98, "left", "top"),
                        "top right": (0.98, 0.98, "right", "top"),
                        "topright": (0.98, 0.98, "right", "top"),
                        "bottom left": (0.02, 0.02, "left", "bottom"),
                        "bottomleft": (0.02, 0.02, "left", "bottom"),
                        "bottom right": (0.98, 0.02, "right", "bottom"),
                        "bottomright": (0.98, 0.02, "right", "bottom"),
                    }
                    tx, ty, ha, va = loc_map.get(loc_raw, (0.02, 0.02, "left", "bottom"))
                    ax.text(
                        float(tx),
                        float(ty),
                        f"({sx:.1f}, {sy:.1f}) um",
                        transform=ax.transAxes,
                        ha=str(ha),
                        va=str(va),
                        fontsize=max(4.0, float(soma_xy_fontsize)),
                        color=str(soma_xy_color),
                        zorder=11,
                        bbox={"facecolor": "black", "alpha": 0.30, "pad": 1.5, "edgecolor": "none"},
                    )
                except Exception:
                    pass
            if None not in (zoom_x0, zoom_x1, zoom_y0, zoom_y1):
                _shared_add_axes_scalebar(
                    ax,
                    zoom_x0=float(zoom_x0),
                    zoom_x1=float(zoom_x1),
                    zoom_y0=float(zoom_y0),
                    zoom_y1=float(zoom_y1),
                )

            if bool(show_minimap):
                try:
                    size = float(minimap_size)
                    size = min(0.45, max(0.08, size))

                    chip_w_mm = max(1e-9, float(minimap_chip_width_mm))
                    chip_h_mm = max(1e-9, float(minimap_chip_height_mm))
                    chip_aspect = chip_w_mm / chip_h_mm
                    ins_w = size
                    ins_h = max(0.06, size / max(1e-9, chip_aspect))

                    pos = str(minimap_position or "bottomright").strip().lower()
                    if pos not in {"topleft", "topright", "bottomleft", "bottomright"}:
                        pos = "bottomright"
                    margin = 0.03
                    if pos == "topleft":
                        x0_ins, y0_ins = margin, 1.0 - ins_h - margin
                    elif pos == "topright":
                        x0_ins, y0_ins = 1.0 - ins_w - margin, 1.0 - ins_h - margin
                    elif pos == "bottomleft":
                        x0_ins, y0_ins = margin, margin
                    else:
                        x0_ins, y0_ins = 1.0 - ins_w - margin, margin

                    mini = ax.inset_axes([x0_ins, y0_ins, ins_w, ins_h])
                    mini.set_facecolor(str(panel_background_color))
                    mini.set_xticks([])
                    mini.set_yticks([])
                    for sp in mini.spines.values():
                        sp.set_visible(False)

                    if bool(minimap_include_footprint):
                        try:
                            ch_pitch_um = 17.5
                            amp_for_mini = np.maximum(amp, 1e-12) if bool(use_log_norm) else np.maximum(amp, 0.0)
                            mini_facecolors = cmap_render(norm(amp_for_mini))
                            for (x, y), fc in zip(locs_xy[:, :2], mini_facecolors):
                                mini.add_patch(
                                    Rectangle(
                                        (float(x) - 0.5 * ch_pitch_um, float(y) - 0.5 * ch_pitch_um),
                                        ch_pitch_um,
                                        ch_pitch_um,
                                        facecolor=fc,
                                        edgecolor="none",
                                        alpha=0.95,
                                        zorder=1,
                                    )
                                )
                        except Exception:
                            pass

                    # Use full-chip coordinate frame when available; fallback to current panel extents.
                    chip_xy = locs_xy
                    try:
                        full_locs_npy = (
                            Path(templates_out_dir)
                            / "templates"
                            / "full"
                            / f"unit_{uid}"
                            / "full_channel_locations_xy.npy"
                        )
                        if full_locs_npy.exists():
                            chip_xy = np.asarray(np.load(str(full_locs_npy), allow_pickle=True), dtype=float)[:, :2]
                    except Exception:
                        chip_xy = locs_xy

                    chip_x0 = float(np.min(chip_xy[:, 0]))
                    chip_x1 = float(np.max(chip_xy[:, 0]))
                    chip_y0 = float(np.min(chip_xy[:, 1]))
                    chip_y1 = float(np.max(chip_xy[:, 1]))

                    chip_w = max(1e-9, chip_x1 - chip_x0)
                    chip_h = max(1e-9, chip_y1 - chip_y0)

                    # Draw a full-chip rectangle in chip-coordinate units.
                    mini.add_patch(
                        Rectangle(
                            (chip_x0, chip_y0),
                            chip_w,
                            chip_h,
                            fill=False,
                            edgecolor=str(minimap_outline_color),
                            linewidth=0.9,
                            zorder=2,
                        )
                    )

                    # Draw current zoom window as a *square* in chip coordinates.
                    if None not in (zoom_x0, zoom_x1, zoom_y0, zoom_y1):
                        zx0 = float(zoom_x0)
                        zx1 = float(zoom_x1)
                        zy0 = float(zoom_y0)
                        zy1 = float(zoom_y1)

                        cx = 0.5 * (zx0 + zx1)
                        cy = 0.5 * (zy0 + zy1)
                        side = max(abs(zx1 - zx0), abs(zy1 - zy0))
                        side = float(np.clip(side, 1e-4, min(chip_w, chip_h)))

                        x0 = cx - 0.5 * side
                        x1 = cx + 0.5 * side
                        y0 = cy - 0.5 * side
                        y1 = cy + 0.5 * side

                        # Keep square box inside chip frame while preserving side length.
                        if x0 < chip_x0:
                            dx = chip_x0 - x0
                            x0 += dx
                            x1 += dx
                        if x1 > chip_x1:
                            dx = x1 - chip_x1
                            x0 -= dx
                            x1 -= dx
                        if y0 < chip_y0:
                            dy = chip_y0 - y0
                            y0 += dy
                            y1 += dy
                        if y1 > chip_y1:
                            dy = y1 - chip_y1
                            y0 -= dy
                            y1 -= dy

                        x0 = float(np.clip(x0, chip_x0, chip_x1))
                        x1 = float(np.clip(x1, chip_x0, chip_x1))
                        y0 = float(np.clip(y0, chip_y0, chip_y1))
                        y1 = float(np.clip(y1, chip_y0, chip_y1))

                        ls = _shared_normalize_minimap_linestyle(minimap_inner_box_linestyle)
                        inner_lw = max(0.1, float(minimap_inner_box_linewidth))

                        mini.add_patch(
                            Rectangle(
                                (x0, y0),
                                max(1e-4, x1 - x0),
                                max(1e-4, y1 - y0),
                                fill=False,
                                edgecolor=str(minimap_outline_color),
                                linewidth=float(inner_lw),
                                linestyle=ls,
                                zorder=3,
                            )
                        )

                    mini.set_xlim(chip_x0, chip_x1)
                    mini.set_ylim(chip_y0, chip_y1)
                except Exception:
                    pass

            if bool(local_color_bars_show):
                try:
                    loc_cb = _shared_normalize_corner_location(local_color_bars_location, default="topright")
                    cb_len = min(0.95, max(0.06, float(local_color_bars_length_fraction)))
                    cb_pad = min(0.20, max(0.0, float(local_color_bars_pad_fraction)))
                    cb_w = 0.035
                    if loc_cb in {"topleft", "bottomleft"}:
                        cb_x = cb_pad
                    else:
                        cb_x = 1.0 - cb_w - cb_pad
                    if loc_cb in {"topleft", "topright"}:
                        cb_y = 1.0 - cb_len - cb_pad
                    else:
                        cb_y = cb_pad
                    cax_local = ax.inset_axes([float(cb_x), float(cb_y), float(cb_w), float(cb_len)])
                    local_sm = mcm.ScalarMappable(norm=norm, cmap=cmap_render)
                    local_sm.set_array([])
                    cb_local = fig.colorbar(local_sm, cax=cax_local)
                    ticks_local, labels_local = _shared_resolve_colorbar_ticks(
                        tick_spec=local_ticks_spec,
                        vmin=float(getattr(norm, "vmin", dyn_vmin)),
                        vmax=float(getattr(norm, "vmax", dyn_vmax)),
                        detected_amp_max=detected_amp_max,
                    )
                    cb_local.set_ticks(ticks_local)
                    if labels_local is not None:
                        cb_local.set_ticklabels(labels_local)
                    cb_local.ax.tick_params(
                        labelsize=max(4.0, float(local_color_bars_fontsize)),
                        colors=str(local_color_bars_fontcolor),
                    )
                    cb_local.outline.set_edgecolor(str(local_color_bars_fontcolor))
                except Exception:
                    pass

            if bool(effective_local_debug):
                try:
                    local_min = float(np.min(amp)) if amp.size else 0.0
                    local_max = float(np.max(amp)) if amp.size else 0.0
                    local_mean = float(np.mean(amp)) if amp.size else 0.0
                    local_median = float(np.median(amp)) if amp.size else 0.0
                    local_p90 = float(np.percentile(amp, 90.0)) if amp.size else 0.0
                    local_p99 = float(np.percentile(amp, 99.0)) if amp.size else 0.0
                    ax.text(
                        0.02,
                        0.98,
                        (
                            f"u{uid} local (µV)\n"
                            f"min={local_min:.2f}  max={local_max:.2f}\n"
                            f"mean={local_mean:.2f}  med={local_median:.2f}\n"
                            f"p90={local_p90:.2f}  p99={local_p99:.2f}"
                        ),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=5.5,
                        color="white",
                        zorder=10,
                        bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.8, "edgecolor": "none"},
                    )
                except Exception:
                    pass

        if bool(effective_global_debug):
            try:
                # Prefer the panel immediately after the last populated subplot so
                # debug stats are separated from unit visuals when there is space.
                if len(top) < len(axes_list):
                    reserved_debug_axis_index = int(len(top))
                    dbg_ax = axes_list[reserved_debug_axis_index]
                    dbg_ax.set_axis_on()
                    dbg_ax.set_facecolor("black")
                    dbg_ax.set_xticks([])
                    dbg_ax.set_yticks([])
                    for spine in dbg_ax.spines.values():
                        spine.set_visible(False)
                else:
                    dbg_ax = axes_list[min(len(top) - 1, len(axes_list) - 1)]

                amp_min_actual = float(np.min(all_amp_raw)) if all_amp_raw.size else 0.0
                amp_max_actual = float(np.max(all_amp_raw)) if all_amp_raw.size else 0.0
                amp_mean_actual = float(np.mean(all_amp_raw)) if all_amp_raw.size else 0.0
                amp_median_actual = float(np.median(all_amp_raw)) if all_amp_raw.size else 0.0
                p10 = float(np.percentile(all_amp_raw, 10.0)) if all_amp_raw.size else 0.0
                p25 = float(np.percentile(all_amp_raw, 25.0)) if all_amp_raw.size else 0.0
                p75 = float(np.percentile(all_amp_raw, 75.0)) if all_amp_raw.size else 0.0
                p90 = float(np.percentile(all_amp_raw, 90.0)) if all_amp_raw.size else 0.0
                p95 = float(np.percentile(all_amp_raw, 95.0)) if all_amp_raw.size else 0.0
                p99 = float(np.percentile(all_amp_raw, 99.0)) if all_amp_raw.size else 0.0
                dbg_ax.text(
                    0.02,
                    0.98,
                    (
                        "dbg global amps (µV)\n"
                        f"n={int(all_amp_raw.size)}\n"
                        f"min={amp_min_actual:.2f}  max={amp_max_actual:.2f}\n"
                        f"mean={amp_mean_actual:.2f}  median={amp_median_actual:.2f}\n"
                        f"p10={p10:.2f}  p25={p25:.2f}\n"
                        f"p75={p75:.2f}  p90={p90:.2f}\n"
                        f"p95={p95:.2f}  p99={p99:.2f}\n"
                        f"cap_raw(p99)={dyn_vmax_raw:.2f}  vmax_used={dyn_vmax:.2f}"
                    ),
                    transform=dbg_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                    color="white",
                    zorder=10,
                    bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
                )
            except Exception:
                pass

        if bool(legend_show):
            try:
                leg_marker = max(1.0, float(legend_marker_size))
                leg_font = max(4.0, float(legend_fontsize))
                leg_handles: list[Any] = []
                if bool(legend_show_footprint_in_legend):
                    leg_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="s",
                            linestyle="None",
                            markerfacecolor="white",
                            markeredgecolor="none",
                            markersize=leg_marker * 1.6,
                            label="Footprint",
                        )
                    )
                leg_handles.extend(
                    [
                        Line2D([0], [0], color=str(branch_color), lw=1.2, label="Branch"),
                        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=str(soma_node_color), markeredgecolor=str(branch_outline_color), markeredgewidth=0.5, markersize=leg_marker * 1.4, label="Soma node"),
                    ]
                )
                if bool(legend_show_nodes_in_legend):
                    leg_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="None",
                            markerfacecolor=str(branch_color),
                            markeredgecolor=str(branch_outline_color),
                            markeredgewidth=0.5,
                            markersize=leg_marker,
                            label="Branch node",
                        )
                    )

                loc_raw = str(legend_location or "first_empty_panel").strip().lower()
                if loc_raw == "first_empty_panel":
                    reserved_idxs: set[int] = set()
                    if reserved_debug_axis_index is not None:
                        reserved_idxs.add(int(reserved_debug_axis_index))
                    leg_idx = None
                    for j in range(len(top), len(axes_list)):
                        if int(j) in reserved_idxs:
                            continue
                        leg_idx = int(j)
                        break
                    if leg_idx is not None:
                        lax = axes_list[leg_idx]
                        lax.set_axis_on()
                        lax.set_facecolor("black")
                        lax.set_xticks([])
                        lax.set_yticks([])
                        for spine in lax.spines.values():
                            spine.set_visible(False)
                        lg = lax.legend(
                            handles=leg_handles,
                            loc="center",
                            frameon=False,
                            fontsize=leg_font,
                            handlelength=1.4,
                            handletextpad=0.6,
                        )
                        for txt in lg.get_texts():
                            txt.set_color(str(legend_fontcolor))
                    elif axes_list:
                        lg = axes_list[0].legend(handles=leg_handles, loc="upper right", frameon=False, fontsize=leg_font)
                        for txt in lg.get_texts():
                            txt.set_color(str(legend_fontcolor))
                else:
                    if axes_list:
                        lg = axes_list[0].legend(handles=leg_handles, loc=loc_raw, frameon=False, fontsize=leg_font)
                        for txt in lg.get_texts():
                            txt.set_color(str(legend_fontcolor))
            except Exception:
                pass

        for j in range(len(top), len(axes_list)):
            if reserved_debug_axis_index is not None and int(j) == int(reserved_debug_axis_index):
                continue
            axes_list[j].set_axis_off()

        if bool(global_color_bar_show):
            try:
                cax = fig.add_axes(
                    _shared_colorbar_axes_bounds(
                        location=global_color_bar_location,
                        length_fraction=float(global_color_bar_length_fraction),
                        pad_fraction=float(global_color_bar_pad_fraction),
                    )
                )
                cb = fig.colorbar(color_mappable, cax=cax)
                cb.set_label(f"{colorbar_label} [µV]", fontsize=max(4.0, float(global_color_bar_fontsize)), color=str(global_color_bar_fontcolor))

                tick_vals, tick_labels = _shared_resolve_colorbar_ticks(
                    tick_spec=global_ticks_spec,
                    vmin=float(getattr(norm, "vmin", dyn_vmin)),
                    vmax=float(getattr(norm, "vmax", dyn_vmax)),
                    detected_amp_max=detected_amp_max,
                )
                cb.set_ticks(tick_vals)
                if tick_labels is not None:
                    cb.set_ticklabels(tick_labels)
                elif bool(use_log_tick_format):
                    cb.ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
                    cb.ax.yaxis.set_minor_locator(mticker.NullLocator())

                cb.ax.tick_params(labelsize=max(4.0, float(global_color_bar_fontsize)), colors=str(global_color_bar_fontcolor))
                cb.outline.set_edgecolor(str(global_color_bar_fontcolor))
            except Exception:
                pass

        try:
            right_margin = 0.90 if bool(global_color_bar_show) else 0.98
            fig.subplots_adjust(left=0.02, right=float(right_margin), bottom=0.03, top=0.97, wspace=0.06, hspace=0.25)
        except Exception:
            pass

        if bool(write_pdf):
            fig.savefig(pdf_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
        if bool(write_png):
            fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    _render_grid(
        norm=log_norm,
        cmap_render=cmap_log,
        colorbar_label="PTP (log)",
        pdf_path=log_pdf_path,
        png_path=log_png_path,
        use_log_norm=True,
        use_log_tick_format=True,
    )
    _render_grid(
        norm=dynamic_piecewise_norm,
        cmap_render=cmap_linear,
        colorbar_label="PTP (dynamic piecewise)",
        pdf_path=linear_pdf_path,
        png_path=linear_png_path,
        use_log_norm=False,
        use_log_tick_format=False,
    )

    ranking_json = out_dir / str(ranking_filename)
    ranking_payload = {
        "metric": ranking_metric,
        "sort_by": ranking_sort_by,
        "top_n_requested": (int(top_n) if top_n is not None else "all"),
        "top_n_written": int(len(top)),
        "units": [
            {
                "unit_id": r.get("unit_id"),
                "density": float(r.get("density", 0.0)),
            "total_branch_length_um": float(r.get("total_branch_length_um", 0.0)),
                "n_waveforms_sum": int(r.get("n_waveforms_sum", 0)),
                "n_channels": int(r.get("n_channels", 0)),
                "area_um2": float(r.get("area_um2", 0.0)),
                "branches_raw_json": r.get("branches_raw_json"),
            }
            for r in top
        ],
    }
    if bool(write_ranking_json):
        try:
            from ..shared_io import write_json

            write_json(ranking_json, ranking_payload)
        except Exception:
            pass

    return {
        "raw_branch_log_footprint_top_density_grid_pdf": (str(log_pdf_path) if bool(write_pdf) else None),
        "raw_branch_log_footprint_top_density_grid_png": (str(log_png_path) if bool(write_png) else None),
        "raw_branch_linear_footprint_top_density_grid_pdf": (str(linear_pdf_path) if bool(write_pdf) else None),
        "raw_branch_linear_footprint_top_density_grid_png": (str(linear_png_path) if bool(write_png) else None),
        "raw_branch_log_footprint_top_density_ranking_json": (str(ranking_json) if bool(write_ranking_json) else None),
        "raw_branch_log_footprint_top_density_units": [r.get("unit_id") for r in top],
    }


