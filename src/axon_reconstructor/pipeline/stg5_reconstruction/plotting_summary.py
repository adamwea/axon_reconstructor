"""Summary plotting helpers for reconstruction (internal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .plotting_core import (
    _force_white_background,
    _read_json,
    _strip_axes_titles,
    _white_bg_rc_params,
    _with_suffix,
)

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

    branches_json = Path(out_unit_dir) / "branches.json"
    heuristics_json = Path(out_unit_dir) / "heuristics.json"
    raw_json = Path(out_unit_dir) / "branches_raw.json"

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

    summary_clean_png = Path(out_unit_dir) / "summary_clean.png"
    summary_raw_png = Path(out_unit_dir) / "summary_raw.png"
    summary_png = Path(out_unit_dir) / "summary.png"

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
    show_scale_debug_text: bool = False,
    show_global_debug_text: bool = False,
    show_local_debug_text: bool = False,
    draw_zoom_range_box: bool = False,
    logger: Any,
) -> dict[str, Any]:
    """Write a top-N grid of raw-branch morphologies over log-zoom footprints.

    Ranking metric (higher is better):
        density = n_waveforms_sum / n_channels / footprint_area_um2

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
            continue

        candidates.append(
            {
                "unit_id": uid,
                "density": float(density),
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

    candidates = sorted(
        candidates,
        key=lambda r: (-float(r.get("density", 0.0)), -int(r.get("n_waveforms_sum", 0)), int(r.get("unit_id", 10**12))),
    )
    if top_n is None:
        top = list(candidates)
    else:
        top = candidates[: max(1, int(top_n))]

    # Rendering.
    out_dir = Path(reconstruction_out_dir) / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_pdf_path = out_dir / "raw_branch_log_footprint_top_density_grid.pdf"
    log_png_path = out_dir / "raw_branch_log_footprint_top_density_grid.png"
    linear_pdf_path = out_dir / "raw_branch_linear_footprint_top_density_grid.pdf"
    linear_png_path = out_dir / "raw_branch_linear_footprint_top_density_grid.png"

    ncols = 5
    nrows = max(1, int(np.ceil(float(len(top)) / float(ncols))))

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

    if all_amp_pos.size:
        log_vmin = float(np.percentile(all_amp_pos, 5))
        log_vmax = float(np.percentile(all_amp_pos, 99.5))
    else:
        log_vmin, log_vmax = 1e-3, 1.0
    if log_vmax <= log_vmin:
        log_vmax = log_vmin * 1.01
    log_norm = mcolors.LogNorm(vmin=max(1e-12, float(log_vmin)), vmax=max(float(log_vmax), float(log_vmin) * 1.01))

    # Dynamic piecewise anchored scaling for the comparison panel,
    # with anchors at [vmin, 1, 10, vmax] and knot positions driven by data percentiles.
    detected_amp_max = float(np.max(all_amp_raw)) if all_amp_raw.size else 1.0
    dyn_vmax_raw = float('nan')
    if all_amp_pos.size:
        dyn_vmax_raw = float(np.percentile(all_amp_pos, 99.0))
        dyn_vmin = 0.0
        if dyn_vmax_raw <= dyn_vmin * 1.01:
            dyn_vmax_raw = dyn_vmin * 1.01
        dyn_vmax = float(max(11.0, _round_up_10(dyn_vmax_raw)))
    else:
        dyn_vmin = 0.0
        dyn_vmax = 100.0
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

    y1 = float(np.clip(_pct_rank(1.0), 0.02, 0.90))
    y2 = float(np.clip(_pct_rank(10.0), y1 + 0.05, 0.98))

    x_knots = np.asarray([float(dyn_vmin), 1.0, 10.0, float(dyn_vmax)], dtype=float)
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
        "black_blue_white_orange_red",
        ["#000000", "#1f4fff", "#ffffff", "#ff8c00", "#ff0000"],
        N=256,
    )

    cmap_linear = LinearSegmentedColormap.from_list(
        "black_blue_white_orange_red_linear_dynamic",
        [
            (0.0, "#000000"),
            (blue_pos, "#1f4fff"),
            (white_pos, "#ffffff"),
            (orange_pos, "#ff8c00"),
            (1.0, "#ff0000"),
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

    from matplotlib.patches import Rectangle

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
            ax.set_facecolor("black")
            branch_points_xy: list[Any] = []

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
            try:
                raw_payload = _read_json(Path(str(row["branches_raw_json"])))
                raw_branches = list((raw_payload or {}).get("branches", []) or [])

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

                for bi, br in enumerate(raw_branches):
                    chs = [int(c) for c in list(br.get("channels", []) or []) if isinstance(c, (int, float))]
                    if len(chs) < 2:
                        continue
                    chs = _map_branch_indices(chs)
                    if len(chs) < 2:
                        continue
                    pts = locs_xy[np.asarray(chs, dtype=int), :2]
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        color="red",
                        lw=0.9,
                        alpha=0.95,
                        zorder=3,
                    )
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        color="red",
                        s=8,
                        linewidths=0,
                        zorder=4,
                    )
                    branch_points_xy.append(np.asarray(pts, dtype=float))
                    n_branches_plotted += 1
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

                if branch_points_xy:
                    all_branch_pts = np.vstack(branch_points_xy)
                    pts_for_zoom = all_branch_pts
                    if high_amp_pts is not None and int(high_amp_pts.shape[0]) > 0:
                        pts_for_zoom = np.vstack([pts_for_zoom, high_amp_pts])

                    xmin, xmax = float(np.min(pts_for_zoom[:, 0])), float(np.max(pts_for_zoom[:, 0]))
                    ymin, ymax = float(np.min(pts_for_zoom[:, 1])), float(np.max(pts_for_zoom[:, 1]))
                    span_x = max(1e-6, (xmax - xmin))
                    span_y = max(1e-6, (ymax - ymin))
                    span = max(span_x, span_y)
                    # Generous branch-centric padding so traces are readable but not cramped.
                    pad = max(45.0, 0.40 * span)
                else:
                    if high_amp_pts is not None and int(high_amp_pts.shape[0]) > 0:
                        xmin, xmax = float(np.min(high_amp_pts[:, 0])), float(np.max(high_amp_pts[:, 0]))
                        ymin, ymax = float(np.min(high_amp_pts[:, 1])), float(np.max(high_amp_pts[:, 1]))
                        span_x = max(1e-6, (xmax - xmin))
                        span_y = max(1e-6, (ymax - ymin))
                        span = max(span_x, span_y)
                        pad = max(45.0, 0.40 * span)
                    else:
                        xmin, xmax = float(np.min(locs_xy[:, 0])), float(np.max(locs_xy[:, 0]))
                        ymin, ymax = float(np.min(locs_xy[:, 1])), float(np.max(locs_xy[:, 1]))
                        pad = 25.0

                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                span_x = max(1e-6, (xmax - xmin))
                span_y = max(1e-6, (ymax - ymin))
                half_span = 0.5 * max(span_x, span_y) + float(pad)
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
                f"u{uid}  d={float(row['density']):.2e}  br={int(n_branches_plotted)}\nWf={int(row['n_waveforms_sum'])} Ch={int(row['n_channels'])}",
                fontsize=7,
                pad=1.5,
                color="white",
            )
            if None not in (zoom_x0, zoom_x1, zoom_y0, zoom_y1):
                _add_scalebar(
                    ax,
                    zoom_x0=float(zoom_x0),
                    zoom_x1=float(zoom_x1),
                    zoom_y0=float(zoom_y0),
                    zoom_y1=float(zoom_y1),
                )

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

        for j in range(len(top), len(axes_list)):
            if reserved_debug_axis_index is not None and int(j) == int(reserved_debug_axis_index):
                continue
            axes_list[j].set_axis_off()

        try:
            cax = fig.add_axes([0.92, 0.16, 0.015, 0.72])
            cb = fig.colorbar(color_mappable, cax=cax)
            cb.set_label(f"{colorbar_label} [µV]", fontsize=8, color="black")
            if bool(use_log_tick_format):
                cb.ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
                cb.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
                cb.ax.yaxis.set_minor_locator(mticker.NullLocator())
                cb.update_ticks()
            else:
                # Dynamic-log panel: show linear-value ticks (1, 10, and top).
                # If detected max exceeds displayed vmax, show top label as ">vmax".
                top_tick = float(dyn_vmax)
                tick_vals = [1.0, 10.0, top_tick]
                tick_vals.append(top_tick)
                dedup_sorted_ticks: list[float] = []
                for tv in sorted({float(t) for t in tick_vals if float(dyn_vmin) <= float(t) <= float(dyn_vmax)}):
                    if (not dedup_sorted_ticks) or abs(tv - dedup_sorted_ticks[-1]) > 1e-9:
                        dedup_sorted_ticks.append(tv)
                if not dedup_sorted_ticks:
                    dedup_sorted_ticks = [float(dyn_vmax)]
                cb.set_ticks(dedup_sorted_ticks)
                cb.update_ticks()
                if detected_amp_max > float(dyn_vmax) + 1e-9:
                    tick_labels: list[str] = []
                    for tv in dedup_sorted_ticks:
                        if abs(tv - top_tick) <= 1e-9:
                            tick_labels.append(f">{_fmt_tick_plain(top_tick)}")
                        else:
                            tick_labels.append(_fmt_tick_plain(tv))
                    cb.set_ticklabels(tick_labels)
                else:
                    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: _fmt_tick_plain(float(x))))
            cb.ax.tick_params(labelsize=7, colors="black")
            cb.outline.set_edgecolor("black")
        except Exception:
            pass

        try:
            fig.subplots_adjust(left=0.02, right=0.90, bottom=0.03, top=0.97, wspace=0.06, hspace=0.25)
        except Exception:
            pass

        fig.savefig(pdf_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
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

    ranking_json = out_dir / "raw_branch_log_footprint_top_density_ranking.json"
    ranking_payload = {
        "metric": "n_waveforms_sum / n_channels / area_um2",
        "top_n_requested": (int(top_n) if top_n is not None else "all"),
        "top_n_written": int(len(top)),
        "units": [
            {
                "unit_id": r.get("unit_id"),
                "density": float(r.get("density", 0.0)),
                "n_waveforms_sum": int(r.get("n_waveforms_sum", 0)),
                "n_channels": int(r.get("n_channels", 0)),
                "area_um2": float(r.get("area_um2", 0.0)),
                "branches_raw_json": r.get("branches_raw_json"),
            }
            for r in top
        ],
    }
    try:
        from ..shared_io import write_json

        write_json(ranking_json, ranking_payload)
    except Exception:
        pass

    return {
        "raw_branch_log_footprint_top_density_grid_pdf": str(log_pdf_path),
        "raw_branch_log_footprint_top_density_grid_png": str(log_png_path),
        "raw_branch_linear_footprint_top_density_grid_pdf": str(linear_pdf_path),
        "raw_branch_linear_footprint_top_density_grid_png": str(linear_png_path),
        "raw_branch_log_footprint_top_density_ranking_json": str(ranking_json),
        "raw_branch_log_footprint_top_density_units": [r.get("unit_id") for r in top],
    }


