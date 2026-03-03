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


