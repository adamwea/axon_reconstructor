"""axon_velocity integrations for templates stage.

This module is intentionally isolated so the core templates stage can run without
requiring `axon_velocity` (and its heavier optional dependencies).

When enabled, it produces a bundle of pre-reconstruction diagnostic plots using
axon_velocity's own plotting + GraphAxonTracking helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class AxonVelocityPlotOutputs:
    out_dir: Path
    amplitude_map_png: Optional[Path]
    peak_latency_map_png: Optional[Path]
    peak_std_map_png: Optional[Path]
    channel_selection_png: Optional[Path]
    graph_png: Optional[Path]
    branches_png: Optional[Path]
    velocities_png: Optional[Path]
    axon_summary_png: Optional[Path]


def try_write_axon_velocity_plots_from_npz(*, npz_path: Path, out_dir: Path, unit_id: Any = None) -> AxonVelocityPlotOutputs:
    """Best-effort plot bundle from an `axon_velocity_inputs.npz` file.

    The NPZ is expected to contain:
      - template_ch_by_t: (n_channels, n_samples)
      - locations_xy: (n_channels, 2)
      - sampling_frequency_hz: float

    Returns paths (may be None if a sub-plot failed).
    """

    import numpy as np  # type: ignore[import-not-found]

    data = np.load(npz_path, allow_pickle=True)
    template = data.get("template_ch_by_t")
    locations = data.get("locations_xy")
    fs = data.get("sampling_frequency_hz")

    return try_write_axon_velocity_plots(
        template_ch_by_t=template,
        locations_xy=locations,
        sampling_frequency_hz=float(fs) if fs is not None else None,
        out_dir=out_dir,
        unit_id=unit_id,
    )


def try_write_axon_velocity_plots(
    *,
    template_ch_by_t: Any,
    locations_xy: Any,
    sampling_frequency_hz: Optional[float],
    out_dir: Path,
    unit_id: Any = None,
    compute_graph: bool = True,
    graph_kwargs: Optional[dict[str, Any]] = None,
) -> AxonVelocityPlotOutputs:
    """Generate axon_velocity plots (best-effort).

    Notes:
    - This uses axon_velocity's own plotting functions, which are based on the
      provided channel set and will *not* expand to the full 220x120 grid.
    - The templates pipeline also produces separate full-chip QC maps.
    """

    import numpy as np  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    amp_png = out_dir / "amplitude_map.png"
    lat_png = out_dir / "peak_latency_map.png"
    std_png = out_dir / "peak_std_map.png"
    sel_png = out_dir / "channel_selection.png"
    graph_png = out_dir / "graph.png"
    branches_png = out_dir / "branches.png"
    vel_png = out_dir / "velocities.png"
    summary_png = out_dir / "axon_summary.png"

    # Default outputs: None until successfully written
    outputs = {
        "amplitude_map_png": None,
        "peak_latency_map_png": None,
        "peak_std_map_png": None,
        "channel_selection_png": None,
        "graph_png": None,
        "branches_png": None,
        "velocities_png": None,
        "axon_summary_png": None,
    }

    tmpl = np.asarray(template_ch_by_t, dtype=float)
    locs = np.asarray(locations_xy, dtype=float)
    if tmpl.ndim != 2 or locs.ndim != 2 or locs.shape[1] < 2 or locs.shape[0] != tmpl.shape[0]:
        return AxonVelocityPlotOutputs(out_dir=out_dir, **outputs)  # type: ignore[arg-type]

    fs = float(sampling_frequency_hz) if sampling_frequency_hz is not None else None
    if fs is None or not (fs > 0):
        compute_graph = False

    try:
        import axon_velocity as av  # type: ignore[import-not-found]
        import axon_velocity.plotting as av_plot
    except Exception:
        return AxonVelocityPlotOutputs(out_dir=out_dir, **outputs)  # type: ignore[arg-type]

    title_suffix = f" (unit {unit_id})" if unit_id is not None else ""

    # Map plots.
    try:
        fig = plt.figure(figsize=(5.2, 4.3))
        ax = fig.add_subplot(111)
        av_plot.plot_amplitude_map(tmpl, locs[:, :2], ax=ax, cmap="viridis", log=True, plot_image=True, colorbar=True)
        ax.set_title("axon_velocity amplitude" + title_suffix)
        fig.tight_layout()
        fig.savefig(amp_png, dpi=200)
        plt.close(fig)
        outputs["amplitude_map_png"] = amp_png
    except Exception:
        pass

    try:
        fig = plt.figure(figsize=(5.2, 4.3))
        ax = fig.add_subplot(111)
        av_plot.plot_peak_latency_map(tmpl, locs[:, :2], float(fs) if fs else 1.0, ax=ax, cmap="viridis", log=False, plot_image=True, colorbar=True)
        ax.set_title("axon_velocity peak latency" + title_suffix)
        fig.tight_layout()
        fig.savefig(lat_png, dpi=200)
        plt.close(fig)
        outputs["peak_latency_map_png"] = lat_png
    except Exception:
        pass

    if fs is not None:
        try:
            fig = plt.figure(figsize=(5.2, 4.3))
            ax = fig.add_subplot(111)
            av_plot.plot_peak_std_map(tmpl, locs[:, :2], float(fs), ax=ax, cmap="viridis", plot_image=True)
            ax.set_title("axon_velocity peak std" + title_suffix)
            fig.tight_layout()
            fig.savefig(std_png, dpi=200)
            plt.close(fig)
            outputs["peak_std_map_png"] = std_png
        except Exception:
            pass

    gtr = None
    if bool(compute_graph) and fs is not None:
        try:
            kwargs = dict(graph_kwargs or {})
            gtr = av.compute_graph_propagation_velocity(template=tmpl, locations=locs[:, :2], fs=float(fs), **kwargs)
        except Exception:
            gtr = None

    if gtr is not None:
        try:
            fig = gtr.plot_channel_selection()
            fig.savefig(sel_png, dpi=200)
            plt.close(fig)
            outputs["channel_selection_png"] = sel_png
        except Exception:
            pass

        try:
            fig = gtr.plot_graph()
            fig.savefig(graph_png, dpi=200)
            plt.close(fig)
            outputs["graph_png"] = graph_png
        except Exception:
            pass

        try:
            fig = gtr.plot_branches()
            fig.savefig(branches_png, dpi=200)
            plt.close(fig)
            outputs["branches_png"] = branches_png
        except Exception:
            pass

        try:
            fig = gtr.plot_velocities()
            fig.savefig(vel_png, dpi=200)
            plt.close(fig)
            outputs["velocities_png"] = vel_png
        except Exception:
            pass

        try:
            fig, _axes = av_plot.plot_axon_summary(gtr)
            fig.savefig(summary_png, dpi=200)
            plt.close(fig)
            outputs["axon_summary_png"] = summary_png
        except Exception:
            pass

    return AxonVelocityPlotOutputs(out_dir=out_dir, **outputs)  # type: ignore[arg-type]


__all__ = [
    "AxonVelocityPlotOutputs",
    "try_write_axon_velocity_plots",
    "try_write_axon_velocity_plots_from_npz",
]
