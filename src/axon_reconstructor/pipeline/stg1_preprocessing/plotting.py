from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def _estimate_pitch_um(x, y) -> float:
    """Estimate electrode pitch (in the same units as x/y) from nearest neighbors."""

    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pitch estimation requires numpy") from e

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0

    # Pairwise distance to nearest neighbor (O(n^2), OK for typical electrode counts).
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 0.0
    return float(np.median(nn))


def detect_electrode_clusters(
    *,
    x,
    y,
    eps: Optional[float] = None,
    max_cluster_size_warn: int = 9,
) -> list[list[int]]:
    """Cluster electrodes by proximity using a simple radius graph + connected components.

    This avoids adding a dependency on scikit-learn. It is intended for the
    electrode layouts here (rectangular grids with local adjacency).

    Returns a list of clusters, where each cluster is a list of indices into x/y.
    """

    import warnings

    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("clustering requires numpy") from e

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    if n == 0:
        return []

    if eps is None:
        pitch = _estimate_pitch_um(x, y)
        # If pitch can't be estimated (degenerate), fall back to a generous radius.
        eps = pitch * 1.6 if pitch > 0 else 50.0

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx * dx + dy * dy
    adj = dist2 <= float(eps) ** 2
    np.fill_diagonal(adj, False)

    visited = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        while stack:
            j = stack.pop()
            comp.append(j)
            nbrs = np.where(adj[j])[0]
            for k in nbrs:
                if not visited[k]:
                    visited[k] = True
                    stack.append(int(k))
        clusters.append(sorted(comp))

    # Largest clusters first (useful for quick inspection).
    clusters.sort(key=len, reverse=True)
    too_big = [c for c in clusters if len(c) > max_cluster_size_warn]
    if too_big:
        warnings.warn(
            f"Found {len(too_big)} clusters larger than {max_cluster_size_warn} electrodes; "
            f"largest={max(len(c) for c in clusters)}. "
            "This may mean eps is too large or the layout is not well-separated.",
            stacklevel=2,
        )

    return clusters


def _pick_representative_index_for_cluster(
    *,
    x,
    y,
    cluster: list[int],
) -> int:
    """Pick a representative index (closest to centroid) for a cluster."""

    import numpy as np

    idx = np.asarray(cluster, dtype=int)
    cx = float(np.mean(np.asarray(x)[idx]))
    cy = float(np.mean(np.asarray(y)[idx]))
    dx = np.asarray(x)[idx] - cx
    dy = np.asarray(y)[idx] - cy
    j = int(idx[int(np.argmin(dx * dx + dy * dy))])
    return j


def _activity_score_rms(
    *,
    recording,
    channel_id,
    num_chunks: int = 6,
    chunk_size: int = 20_000,
    seed: int = 0,
) -> float:
    """Rough activity score: RMS from a few sampled windows.

    This is intentionally lightweight: it reads only `num_chunks * chunk_size`
    samples (or less), rather than scanning the full recording.
    """

    import numpy as np

    total = int(recording.get_num_samples())
    if total <= 0:
        return 0.0

    chunk_size = max(100, int(chunk_size))
    num_chunks = max(1, int(num_chunks))

    # If the recording is shorter than a chunk, just read what we can.
    if total <= chunk_size:
        traces = recording.get_traces(start_frame=0, end_frame=total, channel_ids=[channel_id]).astype(float)
        x = traces[:, 0]
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x)))

    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, total - chunk_size, size=num_chunks, endpoint=False)

    ssq = 0.0
    count = 0
    for start in starts:
        start_i = int(start)
        end_i = start_i + chunk_size
        traces = recording.get_traces(start_frame=start_i, end_frame=end_i, channel_ids=[channel_id]).astype(float)
        x = traces[:, 0]
        ssq += float(np.sum(x * x))
        count += int(x.size)

    return float(np.sqrt(ssq / max(count, 1)))


def _plot_concat_cluster_traces(
    *,
    recording,
    channel_ids: list[int],
    stitch_frames: list[int],
    out_path: Path,
    title: Optional[str] = None,
    target_hz: Optional[float] = None,
    max_points: int = 150_000,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot selected channel traces over time with red stitch markers."""

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = int(recording.get_num_samples())
    if total <= 0:
        return

    fs = float(recording.get_sampling_frequency())
    has_tv = False
    try:
        has_tv = bool(recording.has_time_vector())
    except Exception:
        has_tv = False

    # Decimate to keep plotting lightweight and allow explicit target plotting rate.
    # max_points <= 0 means uncapped for point-based decimation.
    try:
        parsed_max_points = int(max_points)
    except Exception:
        parsed_max_points = 150_000
    if parsed_max_points <= 0:
        step_by_points = 1
    else:
        points_cap = max(1000, parsed_max_points)
        step_by_points = max(1, total // points_cap)

    step_by_rate = 1
    try:
        if target_hz is not None and float(target_hz) > 0.0 and fs > 0.0:
            step_by_rate = max(1, int(round(fs / float(target_hz))))
    except Exception:
        step_by_rate = 1

    step = max(step_by_points, step_by_rate)
    sel_frames = np.arange(0, total, step, dtype=np.int64)
    expected_points = int(sel_frames.size)
    effective_hz = (float(fs) / float(step)) if step > 0 else float(fs)
    if logger is not None:
        logger.info(
            "plot traces: downsample fs=%.2fHz target_hz=%s step=%d effective_hz=%.2f expected_points_per_channel=%d channels=%d out=%s",
            float(fs),
            (f"{float(target_hz):.2f}" if target_hz is not None else "none"),
            int(step),
            float(effective_hz),
            int(expected_points),
            int(len(channel_ids)),
            out_path,
        )
        if expected_points > 200_000:
            logger.warning(
                "plot traces: high point count after downsampling (%d points/channel); consider lowering trace_downsample_hz or setting trace_max_points",
                int(expected_points),
            )

    if has_tv:
        # Use the recording-provided time vector so gaps (e.g. triggered snippets)
        # appear correctly on the x-axis.
        try:
            t = recording.sample_index_to_time(sel_frames)
        except Exception:
            t = (sel_frames.astype(float) / fs)
    else:
        t = (sel_frames.astype(float) / fs)

    fig, axes = plt.subplots(len(channel_ids), 1, figsize=(13.33, 7.5), dpi=180, sharex=True)
    if len(channel_ids) == 1:
        axes = [axes]

    # Read all requested channels once per block (instead of one extractor call per
    # channel per block) so downsampled plotting work scales better.
    block = 200_000
    total_blocks = max(1, int((total + block - 1) // block))
    y_parts_per_channel: list[list[np.ndarray]] = [[] for _ in channel_ids]
    for block_idx, start in enumerate(range(0, total, block), start=1):
        end = min(total, start + block)
        traces_block = recording.get_traces(start_frame=start, end_frame=end, channel_ids=channel_ids)
        offset = (-start) % step
        traces_ds = traces_block[offset::step, :]
        for ch_idx in range(len(channel_ids)):
            y_parts_per_channel[ch_idx].append(np.asarray(traces_ds[:, ch_idx]))

        if logger is not None and (
            block_idx == 1
            or block_idx == total_blocks
            or block_idx % max(1, total_blocks // 10) == 0
        ):
            logger.info(
                "plot traces: load progress %d/%d blocks (%.1f%%) out=%s",
                int(block_idx),
                int(total_blocks),
                float((100.0 * block_idx) / max(1, total_blocks)),
                out_path,
            )

    for ax, ch, y_parts in zip(axes, channel_ids, y_parts_per_channel, strict=False):
        y = np.concatenate(y_parts).astype(float, copy=False) if y_parts else np.asarray([], dtype=float)

        t_plot = t[: y.size]
        y_plot = y
        # If we have an explicit time vector (e.g. triggered snippets), large gaps
        # can create misleading diagonal line segments. Break the line at big jumps.
        if has_tv and y_plot.size > 2:
            try:
                dt = np.diff(t_plot.astype(float))
                baseline = float(step) / float(fs)
                jump_idx = np.where(dt > (5.0 * max(baseline, 1e-9)))[0]
                if jump_idx.size:
                    y_plot = y_plot.astype(float, copy=True)
                    y_plot[jump_idx + 1] = np.nan
            except Exception:
                pass

        ax.plot(t_plot, y_plot, lw=0.2, color="black")
        for sf in stitch_frames:
            if has_tv:
                try:
                    xline = float(recording.sample_index_to_time(int(sf)))
                except Exception:
                    xline = float(sf) / fs
            else:
                xline = float(sf) / fs
            ax.axvline(xline, color="red", lw=0.6, alpha=0.8)
        ax.set_ylabel(f"ch {ch}")
        ax.grid(False)

    axes[-1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    if logger is not None:
        logger.info("plot traces: wrote %s", out_path)


def _plot_stitch_zoom(
    *,
    recording,
    channel_ids: list[int],
    stitch_frame: int,
    out_path: Path,
    window_s: float = 2.0,
    title: Optional[str] = None,
) -> None:
    """Zoomed plot around a stitch boundary.

    Plots a short window centered on the stitch point.
    Time is shown relative to the stitch point (0 s).
    """

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = int(recording.get_num_samples())
    fs = float(recording.get_sampling_frequency())
    if total <= 0 or fs <= 0:
        return

    has_tv = False
    try:
        has_tv = bool(recording.has_time_vector())
    except Exception:
        has_tv = False

    stitch_frame = int(stitch_frame)
    window_frames = max(1, int(round(float(window_s) * fs)))
    start = max(0, stitch_frame - window_frames)
    end = min(total, stitch_frame + window_frames)
    if end <= start:
        return

    traces = recording.get_traces(start_frame=start, end_frame=end, channel_ids=channel_ids).astype(float)
    n = int(traces.shape[0])
    if n <= 0:
        return

    # Decimate only if the zoom window is still huge.
    max_points = 300_000
    step = max(1, n // max_points)
    frames = np.arange(start, end, step, dtype=np.int64)
    if has_tv:
        try:
            t0 = float(recording.sample_index_to_time(int(stitch_frame)))
            t = recording.sample_index_to_time(frames) - t0
        except Exception:
            t = (frames.astype(float) - float(stitch_frame)) / fs
    else:
        t = (frames.astype(float) - float(stitch_frame)) / fs

    fig, axes = plt.subplots(len(channel_ids), 1, figsize=(13.33, 7.5), dpi=220, sharex=True)
    if len(channel_ids) == 1:
        axes = [axes]

    for i, (ax, ch) in enumerate(zip(axes, channel_ids, strict=False)):
        y = traces[::step, i]
        ax.plot(t[: y.size], y, lw=0.2, color="black")
        ax.axvline(0.0, color="red", lw=0.6, alpha=0.9)
        ax.set_ylabel(f"ch {ch}")
        ax.grid(False)

    axes[-1].set_xlabel("time relative to stitch (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _extract_xy_from_contact_vector(contact_vector):
    """Extract (x, y) arrays from a SpikeInterface contact_vector.

    Maxwell/SpikeInterface contact vectors are typically structured arrays with
    an `electrode` field and some flavor of x/y coordinate fields.
    """

    names = set(getattr(getattr(contact_vector, "dtype", None), "names", ()) or ())
    candidates = [
        ("x", "y"),
        ("xpos", "ypos"),
        ("x_um", "y_um"),
        ("xpos_um", "ypos_um"),
    ]
    for x_name, y_name in candidates:
        if x_name in names and y_name in names:
            return contact_vector[x_name], contact_vector[y_name]

    raise RuntimeError(
        f"contact_vector is missing x/y fields; found fields={sorted(names)}"
    )


def _save_channel_layout_plots(
    *,
    h5_path: Path,
    stream_id: str,
    rec_names: list[str],
    common_electrodes: list[int],
    out_dir: Path,
) -> None:
    """Save per-segment layouts + shared layout + a combined slide-ready figure."""

    try:
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        import spikeinterface.extractors as se
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "plotting requires `matplotlib`, `numpy`, and `spikeinterface` installed"
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_set = set(int(x) for x in common_electrodes)
    layouts: list[dict] = []

    shared_color = "#D62728"
    other_color = "#B0B0B0"

    # First pass: load positions once per segment.
    for rec_name in rec_names:
        # SpikeInterface >= 0.103.x: prefer the extractors function API.
        if hasattr(se, "read_maxwell"):
            rec = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
        else:  # pragma: no cover
            rec = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)
        cv = rec.get_property("contact_vector")
        electrodes = np.asarray(cv["electrode"], dtype=int)
        x, y = _extract_xy_from_contact_vector(cv)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        is_shared = np.asarray([int(el) in common_set for el in electrodes], dtype=bool)
        layouts.append(
            {
                "rec_name": rec_name,
                "electrodes": electrodes,
                "x": x,
                "y": y,
                "is_shared": is_shared,
            }
        )

    # Compute global extents so all plots share a consistent frame (rectangular look).
    all_x = np.concatenate([d["x"] for d in layouts]) if layouts else np.asarray([])
    all_y = np.concatenate([d["y"] for d in layouts]) if layouts else np.asarray([])
    if all_x.size and all_y.size:
        pad_x = max(1.0, 0.03 * float(all_x.max() - all_x.min()))
        pad_y = max(1.0, 0.03 * float(all_y.max() - all_y.min()))
        xlim = (float(all_x.min() - pad_x), float(all_x.max() + pad_x))
        ylim = (float(all_y.min() - pad_y), float(all_y.max() + pad_y))
    else:
        xlim = None
        ylim = None

    def plot_one(ax, d: dict, *, title: str, other_s: float = 6, shared_s: float = 10) -> None:
        ax.scatter(d["x"][~d["is_shared"]], d["y"][~d["is_shared"]], s=other_s, c=other_color, linewidths=0)
        ax.scatter(d["x"][d["is_shared"]], d["y"][d["is_shared"]], s=shared_s, c=shared_color, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(False)

    # 1) Per-segment layouts (shared highlighted)
    for d in layouts:
        fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=160)
        plot_one(ax, d, title=f"{stream_id} / {d['rec_name']} (shared highlighted)")
        fig.tight_layout()
        fig.savefig(out_dir / f"layout_{stream_id}_{d['rec_name']}.png")
        plt.close(fig)

    # 2) Shared-only layout
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    # Use first layout as coordinate source; shared electrodes are common across all.
    if layouts:
        d0 = layouts[0]
        ax.scatter(
            d0["x"][d0["is_shared"]],
            d0["y"][d0["is_shared"]],
            s=14,
            c=shared_color,
            linewidths=0,
        )
        ax.set_title(f"{stream_id} / shared electrodes (n={len(common_electrodes)})", fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_dir / f"layout_shared_{stream_id}.png")
    plt.close(fig)

    # 3) Combined slide figure: big shared plot + small per-segment plots
    n = len(layouts)
    # Prefer a consistent thumbnail grid width for slides.
    ncols = 2 if n else 1
    nrows = int(np.ceil(n / ncols)) if n else 1

    # Use a 16:9-ish canvas for slides.
    fig = plt.figure(figsize=(13.33, 7.5), dpi=180)
    outer = GridSpec(1, 2, width_ratios=[4.0, 1.0], wspace=0.05)

    ax_big = fig.add_subplot(outer[0])
    if layouts:
        d0 = layouts[0]
        ax_big.scatter(
            d0["x"][d0["is_shared"]],
            d0["y"][d0["is_shared"]],
            s=12,
            c=shared_color,
            linewidths=0,
        )
        ax_big.set_title(f"Shared electrodes (stream={stream_id}, n={len(common_electrodes)})", fontsize=14)
        ax_big.set_aspect("equal", adjustable="box")
        ax_big.set_xlabel("x")
        ax_big.set_ylabel("y")
        if xlim is not None:
            ax_big.set_xlim(*xlim)
        if ylim is not None:
            ax_big.set_ylim(*ylim)

    # Pack thumbnails tightly; no text, just borders.
    right = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer[1], hspace=0.05, wspace=0.05)
    for idx, d in enumerate(layouts):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(right[r, c])
        plot_one(ax, d, title="", other_s=1, shared_s=1)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    # Avoid tight_layout() here; it tends to re-introduce padding.
    # Leave enough room for the big plot y-label/title.
    fig.subplots_adjust(left=0.06, right=0.995, top=0.965, bottom=0.06)
    fig.savefig(out_dir / f"layout_combined_{stream_id}.png")
    plt.close(fig)
