from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Optional


def print_time_between_segments(
    recording_or_list,
    rec_names: Optional[list[str]] = None,
    *,
    prefix: str = "[axon_reconstructor]",
    tol_s: float = 1e-6,
) -> Optional[list[float]]:
    """Print estimated wall-clock gaps between consecutive segments.

    This relies on SpikeInterface timing metadata:
    - `get_start_time()` / `get_end_time()`
    - optional time vectors (`has_time_vector()`)

    The function only prints *inter-segment gaps* if segment start times appear
    to share a common absolute time base across segments. If start times look
    relative (common case: all segments start at 0), it prints a short message
    and returns None.

    Parameters
    ----------
    recording_or_list:
        Either a list of mono-segment recordings, or a multi-segment recording.
    rec_names:
        Optional human-readable names for segments (same length as segments).
    prefix:
        Prefix used for all printed lines.
    tol_s:
        Tolerance (seconds) used for monotonicity checks.

    Returns
    -------
    gaps_s:
        List of gaps in seconds (length = num_segments - 1), or None if gaps
        cannot be computed from available timing metadata.
    """

    def _to_float_or_none(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    # Normalize input to a list of segments and an accessor that doesn't require
    # callers to worry about segment_index.
    if isinstance(recording_or_list, (list, tuple)):
        segments = list(recording_or_list)

        def get_start(seg, seg_i: int):
            return _to_float_or_none(seg.get_start_time())

        def get_end(seg, seg_i: int):
            return _to_float_or_none(seg.get_end_time())

        def has_tv(seg, seg_i: int):
            try:
                return bool(seg.has_time_vector())
            except Exception:
                return False
    else:
        rec = recording_or_list
        try:
            nseg = int(rec.get_num_segments())
        except Exception:
            nseg = 1
        segments = [rec] * nseg

        def get_start(seg, seg_i: int):
            return _to_float_or_none(rec.get_start_time(segment_index=seg_i))

        def get_end(seg, seg_i: int):
            return _to_float_or_none(rec.get_end_time(segment_index=seg_i))

        def has_tv(seg, seg_i: int):
            try:
                return bool(rec.has_time_vector(segment_index=seg_i))
            except Exception:
                return False

    if rec_names is None:
        rec_names = [f"segment_{i}" for i in range(len(segments))]
    else:
        rec_names = list(rec_names)

    if len(rec_names) != len(segments):
        rec_names = [f"segment_{i}" for i in range(len(segments))]

    if len(segments) <= 1:
        print(f"{prefix} inter-segment gaps: single segment", flush=True)
        return []

    try:
        starts: list[float] = []
        ends: list[float] = []
        has_time_vector: list[bool] = []

        for i, seg in enumerate(segments):
            t0 = get_start(seg, i)
            t1 = get_end(seg, i)
            if t0 is None or t1 is None:
                print(
                    f"{prefix} inter-segment gaps unavailable: missing start/end time metadata",
                    flush=True,
                )
                return None
            starts.append(t0)
            ends.append(t1)
            has_time_vector.append(has_tv(seg, i))

        print(f"{prefix} segment timing metadata:", flush=True)
        for name, t0, t1, tv in zip(rec_names, starts, ends, has_time_vector, strict=False):
            print(f"{prefix}   {name}: start={t0:.6f} s, end={t1:.6f} s, has_time_vector={tv}", flush=True)

        # Heuristic: gaps only make sense if starts vary and are nondecreasing.
        starts_min = min(starts)
        starts_max = max(starts)
        has_variation = (starts_max - starts_min) > float(tol_s)

        nondecreasing = True
        for a, b in zip(starts, starts[1:], strict=False):
            if (b - a) < -float(tol_s):
                nondecreasing = False
                break

        if not (has_variation and nondecreasing):
            print(
                f"{prefix} inter-segment gaps unavailable: segment start times do not look like a shared absolute time base",
                flush=True,
            )
            return None

        gaps_s: list[float] = []
        print(f"{prefix} inter-segment gaps (start_next - end_prev):", flush=True)
        for i in range(1, len(starts)):
            gap = float(starts[i] - ends[i - 1])
            gaps_s.append(gap)
            print(f"{prefix}   {rec_names[i - 1]} -> {rec_names[i]}: gap={gap:.6f} s", flush=True)

        return gaps_s
    except Exception:
        print(f"{prefix} inter-segment gaps unavailable (timing metadata error)", flush=True)
        return None


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
) -> None:
    """Plot selected channel traces over time with red stitch markers."""

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = float(recording.get_sampling_frequency())
    total = int(recording.get_num_samples())
    if total <= 0:
        return

    # Decimate to keep plotting lightweight.
    max_points = 150_000
    step = max(1, total // max_points)
    t = (np.arange(0, total, step, dtype=float) / fs)

    fig, axes = plt.subplots(len(channel_ids), 1, figsize=(13.33, 7.5), dpi=180, sharex=True)
    if len(channel_ids) == 1:
        axes = [axes]

    for ax, ch in zip(axes, channel_ids, strict=False):
        y_parts: list[np.ndarray] = []
        block = 200_000
        for start in range(0, total, block):
            end = min(total, start + block)
            traces = recording.get_traces(start_frame=start, end_frame=end, channel_ids=[ch]).astype(float)
            x = traces[:, 0]
            offset = (-start) % step
            y_parts.append(x[offset::step])
        y = np.concatenate(y_parts) if y_parts else np.asarray([])

        ax.plot(t[: y.size], y, lw=0.2, color="black")
        for sf in stitch_frames:
            ax.axvline(sf / fs, color="red", lw=0.6, alpha=0.8)
        ax.set_ylabel(f"ch {ch}")
        ax.grid(False)

    axes[-1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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

    fs = float(recording.get_sampling_frequency())
    total = int(recording.get_num_samples())
    if total <= 0 or fs <= 0:
        return

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
    frames = np.arange(start, end, step, dtype=float)
    t = (frames - float(stitch_frame)) / fs

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
        import spikeinterface.full as si
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
        rec = si.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)
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


@dataclass(frozen=True)
class RawPreprocessPlan:
    """Plan for building a concatenated recording for spikesorting.

    Today this focuses on two concepts:
    1) discover per-segment channel-location metadata (often via sibling `.cfg` files)
    2) identify the channel/electrode intersection shared across segments

    The plan exists so we can log + reproduce the exact preprocessing decisions.
    """

    h5_path: Path
    stream_id: str
    cfg_files: tuple[Path, ...]


def discover_cfg_files(h5_path: Path) -> list[Path]:
    """Return `.cfg` files adjacent to an `.h5` file.

    Maxwell exports sometimes include per-segment configuration files beside the
    recording. We treat these as *optional* because not all datasets ship them.
    """

    h5_path = Path(h5_path)
    folder = h5_path.parent
    return sorted(folder.glob("*.cfg"))


def parse_cfg_channel_locations(cfg_path: Path) -> dict:
    """Parse a Maxwell-style `.cfg` file.

    We don't yet have a single canonical `.cfg` schema across datasets. This
    parser is deliberately conservative: it attempts INI parsing first and falls
    back to raw text storage.

    Returns a dict with minimally:
    - `path`: cfg path
    - `sections`: parsed INI sections (if any)
    - `raw`: raw file text (always)

    TODO(adam): Once we inspect real `.cfg` files in your datasets, replace this
    with a schema-aware parser that returns electrode ids + (x,y) locations.
    """

    cfg_path = Path(cfg_path)
    raw = cfg_path.read_text(errors="replace")

    parser = configparser.ConfigParser()
    sections: dict[str, dict[str, str]] = {}
    try:
        parser.read_string(raw)
        for section in parser.sections():
            sections[section] = dict(parser.items(section))
    except configparser.Error:
        sections = {}

    return {"path": str(cfg_path), "sections": sections, "raw": raw}


def build_preprocess_plan(
    *,
    h5_path: Path,
    stream_id: str,
    cfg_files: Optional[Iterable[Path]] = None,
) -> RawPreprocessPlan:
    h5_path = Path(h5_path).expanduser().resolve()
    if cfg_files is None:
        cfg_files = discover_cfg_files(h5_path)
    cfg_files_tuple = tuple(Path(p).expanduser().resolve() for p in cfg_files)
    return RawPreprocessPlan(h5_path=h5_path, stream_id=stream_id, cfg_files=cfg_files_tuple)


def find_common_electrodes_from_segments(
    *,
    h5_path: Path,
    stream_id: str,
) -> tuple[list[str], list[int]]:
    """Compute the shared electrode set across all rec segments for a stream.

    Current implementation uses SpikeInterface's `MaxwellRecordingExtractor`
    contact_vector electrode ids.

    This matches the historical behavior in `internal/lib_sorting_functions.py`
    but is now housed in a dedicated preprocessing module.
    """

    try:
        import h5py
        import spikeinterface.full as si
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "raw preprocessing requires `h5py` and `spikeinterface` installed"
        ) from e

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        rec_names = list(h5["wells"][stream_id].keys())

    common: Optional[set[int]] = None
    for rec_name in rec_names:
        rec = si.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)
        electrodes = rec.get_property("contact_vector")["electrode"]
        electrode_set = set(int(x) for x in electrodes)
        if common is None:
            common = electrode_set
        else:
            common &= electrode_set

    return rec_names, sorted(common or set())


def build_concatenated_recording(
    *,
    h5_path: Path,
    stream_id: str,
    n_jobs: int = 8,
    center_chunk_size: int = 10_000,
    plot_output_dir: Optional[Path] = None,
) -> tuple[object, list[int]]:
    """Load per-segment recordings, center, slice to shared electrodes, and concatenate.

    Returns `(multirecording, common_electrodes)`.
    """

    try:
        import numpy as np
        import spikeinterface.full as si
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "raw preprocessing requires `numpy` and `spikeinterface` installed"
        ) from e

    h5_path = Path(h5_path)

    t0 = time.perf_counter()
    print(f"[axon_reconstructor] preprocessing: h5={h5_path} stream={stream_id}", flush=True)

    rec_names, common_el = find_common_electrodes_from_segments(h5_path=h5_path, stream_id=stream_id)

    print(
        f"[axon_reconstructor] found {len(rec_names)} segments; shared electrodes={len(common_el)}; "
        f"intersection took {time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    if plot_output_dir is not None:
        _save_channel_layout_plots(
            h5_path=h5_path,
            stream_id=stream_id,
            rec_names=rec_names,
            common_electrodes=common_el,
            out_dir=Path(plot_output_dir),
        )

        print(
            f"[axon_reconstructor] layout plots written to {Path(plot_output_dir)} "
            f"({time.perf_counter() - t0:.2f}s elapsed)",
            flush=True,
        )

    def process_rec_name(rec_name: str):
        rec = si.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)
        fs = float(rec.get_sampling_frequency())
        n_samples = int(rec.get_num_samples())
        chunk = min(center_chunk_size, rec.get_num_samples()) - 100
        chunk = max(chunk, 100)
        rec_centered = si.center(rec, chunk_size=chunk)

        rec_el = rec.get_property("contact_vector")["electrode"]
        chan_idx = [int(np.where(rec_el == el)[0][0]) for el in common_el]
        sel_channels = rec.get_channel_ids()[chan_idx]
        processed = rec_centered.channel_slice(sel_channels, renamed_channel_ids=list(range(len(chan_idx))))
        return processed, {"rec_name": rec_name, "fs": fs, "n_samples": n_samples, "n_channels": int(processed.get_num_channels())}

    # Keep concurrency modest; these extractors are I/O heavy.
    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(len(rec_names), max(1, int(n_jobs)))
    t_segments = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(process_rec_name, rec_names))

    rec_list = [r for r, _ in results]
    seg_stats = [s for _, s in results]

    # Print per-segment durations (in the original order).
    print(
        f"[axon_reconstructor] segment preprocessing done in {time.perf_counter() - t_segments:.2f}s "
        f"(n_jobs={n_jobs}, workers={max_workers})",
        flush=True,
    )
    for s in seg_stats:
        fs = float(s["fs"])
        n = int(s["n_samples"])
        dur = (n / fs) if fs > 0 else float("nan")
        print(
            f"[axon_reconstructor] segment {s['rec_name']}: fs={fs:.2f} Hz, "
            f"samples={n:,}, duration={dur:.2f} s, channels={int(s['n_channels'])}",
            flush=True,
        )

    # Optional: print real inter-segment gaps, if the extractor exposes absolute times.
    print_time_between_segments(rec_list, rec_names=rec_names)

    # Check sampling-frequency consistency across segments.
    fs_vals = [float(s["fs"]) for s in seg_stats if float(s["fs"]) > 0]
    if fs_vals:
        fs_min = min(fs_vals)
        fs_max = max(fs_vals)
        if abs(fs_max - fs_min) > 1e-6:
            print(
                f"[axon_reconstructor][WARN] segment sampling rates differ: min={fs_min:.6f} Hz max={fs_max:.6f} Hz",
                flush=True,
            )

    t_concat = time.perf_counter()
    multirecording = si.concatenate_recordings(rec_list)

    fs_cat = float(multirecording.get_sampling_frequency())
    n_cat = int(multirecording.get_num_samples())
    dur_cat = (n_cat / fs_cat) if fs_cat > 0 else float("nan")
    print(
        f"[axon_reconstructor] concatenated recording: fs={fs_cat:.2f} Hz, "
        f"samples={n_cat:,}, duration={dur_cat:.2f} s, channels={int(multirecording.get_num_channels())} "
        f"(concat took {time.perf_counter() - t_concat:.2f}s)",
        flush=True,
    )

    if plot_output_dir is not None:
        # Plot concatenation diagnostics: cluster reps over time + stitch markers.
        # We can derive stitch frames directly from segment lengths.
        seg_lengths = [int(r.get_num_samples()) for r in rec_list]
        stitch_frames: list[int] = []
        acc = 0
        for n_frames in seg_lengths[:-1]:
            acc += int(n_frames)
            stitch_frames.append(acc)

        # Build shared-electrode positions using the first segment contact_vector.
        try:
            rec0 = si.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_names[0])
            cv0 = rec0.get_property("contact_vector")
            el0 = np.asarray(cv0["electrode"], dtype=int)
            x0, y0 = _extract_xy_from_contact_vector(cv0)
            x0 = np.asarray(x0, dtype=float)
            y0 = np.asarray(y0, dtype=float)

            # Map each common electrode -> position from segment 0.
            el_to_pos = {int(e): (float(x), float(y)) for e, x, y in zip(el0, x0, y0, strict=False)}
            xs = np.asarray([el_to_pos[int(e)][0] for e in common_el], dtype=float)
            ys = np.asarray([el_to_pos[int(e)][1] for e in common_el], dtype=float)

            clusters = detect_electrode_clusters(x=xs, y=ys, max_cluster_size_warn=9)
            rep_channel_ids: list[int] = []
            for c in clusters:
                rep_idx = _pick_representative_index_for_cluster(x=xs, y=ys, cluster=c)
                rep_channel_ids.append(int(rep_idx))

            # Score reps by activity and keep the most active representative for now.
            scores = [
                _activity_score_rms(recording=multirecording, channel_id=ch)
                for ch in rep_channel_ids
            ]
            rep_sorted = [ch for ch, _ in sorted(zip(rep_channel_ids, scores, strict=False), key=lambda t: t[1], reverse=True)]
            rep_keep = rep_sorted[:1]

            _plot_concat_cluster_traces(
                recording=multirecording,
                channel_ids=rep_keep,
                stitch_frames=stitch_frames,
                out_path=Path(plot_output_dir) / f"concat_cluster_reps_{stream_id}.png",
                title=f"Concat cluster representatives ({stream_id}); red=stitch",
            )

            # Zoom into each stitch for close inspection.
            for stitch_idx, sf in enumerate(stitch_frames, start=1):
                _plot_stitch_zoom(
                    recording=multirecording,
                    channel_ids=rep_keep,
                    stitch_frame=sf,
                    out_path=Path(plot_output_dir) / f"concat_stitch_zoom_{stream_id}_{stitch_idx:03d}.png",
                    window_s=2.0,
                    title=f"Stitch zoom {stitch_idx} ({stream_id}); red=stitch",
                )
        except Exception:
            # Don't fail preprocessing if plotting diagnostics can't be generated.
            pass

    return multirecording, common_el
