from __future__ import annotations

import configparser
import contextlib
from dataclasses import dataclass
import datetime as dt
import io
from pathlib import Path
import sys
import time
from typing import Iterable, Optional


@contextlib.contextmanager
def _tee_stdout_to_file(out_path: Path):
    """Write stdout to both terminal and a file for debug logs."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:

        class _Tee(io.TextIOBase):
            def __init__(self, a, b):
                self._a = a
                self._b = b

            def write(self, s):
                self._a.write(s)
                self._b.write(s)
                return len(s)

            def flush(self):
                try:
                    self._a.flush()
                finally:
                    self._b.flush()

        tee = _Tee(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            yield out_path


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
        import spikeinterface.extractors as se
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "raw preprocessing requires `h5py` and `spikeinterface` installed"
        ) from e

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        rec_names = list(h5["wells"][stream_id].keys())

    common: Optional[set[int]] = None
    for rec_name in rec_names:
        # SpikeInterface <= 0.102.x exposed `MaxwellRecordingExtractor` via `spikeinterface.full`.
        # SpikeInterface >= 0.103.x removed that re-export, so prefer the stable function API.
        if hasattr(se, "read_maxwell"):
            # This appears to print "The h5 compression library for Maxwell is already located in /home/adamm/dev/pkgs/axon_reconstructor/vendor/maxwell_hdf5_plugin/Linux/libcompression.so!"
            # when hdf5plugin is loaded; ignore.
            # but perhaps we dont need to load hdf5plugin at all here?
            rec = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
        else:  # pragma: no cover
            # Old SpikeInterface versions.
            rec = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)
        electrodes = rec.get_property("contact_vector")["electrode"]
        electrode_set = set(int(x) for x in electrodes)
        if common is None:
            common = electrode_set
        else:
            common &= electrode_set

    return rec_names, sorted(common or set())


def _print_h5_stream_summary(
    *,
    h5_path: Path,
    stream_id: Optional[str] = None,
    max_streams_print: int = 25,
    max_segments_print: int = 10,
) -> None:
    """Print a lightweight HDF5 summary (streams/segments) and estimate total duration for a stream.

    This uses `h5py` only (no SpikeInterface) and relies on best-effort heuristics to infer:
    - sampling frequency (from common attribute names)
    - number of samples per segment (from dataset shapes under each segment group)

    It is intended as a debugging aid and should not be treated as a strict file parser.
    """

    try:
        import h5py
        import numpy as np
    except Exception:
        print("[axon_reconstructor][DEBUG] h5 summary skipped (missing h5py/numpy)", flush=True)
        return

    def _lower_keys(d):
        try:
            return {str(k).lower(): k for k in d.keys()}
        except Exception:
            return {}

    def _try_get_attr(obj, names: list[str]):
        key_map = _lower_keys(getattr(obj, "attrs", {}))
        for n in names:
            k = key_map.get(n.lower())
            if k is None:
                continue
            try:
                v = obj.attrs[k]
                # unwrap numpy scalars/0-d arrays
                if isinstance(v, np.ndarray) and v.shape == ():
                    v = v.item()
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode(errors="ignore")
                return v
            except Exception:
                continue
        return None

    def _infer_sampling_frequency(*objs) -> Optional[float]:
        attr_names = [
            "sampling_frequency",
            "sampling rate",
            "sampling_rate",
            "samplerate",
            "sample_rate",
            "fs",
            "frequency",
        ]
        for o in objs:
            if o is None:
                continue
            v = _try_get_attr(o, attr_names)
            if v is None:
                continue
            try:
                fv = float(v)
                if fv > 0:
                    return fv
            except Exception:
                continue
        #return None

        # Fallback: return 10,000 Hz as a common default for Maxwell recordings.
        print("[axon_reconstructor][DEBUG] could not infer fs from attributes; defaulting to 10,000 Hz", flush=True)
        return 10000.0

    def _infer_segment_num_samples(seg_group) -> tuple[Optional[int], Optional[str]]:
        # Heuristic: look for the largest numeric dataset (by number of elements)
        # and use its longest dimension as "n_samples".
        best = (0, None, None)  # (n_elems, n_samples, path)

        def visitor(name, obj):
            nonlocal best
            if not isinstance(obj, h5py.Dataset):
                return
            try:
                if obj.shape is None:
                    return
                if obj.dtype is None:
                    return
                if obj.dtype.kind not in ("i", "u", "f"):
                    return
                if len(obj.shape) == 0:
                    return
                n_elems = int(np.prod(obj.shape))
                if n_elems <= 0:
                    return
                n_samples = int(max(obj.shape))
                if n_elems > best[0]:
                    best = (n_elems, n_samples, name)
            except Exception:
                return

        try:
            seg_group.visititems(visitor)
        except Exception:
            return None, None
        return (best[1] if best[1] else None), best[2]

    h5_path = Path(h5_path).expanduser().resolve()
    try:
        with h5py.File(h5_path, "r") as h5:
            if "wells" not in h5:
                print(f"[axon_reconstructor][DEBUG] h5 has no 'wells' group: {h5_path}", flush=True)
                return

            wells = h5["wells"]
            stream_ids = list(wells.keys())
            print(
                f"[axon_reconstructor][DEBUG] h5 streams: n={len(stream_ids)} (showing up to {max_streams_print})",
                flush=True,
            )
            for sid in stream_ids[: max(0, int(max_streams_print))]:
                try:
                    n_segs = len(wells[sid].keys())
                except Exception:
                    n_segs = -1
                marker = " <==" if (stream_id is not None and str(sid) == str(stream_id)) else ""
                print(f"[axon_reconstructor][DEBUG]  - {sid}: segments={n_segs}{marker}", flush=True)

            if stream_id is None:
                return
            if str(stream_id) not in wells:
                print(
                    f"[axon_reconstructor][DEBUG] requested stream_id={stream_id!r} not found; available={stream_ids}",
                    flush=True,
                )
                return

            stream = wells[str(stream_id)]
            rec_names = list(stream.keys())
            fs = _infer_sampling_frequency(stream, wells, h5)
            if fs is not None:
                print(f"[axon_reconstructor][DEBUG] inferred fs={fs:.6f} Hz for stream={stream_id}", flush=True)
            else:
                print(f"[axon_reconstructor][DEBUG] could not infer fs for stream={stream_id}", flush=True)

            total_samples = 0
            unknown = 0
            for rn in rec_names:
                seg = stream[rn]
                n_samp, ds_path = _infer_segment_num_samples(seg)
                if n_samp is None:
                    unknown += 1
                    continue
                total_samples += int(n_samp)
                if max_segments_print and len(rec_names) <= int(max_segments_print):
                    print(
                        f"[axon_reconstructor][DEBUG]   segment {rn}: samples~{int(n_samp):,} (from dataset '{ds_path}')",
                        flush=True,
                    )

            if fs is not None and total_samples > 0:
                dur_s = float(total_samples) / float(fs)
                print(
                    f"[axon_reconstructor][DEBUG] stream {stream_id}: total_samples~{total_samples:,} => duration~{dur_s/60.0:.2f} min ({dur_s:.1f} s)",
                    flush=True,
                )
            else:
                print(
                    f"[axon_reconstructor][DEBUG] stream {stream_id}: total_samples~{total_samples:,} (duration unknown; fs missing)",
                    flush=True,
                )
            if unknown:
                print(
                    f"[axon_reconstructor][DEBUG] stream {stream_id}: could not infer samples for {unknown}/{len(rec_names)} segments",
                    flush=True,
                )
    except OSError as e:
        print(f"[axon_reconstructor][DEBUG] failed to open h5: {h5_path} ({e})", flush=True)
        return


def _dump_h5_metadata_tree(
    *,
    h5_path: Path,
    out_path: Optional[Path] = None,
    max_attr_value_chars: int = 240,
    max_array_preview_elems: int = 16,
    include_datasets: bool = True,
    include_dataset_preview: bool = False,
) -> None:
    """Dump the full HDF5 tree (groups/datasets) and their attributes.

    Goal: make it easy to find acquisition/settings/config metadata saved in the file.

    Notes:
    - This is a debugging aid. Output can be large.
    - By default, it does NOT read dataset contents (only names/shapes/dtypes/attrs).
    """

    try:
        import h5py
        import numpy as np
    except Exception:
        print("[axon_reconstructor][DEBUG] h5 metadata dump skipped (missing h5py/numpy)", flush=True)
        return

    h5_path = Path(h5_path).expanduser().resolve()
    sink = None
    try:
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sink = open(out_path, "w", encoding="utf-8")

        def emit(line: str) -> None:
            if sink is not None:
                sink.write(line + "\n")
            else:
                print(line, flush=True)

        def fmt_value(v) -> str:
            try:
                if isinstance(v, (bytes, bytearray)):
                    s = v.decode(errors="replace")
                    return repr(s[:max_attr_value_chars]) + ("…" if len(s) > max_attr_value_chars else "")
                if isinstance(v, str):
                    return repr(v[:max_attr_value_chars]) + ("…" if len(v) > max_attr_value_chars else "")
                if isinstance(v, np.ndarray):
                    if v.shape == ():
                        return fmt_value(v.item())
                    preview = v.ravel()[: int(max_array_preview_elems)]
                    return f"ndarray(shape={v.shape}, dtype={v.dtype}, preview={preview!r}{'…' if v.size > max_array_preview_elems else ''})"
                # numpy scalar
                if hasattr(v, "dtype") and hasattr(v, "item"):
                    try:
                        return fmt_value(v.item())
                    except Exception:
                        pass
                s = repr(v)
                return s[:max_attr_value_chars] + ("…" if len(s) > max_attr_value_chars else "")
            except Exception as e:
                return f"<unprintable: {type(v).__name__}: {e}>"

        def dump_attrs(obj, indent: str) -> None:
            try:
                keys = list(getattr(obj, "attrs", {}).keys())
            except Exception:
                keys = []
            if not keys:
                return
            for k in sorted(keys, key=lambda x: str(x)):
                try:
                    v = obj.attrs[k]
                except Exception as e:
                    emit(f"{indent}  @ {k}: <error reading attr: {e}>")
                    continue
                emit(f"{indent}  @ {k}: {fmt_value(v)}")

        emit(f"[axon_reconstructor][DEBUG] HDF5 metadata dump: {h5_path}")
        emit("[axon_reconstructor][DEBUG] (groups/datasets + attributes; dataset contents are not read by default)")

        with h5py.File(h5_path, "r") as h5:
            # Root attributes
            emit("/")
            dump_attrs(h5, indent="")

            def visitor(name: str, obj) -> None:
                # name is path without leading '/'
                path = "/" + name
                if isinstance(obj, h5py.Group):
                    emit(f"{path}  (Group)")
                    dump_attrs(obj, indent="")
                elif isinstance(obj, h5py.Dataset):
                    if not include_datasets:
                        return
                    shape = getattr(obj, "shape", None)
                    dtype = getattr(obj, "dtype", None)
                    emit(f"{path}  (Dataset shape={shape} dtype={dtype})")
                    dump_attrs(obj, indent="")
                    if include_dataset_preview:
                        try:
                            # Keep this extremely conservative; some datasets are huge/compressed.
                            if shape is not None and len(shape) >= 1 and int(shape[0]) > 0:
                                sl = tuple([slice(0, 1)] + [slice(None)] * (len(shape) - 1))
                                arr = obj[sl]
                                emit(f"  preview: {fmt_value(np.asarray(arr))}")
                        except Exception as e:
                            emit(f"  preview: <error reading dataset preview: {e}>")

            h5.visititems(visitor)

        if out_path is not None:
            print(f"[axon_reconstructor][DEBUG] wrote HDF5 metadata dump to: {out_path}", flush=True)
    except OSError as e:
        print(f"[axon_reconstructor][DEBUG] failed to open h5 for metadata dump: {h5_path} ({e})", flush=True)
    finally:
        if sink is not None:
            sink.close()


def _print_assay_settings(*, h5_path: Path) -> None:
    """Print high-level assay metadata from Maxwell-style HDF5 files.

    Expected paths (when present):
    - /assay/run_id
    - /assay/script_id
    - /assay/inputs/record_time
    - /assay/inputs/electrodes
    """

    try:
        import h5py
        import numpy as np
    except Exception:
        print("[axon_reconstructor][DEBUG] assay settings skipped (missing h5py/numpy)", flush=True)
        return

    def _decode_scalar(v):
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            elif v.size == 1:
                v = v.ravel()[0].item()
        if isinstance(v, (bytes, bytearray)):
            return v.decode(errors="replace")
        return v

    def _read_text_dataset(h5, path: str) -> Optional[str]:
        if path not in h5:
            return None
        try:
            ds = h5[path]
            if not hasattr(ds, "shape"):
                return None
            v = ds[()]
            v = _decode_scalar(v)
            return str(v)
        except Exception:
            return None

    h5_path = Path(h5_path).expanduser().resolve()
    try:
        with h5py.File(h5_path, "r") as h5:
            if "assay" not in h5:
                print("[axon_reconstructor][DEBUG] /assay group not present", flush=True)
                return

            run_id = _read_text_dataset(h5, "/assay/run_id")
            script_id = _read_text_dataset(h5, "/assay/script_id")
            record_time = _read_text_dataset(h5, "/assay/inputs/record_time")

            print("[axon_reconstructor][DEBUG] assay settings:", flush=True)
            if run_id is not None:
                print(f"[axon_reconstructor][DEBUG]  - run_id: {run_id}", flush=True)
            if script_id is not None:
                print(f"[axon_reconstructor][DEBUG]  - script_id: {script_id}", flush=True)
            if record_time is not None:
                print(f"[axon_reconstructor][DEBUG]  - record_time: {record_time}", flush=True)

            # Too Much info, we can leave this out.
            # # electrodes can be a very large serialized blob; print size + a short preview.
            # electrodes_path = "/assay/inputs/electrodes"
            # if electrodes_path in h5:
            #     try:
            #         ds = h5[electrodes_path]
            #         raw = ds[()]
            #         raw = _decode_scalar(raw)
            #         raw_str = str(raw)
            #         preview = raw_str[:500]
            #         suffix = "…" if len(raw_str) > 500 else ""
            #         print(
            #             f"[axon_reconstructor][DEBUG]  - electrodes: {len(raw_str):,} chars; preview=\n{preview}{suffix}",
            #             flush=True,
            #         )
            #     except Exception as e:
            #         print(f"[axon_reconstructor][DEBUG]  - electrodes: <error reading: {e}>", flush=True)

    except OSError as e:
        print(f"[axon_reconstructor][DEBUG] assay settings: failed to open h5 ({e})", flush=True)
        return


def _print_data_store_start_stop_durations(
    *,
    h5_path: Path,
    target_stream_id: Optional[str] = None,
    max_entries_print: int = 1000,
) -> None:
    """Print start/stop/duration for each `/data_store/dataXXXX` (stream-config) block.

    In these Maxwell-style files, `/data_store/data0000`, `/data_store/data0001`, ... typically enumerate
    stream/config combinations sequentially. Each block generally contains:
    - `well_id`
    - `settings/*` (gain/hpf/lsb/sampling/spike_threshold/mapping)
    - `start_time` / `stop_time`
    - `groups/routed/raw` with shape (n_channels, n_frames)

    This is a debugging helper; some files may omit fields.
    """

    try:
        import h5py
        import numpy as np
    except Exception:
        print("[axon_reconstructor][DEBUG] data_store timing skipped (missing h5py/numpy)", flush=True)
        return

    def _read_scalar(ds) -> Optional[object]:
        try:
            v = ds[()]
        except Exception:
            return None
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            elif v.size == 1:
                v = v.ravel()[0].item()
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode(errors="replace")
            except Exception:
                return str(v)
        return v

    def _fmt(v) -> str:
        if v is None:
            return "?"
        try:
            return str(v)
        except Exception:
            return repr(v)

    def _well_label(well_id_val: Optional[int]) -> Optional[str]:
        if well_id_val is None:
            return None
        try:
            return f"well{int(well_id_val):03d}"
        except Exception:
            return None

    target_well_label = None
    if target_stream_id is not None:
        s = str(target_stream_id)
        if s.startswith("well") and s[4:].isdigit():
            target_well_label = s

    def _infer_epoch_divisor_to_seconds(values: list[int]) -> tuple[float, str]:
        """Infer whether epoch timestamps are in ms/us/ns and return divisor to seconds."""
        # Typical magnitudes:
        # - seconds since epoch: ~1e9
        # - milliseconds: ~1e12-1e13
        # - microseconds: ~1e15-1e16
        # - nanoseconds: ~1e18-1e19
        if not values:
            return 1.0, "s"
        vmax = max(values)
        if vmax >= 10**18:
            return 1e9, "ns"
        if vmax >= 10**15:
            return 1e6, "us"
        if vmax >= 10**12:
            return 1e3, "ms"
        return 1.0, "s"

    def _fmt_epoch(ts: Optional[int], *, divisor: float) -> str:
        if ts is None:
            return "?"
        try:
            import datetime as _dt

            dt = _dt.datetime.fromtimestamp(float(ts) / float(divisor), tz=_dt.timezone.utc)
            return dt.isoformat()
        except Exception:
            return str(ts)

    h5_path = Path(h5_path).expanduser().resolve()
    try:
        with h5py.File(h5_path, "r") as h5:
            if "data_store" not in h5:
                print("[axon_reconstructor][DEBUG] /data_store group not present", flush=True)
                return
            ds = h5["data_store"]

            def _key_num(k: str) -> int:
                # data0000 -> 0
                digits = "".join(ch for ch in str(k) if ch.isdigit())
                try:
                    return int(digits) if digits else 0
                except Exception:
                    return 0

            data_keys = sorted([k for k in ds.keys() if str(k).startswith("data")], key=_key_num)
            if not data_keys:
                print("[axon_reconstructor][DEBUG] /data_store has no dataXXXX entries", flush=True)
                return

            print(
                f"[axon_reconstructor][DEBUG] data_store blocks: n={len(data_keys)} (showing up to {max_entries_print})",
                flush=True,
            )

            # Track min/max timestamps for robust overall duration.
            starts_all: list[int] = []
            stops_all: list[int] = []
            entries: list[dict[str, object]] = []

            for k in data_keys[: max(0, int(max_entries_print))]:
                g = ds[k]

                well_id = None
                if "well_id" in g:
                    try:
                        well_id = int(_read_scalar(g["well_id"]))
                    except Exception:
                        well_id = None
                wl = _well_label(well_id)

                start = _read_scalar(g["start_time"]) if "start_time" in g else None
                stop = _read_scalar(g["stop_time"]) if "stop_time" in g else None
                try:
                    if start is not None:
                        starts_all.append(int(start))
                    if stop is not None:
                        stops_all.append(int(stop))
                except Exception:
                    pass

                delta = None
                try:
                    if start is not None and stop is not None:
                        delta = int(stop) - int(start)
                except Exception:
                    delta = None

                # Read key settings if present.
                gain = _read_scalar(g["settings/gain"]) if "settings/gain" in g else None
                hpf = _read_scalar(g["settings/hpf"]) if "settings/hpf" in g else None
                lsb = _read_scalar(g["settings/lsb"]) if "settings/lsb" in g else None
                thresh = _read_scalar(g["settings/spike_threshold"]) if "settings/spike_threshold" in g else None
                fs = _read_scalar(g["settings/sampling"]) if "settings/sampling" in g else None
                try:
                    fs = float(fs) if fs is not None else None
                except Exception:
                    fs = None

                # Infer frames/channels from routed/raw.
                n_channels = None
                n_frames = None
                if "groups/routed/raw" in g:
                    try:
                        shape = g["groups/routed/raw"].shape
                        if shape is not None and len(shape) == 2:
                            n_channels = int(shape[0])
                            n_frames = int(shape[1])
                    except Exception:
                        pass
                if n_frames is None and "groups/routed/frame_nos" in g:
                    try:
                        n_frames = int(g["groups/routed/frame_nos"].shape[0])
                    except Exception:
                        pass

                dur_by_frames = None
                if fs is not None and n_frames is not None and fs > 0:
                    dur_by_frames = float(n_frames) / float(fs)

                entries.append(
                    {
                        "key": str(k),
                        "well_label": wl,
                        "well_id": well_id,
                        "start": start,
                        "stop": stop,
                        "delta": delta,
                        "fs": fs,
                        "n_channels": n_channels,
                        "n_frames": n_frames,
                        "dur_by_frames": dur_by_frames,
                        "gain": gain,
                        "hpf": hpf,
                        "lsb": lsb,
                        "thr": thresh,
                    }
                )

            # Infer epoch unit once (ms/us/ns) for consistent per-block prints.
            if starts_all and stops_all:
                min_start = min(starts_all)
                max_stop = max(stops_all)
                divisor, unit = _infer_epoch_divisor_to_seconds([min_start, max_stop])
            else:
                divisor, unit = 1.0, "s"

            # Per-block printing + gaps between consecutive config blocks.
            last_stop_by_well: dict[str, int] = {}
            last_key_by_well: dict[str, str] = {}
            for e in entries:
                wl = e.get("well_label")
                if target_well_label is not None and wl != target_well_label:
                    continue

                k = str(e.get("key"))
                well_id = e.get("well_id")
                start = e.get("start")
                stop = e.get("stop")
                delta = e.get("delta")

                line = (
                    f"[axon_reconstructor][DEBUG]  - {k}: well={wl or _fmt(well_id)} "
                    f"fs={_fmt(e.get('fs'))}Hz nch={_fmt(e.get('n_channels'))} frames={_fmt(e.get('n_frames'))} "
                    f"start={_fmt(start)} ({_fmt_epoch(int(start) if start is not None else None, divisor=divisor)}) "
                    f"stop={_fmt(stop)} ({_fmt_epoch(int(stop) if stop is not None else None, divisor=divisor)}) "
                    f"delta={_fmt(delta)} ({unit})"
                )
                print(line, flush=True)

                # Duration and gap (in seconds) computed from start/stop.
                dur_s = None
                if delta is not None:
                    try:
                        dur_s = float(int(delta)) / float(divisor)
                    except Exception:
                        dur_s = None

                gap_s = None
                prev_key = None
                if wl is not None and start is not None:
                    try:
                        prev_stop = last_stop_by_well.get(wl)
                        prev_key = last_key_by_well.get(wl)
                        if prev_stop is not None:
                            gap_raw = int(start) - int(prev_stop)
                            gap_s = float(gap_raw) / float(divisor)
                    except Exception:
                        gap_s = None

                if wl is not None and stop is not None:
                    try:
                        last_stop_by_well[wl] = int(stop)
                        last_key_by_well[wl] = k
                    except Exception:
                        pass

                cfg_bits = [
                    f"gain={_fmt(e.get('gain'))}",
                    f"hpf={_fmt(e.get('hpf'))}",
                    f"lsb={_fmt(e.get('lsb'))}",
                    f"thr={_fmt(e.get('thr'))}",
                ]
                time_bits = []
                if dur_s is not None:
                    time_bits.append(f"dur={dur_s:.6f}s")
                if gap_s is not None and prev_key is not None:
                    time_bits.append(f"gap(prev {prev_key}->{k})={gap_s:.6f}s")

                print(
                    "[axon_reconstructor][DEBUG]      " + ", ".join(cfg_bits + (["; "] if time_bits else []) + time_bits),
                    flush=True,
                )
            

            # Overall duration summary (earliest start -> latest stop).
            if starts_all and stops_all:
                min_start = min(starts_all)
                max_stop = max(stops_all)
                overall_delta = int(max_stop) - int(min_start)

                overall_delta_s = float(overall_delta) / float(divisor)
                print(
                    f"[axon_reconstructor][DEBUG] data_store overall: min_start={min_start} ({_fmt_epoch(min_start, divisor=divisor)}) "
                    f"max_stop={max_stop} ({_fmt_epoch(max_stop, divisor=divisor)}) delta={overall_delta} ({unit})",
                    flush=True,
                )
                print(
                    f"[axon_reconstructor][DEBUG] data_store overall duration: {overall_delta_s:.3f} s ({overall_delta_s/60.0:.2f} min) assuming epoch-{unit}",
                    flush=True,
                )
                
    except OSError as e:
        print(f"[axon_reconstructor][DEBUG] data_store timing: failed to open h5 ({e})", flush=True)
        return


def _process_rec_segment_for_concatenation(
    *,
    h5_path: Path,
    stream_id: str,
    rec_name: str,
    common_el: list[int],
    center_chunk_size: int,
    expected_xy_by_electrode: Optional[dict[int, tuple[float, float]]] = None,
    expected_xy_atol: float = 0.0,
):
    """Load a segment, center, select the shared electrodes, validate ordering, and normalize channel ids.

    Important behavioral notes (for posterity):

    - SpikeInterface concatenation is strict: it only concatenates recordings when *dtype* and *channel_ids*
      arrays are exactly identical across segments (including order). It does not match by electrode metadata.
    - Historically this pipeline normalized channel ids to 0..n-1 before concatenation.
    - Renaming to 0..n-1 can hide per-segment mismatches (e.g. wrong electrode->channel mapping) because it
      forces channel_ids equality even if the underlying electrode identity differs.
    - To keep concat deterministic without losing identity, we rename channel ids to the *validated* electrode
      ids (`common_el`) after confirming the selected electrodes and (optionally) their x/y locations match.
    """

    try:
        import numpy as np
        import spikeinterface.full as si
        import spikeinterface.extractors as se
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "raw preprocessing requires `numpy` and `spikeinterface` installed"
        ) from e

    if hasattr(se, "read_maxwell"):
        rec = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
    else:  # pragma: no cover
        rec = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)

    fs = float(rec.get_sampling_frequency())
    n_samples = int(rec.get_num_samples())
    chunk = min(center_chunk_size, rec.get_num_samples()) - 100
    chunk = max(chunk, 100)
    rec_centered = si.center(rec, chunk_size=chunk)

    # Map electrode id -> channel index within this segment.
    rec_el = np.asarray(rec.get_property("contact_vector")["electrode"], dtype=int)
    if int(np.unique(rec_el).size) != int(rec_el.size):
        raise RuntimeError(
            f"Duplicate electrode ids found in contact_vector for segment {rec_name}; cannot map electrodes reliably"
        )
    el_to_idx = {int(el): int(i) for i, el in enumerate(rec_el)}
    try:
        chan_idx = [el_to_idx[int(el)] for el in common_el]
    except KeyError as e:
        raise RuntimeError(
            f"Segment {rec_name} is missing expected electrode id={e.args[0]} from the common electrode set"
        ) from e

    sel_channels = np.asarray(rec.get_channel_ids(), dtype=object)[chan_idx]

    # SpikeInterface <=0.102.x (and earlier): this used to work and could also rename ids:
    # rec_centered_sliced = rec_centered.channel_slice(sel_channels, renamed_channel_ids=list(range(len(sel_channels))))
    # SpikeInterface 0.103.x: BaseRecording no longer has `channel_slice()`.
    processed = rec_centered.select_channels(list(sel_channels))

    # Validate that selection did what we asked (before any renaming).
    processed_ch = np.asarray(processed.get_channel_ids(), dtype=object)
    if processed_ch.shape != sel_channels.shape or not np.array_equal(processed_ch, sel_channels):
        raise RuntimeError(
            f"Selected channel_ids mismatch for segment {rec_name}. "
            "This may indicate a channel-id ordering issue during selection."
        )

    # Validate electrode identity and order.
    processed_el = np.asarray(processed.get_property("contact_vector")["electrode"], dtype=int)
    expected_el = np.asarray(common_el, dtype=int)
    if processed_el.shape != expected_el.shape or not np.array_equal(processed_el, expected_el):
        raise RuntimeError(
            f"Selected electrodes mismatch for segment {rec_name}. "
            f"Expected {expected_el.shape[0]} electrodes matching common set; got {processed_el.shape[0]} "
            f"and/or different ordering."
        )

    # Optional: validate electrode locations (x/y) match the reference segment, if available.
    if expected_xy_by_electrode is not None:
        cv = processed.get_property("contact_vector")
        x, y = _extract_xy_from_contact_vector(cv)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        expected_x = np.asarray([expected_xy_by_electrode[int(el)][0] for el in expected_el], dtype=float)
        expected_y = np.asarray([expected_xy_by_electrode[int(el)][1] for el in expected_el], dtype=float)
        if not (np.allclose(x, expected_x, atol=float(expected_xy_atol)) and np.allclose(y, expected_y, atol=float(expected_xy_atol))):
            raise RuntimeError(
                f"Electrode x/y locations differ from reference for segment {rec_name}. "
                "This suggests inconsistent layouts across segments; refusing to concatenate."
            )

    # Now normalize channel ids in a way that preserves identity and makes concat deterministic.
    processed = processed.rename_channels([int(el) for el in expected_el])

    renamed_ch = np.asarray(processed.get_channel_ids(), dtype=object)
    if renamed_ch.shape != expected_el.shape or not np.array_equal(renamed_ch.astype(int), expected_el):
        raise RuntimeError(f"Failed to rename channel ids to electrode ids for segment {rec_name}")

    return processed, {
        "rec_name": rec_name,
        "fs": fs,
        "n_samples": n_samples,
        "n_channels": int(processed.get_num_channels()),
    }


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
        import spikeinterface.extractors as se
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "raw preprocessing requires `numpy` and `spikeinterface` installed"
        ) from e

    h5_path = Path(h5_path)

    t0 = time.perf_counter()
    print(f"[axon_reconstructor] preprocessing: h5={h5_path} stream={stream_id}", flush=True)

    # Save assay + data_store timing stats to a text file while still echoing to terminal.
    stats_dir = Path(plot_output_dir) if plot_output_dir is not None else h5_path.parent
    stats_path = stats_dir / f"assay_stats_{stream_id}.txt"
    try:
        with _tee_stdout_to_file(stats_path) as p:
            print(
                f"[axon_reconstructor][DEBUG] assay_stats file: {p} "
                f"(generated {dt.datetime.now(dt.timezone.utc).isoformat()})",
                flush=True,
            )
            print(f"[axon_reconstructor][DEBUG] assay_stats context: h5={h5_path} stream={stream_id}", flush=True)

            # Quick check for assay-level metadata embedded in the HDF5.
            _print_assay_settings(h5_path=h5_path)

            # Print start/stop/duration for each stream-config block in /data_store.
            _print_data_store_start_stop_durations(h5_path=h5_path, target_stream_id=stream_id)
    except Exception as e:
        print(f"[axon_reconstructor][WARN] failed to write assay_stats file to {stats_path}: {e}", flush=True)
        _print_assay_settings(h5_path=h5_path)
        _print_data_store_start_stop_durations(h5_path=h5_path, target_stream_id=stream_id)

    # Debugging aid: summarize streams/segments and estimate stream duration.
    #_print_h5_stream_summary(h5_path=h5_path, stream_id=stream_id)

    # Debugging aid: dump all HDF5 attributes/settings so we can find acquisition config.
    # This can be very verbose; write to plot_output_dir when provided.
    # Useful tool, but we dont need to run this everytime. Move to a debug libary later.
    # if plot_output_dir is not None:
    #     _dump_h5_metadata_tree(
    #         h5_path=h5_path,
    #         out_path=Path(plot_output_dir) / f"h5_metadata_dump_{stream_id}.txt",
    #         include_datasets=True,
    #         include_dataset_preview=False,
    #     )
    # else:
    #     _dump_h5_metadata_tree(h5_path=h5_path, out_path=None, include_datasets=True, include_dataset_preview=False)

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

    # Build a reference map electrode_id -> (x, y) from the first segment.
    # This lets us confirm that all segments share the same electrode layout/order before concatenation.
    expected_xy_by_electrode: Optional[dict[int, tuple[float, float]]] = None
    try:
        if rec_names:
            if hasattr(se, "read_maxwell"):
                rec0 = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_names[0])
            else:  # pragma: no cover
                rec0 = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_names[0])
            cv0 = rec0.get_property("contact_vector")
            el0 = np.asarray(cv0["electrode"], dtype=int)
            x0, y0 = _extract_xy_from_contact_vector(cv0)
            expected_xy_by_electrode = {
                int(e): (float(x), float(y))
                for e, x, y in zip(el0, np.asarray(x0, dtype=float), np.asarray(y0, dtype=float), strict=False)
            }
    except Exception:
        # If x/y is unavailable or malformed, fall back to electrode-id-only validation.
        expected_xy_by_electrode = None

    # Keep concurrency modest; these extractors are I/O heavy.
    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(len(rec_names), max(1, int(n_jobs)))
    t_segments = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        from functools import partial

        process = partial(
            _process_rec_segment_for_concatenation,
            h5_path=h5_path,
            stream_id=stream_id,
            common_el=common_el,
            center_chunk_size=center_chunk_size,
            expected_xy_by_electrode=expected_xy_by_electrode,
            expected_xy_atol=0.0,
        )
        results = list(ex.map(lambda rn: process(rec_name=rn), rec_names))

    rec_list = [r for r, _ in results]
    seg_stats = [s for _, s in results]

    # # Print per-segment durations (in the original order).
    print(
        f"[axon_reconstructor] segment preprocessing done in {time.perf_counter() - t_segments:.2f}s "
        f"(n_jobs={n_jobs}, workers={max_workers})",
        flush=True,
    )
    # for s in seg_stats:
    #     fs = float(s["fs"])
    #     n = int(s["n_samples"])
    #     dur = (n / fs) if fs > 0 else float("nan")
    #     print(
    #         f"[axon_reconstructor] segment {s['rec_name']}: fs={fs:.2f} Hz, "
    #         f"samples={n:,}, duration={dur:.2f} s, channels={int(s['n_channels'])}",
    #         flush=True,
    #     )

    # # Optional: print real inter-segment gaps, if the extractor exposes absolute times.
    # print_time_between_segments(rec_list, rec_names=rec_names)

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
            if hasattr(se, "read_maxwell"):
                rec0 = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_names[0])
            else:  # pragma: no cover
                rec0 = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_names[0])
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
            mr_channel_ids = list(multirecording.get_channel_ids())
            rep_channel_ids: list[int] = []
            for c in clusters:
                rep_idx = _pick_representative_index_for_cluster(x=xs, y=ys, cluster=c)
                # rep_idx is an index into the shared-electrode arrays (xs/ys), NOT necessarily a channel id.
                # Map it to the actual channel id used by the (possibly non-renamed) recording.
                rep_channel_ids.append(int(mr_channel_ids[int(rep_idx)]))

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
