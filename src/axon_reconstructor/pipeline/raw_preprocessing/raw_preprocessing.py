from __future__ import annotations

import json
import configparser
from dataclasses import dataclass
import datetime as dt
from pathlib import Path
import time
from typing import Iterable, Optional


def _ensure_maxwell_hdf5_plugin_path(*, prefix: str = "[axon_reconstructor]") -> None:
    """Best-effort fix for Maxwell HDF5 decompression plugin discovery.

    Some environments end up with `HDF5_PLUGIN_PATH` pointing at a non-existent
    directory, which causes HDF5 reads to fail when the Maxwell compression
    filter is encountered.

    We prefer the vendored plugin shipped with this repo when available.
    """

    import os

    env = os.environ.get("HDF5_PLUGIN_PATH")
    if env:
        try:
            if not Path(env).expanduser().exists():
                print(f"{prefix}[WARN] HDF5_PLUGIN_PATH points to missing dir: {env}; ignoring", flush=True)
                os.environ.pop("HDF5_PLUGIN_PATH", None)
        except Exception:
            pass

    if os.environ.get("HDF5_PLUGIN_PATH"):
        return

    # Look for a vendored plugin directory.
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        cand_dir = parent / "vendor" / "maxwell_hdf5_plugin" / "Linux"
        if (cand_dir / "libcompression.so").exists():
            os.environ["HDF5_PLUGIN_PATH"] = str(cand_dir)
            print(f"{prefix}[DEBUG] set HDF5_PLUGIN_PATH={cand_dir}", flush=True)
            return

from .h5_helpers import (
    _read_well_rec_frame_nos_and_trigger_settings,
    _print_assay_settings,
    _print_data_store_start_stop_durations,
    _tee_stdout_to_file,
    print_time_between_segments,
)
from .plotting import (
    _activity_score_rms,
    _extract_xy_from_contact_vector,
    _pick_representative_index_for_cluster,
    _plot_concat_cluster_traces,
    _plot_stitch_zoom,
    _save_channel_layout_plots,
    detect_electrode_clusters,
)


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

    _ensure_maxwell_hdf5_plugin_path()

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

    _ensure_maxwell_hdf5_plugin_path()

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
        if not (
            np.allclose(x, expected_x, atol=float(expected_xy_atol))
            and np.allclose(y, expected_y, atol=float(expected_xy_atol))
        ):
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
    epoch_markers_output_dir: Optional[Path] = None,
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

    _ensure_maxwell_hdf5_plugin_path()

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

    rec_names, common_el = find_common_electrodes_from_segments(h5_path=h5_path, stream_id=stream_id)

    print(
        f"[axon_reconstructor] found {len(rec_names)} segments; shared electrodes={len(common_el)}; "
        f"intersection took {time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    if plot_output_dir is not None:
        plot_output_dir = Path(plot_output_dir)
        channel_layouts_dir = plot_output_dir / "channel_layouts"
        channel_layouts_dir.mkdir(parents=True, exist_ok=True)
        _save_channel_layout_plots(
            h5_path=h5_path,
            stream_id=stream_id,
            rec_names=rec_names,
            common_electrodes=common_el,
            out_dir=channel_layouts_dir,
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

    print(
        f"[axon_reconstructor] segment preprocessing done in {time.perf_counter() - t_segments:.2f}s "
        f"(n_jobs={n_jobs}, workers={max_workers})",
        flush=True,
    )

    # Optional: print real inter-segment gaps, if the extractor exposes absolute times.
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

    # Precompute segment stitch epochs in concatenated sample coordinates.
    seg_lengths = [int(r.get_num_samples()) for r in rec_list]
    seg_offsets: list[int] = []
    acc = 0
    for n_frames in seg_lengths:
        seg_offsets.append(acc)
        acc += int(n_frames)

    concat_epochs: list[dict] = []
    for i, (rn, n_frames, start) in enumerate(zip(rec_names, seg_lengths, seg_offsets, strict=False)):
        concat_epochs.append(
            {
                "segment_index": int(i),
                "rec_name": str(rn),
                "start_sample": int(start),
                "end_sample": int(start + int(n_frames)),
                "n_samples": int(n_frames),
            }
        )

    # Reconstruct accurate time vectors from `frame_nos`.
    #
    # Maxwell `.raw.h5` files can store *triggered* snippets spread across a longer
    # wall-clock span. In that case `get_num_samples()/fs` underestimates the real
    # span, and plots vs. `sample_index/fs` look too short with no gaps.
    #
    # We use `/wells/<stream>/<rec>/groups/routed/frame_nos` to build:
    #  - per-segment *relative* times (0..~record_time) for plotting individual segments
    #  - concatenated *absolute-ish* times (aligned to the first segment) so inter-segment
    #    gaps appear when plotting the concatenated recording.
    try:
        import numpy as np

        time_vectors: list[np.ndarray] = []
        t0_epoch_s: Optional[float] = None

        # Epochs of contiguous samples within Maxwell triggered/snippet recording strategy.
        # Expressed in *concatenated* sample coordinates (multirecording time axis).
        maxwell_epochs: list[dict] = []

        for seg_index, (rn, st) in enumerate(zip(rec_names, seg_stats, strict=False)):
            info = _read_well_rec_frame_nos_and_trigger_settings(
                h5_path=h5_path,
                stream_id=stream_id,
                rec_name=rn,
            )

            fs = float(st.get("fs", 0.0) or 0.0)
            if fs <= 0:
                continue

            start_epoch_s = float(info["start_ms"]) / 1000.0
            stop_epoch_s = float(info["stop_ms"]) / 1000.0
            if t0_epoch_s is None:
                t0_epoch_s = start_epoch_s

            frame_nos = np.asarray(info["frame_nos"], dtype=np.int64)
            n_samples = int(st.get("n_samples", frame_nos.size))
            if frame_nos.size != n_samples:
                print(
                    f"[axon_reconstructor][WARN] {rn}: frame_nos length ({frame_nos.size}) != n_samples ({n_samples}); skipping time vector",
                    flush=True,
                )
                continue

            # Detect discontinuities in the saved frames (triggered/snippet gaps show up here).
            # A contiguous epoch is a run where diff(frame_nos)==1.
            diffs = np.diff(frame_nos)
            split_points = np.flatnonzero(diffs != 1) + 1
            run_starts = np.concatenate(([0], split_points))
            run_ends = np.concatenate((split_points, [frame_nos.size]))
            seg_offset = int(seg_offsets[seg_index]) if seg_index < len(seg_offsets) else 0

            for rs, re in zip(run_starts, run_ends, strict=False):
                rs_i = int(rs)
                re_i = int(re)
                if re_i <= rs_i:
                    continue
                maxwell_epochs.append(
                    {
                        "segment_index": int(seg_index),
                        "rec_name": str(rn),
                        "start_sample": int(seg_offset + rs_i),
                        "end_sample": int(seg_offset + re_i),
                        "segment_start_sample": int(rs_i),
                        "segment_end_sample": int(re_i),
                        "frame_no_start": int(frame_nos[rs_i]),
                        "frame_no_end": int(frame_nos[re_i - 1]),
                    }
                )

            frame0 = int(frame_nos[0])
            times_rel = (frame_nos - frame0) / fs
            times_abs = (start_epoch_s - float(t0_epoch_s)) + times_rel

            # For per-segment plots, use relative time so each figure spans ~0..100s.
            if plot_output_dir is not None:
                try:
                    rec_list[seg_index].set_times(times_rel.astype(float, copy=False))
                except Exception:
                    pass

            time_vectors.append(times_abs.astype(float, copy=False))

            wall_dur = stop_epoch_s - start_epoch_s
            sample_dur = n_samples / fs
            frame_span_dur = (int(frame_nos[-1]) - int(frame_nos[0])) / fs

            triggered = info.get("triggered")
            if triggered == 1 or (wall_dur > 0 and (sample_dur / wall_dur) < 0.8):
                pre = info.get("trigger_pre")
                post = info.get("trigger_post")
                snip_ms = None
                try:
                    if pre is not None and post is not None and fs > 0:
                        snip_ms = 1000.0 * (float(pre) + float(post)) / fs
                except Exception:
                    snip_ms = None

                msg = (
                    f"[axon_reconstructor][DEBUG] {rn}: wall_dur={wall_dur:.3f}s, "
                    f"frame_span={frame_span_dur:.3f}s, stored_samples_dur={sample_dur:.3f}s"
                )
                if triggered == 1:
                    msg += " (triggered recording)"
                if snip_ms is not None:
                    msg += f"; trigger_pre+post~{snip_ms:.1f}ms"
                print(msg, flush=True)

        if time_vectors and t0_epoch_s is not None:
            concat_times = np.concatenate(time_vectors)
            if concat_times.size == int(multirecording.get_num_samples()):
                multirecording.set_times(concat_times)
            else:
                print(
                    "[axon_reconstructor][WARN] concatenated time vector length mismatch; "
                    f"times={concat_times.size:,} samples={int(multirecording.get_num_samples()):,}; leaving default times",
                    flush=True,
                )
    except Exception as e:
        print(f"[axon_reconstructor][WARN] failed to set time vector from frame_nos: {e}", flush=True)

    # Persist epoch marker JSON artifacts for later analysis.
    if epoch_markers_output_dir is not None:
        out_dir = Path(epoch_markers_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        maxwell_path = out_dir / f"maxwell_contiguous_epochs_{stream_id}.json"
        concat_path = out_dir / f"concatenation_stitch_epochs_{stream_id}.json"

        try:
            # maxwell_epochs is defined inside the try block above; fall back to empty if unavailable.
            maxwell_epochs_payload = locals().get("maxwell_epochs", [])
            with open(maxwell_path, "w", encoding="utf-8") as f:
                json.dump(list(maxwell_epochs_payload), f, indent=2)

            with open(concat_path, "w", encoding="utf-8") as f:
                json.dump(list(concat_epochs), f, indent=2)

            print(
                f"[axon_reconstructor] wrote epoch markers: maxwell={maxwell_path.name} concat={concat_path.name}",
                flush=True,
            )
        except Exception as e:
            print(f"[axon_reconstructor][WARN] failed to write epoch marker JSON: {e}", flush=True)

    fs_cat = float(multirecording.get_sampling_frequency())
    n_cat = int(multirecording.get_num_samples())
    dur_cat = (n_cat / fs_cat) if fs_cat > 0 else float("nan")

    dur_timevec = None
    try:
        if bool(multirecording.has_time_vector()):
            dur_timevec = float(multirecording.get_end_time() - multirecording.get_start_time())
    except Exception:
        dur_timevec = None

    print(
        f"[axon_reconstructor] concatenated recording: fs={fs_cat:.2f} Hz, "
        f"samples={n_cat:,}, duration_samples/fs={dur_cat:.2f} s"
        + (
            f", duration_time_vector={dur_timevec:.2f} s" if dur_timevec is not None else ""
        )
        + f", channels={int(multirecording.get_num_channels())} "
        f"(concat took {time.perf_counter() - t_concat:.2f}s)",
        flush=True,
    )

    if plot_output_dir is not None:
        # Plot concatenation diagnostics: cluster reps over time + stitch markers.
        # We can derive stitch frames directly from segment lengths.
        plot_output_dir = Path(plot_output_dir)
        segment_traces_dir = plot_output_dir / "segment_traces"
        segment_traces_dir.mkdir(parents=True, exist_ok=True)
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
            scores = [_activity_score_rms(recording=multirecording, channel_id=ch) for ch in rep_channel_ids]
            rep_sorted = [
                ch
                for ch, _ in sorted(
                    zip(rep_channel_ids, scores, strict=False), key=lambda t: t[1], reverse=True
                )
            ]
            rep_keep = rep_sorted[:4] # keep and plot top 2 representatives

            _plot_concat_cluster_traces(
                recording=multirecording,
                channel_ids=rep_keep,
                stitch_frames=stitch_frames,
                out_path=plot_output_dir / f"concat_cluster_reps_{stream_id}.png",
                title=f"Concat cluster representatives ({stream_id}); red=stitch",
            )

            # Also plot the same representative channel for each segment individually.
            # These segment plots use the per-segment time vector derived from `frame_nos`
            # (relative seconds within segment) so internal gaps/snippets are visible.
            for rn, seg_rec in zip(rec_names, rec_list, strict=False):
                _plot_concat_cluster_traces(
                    recording=seg_rec,
                    channel_ids=rep_keep,
                    stitch_frames=[],
                    out_path=segment_traces_dir / f"segment_trace_{stream_id}_{rn}.png",
                    title=f"Segment trace ({stream_id} / {rn})",
                )
            
        except Exception:
            # Don't fail preprocessing if plotting diagnostics can't be generated.
            pass

    return multirecording, common_el
