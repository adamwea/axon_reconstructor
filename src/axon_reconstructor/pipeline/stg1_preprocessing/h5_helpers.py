from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import threading
from typing import Optional


_STDOUT_TEE_LOCK = threading.Lock()


def _print_assay_settings(*, h5_path: Path, prefix: str = "[axon_reconstructor]") -> None:
    """Best-effort print a small set of Maxwell assay metadata.

    This is debug/logging only: failure to read metadata should not stop the pipeline.
    """

    try:
        import h5py
    except Exception:
        print(f"{prefix} assay settings: h5py not available", flush=True)
        return

    h5_path = Path(h5_path).expanduser().resolve()
    try:
        with h5py.File(h5_path, "r") as h5:
            if "assay" not in h5:
                print(f"{prefix} assay settings: no /assay group", flush=True)
                return
            assay = h5["assay"]
            keys = list(assay.keys())
            print(f"{prefix} assay settings: /assay keys={keys}", flush=True)
    except Exception as e:
        print(f"{prefix} assay settings: failed to read: {e}", flush=True)


def _print_data_store_start_stop_durations(
    *,
    h5_path: Path,
    target_stream_id: Optional[str] = None,
    prefix: str = "[axon_reconstructor]",
) -> None:
    """Print start/stop/duration for `/data_store` entries.

    Maxwell exports sometimes include one or more timing blocks in `/data_store`.
    This is used for debugging segment-time alignment, but should not be required
    for processing.
    """

    try:
        import h5py
        import numpy as np
    except Exception:
        print(f"{prefix} data_store: h5py/numpy not available", flush=True)
        return

    h5_path = Path(h5_path).expanduser().resolve()
    try:
        with h5py.File(h5_path, "r") as h5:
            if "data_store" not in h5:
                print(f"{prefix} data_store: no /data_store group", flush=True)
                return

            ds = h5["data_store"]
            stream_ids = sorted(ds.keys())
            if target_stream_id is not None:
                stream_ids = [s for s in stream_ids if str(s) == str(target_stream_id)]
                if not stream_ids:
                    print(
                        f"{prefix} data_store: target_stream_id={target_stream_id} not found", flush=True
                    )
                    return

            for stream_id in stream_ids:
                stream = ds[str(stream_id)]
                cfg_names = sorted(stream.keys())
                for cfg_name in cfg_names:
                    cfg = stream[str(cfg_name)]

                    def _read_ms(name: str) -> Optional[int]:
                        if name not in cfg:
                            return None
                        try:
                            return int(np.asarray(cfg[name][()]).ravel()[0])
                        except Exception:
                            return None

                    start_ms = _read_ms("start_time")
                    stop_ms = _read_ms("stop_time")
                    if start_ms is None or stop_ms is None:
                        print(
                            f"{prefix} data_store: stream={stream_id} cfg={cfg_name} start/stop unavailable",
                            flush=True,
                        )
                        continue

                    dur_s = (stop_ms - start_ms) / 1000.0
                    print(
                        f"{prefix} data_store: stream={stream_id} cfg={cfg_name} "
                        f"start_ms={start_ms} stop_ms={stop_ms} dur_s={dur_s:.3f}",
                        flush=True,
                    )
    except Exception as e:
        print(f"{prefix} data_store: failed to read: {e}", flush=True)


def _read_well_rec_frame_nos_and_trigger_settings(
    *,
    h5_path: Path,
    stream_id: str,
    rec_name: str,
):
    """Read per-recording timing info from `/wells/<stream>/<rec>`.

    This is used to reconcile the apparent mismatch between:
    - wall-clock timing (`start_time`/`stop_time` epoch timestamps)
    - stored sample count (`groups/routed/raw` length)

    Many Maxwell exports store *triggered* snippets: a small number of frames
    around each trigger, spread across a longer wall-clock span. In that case
    `frame_nos` spans ~`record_time * fs`, while the stored sample count is much
    smaller.
    """

    try:
        import h5py
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("reading well timing requires h5py/numpy") from e

    h5_path = Path(h5_path).expanduser().resolve()
    with h5py.File(h5_path, "r") as h5:
        rec = h5["wells"][str(stream_id)][str(rec_name)]

        # These are typically 1-element arrays.
        start_raw = rec["start_time"][()]
        stop_raw = rec["stop_time"][()]
        start_ms = int(np.asarray(start_raw).ravel()[0])
        stop_ms = int(np.asarray(stop_raw).ravel()[0])

        routed = rec["groups"]["routed"]
        frame_nos = np.asarray(routed["frame_nos"], dtype=np.int64)

        def _read_int_1(name: str) -> Optional[int]:
            if name not in routed:
                return None
            try:
                return int(np.asarray(routed[name][()]).ravel()[0])
            except Exception:
                return None

        def _read_float_1(name: str) -> Optional[float]:
            if name not in routed:
                return None
            try:
                return float(np.asarray(routed[name][()]).ravel()[0])
            except Exception:
                return None

        triggered = _read_int_1("triggered")
        trigger_pre = _read_int_1("trigger_pre")
        trigger_post = _read_int_1("trigger_post")
        trigger_minamp = _read_float_1("trigger_minamp")
        trigger_maxamp = _read_float_1("trigger_maxamp")

    return {
        "start_ms": start_ms,
        "stop_ms": stop_ms,
        "frame_nos": frame_nos,
        "triggered": triggered,
        "trigger_pre": trigger_pre,
        "trigger_post": trigger_post,
        "trigger_minamp": trigger_minamp,
        "trigger_maxamp": trigger_maxamp,
    }


@contextlib.contextmanager
def _tee_stdout_to_file(out_path: Path):
    """Write stdout to both terminal and a file for debug logs."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _STDOUT_TEE_LOCK, open(out_path, "w", encoding="utf-8") as f:

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
