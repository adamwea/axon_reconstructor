from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
import time
from typing import Optional

from .utils import _ensure_maxwell_hdf5_plugin_path

from .planning import RawPreprocessPlan, build_preprocess_plan, discover_cfg_files, parse_cfg_channel_locations

from .concatenation import (
    find_common_electrodes_from_segments,
    _load_centered_segment_with_electrode_channel_ids,
)
from .preprocessing import apply_standard_preprocessing

from .h5_helpers import (
    _read_well_rec_frame_nos_and_trigger_settings,
    _tee_stdout_to_file,
    _print_assay_settings,
    _print_data_store_start_stop_durations,
)

from .plotting import (
    _activity_score_rms,
    _extract_xy_from_contact_vector,
    _pick_representative_index_for_cluster,
    _plot_concat_cluster_traces,
    _save_channel_layout_plots,
    detect_electrode_clusters,
)


def build_concatenated_recording(
    *,
    h5_path: Path,
    stream_id: str,
    n_jobs: int = 8,
    center_chunk_size: int = 10_000,
    temporal_resample_factor: Optional[int] = None,
    temporal_resample_rate_hz: Optional[int] = None,
    temporal_resample_margin_ms: float = 100.0,
    temporal_resample_dtype: Optional[str] = None,
    plot_output_dir: Optional[Path] = None,
    plot_segment_traces: bool = True,
    epoch_markers_output_dir: Optional[Path] = None,
    return_artifacts: bool = False,
) -> tuple[object, list[int]] | tuple[object, list[int], dict[str, object]]:
    """Load per-segment recordings, center + preprocess full segments, then slice for concat.

    Returns `(multirecording, common_electrodes)`.
    """

    try:
        import numpy as np
        import spikeinterface.full as si
        import spikeinterface.extractors as se
        import h5py
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "raw preprocessing requires `numpy`, `h5py`, and `spikeinterface` installed"
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

    with h5py.File(h5_path, "r") as h5:
        rec_names = list(h5["wells"][stream_id].keys())

    if not rec_names:
        raise RuntimeError(f"No recording segments found under /wells/{stream_id} in {h5_path}")

    is_single_segment = len(rec_names) == 1
    common_el: list[int] = []

    if is_single_segment:
        rec_name = rec_names[0]
        print(
            f"[axon_reconstructor] single-segment recording detected ({rec_name}); "
            "skipping cross-segment channel intersection/slicing and concatenation",
            flush=True,
        )
        rec_single, seg_stat = _load_centered_segment_with_electrode_channel_ids(
            h5_path=h5_path,
            stream_id=stream_id,
            rec_name=rec_name,
            center_chunk_size=center_chunk_size,
        )

        common_el = [int(e) for e in np.asarray(rec_single.get_channel_ids(), dtype=int).tolist()]
        rec_list_full = [rec_single]
        seg_stats = [seg_stat]
    else:
        rec_names, common_el = find_common_electrodes_from_segments(h5_path=h5_path, stream_id=stream_id)
        print(
            f"[axon_reconstructor] found {len(rec_names)} segments; shared electrodes={len(common_el)}; "
            f"intersection took {time.perf_counter() - t0:.2f}s",
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
                _load_centered_segment_with_electrode_channel_ids,
                h5_path=h5_path,
                stream_id=stream_id,
                center_chunk_size=center_chunk_size,
            )
            results = list(ex.map(lambda rn: process(rec_name=rn), rec_names))

        rec_list_full = [r for r, _ in results]
        seg_stats = [s for _, s in results]

        print(
            f"[axon_reconstructor] segment preprocessing done in {time.perf_counter() - t_segments:.2f}s "
            f"(n_jobs={n_jobs}, workers={max_workers})",
            flush=True,
        )

    if plot_output_dir is not None:
        plot_output_dir = Path(plot_output_dir)
        if is_single_segment:
            print(
                "[axon_reconstructor] single-segment recording: skipping channel-layout summary plots",
                flush=True,
            )
        else:
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

    # Apply preprocessing parity before concatenation so both concat and saved segments
    # can be derived from identically preprocessed segment recordings.
    pre_t0 = time.perf_counter()
    preprocessed_rec_list_full = [apply_standard_preprocessing(recording=rec) for rec in rec_list_full]

    if is_single_segment:
        preprocessed_rec_list_concat = preprocessed_rec_list_full
    else:
        preprocessed_rec_list_concat = [
            seg_rec.select_channels([int(el) for el in common_el])
            for seg_rec in preprocessed_rec_list_full
        ]

        # Validate that selected channels remain aligned with the shared electrode set.
        for rn, seg_rec in zip(rec_names, preprocessed_rec_list_concat, strict=False):
            seg_ch = np.asarray(seg_rec.get_channel_ids(), dtype=int)
            exp_ch = np.asarray(common_el, dtype=int)
            if seg_ch.shape != exp_ch.shape or not np.array_equal(seg_ch, exp_ch):
                raise RuntimeError(
                    f"Post-preprocess common-channel selection mismatch for segment {rn}; "
                    "refusing to concatenate potentially misaligned channels."
                )

    print(
        f"[axon_reconstructor] MEA-style preprocessing applied to {len(preprocessed_rec_list_full)} segment recording(s) "
        f"in {time.perf_counter() - pre_t0:.2f}s",
        flush=True,
    )

    t_concat = time.perf_counter()
    if is_single_segment:
        multirecording = preprocessed_rec_list_concat[0]
    else:
        multirecording = si.concatenate_recordings(preprocessed_rec_list_concat)

    # Precompute segment stitch epochs in concatenated sample coordinates.
    seg_lengths = [int(r.get_num_samples()) for r in preprocessed_rec_list_concat]
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
    concat_times = None
    maxwell_epochs: list[dict] = []
    try:
        import numpy as np

        time_vectors: list[np.ndarray] = []
        t0_epoch_s: Optional[float] = None

        # Epochs of contiguous samples within Maxwell triggered/snippet recording strategy.
        # Expressed in *concatenated* sample coordinates (multirecording time axis).

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
                    preprocessed_rec_list_full[seg_index].set_times(times_rel.astype(float, copy=False))
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

    # Optional: temporal resampling (e.g. 10x) to emulate higher sampling rate.
    if temporal_resample_rate_hz is not None or temporal_resample_factor is not None:
        try:
            import numpy as np
            import spikeinterface.preprocessing as spre

            old_fs = float(multirecording.get_sampling_frequency())
            if old_fs <= 0:
                raise RuntimeError("Could not determine recording sampling frequency")

            if temporal_resample_rate_hz is not None:
                new_fs = float(int(temporal_resample_rate_hz))
            else:
                f = int(temporal_resample_factor or 0)
                if f < 2:
                    raise ValueError(f"temporal_resample_factor must be >=2, got {f}")
                new_fs = float(old_fs) * float(f)

            new_fs_int = int(round(new_fs))
            if new_fs_int <= 0:
                raise ValueError(f"Invalid target resample rate: {new_fs_int}")

            ratio = float(new_fs_int) / float(old_fs)
            print(
                f"[axon_reconstructor] temporal resample: fs {old_fs:.2f} -> {new_fs_int} Hz (x{ratio:.4f})",
                flush=True,
            )

            dtype = None
            if temporal_resample_dtype is not None:
                try:
                    dtype = np.dtype(str(temporal_resample_dtype))
                except Exception:
                    dtype = None

            multirecording = spre.resample(
                multirecording,
                resample_rate=int(new_fs_int),
                margin_ms=float(temporal_resample_margin_ms),
                dtype=dtype,
                skip_checks=False,
            )

            # Scale epoch markers to match the resampled sample coordinates.
            def _scale_epoch(ep: dict) -> dict:
                out = dict(ep)
                for k in ("start_sample", "end_sample", "segment_start_sample", "segment_end_sample"):
                    if k in out and out[k] is not None:
                        out[k] = int(round(float(out[k]) * ratio))
                if "n_samples" in out and out["n_samples"] is not None:
                    out["n_samples"] = int(round(float(out["n_samples"]) * ratio))
                return out

            concat_epochs = [_scale_epoch(ep) for ep in concat_epochs]
            maxwell_epochs = [_scale_epoch(ep) for ep in maxwell_epochs]

            # Fail fast if epoch indices drift out of bounds after resampling.
            n_new = int(multirecording.get_num_samples())
            if n_new <= 0:
                raise RuntimeError("Resampled recording has non-positive sample count")

            def _max_end_sample(epochs: list[dict]) -> int:
                ends: list[int] = []
                for e in epochs:
                    try:
                        end = e.get("end_sample")
                        if end is None:
                            continue
                        ends.append(int(end))
                    except Exception:
                        continue
                return max(ends) if ends else 0

            maxwell_max_end = _max_end_sample(maxwell_epochs)
            concat_max_end = _max_end_sample(concat_epochs)
            if maxwell_max_end > n_new or concat_max_end > n_new:
                raise RuntimeError(
                    "Epoch markers exceed resampled recording length: "
                    f"n_samples={n_new} maxwell_max_end={maxwell_max_end} concat_max_end={concat_max_end} "
                    f"(ratio={ratio:.6f})"
                )

            # Re-interpolate concat time vector to match new length, if available.
            if concat_times is not None:
                n_old = int(concat_times.size)
                n_new = int(multirecording.get_num_samples())
                if n_old > 1 and n_new > 1:
                    x_old = np.arange(n_old, dtype=float)
                    x_new = np.linspace(0.0, float(n_old - 1), num=n_new, dtype=float)
                    t_new = np.interp(x_new, x_old, np.asarray(concat_times, dtype=float))
                    if t_new.size == n_new:
                        try:
                            multirecording.set_times(t_new)
                        except Exception:
                            pass
        except Exception as e:
            raise RuntimeError(f"Temporal resampling failed: {e}") from e

    # Persist epoch marker JSON artifacts for later analysis.
    if epoch_markers_output_dir is not None:
        out_dir = Path(epoch_markers_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        maxwell_path = out_dir / f"maxwell_contiguous_epochs_{stream_id}.json"
        concat_path = out_dir / f"concatenation_stitch_epochs_{stream_id}.json"

        try:
            with open(maxwell_path, "w", encoding="utf-8") as f:
                json.dump(list(maxwell_epochs), f, indent=2)

            if is_single_segment:
                print(
                    f"[axon_reconstructor] wrote epoch markers: maxwell={maxwell_path.name} "
                    "(single-segment; no concatenation_stitch_epochs file)",
                    flush=True,
                )
            else:
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

    record_kind = "single-segment recording" if is_single_segment else "concatenated recording"
    timing_label = "prepare took" if is_single_segment else "concat took"
    print(
        f"[axon_reconstructor] {record_kind}: fs={fs_cat:.2f} Hz, "
        f"samples={n_cat:,}, duration_samples/fs={dur_cat:.2f} s"
        + (
            f", duration_time_vector={dur_timevec:.2f} s" if dur_timevec is not None else ""
        )
        + f", channels={int(multirecording.get_num_channels())} "
        f"({timing_label} {time.perf_counter() - t_concat:.2f}s)",
        flush=True,
    )

    if plot_output_dir is not None and not is_single_segment:
        # Plot concatenation diagnostics: cluster reps over time + stitch markers.
        # We derive stitch frames from `concat_epochs` so they remain correct after
        # optional temporal resampling (which scales sample indices).
        plot_output_dir = Path(plot_output_dir)
        segment_traces_dir = plot_output_dir / "segment_traces"
        if bool(plot_segment_traces):
            segment_traces_dir.mkdir(parents=True, exist_ok=True)
        stitch_frames: list[int] = []
        try:
            # Stitch points are the start of each segment after the first.
            stitch_frames = [int(ep["start_sample"]) for ep in concat_epochs[1:]]
        except Exception:
            # Fallback: derive from (pre-resample) segment lengths.
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
            if bool(plot_segment_traces):
                for rn, seg_rec in zip(rec_names, preprocessed_rec_list_full, strict=False):
                    _plot_concat_cluster_traces(
                        recording=seg_rec,
                        channel_ids=rep_keep,
                        stitch_frames=[],
                        out_path=segment_traces_dir / f"segment_trace_{stream_id}_{rn}.png",
                        title=f"Segment trace ({stream_id} / {rn})",
                    )
            else:
                print("[axon_reconstructor] segment trace plots disabled by config", flush=True)
            
        except Exception:
            # Don't fail preprocessing if plotting diagnostics can't be generated.
            pass

    if bool(return_artifacts):
        artifacts: dict[str, object] = {
            "rec_names": [str(rn) for rn in rec_names],
            "segment_recordings_preprocessed": preprocessed_rec_list_full,
            "segment_recordings_preprocessed_concat": preprocessed_rec_list_concat,
            "segment_stats": seg_stats,
        }
        return multirecording, common_el, artifacts

    return multirecording, common_el
