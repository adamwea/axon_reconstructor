from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from .pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from .pipeline_driver import PREPROCESS_OUTPUTS_DIRNAME, _compute_mea_analysis_output_dir
from .raw_preprocessing.raw_preprocessing import _ensure_maxwell_hdf5_plugin_path


WAVEFORMS_OUTPUTS_DIRNAME = "waveforms_outputs"


def _compute_waveforms_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    """Use a dedicated checkpoint file for waveforms.

    The MEA_Analysis-style stage machine (PREPROCESSING..REPORTS_COMPLETE) doesn't
    include waveforms, so storing waveforms progress in the main checkpoint can
    accidentally *regress* stage numbers (e.g. REPORTS_COMPLETE -> ANALYZER_COMPLETE).
    """

    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_waveforms_checkpoint.json"
    else:
        name = main_ckpt.stem + "_waveforms_checkpoint.json"
    return main_ckpt.with_name(name)


@dataclass(frozen=True)
class WaveformExtractInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path
    sorter: str = "kilosort4"

    # Waveform window. If None, try to infer from trigger_pre/post.
    ms_before: Optional[float] = None
    ms_after: Optional[float] = None

    n_jobs: int = 8
    max_spikes_per_unit: Optional[int] = None

    # If True, also extract waveforms per concatenated segment.
    per_segment: bool = True

    # Resume/overwrite controls
    force_restart: bool = False

    # If True, drop spikes whose waveform window would cross Maxwell snippet boundaries.
    filter_by_maxwell_epochs: bool = True


@dataclass(frozen=True)
class WaveformExtractOutputs:
    well_out_dir: Path
    waveforms_out_dir: Path
    concat_waveforms_dir: Path
    segment_waveforms_dir: Optional[Path]
    params_json: Path
    filtering_json: Path


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _infer_cutout_ms(*, h5_path: Path, stream_id: str, fs_hz: float) -> tuple[float, float]:
    """Infer ms_before/ms_after from trigger_pre/trigger_post when available."""

    try:
        from .raw_preprocessing.h5_helpers import _read_well_rec_frame_nos_and_trigger_settings

        # Grab settings from the first rec in the stream.
        import h5py  # type: ignore[import-not-found]

        with h5py.File(h5_path, "r") as h5:
            rec_name = list(h5["wells"][stream_id].keys())[0]

        info = _read_well_rec_frame_nos_and_trigger_settings(
            h5_path=h5_path,
            stream_id=stream_id,
            rec_name=rec_name,
        )
        pre = info.get("trigger_pre")
        post = info.get("trigger_post")
        if pre is None or post is None:
            raise ValueError("trigger_pre/post missing")

        # trigger_pre/post are in samples.
        ms_before = float(pre) / (fs_hz / 1000.0)
        ms_after = float(post) / (fs_hz / 1000.0)

        # Guard against nonsense.
        if not (0 < ms_before < 50 and 0 < ms_after < 50):
            raise ValueError(f"unexpected cutout ms: {ms_before}, {ms_after}")

        return ms_before, ms_after
    except Exception:
        # Fall back to Mandar defaults.
        return 1.0, 2.0


def _compute_waveforms_out_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
    return _compute_mea_analysis_output_dir(output_root=output_root, data_file=data_file, well=well) / WAVEFORMS_OUTPUTS_DIRNAME


def _load_preprocessed_recording(*, well_out_dir: Path) -> Any:
    recording_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME / "preprocessed_recording"
    if not recording_dir.exists():
        raise FileNotFoundError(f"preprocessed_recording not found: {recording_dir}")

    import spikeinterface.full as si  # type: ignore[import-not-found]

    try:
        return si.load(recording_dir)
    except Exception:
        return si.load_extractor(recording_dir)


def _load_raw_segment_recording_full_channels(
    *,
    h5_path: Path,
    stream_id: str,
    rec_name: str,
    center_chunk_size: int = 10_000,
) -> Any:
    """Load a *single* raw Maxwell rec segment and keep its full channel set.

    This is the key difference vs. the concatenated recording:
    concatenation slices to the shared electrode intersection, which drops
    channels that are not present in every segment.

    In other words:
    - concat recording: fewer channels (intersection), but a single continuous time axis
    - raw segment recording: many more channels, but only one segment worth of time
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import spikeinterface.full as si  # type: ignore[import-not-found]
        import spikeinterface.extractors as se  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("waveform extraction requires spikeinterface/numpy installed") from e

    _ensure_maxwell_hdf5_plugin_path()

    if hasattr(se, "read_maxwell"):
        rec = se.read_maxwell(file_path=str(h5_path), stream_id=stream_id, rec_name=rec_name)
    else:  # pragma: no cover
        rec = se.MaxwellRecordingExtractor(str(h5_path), stream_id=stream_id, rec_name=rec_name)

    chunk = min(center_chunk_size, int(rec.get_num_samples())) - 100
    chunk = max(int(chunk), 100)
    rec_centered = si.center(rec, chunk_size=int(chunk))

    # Rename channel_ids to electrode ids for identity stability.
    # This mirrors how preprocessing renames the *common* channels before concatenation,
    # but here we do it for the full channel set so downstream waveforms are tied to
    # physical electrode ids when possible.
    try:
        electrodes = np.asarray(rec_centered.get_property("contact_vector")["electrode"], dtype=int)
        if int(np.unique(electrodes).size) != int(electrodes.size):
            raise RuntimeError(f"Duplicate electrode ids in contact_vector for rec={rec_name}")
        rec_centered = rec_centered.rename_channels([int(e) for e in electrodes])
    except Exception:
        # Best-effort: proceed without renaming.
        pass

    return rec_centered


def _resolve_mea_sorter_output_dir(*, well_out_dir: Path) -> Path:
    # New step-style layout: spikesorting_outputs/sorter_output
    p = well_out_dir / "spikesorting_outputs" / "sorter_output"
    if p.exists():
        return p

    # Legacy fallback.
    legacy = well_out_dir / "sorter_output"
    if legacy.exists():
        return legacy

    return p


def _load_sorting_from_sorter_output_dir(*, sorter_output_dir: Path, sorter: str) -> Any:
    import spikeinterface.full as si  # type: ignore[import-not-found]

    # Prefer the generic entry point when available.
    if hasattr(si, "read_sorter_folder"):
        try:
            return si.read_sorter_folder(sorter_output_dir, sorter_name=sorter)
        except TypeError:
            # Older SpikeInterface versions use a positional arg (no keyword).
            return si.read_sorter_folder(sorter_output_dir, sorter)

    # Fallbacks: try to load as an extractor.
    try:
        return si.load_extractor(sorter_output_dir)
    except Exception:
        pass

    raise RuntimeError(
        f"Could not load sorting from sorter_output_dir={sorter_output_dir} (sorter={sorter})."
    )


def _epochs_to_intervals(epochs: list[dict]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for e in epochs:
        try:
            start = int(e["start_sample"])
            end = int(e["end_sample"])
        except Exception:
            continue
        if end > start:
            intervals.append((start, end))
    intervals.sort()
    return intervals


def _filter_spike_train_by_intervals(
    *,
    spike_train: list[int],
    intervals: list[tuple[int, int]],
    pre_samples: int,
    post_samples: int,
) -> tuple[list[int], int]:
    """Keep spikes whose cutout window stays within some interval."""

    if not intervals:
        return spike_train, 0

    kept: list[int] = []
    removed = 0

    # Two-pointer scan because both spike_train and intervals are sorted.
    i = 0
    for t in spike_train:
        t0 = int(t) - int(pre_samples)
        t1 = int(t) + int(post_samples)

        while i < len(intervals) and intervals[i][1] <= t0:
            i += 1

        ok = False
        if i < len(intervals):
            start, end = intervals[i]
            if t0 >= start and t1 < end:
                ok = True

        if ok:
            kept.append(int(t))
        else:
            removed += 1

    return kept, removed


def _to_numpy_sorting(*, unit_trains: dict[int, list[int]], fs_hz: float) -> Any:
    import numpy as np  # type: ignore[import-not-found]
    from spikeinterface.core import NumpySorting  # type: ignore[import-not-found]

    unit_ids = sorted(unit_trains.keys())
    all_times: list[int] = []
    all_labels: list[int] = []
    for u in unit_ids:
        times = unit_trains[u]
        all_times.extend(times)
        all_labels.extend([u] * len(times))

    if not all_times:
        # Empty sorting: still return a valid object with unit_ids.
        sorting = NumpySorting.from_unit_dict(unit_trains, sampling_frequency=float(fs_hz))
        return sorting

    times_arr = np.asarray(all_times, dtype=np.int64)
    labels_arr = np.asarray(all_labels, dtype=np.int64)

    # SpikeInterface expects sorted times within segment.
    order = np.argsort(times_arr, kind="mergesort")
    times_arr = times_arr[order]
    labels_arr = labels_arr[order]

    sorting = NumpySorting.from_times_labels(
        times_list=[times_arr],
        labels_list=[labels_arr],
        sampling_frequency=float(fs_hz),
    )
    return sorting


def extract_waveforms(
    *,
    inputs: WaveformExtractInputs,
    logger_name_prefix: str = "axon_reconstructor",
) -> WaveformExtractOutputs:
    """Extract waveforms from the preprocessed recording and sorter output.

    Produces:
      <well>/waveforms_outputs/concat_waveforms/
      <well>/waveforms_outputs/segment_waveforms/ (optional)
      plus JSON summaries.

    Uses existing epoch marker JSONs (from preprocessing) to avoid extracting
    waveforms that cross Maxwell snippet discontinuities.
    """

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=inputs.h5_path, stream_id=inputs.stream_id)
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}",
        verbose=True,
    )

    waveforms_out_dir = _compute_waveforms_out_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )
    waveforms_out_dir.mkdir(parents=True, exist_ok=True)

    concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"
    segment_waveforms_dir = waveforms_out_dir / "segment_waveforms" if inputs.per_segment else None

    params_json = waveforms_out_dir / "waveform_extraction_params.json"
    filtering_json = waveforms_out_dir / "waveform_filtering_summary.json"

    ckpt_file = _compute_waveforms_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # Resume shortcut: trust existing artifacts if present.
    if not inputs.force_restart and concat_waveforms_dir.exists():
        logger.info("Resuming waveforms: existing outputs found at %s", concat_waveforms_dir)
        return WaveformExtractOutputs(
            well_out_dir=well_out_dir,
            waveforms_out_dir=waveforms_out_dir,
            concat_waveforms_dir=concat_waveforms_dir,
            segment_waveforms_dir=segment_waveforms_dir,
            params_json=params_json,
            filtering_json=filtering_json,
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={
            "waveforms_out_dir": str(waveforms_out_dir),
        },
    )

    logger.info("Waveform extraction starting: well_out_dir=%s", well_out_dir)

    try:
        recording = _load_preprocessed_recording(well_out_dir=well_out_dir)
        fs_hz = float(recording.get_sampling_frequency())

        ms_before = float(inputs.ms_before) if inputs.ms_before is not None else None
        ms_after = float(inputs.ms_after) if inputs.ms_after is not None else None
        if ms_before is None or ms_after is None:
            inferred_before, inferred_after = _infer_cutout_ms(h5_path=inputs.h5_path, stream_id=inputs.stream_id, fs_hz=fs_hz)
            ms_before = inferred_before if ms_before is None else ms_before
            ms_after = inferred_after if ms_after is None else ms_after

        pre_samples = int(math.ceil(ms_before * fs_hz / 1000.0))
        post_samples = int(math.ceil(ms_after * fs_hz / 1000.0))

        sorter_output_dir = _resolve_mea_sorter_output_dir(well_out_dir=well_out_dir)
        sorting = _load_sorting_from_sorter_output_dir(sorter_output_dir=sorter_output_dir, sorter=inputs.sorter)

        # Load epoch markers from preprocessing.
        preprocess_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
        maxwell_epochs_path = preprocess_dir / f"maxwell_contiguous_epochs_{inputs.stream_id}.json"
        concat_epochs_path = preprocess_dir / f"concatenation_stitch_epochs_{inputs.stream_id}.json"

        maxwell_intervals: list[tuple[int, int]] = []
        concat_epochs: list[dict] = []

        if maxwell_epochs_path.exists():
            maxwell_epochs = list(_read_json(maxwell_epochs_path))
            maxwell_intervals = _epochs_to_intervals(maxwell_epochs)
        if concat_epochs_path.exists():
            concat_epochs = list(_read_json(concat_epochs_path))

        # Filter spikes by Maxwell epochs (avoid snippet boundary crossings).
        filtered_sorting = sorting
        filtering_summary: dict[str, Any] = {
            "filter_by_maxwell_epochs": bool(inputs.filter_by_maxwell_epochs),
            "ms_before": ms_before,
            "ms_after": ms_after,
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "maxwell_epochs_path": str(maxwell_epochs_path) if maxwell_epochs_path.exists() else None,
            "concat_epochs_path": str(concat_epochs_path) if concat_epochs_path.exists() else None,
            "removed_spikes_total": 0,
            "kept_spikes_total": 0,
        }

        if inputs.filter_by_maxwell_epochs and maxwell_intervals:
            unit_trains: dict[int, list[int]] = {}
            removed_total = 0
            kept_total = 0

            unit_ids = list(sorting.get_unit_ids())
            for u in unit_ids:
                st = sorting.get_unit_spike_train(u)
                st_list = [int(x) for x in st]
                st_list.sort()

                kept, removed = _filter_spike_train_by_intervals(
                    spike_train=st_list,
                    intervals=maxwell_intervals,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                )
                unit_trains[int(u)] = kept
                removed_total += removed
                kept_total += len(kept)

            filtered_sorting = _to_numpy_sorting(unit_trains=unit_trains, fs_hz=fs_hz)
            filtering_summary["removed_spikes_total"] = int(removed_total)
            filtering_summary["kept_spikes_total"] = int(kept_total)
            logger.info(
                "Filtered spikes by Maxwell epochs: kept=%d removed=%d (pre=%d post=%d samples)",
                kept_total,
                removed_total,
                pre_samples,
                post_samples,
            )
        else:
            # Best-effort counts.
            try:
                n_total = sum(len(sorting.get_unit_spike_train(u)) for u in sorting.get_unit_ids())
                filtering_summary["kept_spikes_total"] = int(n_total)
            except Exception:
                pass

        _write_json(params_json, {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "sorter": inputs.sorter,
            "sorter_output_dir": str(sorter_output_dir),
            "ms_before": ms_before,
            "ms_after": ms_after,
            "n_jobs": int(inputs.n_jobs),
            "max_spikes_per_unit": inputs.max_spikes_per_unit,
            "per_segment": bool(inputs.per_segment),
            "per_segment_recording_source": "raw_maxwell_full_channels" if inputs.per_segment else None,
        })
        _write_json(filtering_json, filtering_summary)

        # Extract concatenated waveforms.
        import shutil
        import spikeinterface.full as si  # type: ignore[import-not-found]

        if concat_waveforms_dir.exists() and inputs.force_restart:
            shutil.rmtree(concat_waveforms_dir)

        logger.info("Extracting concat waveforms -> %s", concat_waveforms_dir)
        si.extract_waveforms(
            recording=recording,
            sorting=filtered_sorting,
            folder=concat_waveforms_dir,
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            max_spikes_per_unit=inputs.max_spikes_per_unit,
            return_scaled=True,
            allow_unfiltered=True,
            overwrite=None,
            n_jobs=int(inputs.n_jobs),
            progress_bar=False,
        )

        # Optional: per-segment extraction using concatenation epochs.
        if inputs.per_segment and concat_epochs:
            assert segment_waveforms_dir is not None
            segment_waveforms_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Extracting per-segment waveforms -> %s", segment_waveforms_dir)
            for seg in concat_epochs:
                try:
                    seg_index = int(seg["segment_index"])
                    rec_name = str(seg.get("rec_name", f"seg{seg_index}"))
                    start = int(seg["start_sample"])
                    end = int(seg["end_sample"])
                except Exception:
                    continue

                seg_dir = segment_waveforms_dir / f"seg{seg_index:02d}_{rec_name}"
                if seg_dir.exists() and inputs.force_restart:
                    shutil.rmtree(seg_dir)

                # STEP 1) Load the *raw* segment recording with its full channel set.
                #
                # This is the point of the per-segment extraction: the concatenated
                # recording (used for sorting) only contains electrodes shared across
                # all segments. Any electrodes present in *some* segments but not all
                # are intentionally dropped during concatenation.
                #
                # By extracting waveforms on the raw segment, we can recover waveforms
                # on those dropped electrodes for this segment.
                seg_rec = _load_raw_segment_recording_full_channels(
                    h5_path=inputs.h5_path,
                    stream_id=inputs.stream_id,
                    rec_name=rec_name,
                    center_chunk_size=10_000,
                )

                # Sanity: concat_epochs were computed from the preprocessed segments.
                # Channel slicing does not change sample count, so this should match.
                seg_len_expected = int(end - start)
                try:
                    seg_len = int(seg_rec.get_num_samples())
                    if seg_len != seg_len_expected:
                        logger.warning(
                            "Segment length mismatch for %s: raw=%d expected=%d (start=%d end=%d). Proceeding.",
                            rec_name,
                            seg_len,
                            seg_len_expected,
                            start,
                            end,
                        )
                except Exception:
                    seg_len = seg_len_expected

                try:
                    logger.info(
                        "Segment %s: raw channels=%d (concat channels=%d)",
                        rec_name,
                        int(seg_rec.get_num_channels()),
                        int(recording.get_num_channels()),
                    )
                except Exception:
                    pass

                # STEP 2) Build a segment-local sorting from the *concatenated* sorting.
                #
                # Conceptually:
                # - `filtered_sorting` spike times are in the concatenated time base.
                # - `seg_rec` is a single raw recording segment with its own 0..N time base.
                # So we:
                #   - select spikes that fall inside this concatenated segment window [start, end)
                #   - shift them to segment-local coordinates by subtracting `start`
                #   - drop spikes too close to segment edges so waveform windows fit.
                unit_trains_seg: dict[int, list[int]] = {}
                for u in filtered_sorting.get_unit_ids():
                    st = filtered_sorting.get_unit_spike_train(u)
                    st_list = [int(x) for x in st]

                    # Keep only spikes in the segment, shift to segment-local coordinates.
                    # Also drop spikes too close to segment edges for waveform windows.
                    kept: list[int] = []
                    for t in st_list:
                        if t < start or t >= end:
                            continue
                        t_local = int(t - start)
                        if t_local - pre_samples < 0:
                            continue
                        if t_local + post_samples >= int(seg_len):
                            continue
                        kept.append(t_local)
                    unit_trains_seg[int(u)] = kept

                # Convert the segment-local spike trains into a SortingExtractor.
                # This is the bridge that lets us use the holistic concat sorting
                # to drive waveform extraction on each raw segment.
                seg_sort = _to_numpy_sorting(unit_trains=unit_trains_seg, fs_hz=fs_hz)

                # STEP 3) Register the sorting to the raw segment recording.
                #
                # Not strictly required for `si.extract_waveforms(recording=..., sorting=...)`,
                # but this matches the intent/semantics of the legacy pipeline and helps
                # downstream utilities that expect the sorting to “know” its recording.
                try:
                    seg_sort.register_recording(seg_rec)
                except Exception:
                    pass

                # STEP 4) Extract waveforms on the raw segment.
                #
                # Key outcome: these waveforms include channels that are absent from the
                # concatenated recording (i.e. the non-shared electrodes), which is the
                # whole reason we do this per-segment pass.
                si.extract_waveforms(
                    recording=seg_rec,
                    sorting=seg_sort,
                    folder=seg_dir,
                    ms_before=float(ms_before),
                    ms_after=float(ms_after),
                    max_spikes_per_unit=inputs.max_spikes_per_unit,
                    return_scaled=True,
                    allow_unfiltered=True,
                    overwrite=None,
                    n_jobs=max(1, int(inputs.n_jobs)),
                    progress_bar=False,
                )

        ckpt = save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "waveforms_out_dir": str(waveforms_out_dir),
                "concat_waveforms_dir": str(concat_waveforms_dir),
                "segment_waveforms_dir": str(segment_waveforms_dir) if segment_waveforms_dir else None,
                "waveforms_params_json": str(params_json),
                "waveforms_filtering_json": str(filtering_json),
            },
        )

        logger.info("Waveform extraction complete")

        return WaveformExtractOutputs(
            well_out_dir=well_out_dir,
            waveforms_out_dir=waveforms_out_dir,
            concat_waveforms_dir=concat_waveforms_dir,
            segment_waveforms_dir=segment_waveforms_dir,
            params_json=params_json,
            filtering_json=filtering_json,
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage="WAVEFORMS",
            error=exception_to_error_dict(e),
        )
        logger.exception("Waveform extraction FAILED")
        raise
