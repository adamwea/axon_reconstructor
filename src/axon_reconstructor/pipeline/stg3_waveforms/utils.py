from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

from ..stg1_preprocessing.constants import PREPROCESS_OUTPUTS_DIRNAME
from ..stg1_preprocessing.preprocessing import apply_standard_preprocessing
from ..stg1_preprocessing.utils import _ensure_maxwell_hdf5_plugin_path

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_preprocess_requested_cfg(*, preprocess_dir: Path) -> dict[str, Any] | None:
    """Load preprocessing config written by Stage 01 (best effort).

    This is used to ensure downstream stages (especially per-segment waveforms)
    can reproduce temporal resampling settings when the concatenated recording
    was upsampled during preprocessing.
    """

    preprocess_dir = Path(preprocess_dir)
    cfg_path = preprocess_dir / "preprocess_config.json"
    if not cfg_path.exists():
        return None

    try:
        payload = _read_json(cfg_path)
        if isinstance(payload, dict):
            requested = payload.get("requested_cfg")
            return requested if isinstance(requested, dict) else None
    except Exception:
        return None

    return None


def _infer_cutout_ms(*, h5_path: Path, stream_id: str, fs_hz: float) -> tuple[float, float]:
    """Infer ms_before/ms_after from trigger_pre/trigger_post when available."""

    try:
        from ..stg1_preprocessing.h5_helpers import _read_well_rec_frame_nos_and_trigger_settings

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

        ms_before = float(pre) / (fs_hz / 1000.0)
        ms_after = float(post) / (fs_hz / 1000.0)

        if not (0 < ms_before < 50 and 0 < ms_after < 50):
            raise ValueError(f"unexpected cutout ms: {ms_before}, {ms_after}")

        return ms_before, ms_after
    except Exception:
        return 1.0, 2.0


def _load_preprocessed_recording(*, well_out_dir: Path) -> Any:
    recording_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME / "preprocessed_recording"
    if not recording_dir.exists():
        raise FileNotFoundError(f"preprocessed_recording not found: {recording_dir}")

    import spikeinterface.full as si  # type: ignore[import-not-found]

    try:
        return si.load(recording_dir)
    except Exception:
        return si.load_extractor(recording_dir)


def _load_per_segment_manifest(*, preprocess_dir: Path) -> dict[str, Any] | None:
    manifest_path = Path(preprocess_dir) / "per_segment_preprocessed" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = _read_json(manifest_path)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _resample_recording_to_target(
    *,
    recording: Any,
    rec_name: str,
    target_sampling_frequency_hz: float | None,
    temporal_resample_margin_ms: float,
    temporal_resample_dtype: str | None,
) -> Any:
    if target_sampling_frequency_hz is None:
        return recording

    try:
        import numpy as np  # type: ignore[import-not-found]
        import spikeinterface.preprocessing as spre  # type: ignore[import-not-found]

        current_fs = float(recording.get_sampling_frequency())
        target_fs = float(target_sampling_frequency_hz)

        current_fs_i = int(round(current_fs))
        target_fs_i = int(round(target_fs))
        if current_fs_i <= 0 or target_fs_i <= 0:
            raise RuntimeError(f"Invalid sampling frequency (current={current_fs}, target={target_fs})")

        if current_fs_i == target_fs_i:
            return recording

        dtype = None
        if temporal_resample_dtype is not None:
            try:
                dtype = np.dtype(str(temporal_resample_dtype))
            except Exception:
                dtype = None

        logger.info(
            "Resampling segment recording %s: fs %d -> %d Hz (margin_ms=%.1f dtype=%s)",
            str(rec_name),
            int(current_fs_i),
            int(target_fs_i),
            float(temporal_resample_margin_ms),
            str(temporal_resample_dtype),
        )

        return spre.resample(
            recording,
            resample_rate=int(target_fs_i),
            margin_ms=float(temporal_resample_margin_ms),
            dtype=dtype,
            skip_checks=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to resample segment recording rec={rec_name} to target_fs={target_sampling_frequency_hz}: {e}"
        ) from e


def _load_preprocessed_segment_recording(
    *,
    preprocess_dir: Path,
    segment_index: int,
    rec_name: str,
    target_sampling_frequency_hz: float | None = None,
    temporal_resample_margin_ms: float = 100.0,
    temporal_resample_dtype: str | None = None,
) -> Any:
    """Load a preprocessed segment recording persisted by stage-1 outputs."""

    try:
        import spikeinterface.full as si  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("waveform extraction requires spikeinterface installed") from e

    preprocess_dir = Path(preprocess_dir)
    segment_root = preprocess_dir / "per_segment_preprocessed"
    manifest = _load_per_segment_manifest(preprocess_dir=preprocess_dir)

    seg_dir: Path | None = None
    if isinstance(manifest, dict):
        segments = manifest.get("segments")
        if isinstance(segments, list):
            for item in segments:
                if not isinstance(item, dict):
                    continue
                try:
                    item_idx = int(item.get("segment_index"))
                except Exception:
                    continue
                item_name = str(item.get("rec_name", ""))
                if item_idx == int(segment_index) and item_name == str(rec_name):
                    item_folder = item.get("folder")
                    if item_folder is not None:
                        seg_dir = Path(str(item_folder))
                    break

    if seg_dir is None:
        seg_token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(rec_name)).strip("_")
        if not seg_token:
            seg_token = f"segment_{int(segment_index):03d}"
        seg_dir = segment_root / f"{int(segment_index):03d}_{seg_token}"

    if not seg_dir.exists():
        raise FileNotFoundError(f"preprocessed segment recording not found: {seg_dir}")

    try:
        seg_rec = si.load(seg_dir)
    except Exception:
        seg_rec = si.load_extractor(seg_dir)

    seg_rec = _resample_recording_to_target(
        recording=seg_rec,
        rec_name=str(rec_name),
        target_sampling_frequency_hz=target_sampling_frequency_hz,
        temporal_resample_margin_ms=temporal_resample_margin_ms,
        temporal_resample_dtype=temporal_resample_dtype,
    )
    return seg_rec


def _load_raw_segment_recording_segment_channels(
    *,
    h5_path: Path,
    stream_id: str,
    rec_name: str,
    center_chunk_size: int = 10_000,
    preprocess_like_mea_analysis: bool = True,
    target_sampling_frequency_hz: float | None = None,
    temporal_resample_margin_ms: float = 100.0,
    temporal_resample_dtype: str | None = None,
) -> Any:
    """Load a single raw Maxwell rec segment and keep its segment channel set.

    Terminology:
    - "segment channels" means the full electrode set available in *this* segment.
    - This is not "all channels" across all segments.
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

    try:
        electrodes = np.asarray(rec_centered.get_property("contact_vector")["electrode"], dtype=int)
        if int(np.unique(electrodes).size) != int(electrodes.size):
            raise RuntimeError(f"Duplicate electrode ids in contact_vector for rec={rec_name}")
        rec_centered = rec_centered.rename_channels([int(e) for e in electrodes])
    except Exception:
        pass

    if preprocess_like_mea_analysis:
        rec_centered = apply_standard_preprocessing(recording=rec_centered, logger=logger)

    # IMPORTANT: If preprocessing applied temporal resampling, the concatenated recording
    # (and all downstream spike times/epoch markers) are in the resampled time base.
    # Per-segment waveforms load raw segments directly from the H5, so we must resample
    # them here to keep segment-local indexing consistent.
    return _resample_recording_to_target(
        recording=rec_centered,
        rec_name=str(rec_name),
        target_sampling_frequency_hz=target_sampling_frequency_hz,
        temporal_resample_margin_ms=temporal_resample_margin_ms,
        temporal_resample_dtype=temporal_resample_dtype,
    )


def _load_raw_segment_recording_full_channels(
    *,
    h5_path: Path,
    stream_id: str,
    rec_name: str,
    center_chunk_size: int = 10_000,
    preprocess_like_mea_analysis: bool = True,
) -> Any:
    """Backward-compatible alias.

    Historical name used "full_channels" to mean "full set for the segment".
    Prefer `_load_raw_segment_recording_segment_channels`.
    """

    return _load_raw_segment_recording_segment_channels(
        h5_path=h5_path,
        stream_id=stream_id,
        rec_name=rec_name,
        center_chunk_size=center_chunk_size,
        preprocess_like_mea_analysis=preprocess_like_mea_analysis,
    )


def _resolve_mea_sorter_output_dir(
    *,
    well_out_dir: Path,
    merged_sorting_dir: Path | None = None,
    prefer_merged_sorting: bool = False,
) -> Path:
    if merged_sorting_dir is not None:
        explicit_merged = Path(merged_sorting_dir)
        if explicit_merged.exists():
            return explicit_merged

    canonical_merged = well_out_dir / "stg2_spikesorting_outputs" / "unitmatch_outputs" / "final_merged_sorting"
    if bool(prefer_merged_sorting) and canonical_merged.exists():
        return canonical_merged

    p = well_out_dir / "stg2_spikesorting_outputs" / "sorter_output"
    if p.exists():
        return p

    legacy = well_out_dir / "sorter_output"
    if legacy.exists():
        return legacy

    return p


def _load_sorting_from_sorter_output_dir(*, sorter_output_dir: Path, sorter: str) -> Any:
    import spikeinterface.full as si  # type: ignore[import-not-found]

    if hasattr(si, "read_sorter_folder"):
        try:
            return si.read_sorter_folder(sorter_output_dir, sorter_name=sorter)
        except TypeError:
            try:
                return si.read_sorter_folder(sorter_output_dir, sorter)
            except Exception:
                pass
        except Exception:
            pass

    try:
        return si.load_extractor(sorter_output_dir)
    except Exception:
        pass

    raise RuntimeError(f"Could not load sorting from sorter_output_dir={sorter_output_dir} (sorter={sorter}).")


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


def _maxwell_epochs_to_segment_local_intervals(*, maxwell_epochs: list[dict], segment_index: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for e in maxwell_epochs:
        try:
            if int(e.get("segment_index")) != int(segment_index):
                continue
            start = int(e["segment_start_sample"])
            end = int(e["segment_end_sample"])
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
) -> tuple[list[int], int, int, list[int], list[int]]:
    if not intervals:
        return spike_train, 0, 0, [], []

    kept: list[int] = []
    removed_outside = 0
    removed_edge = 0
    removed_outside_spikes: list[int] = []
    removed_edge_spikes: list[int] = []

    i = 0
    for t in spike_train:
        t_int = int(t)
        t0 = t_int - int(pre_samples)
        t1 = t_int + int(post_samples)

        while i < len(intervals) and intervals[i][1] <= t_int:
            i += 1

        in_interval = False
        ok = False
        if i < len(intervals):
            start, end = intervals[i]
            if start <= t_int < end:
                in_interval = True
                if t0 >= start and t1 < end:
                    ok = True

        if ok:
            kept.append(t_int)
        else:
            if in_interval:
                removed_edge += 1
                removed_edge_spikes.append(t_int)
            else:
                removed_outside += 1
                removed_outside_spikes.append(t_int)

    return kept, removed_outside, removed_edge, removed_outside_spikes, removed_edge_spikes


def _to_numpy_sorting(*, unit_trains: dict[int, list[int]], fs_hz: float) -> Any:
    import numpy as np  # type: ignore[import-not-found]
    from spikeinterface.core import NumpySorting  # type: ignore[import-not-found]

    unit_trains_np: dict[int, "np.ndarray"] = {
        int(u): np.asarray(times, dtype=np.int64) for u, times in unit_trains.items()
    }

    unit_ids = sorted(unit_trains_np.keys())
    all_times: list[int] = []
    all_labels: list[int] = []
    for u in unit_ids:
        times_arr = unit_trains_np[u]
        if times_arr.size:
            all_times.extend(times_arr.tolist())
            all_labels.extend([u] * int(times_arr.size))

    if not all_times:
        return NumpySorting.from_unit_dict(unit_trains_np, sampling_frequency=float(fs_hz))

    times_arr = np.asarray(all_times, dtype=np.int64)
    labels_arr = np.asarray(all_labels, dtype=np.int64)

    order = np.argsort(times_arr, kind="mergesort")
    times_arr = times_arr[order]
    labels_arr = labels_arr[order]

    return NumpySorting.from_times_labels(
        times_list=[times_arr],
        labels_list=[labels_arr],
        sampling_frequency=float(fs_hz),
    )


__all__ = [
    "_read_json",
    "_write_json",
    "_infer_cutout_ms",
    "_load_preprocessed_recording",
    "_load_preprocessed_segment_recording",
    "_load_raw_segment_recording_full_channels",
    "_load_raw_segment_recording_segment_channels",
    "_resolve_mea_sorter_output_dir",
    "_load_sorting_from_sorter_output_dir",
    "_epochs_to_intervals",
    "_maxwell_epochs_to_segment_local_intervals",
    "_filter_spike_train_by_intervals",
    "_load_preprocess_requested_cfg",
    "_to_numpy_sorting",
]
