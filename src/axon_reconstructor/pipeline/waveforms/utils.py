from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..pipeline_driver import PREPROCESS_OUTPUTS_DIRNAME
from ..raw_preprocessing.utils import _ensure_maxwell_hdf5_plugin_path


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
        from ..raw_preprocessing.h5_helpers import _read_well_rec_frame_nos_and_trigger_settings

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


def _load_raw_segment_recording_full_channels(
    *,
    h5_path: Path,
    stream_id: str,
    rec_name: str,
    center_chunk_size: int = 10_000,
) -> Any:
    """Load a single raw Maxwell rec segment and keep its full channel set."""

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

    return rec_centered


def _resolve_mea_sorter_output_dir(*, well_out_dir: Path) -> Path:
    p = well_out_dir / "spikesorting_outputs" / "sorter_output"
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
            return si.read_sorter_folder(sorter_output_dir, sorter)

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

    unit_ids = sorted(unit_trains.keys())
    all_times: list[int] = []
    all_labels: list[int] = []
    for u in unit_ids:
        times = unit_trains[u]
        all_times.extend(times)
        all_labels.extend([u] * len(times))

    if not all_times:
        return NumpySorting.from_unit_dict(unit_trains, sampling_frequency=float(fs_hz))

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
    "_load_raw_segment_recording_full_channels",
    "_resolve_mea_sorter_output_dir",
    "_load_sorting_from_sorter_output_dir",
    "_epochs_to_intervals",
    "_maxwell_epochs_to_segment_local_intervals",
    "_filter_spike_train_by_intervals",
    "_to_numpy_sorting",
]
