from __future__ import annotations

from typing import Any

from .utils import _filter_spike_train_by_intervals, _to_numpy_sorting


def _init_filtering_summary(*, inputs, window, epochs) -> dict[str, Any]:
    return {
        "filter_by_maxwell_epochs": bool(inputs.filter_by_maxwell_epochs),
        "filter_by_segment_bounds": bool(getattr(inputs, "filter_by_segment_bounds", True)),
        "ms_before": float(window.ms_before),
        "ms_after": float(window.ms_after),
        "pre_samples": int(window.pre_samples),
        "post_samples": int(window.post_samples),
        "maxwell_epochs_path": str(epochs.maxwell_epochs_path) if epochs.maxwell_epochs_path.exists() else None,
        "concat_epochs_path": str(epochs.concat_epochs_path) if epochs.concat_epochs_path.exists() else None,
        "removed_spikes_total": 0,
        "kept_spikes_total": 0,
        "removed_by_maxwell_epoch_total": 0,
        "removed_by_edge_total": 0,
        "per_segment": {
            "enabled": bool(inputs.per_segment),
            "removed_by_maxwell_epoch_total": 0,
            "removed_by_edge_total": 0,
            "kept_spikes_total": 0,
            "segments": [],
        },
    }


def _init_wf_rejection_log_fields(*, inputs, window) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wf_rejection_rows: list[dict[str, Any]] = []
    base_rej_fields: dict[str, Any] = {
        "stream_id": inputs.stream_id,
        "sorter": inputs.sorter,
        "h5_path": str(inputs.h5_path),
        "fs_hz": float(window.fs_hz),
        "ms_before": float(window.ms_before),
        "ms_after": float(window.ms_after),
        "pre_samples": int(window.pre_samples),
        "post_samples": int(window.post_samples),
    }
    return wf_rejection_rows, base_rej_fields


def _filter_sorting_by_maxwell_epochs(
    *,
    inputs,
    sorting: Any,
    epochs,
    window,
    logger: Any,
    filtering_summary: dict[str, Any],
    wf_rejection_rows: list[dict[str, Any]],
    base_rej_fields: dict[str, Any],
) -> Any:
    filtered_sorting = sorting
    if inputs.filter_by_maxwell_epochs and epochs.maxwell_intervals:
        unit_trains: dict[int, list[int]] = {}
        removed_total = 0
        removed_outside_total = 0
        removed_edge_total = 0
        kept_total = 0

        unit_ids = list(sorting.get_unit_ids())
        for u in unit_ids:
            st = sorting.get_unit_spike_train(u)
            st_list = [int(x) for x in st]
            st_list.sort()

            kept, removed_outside, removed_edge, removed_outside_spikes, removed_edge_spikes = _filter_spike_train_by_intervals(
                spike_train=st_list,
                intervals=epochs.maxwell_intervals,
                pre_samples=window.pre_samples,
                post_samples=window.post_samples,
            )
            unit_trains[int(u)] = kept
            removed_outside_total += int(removed_outside)
            removed_edge_total += int(removed_edge)
            removed_total += int(removed_outside) + int(removed_edge)
            kept_total += len(kept)

            for t in removed_outside_spikes:
                wf_rejection_rows.append(
                    {
                        **base_rej_fields,
                        "scope": "concat",
                        "source_name": "concat",
                        "segment_index": None,
                        "rec_name": None,
                        "unit_id": int(u),
                        "spike_sample_local": None,
                        "spike_sample_concat": int(t),
                        "spike_time_s": float(t) / float(window.fs_hz),
                        "reason": "outside_maxwell_epoch",
                    }
                )
            for t in removed_edge_spikes:
                wf_rejection_rows.append(
                    {
                        **base_rej_fields,
                        "scope": "concat",
                        "source_name": "concat",
                        "segment_index": None,
                        "rec_name": None,
                        "unit_id": int(u),
                        "spike_sample_local": None,
                        "spike_sample_concat": int(t),
                        "spike_time_s": float(t) / float(window.fs_hz),
                        "reason": "waveform_window_crosses_epoch_edge",
                    }
                )

        filtered_sorting = _to_numpy_sorting(unit_trains=unit_trains, fs_hz=window.fs_hz)
        filtering_summary["removed_spikes_total"] = int(removed_total)
        filtering_summary["kept_spikes_total"] = int(kept_total)
        filtering_summary["removed_by_maxwell_epoch_total"] = int(removed_outside_total)
        filtering_summary["removed_by_edge_total"] = int(removed_edge_total)
        logger.info(
            "Filtered spikes by Maxwell epochs: kept=%d removed=%d (pre=%d post=%d samples)",
            kept_total,
            removed_total,
            window.pre_samples,
            window.post_samples,
        )
    else:
        try:
            n_total = sum(len(sorting.get_unit_spike_train(u)) for u in sorting.get_unit_ids())
            filtering_summary["kept_spikes_total"] = int(n_total)
        except Exception:
            pass

    return filtered_sorting


__all__ = [
    "_filter_sorting_by_maxwell_epochs",
    "_init_filtering_summary",
    "_init_wf_rejection_log_fields",
]
