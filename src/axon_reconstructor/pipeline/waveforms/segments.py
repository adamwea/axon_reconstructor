from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .utils import _maxwell_epochs_to_segment_local_intervals


@dataclass(frozen=True)
class _SegmentSpec:
    segment_index: int
    rec_name: str
    start_sample_concat: int
    end_sample_concat: int
    seg_dir: Path


def _parse_concat_epoch_segment(*, seg: dict[str, Any], segment_waveforms_dir: Path) -> Optional[_SegmentSpec]:
    try:
        seg_index = int(seg["segment_index"])
        rec_name = str(seg.get("rec_name", f"seg{seg_index}"))
        start = int(seg["start_sample"])
        end = int(seg["end_sample"])
    except Exception:
        return None

    seg_dir = segment_waveforms_dir / f"seg{seg_index:02d}_{rec_name}"
    return _SegmentSpec(
        segment_index=int(seg_index),
        rec_name=str(rec_name),
        start_sample_concat=int(start),
        end_sample_concat=int(end),
        seg_dir=seg_dir,
    )


def _count_spikes_in_concat_window(*, sorting: Any, start: int, end: int) -> int:
    total = 0
    for u in sorting.get_unit_ids():
        st = sorting.get_unit_spike_train(u)
        for t in st:
            tt = int(t)
            if start <= tt < end:
                total += 1
    return int(total)


def _compute_segment_maxwell_intervals(
    *,
    inputs,
    epochs,
    segment_index: int,
    seg_len: int,
    logger: Any,
    rec_name: str,
) -> list[tuple[int, int]]:
    seg_maxwell_intervals: list[tuple[int, int]] = []
    if inputs.filter_by_maxwell_epochs and epochs.maxwell_epochs:
        seg_maxwell_intervals = _maxwell_epochs_to_segment_local_intervals(
            maxwell_epochs=epochs.maxwell_epochs,
            segment_index=segment_index,
        )

    for a, b in seg_maxwell_intervals[:10]:
        if a < 0 or b > int(seg_len):
            logger.warning(
                "Segment %s: segment-local Maxwell interval out of bounds: (%d, %d) with seg_len=%d",
                rec_name,
                int(a),
                int(b),
                int(seg_len),
            )
            break

    return seg_maxwell_intervals


def _append_segment_skip_summary(
    *,
    filtering_summary: dict[str, Any],
    seg_index: int,
    rec_name: str,
    raw_channels_total: Optional[int],
    excluded_common_channels_total: Optional[int],
    kept_additional_channels_total: Optional[int],
    skipped_reason: str,
) -> None:
    filtering_summary["per_segment"]["segments"].append(
        {
            "segment_index": int(seg_index),
            "rec_name": str(rec_name),
            "spikes_in_segment_total": 0,
            "removed_by_maxwell_epoch": 0,
            "removed_by_edge": 0,
            "kept_spikes_total": 0,
            "maxwell_intervals_in_segment": None,
            "raw_channels_total": raw_channels_total,
            "excluded_common_channels_total": excluded_common_channels_total,
            "kept_additional_channels_total": kept_additional_channels_total,
            "skipped_reason": str(skipped_reason),
        }
    )


def _append_segment_summary(
    *,
    filtering_summary: dict[str, Any],
    seg_index: int,
    rec_name: str,
    seg_spikes_total: int,
    seg_removed_epoch_total: int,
    seg_removed_edge_total: int,
    seg_kept_total: int,
    seg_maxwell_intervals_len: int,
    raw_channels_total: Optional[int],
    excluded_common_channels_total: Optional[int],
    kept_additional_channels_total: Optional[int],
) -> None:
    filtering_summary["per_segment"]["removed_by_maxwell_epoch_total"] += int(seg_removed_epoch_total)
    filtering_summary["per_segment"]["removed_by_edge_total"] += int(seg_removed_edge_total)
    filtering_summary["per_segment"]["kept_spikes_total"] += int(seg_kept_total)
    filtering_summary["per_segment"]["segments"].append(
        {
            "segment_index": int(seg_index),
            "rec_name": str(rec_name),
            "spikes_in_segment_total": int(seg_spikes_total),
            "removed_by_maxwell_epoch": int(seg_removed_epoch_total),
            "removed_by_edge": int(seg_removed_edge_total),
            "kept_spikes_total": int(seg_kept_total),
            "maxwell_intervals_in_segment": int(seg_maxwell_intervals_len),
            "raw_channels_total": raw_channels_total,
            "excluded_common_channels_total": excluded_common_channels_total,
            "kept_additional_channels_total": kept_additional_channels_total,
        }
    )


__all__ = [
    "_SegmentSpec",
    "_append_segment_skip_summary",
    "_append_segment_summary",
    "_compute_segment_maxwell_intervals",
    "_count_spikes_in_concat_window",
    "_parse_concat_epoch_segment",
]
