from __future__ import annotations

from typing import Any

from .segments import (
    _append_segment_skip_summary,
    _append_segment_summary,
    _compute_segment_maxwell_intervals,
    _count_spikes_in_concat_window,
    _parse_concat_epoch_segment,
)
from .utils import _load_raw_segment_recording_full_channels, _to_numpy_sorting


def _extract_concat_waveforms(
    *,
    inputs,
    filtered_sorting: Any,
    recording: Any,
    concat_waveforms_dir,
    window,
    quality_metrics_params: dict[str, Any],
    logger: Any,
) -> None:
    import shutil

    import spikeinterface.full as si  # type: ignore[import-not-found]

    if concat_waveforms_dir.exists() and inputs.force_restart:
        shutil.rmtree(concat_waveforms_dir)

    logger.info("Extracting concat waveforms -> %s", concat_waveforms_dir)
    concat_analyzer = si.create_sorting_analyzer(
        filtered_sorting,
        recording,
        format="binary_folder",
        folder=concat_waveforms_dir,
        return_in_uV=True,
    )
    concat_analyzer.compute(
        ["random_spikes", "waveforms"],
        extension_params={
            "random_spikes": {
                "method": "uniform",
                "max_spikes_per_unit": int(inputs.max_spikes_per_unit),
                "seed": 0,
            },
            "waveforms": {"ms_before": float(window.ms_before), "ms_after": float(window.ms_after)},
        },
        verbose=False,
        n_jobs=int(inputs.n_jobs),
    )

    concat_analyzer.compute(
        [
            "spike_amplitudes",
            "templates",
            "noise_levels",
            "quality_metrics",
            "template_metrics",
            "unit_locations",
        ],
        extension_params={
            "unit_locations": {"method": "monopolar_triangulation"},
            "quality_metrics": dict(quality_metrics_params),
        },
        verbose=False,
        n_jobs=int(inputs.n_jobs),
    )


def _extract_per_segment_waveforms(
    *,
    inputs,
    recording: Any,
    sorting_unfiltered: Any,
    filtered_sorting: Any,
    epochs,
    window,
    segment_waveforms_dir,
    common_channel_ids: set[int],
    filtering_summary: dict[str, Any],
    wf_rejection_rows: list[dict[str, Any]],
    base_rej_fields: dict[str, Any],
    quality_metrics_params: dict[str, Any],
    logger: Any,
) -> None:
    if inputs.per_segment and epochs.concat_epochs:
        segment_waveforms_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        import spikeinterface.full as si  # type: ignore[import-not-found]

        logger.info("Extracting per-segment waveforms -> %s", segment_waveforms_dir)
        for seg in epochs.concat_epochs:
            spec = _parse_concat_epoch_segment(seg=seg, segment_waveforms_dir=segment_waveforms_dir)
            if spec is None:
                continue

            seg_index = int(spec.segment_index)
            rec_name = str(spec.rec_name)
            start = int(spec.start_sample_concat)
            end = int(spec.end_sample_concat)
            seg_dir = spec.seg_dir

            if seg_dir.exists() and inputs.force_restart:
                shutil.rmtree(seg_dir)

            seg_rec = _load_raw_segment_recording_full_channels(
                h5_path=inputs.h5_path,
                stream_id=inputs.stream_id,
                rec_name=rec_name,
                center_chunk_size=10_000,
            )

            raw_channels_total = None
            excluded_common_channels_total = None
            kept_additional_channels_total = None
            try:
                raw_channels_total = int(seg_rec.get_num_channels())
            except Exception:
                pass

            if inputs.per_segment_only_additional_channels and common_channel_ids:
                try:
                    import numpy as np  # type: ignore[import-not-found]

                    cv = seg_rec.get_property("contact_vector")
                    electrodes = np.asarray(cv["electrode"], dtype=int)
                    ch_ids = list(seg_rec.get_channel_ids())
                    if len(electrodes) == len(ch_ids):
                        keep_mask = [int(e) not in common_channel_ids for e in electrodes]
                        keep_channel_ids = [ch_ids[i] for i, keep in enumerate(keep_mask) if keep]
                        kept_electrodes = [int(electrodes[i]) for i, keep in enumerate(keep_mask) if keep]
                        seg_rec = seg_rec.select_channels(keep_channel_ids)
                        excluded_common_channels_total = int(len(ch_ids) - len(keep_channel_ids))
                        kept_additional_channels_total = int(len(keep_channel_ids))

                        try:
                            if len(kept_electrodes) == int(seg_rec.get_num_channels()):
                                if int(np.unique(np.asarray(kept_electrodes)).size) == int(len(kept_electrodes)):
                                    seg_rec = seg_rec.rename_channels([int(e) for e in kept_electrodes])
                        except Exception:
                            pass
                except Exception:
                    pass

            if inputs.per_segment_only_additional_channels:
                try:
                    if int(seg_rec.get_num_channels()) == 0:
                        logger.info(
                            "Segment %s: no additional channels after excluding common set; skipping per-segment waveforms.",
                            rec_name,
                        )
                        _append_segment_skip_summary(
                            filtering_summary=filtering_summary,
                            seg_index=seg_index,
                            rec_name=rec_name,
                            raw_channels_total=raw_channels_total,
                            excluded_common_channels_total=excluded_common_channels_total,
                            kept_additional_channels_total=kept_additional_channels_total,
                            skipped_reason="no_additional_channels",
                        )
                        continue
                except Exception:
                    pass

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
                if inputs.per_segment_only_additional_channels:
                    logger.info(
                        "Segment %s: raw channels=%s, kept additional=%s (excluded common=%s, concat/common=%d)",
                        rec_name,
                        raw_channels_total,
                        kept_additional_channels_total,
                        excluded_common_channels_total,
                        int(recording.get_num_channels()),
                    )
                else:
                    logger.info(
                        "Segment %s: raw channels=%d (concat/common=%d)",
                        rec_name,
                        int(seg_rec.get_num_channels()),
                        int(recording.get_num_channels()),
                    )
            except Exception:
                pass

            try:
                in_seg_unfiltered = _count_spikes_in_concat_window(sorting=sorting_unfiltered, start=start, end=end)
                in_seg_filtered = _count_spikes_in_concat_window(sorting=filtered_sorting, start=start, end=end)
                logger.info(
                    "Segment %s: spikes in concat window (unfiltered=%d, after-concat-filter=%d, excluded-at-concat=%d)",
                    rec_name,
                    int(in_seg_unfiltered),
                    int(in_seg_filtered),
                    int(in_seg_unfiltered - in_seg_filtered),
                )
            except Exception:
                pass

            seg_maxwell_intervals = _compute_segment_maxwell_intervals(
                inputs=inputs,
                epochs=epochs,
                segment_index=seg_index,
                seg_len=int(seg_len),
                logger=logger,
                rec_name=rec_name,
            )

            unit_trains_seg: dict[int, list[int]] = {}

            seg_spikes_total = 0
            seg_removed_epoch_total = 0
            seg_removed_edge_total = 0
            seg_kept_total = 0

            for u in filtered_sorting.get_unit_ids():
                st = filtered_sorting.get_unit_spike_train(u)
                st_list = [int(x) for x in st]

                local_all = [int(t - start) for t in st_list if start <= t < end]
                local_all.sort()
                seg_spikes_total += len(local_all)

                kept_edges: list[int] = []
                for t_local in local_all:
                    if t_local - window.pre_samples < 0:
                        continue
                    if t_local + window.post_samples >= int(seg_len):
                        continue
                    kept_edges.append(int(t_local))

                try:
                    source_name = f"seg{int(seg_index):02d}_{rec_name}"
                    kept_edge_set = set(int(x) for x in kept_edges)
                    for t_local in local_all:
                        if int(t_local) in kept_edge_set:
                            continue
                        if (int(t_local) - int(window.pre_samples) < 0) or (
                            int(t_local) + int(window.post_samples) >= int(seg_len)
                        ):
                            t_concat = int(t_local) + int(start)
                            wf_rejection_rows.append(
                                {
                                    **base_rej_fields,
                                    "scope": "segment",
                                    "source_name": str(source_name),
                                    "segment_index": int(seg_index),
                                    "rec_name": str(rec_name),
                                    "unit_id": int(u),
                                    "spike_sample_local": int(t_local),
                                    "spike_sample_concat": int(t_concat),
                                    "spike_time_s": float(t_concat) / float(window.fs_hz),
                                    "reason": "waveform_window_outside_segment_bounds",
                                }
                            )
                except Exception:
                    pass

                removed_edge = int(len(local_all) - len(kept_edges))
                seg_removed_edge_total += int(removed_edge)
                seg_kept_total += int(len(kept_edges))

                unit_trains_seg[int(u)] = kept_edges

            try:
                logger.info(
                    "Segment %s: per-segment spike train prepared=%d (in_window=%d, excluded_by_segment_edge=%d)",
                    rec_name,
                    int(seg_kept_total),
                    int(seg_spikes_total),
                    int(seg_removed_edge_total),
                )
            except Exception:
                pass

            seg_sort = _to_numpy_sorting(unit_trains=unit_trains_seg, fs_hz=window.fs_hz)

            try:
                seg_sort.register_recording(seg_rec)
            except Exception:
                pass

            seg_analyzer = si.create_sorting_analyzer(
                seg_sort,
                seg_rec,
                format="binary_folder",
                folder=seg_dir,
                return_in_uV=True,
            )
            seg_analyzer.compute(
                ["random_spikes", "waveforms"],
                extension_params={
                    "random_spikes": {
                        "method": "uniform",
                        "max_spikes_per_unit": int(inputs.max_spikes_per_unit),
                        "seed": 0,
                    },
                    "waveforms": {"ms_before": float(window.ms_before), "ms_after": float(window.ms_after)},
                },
                verbose=False,
                n_jobs=max(1, int(inputs.n_jobs)),
            )

            seg_analyzer.compute(
                [
                    "spike_amplitudes",
                    "templates",
                    "noise_levels",
                    "quality_metrics",
                    "template_metrics",
                    "unit_locations",
                ],
                extension_params={
                    "unit_locations": {"method": "monopolar_triangulation"},
                    "quality_metrics": dict(quality_metrics_params),
                },
                verbose=False,
                n_jobs=max(1, int(inputs.n_jobs)),
            )

            if inputs.filter_by_maxwell_epochs and seg_maxwell_intervals:
                try:
                    from .exclusions import _get_random_spike_samples  # type: ignore

                    source_name = f"seg{int(seg_index):02d}_{rec_name}"
                    removed_outside_total = 0
                    removed_edge_epoch_total = 0
                    selected_total = 0

                    intervals = list(seg_maxwell_intervals) if seg_maxwell_intervals else []

                    def _classify_maxwell_exclusion(t_local: int):
                        if not intervals:
                            return None

                        containing = None
                        for a, b in intervals:
                            if int(a) <= int(t_local) < int(b):
                                containing = (int(a), int(b))
                                break

                        if containing is None:
                            return "outside_maxwell_epoch"

                        a, b = containing
                        left_bound = int(t_local) - int(window.pre_samples)
                        right_bound = int(t_local) + int(window.post_samples)
                        if left_bound < int(a) or right_bound >= int(b):
                            return "waveform_window_crosses_epoch_edge"
                        return None

                    for u in seg_sort.get_unit_ids():
                        samples = _get_random_spike_samples(analyzer=seg_analyzer, unit_id=u)
                        if samples is None:
                            continue
                        try:
                            samples_iter = list(samples)
                        except Exception:
                            continue

                        selected_total += int(len(samples_iter))
                        for t_local in samples_iter:
                            try:
                                t_local_i = int(t_local)
                            except Exception:
                                continue

                            reason = _classify_maxwell_exclusion(t_local_i)
                            if reason is None:
                                continue

                            if reason == "outside_maxwell_epoch":
                                removed_outside_total += 1
                            elif reason == "waveform_window_crosses_epoch_edge":
                                removed_edge_epoch_total += 1

                            t_concat = int(t_local_i) + int(start)
                            wf_rejection_rows.append(
                                {
                                    **base_rej_fields,
                                    "scope": "segment",
                                    "source_name": str(source_name),
                                    "segment_index": int(seg_index),
                                    "rec_name": str(rec_name),
                                    "unit_id": int(u),
                                    "spike_sample_local": int(t_local_i),
                                    "spike_sample_concat": int(t_concat),
                                    "spike_time_s": float(t_concat) / float(window.fs_hz),
                                    "reason": str(reason),
                                }
                            )

                    seg_removed_epoch_total = int(removed_outside_total + removed_edge_epoch_total)
                    logger.info(
                        "Segment %s: flagged extracted waveforms by maxwell epochs=%d (outside=%d edge=%d, selected_random_spikes=%d)",
                        rec_name,
                        int(seg_removed_epoch_total),
                        int(removed_outside_total),
                        int(removed_edge_epoch_total),
                        int(selected_total),
                    )
                except Exception:
                    seg_removed_epoch_total = 0
            else:
                seg_removed_epoch_total = 0

            try:
                _append_segment_summary(
                    filtering_summary=filtering_summary,
                    seg_index=seg_index,
                    rec_name=rec_name,
                    seg_spikes_total=int(seg_spikes_total),
                    seg_removed_epoch_total=int(seg_removed_epoch_total),
                    seg_removed_edge_total=int(seg_removed_edge_total),
                    seg_kept_total=int(seg_kept_total),
                    seg_maxwell_intervals_len=int(len(seg_maxwell_intervals)),
                    raw_channels_total=raw_channels_total,
                    excluded_common_channels_total=excluded_common_channels_total,
                    kept_additional_channels_total=kept_additional_channels_total,
                )
            except Exception:
                pass


__all__ = [
    "_extract_concat_waveforms",
    "_extract_per_segment_waveforms",
]
