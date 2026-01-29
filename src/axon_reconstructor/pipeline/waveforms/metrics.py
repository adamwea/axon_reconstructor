from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class SegmentAnalyzerSource:
    source_name: str
    analyzer_dir: Path
    segment_index: int
    start_sample_concat: int
    end_sample_concat: int


@dataclass(frozen=True)
class SourceKey:
    scope: str  # "concat" | "segment" | "merged"
    source_name: str
    segment_index: Optional[int] = None
    rec_name: Optional[str] = None


def _require_extension(*, analyzer: Any, name: str) -> Any:
    """Return analyzer extension or raise with a helpful message."""

    try:
        return analyzer.get_extension(name)
    except Exception as e:
        raise RuntimeError(
            f"Missing analyzer extension '{name}'. "
            "This should be computed during waveform extraction (concat + per-segment). "
            "Re-run waveforms with force_restart to regenerate analyzers."
        ) from e


def load_and_compute_metrics(
    *,
    analyzer_dir: Path,
    ms_before: float,
    ms_after: float,
    n_jobs: int,
    logger: Any,
) -> tuple[Any, Any]:
    """Load an existing SortingAnalyzer folder and return (q_metrics, t_metrics).

    Returns pandas DataFrames.
    """

    try:
        import spikeinterface.full as si  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("spikeinterface is required to load metrics") from e

    analyzer = si.load_sorting_analyzer(analyzer_dir)

    q_metrics = _require_extension(analyzer=analyzer, name="quality_metrics").get_data()
    t_metrics = _require_extension(analyzer=analyzer, name="template_metrics").get_data()

    # Match MEA_Analysis behavior: include unit locations into quality metrics.
    try:
        locations = _require_extension(analyzer=analyzer, name="unit_locations").get_data()
        q_metrics = q_metrics.copy()
        q_metrics["loc_x"] = locations[:, 0]
        q_metrics["loc_y"] = locations[:, 1]
    except Exception:
        pass

    return q_metrics, t_metrics


def merge_quality_metrics(
    *,
    concat_qm: Any,
    segment_qm_by_source: dict[str, Any],
    prefer_concat_for: Optional[set[str]] = None,
) -> Any:
    """Merge concat + per-segment quality metrics into one DataFrame.

    Policy is column-based with safe defaults that align with MEA_Analysis
    thresholds ("worst-case" aggregation across contexts):
    - Higher-is-better columns => min across sources
    - Lower-is-better columns => max across sources
    - Unknown numeric columns => concat-preferred, else mean fallback

    Args:
        concat_qm: DataFrame indexed by unit_id.
        segment_qm_by_source: source_name -> DataFrame indexed by unit_id.
        prefer_concat_for: columns where concat value should always win (e.g. loc_x/loc_y).

    Returns:
        merged_qm: DataFrame indexed by unit_id.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy/pandas are required to merge metrics") from e

    prefer_concat_for = prefer_concat_for or {"loc_x", "loc_y"}

    higher_is_better_min = {
        "presence_ratio",
        "firing_rate",
    }
    lower_is_better_max = {
        "rp_contamination",
        "amplitude_median",  # less negative is worse => take max
        "amplitude_cv_median",
    }

    all_sources = {"concat": concat_qm, **segment_qm_by_source}
    unit_ids = set()
    cols = set()
    for df in all_sources.values():
        unit_ids |= set(df.index.values)
        cols |= set(df.columns)

    unit_ids_sorted = sorted(unit_ids)
    out = pd.DataFrame(index=unit_ids_sorted)

    for col in sorted(cols):
        if col in prefer_concat_for:
            # Prefer concat if present; fall back to first non-null from segments.
            series = concat_qm[col] if col in concat_qm.columns else pd.Series(index=unit_ids_sorted, dtype=float)
            out[col] = series.reindex(unit_ids_sorted)
            if out[col].isna().any():
                for src, df in segment_qm_by_source.items():
                    if col not in df.columns:
                        continue
                    fill = df[col].reindex(unit_ids_sorted)
                    out[col] = out[col].where(~out[col].isna(), fill)
            continue

        values = []
        for src, df in all_sources.items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").reindex(unit_ids_sorted)
            values.append(s)

        if not values:
            continue

        mat = np.vstack([v.to_numpy(dtype=float) for v in values])

        if col in higher_is_better_min:
            out[col] = np.nanmin(mat, axis=0)
        elif col in lower_is_better_max:
            out[col] = np.nanmax(mat, axis=0)
        else:
            # Default: concat-preferred if available else mean.
            if col in concat_qm.columns:
                out[col] = pd.to_numeric(concat_qm[col], errors="coerce").reindex(unit_ids_sorted)
            else:
                out[col] = np.nanmean(mat, axis=0)

    return out


def merge_template_metrics(
    *,
    concat_tm: Any,
    segment_tm_by_source: dict[str, Any],
) -> Any:
    """Merge template metrics.

    For now this is concat-preferred with fallback to mean across segments for
    numeric columns when concat is missing.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy/pandas are required to merge metrics") from e

    all_sources = {"concat": concat_tm, **segment_tm_by_source}
    unit_ids = set()
    cols = set()
    for df in all_sources.values():
        unit_ids |= set(df.index.values)
        cols |= set(df.columns)

    unit_ids_sorted = sorted(unit_ids)
    out = pd.DataFrame(index=unit_ids_sorted)

    for col in sorted(cols):
        if col in concat_tm.columns:
            out[col] = concat_tm[col].reindex(unit_ids_sorted)
            continue

        values = []
        for df in segment_tm_by_source.values():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").reindex(unit_ids_sorted)
            values.append(s)

        if values:
            mat = np.vstack([v.to_numpy(dtype=float) for v in values])
            out[col] = np.nanmean(mat, axis=0)

    return out


def recompute_merged_quality_metrics_from_deduplicated_spikes(
    *,
    concat_analyzer_dir: Path,
    segment_sources: list[SegmentAnalyzerSource],
    logger: Any,
) -> Any:
    """Recompute key quality metrics from a deduplicated spike train + amplitudes.

    This is used for merged metrics across concat + per-segment analyzers.

    The goal is to avoid heuristic scalar merges and instead mirror SpikeInterface
    definitions on a single deduplicated representation of unit activity.

    Metrics recomputed/overwritten (when possible):
    - num_spikes
    - firing_rate
    - presence_ratio
    - rp_contamination (and rp_violations count)
    - amplitude_median
    - amplitude_mean (new column)
    - amplitude_cv_median
    - amplitude_cv_range
    """

    try:
        import math
        import json

        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy/pandas are required to recompute merged metrics") from e

    try:
        import spikeinterface.full as si  # type: ignore[import-not-found]
        import spikeinterface.qualitymetrics.misc_metrics as mm  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("spikeinterface is required to recompute merged metrics") from e

    concat_analyzer = si.load_sorting_analyzer(concat_analyzer_dir)

    # Baseline ("old") metrics computed directly on the concat analyzer.
    concat_qm = None
    concat_amps_by_unit: dict[Any, Any] = {}
    try:
        concat_qm = concat_analyzer.get_extension("quality_metrics").get_data()
    except Exception:
        concat_qm = None
    try:
        concat_amps_by_unit = concat_analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit", concatenated=True)
    except Exception:
        concat_amps_by_unit = {}

    fs_hz = float(concat_analyzer.sampling_frequency)
    total_samples = int(concat_analyzer.get_total_samples())
    total_duration_s = float(concat_analyzer.get_total_duration())

    # Load the exact metric parameters used when computing the analyzer's
    # quality_metrics extension, so recomputation matches SpikeInterface.
    metric_params: dict[str, Any] = {}
    try:
        qm_ext = concat_analyzer.get_extension("quality_metrics")
        metric_params = dict(qm_ext.params.get("metric_params", {})) if getattr(qm_ext, "params", None) else {}
    except Exception:
        metric_params = {}

    # Best-effort fallback to on-disk params.json if extension object doesn't expose params.
    if not metric_params:
        try:
            params_path = concat_analyzer_dir / "extensions" / "quality_metrics" / "params.json"
            if params_path.exists():
                metric_params = json.loads(params_path.read_text()).get("metric_params", {})
        except Exception:
            metric_params = {}

    presence_cfg = dict(metric_params.get("presence_ratio", {}))
    bin_duration_s = float(presence_cfg.get("bin_duration_s", 60.0))
    mean_fr_ratio_thresh = float(presence_cfg.get("mean_fr_ratio_thresh", 0.0))

    rp_cfg = dict(metric_params.get("rp_violation", {}))
    refractory_period_ms = float(rp_cfg.get("refractory_period_ms", 1.0))
    censored_period_ms = float(rp_cfg.get("censored_period_ms", 0.0))

    amp_cv_cfg = dict(metric_params.get("amplitude_cv", {}))
    average_num_spikes_per_bin = int(amp_cv_cfg.get("average_num_spikes_per_bin", 50))
    percentiles = tuple(amp_cv_cfg.get("percentiles", (5, 95)))
    min_num_bins = int(amp_cv_cfg.get("min_num_bins", 10))

    # Build segment boundaries (original pre-concat segments) so recomputation
    # matches SpikeInterface's multi-segment semantics (e.g. rp violations do not
    # count across segment boundaries; amplitude_cv is binned per segment).
    segments = sorted(
        {
            (int(s.segment_index), int(s.start_sample_concat), int(s.end_sample_concat))
            for s in segment_sources
            if int(s.end_sample_concat) > int(s.start_sample_concat)
        },
        key=lambda x: x[1],
    )
    if not segments:
        segments = [(0, 0, int(total_samples))]

    seg_indices = [s[0] for s in segments]
    seg_starts = np.array([s[1] for s in segments], dtype=np.int64)
    seg_ends = np.array([s[2] for s in segments], dtype=np.int64)
    seg_lengths = (seg_ends - seg_starts).astype(np.int64)
    total_length_samples = int(np.sum(seg_lengths))

    # Map segment_index -> cumulative start (continuous index) for presence ratio.
    cum_starts: dict[int, int] = {}
    running = 0
    for (seg_idx, _start, _end), seg_len in zip(segments, seg_lengths):
        cum_starts[int(seg_idx)] = int(running)
        running += int(seg_len)

    # Prepare per-unit maps: sample_index (concat time) -> amplitude.
    # We build two separate sources (concat + segments), then select per-spike
    # which amplitude to keep based on larger magnitude (|amp|).
    concat_amp_maps_by_unit: dict[Any, dict[int, float]] = {}
    segment_best_amp_maps_by_unit: dict[Any, dict[int, float]] = {}

    def _add_spikes_from_analyzer(*, analyzer: Any, sample_offset: int, out_maps_by_unit: dict[Any, dict[int, float]], keep_best_abs: bool) -> None:
        sorting = analyzer.sorting
        unit_ids = list(sorting.unit_ids)

        spikes = sorting.to_spike_vector()
        try:
            amps = analyzer.get_extension("spike_amplitudes").get_data()
        except Exception as e:
            raise RuntimeError("Missing 'spike_amplitudes' extension; re-run waveforms with metrics enabled") from e

        if len(spikes) != len(amps):
            raise RuntimeError(
                "Spike vector and spike amplitudes are misaligned "
                f"(len(spikes)={len(spikes)} != len(amps)={len(amps)})."
            )

        for i in range(len(spikes)):
            unit_id = unit_ids[int(spikes[i]["unit_index"])]
            sample_concat = int(spikes[i]["sample_index"]) + int(sample_offset)
            if sample_concat < 0 or sample_concat >= total_samples:
                continue

            per_unit = out_maps_by_unit.get(unit_id)
            if per_unit is None:
                per_unit = {}

                out_maps_by_unit[unit_id] = per_unit

            amp_v = float(amps[i])

            if sample_concat not in per_unit:
                per_unit[sample_concat] = amp_v
            elif keep_best_abs:
                # If the same spike time appears multiple times in this analyzer
                # (should be rare), keep the entry with the larger |amplitude|.
                try:
                    if abs(amp_v) > abs(float(per_unit[sample_concat])):
                        per_unit[sample_concat] = amp_v
                except Exception:
                    per_unit[sample_concat] = amp_v

    # Add concat baseline.
    _add_spikes_from_analyzer(
        analyzer=concat_analyzer,
        sample_offset=0,
        out_maps_by_unit=concat_amp_maps_by_unit,
        keep_best_abs=False,
    )

    # Then fill from segments.
    for src in segment_sources:
        try:
            seg_analyzer = si.load_sorting_analyzer(src.analyzer_dir)
        except Exception as e:
            logger.warning("Failed to load segment analyzer %s: %s", src.analyzer_dir, e)
            continue

        _add_spikes_from_analyzer(
            analyzer=seg_analyzer,
            sample_offset=int(src.start_sample_concat),
            out_maps_by_unit=segment_best_amp_maps_by_unit,
            keep_best_abs=True,
        )

    # Merge the two sources by selecting, per spike time, the amplitude with
    # larger magnitude (|amp|). This is the expected behavior when segment
    # analyzers include additional channels and can better capture peak amplitude.
    AMP_TIE_ABS_TOL = 1e-6

    amp_maps_by_unit: dict[Any, dict[int, float]] = {}
    # Debug: how often segment vs concat wins the amplitude comparison.
    # unit_id -> (segment_wins, concat_wins, ties, only_segment, only_concat)
    pick_stats_by_unit: dict[Any, tuple[int, int, int, int, int]] = {}

    all_unit_ids = set(concat_amp_maps_by_unit.keys()) | set(segment_best_amp_maps_by_unit.keys())
    for unit_id in all_unit_ids:
        c_map = concat_amp_maps_by_unit.get(unit_id, {})
        s_map = segment_best_amp_maps_by_unit.get(unit_id, {})

        samples_all = set(c_map.keys()) | set(s_map.keys())
        if not samples_all:
            continue

        seg_wins = 0
        con_wins = 0
        ties = 0
        only_seg = 0
        only_con = 0

        merged: dict[int, float] = {}
        for sample_concat in sorted(samples_all):
            has_c = sample_concat in c_map
            has_s = sample_concat in s_map

            if has_c and has_s:
                a_c = float(c_map[sample_concat])
                a_s = float(s_map[sample_concat])
                d = abs(a_s) - abs(a_c)
                if d > AMP_TIE_ABS_TOL:
                    merged[sample_concat] = a_s
                    seg_wins += 1
                elif d < -AMP_TIE_ABS_TOL:
                    merged[sample_concat] = a_c
                    con_wins += 1
                else:
                    # Tie (or near-tie): keep segment deterministically.
                    merged[sample_concat] = a_s
                    ties += 1
            elif has_s:
                merged[sample_concat] = float(s_map[sample_concat])
                only_seg += 1
            else:
                merged[sample_concat] = float(c_map[sample_concat])
                only_con += 1

        amp_maps_by_unit[unit_id] = merged
        pick_stats_by_unit[unit_id] = (seg_wins, con_wins, ties, only_seg, only_con)

    # Helpers that mirror SpikeInterface logic.
    def _count_rp_violations_pairs(*, spike_train: Any, t_r: int) -> int:
        # spike_train must be sorted int array.
        n = int(spike_train.size)
        if n <= 1:
            return 0
        j = 1
        count = 0
        for i in range(n):
            if j < i + 1:
                j = i + 1
            while j < n and (int(spike_train[j]) - int(spike_train[i])) <= t_r:
                j += 1
            count += (j - i - 1)
        return int(count)

    # Pre-compute bin edges for presence ratio.
    bin_duration_samples = int(bin_duration_s * fs_hz)
    if bin_duration_samples <= 0:
        bin_duration_samples = int(60.0 * fs_hz)
    num_bin_edges = total_length_samples // bin_duration_samples + 1
    bin_edges = np.arange(num_bin_edges, dtype=np.int64) * int(bin_duration_samples)

    t_c = int(round(censored_period_ms * fs_hz * 1e-3))
    t_r = int(round(refractory_period_ms * fs_hz * 1e-3))

    rows: dict[str, list[Any]] = {
        "unit_id": [],
        "num_spikes": [],
        "firing_rate": [],
        "presence_ratio": [],
        "rp_contamination": [],
        "rp_violations": [],
        "amplitude_mean": [],
        "amplitude_median": [],
        "amplitude_cv_median": [],
        "amplitude_cv_range": [],
    }

    # Debug bookkeeping so we can emit OLD vs NEW comparisons *after* NaN-fallback
    # has been applied to the final dataframe.
    debug_old_by_unit: dict[Any, dict[str, Any]] = {}
    debug_new_pre_fallback_by_unit: dict[Any, dict[str, Any]] = {}
    debug_pick_stats_by_unit: dict[Any, tuple[int, int, int, int, int]] = {}

    for unit_id, sample_to_amp in amp_maps_by_unit.items():
        samples = np.array(sorted(sample_to_amp.keys()), dtype=np.int64)
        amps = np.array([sample_to_amp[int(s)] for s in samples], dtype=float)
        n_spikes = int(samples.size)

        # Assign spikes to segments based on concat sample, and compute segment-local arrays.
        seg_pos = np.searchsorted(seg_starts, samples, side="right") - 1
        valid = (seg_pos >= 0) & (seg_pos < len(seg_starts)) & (samples < seg_ends[seg_pos])
        seg_pos = seg_pos[valid]
        seg_samples_concat = samples[valid]
        seg_amps = amps[valid]

        samples_local_by_seg: dict[int, np.ndarray] = {}
        amps_by_seg: dict[int, np.ndarray] = {}
        if seg_samples_concat.size:
            for i_seg, seg_idx in enumerate(seg_indices):
                mask = seg_pos == i_seg
                if not np.any(mask):
                    continue
                local = (seg_samples_concat[mask] - int(seg_starts[i_seg])).astype(np.int64)
                order = np.argsort(local)
                samples_local_by_seg[int(seg_idx)] = local[order]
                amps_by_seg[int(seg_idx)] = seg_amps[mask][order]

        # num_spikes + firing rate
        rows["unit_id"].append(unit_id)
        rows["num_spikes"].append(n_spikes)

        fr = float(n_spikes / total_duration_s) if total_duration_s > 0 else float("nan")
        rows["firing_rate"].append(fr)

        # presence ratio (match SI: concat segments into continuous sample index)
        if total_length_samples < bin_duration_samples:
            pr = float("nan")
        else:
            unit_fr = float(n_spikes / total_duration_s) if total_duration_s > 0 else 0.0
            bin_n_spikes_thres = math.floor(unit_fr * bin_duration_s * mean_fr_ratio_thresh)
            merged_train_parts = []
            for seg_idx in seg_indices:
                loc = samples_local_by_seg.get(int(seg_idx))
                if loc is None:
                    continue
                merged_train_parts.append(loc + int(cum_starts[int(seg_idx)]))
            merged_train = np.concatenate(merged_train_parts) if merged_train_parts else np.array([], dtype=np.int64)

            pr = float(
                mm.presence_ratio(
                    merged_train,
                    total_length_samples,
                    bin_edges=bin_edges,
                    bin_n_spikes_thres=int(bin_n_spikes_thres),
                )
            )
        rows["presence_ratio"].append(pr)

        # rp contamination (match SI: count violations within each segment)
        if n_spikes == 0 or (t_r - t_c) <= 0:
            rows["rp_violations"].append(float("nan"))
            rows["rp_contamination"].append(float("nan"))
        else:
            n_v_total = 0
            for seg_idx in seg_indices:
                loc = samples_local_by_seg.get(int(seg_idx))
                if loc is None:
                    continue
                n_v_total += _count_rp_violations_pairs(spike_train=loc, t_r=int(t_r))

            n_v = int(n_v_total)
            rows["rp_violations"].append(int(n_v))
            N = float(n_spikes)
            T = float(total_length_samples)
            D = 1.0 - float(n_v) * (T - 2.0 * N * float(t_c)) / (N**2 * float(t_r - t_c))
            rows["rp_contamination"].append(1.0 - math.sqrt(D) if D >= 0 else 1.0)

        # amplitude stats
        if n_spikes == 0:
            rows["amplitude_mean"].append(float("nan"))
            rows["amplitude_median"].append(float("nan"))
        else:
            rows["amplitude_mean"].append(float(np.mean(amps)))
            rows["amplitude_median"].append(float(np.median(amps)))

        # amplitude CV (match SI: compute temporal bin size from overall FR,
        # then compute per-segment spreads using segment-local bins)
        if n_spikes == 0 or not np.isfinite(fr) or fr <= 0:
            rows["amplitude_cv_median"].append(float("nan"))
            rows["amplitude_cv_range"].append(float("nan"))
        else:
            temporal_bin_size_samples = int((average_num_spikes_per_bin / fr) * fs_hz)
            if temporal_bin_size_samples <= 0:
                rows["amplitude_cv_median"].append(float("nan"))
                rows["amplitude_cv_range"].append(float("nan"))
            else:
                amp_mean = float(np.abs(np.mean(amps)))
                amp_spreads: list[float] = []
                if amp_mean != 0:
                    for i_seg, seg_idx in enumerate(seg_indices):
                        loc = samples_local_by_seg.get(int(seg_idx))
                        if loc is None:
                            continue
                        a = amps_by_seg.get(int(seg_idx))
                        if a is None:
                            continue
                        seg_len = int(seg_lengths[i_seg])
                        sample_bin_edges = np.arange(0, seg_len + 1, temporal_bin_size_samples, dtype=np.int64)
                        for t0, t1 in zip(sample_bin_edges[:-1], sample_bin_edges[1:]):
                            i0 = int(np.searchsorted(loc, int(t0)))
                            i1 = int(np.searchsorted(loc, int(t1)))
                            amp_spreads.append(float(np.std(a[i0:i1]) / amp_mean))

                if len(amp_spreads) < min_num_bins:
                    rows["amplitude_cv_median"].append(float("nan"))
                    rows["amplitude_cv_range"].append(float("nan"))
                else:
                    rows["amplitude_cv_median"].append(float(np.median(amp_spreads)))
                    p_lo, p_hi = float(percentiles[0]), float(percentiles[1])
                    rows["amplitude_cv_range"].append(
                        float(np.percentile(amp_spreads, p_hi) - np.percentile(amp_spreads, p_lo))
                    )

        # Debug: per-unit recomputed/adjusted values used for merged curation.
        # This is intentionally verbose when logger is in DEBUG level.
        # IMPORTANT: we *store* values here but only *log* them after fallback.
        try:
            if hasattr(logger, "debug"):
                debug_pick_stats_by_unit[unit_id] = pick_stats_by_unit.get(unit_id, (0, 0, 0, 0, 0))

                rp_contam_v = rows["rp_contamination"][-1] if rows["rp_contamination"] else float("nan")
                rp_v = rows["rp_violations"][-1] if rows["rp_violations"] else float("nan")
                amp_mean_v = rows["amplitude_mean"][-1] if rows["amplitude_mean"] else float("nan")
                amp_median_v = rows["amplitude_median"][-1] if rows["amplitude_median"] else float("nan")
                amp_cv_med_v = rows["amplitude_cv_median"][-1] if rows["amplitude_cv_median"] else float("nan")

                # OLD values: direct concat quality_metrics (plus rp_contam/amp_mean computed here).
                old_num_spikes = float("nan")
                old_fr = float("nan")
                old_pr = float("nan")
                old_amp_median = float("nan")
                old_amp_cv_median = float("nan")
                old_rp_contam = float("nan")
                old_rp_v = float("nan")
                old_amp_mean = float("nan")
                try:
                    if concat_qm is not None and unit_id in concat_qm.index:
                        row = concat_qm.loc[unit_id]
                        old_num_spikes = row.get("num_spikes", float("nan"))
                        old_fr = row.get("firing_rate", float("nan"))
                        old_pr = row.get("presence_ratio", float("nan"))
                        old_amp_median = row.get("amplitude_median", float("nan"))
                        old_amp_cv_median = row.get("amplitude_cv_median", float("nan"))
                except Exception:
                    pass

                try:
                    a0 = concat_amps_by_unit.get(unit_id)
                    if a0 is not None and len(a0) > 0:
                        old_amp_mean = float(np.mean(np.asarray(a0, dtype=float)))
                except Exception:
                    pass

                try:
                    # Old rp contamination treats concat as a single segment.
                    st0 = concat_analyzer.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)
                    st0 = np.asarray(st0, dtype=np.int64)
                    st0.sort()
                    n0 = int(st0.size)
                    if n0 > 0 and (t_r - t_c) > 0:
                        old_rp_v = int(_count_rp_violations_pairs(spike_train=st0, t_r=int(t_r)))
                        N0 = float(n0)
                        T0 = float(total_samples)
                        D0 = 1.0 - float(old_rp_v) * (T0 - 2.0 * N0 * float(t_c)) / (N0**2 * float(t_r - t_c))
                        old_rp_contam = 1.0 - math.sqrt(D0) if D0 >= 0 else 1.0
                except Exception:
                    pass

                debug_old_by_unit[unit_id] = {
                    "num_spikes": old_num_spikes,
                    "firing_rate": old_fr,
                    "presence_ratio": old_pr,
                    "rp_contamination": old_rp_contam,
                    "rp_violations": old_rp_v,
                    "amplitude_mean": old_amp_mean,
                    "amplitude_median": old_amp_median,
                    "amplitude_cv_median": old_amp_cv_median,
                }
                debug_new_pre_fallback_by_unit[unit_id] = {
                    "num_spikes": float(n_spikes),
                    "firing_rate": fr,
                    "presence_ratio": pr,
                    "rp_contamination": rp_contam_v,
                    "rp_violations": rp_v,
                    "amplitude_mean": amp_mean_v,
                    "amplitude_median": amp_median_v,
                    "amplitude_cv_median": amp_cv_med_v,
                    "amplitude_cv_range": rows["amplitude_cv_range"][-1]
                    if rows["amplitude_cv_range"]
                    else float("nan"),
                }
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("unit_id")

    # Optional fallback: if a recomputed value is NaN, use the concat analyzer
    # value when it exists and is not NaN. This keeps downstream curation stable
    # for edge cases (too-few-spikes, tiny segments, etc.) without overriding
    # valid recomputed values.
    try:
        if df is not None and (not df.empty):
            # Straight column-wise fallbacks where concat_qm has matching columns.
            if concat_qm is not None:
                for col in [
                    "firing_rate",
                    "presence_ratio",
                    "amplitude_median",
                    "amplitude_cv_median",
                    "amplitude_cv_range",
                ]:
                    try:
                        if col in df.columns and col in concat_qm.columns:
                            old_vals = concat_qm[col].reindex(df.index)
                            df[col] = df[col].where(~df[col].isna(), old_vals)
                    except Exception:
                        continue

            # amplitude_mean fallback: concat_qm typically doesn't include this.
            try:
                if "amplitude_mean" in df.columns and concat_amps_by_unit:
                    for unit_id in df.index:
                        try:
                            if not pd.isna(df.at[unit_id, "amplitude_mean"]):
                                continue
                            a0 = concat_amps_by_unit.get(unit_id)
                            if a0 is None:
                                continue
                            a0 = np.asarray(a0, dtype=float)
                            if a0.size == 0:
                                continue
                            v = float(np.mean(a0))
                            if np.isfinite(v):
                                df.at[unit_id, "amplitude_mean"] = v
                        except Exception:
                            continue
            except Exception:
                pass

            # rp_contamination/rp_violations fallback: concat_qm doesn't store
            # rp_contamination, so recompute the concat-based version only when
            # needed.
            try:
                if ("rp_contamination" in df.columns) or ("rp_violations" in df.columns):
                    needs = []
                    for unit_id in df.index:
                        try:
                            need = False
                            if "rp_contamination" in df.columns and pd.isna(df.at[unit_id, "rp_contamination"]):
                                need = True
                            if "rp_violations" in df.columns and pd.isna(df.at[unit_id, "rp_violations"]):
                                need = True
                            if need:
                                needs.append(unit_id)
                        except Exception:
                            continue

                    if needs and (t_r - t_c) > 0:
                        for unit_id in needs:
                            try:
                                st0 = concat_analyzer.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)
                                st0 = np.asarray(st0, dtype=np.int64)
                                if st0.size == 0:
                                    continue
                                st0.sort()
                                n0 = int(st0.size)
                                old_rp_v = int(_count_rp_violations_pairs(spike_train=st0, t_r=int(t_r)))

                                # Fill violations count if missing.
                                if "rp_violations" in df.columns and pd.isna(df.at[unit_id, "rp_violations"]):
                                    df.at[unit_id, "rp_violations"] = float(old_rp_v)

                                # Fill contamination if missing.
                                if "rp_contamination" in df.columns and pd.isna(df.at[unit_id, "rp_contamination"]):
                                    N0 = float(n0)
                                    T0 = float(total_samples)
                                    D0 = 1.0 - float(old_rp_v) * (T0 - 2.0 * N0 * float(t_c)) / (N0**2 * float(t_r - t_c))
                                    old_rp_contam = 1.0 - math.sqrt(D0) if D0 >= 0 else 1.0
                                    if np.isfinite(float(old_rp_contam)):
                                        df.at[unit_id, "rp_contamination"] = float(old_rp_contam)
                            except Exception:
                                continue
            except Exception:
                pass
    except Exception:
        pass

    # Emit debug AFTER fallback so NEW values reflect what downstream uses.
    try:
        if hasattr(logger, "debug") and df is not None and (not df.empty) and debug_old_by_unit:

            def _fmt(x: Any) -> str:
                try:
                    xf = float(x)
                    if not np.isfinite(xf):
                        return "nan"
                    return f"{xf:.6g}"
                except Exception:
                    return str(x)

            def _fmtw(x: Any, width: int = 10) -> str:
                s = _fmt(x)
                try:
                    float(s)
                    return f"{s:>{width}}"
                except Exception:
                    return f"{s:<{width}}"

            fmt_line = (
                "QM unit=%s %-3s"
                " n=%s fr=%s pr=%s"
                " rpC=%s rpV=%s"
                " ampMean=%s ampMed=%s ampCvMed=%s"
                " segWin=%s conWin=%s tie=%s onlySeg=%s onlyCon=%s"
            )

            # Determine whether fallback actually changed anything.
            fallback_units = 0
            fallback_fields = 0
            for unit_id in df.index:
                pre = debug_new_pre_fallback_by_unit.get(unit_id)
                if pre is None:
                    continue
                changed_cols = []
                for col in [
                    "firing_rate",
                    "presence_ratio",
                    "rp_contamination",
                    "rp_violations",
                    "amplitude_mean",
                    "amplitude_median",
                    "amplitude_cv_median",
                    "amplitude_cv_range",
                ]:
                    try:
                        if col not in df.columns:
                            continue
                        pre_v = pre.get(col, float("nan"))
                        post_v = df.at[unit_id, col]
                        if pd.isna(pre_v) and (not pd.isna(post_v)):
                            changed_cols.append(col)
                    except Exception:
                        continue

                if changed_cols:
                    fallback_units += 1
                    fallback_fields += len(changed_cols)
                    logger.debug(
                        "QM NaN-fallback activated unit=%s cols=%s",
                        str(unit_id),
                        ",".join(changed_cols),
                    )

            if fallback_units > 0:
                logger.debug(
                    "QM NaN-fallback filled %d fields across %d units",
                    int(fallback_fields),
                    int(fallback_units),
                )

            # Print aligned OLD vs NEW (post-fallback) rows.
            for unit_id in df.index:
                old = debug_old_by_unit.get(unit_id)
                if old is None:
                    continue

                seg_wins, con_wins, ties, only_seg, only_con = debug_pick_stats_by_unit.get(unit_id, (0, 0, 0, 0, 0))

                logger.debug(
                    fmt_line,
                    str(unit_id),
                    "OLD",
                    _fmtw(old.get("num_spikes"), 8),
                    _fmtw(old.get("firing_rate"), 9),
                    _fmtw(old.get("presence_ratio"), 7),
                    _fmtw(old.get("rp_contamination"), 8),
                    _fmtw(old.get("rp_violations"), 5),
                    _fmtw(old.get("amplitude_mean"), 10),
                    _fmtw(old.get("amplitude_median"), 10),
                    _fmtw(old.get("amplitude_cv_median"), 9),
                    _fmtw("-", 6),
                    _fmtw("-", 6),
                    _fmtw("-", 4),
                    _fmtw("-", 7),
                    _fmtw("-", 7),
                )
                logger.debug(
                    fmt_line,
                    str(unit_id),
                    "NEW",
                    _fmtw(df.at[unit_id, "num_spikes"] if "num_spikes" in df.columns else float("nan"), 8),
                    _fmtw(df.at[unit_id, "firing_rate"] if "firing_rate" in df.columns else float("nan"), 9),
                    _fmtw(df.at[unit_id, "presence_ratio"] if "presence_ratio" in df.columns else float("nan"), 7),
                    _fmtw(df.at[unit_id, "rp_contamination"] if "rp_contamination" in df.columns else float("nan"), 8),
                    _fmtw(df.at[unit_id, "rp_violations"] if "rp_violations" in df.columns else float("nan"), 5),
                    _fmtw(df.at[unit_id, "amplitude_mean"] if "amplitude_mean" in df.columns else float("nan"), 10),
                    _fmtw(df.at[unit_id, "amplitude_median"] if "amplitude_median" in df.columns else float("nan"), 10),
                    _fmtw(df.at[unit_id, "amplitude_cv_median"] if "amplitude_cv_median" in df.columns else float("nan"), 9),
                    _fmtw(seg_wins, 6),
                    _fmtw(con_wins, 6),
                    _fmtw(ties, 4),
                    _fmtw(only_seg, 7),
                    _fmtw(only_con, 7),
                )

            # Aggregate selection stats across units for quick sanity-checks.
            try:
                tot_seg = 0
                tot_con = 0
                tot_tie = 0
                tot_only_seg = 0
                tot_only_con = 0
                for unit_id in df.index:
                    seg_wins, con_wins, ties, only_seg, only_con = debug_pick_stats_by_unit.get(unit_id, (0, 0, 0, 0, 0))
                    tot_seg += int(seg_wins)
                    tot_con += int(con_wins)
                    tot_tie += int(ties)
                    tot_only_seg += int(only_seg)
                    tot_only_con += int(only_con)

                logger.debug(
                    "QM pick-stats total segWin=%d conWin=%d tie=%d onlySeg=%d onlyCon=%d (units=%d)",
                    int(tot_seg),
                    int(tot_con),
                    int(tot_tie),
                    int(tot_only_seg),
                    int(tot_only_con),
                    int(len(df.index)),
                )
            except Exception:
                pass
    except Exception:
        pass
    return df
