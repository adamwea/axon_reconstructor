from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

from .segments import (
    _append_segment_skip_summary,
    _append_segment_summary,
    _compute_segment_maxwell_intervals,
    _count_spikes_in_concat_window,
    _parse_concat_epoch_segment,
)
from .utils import (
    _load_preprocess_requested_cfg,
    _load_preprocessed_segment_recording,
    _load_raw_segment_recording_segment_channels,
    _to_numpy_sorting,
)


def _best_ptp_channel_by_unit_from_templates(*, analyzer: Any) -> dict[Any, tuple[float, Any]]:
    """Return {unit_id: (best_ptp_uv, best_channel_id)} from analyzer templates.

    Notes:
    - Uses templates (mean waveforms) since they exist for both concat and segment analyzers.
    - Handles both dense (ndarray) and sparse (dict) template representations.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        return {}

    try:
        ch_ids = list(analyzer.get_channel_ids())
    except Exception:
        try:
            ch_ids = list(analyzer.recording.get_channel_ids())
        except Exception:
            ch_ids = None

    try:
        ext = analyzer.get_extension("templates")
        templates = ext.get_data()
    except Exception:
        templates = None
        try:
            templates = analyzer.get_templates()
        except Exception:
            return {}

    try:
        unit_ids = list(analyzer.unit_ids)
    except Exception:
        try:
            unit_ids = list(analyzer.sorting.get_unit_ids())
        except Exception:
            unit_ids = []

    out: dict[Any, tuple[float, Any]] = {}

    def _best_for_template(tmpl_2d: "np.ndarray") -> tuple[float, int] | None:
        if tmpl_2d.ndim != 2:
            return None
        # tmpl_2d: (n_samples, n_channels)
        ptp = np.nanmax(tmpl_2d, axis=0) - np.nanmin(tmpl_2d, axis=0)
        if ptp.size == 0 or not np.any(np.isfinite(ptp)):
            return None
        j = int(np.nanargmax(ptp))
        v = float(ptp[j])
        if not np.isfinite(v):
            return None
        return v, j

    if isinstance(templates, dict):
        for u in unit_ids:
            try:
                tmpl = np.asarray(templates[u])
            except Exception:
                continue
            best = _best_for_template(tmpl)
            if best is None:
                continue
            v, j = best
            if ch_ids is not None and 0 <= j < len(ch_ids):
                out[u] = (float(v), ch_ids[j])
            else:
                out[u] = (float(v), int(j))
        return out

    arr = np.asarray(templates)
    if arr.ndim != 3:
        return out

    n_units, _, n_ch = arr.shape
    if n_units != len(unit_ids):
        unit_ids = [i for i in range(int(n_units))]

    for i, u in enumerate(unit_ids):
        try:
            tmpl = np.asarray(arr[int(i)])
        except Exception:
            continue
        best = _best_for_template(tmpl)
        if best is None:
            continue
        v, j = best
        if ch_ids is not None and 0 <= j < len(ch_ids):
            out[u] = (float(v), ch_ids[j])
        else:
            out[u] = (float(v), int(j))

    return out


def _warn_early_negative_peaks_from_templates(*, analyzer: Any, window: Any, logger: Any) -> None:
    """Warn if mean waveforms appear clipped at the start.

    Scientific rationale:
    - Spike times are usually referenced to a unit's "main" channel.
    - Other channels can legitimately peak earlier/later due to propagation.
    - However, if a non-trivial fraction of channels have their negative peak at
      the first 0-1 samples, it's often a sign that ms_before is too small (peak
      clipped) or that spike-time alignment is off.

    This is a lightweight QC check using the computed templates (mean waveforms)
    rather than iterating over all spike snippets.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        ext = analyzer.get_extension("templates")
        templates = ext.get_data()
    except Exception:
        templates = None
        try:
            templates = analyzer.get_templates()
        except Exception:
            return

    try:
        unit_ids = list(analyzer.unit_ids)
    except Exception:
        try:
            unit_ids = list(analyzer.sorting.get_unit_ids())
        except Exception:
            unit_ids = []

    # IMPORTANT: SpikeInterface templates are often *sparse* (many channels are
    # exactly/near-zero for compute efficiency). Those near-zero channels will
    # trivially have their argmin at sample 0 and would create a massive false
    # positive. So we only evaluate channels with non-trivial PTP.

    # Thresholds tuned to catch obvious clipping without being noisy.
    early_samples = 1
    warn_frac = 0.05
    warn_abs = 3

    # Channel activity thresholding: active if PTP exceeds both an absolute and
    # relative threshold (relative to the unit's strongest channel).
    ptp_abs_uv = 1.0
    ptp_rel = 0.02

    def _active_mask_from_template(tmpl_2d: "np.ndarray") -> "np.ndarray":
        ptp = np.nanmax(tmpl_2d, axis=0) - np.nanmin(tmpl_2d, axis=0)
        max_ptp = float(np.nanmax(ptp)) if ptp.size else float("nan")
        rel_thresh = float(ptp_rel) * max_ptp if np.isfinite(max_ptp) else float("inf")
        thr = max(float(ptp_abs_uv), float(rel_thresh))
        return ptp > thr

    if isinstance(templates, dict):
        for u in unit_ids:
            try:
                tmpl = np.asarray(templates[u])
            except Exception:
                continue
            if tmpl.ndim != 2:
                continue
            n_samples, n_ch = tmpl.shape
            if n_samples < 3 or n_ch < 1:
                continue
            try:
                active = _active_mask_from_template(tmpl)
                active_n = int(np.sum(active))
                if active_n <= 0:
                    continue
                imin = np.nanargmin(tmpl[:, active], axis=0)
            except Exception:
                continue

            n0 = int(np.sum(imin == 0))
            n_early = int(np.sum(imin <= int(early_samples)))
            if n_early >= warn_abs or (active_n > 0 and (n_early / float(active_n)) >= warn_frac):
                logger.warning(
                    "Unit %s: %d/%d active channels have neg-peak at <=%d samples (sample0=%d). "
                    "This can indicate waveform clipping (ms_before too small) or misalignment. "
                    "Consider increasing ms_before (currently %.3f ms) or reviewing spike-time alignment.",
                    u,
                    n_early,
                    int(active_n),
                    int(early_samples),
                    int(n0),
                    float(getattr(window, "ms_before", float("nan"))),
                )
        return

    templates_arr = np.asarray(templates)
    if templates_arr.ndim != 3:
        return

    n_units, n_samples, n_ch = templates_arr.shape
    if n_units != len(unit_ids):
        # Fall back to best-effort indexing if ordering isn't available.
        unit_ids = [str(i) for i in range(int(n_units))]

    for i, u in enumerate(unit_ids):
        tmpl = templates_arr[i]
        if tmpl.shape != (n_samples, n_ch):
            continue
        try:
            active = _active_mask_from_template(tmpl)
            active_n = int(np.sum(active))
            if active_n <= 0:
                continue
            imin = np.nanargmin(tmpl[:, active], axis=0)
        except Exception:
            continue

        n0 = int(np.sum(imin == 0))
        n_early = int(np.sum(imin <= int(early_samples)))
        if n_early >= warn_abs or (active_n > 0 and (n_early / float(active_n)) >= warn_frac):
            logger.warning(
                "Unit %s: %d/%d active channels have neg-peak at <=%d samples (sample0=%d). "
                "This can indicate waveform clipping (ms_before too small) or misalignment. "
                "Consider increasing ms_before (currently %.3f ms) or reviewing spike-time alignment.",
                u,
                n_early,
                int(active_n),
                int(early_samples),
                int(n0),
                float(getattr(window, "ms_before", float("nan"))),
            )


def _extract_concat_waveforms(
    *,
    inputs,
    filtered_sorting: Any,
    recording: Any,
    concat_waveforms_dir,
    window,
    logger: Any,
) -> None:
    import shutil

    import spikeinterface.full as si  # type: ignore[import-not-found]

    if concat_waveforms_dir.exists() and inputs.force_restart:
        shutil.rmtree(concat_waveforms_dir)

    if concat_waveforms_dir.exists() and not inputs.force_restart:
        try:
            logger.info("Reusing existing concat waveforms analyzer -> %s", concat_waveforms_dir)
            concat_analyzer = si.load_sorting_analyzer(concat_waveforms_dir)
            _warn_early_negative_peaks_from_templates(analyzer=concat_analyzer, window=window, logger=logger)
            return _best_ptp_channel_by_unit_from_templates(analyzer=concat_analyzer)
        except Exception:
            logger.warning(
                "Failed to load existing concat analyzer at %s; recomputing concat waveforms.",
                concat_waveforms_dir,
                exc_info=True,
            )
            try:
                shutil.rmtree(concat_waveforms_dir)
            except Exception:
                pass

    logger.info("Extracting concat waveforms -> %s", concat_waveforms_dir)
    concat_analyzer = si.create_sorting_analyzer(
        filtered_sorting,
        recording,
        format="binary_folder",
        folder=concat_waveforms_dir,
        return_in_uV=True,
    )
    random_spikes_params = {
        "method": "uniform",
        "seed": 0,
    }
    if inputs.max_spikes_per_unit is not None and int(inputs.max_spikes_per_unit) >= 0:
        random_spikes_params["max_spikes_per_unit"] = int(inputs.max_spikes_per_unit)

    concat_analyzer.compute(
        ["random_spikes", "waveforms"],
        extension_params={
            "random_spikes": random_spikes_params,
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
            "unit_locations",
        ],
        extension_params={
            "unit_locations": {"method": "monopolar_triangulation"},
        },
        verbose=False,
        n_jobs=int(inputs.n_jobs),
    )

    _warn_early_negative_peaks_from_templates(analyzer=concat_analyzer, window=window, logger=logger)

    return _best_ptp_channel_by_unit_from_templates(analyzer=concat_analyzer)


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
    logger: Any,
    channel_groups: dict[str, Any] | None = None,
    checkpoint_completed_sources: set[str] | None = None,
    on_segment_completed: Callable[[str], None] | None = None,
) -> dict[Any, tuple[float, Any, str]]:
    """Extract per-segment waveforms and return best channel info across segments.

    Returns:
        best_by_unit: {unit_id: (best_ptp_uv, best_channel_id, source_name)}
    """

    best_by_unit: dict[Any, tuple[float, Any, str]] = {}

    requested_cfg = None
    try:
        preprocess_dir = getattr(epochs, "preprocess_dir", None)
        if preprocess_dir is not None:
            requested_cfg = _load_preprocess_requested_cfg(preprocess_dir=preprocess_dir)
    except Exception:
        requested_cfg = None

    # Resampling params (best-effort); if preprocessing upsampled the concatenated recording,
    # we must resample each raw segment to the same fs as `recording`/`window.fs_hz`.
    target_fs_hz = float(getattr(window, "fs_hz", recording.get_sampling_frequency()))
    seg_resample_margin_ms = 100.0
    seg_resample_dtype = None
    if isinstance(requested_cfg, dict):
        try:
            seg_resample_margin_ms = float(requested_cfg.get("temporal_resample_margin_ms", seg_resample_margin_ms))
        except Exception:
            seg_resample_margin_ms = 100.0
        try:
            seg_resample_dtype = requested_cfg.get("temporal_resample_dtype")
            if seg_resample_dtype is not None:
                seg_resample_dtype = str(seg_resample_dtype)
        except Exception:
            seg_resample_dtype = None

    # Channel-group bookkeeping for documentation/QC.
    # We track *electrode ids* when they are available via contact_vector['electrode'].
    seg_electrode_sets: dict[str, set[int]] = {}
    seg_channel_id_sets: dict[str, set[Any]] = {}
    seg_additional_electrode_sets: dict[str, set[int]] = {}

    if inputs.per_segment and epochs.concat_epochs:
        segment_waveforms_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        import spikeinterface.full as si  # type: ignore[import-not-found]

        segment_manifest: dict[str, Any] = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "force_restart": bool(getattr(inputs, "force_restart", False)),
            "recompute_channel_groups_for_reused_segments": bool(
                getattr(inputs, "recompute_channel_groups_for_reused_segments", False)
            ),
            "segments": [],
            "summary": {
                "n_expected": int(len(list(epochs.concat_epochs))),
                "n_processed": 0,
                "n_reused": 0,
                "n_computed": 0,
                "n_skipped": 0,
                "timing_seconds": {
                    "load_preprocess": 0.0,
                    "spike_prep": 0.0,
                    "analyzer": 0.0,
                    "postprocess": 0.0,
                    "total": 0.0,
                },
            },
        }

        cache_numpy = None
        filtered_spike_cache: dict[Any, Any] = {}
        unfiltered_spike_cache: dict[Any, Any] = {}
        try:
            import numpy as np  # type: ignore[import-not-found]

            cache_numpy = np

            for u in filtered_sorting.get_unit_ids():
                arr = np.asarray(filtered_sorting.get_unit_spike_train(u), dtype=np.int64)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                filtered_spike_cache[u] = arr

            for u in sorting_unfiltered.get_unit_ids():
                arr = np.asarray(sorting_unfiltered.get_unit_spike_train(u), dtype=np.int64)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                unfiltered_spike_cache[u] = arr
        except Exception:
            cache_numpy = None
            filtered_spike_cache = {}
            unfiltered_spike_cache = {}

        def _count_spikes_cached(*, cache: dict[Any, Any], start: int, end: int) -> int:
            if cache_numpy is None or not cache:
                return -1
            total = 0
            for arr in cache.values():
                try:
                    left = int(cache_numpy.searchsorted(arr, int(start), side="left"))
                    right = int(cache_numpy.searchsorted(arr, int(end), side="left"))
                    total += int(max(0, right - left))
                except Exception:
                    continue
            return int(total)

        logger.info("Extracting per-segment waveforms -> %s", segment_waveforms_dir)
        for seg in epochs.concat_epochs:
            seg_t_total_start = perf_counter()
            seg_t_load_preprocess = 0.0
            seg_t_spike_prep = 0.0
            seg_t_analyzer = 0.0
            seg_t_postprocess = 0.0

            spec = _parse_concat_epoch_segment(seg=seg, segment_waveforms_dir=segment_waveforms_dir)
            if spec is None:
                continue

            seg_index = int(spec.segment_index)
            rec_name = str(spec.rec_name)
            start = int(spec.start_sample_concat)
            end_epoch = int(spec.end_sample_concat)
            end = int(end_epoch)
            seg_dir = spec.seg_dir

            if seg_dir.exists() and inputs.force_restart:
                shutil.rmtree(seg_dir)

            source_name = f"seg{int(seg_index):02d}_{rec_name}"
            was_checkpoint_completed = bool(checkpoint_completed_sources and source_name in checkpoint_completed_sources)

            manifest_entry: dict[str, Any] = {
                "segment_index": int(seg_index),
                "rec_name": str(rec_name),
                "source_name": str(source_name),
                "segment_dir": str(seg_dir),
                "start_sample_concat": int(start),
                "end_sample_concat": int(end_epoch),
                "checkpoint_completed": bool(was_checkpoint_completed),
                "status": "pending",
                "timing_seconds": {},
            }

            if (
                seg_dir.exists()
                and not inputs.force_restart
                and not bool(getattr(inputs, "recompute_channel_groups_for_reused_segments", False))
            ):
                try:
                    seg_analyzer_existing = si.load_sorting_analyzer(seg_dir)

                    try:
                        seg_channel_ids_existing = list(seg_analyzer_existing.get_channel_ids())
                    except Exception:
                        seg_channel_ids_existing = []
                    if seg_channel_ids_existing:
                        seg_channel_id_sets[str(source_name)] = set(seg_channel_ids_existing)
                        try:
                            seg_additional_electrode_sets[str(source_name)] = set(int(x) for x in seg_channel_ids_existing)
                        except Exception:
                            pass

                    try:
                        seg_best_existing = _best_ptp_channel_by_unit_from_templates(analyzer=seg_analyzer_existing)
                        for u, (ptp_uv, ch_id) in seg_best_existing.items():
                            try:
                                ptp_f = float(ptp_uv)
                            except Exception:
                                continue
                            prev = best_by_unit.get(u)
                            if prev is None or ptp_f > float(prev[0]):
                                best_by_unit[u] = (ptp_f, ch_id, str(source_name))
                    except Exception:
                        pass

                    try:
                        in_seg_unfiltered_cached = _count_spikes_cached(cache=unfiltered_spike_cache, start=start, end=end)
                        in_seg_filtered_cached = _count_spikes_cached(cache=filtered_spike_cache, start=start, end=end)
                        if in_seg_unfiltered_cached >= 0 and in_seg_filtered_cached >= 0:
                            logger.info(
                                "Segment %s: reuse existing analyzer (spikes in concat window: unfiltered=%d, after-concat-filter=%d)",
                                rec_name,
                                int(in_seg_unfiltered_cached),
                                int(in_seg_filtered_cached),
                            )
                    except Exception:
                        pass

                    _append_segment_skip_summary(
                        filtering_summary=filtering_summary,
                        seg_index=seg_index,
                        rec_name=rec_name,
                        raw_channels_total=None,
                        excluded_common_channels_total=None,
                        kept_additional_channels_total=None,
                        skipped_reason="reused_existing_analyzer",
                    )

                    logger.info(
                        "Segment %s: reusing existing waveform analyzer -> %s%s",
                        rec_name,
                        seg_dir,
                        " (from checkpoint)" if was_checkpoint_completed else "",
                    )

                    seg_t_total = float(perf_counter() - seg_t_total_start)
                    manifest_entry["status"] = "reused_existing_analyzer"
                    manifest_entry["timing_seconds"] = {
                        "load_preprocess": float(seg_t_load_preprocess),
                        "spike_prep": float(seg_t_spike_prep),
                        "analyzer": float(seg_t_analyzer),
                        "postprocess": float(seg_t_postprocess),
                        "total": float(seg_t_total),
                    }
                    segment_manifest["segments"].append(manifest_entry)

                    segment_manifest["summary"]["n_processed"] += 1
                    segment_manifest["summary"]["n_reused"] += 1
                    segment_manifest["summary"]["timing_seconds"]["total"] += float(seg_t_total)

                    logger.info(
                        "Segment %s timings (s): load_preprocess=%.2f spike_prep=%.2f analyzer=%.2f postprocess=%.2f total=%.2f status=%s",
                        rec_name,
                        float(seg_t_load_preprocess),
                        float(seg_t_spike_prep),
                        float(seg_t_analyzer),
                        float(seg_t_postprocess),
                        float(seg_t_total),
                        "reused_existing_analyzer",
                    )

                    if on_segment_completed is not None:
                        try:
                            on_segment_completed(str(source_name))
                        except Exception:
                            pass
                    continue
                except Exception:
                    logger.warning(
                        "Segment %s: existing analyzer at %s is not loadable; recomputing.",
                        rec_name,
                        seg_dir,
                        exc_info=True,
                    )
                    try:
                        shutil.rmtree(seg_dir)
                    except Exception:
                        pass

            seg_t0 = perf_counter()
            seg_rec = None
            preprocess_dir = getattr(epochs, "preprocess_dir", None)
            if preprocess_dir is not None:
                try:
                    seg_rec = _load_preprocessed_segment_recording(
                        preprocess_dir=preprocess_dir,
                        segment_index=int(seg_index),
                        rec_name=str(rec_name),
                        target_sampling_frequency_hz=float(target_fs_hz),
                        temporal_resample_margin_ms=float(seg_resample_margin_ms),
                        temporal_resample_dtype=(str(seg_resample_dtype) if seg_resample_dtype is not None else None),
                    )
                    logger.info(
                        "Segment %s: loaded preprocessed segment recording from stg1 outputs",
                        rec_name,
                    )
                except Exception as e:
                    logger.warning(
                        "Segment %s: failed loading stg1 preprocessed segment recording; "
                        "falling back to raw+local preprocessing (%s)",
                        rec_name,
                        e,
                    )

            if seg_rec is None:
                seg_rec = _load_raw_segment_recording_segment_channels(
                    h5_path=inputs.h5_path,
                    stream_id=inputs.stream_id,
                    rec_name=rec_name,
                    center_chunk_size=10_000,
                    preprocess_like_mea_analysis=bool(
                        getattr(inputs, "per_segment_preprocess_like_mea_analysis", True)
                    ),
                    target_sampling_frequency_hz=float(target_fs_hz),
                    temporal_resample_margin_ms=float(seg_resample_margin_ms),
                    temporal_resample_dtype=(str(seg_resample_dtype) if seg_resample_dtype is not None else None),
                )
            seg_t_load_preprocess += float(perf_counter() - seg_t0)

            # Attempt to record electrode ids (preferred) and raw channel ids (fallback).
            electrodes_all: list[int] | None = None
            try:
                import numpy as np  # type: ignore[import-not-found]

                cv_all = seg_rec.get_property("contact_vector")
                electrodes_all = [int(x) for x in np.asarray(cv_all["electrode"], dtype=int).tolist()]
            except Exception:
                electrodes_all = None

            try:
                seg_channel_ids_all = list(seg_rec.get_channel_ids())
            except Exception:
                seg_channel_ids_all = []

            if electrodes_all is not None and len(electrodes_all) == len(seg_channel_ids_all):
                seg_electrode_sets[str(source_name)] = set(int(e) for e in electrodes_all)
            seg_channel_id_sets[str(source_name)] = set(seg_channel_ids_all)

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
                    logger.warning(
                        "Segment %s: could not map channel ids to electrode ids; cannot reliably exclude common channels. "
                        "Per-segment waveforms may include common channels.",
                        str(rec_name),
                    )

            # After selection, track additional-channel electrode ids when possible.
            try:
                import numpy as np  # type: ignore[import-not-found]

                cv_sel = seg_rec.get_property("contact_vector")
                electrodes_sel = [int(x) for x in np.asarray(cv_sel["electrode"], dtype=int).tolist()]
                if electrodes_sel:
                    seg_additional_electrode_sets[str(source_name)] = set(int(e) for e in electrodes_sel)
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

                        seg_t_total = float(perf_counter() - seg_t_total_start)
                        manifest_entry["status"] = "skipped_no_additional_channels"
                        manifest_entry["timing_seconds"] = {
                            "load_preprocess": float(seg_t_load_preprocess),
                            "spike_prep": float(seg_t_spike_prep),
                            "analyzer": float(seg_t_analyzer),
                            "postprocess": float(seg_t_postprocess),
                            "total": float(seg_t_total),
                        }
                        segment_manifest["segments"].append(manifest_entry)
                        segment_manifest["summary"]["n_processed"] += 1
                        segment_manifest["summary"]["n_skipped"] += 1
                        segment_manifest["summary"]["timing_seconds"]["load_preprocess"] += float(seg_t_load_preprocess)
                        segment_manifest["summary"]["timing_seconds"]["total"] += float(seg_t_total)

                        logger.info(
                            "Segment %s timings (s): load_preprocess=%.2f spike_prep=%.2f analyzer=%.2f postprocess=%.2f total=%.2f status=%s",
                            rec_name,
                            float(seg_t_load_preprocess),
                            float(seg_t_spike_prep),
                            float(seg_t_analyzer),
                            float(seg_t_postprocess),
                            float(seg_t_total),
                            "skipped_no_additional_channels",
                        )
                        continue
                except Exception:
                    pass

            seg_len_expected = int(end_epoch - start)
            try:
                seg_len = int(seg_rec.get_num_samples())
                # Clamp concat-window end to the actual segment length (resampling can
                # introduce +/- 1 sample drift vs simple ratio scaling).
                end_clamped = min(int(end_epoch), int(start) + int(seg_len))
                if end_clamped != int(end_epoch) or seg_len != seg_len_expected:
                    logger.warning(
                        "Segment length mismatch for %s: raw=%d expected=%d (start=%d end_epoch=%d end_clamped=%d). Proceeding.",
                        rec_name,
                        seg_len,
                        seg_len_expected,
                        start,
                        end_epoch,
                        end_clamped,
                    )
                end = int(end_clamped)
            except Exception:
                seg_len = seg_len_expected
                end = int(min(int(end_epoch), int(start) + int(seg_len)))

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
                in_seg_unfiltered = _count_spikes_cached(cache=unfiltered_spike_cache, start=start, end=end)
                in_seg_filtered = _count_spikes_cached(cache=filtered_spike_cache, start=start, end=end)
                if in_seg_unfiltered < 0 or in_seg_filtered < 0:
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

            seg_t1 = perf_counter()
            for u in filtered_sorting.get_unit_ids():
                local_all: list[int]
                if cache_numpy is not None and u in filtered_spike_cache:
                    arr = filtered_spike_cache[u]
                    left = int(cache_numpy.searchsorted(arr, int(start), side="left"))
                    right = int(cache_numpy.searchsorted(arr, int(end), side="left"))
                    if right > left:
                        local_all = [int(x) for x in (arr[left:right] - int(start)).tolist()]
                    else:
                        local_all = []
                else:
                    st = filtered_sorting.get_unit_spike_train(u)
                    st_list = [int(x) for x in st]
                    local_all = [int(t - start) for t in st_list if start <= t < end]
                    local_all.sort()
                seg_spikes_total += len(local_all)

                if bool(getattr(inputs, "filter_by_segment_bounds", True)):
                    kept_edges: list[int] = []
                    for t_local in local_all:
                        if t_local - window.pre_samples < 0:
                            continue
                        if t_local + window.post_samples >= int(seg_len):
                            continue
                        kept_edges.append(int(t_local))

                    try:
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
                else:
                    kept_edges = [int(t_local) for t_local in local_all]

                removed_edge = int(len(local_all) - len(kept_edges))
                seg_removed_edge_total += int(removed_edge)
                seg_kept_total += int(len(kept_edges))

                unit_trains_seg[int(u)] = kept_edges

            seg_t_spike_prep += float(perf_counter() - seg_t1)

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

            # MEA_Analysis-style safety cleanup (segment-local):
            # even though we try to keep only in-bounds spikes above, this is a cheap
            # no-op in the common case and prevents analyzer-time indexing errors if
            # any spike times fall outside `seg_rec`.
            # Also drops units that ended up with zero kept spikes in this segment.
            # NOTE: upstream spikesorting may already take care of this already... but I guess it doesnt hurt.
            # -- aw 2026-02-01 21:51:12
            if bool(getattr(inputs, "segment_sort_safety_cleanup", True)):
                try:
                    import spikeinterface.full as si  # type: ignore[import-not-found]

                    seg_sort = si.remove_excess_spikes(seg_sort, seg_rec)
                    seg_sort = seg_sort.remove_empty_units()
                except Exception:
                    logger.debug(
                        "Segment %s: sorting cleanup (remove_excess_spikes/remove_empty_units) failed; continuing.",
                        rec_name,
                        exc_info=True,
                    )

            # SpikeInterface's SortingAnalyzer creation/auto-sparsity estimation
            # fails when there are zero units or zero spikes (e.g. empty segment
            # after edge/epoch filtering). Skip those segments.
            try:
                unit_ids = list(getattr(seg_sort, "unit_ids", []))
            except Exception:
                unit_ids = []
            if not unit_ids:
                try:
                    logger.info("Segment %s: no units after filtering; skipping per-segment waveforms.", rec_name)
                except Exception:
                    pass
                if on_segment_completed is not None:
                    try:
                        on_segment_completed(str(source_name))
                    except Exception:
                        pass

                seg_t_total = float(perf_counter() - seg_t_total_start)
                manifest_entry["status"] = "skipped_no_units_after_filtering"
                manifest_entry["timing_seconds"] = {
                    "load_preprocess": float(seg_t_load_preprocess),
                    "spike_prep": float(seg_t_spike_prep),
                    "analyzer": float(seg_t_analyzer),
                    "postprocess": float(seg_t_postprocess),
                    "total": float(seg_t_total),
                }
                segment_manifest["segments"].append(manifest_entry)
                segment_manifest["summary"]["n_processed"] += 1
                segment_manifest["summary"]["n_skipped"] += 1
                segment_manifest["summary"]["timing_seconds"]["load_preprocess"] += float(seg_t_load_preprocess)
                segment_manifest["summary"]["timing_seconds"]["spike_prep"] += float(seg_t_spike_prep)
                segment_manifest["summary"]["timing_seconds"]["total"] += float(seg_t_total)

                logger.info(
                    "Segment %s timings (s): load_preprocess=%.2f spike_prep=%.2f analyzer=%.2f postprocess=%.2f total=%.2f status=%s",
                    rec_name,
                    float(seg_t_load_preprocess),
                    float(seg_t_spike_prep),
                    float(seg_t_analyzer),
                    float(seg_t_postprocess),
                    float(seg_t_total),
                    "skipped_no_units_after_filtering",
                )
                continue

            try:
                has_any_spikes = False
                for u in unit_ids:
                    st = seg_sort.get_unit_spike_train(unit_id=u, segment_index=0)
                    if len(st):
                        has_any_spikes = True
                        break
                if not has_any_spikes:
                    try:
                        logger.info("Segment %s: no spikes after filtering; skipping per-segment waveforms.", rec_name)
                    except Exception:
                        pass
                    if on_segment_completed is not None:
                        try:
                            on_segment_completed(str(source_name))
                        except Exception:
                            pass

                    seg_t_total = float(perf_counter() - seg_t_total_start)
                    manifest_entry["status"] = "skipped_no_spikes_after_filtering"
                    manifest_entry["timing_seconds"] = {
                        "load_preprocess": float(seg_t_load_preprocess),
                        "spike_prep": float(seg_t_spike_prep),
                        "analyzer": float(seg_t_analyzer),
                        "postprocess": float(seg_t_postprocess),
                        "total": float(seg_t_total),
                    }
                    segment_manifest["segments"].append(manifest_entry)
                    segment_manifest["summary"]["n_processed"] += 1
                    segment_manifest["summary"]["n_skipped"] += 1
                    segment_manifest["summary"]["timing_seconds"]["load_preprocess"] += float(seg_t_load_preprocess)
                    segment_manifest["summary"]["timing_seconds"]["spike_prep"] += float(seg_t_spike_prep)
                    segment_manifest["summary"]["timing_seconds"]["total"] += float(seg_t_total)

                    logger.info(
                        "Segment %s timings (s): load_preprocess=%.2f spike_prep=%.2f analyzer=%.2f postprocess=%.2f total=%.2f status=%s",
                        rec_name,
                        float(seg_t_load_preprocess),
                        float(seg_t_spike_prep),
                        float(seg_t_analyzer),
                        float(seg_t_postprocess),
                        float(seg_t_total),
                        "skipped_no_spikes_after_filtering",
                    )
                    continue
            except Exception:
                # Best-effort: if we can't determine spike counts, continue.
                pass

            try:
                seg_sort.register_recording(seg_rec)
            except Exception:
                pass

            loaded_existing_seg_analyzer = None
            seg_t2 = perf_counter()
            if seg_dir.exists() and not inputs.force_restart:
                try:
                    loaded_existing_seg_analyzer = si.load_sorting_analyzer(seg_dir)
                    logger.info(
                        "Segment %s: reusing existing waveform analyzer -> %s%s",
                        rec_name,
                        seg_dir,
                        " (from checkpoint)" if was_checkpoint_completed else "",
                    )
                except Exception:
                    logger.warning(
                        "Segment %s: existing analyzer at %s is not loadable; recomputing.",
                        rec_name,
                        seg_dir,
                        exc_info=True,
                    )
                    try:
                        shutil.rmtree(seg_dir)
                    except Exception:
                        pass

            seg_analyzer = loaded_existing_seg_analyzer
            if seg_analyzer is None:
                seg_analyzer = si.create_sorting_analyzer(
                    seg_sort,
                    seg_rec,
                    format="binary_folder",
                    folder=seg_dir,
                    return_in_uV=True,
                )
                random_spikes_params = {
                    "method": "uniform",
                    "seed": 0,
                }
                if inputs.max_spikes_per_unit is not None and int(inputs.max_spikes_per_unit) >= 0:
                    random_spikes_params["max_spikes_per_unit"] = int(inputs.max_spikes_per_unit)

                seg_analyzer.compute(
                    ["random_spikes", "waveforms"],
                    extension_params={
                        "random_spikes": random_spikes_params,
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
                        "unit_locations",
                    ],
                    extension_params={
                        "unit_locations": {"method": "monopolar_triangulation"},
                    },
                    verbose=False,
                    n_jobs=max(1, int(inputs.n_jobs)),
                )

            seg_t_analyzer += float(perf_counter() - seg_t2)

            # Track best channel by PTP for this segment (mean template).
            seg_t3 = perf_counter()
            try:
                source_name = f"seg{int(seg_index):02d}_{rec_name}"
                seg_best = _best_ptp_channel_by_unit_from_templates(analyzer=seg_analyzer)
                for u, (ptp_uv, ch_id) in seg_best.items():
                    try:
                        ptp_f = float(ptp_uv)
                    except Exception:
                        continue
                    prev = best_by_unit.get(u)
                    if prev is None or ptp_f > float(prev[0]):
                        best_by_unit[u] = (ptp_f, ch_id, str(source_name))
            except Exception:
                pass

            # Deprecated (2026-01): we previously *flagged* extracted per-segment random_spikes
            # against segment-local Maxwell intervals after waveforms were computed. This does
            # not mutate the analyzer and is redundant when concat-time filtering is authoritative.
            if (
                bool(getattr(inputs, "deprecated_flag_segment_random_spikes_by_epochs", False))
                and inputs.filter_by_maxwell_epochs
                and seg_maxwell_intervals
            ):
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
                    logger.warning(
                        "DEPRECATED: Segment %s: flagged extracted random_spikes by maxwell epochs=%d (outside=%d edge=%d, selected_random_spikes=%d)",
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

            seg_t_postprocess += float(perf_counter() - seg_t3)

            seg_t_total = float(perf_counter() - seg_t_total_start)
            manifest_entry["status"] = "computed"
            manifest_entry["timing_seconds"] = {
                "load_preprocess": float(seg_t_load_preprocess),
                "spike_prep": float(seg_t_spike_prep),
                "analyzer": float(seg_t_analyzer),
                "postprocess": float(seg_t_postprocess),
                "total": float(seg_t_total),
            }
            segment_manifest["segments"].append(manifest_entry)
            segment_manifest["summary"]["n_processed"] += 1
            segment_manifest["summary"]["n_computed"] += 1
            segment_manifest["summary"]["timing_seconds"]["load_preprocess"] += float(seg_t_load_preprocess)
            segment_manifest["summary"]["timing_seconds"]["spike_prep"] += float(seg_t_spike_prep)
            segment_manifest["summary"]["timing_seconds"]["analyzer"] += float(seg_t_analyzer)
            segment_manifest["summary"]["timing_seconds"]["postprocess"] += float(seg_t_postprocess)
            segment_manifest["summary"]["timing_seconds"]["total"] += float(seg_t_total)

            logger.info(
                "Segment %s timings (s): load_preprocess=%.2f spike_prep=%.2f analyzer=%.2f postprocess=%.2f total=%.2f status=%s",
                rec_name,
                float(seg_t_load_preprocess),
                float(seg_t_spike_prep),
                float(seg_t_analyzer),
                float(seg_t_postprocess),
                float(seg_t_total),
                "computed",
            )

            if on_segment_completed is not None:
                try:
                    on_segment_completed(str(source_name))
                except Exception:
                    pass

        try:
            import json

            manifest_path = segment_waveforms_dir / "segment_execution_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(segment_manifest, f, indent=2)
            logger.info("Waveforms segment manifest written -> %s", manifest_path)

            ts = segment_manifest.get("summary", {}).get("timing_seconds", {})
            logger.info(
                "Per-segment timings aggregate (s): expected=%d processed=%d reused=%d computed=%d skipped=%d load_preprocess=%.2f spike_prep=%.2f analyzer=%.2f postprocess=%.2f total=%.2f",
                int(segment_manifest.get("summary", {}).get("n_expected", 0)),
                int(segment_manifest.get("summary", {}).get("n_processed", 0)),
                int(segment_manifest.get("summary", {}).get("n_reused", 0)),
                int(segment_manifest.get("summary", {}).get("n_computed", 0)),
                int(segment_manifest.get("summary", {}).get("n_skipped", 0)),
                float(ts.get("load_preprocess", 0.0)),
                float(ts.get("spike_prep", 0.0)),
                float(ts.get("analyzer", 0.0)),
                float(ts.get("postprocess", 0.0)),
                float(ts.get("total", 0.0)),
            )
        except Exception:
            logger.debug("Failed to persist or log segment execution manifest", exc_info=True)

        # Once segments are processed, compute derived channel groups if requested.
        if channel_groups is not None:
            try:
                n_segments = len(list(epochs.concat_epochs))

                # Prefer electrode-id space (only if we have it for at least one segment).
                any_electrodes = len(seg_electrode_sets) > 0

                # Version bump: key names aligned to docs/methods_waveforms_channel_sets.md.
                channel_groups.setdefault("version", 2)
                channel_groups.setdefault("n_segments", int(n_segments))
                channel_groups.setdefault("common_channel_ids", sorted(int(x) for x in common_channel_ids))

                def _json_id(x: Any) -> int | str:
                    try:
                        # numpy scalars / strings that represent ints
                        return int(x)
                    except Exception:
                        try:
                            return str(x)
                        except Exception:
                            return "<unserializable>"

                segments_payload: dict[str, Any] = {}
                for src in sorted(set(list(seg_channel_id_sets.keys()) + list(seg_electrode_sets.keys()))):
                    seg_payload: dict[str, Any] = {
                        "source_name": str(src),
                        "segment_channel_ids": sorted((_json_id(x) for x in seg_channel_id_sets.get(src, set())), key=str),
                        "segment_channel_id_space": "electrode_id" if src in seg_electrode_sets else "unknown",
                    }

                    if src in seg_electrode_sets:
                        seg_e = set(int(x) for x in seg_electrode_sets[src])
                        seg_payload["segment_electrode_ids"] = sorted(seg_e)
                        seg_payload["non_common_segment_electrode_ids"] = sorted(seg_e - set(common_channel_ids))

                    if src in seg_additional_electrode_sets:
                        # This is the electrode set after optional channel selection.
                        seg_payload["waveforms_analyzer_electrode_ids"] = sorted(
                            set(int(x) for x in seg_additional_electrode_sets[src])
                        )

                    segments_payload[str(src)] = seg_payload

                channel_groups["segments"] = segments_payload

                if any_electrodes:
                    # Build channel occurrence counts across segments.
                    counts: dict[int, int] = {}
                    for seg_set in seg_electrode_sets.values():
                        for c in seg_set:
                            counts[int(c)] = int(counts.get(int(c), 0) + 1)

                    all_union = set(counts.keys())
                    # Dataset-level: union of electrode ids present in at least one segment.
                    channel_groups["all_recorded_electrode_ids"] = sorted(all_union)

                    # Compute intersection from segments and compare to the concat/common set.
                    seg_sets = list(seg_electrode_sets.values())
                    seg_intersection = set(seg_sets[0]) if seg_sets else set()
                    for s in seg_sets[1:]:
                        seg_intersection &= set(s)

                    channel_groups["common_channels_from_segment_intersection_electrode_ids"] = sorted(seg_intersection)
                    if set(common_channel_ids) and seg_intersection and set(common_channel_ids) != seg_intersection:
                        logger.warning(
                            "Common channel mismatch: concat/common has %d channels but segment intersection has %d. "
                            "This can indicate inconsistent channel-id naming or a concatenation/channel-selection mismatch.",
                            int(len(set(common_channel_ids))),
                            int(len(seg_intersection)),
                        )

                    # Unique / non-unique (excluding common).
                    unique_by_src: dict[str, list[int]] = {}
                    for src, seg_set in seg_electrode_sets.items():
                        uniq = [int(c) for c in seg_set if counts.get(int(c), 0) == 1 and int(c) not in common_channel_ids]
                        unique_by_src[str(src)] = sorted(uniq)
                    channel_groups["unique_segment_electrode_ids_by_source"] = unique_by_src

                    non_unique_non_common = [
                        int(c)
                        for c, k in counts.items()
                        if int(k) > 1 and int(k) < int(n_segments) and int(c) not in common_channel_ids
                    ]
                    channel_groups["non_unique_non_common_electrode_ids"] = sorted(non_unique_non_common)

                    channel_groups.setdefault("counts", {})
                    channel_groups["counts"].update(
                        {
                            "all_channels_union": int(len(all_union)),
                            "common_channels_concat": int(len(set(common_channel_ids))),
                            "common_channels_segment_intersection": int(len(seg_intersection)),
                            "non_unique_non_common": int(len(non_unique_non_common)),
                        }
                    )
            except Exception:
                logger.debug("Failed to compute channel group summaries", exc_info=True)

    return best_by_unit


__all__ = [
    "_extract_concat_waveforms",
    "_extract_per_segment_waveforms",
]
