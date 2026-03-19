from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


def robust_baseline_pre_negative_peak(
    wf: Any,
    *,
    win_frac: float = 0.10,
    guard_frac: float = 0.05,
    min_win: int = 5,
    min_guard: int = 2,
) -> float:
    """Estimate baseline using a window *before* the waveform's negative peak.

    Why this exists:
        Using the first N samples as "baseline" is fragile when waveforms are not
        temporally centered (AP can start very early), which can bias the baseline
        estimate and leave DC offsets in merged templates.

    Fallback behavior:
        If the negative peak is too close to the beginning to permit a pre-peak
        window, fall back to a trailing window at the end of the waveform.
    """

    import numpy as np  # type: ignore[import-not-found]

    x = np.asarray(wf, dtype=float)
    if x.ndim != 1 or x.size == 0:
        return 0.0

    n = int(x.size)
    win = max(int(min_win), int(round(float(win_frac) * n)))
    guard = max(int(min_guard), int(round(float(guard_frac) * n)))

    try:
        imin = int(np.nanargmin(x))
    except Exception:
        return 0.0

    end = max(0, imin - guard)
    start = max(0, end - win)
    if (end - start) < 3:
        # min is too early; use a trailing window instead.
        start = max(0, n - win)
        end = n

    try:
        b = float(np.nanmedian(x[start:end]))
    except Exception:
        b = 0.0
    return b


@dataclass(frozen=True)
class WaveformContribution:
    """Pointer to waveforms for a (source, unit, channel) tuple.

    `channel_ref` is a best-effort reference to a channel for this source.
    It can be:
      - a channel id from `recording.get_channel_ids()` (preferred)
      - an electrode/contact id from per-channel properties (common for Maxwell)
      - a global channel index in the recording channel list

    This module maps `channel_ref` into the waveforms extension's local channel
    axis index (sparsity-aware) before selecting waveforms.
    """

    source_name: str
    analyzer: Any
    unit_id: Any
    channel_ref: Any


def _try_get_recording_channel_ids(analyzer: Any) -> Optional[list[Any]]:
    try:
        rec = getattr(analyzer, "recording", None)
        if rec is None:
            return None
        if hasattr(rec, "get_channel_ids"):
            return list(rec.get_channel_ids())
    except Exception:
        return None
    return None


def _try_get_recording_electrode_ids(analyzer: Any) -> Optional[list[Any]]:
    """Best-effort electrode/contact id per recording channel."""

    rec = getattr(analyzer, "recording", None)
    if rec is None:
        return None

    for key in (
        "electrode_id",
        "electrode",
        "contact_id",
        "contact_ids",
        "contact",
        "site_id",
        "site",
    ):
        try:
            if hasattr(rec, "get_property_keys") and key not in set(rec.get_property_keys()):
                continue
            if hasattr(rec, "get_property"):
                vals = rec.get_property(key)
                if vals is None:
                    continue
                return list(vals)
        except Exception:
            continue

    # Maxwell-specific: contact_vector dict with an 'electrode' field.
    try:
        if hasattr(rec, "get_property"):
            cv = rec.get_property("contact_vector")
            if isinstance(cv, dict) and "electrode" in cv:
                return list(cv["electrode"])
    except Exception:
        pass

    return None


def _resolve_global_channel_index(*, analyzer: Any, channel_ref: Any) -> Optional[int]:
    """Resolve a channel reference to a global channel index (recording order)."""

    ch_ids = _try_get_recording_channel_ids(analyzer)
    if ch_ids:
        # Exact match against channel_ids (handles int/str ids).
        try:
            return int(ch_ids.index(channel_ref))
        except Exception:
            pass

        # Common: ids are numeric but may be stringified (or vice versa).
        try:
            ref_str = str(channel_ref)
            for i, cid in enumerate(ch_ids):
                if str(cid) == ref_str:
                    return int(i)
        except Exception:
            pass

    # Some pipelines key overlaps by electrode/contact id instead of channel_id.
    el_ids = _try_get_recording_electrode_ids(analyzer)
    if el_ids:
        try:
            return int(el_ids.index(channel_ref))
        except Exception:
            pass
        try:
            ref_str = str(channel_ref)
            for i, eid in enumerate(el_ids):
                if str(eid) == ref_str:
                    return int(i)
        except Exception:
            pass

    # Fallback: treat as an already-global index.
    try:
        idx = int(channel_ref)
        if ch_ids and 0 <= idx < len(ch_ids):
            return idx
    except Exception:
        pass

    return None


def _try_get_unit_sparse_global_channel_indices(*, analyzer: Any, unit_id: Any) -> Optional[list[int]]:
    """Return the per-unit sparse global channel indices if available."""

    try:
        wf_ext = analyzer.get_extension("waveforms")
    except Exception:
        wf_ext = None

    sparsity = None
    for obj in (analyzer, wf_ext):
        if obj is None:
            continue
        try:
            sparsity = getattr(obj, "sparsity", None)
        except Exception:
            sparsity = None
        if sparsity is not None:
            break

    if sparsity is None:
        return None

    # Support a few SpikeInterface versions.
    try:
        mapping = getattr(sparsity, "unit_id_to_channel_indices", None)
        if callable(mapping):
            out = mapping(unit_id)
            return [int(x) for x in list(out)]
        if isinstance(mapping, dict):
            if unit_id in mapping:
                return [int(x) for x in list(mapping[unit_id])]
            # Best-effort by string match.
            uid_str = str(unit_id)
            for k, v in mapping.items():
                if str(k) == uid_str:
                    return [int(x) for x in list(v)]
    except Exception:
        pass

    for attr in ("get_channel_indices", "get_channel_indices_for_unit"):
        if hasattr(sparsity, attr):
            fn = getattr(sparsity, attr)
            if callable(fn):
                try:
                    out = fn(unit_id)
                    return [int(x) for x in list(out)]
                except Exception:
                    pass

    return None


def _resolve_waveforms_channel_axis_index(*, analyzer: Any, unit_id: Any, channel_ref: Any) -> Optional[int]:
    """Map channel_ref to the local waveforms channel axis index (sparse-aware)."""

    global_idx = _resolve_global_channel_index(analyzer=analyzer, channel_ref=channel_ref)
    if global_idx is None:
        return None

    sparse_global = _try_get_unit_sparse_global_channel_indices(analyzer=analyzer, unit_id=unit_id)
    if sparse_global is None:
        # No sparsity: waveforms axis is the full recording channel list.
        return int(global_idx)

    try:
        return int(list(sparse_global).index(int(global_idx)))
    except Exception:
        return None


def _try_get_waveforms_one_unit(*, analyzer: Any, unit_id: Any) -> Optional[Any]:
    try:
        if hasattr(analyzer, "has_extension") and analyzer.has_extension("waveforms"):
            wf_ext = analyzer.get_extension("waveforms")
        else:
            wf_ext = analyzer.get_extension("waveforms")
    except Exception:
        return None

    for attr in ("get_waveforms_one_unit", "get_waveforms"):
        if hasattr(wf_ext, attr):
            try:
                fn = getattr(wf_ext, attr)
                if attr == "get_waveforms_one_unit":
                    return fn(unit_id)
                # Older APIs: get_waveforms(unit_id)
                return fn(unit_id)
            except Exception:
                return None

    return None


def mean_waveform_from_contributions(
    *,
    contributions: list[WaveformContribution],
    max_spikes_per_contribution: Optional[int] = 500,
    rng_choice: Optional[Callable[[int, int], Any]] = None,
    logger: Any = None,
):
    """Stack waveforms across contributions and compute mean (default strategy).

    Returns:
        mean_waveform: np.ndarray with shape (n_samples,), or None if unavailable.

    Notes:
        - We intentionally average *raw waveforms* (not templates) to mimic the
          conceptual template definition.
        - `max_spikes_per_contribution` caps per-source memory use.
    """

    import numpy as np  # type: ignore[import-not-found]

    debug_enabled = bool(getattr(logger, "isEnabledFor", lambda *_: False)(10))

    if not contributions:
        if debug_enabled:
            logger.debug("Overlap: no contributions provided")
        return None

    if debug_enabled:
        logger.debug(
            "Overlap: begin mean-waveform merge with n_contributions=%d max_spikes_per_contribution=%s",
            int(len(contributions)),
            str(max_spikes_per_contribution),
        )

    stacked: list[np.ndarray] = []
    used = 0

    # Cache waveforms per (analyzer, unit_id) to avoid repeated heavy loads.
    wf_cache: dict[tuple[int, Any], Optional[np.ndarray]] = {}

    for c in contributions:
        cache_key = (id(c.analyzer), c.unit_id)
        if cache_key in wf_cache:
            wfs = wf_cache[cache_key]
        else:
            wfs_raw = _try_get_waveforms_one_unit(analyzer=c.analyzer, unit_id=c.unit_id)
            try:
                wfs = None if wfs_raw is None else np.asarray(wfs_raw)
            except Exception:
                wfs = None
            wf_cache[cache_key] = wfs

        if debug_enabled:
            logger.debug(
                "Overlap: contribution source=%s unit=%s channel_ref=%s waveforms_shape=%s",
                c.source_name,
                c.unit_id,
                c.channel_ref,
                (None if wfs is None else tuple(wfs.shape)),
            )

        if wfs is None:
            if logger is not None:
                logger.debug("Overlap: no waveforms for %s unit=%s", c.source_name, c.unit_id)
            continue

        if wfs.ndim != 3:
            if logger is not None:
                logger.debug(
                    "Overlap: unexpected waveforms shape for %s unit=%s: %s",
                    c.source_name,
                    c.unit_id,
                    getattr(wfs, "shape", None),
                )
            continue

        n_spikes, n_samples, n_ch = wfs.shape
        if n_spikes <= 0 or n_samples <= 0 or n_ch <= 0:
            if debug_enabled:
                logger.debug(
                    "Overlap: invalid waveform dimensions source=%s unit=%s shape=%s",
                    c.source_name,
                    c.unit_id,
                    tuple(wfs.shape),
                )
            continue

        ch_i = _resolve_waveforms_channel_axis_index(
            analyzer=c.analyzer,
            unit_id=c.unit_id,
            channel_ref=c.channel_ref,
        )
        if ch_i is None or int(ch_i) < 0 or int(ch_i) >= n_ch:
            if logger is not None:
                logger.debug(
                    "Overlap: channel_ref unresolved/out-of-bounds for %s unit=%s: ref=%s -> idx=%s not in [0, %s)",
                    c.source_name,
                    c.unit_id,
                    c.channel_ref,
                    ch_i,
                    n_ch,
                )
            continue

        sel = wfs[:, :, int(ch_i)]

        if debug_enabled:
            logger.debug(
                "Overlap: selected waveforms source=%s unit=%s channel_axis_index=%d n_spikes=%d n_samples=%d",
                c.source_name,
                c.unit_id,
                int(ch_i),
                int(sel.shape[0]),
                int(sel.shape[1]),
            )

        # Robust per-spike baseline subtraction to avoid DC offsets propagating into
        # overlap-resolved templates. We intentionally do this here (templates stage)
        # because source templates may have slightly different baselines across segments.
        try:
            baseline = np.zeros((int(sel.shape[0]), 1), dtype=float)
            for i in range(int(sel.shape[0])):
                baseline[i, 0] = robust_baseline_pre_negative_peak(sel[i, :])
            sel = sel - baseline
        except Exception:
            pass

        # Optional downsampling of spikes to cap memory.
        if max_spikes_per_contribution is not None and n_spikes > int(max_spikes_per_contribution):
            k = int(max_spikes_per_contribution)
            if rng_choice is None:
                idx = np.random.choice(n_spikes, size=k, replace=False)
            else:
                idx = rng_choice(n_spikes, k)
            sel = sel[idx, :]
            if debug_enabled:
                logger.debug(
                    "Overlap: downsampled contribution source=%s unit=%s from %d to %d spikes",
                    c.source_name,
                    c.unit_id,
                    int(n_spikes),
                    int(k),
                )

        if sel.ndim == 2 and sel.shape[0] > 0 and sel.shape[1] > 0:
            stacked.append(np.asarray(sel, dtype=float))
            used += int(sel.shape[0])
            if debug_enabled:
                logger.debug(
                    "Overlap: accepted contribution source=%s unit=%s stacked_spikes_now=%d",
                    c.source_name,
                    c.unit_id,
                    int(used),
                )

    if not stacked:
        if debug_enabled:
            logger.debug("Overlap: merge aborted because no valid contributions remained after filtering")
        return None

    all_wfs = np.concatenate(stacked, axis=0)
    if all_wfs.ndim != 2 or all_wfs.shape[0] == 0:
        if debug_enabled:
            logger.debug(
                "Overlap: merge aborted after concatenate due to invalid shape=%s",
                tuple(all_wfs.shape) if hasattr(all_wfs, "shape") else None,
            )
        return None

    mean_wf = np.mean(all_wfs, axis=0)

    # Re-center the mean waveform (in case some contributions had different baseline windows).
    try:
        mean_wf = mean_wf - robust_baseline_pre_negative_peak(mean_wf)
    except Exception:
        pass

    if logger is not None and debug_enabled:
        logger.debug(
            "Overlap: mean-waveform merge used %d spikes across %d contributions",
            int(used),
            int(len(contributions)),
        )
        logger.debug(
            "Overlap: merge success output_samples=%d",
            int(mean_wf.shape[0]),
        )

    return np.asarray(mean_wf, dtype=float)


def ptp_amplitude_from_mean_waveform(mean_wf: Any) -> Optional[float]:
    """Compute peak-to-peak amplitude (compatible with footprinting amp)."""

    import numpy as np  # type: ignore[import-not-found]

    if mean_wf is None:
        return None

    x = np.asarray(mean_wf, dtype=float)
    if x.ndim != 1 or x.size == 0:
        return None

    return float(np.ptp(x))


OVERLAP_STRATEGIES: dict[str, Callable[..., Any]] = {
    "mean_waveforms": mean_waveform_from_contributions,
}
