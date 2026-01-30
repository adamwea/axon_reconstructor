from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class WaveformContribution:
    """Pointer to waveforms for a (source, unit, channel) tuple.

    `channel_ref` is a best-effort reference to a channel for this source.
    It can be:
      - a channel id from `recording.get_channel_ids()` (preferred)
      - a global channel index in the recording channel list (common when ids
        are 0..N-1)

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


def _resolve_global_channel_index(*, analyzer: Any, channel_ref: Any) -> Optional[int]:
    """Resolve a channel reference to a global channel index (recording order)."""

    ch_ids = _try_get_recording_channel_ids(analyzer)
    if not ch_ids:
        return None

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

    # Fallback: treat as an already-global index.
    try:
        idx = int(channel_ref)
        if 0 <= idx < len(ch_ids):
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

    if not contributions:
        return None

    stacked: list[np.ndarray] = []
    used = 0

    for c in contributions:
        wfs = _try_get_waveforms_one_unit(analyzer=c.analyzer, unit_id=c.unit_id)
        if wfs is None:
            if logger is not None:
                logger.debug("Overlap: no waveforms for %s unit=%s", c.source_name, c.unit_id)
            continue

        wfs = np.asarray(wfs)
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

        # Optional downsampling of spikes to cap memory.
        if max_spikes_per_contribution is not None and n_spikes > int(max_spikes_per_contribution):
            k = int(max_spikes_per_contribution)
            if rng_choice is None:
                idx = np.random.choice(n_spikes, size=k, replace=False)
            else:
                idx = rng_choice(n_spikes, k)
            sel = sel[idx, :]

        if sel.ndim == 2 and sel.shape[0] > 0 and sel.shape[1] > 0:
            stacked.append(np.asarray(sel, dtype=float))
            used += int(sel.shape[0])

    if not stacked:
        return None

    all_wfs = np.concatenate(stacked, axis=0)
    if all_wfs.ndim != 2 or all_wfs.shape[0] == 0:
        return None

    mean_wf = np.mean(all_wfs, axis=0)

    if logger is not None:
        logger.info(
            "Overlap: mean-waveform merge used %d spikes across %d contributions",
            int(used),
            int(len(contributions)),
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
