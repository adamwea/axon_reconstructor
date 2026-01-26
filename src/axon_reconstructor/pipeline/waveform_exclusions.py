"""Waveform exclusion artifact helpers.

This module provides a *spike-level* exclusion mechanism that can be produced in
`waveforms` and consumed later by `footprinting`/`templates`.

Key idea:
- Represent exclusions by (source_name, unit_id, spike_sample) rather than by
  waveform-value matching (more stable, cheaper, and joinable).
- Use `random_spikes` + `waveforms` extensions to compute per-unit templates
  while dropping excluded spikes.

Artifacts:
- `wf_exclusions.npz` (written under `<well>/waveforms_outputs/`)

The `.npz` stores one row per excluded spike with metadata fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


WF_EXCLUSIONS_NPZ_NAME = "wf_exclusions.npz"


def _py_scalar(v: Any) -> Any:
    try:
        return v.item() if hasattr(v, "item") else v
    except Exception:
        return v


def normalize_unit_id(unit_id: Any) -> Any:
    """Normalize unit IDs for dict/set lookups across numpy/python scalar types."""
    unit_id = _py_scalar(unit_id)
    # Preserve ints when possible.
    try:
        if isinstance(unit_id, bool):
            return unit_id
        if isinstance(unit_id, (int,)):
            return int(unit_id)
    except Exception:
        pass
    # Preserve numeric strings as ints.
    try:
        s = str(unit_id)
        if s.isdigit():
            return int(s)
    except Exception:
        pass
    return unit_id


def _as_int(v: Any, *, none_value: int = -1) -> int:
    if v is None:
        return int(none_value)
    try:
        return int(_py_scalar(v))
    except Exception:
        return int(none_value)


def write_wf_exclusions_npz(
    *,
    wf_exclusions_npz: Path,
    rows: list[dict[str, Any]],
    force_restart: bool,
    logger,
) -> None:
    """Write exclusions as a compact `.npz` for fast downstream loading."""

    if wf_exclusions_npz.exists() and (not force_restart):
        logger.info("wf_exclusions.npz exists; not overwriting: %s", wf_exclusions_npz)
        return

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required to write wf_exclusions.npz") from e

    wf_exclusions_npz.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        np.savez_compressed(
            wf_exclusions_npz,
            scope=np.asarray([], dtype=object),
            source_name=np.asarray([], dtype=object),
            unit_id=np.asarray([], dtype=object),
            spike_sample_local=np.asarray([], dtype=np.int64),
            spike_sample_concat=np.asarray([], dtype=np.int64),
            reason=np.asarray([], dtype=object),
        )
        logger.info("Wrote wf_exclusions.npz -> %s (rows=0)", wf_exclusions_npz)
        return

    scope = []
    source_name = []
    unit_id = []
    spike_sample_local = []
    spike_sample_concat = []
    reason = []

    # Optional metadata (kept as best-effort, object-typed)
    segment_index = []
    rec_name = []

    for r in rows:
        scope.append(str(r.get("scope")))
        source_name.append(str(r.get("source_name")))
        unit_id.append(_py_scalar(r.get("unit_id")))
        spike_sample_local.append(_as_int(r.get("spike_sample_local")))
        spike_sample_concat.append(_as_int(r.get("spike_sample_concat")))
        reason.append(str(r.get("reason")))
        segment_index.append(_py_scalar(r.get("segment_index")))
        rec_name.append(_py_scalar(r.get("rec_name")))

    np.savez_compressed(
        wf_exclusions_npz,
        scope=np.asarray(scope, dtype=object),
        source_name=np.asarray(source_name, dtype=object),
        unit_id=np.asarray(unit_id, dtype=object),
        spike_sample_local=np.asarray(spike_sample_local, dtype=np.int64),
        spike_sample_concat=np.asarray(spike_sample_concat, dtype=np.int64),
        reason=np.asarray(reason, dtype=object),
        segment_index=np.asarray(segment_index, dtype=object),
        rec_name=np.asarray(rec_name, dtype=object),
    )

    logger.info("Wrote wf_exclusions.npz -> %s (rows=%d)", wf_exclusions_npz, int(len(rows)))


def load_wf_exclusions_by_source(
    *,
    well_out_dir: Path,
    logger,
) -> dict[str, dict[Any, set[int]]]:
    """Load exclusions into a lookup map.

    Returns:
        source_name -> normalized_unit_id -> set(spike_sample)

    Notes:
    - For `concat` source, spike samples are interpreted in concat coordinates.
    - For segment sources, spike samples are interpreted in segment-local coordinates.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required to load wf_exclusions") from e

    wf_exclusions_npz = well_out_dir / "waveforms_outputs" / WF_EXCLUSIONS_NPZ_NAME
    if not wf_exclusions_npz.exists():
        return {}

    try:
        data = np.load(wf_exclusions_npz, allow_pickle=True)
        source_names = list(data["source_name"])
        unit_ids = list(data["unit_id"])
        scope = list(data["scope"]) if "scope" in data else [None] * len(unit_ids)
        ssl = data["spike_sample_local"]
        ssc = data["spike_sample_concat"]
    except Exception as e:
        logger.warning("Failed to read wf_exclusions.npz (%s): %s", wf_exclusions_npz, e)
        return {}

    out: dict[str, dict[Any, set[int]]] = {}
    for i in range(len(unit_ids)):
        try:
            src = str(source_names[i])
            uid_norm = normalize_unit_id(unit_ids[i])
            sc = str(scope[i]) if scope[i] is not None else ""

            # Decide which coordinate system to use.
            if src == "concat" or sc == "concat":
                sample = int(ssc[i])
            else:
                sample = int(ssl[i])

            if sample < 0:
                continue

            out.setdefault(src, {}).setdefault(uid_norm, set()).add(sample)
        except Exception:
            continue

    if out:
        logger.info("Loaded wf exclusions from %s (sources=%d)", wf_exclusions_npz, len(out))
    return out


def _get_random_spike_samples(*, analyzer, unit_id: Any) -> Optional["Any"]:
    """Return per-waveform spike samples corresponding to waveforms array order."""

    try:
        rs_ext = analyzer.get_extension("random_spikes")
    except Exception:
        return None

    # Newer SI (e.g. 0.103.x): derive sample indices from selected indices in spike train.
    if hasattr(rs_ext, "get_selected_indices_in_spike_train") and hasattr(analyzer, "sorting"):
        try:
            nseg = int(getattr(analyzer.sorting, "get_num_segments", lambda: 1)())
        except Exception:
            nseg = 1

        # The analyzers produced in this pipeline (concat + per-segment) are typically single-segment.
        seg_indices = [0] if nseg <= 1 else [0]

        for seg_index in seg_indices:
            try:
                try:
                    sel = rs_ext.get_selected_indices_in_spike_train(unit_id, seg_index)
                except TypeError:
                    sel = rs_ext.get_selected_indices_in_spike_train(unit_id=unit_id, segment_index=seg_index)

                st = analyzer.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=seg_index)
                return st[sel]
            except Exception:
                continue

    # Legacy/common APIs across SI versions.
    for attr in ("get_random_spikes", "get_unit_random_spikes"):
        if hasattr(rs_ext, attr):
            try:
                fn = getattr(rs_ext, attr)
                return fn(unit_id=unit_id)
            except TypeError:
                try:
                    return fn(unit_id)
                except Exception:
                    pass
            except Exception:
                pass

    # Fallback: a dict-like attribute.
    for attr in ("random_spikes", "random_spikes_indices"):
        if hasattr(rs_ext, attr):
            try:
                d = getattr(rs_ext, attr)
                if isinstance(d, dict) and unit_id in d:
                    return d[unit_id]
                # Try normalized unit id match.
                if isinstance(d, dict):
                    uid_norm = normalize_unit_id(unit_id)
                    for k, v in d.items():
                        if normalize_unit_id(k) == uid_norm:
                            return v
            except Exception:
                pass

    return None


def _get_unit_waveforms(*, analyzer, unit_id: Any) -> Optional["Any"]:
    try:
        wf_ext = analyzer.get_extension("waveforms")
    except Exception:
        return None

    # SpikeInterface API varies across versions:
    # - older: get_waveforms / get_unit_waveforms
    # - newer (e.g. SI 0.103.x): ComputeWaveforms.get_waveforms_one_unit
    for attr in ("get_waveforms", "get_unit_waveforms", "get_waveforms_one_unit"):
        if hasattr(wf_ext, attr):
            try:
                fn = getattr(wf_ext, attr)
                return fn(unit_id=unit_id)
            except TypeError:
                try:
                    return fn(unit_id)
                except Exception:
                    pass
            except Exception:
                pass

    return None


@dataclass(frozen=True)
class TemplateFromWaveformsResult:
    template: "Any"
    n_waveforms_total: int
    n_waveforms_kept: int


def compute_unit_template_from_waveforms(
    *,
    analyzer,
    unit_id: Any,
    excluded_spike_samples: Optional[set[int]],
    logger,
) -> Optional[TemplateFromWaveformsResult]:
    """Compute a unit template by averaging waveforms while excluding spikes.

    Exclusions are applied by comparing per-waveform spike sample indices from
    `random_spikes` against `excluded_spike_samples`.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy required") from e

    # Some analyzers may not contain the unit.
    try:
        if hasattr(analyzer, "sorting") and hasattr(analyzer.sorting, "id_to_index"):
            analyzer.sorting.id_to_index(unit_id)
    except Exception:
        return None

    waveforms = _get_unit_waveforms(analyzer=analyzer, unit_id=unit_id)
    if waveforms is None:
        return None

    waveforms = np.asarray(waveforms)
    if waveforms.ndim != 3 or waveforms.size == 0:
        return None

    n_total = int(waveforms.shape[0])

    if not excluded_spike_samples:
        tmpl = np.nanmean(waveforms, axis=0)
        return TemplateFromWaveformsResult(template=tmpl, n_waveforms_total=n_total, n_waveforms_kept=n_total)

    rs = _get_random_spike_samples(analyzer=analyzer, unit_id=unit_id)
    if rs is None:
        # We can still compute a template, but cannot reliably align exclusions.
        logger.warning(
            "No random_spikes alignment for unit %s; computing template without exclusions",
            unit_id,
        )
        tmpl = np.nanmean(waveforms, axis=0)
        return TemplateFromWaveformsResult(template=tmpl, n_waveforms_total=n_total, n_waveforms_kept=n_total)

    rs = np.asarray(rs).astype(np.int64, copy=False)
    if rs.shape[0] != waveforms.shape[0]:
        logger.warning(
            "random_spikes mismatch for unit %s: random_spikes=%d waveforms=%d; skipping exclusions",
            unit_id,
            int(rs.shape[0]),
            int(waveforms.shape[0]),
        )
        tmpl = np.nanmean(waveforms, axis=0)
        return TemplateFromWaveformsResult(template=tmpl, n_waveforms_total=n_total, n_waveforms_kept=n_total)

    keep_mask = np.asarray([int(s) not in excluded_spike_samples for s in rs], dtype=bool)
    n_kept = int(keep_mask.sum())

    if n_kept <= 0:
        logger.warning("All waveforms excluded for unit %s; returning None", unit_id)
        return None

    tmpl = np.nanmean(waveforms[keep_mask, :, :], axis=0)
    return TemplateFromWaveformsResult(template=tmpl, n_waveforms_total=n_total, n_waveforms_kept=n_kept)
