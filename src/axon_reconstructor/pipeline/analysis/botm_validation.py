from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class BotmValidationInputs:
    """Inputs for BOTM validation scoring.

    This is intentionally analysis-stage friendly:
    - best-effort (missing deps/inputs should be handled gracefully by callers)
    - uses existing on-disk artifacts (templates + waveforms analyzers)

    Notes on scope (MVP):
    - Uses merged_contributing template outputs.
    - Uses concat waveforms analyzer.
    - Uses a diagonal noise model with per-channel sigma.
    - Generates a synthetic Gaussian noise score distribution (no raw recording reads).
    """

    well_out_dir: Path

    templates_out_dir: Optional[Path] = None
    waveforms_out_dir: Optional[Path] = None

    unit_ids: Optional[list[Any]] = None

    n_spike: int = 200
    n_noise: int = 200

    # Noise model controls (MVP supports "diag").
    noise_model: str = "diag"

    # Which waveforms sources to score against the merged template.
    # - "concat": concat_waveforms only
    # - "concat+segments": concat + all readable segment analyzers
    waveforms_source: str = "concat"

    # Negative sampling mode.
    # - "gaussian": synthetic Gaussian noise using estimated per-channel sigma (default)
    # - "recording_random": random windows sampled from the concat recording (best-effort)
    negatives_mode: str = "gaussian"

    # Baseline region used to estimate noise sigma from spike snippets.
    # This does NOT need to match the full template length.
    baseline_frac: float = 0.25

    # RNG seed for subsampling spikes and generating synthetic noise.
    seed: int = 0

    # Output control
    out_dir: Optional[Path] = None
    force_restart: bool = False


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _try_load_concat_analyzer(*, waveforms_out_dir: Path, logger: logging.Logger):
    try:
        import spikeinterface.full as si  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(f"SpikeInterface unavailable: {e}") from e

    concat_waveforms_dir = Path(waveforms_out_dir) / "concat_waveforms"
    if not concat_waveforms_dir.exists():
        raise FileNotFoundError(f"Missing concat waveforms analyzer at {concat_waveforms_dir}")

    logger.info("Loading concat waveforms analyzer: %s", concat_waveforms_dir)
    return si.load_sorting_analyzer(concat_waveforms_dir)


def _try_load_waveforms_analyzers(
    *,
    well_out_dir: Path,
    waveforms_out_dir: Path,
    waveforms_source: str,
    logger: logging.Logger,
) -> list[tuple[str, Any]]:
    """Load waveforms analyzers (concat and optionally segments).

    Uses the canonical waveforms stage folder structure under the well output dir.
    """

    src = str(waveforms_source).lower().strip()
    include_concat = True
    include_segments = src in {"concat+segments", "all", "segments+concat"}

    # Reuse existing helper from templates stage so conventions stay aligned.
    try:
        from ..templates.multi_source_utils import _load_waveforms_analyzers  # type: ignore

        return _load_waveforms_analyzers(
            well_out_dir=Path(well_out_dir),
            include_concat=include_concat,
            include_segments=include_segments,
            logger=logger,
        )
    except Exception:
        # Fallback: concat only.
        an = _try_load_concat_analyzer(waveforms_out_dir=waveforms_out_dir, logger=logger)
        return [("concat", an)]


def _try_get_waveforms_one_unit(*, analyzer: Any, unit_id: Any):
    """Compatibility helper for SpikeInterface waveforms extension."""

    try:
        wf_ext = analyzer.get_extension("waveforms")
    except Exception:
        wf_ext = None

    if wf_ext is None:
        return None

    for attr in ("get_waveforms_one_unit", "get_waveforms"):
        if hasattr(wf_ext, attr):
            fn = getattr(wf_ext, attr)
            if callable(fn):
                try:
                    if attr == "get_waveforms_one_unit":
                        return fn(unit_id=unit_id)
                    return fn(unit_id)
                except Exception:
                    return None

    return None


def _jsonable(x: Any) -> Any:
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass

    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x

    try:
        return str(x)
    except Exception:
        return repr(x)


def _resolve_unit_channel_ids(*, analyzer: Any, unit_id: Any) -> Optional[list[Any]]:
    """Return the waveforms channel-id axis for this unit (sparsity aware)."""

    # Newer analyzers: sparsity may exist and expose per-unit channel ids.
    try:
        sp = getattr(analyzer, "sparsity", None)
        if sp is not None:
            mapping = getattr(sp, "unit_id_to_channel_ids", None)
            if isinstance(mapping, dict) and unit_id in mapping:
                return list(mapping[unit_id])
    except Exception:
        pass

    # Fallback: no sparsity => waveforms channel axis matches recording channel ids.
    for attr in ("get_channel_ids",):
        if hasattr(analyzer, attr):
            try:
                return list(getattr(analyzer, attr)())
            except Exception:
                pass

    try:
        rec = getattr(analyzer, "recording", None)
        if rec is not None and hasattr(rec, "get_channel_ids"):
            return list(rec.get_channel_ids())
    except Exception:
        pass

    return None


def _subset_waveforms_to_template_channels(
    *,
    waveforms: Any,
    waveforms_channel_ids: Optional[list[Any]],
    template_channel_ids: Optional[list[Any]],
):
    """Return waveforms subsetted to match template channel axis (best effort).

    waveforms: (n_spikes, n_samples, n_unit_channels)
    template expects channels == template_channel_ids length.
    """

    import numpy as np  # type: ignore[import-not-found]

    wfs = np.asarray(waveforms)
    if wfs.ndim != 3:
        raise ValueError(f"Unexpected waveforms shape: {wfs.shape}")

    if template_channel_ids is None:
        # Nothing to map; only ok if counts match.
        return wfs

    if waveforms_channel_ids is None:
        # Nothing to map; only ok if counts match.
        return wfs

    # Compare as strings to handle int/np scalar drift.
    wf_ids = [str(x) for x in waveforms_channel_ids]
    tpl_ids = [str(x) for x in template_channel_ids]

    inds: list[int] = []
    for tid in tpl_ids:
        try:
            inds.append(int(wf_ids.index(str(tid))))
        except Exception as e:
            raise KeyError(f"template channel_id {tid!r} not found in waveforms channel ids") from e

    return wfs[:, :, np.asarray(inds, dtype=int)]


def _embed_waveforms_into_template_channel_order(
    *,
    waveforms: Any,
    waveforms_channel_ids: Optional[list[Any]],
    template_channel_ids: Optional[list[Any]],
) -> Any:
    """Embed per-source waveforms into the merged-template channel order.

    Returns an array shaped (n_spikes, n_samples, n_template_channels).
    Channels not present in this source are filled with zeros.

    This is the key helper that lets us score a merged template even when its
    channels come from multiple analyzers (concat + per-segment additional channels).
    """

    import numpy as np  # type: ignore[import-not-found]

    wfs = np.asarray(waveforms)
    if wfs.ndim != 3:
        raise ValueError(f"Unexpected waveforms shape: {wfs.shape}")

    if template_channel_ids is None:
        # No stable ids to map against; we can only pass through.
        return wfs

    if waveforms_channel_ids is None:
        raise ValueError("missing waveforms_channel_ids; cannot embed into template channel order")

    tpl_ids = [str(x) for x in template_channel_ids]
    wf_ids = [str(x) for x in waveforms_channel_ids]

    n_spikes, n_samples, _ = wfs.shape
    out = np.zeros((n_spikes, n_samples, int(len(tpl_ids))), dtype=wfs.dtype)

    for j, wf_id in enumerate(wf_ids):
        try:
            k = int(tpl_ids.index(wf_id))
        except Exception:
            continue
        out[:, :, k] = wfs[:, :, int(j)]

    return out


def _estimate_sigma_per_channel_from_baseline(*, waveforms: Any, baseline_frac: float) -> Any:
    """Estimate per-channel sigma from baseline portion of spike snippets.

    This is intentionally simple and robust:
    - baseline window length = max(1, floor(T * baseline_frac))
    - sigma[ch] = std over (spikes, time) samples in that baseline region
    """

    import numpy as np  # type: ignore[import-not-found]

    wfs = np.asarray(waveforms)
    if wfs.ndim != 3:
        raise ValueError(f"Unexpected waveforms shape for sigma estimation: {wfs.shape}")

    n_spikes, n_samples, n_ch = wfs.shape
    if n_spikes <= 0 or n_samples <= 0 or n_ch <= 0:
        raise ValueError("Empty waveforms")

    frac = float(baseline_frac)
    if not (0.0 < frac <= 1.0):
        frac = 0.25

    t0 = 0
    t1 = max(1, int(np.floor(n_samples * frac)))

    base = wfs[:, t0:t1, :]
    # std over spikes and time => per-channel
    sigma = np.nanstd(base.reshape(-1, n_ch), axis=0)

    # Avoid degenerate values.
    eps = 1e-9
    sigma = np.where(np.isfinite(sigma) & (sigma > eps), sigma, eps)
    return sigma


def _botm_filter_from_template(*, template_t_by_c: Any, sigma_ch: Any) -> Any:
    """Return diagonal-cov BOTM filter f with shape (T, C)."""

    import numpy as np  # type: ignore[import-not-found]

    tmpl = np.asarray(template_t_by_c)
    if tmpl.ndim != 2:
        raise ValueError(f"Unexpected template shape: {tmpl.shape}")

    sigma = np.asarray(sigma_ch)
    if sigma.ndim != 1 or sigma.shape[0] != tmpl.shape[1]:
        raise ValueError(f"Unexpected sigma shape: {sigma.shape}, expected ({tmpl.shape[1]},)")

    denom = (sigma**2)[None, :]
    eps = 1e-12
    denom = np.where(denom > eps, denom, eps)
    return tmpl / denom


def _scores_for_waveforms(*, waveforms_t_by_c: Any, f_t_by_c: Any) -> Any:
    """Compute s(w) = sum_{t,c} f[t,c] * w[t,c] for each waveform."""

    import numpy as np  # type: ignore[import-not-found]

    wfs = np.asarray(waveforms_t_by_c)
    f = np.asarray(f_t_by_c)

    if wfs.ndim != 3:
        raise ValueError(f"Unexpected waveforms shape: {wfs.shape}")
    if f.ndim != 2:
        raise ValueError(f"Unexpected filter shape: {f.shape}")

    if wfs.shape[1] != f.shape[0] or wfs.shape[2] != f.shape[1]:
        raise ValueError(f"Shape mismatch: waveforms={wfs.shape}, filter={f.shape}")

    return np.sum(wfs * f[None, :, :], axis=(1, 2))


def _linear_resample_time_t_by_c(*, x_t_by_c: Any, target_t: int) -> Any:
    """Linearly resample (T, C) -> (target_t, C) on a uniform grid."""

    import numpy as np  # type: ignore[import-not-found]

    x = np.asarray(x_t_by_c)
    if x.ndim != 2:
        raise ValueError(f"expected (T, C), got shape={x.shape}")

    old_t = int(x.shape[0])
    target_t = int(target_t)
    if target_t <= 0:
        raise ValueError(f"target_t must be > 0 (got {target_t})")
    if old_t == target_t:
        return x
    if old_t < 2:
        raise ValueError(f"cannot resample from old_t={old_t}")

    pos = np.linspace(0.0, float(old_t - 1), num=target_t, dtype=float)
    idx0 = np.floor(pos).astype(int)
    idx1 = np.minimum(idx0 + 1, old_t - 1)
    w = (pos - idx0).astype(float)
    w = w[:, None]  # (target_t, 1)

    y0 = x[idx0, :]
    y1 = x[idx1, :]
    return (1.0 - w) * y0 + w * y1


def _linear_resample_time_n_t_by_c(*, x_n_t_by_c: Any, target_t: int) -> Any:
    """Linearly resample (N, T, C) -> (N, target_t, C) on a uniform grid."""

    import numpy as np  # type: ignore[import-not-found]

    x = np.asarray(x_n_t_by_c)
    if x.ndim != 3:
        raise ValueError(f"expected (N, T, C), got shape={x.shape}")

    old_t = int(x.shape[1])
    target_t = int(target_t)
    if target_t <= 0:
        raise ValueError(f"target_t must be > 0 (got {target_t})")
    if old_t == target_t:
        return x
    if old_t < 2:
        raise ValueError(f"cannot resample from old_t={old_t}")

    pos = np.linspace(0.0, float(old_t - 1), num=target_t, dtype=float)
    idx0 = np.floor(pos).astype(int)
    idx1 = np.minimum(idx0 + 1, old_t - 1)
    w = (pos - idx0).astype(float)

    y0 = x[:, idx0, :]
    y1 = x[:, idx1, :]
    return (1.0 - w)[None, :, None] * y0 + w[None, :, None] * y1

def _resample_waveforms_time_to_target_t(*, waveforms_n_t_by_c: Any, target_t: int, method: str = "sinc") -> Any:
    """Resample waveforms to a target time length.

    We prefer to keep templates at their native timebase (which may be upsampled
    upstream) and instead upsample/downsample waveforms to match.

    If the ratio is an integer factor, we reuse the templates-stage resampling
    implementation (polyphase/sinc via resample_poly when available).
    Falls back to local linear interpolation for non-integer ratios.
    """

    import numpy as np  # type: ignore[import-not-found]

    wfs = np.asarray(waveforms_n_t_by_c)
    if wfs.ndim != 3:
        raise ValueError(f"expected (N, T, C), got shape={wfs.shape}")

    old_t = int(wfs.shape[1])
    target_t = int(target_t)
    if target_t <= 0:
        raise ValueError(f"target_t must be > 0 (got {target_t})")
    if old_t == target_t:
        return wfs
    if old_t < 2:
        raise ValueError(f"cannot resample from old_t={old_t}")

    up = None
    down = None
    if old_t > 0 and target_t % old_t == 0:
        up = int(target_t // old_t)
        down = 1
    elif target_t > 0 and old_t % target_t == 0:
        up = 1
        down = int(old_t // target_t)

    if up is not None and down is not None:
        try:
            from axon_reconstructor.pipeline.templates.utils import _resample_template_time  # type: ignore

            # Reuse the templates-stage resampler by flattening (N, C) into a single channel axis.
            n, _, c = wfs.shape
            x = np.transpose(wfs, (1, 0, 2)).reshape(old_t, int(n * c))  # (T, N*C)
            y = _resample_template_time(template=x, up=int(up), down=int(down), method=str(method))
            y = np.asarray(y)

            # Enforce exact target_t if needed.
            if int(y.shape[0]) > int(target_t):
                y = y[: int(target_t), :]
            elif int(y.shape[0]) < int(target_t):
                pad = np.zeros((int(target_t) - int(y.shape[0]), int(y.shape[1])), dtype=y.dtype)
                y = np.concatenate([y, pad], axis=0)

            y = y.reshape(int(target_t), int(n), int(c))
            return np.transpose(y, (1, 0, 2))  # (N, target_t, C)
        except Exception:
            pass

    return _linear_resample_time_n_t_by_c(x_n_t_by_c=wfs, target_t=int(target_t))


def _auc_from_scores(*, pos: Any, neg: Any) -> float:
    """Compute AUC = P(pos > neg) using rank statistics.

    Returns NaN if inputs are insufficient.
    """

    import numpy as np  # type: ignore[import-not-found]

    pos = np.asarray(pos).astype(float, copy=False)
    neg = np.asarray(neg).astype(float, copy=False)

    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    n1 = int(pos.size)
    n0 = int(neg.size)
    if n1 == 0 or n0 == 0:
        return float("nan")

    scores = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(n1, dtype=int), np.zeros(n0, dtype=int)], axis=0)

    # Average ranks with ties.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)

    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j - 1]])
            ranks[order[i:j]] = avg
        i = j

    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    u = rank_sum_pos - (n1 * (n1 + 1) / 2.0)
    auc = u / float(n1 * n0)
    return float(auc)


def _dprime_from_scores(*, pos: Any, neg: Any) -> float:
    import numpy as np  # type: ignore[import-not-found]

    pos = np.asarray(pos).astype(float, copy=False)
    neg = np.asarray(neg).astype(float, copy=False)

    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    if pos.size == 0 or neg.size == 0:
        return float("nan")

    mu_s = float(np.mean(pos))
    mu_n = float(np.mean(neg))
    s2 = float(np.var(pos))
    n2 = float(np.var(neg))

    denom = (0.5 * (s2 + n2)) ** 0.5
    if denom <= 0:
        return float("nan")

    return float((mu_s - mu_n) / denom)


def compute_botm_template_metrics_for_unit(
    *,
    uid: Any,
    well_out_dir: Path,
    logger: logging.Logger,
    templates_out_dir: Optional[Path] = None,
    waveforms_out_dir: Optional[Path] = None,
    n_spike: int = 200,
    n_noise: int = 200,
    noise_model: str = "diag",
    baseline_frac: float = 0.25,
    seed: int = 0,
    waveforms_source: str = "concat",
    negatives_mode: str = "gaussian",
) -> dict[str, Any]:
    """Compute BOTM-style validation metrics for one unit.

    MVP behavior:
    - template source: merged_contributing template from templates stage
    - waveforms source: concat waveforms analyzer
    - covariance: diagonal with per-channel sigma estimated from baseline region
    - negatives: synthetic Gaussian noise (same sigma)

    Returns a JSON-serializable dict with keys: unit_id, metrics, status, error, etc.
    """

    record: dict[str, Any] = {
        "unit_id": _jsonable(uid),
        "status": "ok",
        "error": None,
        "inputs": {},
        "metrics": {},
    }

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"numpy unavailable: {e}"
        return record

    templates_out_dir = Path(templates_out_dir) if templates_out_dir is not None else Path(well_out_dir) / "templates_outputs"
    waveforms_out_dir = Path(waveforms_out_dir) if waveforms_out_dir is not None else Path(well_out_dir) / "waveforms_outputs"

    unit_dir = templates_out_dir / "templates" / "merged" / f"unit_{uid}"
    if not unit_dir.exists():
        # Backward compatibility
        legacy = templates_out_dir / "merged_units" / f"unit_{uid}"
        if legacy.exists():
            unit_dir = legacy

    tmpl_path = unit_dir / "merged_contributing_template.npy"
    meta_path = unit_dir / "merged_contributing_template_meta.json"

    record["inputs"].update(
        {
            "templates_out_dir": str(templates_out_dir),
            "waveforms_out_dir": str(waveforms_out_dir),
            "template_npy": str(tmpl_path),
            "template_meta_json": str(meta_path),
        }
    )

    if not tmpl_path.exists() or not meta_path.exists():
        record["status"] = "error"
        record["error"] = "missing template inputs"
        return record

    try:
        tmpl = np.load(tmpl_path)
        meta = _read_json(meta_path)
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"failed loading template/meta: {e}"
        return record

    if tmpl.ndim != 2:
        record["status"] = "error"
        record["error"] = f"unexpected template shape: {tuple(tmpl.shape)}"
        return record

    # Keep templates at their native timebase (may be upsampled upstream).
    tmpl_use = tmpl

    template_channel_ids = None
    try:
        ch_ids = meta.get("channel_ids")
        if isinstance(ch_ids, list) and len(ch_ids) == int(tmpl.shape[1]):
            template_channel_ids = ch_ids
    except Exception:
        template_channel_ids = None

    record["inputs"].update(
        {
            "template_shape": [int(x) for x in tmpl.shape],
            "template_channel_ids_present": bool(template_channel_ids is not None),
            "sampling_frequency_hz": meta.get("sampling_frequency_hz"),
            "ms_before": meta.get("ms_before"),
            "ms_after": meta.get("ms_after"),
            "noise_model": str(noise_model),
            "seed": int(seed),
            "waveforms_source": str(waveforms_source),
            "negatives_mode": str(negatives_mode),
        }
    )

    if str(noise_model).lower() != "diag":
        record["status"] = "error"
        record["error"] = f"unsupported noise_model for MVP: {noise_model!r}"
        return record

    rng = np.random.default_rng(int(seed))

    # Load waveforms analyzers (concat, optionally segments) and embed their waveforms
    # into the merged-template channel order.
    try:
        analyzers = _try_load_waveforms_analyzers(
            well_out_dir=Path(well_out_dir),
            waveforms_out_dir=Path(waveforms_out_dir),
            waveforms_source=str(waveforms_source),
            logger=logger,
        )
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"failed loading waveforms analyzers: {e}"
        return record

    wfs_sources: list[np.ndarray] = []
    spikes_by_source: dict[str, int] = {}
    used_sources: list[str] = []

    target_t: int = int(tmpl_use.shape[0])
    concat_waveforms_native_t: Optional[int] = None
    waveforms_native_t_by_source: dict[str, int] = {}

    for src_name, analyzer in analyzers:
        waveforms = _try_get_waveforms_one_unit(analyzer=analyzer, unit_id=uid)
        if waveforms is None:
            continue

        try:
            wfs_src = np.asarray(waveforms)
        except Exception:
            continue

        if wfs_src.ndim != 3 or int(wfs_src.shape[0]) == 0:
            continue

        try:
            waveforms_native_t_by_source[str(src_name)] = int(wfs_src.shape[1])
        except Exception:
            pass

        # Remember the native (pre-resample) waveform window length for the concat analyzer.
        # This is the most appropriate window length for sampling random negatives from the
        # concat recording (before upsampling those windows to match the template timebase).
        if concat_waveforms_native_t is None and str(src_name) == "concat":
            try:
                concat_waveforms_native_t = int(wfs_src.shape[1])
            except Exception:
                concat_waveforms_native_t = None

        # Resample waveforms to match template time length (common case: 30 -> 300).
        if int(wfs_src.shape[1]) != int(target_t):
            old_t = int(wfs_src.shape[1])
            try:
                wfs_src = _resample_waveforms_time_to_target_t(
                    waveforms_n_t_by_c=wfs_src,
                    target_t=int(target_t),
                    method="sinc",
                )
                record.setdefault("warnings", [])
                record["warnings"].append(
                    f"resampled waveforms time axis for source={str(src_name)!r} from T={old_t} to T={int(target_t)}"
                )
            except Exception:
                continue

        # Cap per-source spikes to avoid huge memory.
        n_have = int(wfs_src.shape[0])
        cap = min(n_have, int(max(1, n_spike)))
        try:
            keep_idx = rng.choice(n_have, size=cap, replace=False)
            wfs_src = wfs_src[keep_idx]
        except Exception:
            wfs_src = wfs_src[:cap]

        wf_channel_ids = _resolve_unit_channel_ids(analyzer=analyzer, unit_id=uid)
        try:
            if template_channel_ids is not None:
                wfs_src = _embed_waveforms_into_template_channel_order(
                    waveforms=wfs_src,
                    waveforms_channel_ids=wf_channel_ids,
                    template_channel_ids=template_channel_ids,
                )
            else:
                # No mapping available: only accept if channel counts match.
                if int(wfs_src.shape[2]) != int(tmpl.shape[1]):
                    continue
        except Exception:
            continue

        if int(wfs_src.shape[2]) != int(tmpl.shape[1]):
            continue

        used_sources.append(str(src_name))
        spikes_by_source[str(src_name)] = int(wfs_src.shape[0])
        wfs_sources.append(wfs_src)

    if not wfs_sources:
        record["status"] = "error"
        record["error"] = "missing waveforms for unit across requested sources"
        return record

    record["inputs"].update({"waveforms_target_t": int(target_t), "template_time_resampled": False})
    if concat_waveforms_native_t is not None:
        record["inputs"]["concat_waveforms_native_t"] = int(concat_waveforms_native_t)

    try:
        wfs_all = np.concatenate(wfs_sources, axis=0)
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"failed concatenating waveforms across sources: {e}"
        return record

    # Final subsample to n_spike.
    n_have_all = int(wfs_all.shape[0])
    n_keep_all = min(int(max(1, n_spike)), n_have_all)
    try:
        keep_idx = rng.choice(n_have_all, size=n_keep_all, replace=False)
        wfs = wfs_all[keep_idx]
    except Exception:
        wfs = wfs_all[:n_keep_all]

    record["inputs"].update(
        {
            "waveforms_sources_used": used_sources,
            "n_waveforms_by_source": {k: int(v) for k, v in spikes_by_source.items()},
            "n_waveforms_total_before_final_subsample": int(n_have_all),
            "n_waveforms_used": int(wfs.shape[0]),
        }
    )

    # Estimate sigma (per channel) from baseline region.
    try:
        sigma_ch = _estimate_sigma_per_channel_from_baseline(waveforms=wfs, baseline_frac=float(baseline_frac))
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"sigma estimation failed: {e}"
        return record

    # BOTM filter and spike scores.
    try:
        f = _botm_filter_from_template(template_t_by_c=tmpl_use, sigma_ch=sigma_ch)
        scores_spike = _scores_for_waveforms(waveforms_t_by_c=wfs, f_t_by_c=f)
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"score computation failed: {e}"
        return record

    # Noise scores: default to synthetic Gaussian; optionally sample from concat recording.
    mode = str(negatives_mode).lower().strip()

    if mode in {"gaussian", "synthetic", "normal"}:
        try:
            nN = int(max(1, n_noise))
            noise = rng.normal(
                loc=0.0,
                scale=np.asarray(sigma_ch)[None, None, :],
                size=(nN, int(tmpl_use.shape[0]), int(tmpl_use.shape[1])),
            )
            scores_noise = _scores_for_waveforms(waveforms_t_by_c=noise, f_t_by_c=f)
        except Exception as e:
            record["status"] = "error"
            record["error"] = f"noise score generation failed: {e}"
            return record

    elif mode in {"recording_random", "random_recording", "recording"}:
        # Best-effort: sample random windows from the concat recording.
        # If the required channels are not available, fall back to Gaussian.
        scores_noise = None
        try:
            if template_channel_ids is None:
                raise RuntimeError("cannot sample recording_random negatives without template channel ids")

            nN = int(max(1, n_noise))
            T = int(tmpl_use.shape[0])
            c_template = int(tmpl_use.shape[1])

            # Use any analyzers that have a recording attached (concat + segments).
            sources_with_rec = []
            for src_name, an in analyzers:
                rec = getattr(an, "recording", None)
                if rec is None:
                    continue
                sources_with_rec.append((str(src_name), rec))

            if not sources_with_rec:
                raise RuntimeError("no recordings attached to any waveforms analyzers")

            # Distribute noise windows across sources (best-effort).
            n_sources = int(len(sources_with_rec))
            n_per = int((nN + n_sources - 1) // n_sources)
            noise_chunks: list[np.ndarray] = []
            used_noise_sources: dict[str, int] = {}

            for src_name, rec in sources_with_rec:
                try:
                    rec_ch_ids = list(rec.get_channel_ids())
                except Exception:
                    continue

                rec_id_to_idx = {str(cid): int(i) for i, cid in enumerate(rec_ch_ids)}
                # Which template channels exist in this recording?
                template_positions: list[int] = []
                rec_indices: list[int] = []
                for j, cid in enumerate(template_channel_ids):
                    idx = rec_id_to_idx.get(str(cid))
                    if idx is None:
                        continue
                    template_positions.append(int(j))
                    rec_indices.append(int(idx))

                if not rec_indices:
                    continue

                # Choose native window length for this source (prefer its waveforms length).
                T_native = int(waveforms_native_t_by_source.get(str(src_name)) or concat_waveforms_native_t or T)
                n_total = int(rec.get_num_samples())
                if n_total <= T_native + 1:
                    continue

                starts = rng.integers(low=0, high=max(1, n_total - T_native), size=int(n_per), dtype=int)
                windows_src: list[np.ndarray] = []
                for s in starts.tolist():
                    tr = rec.get_traces(start_frame=int(s), end_frame=int(s + T_native))
                    tr = np.asarray(tr)[:, rec_indices]  # (T_native, n_found_channels)

                    # Embed into full merged-template channel order with zeros for missing channels.
                    full = np.zeros((int(T_native), int(c_template)), dtype=float)
                    full[:, template_positions] = tr
                    windows_src.append(full)

                if not windows_src:
                    continue

                noise_native = np.stack(windows_src, axis=0)  # (n_per, T_native, C_template)
                if int(noise_native.shape[1]) != int(T):
                    noise_src = _resample_waveforms_time_to_target_t(
                        waveforms_n_t_by_c=noise_native,
                        target_t=int(T),
                        method="sinc",
                    )
                else:
                    noise_src = noise_native

                noise_chunks.append(np.asarray(noise_src))
                used_noise_sources[str(src_name)] = int(noise_src.shape[0])

                if int(sum(int(x.shape[0]) for x in noise_chunks)) >= int(nN):
                    break

            if not noise_chunks:
                raise RuntimeError("no noise windows could be sampled from any recording source")

            noise_wfs = np.concatenate(noise_chunks, axis=0)[: int(nN), :, :]
            record["inputs"].setdefault("recording_random_noise_sources", used_noise_sources)
            scores_noise = _scores_for_waveforms(waveforms_t_by_c=noise_wfs, f_t_by_c=f)
        except Exception as e:
            record.setdefault("warnings", [])
            record["warnings"].append(f"recording_random negatives failed; falling back to gaussian: {e}")
            try:
                nN = int(max(1, n_noise))
                noise = rng.normal(
                    loc=0.0,
                    scale=np.asarray(sigma_ch)[None, None, :],
                    size=(nN, int(tmpl_use.shape[0]), int(tmpl_use.shape[1])),
                )
                scores_noise = _scores_for_waveforms(waveforms_t_by_c=noise, f_t_by_c=f)
            except Exception as e2:
                record["status"] = "error"
                record["error"] = f"noise score generation failed: {e2}"
                return record

        if scores_noise is None:
            record["status"] = "error"
            record["error"] = "noise negatives mode failed"
            return record

    else:
        record["status"] = "error"
        record["error"] = f"unsupported negatives_mode: {negatives_mode!r}"
        return record

    # Summary metrics.
    try:
        auc = _auc_from_scores(pos=scores_spike, neg=scores_noise)
        dprime = _dprime_from_scores(pos=scores_spike, neg=scores_noise)
        record["metrics"].update(
            {
                "n_spike": int(np.asarray(scores_spike).size),
                "n_noise": int(np.asarray(scores_noise).size),
                "score_mean_spike": float(np.nanmean(scores_spike)),
                "score_std_spike": float(np.nanstd(scores_spike)),
                "score_mean_noise": float(np.nanmean(scores_noise)),
                "score_std_noise": float(np.nanstd(scores_noise)),
                "auc": float(auc),
                "dprime": float(dprime),
            }
        )
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"metric computation failed: {e}"
        return record

    return record


def write_botm_validation_outputs(
    *,
    inputs: BotmValidationInputs,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Compute and persist BOTM validation metrics for units.

    Writes JSON artifacts under:
      <analysis_outputs>/botm_validation/
        summary.json
        by_unit/unit_<uid>_botm_metrics.json

    This is designed to be called from analysis stage, but is not wired in yet.

    Returns the run summary dict.
    """

    if logger is None:
        logger = logging.getLogger("axon_reconstructor.botm_validation")

    out_dir = Path(inputs.out_dir) if inputs.out_dir is not None else Path(inputs.well_out_dir) / "analysis_outputs" / "botm_validation"
    by_unit_dir = out_dir / "by_unit"

    if inputs.force_restart and out_dir.exists():
        # Best-effort cleanup: only delete known JSON outputs.
        try:
            for p in by_unit_dir.glob("unit_*_botm_metrics.json"):
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    by_unit_dir.mkdir(parents=True, exist_ok=True)

    # Unit discovery: prefer explicit unit_ids, else discover from templates merged dir.
    unit_ids: list[Any]
    if inputs.unit_ids is not None:
        unit_ids = list(inputs.unit_ids)
    else:
        templates_out_dir = Path(inputs.templates_out_dir) if inputs.templates_out_dir is not None else Path(inputs.well_out_dir) / "templates_outputs"
        merged_units_dir = templates_out_dir / "templates" / "merged"
        if not merged_units_dir.exists():
            legacy = templates_out_dir / "merged_units"
            if legacy.exists():
                merged_units_dir = legacy
        discovered: list[Any] = []
        for p in sorted(merged_units_dir.glob("unit_*") if merged_units_dir.exists() else []):
            if not p.is_dir():
                continue
            try:
                discovered.append(int(p.name.split("unit_", 1)[1]))
            except Exception:
                discovered.append(p.name.split("unit_", 1)[1])
        unit_ids = discovered

    results: list[dict[str, Any]] = []
    for uid in unit_ids:
        logger.info("BOTM validation: unit=%s", str(uid))
        rec = compute_botm_template_metrics_for_unit(
            uid=uid,
            well_out_dir=inputs.well_out_dir,
            logger=logger,
            templates_out_dir=inputs.templates_out_dir,
            waveforms_out_dir=inputs.waveforms_out_dir,
            n_spike=int(inputs.n_spike),
            n_noise=int(inputs.n_noise),
            noise_model=str(inputs.noise_model),
            baseline_frac=float(inputs.baseline_frac),
            seed=int(inputs.seed),
            waveforms_source=str(inputs.waveforms_source),
            negatives_mode=str(inputs.negatives_mode),
        )

        out_json = by_unit_dir / f"unit_{uid}_botm_metrics.json"
        _write_json(out_json, rec)

        # Include pointers in the summary.
        rec2 = dict(rec)
        rec2.setdefault("artifacts", {})
        rec2["artifacts"].update({"metrics_json": str(out_json)})
        results.append(rec2)

    summary = {
        "well_out_dir": str(inputs.well_out_dir),
        "out_dir": str(out_dir),
        "by_unit_dir": str(by_unit_dir),
        "n_units": int(len(unit_ids)),
        "params": {
            "n_spike": int(inputs.n_spike),
            "n_noise": int(inputs.n_noise),
            "noise_model": str(inputs.noise_model),
            "waveforms_source": str(inputs.waveforms_source),
            "negatives_mode": str(inputs.negatives_mode),
            "baseline_frac": float(inputs.baseline_frac),
            "seed": int(inputs.seed),
        },
        "units": results,
    }

    _write_json(out_dir / "summary.json", summary)
    return summary
