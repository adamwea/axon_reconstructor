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

    Notes on scope (current):
    - Uses merged_contributing template outputs.
    - Can score waveforms from concat and/or per-segment analyzers.
    - Supports diagonal or (channel) covariance noise models.
    - Can sample negatives from recordings, optionally restricted to spike-free windows.
    """

    well_out_dir: Path

    templates_out_dir: Optional[Path] = None
    waveforms_out_dir: Optional[Path] = None

    unit_ids: Optional[list[Any]] = None

    n_spike: int = 200
    n_noise: int = 200

    # If True (default), each waveforms source is capped to at most n_spike
    # before concatenation. This keeps memory bounded when using many sources.
    # If False, all available waveforms from each source are kept (can be large).
    cap_waveforms_per_source_to_n_spike: bool = True

    # Noise model controls.
    # - "cov" (default): channel covariance (C x C) estimated from spike-free recording windows when possible
    # - "diag": diagonal covariance using per-channel sigma
    noise_model: str = "cov"

    # Which waveforms sources to score against the merged template.
    # - "concat": concat_waveforms only
    # - "concat+segments": concat + all readable segment analyzers
    waveforms_source: str = "concat"

    # Negative sampling mode.
    # - "recording_spikefree": windows sampled from the recording excluding detected spike times (best-effort)
    # - "recording_random": random windows sampled from the recording (best-effort)
    # - "gaussian": synthetic Gaussian noise using estimated noise stats
    negatives_mode: str = "recording_spikefree"

    # Baseline region used to estimate noise sigma from spike snippets.
    # This does NOT need to match the full template length.
    baseline_frac: float = 0.25

    # RNG seed for subsampling spikes and generating synthetic noise.
    seed: int = 0

    # Channel-level matching (Figure-5-style / 2023 validation):
    # For each merged-template channel, compute the fraction of spike waveforms whose
    # trials "match" a template.
    #
    # In the 2023 Figure 5 description, the waveform of the template is compared
    # against each single trial and the % of "matching" trials is computed per peak.
    # In our pipeline, we treat each template channel as a peak location and compute
    # the per-channel % matching across waveforms.
    #
    # Default behavior uses the Franke 2015 BOTM discriminant (colored Gaussian noise):
    #   D(x) = x^T C^{-1} xi - 0.5 * xi^T C^{-1} xi + ln(p_signal)
    # and declares a match when:
    #   D(x) >= ln(1 - p_signal)
    # (single-template vs noise model; see Franke 2015 eqns referenced in the paper).
    #
    # The prior p_signal is the probability of a signal/template being present at the
    # evaluated alignment position. In this validation context we default to 0.5
    # (symmetric decision at a known candidate location), but it can be set.
    #
    # Legacy behavior (optional) uses a residual-RMS threshold:
    #   match iff RMS(w-template) <= (noise_std_level * sigma_noise)
    channel_match_fraction_threshold: float = 0.70
    channel_match_method: str = "botm_franke2015"
    channel_match_prior_signal: float = 0.5
    channel_match_noise_std_level: float = 3.0

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


def _try_load_reconstruction_channel_sets_for_unit(*, well_out_dir: Path, uid: Any) -> tuple[set[Any], set[Any], dict[str, str]]:
    """Best-effort load of reconstruction channel sets.

    The reconstruction stage persists stable JSONs:
      - reconstruction_outputs/by_unit/unit_<uid>/branches_raw.json  (raw)
      - reconstruction_outputs/by_unit/unit_<uid>/branches.json      (cleaned)

    We treat:
      - "raw" nodes as the union of per-branch channels in branches_raw.json
      - "cleaned" nodes as the union of per-branch channels in branches.json
    """

    base = Path(well_out_dir) / "reconstruction_outputs" / "by_unit" / f"unit_{uid}"
    branches_raw_path = base / "branches_raw.json"
    branches_path = base / "branches.json"
    paths: dict[str, str] = {}

    raw_nodes: set[Any] = set()
    cleaned_nodes: set[Any] = set()

    if branches_raw_path.exists():
        paths["branches_raw_json"] = str(branches_raw_path)
        try:
            payload = _read_json(branches_raw_path)
            branches = payload.get("branches", []) if isinstance(payload, dict) else []
            if isinstance(branches, list):
                for b in branches:
                    if not isinstance(b, dict):
                        continue
                    ch = b.get("channels", [])
                    if isinstance(ch, list):
                        raw_nodes.update(ch)
        except Exception:
            pass

    if branches_path.exists():
        paths["branches_json"] = str(branches_path)
        try:
            payload = _read_json(branches_path)
            branches = payload.get("branches", []) if isinstance(payload, dict) else []
            if isinstance(branches, list):
                for b in branches:
                    if not isinstance(b, dict):
                        continue
                    ch = b.get("channels", [])
                    if isinstance(ch, list):
                        cleaned_nodes.update(ch)
        except Exception:
            pass

    return raw_nodes, cleaned_nodes, paths


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


def _cov_reg_identity(*, cov: Any, reg_frac: float) -> Any:
    """Add a small ridge term to a covariance matrix for numerical stability."""

    import numpy as np  # type: ignore[import-not-found]

    if reg_frac <= 0:
        return cov
    cov = np.asarray(cov, dtype=float)
    n = int(cov.shape[0])
    if n == 0:
        return cov
    tr = float(np.trace(cov))
    if not np.isfinite(tr) or tr <= 0:
        tr = 1.0
    return cov + (reg_frac * (tr / max(1, n))) * np.eye(n, dtype=float)


def _botm_discriminant_franke2015_one_channel(
    *,
    waveforms_nt: Any,  # (N,T)
    template_t: Any,  # (T,)
    noise_nt: Optional[Any],  # (M,T) or None
    prior_signal: float,
    reg_frac: float = 1e-3,
) -> tuple[Any, float, dict[str, Any]]:
    """Compute Franke 2015 BOTM discriminant and threshold for a single channel.

    Returns:
      - D_n: discriminant values per trial (N,)
      - thr: threshold on D
      - meta: diagnostic payload (cov source, conditioning warnings)
    """

    import numpy as np  # type: ignore[import-not-found]

    x = np.asarray(waveforms_nt, dtype=float)
    xi = np.asarray(template_t, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"waveforms_nt must be 2D (N,T), got shape={x.shape}")
    if xi.ndim != 1:
        raise ValueError(f"template_t must be 1D (T,), got shape={xi.shape}")
    n_trials, t_len = int(x.shape[0]), int(x.shape[1])
    if int(xi.shape[0]) != t_len:
        raise ValueError(f"template length {int(xi.shape[0])} != waveform T {t_len}")

    p = float(prior_signal)
    if not (0.0 < p < 1.0):
        raise ValueError(f"channel_match_prior_signal must be in (0,1), got {p}")

    # Estimate temporal covariance C (T x T) from noise snippets.
    cov_src = "noise_windows"
    if noise_nt is None:
        cov_src = "waveforms_baseline_fallback"
        # Fallback: treat baseline-only noise as independent over time.
        # This is less faithful to Franke 2015 colored-noise assumption but keeps scoring defined.
        # Use variance across trials for each time point as a diagonal C.
        v = np.nanvar(x, axis=0)
        v = np.where(np.isfinite(v) & (v > 1e-12), v, 1e-12)
        C = np.diag(v)
    else:
        n = np.asarray(noise_nt, dtype=float)
        if n.ndim != 2 or int(n.shape[1]) != t_len:
            raise ValueError(f"noise_nt must be (M,T={t_len}), got shape={n.shape}")
        # Center noise.
        n0 = n - np.nanmean(n, axis=0, keepdims=True)
        # Covariance over time.
        C = np.cov(n0, rowvar=False, bias=False)

    C = _cov_reg_identity(cov=C, reg_frac=float(reg_frac))

    # Solve for f = C^{-1} xi (avoid explicit inverse).
    try:
        f = np.linalg.solve(C, xi)
        ok = True
        warn = None
    except Exception as e:
        ok = False
        warn = f"covariance solve failed: {e}"
        # Last-resort: diagonal approximation.
        diag = np.diag(C)
        diag = np.where(np.isfinite(diag) & (diag > 1e-12), diag, 1e-12)
        f = xi / diag

    xiCf = float(np.dot(xi, f))
    # D = x^T f - 0.5 xi^T f + ln(p)
    D = (x @ f) - 0.5 * xiCf + float(np.log(p))
    thr = float(np.log(1.0 - p))
    meta = {
        "method": "botm_franke2015_discriminant",
        "covariance_source": cov_src,
        "prior_signal": p,
        "threshold": thr,
        "reg_frac": float(reg_frac),
        "solve_ok": bool(ok),
    }
    if warn:
        meta["warning"] = str(warn)
    return np.asarray(D, dtype=float).reshape(n_trials), thr, meta


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


def _botm_offset_constant(*, template_t_by_c: Any, f_t_by_c: Any) -> float:
    """Compute BOTM constant term 0.5 * xi^T C^{-1} xi (ignoring priors).

    For our (time-independent) channel-covariance approximation, this reduces to:
      0.5 * sum_{t,c} template[t,c] * f[t,c]
    """

    import numpy as np  # type: ignore[import-not-found]

    tmpl = np.asarray(template_t_by_c)
    f = np.asarray(f_t_by_c)
    if tmpl.shape != f.shape:
        raise ValueError(f"shape mismatch for offset: template={tmpl.shape} f={f.shape}")
    return float(0.5 * float(np.sum(tmpl * f)))


def _scores_for_waveforms_with_offset(*, waveforms_t_by_c: Any, f_t_by_c: Any, offset: float) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    s = _scores_for_waveforms(waveforms_t_by_c=waveforms_t_by_c, f_t_by_c=f_t_by_c)
    return np.asarray(s, dtype=float) - float(offset)


def _estimate_channel_covariance_from_windows(*, windows_n_t_by_c: Any, diag_load: float = 1e-3) -> Any:
    """Estimate channel covariance (C x C) from noise windows.

    Approximates the full spatio-temporal covariance used in Franke et al. by a
    time-independent channel covariance (shared across t). This is much cheaper
    and still captures cross-channel correlations.
    """

    import numpy as np  # type: ignore[import-not-found]

    w = np.asarray(windows_n_t_by_c)
    if w.ndim != 3:
        raise ValueError(f"expected (N, T, C), got {w.shape}")

    n, t, c = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
    if n <= 0 or t <= 0 or c <= 0:
        raise ValueError("empty windows")

    x = w.reshape(-1, c).astype(float, copy=False)
    # Centering is cheap and makes covariance more stable.
    mu = np.mean(x, axis=0)
    x = x - mu[None, :]
    denom = max(1, int(x.shape[0]))
    cov = (x.T @ x) / float(denom)

    # Diagonal loading for numerical stability.
    diag = np.diag(cov)
    scale = float(np.nanmean(diag)) if diag.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    lam = float(max(0.0, diag_load)) * scale + 1e-12
    cov = cov + lam * np.eye(int(c), dtype=float)
    return cov


def _botm_filter_from_template_cov(*, template_t_by_c: Any, cov_c_by_c: Any) -> Any:
    """Return BOTM filter f[t] = C^{-1} * template[t] for channel covariance C."""

    import numpy as np  # type: ignore[import-not-found]

    tmpl = np.asarray(template_t_by_c)
    cov = np.asarray(cov_c_by_c)
    if tmpl.ndim != 2:
        raise ValueError(f"Unexpected template shape: {tmpl.shape}")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != tmpl.shape[1]:
        raise ValueError(f"Unexpected cov shape: {cov.shape}, expected ({tmpl.shape[1]}, {tmpl.shape[1]})")

    # Solve C * F^T = template^T  =>  F = (solve(C, template^T))^T
    f = np.linalg.solve(cov, tmpl.T).T
    return f


def _union_spike_times_from_sorting(*, sorting: Any) -> Any:
    """Union spike times across units for segment 0, returned sorted int array."""

    import numpy as np  # type: ignore[import-not-found]

    unit_ids = []
    try:
        unit_ids = list(sorting.get_unit_ids())
    except Exception:
        unit_ids = []

    all_times: list[np.ndarray] = []
    for uid in unit_ids:
        try:
            st = sorting.get_unit_spike_train(unit_id=uid, segment_index=0)
        except Exception:
            try:
                st = sorting.get_unit_spike_train(uid)
            except Exception:
                continue
        try:
            st = np.asarray(st, dtype=int)
        except Exception:
            continue
        if st.size:
            all_times.append(st)

    if not all_times:
        return np.asarray([], dtype=int)
    out = np.unique(np.concatenate(all_times, axis=0).astype(int, copy=False))
    out.sort()
    return out


def _sample_spikefree_starts(
    *,
    rng: Any,
    spike_times_sorted: Any,
    n_total: int,
    window_len: int,
    pad: int,
    n_needed: int,
) -> list[int]:
    """Sample start indices such that no spikes fall within [start-pad, start+window_len+pad)."""

    import numpy as np  # type: ignore[import-not-found]

    n_total = int(n_total)
    window_len = int(window_len)
    pad = int(max(0, pad))
    n_needed = int(max(0, n_needed))
    if n_needed <= 0:
        return []

    st = np.asarray(spike_times_sorted, dtype=int)
    if st.size == 0:
        # Nothing to exclude.
        hi = max(1, n_total - window_len)
        starts = rng.integers(low=0, high=hi, size=int(n_needed), dtype=int)
        return [int(x) for x in starts.tolist()]

    max_start = int(n_total - window_len)
    if max_start <= 1:
        return []

    starts: list[int] = []
    max_tries = int(max(500, 50 * n_needed))
    for _ in range(max_tries):
        s = int(rng.integers(low=0, high=max_start, size=1, dtype=int)[0])
        lo = int(max(0, s - pad))
        hi = int(min(n_total, s + window_len + pad))

        i = int(np.searchsorted(st, lo, side="left"))
        if i < int(st.size) and int(st[i]) < hi:
            continue
        starts.append(s)
        if len(starts) >= n_needed:
            break

    return starts


def _sample_noise_windows_multi_source(
    *,
    analyzers: list[tuple[str, Any]],
    template_channel_ids: list[Any],
    c_template: int,
    target_t: int,
    waveforms_native_t_by_source: dict[str, int],
    concat_waveforms_native_t: Optional[int],
    rng: Any,
    n_needed: int,
    spikefree: bool,
) -> tuple[Any, dict[str, int], list[str]]:
    """Sample noise windows from recordings (concat + segments), embedded into template channel order.

    Returns (noise_wfs_target_t, used_noise_sources, warnings).
    """

    import numpy as np  # type: ignore[import-not-found]

    n_needed = int(max(1, n_needed))
    target_t = int(target_t)
    c_template = int(c_template)
    warnings: list[str] = []

    sources_with_rec: list[tuple[str, Any, Any]] = []
    for src_name, an in analyzers:
        rec = getattr(an, "recording", None)
        if rec is None:
            continue
        sources_with_rec.append((str(src_name), an, rec))

    if not sources_with_rec:
        raise RuntimeError("no recordings attached to any waveforms analyzers")

    n_sources = int(len(sources_with_rec))
    n_per = int((n_needed + n_sources - 1) // n_sources)
    noise_chunks: list[np.ndarray] = []
    used_noise_sources: dict[str, int] = {}

    for src_name, an, rec in sources_with_rec:
        try:
            rec_ch_ids = list(rec.get_channel_ids())
        except Exception:
            continue

        rec_id_to_idx = {str(cid): int(i) for i, cid in enumerate(rec_ch_ids)}
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

        T_native = int(waveforms_native_t_by_source.get(str(src_name)) or concat_waveforms_native_t or target_t)
        n_total = int(rec.get_num_samples())
        if n_total <= T_native + 1:
            continue

        starts: list[int]
        if spikefree:
            sorting = getattr(an, "sorting", None)
            if sorting is None:
                warnings.append(f"source={src_name!r} has no sorting; spikefree sampling falling back to random")
                hi = max(1, n_total - T_native)
                arr = rng.integers(low=0, high=hi, size=int(n_per), dtype=int)
                starts = [int(x) for x in arr.tolist()]
            else:
                st = _union_spike_times_from_sorting(sorting=sorting)
                pad = int(max(1, round(0.5 * float(T_native))))
                starts = _sample_spikefree_starts(
                    rng=rng,
                    spike_times_sorted=st,
                    n_total=int(n_total),
                    window_len=int(T_native),
                    pad=int(pad),
                    n_needed=int(n_per),
                )
                if len(starts) < int(n_per):
                    warnings.append(
                        f"source={src_name!r} spikefree windows: requested={int(n_per)} got={int(len(starts))}"
                    )
        else:
            hi = max(1, n_total - T_native)
            arr = rng.integers(low=0, high=hi, size=int(n_per), dtype=int)
            starts = [int(x) for x in arr.tolist()]

        windows_src: list[np.ndarray] = []
        for s in starts:
            try:
                tr = rec.get_traces(start_frame=int(s), end_frame=int(s + T_native))
                tr = np.asarray(tr)[:, rec_indices]
            except Exception:
                continue

            full = np.zeros((int(T_native), int(c_template)), dtype=float)
            full[:, template_positions] = tr
            windows_src.append(full)

        if not windows_src:
            continue

        noise_native = np.stack(windows_src, axis=0)
        if int(noise_native.shape[1]) != int(target_t):
            noise_src = _resample_waveforms_time_to_target_t(
                waveforms_n_t_by_c=noise_native,
                target_t=int(target_t),
                method="sinc",
            )
        else:
            noise_src = noise_native

        noise_chunks.append(np.asarray(noise_src))
        used_noise_sources[str(src_name)] = int(noise_src.shape[0])

        if int(sum(int(x.shape[0]) for x in noise_chunks)) >= int(n_needed):
            break

    if not noise_chunks:
        raise RuntimeError("no noise windows could be sampled from any recording source")

    noise_wfs = np.concatenate(noise_chunks, axis=0)[: int(n_needed), :, :]
    return noise_wfs, used_noise_sources, warnings


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
    cap_waveforms_per_source_to_n_spike: bool = True,
    noise_model: str = "cov",
    baseline_frac: float = 0.25,
    seed: int = 0,
    waveforms_source: str = "concat",
    negatives_mode: str = "recording_spikefree",
    channel_match_fraction_threshold: float = 0.70,
    channel_match_method: str = "botm_franke2015",
    channel_match_prior_signal: float = 0.5,
    channel_match_noise_std_level: float = 3.0,
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
            "cap_waveforms_per_source_to_n_spike": bool(cap_waveforms_per_source_to_n_spike),
            "noise_model": str(noise_model),
            "seed": int(seed),
            "waveforms_source": str(waveforms_source),
            "negatives_mode": str(negatives_mode),
            "channel_match_fraction_threshold": float(channel_match_fraction_threshold),
            "channel_match_method": str(channel_match_method),
            "channel_match_prior_signal": float(channel_match_prior_signal),
            "channel_match_noise_std_level": float(channel_match_noise_std_level),
        }
    )

    noise_model_req = str(noise_model).lower().strip()
    if noise_model_req not in {"diag", "cov", "covariance"}:
        record["status"] = "error"
        record["error"] = f"unsupported noise_model: {noise_model!r}"
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

        # Cap per-source spikes to avoid huge memory (optional).
        if bool(cap_waveforms_per_source_to_n_spike):
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

    # --- Noise statistics windows (spike-free when possible) ---
    noise_stats_wfs = None
    noise_stats_sources: Optional[dict[str, int]] = None
    if template_channel_ids is not None:
        n_stats = int(min(2000, max(200, int(max(1, n_noise)))))
        try:
            noise_stats_wfs, noise_stats_sources, stats_warnings = _sample_noise_windows_multi_source(
                analyzers=analyzers,
                template_channel_ids=list(template_channel_ids),
                c_template=int(tmpl_use.shape[1]),
                target_t=int(target_t),
                waveforms_native_t_by_source=waveforms_native_t_by_source,
                concat_waveforms_native_t=concat_waveforms_native_t,
                rng=rng,
                n_needed=int(n_stats),
                spikefree=True,
            )
            record["inputs"].setdefault("noise_stats_mode", "recording_spikefree")
            record["inputs"].setdefault("noise_stats_n", int(np.asarray(noise_stats_wfs).shape[0]))
            record["inputs"].setdefault("noise_stats_sources", noise_stats_sources)
            if stats_warnings:
                record.setdefault("warnings", [])
                record["warnings"].extend([f"noise_stats: {w}" for w in stats_warnings])
        except Exception as e:
            record.setdefault("warnings", [])
            record["warnings"].append(f"noise_stats sampling failed (will fall back): {e}")

    # --- Channel-level matching (per template channel) ---
    # Mirrors the 2023 Figure-5-style reporting:
    #   - compare the template waveform to each single trial
    #   - count the % of trials that "match"
    # Default: Franke 2015 BOTM discriminant with an analytic threshold.
    if template_channel_ids is not None:
        try:
            tmpl_c = int(tmpl_use.shape[1])
            if int(wfs.shape[2]) != tmpl_c:
                raise RuntimeError(f"waveforms C={int(wfs.shape[2])} does not match template C={tmpl_c}")

            method_req = str(channel_match_method).lower().strip()
            # Back-compat aliases:
            # - previous default: "botm_llr" (diagonal, no-prior) -> now replaced by Franke 2015 discriminant
            #   with temporal covariance and a prior.
            botm_req = method_req in {
                "botm",
                "botm_franke2015",
                "franke2015",
                "botm_discriminant",
                "botm_llr",
                "llr",
            }

            if botm_req:
                # Franke 2015 per-channel BOTM discriminant:
                #   D = x^T C^{-1} xi - 0.5 xi^T C^{-1} xi + ln(p)
                #   match iff D >= ln(1-p)
                prior_signal = float(channel_match_prior_signal)
                scores_nc = np.zeros((int(wfs.shape[0]), tmpl_c), dtype=float)
                thr_c = np.zeros((tmpl_c,), dtype=float)
                solve_ok_c: list[bool] = []
                cov_src_c: list[str] = []
                warn_c: list[Optional[str]] = []

                # Use spike-free recording windows when available.
                # noise_stats_wfs shape: (M, T, C)
                noise_src = "waveforms_baseline_fallback"
                noise_stats_arr = None
                if noise_stats_wfs is not None:
                    noise_stats_arr = np.asarray(noise_stats_wfs, dtype=float)
                    if noise_stats_arr.ndim != 3 or int(noise_stats_arr.shape[2]) != tmpl_c:
                        noise_stats_arr = None
                    else:
                        noise_src = "recording_spikefree"

                for ci in range(tmpl_c):
                    noise_nt = None
                    if noise_stats_arr is not None:
                        noise_nt = np.asarray(noise_stats_arr[:, :, ci], dtype=float)

                    Dn, thr, meta = _botm_discriminant_franke2015_one_channel(
                        waveforms_nt=np.asarray(wfs[:, :, ci], dtype=float),
                        template_t=np.asarray(tmpl_use[:, ci], dtype=float),
                        noise_nt=noise_nt,
                        prior_signal=prior_signal,
                    )
                    scores_nc[:, ci] = Dn
                    thr_c[ci] = float(thr)
                    solve_ok_c.append(bool(meta.get("solve_ok", True)))
                    cov_src_c.append(str(meta.get("covariance_source", noise_src)))
                    warn_c.append(meta.get("warning"))

                # Per-channel thresholds from the Bayes prior (same scalar here), but keep vector for clarity.
                match = scores_nc >= thr_c[None, :]
                chm_method = "botm_franke2015_discriminant_ge_thr"
                chm_thr_payload = {
                    "prior_signal": float(prior_signal),
                    "threshold_by_channel": [float(x) for x in thr_c.tolist()],
                    "covariance_source": str(noise_src),
                    "cov_solve_ok_by_channel": [bool(v) for v in solve_ok_c],
                    "cov_source_by_channel": [str(v) for v in cov_src_c],
                    "cov_warning_by_channel": [None if w is None else str(w) for w in warn_c],
                }
            else:
                # Legacy residual-RMS thresholding (kept for backwards comparisons).
                eps = 1e-12
                sigma_src = "waveforms_baseline"
                if noise_stats_wfs is not None:
                    x = np.asarray(noise_stats_wfs, dtype=float).reshape(-1, tmpl_c)
                    sigma_ch = np.nanstd(x, axis=0)
                    sigma_src = "recording_spikefree"
                else:
                    sigma_ch = _estimate_sigma_per_channel_from_baseline(waveforms=wfs, baseline_frac=float(baseline_frac))

                sigma_ch = np.asarray(sigma_ch, dtype=float)
                sigma_ch = np.where(np.isfinite(sigma_ch) & (sigma_ch > eps), sigma_ch, eps)

                residual = np.asarray(wfs, dtype=float) - np.asarray(tmpl_use, dtype=float)[None, :, :]
                residual_rms = np.sqrt(np.nanmean(residual * residual, axis=1))  # (N, C)
                thr = float(channel_match_noise_std_level) * sigma_ch[None, :]
                match = residual_rms <= thr
                scores_nc = -residual_rms
                chm_method = "residual_rms_le_k_sigma"
                chm_thr_payload = {"noise_std_level": float(channel_match_noise_std_level), "sigma_source": str(sigma_src)}

            n_trials = int(match.shape[0])
            n_match = np.sum(match, axis=0).astype(int)
            frac = (n_match.astype(float) / max(1, n_trials)).astype(float)
            good_mask = frac > float(channel_match_fraction_threshold)

            def _ch_id(v: Any) -> Any:
                try:
                    return int(v)
                except Exception:
                    return str(v)

            ch_ids = [_ch_id(v) for v in list(template_channel_ids)]
            good_channel_ids = [ch_ids[i] for i in np.where(good_mask)[0].tolist()]

            record.setdefault("channel_match", {})
            record["channel_match"].update(
                {
                    "method": str(chm_method),
                    "n_trials": int(n_trials),
                    "fraction_threshold": float(channel_match_fraction_threshold),
                    **chm_thr_payload,
                    "template_channel_ids": ch_ids,
                    "match_fraction_by_channel": [float(x) for x in frac.tolist()],
                    "n_matches_by_channel": [int(x) for x in n_match.tolist()],
                    "n_mismatches_by_channel": [int(n_trials - int(x)) for x in n_match.tolist()],
                    "good_channel_ids": good_channel_ids,
                    "n_good_channels": int(len(good_channel_ids)),
                }
            )

            # Keep scores only as summary stats to avoid huge JSON.
            try:
                if isinstance(scores_nc, np.ndarray) and scores_nc.ndim == 2:
                    record["channel_match"]["score_mean_by_channel"] = [float(x) for x in np.nanmean(scores_nc, axis=0).tolist()]
                    record["channel_match"]["score_std_by_channel"] = [float(x) for x in np.nanstd(scores_nc, axis=0).tolist()]
            except Exception:
                pass
        except Exception as e:
            record.setdefault("warnings", [])
            record["warnings"].append(f"channel_match computation failed: {e}")

    # --- Reconstruction intersections (raw vs cleaned) ---
    # Report how the Figure-5-style "good" channels relate to reconstruction nodes.
    try:
        raw_nodes, cleaned_nodes, recon_paths = _try_load_reconstruction_channel_sets_for_unit(
            well_out_dir=Path(well_out_dir),
            uid=uid,
        )
        if recon_paths:
            record.setdefault("inputs", {})
            record["inputs"].setdefault("reconstruction_artifacts", recon_paths)

        chm = record.get("channel_match") if isinstance(record, dict) else None
        if isinstance(chm, dict) and (raw_nodes or cleaned_nodes):
            template_ch_ids = chm.get("template_channel_ids", [])
            good_ch_ids = chm.get("good_channel_ids", [])
            if isinstance(template_ch_ids, list) and isinstance(good_ch_ids, list):
                template_set = set(template_ch_ids)
                good_set = set(good_ch_ids)
                bad_set = set(template_ch_ids) - good_set

                def _sorted_list(xs: set[Any]) -> list[Any]:
                    try:
                        return sorted(xs)
                    except Exception:
                        return [str(x) for x in sorted([str(v) for v in xs])]

                raw_set = set(raw_nodes)
                cleaned_set = set(cleaned_nodes)

                good_in_raw = good_set & raw_set
                good_in_cleaned = good_set & cleaned_set
                bad_in_raw = bad_set & raw_set
                bad_in_cleaned = bad_set & cleaned_set

                raw_in_template = raw_set & template_set
                cleaned_in_template = cleaned_set & template_set

                raw_outside_template = raw_set - template_set
                cleaned_outside_template = cleaned_set - template_set

                record["reconstruction_intersections"] = {
                    "raw_nodes_total": int(len(raw_set)),
                    "cleaned_nodes_total": int(len(cleaned_set)),
                    "raw_nodes_in_template": int(len(raw_in_template)),
                    "cleaned_nodes_in_template": int(len(cleaned_in_template)),
                    "raw_nodes_outside_template": int(len(raw_outside_template)),
                    "cleaned_nodes_outside_template": int(len(cleaned_outside_template)),
                    "good_channels_total": int(len(good_set)),
                    "bad_channels_total": int(len(bad_set)),
                    "good_channels_in_raw_nodes": int(len(good_in_raw)),
                    "good_channels_in_cleaned_nodes": int(len(good_in_cleaned)),
                    "bad_channels_in_raw_nodes": int(len(bad_in_raw)),
                    "bad_channels_in_cleaned_nodes": int(len(bad_in_cleaned)),
                    "good_in_raw_fraction": float(len(good_in_raw) / max(1, len(good_set))),
                    "good_in_cleaned_fraction": float(len(good_in_cleaned) / max(1, len(good_set))),
                    "raw_nodes_bad_fraction_over_template": float(len(bad_in_raw) / max(1, len(raw_in_template))),
                    "cleaned_nodes_bad_fraction_over_template": float(len(bad_in_cleaned) / max(1, len(cleaned_in_template))),
                    "good_channel_ids_in_raw_nodes": _sorted_list(good_in_raw),
                    "good_channel_ids_in_cleaned_nodes": _sorted_list(good_in_cleaned),
                    "bad_channel_ids_in_raw_nodes": _sorted_list(bad_in_raw),
                    "bad_channel_ids_in_cleaned_nodes": _sorted_list(bad_in_cleaned),
                }
    except Exception as e:
        record.setdefault("warnings", [])
        record["warnings"].append(f"reconstruction intersections failed: {e}")

    # --- BOTM filter and spike scores ---
    f = None
    offset = 0.0
    scores_spike = None
    noise_model_used = "diag"
    try:
        if noise_model_req in {"cov", "covariance"}:
            if noise_stats_wfs is None:
                raise RuntimeError("cov noise_model requires recording noise windows; none available")
            cov = _estimate_channel_covariance_from_windows(windows_n_t_by_c=noise_stats_wfs, diag_load=1e-3)
            f = _botm_filter_from_template_cov(template_t_by_c=tmpl_use, cov_c_by_c=cov)
            offset = _botm_offset_constant(template_t_by_c=tmpl_use, f_t_by_c=f)
            scores_spike = _scores_for_waveforms_with_offset(waveforms_t_by_c=wfs, f_t_by_c=f, offset=float(offset))
            noise_model_used = "cov"
            record["inputs"].update({"noise_model_used": noise_model_used, "botm_offset": float(offset)})
        else:
            # Diagonal covariance model.
            if noise_stats_wfs is not None:
                # Estimate sigma from recording noise windows (closer to Franke et al.).
                x = np.asarray(noise_stats_wfs).reshape(-1, int(tmpl_use.shape[1]))
                sigma_ch = np.nanstd(x.astype(float, copy=False), axis=0)
                eps = 1e-9
                sigma_ch = np.where(np.isfinite(sigma_ch) & (sigma_ch > eps), sigma_ch, eps)
                record["inputs"].setdefault("sigma_estimation", "recording_spikefree")
            else:
                sigma_ch = _estimate_sigma_per_channel_from_baseline(waveforms=wfs, baseline_frac=float(baseline_frac))
                record["inputs"].setdefault("sigma_estimation", "waveforms_baseline")

            f = _botm_filter_from_template(template_t_by_c=tmpl_use, sigma_ch=sigma_ch)
            offset = _botm_offset_constant(template_t_by_c=tmpl_use, f_t_by_c=f)
            scores_spike = _scores_for_waveforms_with_offset(waveforms_t_by_c=wfs, f_t_by_c=f, offset=float(offset))
            noise_model_used = "diag"
            record["inputs"].update({"noise_model_used": noise_model_used, "botm_offset": float(offset)})

    except Exception as e:
        # Fall back from covariance to diagonal if anything goes wrong.
        record.setdefault("warnings", [])
        record["warnings"].append(f"noise_model {noise_model_req!r} failed; falling back to diag: {e}")
        try:
            sigma_ch = _estimate_sigma_per_channel_from_baseline(waveforms=wfs, baseline_frac=float(baseline_frac))
            f = _botm_filter_from_template(template_t_by_c=tmpl_use, sigma_ch=sigma_ch)
            offset = _botm_offset_constant(template_t_by_c=tmpl_use, f_t_by_c=f)
            scores_spike = _scores_for_waveforms_with_offset(waveforms_t_by_c=wfs, f_t_by_c=f, offset=float(offset))
            noise_model_used = "diag"
            record["inputs"].update({"noise_model_used": noise_model_used, "botm_offset": float(offset)})
        except Exception as e2:
            record["status"] = "error"
            record["error"] = f"score computation failed: {e2}"
            return record

    if f is None or scores_spike is None:
        record["status"] = "error"
        record["error"] = "score computation failed: missing filter/scores"
        return record

    # Noise scores: default to synthetic Gaussian; optionally sample from concat recording.
    mode = str(negatives_mode).lower().strip()

    if mode in {"gaussian", "synthetic", "normal"}:
        try:
            nN = int(max(1, n_noise))
            if str(noise_model_used) == "cov":
                # Gaussian with channel covariance; time samples are drawn i.i.d. across t.
                if noise_stats_wfs is None:
                    raise RuntimeError("cov gaussian negatives require noise_stats windows")
                cov = _estimate_channel_covariance_from_windows(windows_n_t_by_c=noise_stats_wfs, diag_load=1e-3)
                L = int(tmpl_use.shape[0])
                C = int(tmpl_use.shape[1])
                chol = np.linalg.cholesky(np.asarray(cov, dtype=float))
                z = rng.normal(loc=0.0, scale=1.0, size=(int(nN) * int(L), int(C)))
                x = (z @ chol.T).reshape(int(nN), int(L), int(C))
                scores_noise = _scores_for_waveforms_with_offset(waveforms_t_by_c=x, f_t_by_c=f, offset=float(offset))
            else:
                # Diagonal covariance synthetic noise.
                # Use sigma estimated above (recording-based if available, else baseline).
                if noise_stats_wfs is not None and str(record["inputs"].get("sigma_estimation")) == "recording_spikefree":
                    x0 = np.asarray(noise_stats_wfs).reshape(-1, int(tmpl_use.shape[1]))
                    sigma_ch2 = np.nanstd(x0.astype(float, copy=False), axis=0)
                    eps = 1e-9
                    sigma_ch2 = np.where(np.isfinite(sigma_ch2) & (sigma_ch2 > eps), sigma_ch2, eps)
                    sigma_use = sigma_ch2
                else:
                    sigma_use = _estimate_sigma_per_channel_from_baseline(waveforms=wfs, baseline_frac=float(baseline_frac))

                noise = rng.normal(
                    loc=0.0,
                    scale=np.asarray(sigma_use)[None, None, :],
                    size=(nN, int(tmpl_use.shape[0]), int(tmpl_use.shape[1])),
                )
                scores_noise = _scores_for_waveforms_with_offset(
                    waveforms_t_by_c=noise,
                    f_t_by_c=f,
                    offset=float(offset),
                )
        except Exception as e:
            record["status"] = "error"
            record["error"] = f"noise score generation failed: {e}"
            return record

    elif mode in {"recording_random", "random_recording", "recording", "recording_spikefree", "spikefree"}:
        # Best-effort: sample windows from recordings (concat + segments).
        # In spikefree mode, exclude windows overlapping detected spikes (per-source sorting when available).
        # If sampling fails, fall back to Gaussian.
        scores_noise = None
        try:
            if template_channel_ids is None:
                raise RuntimeError("cannot sample recording_random negatives without template channel ids")

            nN = int(max(1, n_noise))
            T = int(tmpl_use.shape[0])
            c_template = int(tmpl_use.shape[1])
            do_spikefree = mode in {"recording_spikefree", "spikefree"}

            noise_wfs, used_noise_sources, neg_warnings = _sample_noise_windows_multi_source(
                analyzers=analyzers,
                template_channel_ids=list(template_channel_ids),
                c_template=int(c_template),
                target_t=int(T),
                waveforms_native_t_by_source=waveforms_native_t_by_source,
                concat_waveforms_native_t=concat_waveforms_native_t,
                rng=rng,
                n_needed=int(nN),
                spikefree=bool(do_spikefree),
            )

            key = "recording_spikefree_noise_sources" if do_spikefree else "recording_random_noise_sources"
            record["inputs"].setdefault(key, used_noise_sources)
            if neg_warnings:
                record.setdefault("warnings", [])
                record["warnings"].extend([f"negatives: {w}" for w in neg_warnings])

            scores_noise = _scores_for_waveforms_with_offset(
                waveforms_t_by_c=noise_wfs,
                f_t_by_c=f,
                offset=float(offset),
            )
        except Exception as e:
            record.setdefault("warnings", [])
            record["warnings"].append(f"recording_random negatives failed; falling back to gaussian: {e}")
            try:
                nN = int(max(1, n_noise))
                # Last-resort diagonal synthetic noise.
                sigma_use = _estimate_sigma_per_channel_from_baseline(waveforms=wfs, baseline_frac=float(baseline_frac))
                noise = rng.normal(
                    loc=0.0,
                    scale=np.asarray(sigma_use)[None, None, :],
                    size=(nN, int(tmpl_use.shape[0]), int(tmpl_use.shape[1])),
                )
                scores_noise = _scores_for_waveforms_with_offset(
                    waveforms_t_by_c=noise,
                    f_t_by_c=f,
                    offset=float(offset),
                )
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
        metrics_update: dict[str, Any] = {
            "n_spike": int(np.asarray(scores_spike).size),
            "n_noise": int(np.asarray(scores_noise).size),
            "score_mean_spike": float(np.nanmean(scores_spike)),
            "score_std_spike": float(np.nanstd(scores_spike)),
            "score_mean_noise": float(np.nanmean(scores_noise)),
            "score_std_noise": float(np.nanstd(scores_noise)),
            "auc": float(auc),
            "dprime": float(dprime),
            # Convenience: explicit naming for “how many waveforms were scored”.
            "n_waveforms_scored": int(np.asarray(scores_spike).size),
        }

        # Add channel-level match summary counts if available.
        try:
            chm = record.get("channel_match") or {}
            n_trials = chm.get("n_trials")
            n_matches_by_ch = chm.get("n_matches_by_channel")
            n_mismatches_by_ch = chm.get("n_mismatches_by_channel")
            tmpl_ch_ids = chm.get("template_channel_ids")
            good_ch_ids = chm.get("good_channel_ids")
            if isinstance(n_trials, int) and isinstance(n_matches_by_ch, list) and isinstance(n_mismatches_by_ch, list):
                c = int(len(n_matches_by_ch))
                total_trials = int(n_trials) * c
                total_false = int(sum(int(x) for x in n_mismatches_by_ch))
                metrics_update.update(
                    {
                        "channel_match_n_trials": int(n_trials),
                        "channel_match_n_channels": int(c),
                        "channel_match_total_trial_channel": int(total_trials),
                        "channel_match_false_matches_total": int(total_false),
                        "channel_match_false_match_fraction": float(total_false / max(1, total_trials)),
                    }
                )
            if isinstance(tmpl_ch_ids, list) and isinstance(good_ch_ids, list):
                metrics_update.update(
                    {
                        "channel_match_n_good_channels": int(len(good_ch_ids)),
                        "channel_match_n_bad_channels": int(max(0, len(tmpl_ch_ids) - len(good_ch_ids))),
                    }
                )
        except Exception:
            pass

        # Add reconstruction intersection summaries if available.
        try:
            ri = record.get("reconstruction_intersections") or {}
            if isinstance(ri, dict):
                for k in [
                    "raw_nodes_total",
                    "cleaned_nodes_total",
                    "good_channels_in_raw_nodes",
                    "good_channels_in_cleaned_nodes",
                    "bad_channels_in_raw_nodes",
                    "bad_channels_in_cleaned_nodes",
                    "raw_nodes_outside_template",
                    "cleaned_nodes_outside_template",
                ]:
                    if k in ri:
                        metrics_update[f"recon_{k}"] = ri[k]
        except Exception:
            pass

        record["metrics"].update(metrics_update)
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
            cap_waveforms_per_source_to_n_spike=bool(inputs.cap_waveforms_per_source_to_n_spike),
            noise_model=str(inputs.noise_model),
            baseline_frac=float(inputs.baseline_frac),
            seed=int(inputs.seed),
            waveforms_source=str(inputs.waveforms_source),
            negatives_mode=str(inputs.negatives_mode),
            channel_match_fraction_threshold=float(inputs.channel_match_fraction_threshold),
            channel_match_method=str(inputs.channel_match_method),
            channel_match_prior_signal=float(inputs.channel_match_prior_signal),
            channel_match_noise_std_level=float(inputs.channel_match_noise_std_level),
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
            "cap_waveforms_per_source_to_n_spike": bool(inputs.cap_waveforms_per_source_to_n_spike),
            "noise_model": str(inputs.noise_model),
            "waveforms_source": str(inputs.waveforms_source),
            "negatives_mode": str(inputs.negatives_mode),
            "baseline_frac": float(inputs.baseline_frac),
            "seed": int(inputs.seed),
            "channel_match_fraction_threshold": float(inputs.channel_match_fraction_threshold),
            "channel_match_method": str(inputs.channel_match_method),
            "channel_match_prior_signal": float(inputs.channel_match_prior_signal),
            "channel_match_noise_std_level": float(inputs.channel_match_noise_std_level),
        },
        "units": results,
    }

    _write_json(out_dir / "summary.json", summary)
    return summary
