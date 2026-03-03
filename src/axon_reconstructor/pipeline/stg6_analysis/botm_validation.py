from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class BotmValidationInputs:
    """Raw-recording BOTM validation (analysis-stage).

    Implements the procedure we discussed:
      - templates: merged template from templates stage
      - candidate events: unit spike times from the *concat* sorting
      - noise: spike-free windows from the raw recording (all units excluded)
      - evaluation: for each template channel, compute % of candidate events that
        match that channel's template under a BOTM (Bayes) decision rule

    This intentionally does NOT use SpikeInterface waveforms extractors/analyzers,
    does not compute AUC/d', and does not keep legacy negative-sampling modes.
    """

    well_out_dir: Path
    h5_path: Path
    stream_id: str

    unit_ids: Optional[list[Any]] = None

    # Spikes (candidate events)
    n_events: int = 200

    # Noise windows (for sigma estimation)
    n_noise_windows: int = 2000

    # RNG seed for subsampling events and sampling noise windows
    seed: int = 0

    # BOTM/Bayes prior P(signal) at a candidate event time.
    prior_signal: float = 0.5

    # Good-channel cutoff from the 2023 description: strictly > 70%
    match_fraction_threshold: float = 0.70

    # Spike sorting loader needs a sorter name for some SpikeInterface versions.
    sorter: str = "kilosort4"

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


def _load_merged_template_for_unit(*, templates_out_dir: Path, uid: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np  # type: ignore[import-not-found]

    unit_dir = Path(templates_out_dir) / "templates" / "merged" / f"unit_{uid}"
    if not unit_dir.exists():
        raise FileNotFoundError(f"Missing merged template unit dir: {unit_dir}")

    tmpl_path = unit_dir / "merged_contributing_template.npy"
    meta_path = unit_dir / "merged_contributing_template_meta.json"
    if (not tmpl_path.exists()) or (not meta_path.exists()):
        raise FileNotFoundError(f"Missing merged template inputs for unit={uid}: {tmpl_path.name}, {meta_path.name}")

    tmpl = np.load(tmpl_path)
    if getattr(tmpl, "ndim", None) != 2:
        raise ValueError(f"Unexpected template shape for unit={uid}: {getattr(tmpl, 'shape', None)}")

    meta = _read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"Unexpected template meta payload type for unit={uid}: {type(meta)}")

    return tmpl.astype(float, copy=False), meta


def _load_concat_epochs(*, preprocess_out_dir: Path, stream_id: str) -> list[dict[str, Any]]:
    path = Path(preprocess_out_dir) / f"concatenation_stitch_epochs_{stream_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing concat epoch markers: {path}")
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected concat epoch payload type: {type(payload)}")

    # Required keys: segment_index, rec_name, start_sample, end_sample
    epochs: list[dict[str, Any]] = []
    for e in payload:
        if not isinstance(e, dict):
            continue
        for k in ("segment_index", "rec_name", "start_sample", "end_sample"):
            if k not in e:
                raise ValueError(f"Epoch marker missing key={k}: {e}")
        epochs.append(dict(e))
    if not epochs:
        raise ValueError("No concat epochs found")
    epochs.sort(key=lambda d: int(d["start_sample"]))
    return epochs


def _find_epoch_for_sample(*, epochs: list[dict[str, Any]], sample_index: int) -> dict[str, Any]:
    s = int(sample_index)
    lo = 0
    hi = len(epochs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        e = epochs[mid]
        a = int(e["start_sample"])
        b = int(e["end_sample"])
        if s < a:
            hi = mid - 1
        elif s >= b:
            lo = mid + 1
        else:
            return e
    raise ValueError(f"Sample index {s} not within any concat epoch")


def _load_sorting(*, sorter_output_dir: Path, sorter: str) -> Any:
    import spikeinterface.full as si  # type: ignore[import-not-found]

    if hasattr(si, "read_sorter_folder"):
        try:
            return si.read_sorter_folder(sorter_output_dir, sorter_name=str(sorter))
        except TypeError:
            return si.read_sorter_folder(sorter_output_dir, str(sorter))

    try:
        return si.load_extractor(sorter_output_dir)
    except Exception as e:
        raise RuntimeError(
            f"Could not load sorting from sorter_output_dir={sorter_output_dir} (sorter={sorter}): {e}"
        ) from e


def _load_branch_channels(*, branches_json: Path) -> set[int]:
    branches_json = Path(branches_json)
    if not branches_json.exists():
        raise FileNotFoundError(f"Missing branches JSON: {branches_json}")
    payload = _read_json(branches_json)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected branches JSON payload type: {type(payload)}")
    branches = payload.get("branches")
    if not isinstance(branches, list):
        raise ValueError("Branches JSON missing 'branches' list")
    out: set[int] = set()
    for b in branches:
        if not isinstance(b, dict):
            continue
        ch = b.get("channels")
        if not isinstance(ch, list):
            continue
        for v in ch:
            try:
                out.add(int(v))
            except Exception:
                continue
    return out


def _resolve_sorter_output_dir(*, well_out_dir: Path) -> Path:
    p = Path(well_out_dir) / "stg2_spikesorting_outputs" / "sorter_output"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing sorter output dir under well_out_dir: {p} (and no legacy)")


def _load_preprocessed_concat_fs_hz(*, well_out_dir: Path) -> float:
    import spikeinterface.full as si  # type: ignore[import-not-found]

    rec_dir = Path(well_out_dir) / "stg1_preprocess_outputs" / "preprocessed_recording"
    if not rec_dir.exists():
        raise FileNotFoundError(f"Missing preprocessed recording dir: {rec_dir}")

    try:
        rec = si.load(rec_dir)
    except Exception:
        rec = si.load_extractor(rec_dir)

    fs = float(rec.get_sampling_frequency())
    if fs <= 0:
        raise RuntimeError("Could not determine sampling frequency from preprocessed recording")
    return fs


def _resample_time_axis_to_length(*, x_t_by_c: Any, target_t: int) -> Any:
    """Sinc-like resampling via scipy.signal.resample_poly.

    Requires SciPy; raises if unavailable.
    """

    import numpy as np  # type: ignore[import-not-found]

    x = np.asarray(x_t_by_c, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D (T,C), got shape={x.shape}")
    t0 = int(x.shape[0])
    t1 = int(target_t)
    if t0 == t1:
        return x
    if t0 <= 1 or t1 <= 1:
        raise ValueError(f"Invalid resample lengths: t0={t0} t1={t1}")

    try:
        from scipy.signal import resample_poly  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(f"SciPy required for resampling but unavailable: {e}") from e

    # Use rational approximation by simple integer up/down based on lengths.
    # We deliberately keep this simple: resample_poly handles anti-alias filtering.
    y = resample_poly(x, up=t1, down=t0, axis=0)
    if int(y.shape[0]) != t1:
        # resample_poly can off-by-1 depending on filter padding; crop/pad deterministically.
        if int(y.shape[0]) > t1:
            y = y[:t1, :]
        else:
            pad = np.zeros((t1 - int(y.shape[0]), int(y.shape[1])), dtype=float)
            y = np.concatenate([y, pad], axis=0)
    return np.asarray(y, dtype=float)


def _subsample_indices(*, n_have: int, n_keep: int, rng: Any) -> list[int]:
    import numpy as np  # type: ignore[import-not-found]

    n_have_i = int(n_have)
    n_keep_i = int(n_keep)
    if n_keep_i >= n_have_i:
        return list(range(n_have_i))
    idx = rng.choice(n_have_i, size=n_keep_i, replace=False)
    return [int(i) for i in np.asarray(idx, dtype=int).tolist()]


def _any_spike_in_interval(*, spikes_sorted: Any, start: int, end: int) -> bool:
    import numpy as np  # type: ignore[import-not-found]

    s = int(start)
    e = int(end)
    if e <= s:
        return True
    sp = np.asarray(spikes_sorted, dtype=np.int64)
    if sp.size == 0:
        return False
    i = int(np.searchsorted(sp, s, side="left"))
    return bool(i < int(sp.size) and int(sp[i]) < e)


def compute_botm_validation_for_unit(*, inputs: BotmValidationInputs, uid: Any, logger: logging.Logger) -> dict[str, Any]:
    """Compute per-channel match fractions for one unit using raw snippets."""

    import numpy as np  # type: ignore[import-not-found]

    rec: dict[str, Any] = {
        "unit_id": _jsonable(uid),
        "status": "ok",
        "error": None,
        "params": {},
        "channel_match": {},
    }

    # Validate prior.
    p = float(inputs.prior_signal)
    if not (0.0 < p < 1.0):
        raise ValueError(f"prior_signal must be in (0,1), got {p}")
    thr = float(np.log(1.0 - p))

    # Load template.
    templates_out_dir = Path(inputs.well_out_dir) / "stg4_templates_outputs"
    tmpl_t_by_c, meta = _load_merged_template_for_unit(templates_out_dir=templates_out_dir, uid=uid)
    t_template = int(tmpl_t_by_c.shape[0])
    c_template = int(tmpl_t_by_c.shape[1])

    ch_ids = meta.get("channel_ids")
    if not (isinstance(ch_ids, list) and len(ch_ids) == c_template):
        raise ValueError("Template meta missing channel_ids or length mismatch")

    ms_before = meta.get("ms_before")
    ms_after = meta.get("ms_after")
    if ms_before is None or ms_after is None:
        raise ValueError("Template meta missing ms_before/ms_after")
    ms_before_f = float(ms_before)
    ms_after_f = float(ms_after)

    fs_template = float(meta.get("sampling_frequency_hz"))
    fs_native = float(meta.get("native_sampling_frequency_hz"))
    if fs_template <= 0 or fs_native <= 0:
        raise ValueError("Template meta missing/invalid sampling_frequency_hz/native_sampling_frequency_hz")

    # Recording time base (concat) for spike times / epochs.
    fs_concat = _load_preprocessed_concat_fs_hz(well_out_dir=Path(inputs.well_out_dir))

    # We require the concat spike times to be in the concat/preprocessed time base.
    # If preprocessing temporally resampled, fs_concat reflects that.
    if abs(fs_concat - fs_native) > 1e-3:
        # No fallback: enforce consistency. If templates were built at a different native fs,
        # indexing will be wrong.
        raise RuntimeError(
            "Native template fs does not match concat/preprocessed fs; cannot align raw snippets. "
            f"fs_native={fs_native} fs_concat={fs_concat}"
        )

    pre_samples = int(round(ms_before_f * fs_native / 1000.0))
    post_samples = int(round(ms_after_f * fs_native / 1000.0))
    win_native = int(pre_samples + post_samples)
    if win_native <= 2:
        raise ValueError(f"Invalid native window length: {win_native} samples")

    # Load concat epochs and sorting.
    preprocess_out_dir = Path(inputs.well_out_dir) / "stg1_preprocess_outputs"
    epochs = _load_concat_epochs(preprocess_out_dir=preprocess_out_dir, stream_id=str(inputs.stream_id))

    sorter_output_dir = _resolve_sorter_output_dir(well_out_dir=Path(inputs.well_out_dir))
    sorting = _load_sorting(sorter_output_dir=sorter_output_dir, sorter=str(inputs.sorter))

    # Candidate spikes: unit spike train in concat time base.
    st_unit = sorting.get_unit_spike_train(unit_id=uid, segment_index=0)
    st_unit = np.asarray(st_unit, dtype=np.int64)
    if st_unit.size == 0:
        raise RuntimeError("Unit has zero spikes in concat sorting")

    rng = np.random.default_rng(int(inputs.seed))
    keep_idx = _subsample_indices(n_have=int(st_unit.size), n_keep=int(inputs.n_events), rng=rng)
    st_events = np.asarray(st_unit[keep_idx], dtype=np.int64)
    st_events.sort()
    if int(st_events.size) < int(inputs.n_events):
        # No fallback: enforce requested count for reproducibility.
        raise RuntimeError(f"Not enough unit spikes for n_events={int(inputs.n_events)} (have {int(st_unit.size)})")

    # Spike-free constraint: union spikes across ALL units.
    all_spikes: list[np.ndarray] = []
    for u2 in sorting.get_unit_ids():
        st = sorting.get_unit_spike_train(unit_id=u2, segment_index=0)
        st = np.asarray(st, dtype=np.int64)
        if st.size:
            all_spikes.append(st)
    spikes_all = np.unique(np.concatenate(all_spikes, axis=0)).astype(np.int64, copy=False) if all_spikes else np.asarray([], dtype=np.int64)

    # Per-channel accumulators.
    # We compute sigma (noise std) from spike-free windows and then compute BOTM discriminant:
    #   D = (x·xi)/sigma^2 - 0.5*(xi·xi)/sigma^2 + ln(p)
    # match iff D >= ln(1-p)
    ch_ids_norm: list[int] = []
    for v in ch_ids:
        try:
            ch_ids_norm.append(int(v))
        except Exception as e:
            raise ValueError(f"Template channel id not int-like: {v} ({e})") from e

    ch_to_ci = {int(ch): int(i) for i, ch in enumerate(ch_ids_norm)}

    tmpl = np.asarray(tmpl_t_by_c, dtype=float)
    tmpl_energy = np.sum(tmpl * tmpl, axis=0).astype(float)  # (C,)

    # Noise stats running sums on the *resampled-to-template* time base.
    sum_x = np.zeros((c_template,), dtype=float)
    sum_x2 = np.zeros((c_template,), dtype=float)
    n_x = np.zeros((c_template,), dtype=np.int64)

    # Per-channel match counts.
    n_trials = np.zeros((c_template,), dtype=np.int64)
    n_match = np.zeros((c_template,), dtype=np.int64)

    # Lazy-load per-segment recordings (full segment channels), resampled to fs_native if needed.
    # Note: raw segment recordings are loaded from the H5 and preprocessed like MEA_Analysis.
    from ..waveforms.utils import _load_raw_segment_recording_segment_channels  # type: ignore

    seg_recordings: dict[int, Any] = {}
    seg_channel_sets: dict[int, set[Any]] = {}

    def _get_seg_rec(seg_index: int, rec_name: str):
        si = int(seg_index)
        if si in seg_recordings:
            return seg_recordings[si]
        r = _load_raw_segment_recording_segment_channels(
            h5_path=Path(inputs.h5_path),
            stream_id=str(inputs.stream_id),
            rec_name=str(rec_name),
            preprocess_like_mea_analysis=True,
            target_sampling_frequency_hz=float(fs_native),
        )
        seg_recordings[si] = r
        try:
            seg_channel_sets[si] = set(r.get_channel_ids())
        except Exception:
            seg_channel_sets[si] = set()
        return r

    # Helper: extract and resample a snippet (T_native,C_present) -> (T_template,C_present).
    def _extract_resampled_snippet(*, seg_index: int, rec_name: str, center_sample_local: int, present_ch_ids: list[Any]):
        rec_seg = _get_seg_rec(seg_index=seg_index, rec_name=rec_name)
        t0 = int(center_sample_local) - int(pre_samples)
        t1 = int(center_sample_local) + int(post_samples)
        if t0 < 0 or t1 > int(rec_seg.get_num_samples()):
            raise RuntimeError("snippet window outside segment bounds")
        x = rec_seg.get_traces(start_frame=int(t0), end_frame=int(t1), channel_ids=list(present_ch_ids))
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or int(x.shape[0]) != int(win_native):
            raise RuntimeError(f"unexpected snippet shape: {x.shape}")
        return _resample_time_axis_to_length(x_t_by_c=x, target_t=int(t_template))

    # Sample spike-free noise windows.
    # We sample uniformly over epochs; for each sample we reject if any spike overlaps the window.
    epochs_arr = epochs
    epoch_weights = np.asarray([int(e["end_sample"]) - int(e["start_sample"]) for e in epochs_arr], dtype=float)
    if not np.all(epoch_weights > 0):
        raise RuntimeError("invalid concat epoch lengths")
    epoch_weights = epoch_weights / float(np.sum(epoch_weights))

    max_tries = int(max(10_000, 20 * int(inputs.n_noise_windows)))
    n_accepted = 0
    tries = 0
    while n_accepted < int(inputs.n_noise_windows) and tries < max_tries:
        tries += 1
        ep_i = int(rng.choice(len(epochs_arr), p=epoch_weights))
        ep = epochs_arr[ep_i]
        seg_index = int(ep["segment_index"])
        rec_name = str(ep["rec_name"])
        start_c = int(ep["start_sample"])
        end_c = int(ep["end_sample"])

        # Pick a concat sample where the window stays within the epoch.
        center_c_min = start_c + int(pre_samples)
        center_c_max = end_c - int(post_samples)
        if center_c_max <= center_c_min:
            continue
        center_c = int(rng.integers(center_c_min, center_c_max))
        if _any_spike_in_interval(spikes_sorted=spikes_all, start=center_c - int(pre_samples), end=center_c + int(post_samples)):
            continue

        # Map to segment-local.
        center_local = int(center_c - int(start_c))

        # Determine which template channels exist in this segment.
        rec_seg = _get_seg_rec(seg_index=seg_index, rec_name=rec_name)
        ch_set = seg_channel_sets.get(seg_index)
        if ch_set is None:
            ch_set = set(rec_seg.get_channel_ids())
            seg_channel_sets[seg_index] = ch_set
        present = [ch for ch in ch_ids_norm if ch in ch_set]
        if not present:
            continue

        y = _extract_resampled_snippet(
            seg_index=seg_index,
            rec_name=rec_name,
            center_sample_local=center_local,
            present_ch_ids=present,
        )
        # Update running noise stats for present channels.
        # y is (T_template, C_present) in the same order as `present`.
        y = np.asarray(y, dtype=float)
        for j, ch in enumerate(present):
            ci = ch_to_ci.get(int(ch))
            if ci is None:
                continue
            v = y[:, j]
            sum_x[ci] += float(np.sum(v))
            sum_x2[ci] += float(np.sum(v * v))
            n_x[ci] += int(v.size)

        n_accepted += 1

    if n_accepted < int(inputs.n_noise_windows):
        raise RuntimeError(
            f"Failed to sample required spike-free noise windows: needed={int(inputs.n_noise_windows)} got={n_accepted} tries={tries}"
        )

    # Compute sigma^2 per channel.
    eps = 1e-12
    sigma2 = np.zeros((c_template,), dtype=float)
    mu = np.zeros((c_template,), dtype=float)
    for ci in range(c_template):
        if int(n_x[ci]) <= 1:
            raise RuntimeError(f"Insufficient noise samples for channel index {ci}")
        mean = float(sum_x[ci]) / float(n_x[ci])
        ex2 = float(sum_x2[ci]) / float(n_x[ci])
        var = float(max(ex2 - mean * mean, 0.0))
        mu[ci] = mean
        sigma2[ci] = var if var > eps else eps

    # Evaluate candidate events.
    for t_c in st_events.tolist():
        t_c_i = int(t_c)
        ep = _find_epoch_for_sample(epochs=epochs_arr, sample_index=t_c_i)
        seg_index = int(ep["segment_index"])
        rec_name = str(ep["rec_name"])
        start_c = int(ep["start_sample"])
        center_local = int(t_c_i - start_c)

        rec_seg = _get_seg_rec(seg_index=seg_index, rec_name=rec_name)
        ch_set = seg_channel_sets.get(seg_index)
        if ch_set is None:
            ch_set = set(rec_seg.get_channel_ids())
            seg_channel_sets[seg_index] = ch_set
        present = [ch for ch in ch_ids_norm if ch in ch_set]
        if not present:
            continue

        y = _extract_resampled_snippet(
            seg_index=seg_index,
            rec_name=rec_name,
            center_sample_local=center_local,
            present_ch_ids=present,
        )
        y = np.asarray(y, dtype=float)

        for j, ch in enumerate(present):
            ci = ch_to_ci.get(int(ch))
            if ci is None:
                continue
            # BOTM assumes zero-mean noise; we center snippets by the spike-free mean.
            x = y[:, j] - float(mu[ci])
            xi = tmpl[:, ci]
            s = float(np.dot(x, xi))
            D = (s / float(sigma2[ci])) - 0.5 * float(tmpl_energy[ci] / float(sigma2[ci])) + float(np.log(p))
            n_trials[ci] += 1
            if D >= thr:
                n_match[ci] += 1

    if int(np.max(n_trials)) == 0:
        raise RuntimeError("No candidate events overlapped any template channels")

    frac = np.zeros((c_template,), dtype=float)
    for ci in range(c_template):
        if int(n_trials[ci]) > 0:
            frac[ci] = float(n_match[ci]) / float(n_trials[ci])
        else:
            frac[ci] = float("nan")

    good_mask = (frac > float(inputs.match_fraction_threshold)) & np.isfinite(frac)
    good_channel_ids = [ch_ids_norm[i] for i in np.where(good_mask)[0].tolist()]

    rec["params"] = {
        "n_events": int(inputs.n_events),
        "n_noise_windows": int(inputs.n_noise_windows),
        "seed": int(inputs.seed),
        "prior_signal": float(p),
        "decision_threshold": float(thr),
        "match_fraction_threshold": float(inputs.match_fraction_threshold),
        "sorter": str(inputs.sorter),
        "template_sampling_frequency_hz": float(fs_template),
        "native_sampling_frequency_hz": float(fs_native),
        "ms_before": float(ms_before_f),
        "ms_after": float(ms_after_f),
        "template_shape": [int(t_template), int(c_template)],
    }

    rec["channel_match"] = {
        "template_channel_ids": ch_ids_norm,
        "n_trials_by_channel": [int(x) for x in n_trials.tolist()],
        "n_matches_by_channel": [int(x) for x in n_match.tolist()],
        "match_fraction_by_channel": [float(x) if np.isfinite(x) else None for x in frac.tolist()],
        "noise_mean_by_channel": [float(x) for x in mu.tolist()],
        "sigma2_by_channel": [float(x) for x in sigma2.tolist()],
        "good_channel_ids": good_channel_ids,
        "n_good_channels": int(len(good_channel_ids)),
    }

    # Strict overlap with reconstruction node channels.
    recon_unit_dir = Path(inputs.well_out_dir) / "stg5_reconstruction_outputs" / "by_unit" / f"unit_{uid}"
    raw_json = recon_unit_dir / "branches_raw.json"
    clean_json = recon_unit_dir / "branches.json"
    raw_nodes = _load_branch_channels(branches_json=raw_json)
    clean_nodes = _load_branch_channels(branches_json=clean_json)

    good_set = set(int(x) for x in good_channel_ids)
    good_in_raw = sorted(good_set.intersection(raw_nodes))
    good_in_clean = sorted(good_set.intersection(clean_nodes))

    n_good = int(len(good_set))
    n_raw = int(len(raw_nodes))
    n_clean = int(len(clean_nodes))
    n_good_in_raw = int(len(good_in_raw))
    n_good_in_clean = int(len(good_in_clean))

    rec["reconstruction_overlap"] = {
        "branches_raw_json": str(raw_json),
        "branches_json": str(clean_json),
        "raw_node_channel_ids": sorted(raw_nodes),
        "clean_node_channel_ids": sorted(clean_nodes),
        "good_channel_ids": sorted(good_set),
        "good_in_raw_node_channel_ids": good_in_raw,
        "good_in_clean_node_channel_ids": good_in_clean,
        "n_raw_nodes": n_raw,
        "n_clean_nodes": n_clean,
        "n_good_channels": n_good,
        "n_good_in_raw_nodes": n_good_in_raw,
        "n_good_in_clean_nodes": n_good_in_clean,
        "frac_good_in_raw_nodes": (float(n_good_in_raw) / float(n_good) if n_good > 0 else None),
        "frac_good_in_clean_nodes": (float(n_good_in_clean) / float(n_good) if n_good > 0 else None),
        "frac_raw_nodes_good": (float(n_good_in_raw) / float(n_raw) if n_raw > 0 else None),
        "frac_clean_nodes_good": (float(n_good_in_clean) / float(n_clean) if n_clean > 0 else None),
    }

    logger.info(
        "BOTM raw validation unit=%s: good_channels=%d/%d",
        str(uid),
        int(len(good_channel_ids)),
        int(c_template),
    )

    return rec


def write_botm_validation_outputs(*, inputs: BotmValidationInputs, logger: Optional[logging.Logger] = None) -> dict[str, Any]:
    if logger is None:
        logger = logging.getLogger("axon_reconstructor.botm_validation")

    out_dir = Path(inputs.out_dir) if inputs.out_dir is not None else Path(inputs.well_out_dir) / "stg6_analysis_outputs" / "botm_validation"
    by_unit_dir = out_dir / "by_unit"

    if inputs.force_restart and out_dir.exists():
        for p in by_unit_dir.glob("unit_*_botm_metrics.json"):
            p.unlink(missing_ok=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    by_unit_dir.mkdir(parents=True, exist_ok=True)

    # Unit discovery (templates merged dir).
    if inputs.unit_ids is not None:
        unit_ids = list(inputs.unit_ids)
    else:
        merged_units_dir = Path(inputs.well_out_dir) / "stg4_templates_outputs" / "templates" / "merged"
        if not merged_units_dir.exists():
            raise FileNotFoundError(f"Missing merged templates dir: {merged_units_dir}")
        unit_ids = []
        for p in sorted(merged_units_dir.glob("unit_*") if merged_units_dir.exists() else []):
            if not p.is_dir():
                continue
            try:
                unit_ids.append(int(p.name.split("unit_", 1)[1]))
            except Exception:
                unit_ids.append(p.name.split("unit_", 1)[1])

    results: list[dict[str, Any]] = []
    for uid in unit_ids:
        logger.info("BOTM raw validation: unit=%s", str(uid))
        rec = compute_botm_validation_for_unit(inputs=inputs, uid=uid, logger=logger)
        out_json = by_unit_dir / f"unit_{uid}_botm_metrics.json"
        _write_json(out_json, rec)
        rec2 = dict(rec)
        rec2.setdefault("artifacts", {})
        rec2["artifacts"].update({"metrics_json": str(out_json)})
        results.append(rec2)

    summary = {
        "well_out_dir": str(inputs.well_out_dir),
        "h5_path": str(inputs.h5_path),
        "stream_id": str(inputs.stream_id),
        "out_dir": str(out_dir),
        "by_unit_dir": str(by_unit_dir),
        "n_units": int(len(unit_ids)),
        "params": {
            "n_events": int(inputs.n_events),
            "n_noise_windows": int(inputs.n_noise_windows),
            "seed": int(inputs.seed),
            "prior_signal": float(inputs.prior_signal),
            "match_fraction_threshold": float(inputs.match_fraction_threshold),
            "sorter": str(inputs.sorter),
        },
        "units": results,
    }

    _write_json(out_dir / "summary.json", summary)
    return summary
