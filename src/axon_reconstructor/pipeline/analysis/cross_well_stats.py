from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

def _annotate_rows_with_div(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate rows with `recording_date`, `div_index`, and `div_label`.

    DIV is computed relative to the earliest recording date present in rows.
    """

    out: list[dict[str, Any]] = []

    parsed_dates: list[datetime] = []
    for r in rows:
        explicit_div = r.get("div")
        if explicit_div is not None:
            continue
        d_raw = r.get("recording_date")
        if d_raw is None:
            continue
        try:
            parsed_dates.append(datetime.strptime(str(d_raw), "%Y-%m-%d"))
        except Exception:
            continue

    min_date = min(parsed_dates) if parsed_dates else None

    for r in rows:
        rr = dict(r)
        explicit_div = rr.get("div")
        if explicit_div is not None:
            try:
                div_value = int(explicit_div)
                rr["div"] = div_value
                rr["div_index"] = div_value
                rr["div_label"] = f"DIV{div_value}"
                out.append(rr)
                continue
            except Exception:
                pass

        d_raw = rr.get("recording_date")
        if d_raw is None or min_date is None:
            rr["div"] = None
            rr["div_index"] = None
            rr["div_label"] = None
            out.append(rr)
            continue

        try:
            d = datetime.strptime(str(d_raw), "%Y-%m-%d")
            div_index = int((d - min_date).days)
            rr["div"] = div_index
            rr["div_index"] = div_index
            rr["div_label"] = f"DIV{div_index}"
        except Exception:
            rr["div"] = None
            rr["div_index"] = None
            rr["div_label"] = None
        out.append(rr)

    return out


def _compute_div_density_group_order(*, rows: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    seen: set[tuple[int, int, str]] = set()
    order: list[tuple[int, int, str]] = []
    for r in rows:
        div = r.get("div")
        density = r.get("plating_density_nbp")
        cond = r.get("condition")
        if div is None or density is None or cond is None:
            continue
        try:
            key = (int(div), int(density), str(cond))
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            order.append(key)
    order.sort(key=lambda x: (x[0], x[1], x[2]))
    return order


def _values_by_div_density(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    group_order: list[tuple[int, int, str]],
    where: dict[str, Any] | None = None,
    min_value: float | None = None,
) -> tuple[list[np.ndarray], list[str], list[str], list[str], list[str]]:
    groups: list[np.ndarray] = []
    group_labels: list[str] = []
    density_tick_labels: list[str] = []
    div_labels: list[str] = []
    conditions_for_groups: list[str] = []

    for div, density, cond in group_order:
        vals: list[float] = []
        for r in rows:
            if where is not None:
                ok = True
                for k, v in where.items():
                    if r.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue

            try:
                if int(r.get("div")) != int(div):
                    continue
                if int(r.get("plating_density_nbp")) != int(density):
                    continue
            except Exception:
                continue
            if str(r.get("condition")) != str(cond):
                continue

            v = r.get(metric)
            if v is None:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            if min_value is not None and vv < float(min_value):
                continue
            vals.append(vv)

        groups.append(np.asarray(vals, dtype=float))
        div_label = f"DIV{int(div)}"
        group_labels.append(f"{div_label} | {cond}")
        density_tick_labels.append(str(cond))
        div_labels.append(div_label)
        conditions_for_groups.append(str(cond))

    return groups, group_labels, div_labels, density_tick_labels, conditions_for_groups


def _rows_with_positive_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv > 0:
            out.append(r)
    return out


def _p_to_stars(p: float) -> str | None:
    if p < 0.001:
        return "***"
    if p < 0.005:
        return "**"
    if p < 0.05:
        return "*"
    return None


def _pairwise_mannwhitneyu(groups: list[np.ndarray], group_labels: list[str]) -> list[dict[str, Any]]:
    from scipy.stats import mannwhitneyu

    out: list[dict[str, Any]] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = groups[i]
            b = groups[j]
            if a.size == 0 or b.size == 0:
                continue
            res = mannwhitneyu(a, b, alternative="two-sided")
            p = float(res.pvalue)
            out.append(
                {
                    "group_a": group_labels[i],
                    "group_b": group_labels[j],
                    "n_a": int(a.size),
                    "n_b": int(b.size),
                    "u": float(res.statistic),
                    "p": p,
                    "stars": _p_to_stars(p) or "",
                }
            )
    return out


def _pairwise_mannwhitneyu_within_blocks(
    groups: list[np.ndarray],
    group_labels: list[str],
    *,
    block_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    from scipy.stats import mannwhitneyu

    out: list[dict[str, Any]] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if block_labels is not None:
                if i >= len(block_labels) or j >= len(block_labels):
                    continue
                if str(block_labels[i]) != str(block_labels[j]):
                    continue

            a = groups[i]
            b = groups[j]
            if a.size == 0 or b.size == 0:
                continue
            res = mannwhitneyu(a, b, alternative="two-sided")
            p = float(res.pvalue)
            out.append(
                {
                    "group_a": group_labels[i],
                    "group_b": group_labels[j],
                    "n_a": int(a.size),
                    "n_b": int(b.size),
                    "u": float(res.statistic),
                    "p": p,
                    "stars": _p_to_stars(p) or "",
                }
            )
    return out


def _iqr_outlier_mask(values: np.ndarray, *, k: float = 1.5) -> np.ndarray:
    """Return a boolean mask marking IQR outliers.

    Outliers are values < Q1 - k*IQR or > Q3 + k*IQR.
    For small samples (<4) or constant arrays, returns all-False.
    """

    v = np.asarray(values, dtype=float)
    if v.size < 4:
        return np.zeros(v.shape, dtype=bool)
    q1 = float(np.percentile(v, 25))
    q3 = float(np.percentile(v, 75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return np.zeros(v.shape, dtype=bool)
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (v < lo) | (v > hi)


def _prepare_groups_for_tests(
    groups: list[np.ndarray],
    labels: list[str],
    *,
    exclude_outliers: bool,
    outlier_k: float,
) -> tuple[list[np.ndarray], dict[str, dict[str, int]]]:
    """Optionally exclude IQR outliers per group; return groups used + stats per label."""

    used: list[np.ndarray] = []
    stats: dict[str, dict[str, int]] = {}
    for g, lab in zip(groups, labels, strict=True):
        g = np.asarray(g, dtype=float)
        mask = _iqr_outlier_mask(g, k=outlier_k) if exclude_outliers else np.zeros(g.shape, dtype=bool)
        g_used = g[~mask]
        used.append(g_used)
        stats[lab] = {
            "n_total": int(g.size),
            "n_outliers": int(mask.sum()),
            "n_used": int(g_used.size),
        }
    return used, stats


def _exclude_group_outliers(
    groups: list[np.ndarray],
    *,
    outlier_k: float,
) -> tuple[list[np.ndarray], dict[int, dict[str, int]]]:
    """Return outlier-filtered groups and per-group exclusion stats."""

    filtered: list[np.ndarray] = []
    stats: dict[int, dict[str, int]] = {}
    for idx, g in enumerate(groups):
        arr = np.asarray(g, dtype=float)
        mask = _iqr_outlier_mask(arr, k=outlier_k)
        kept = arr[~mask]
        filtered.append(kept)
        stats[idx] = {
            "n_total": int(arr.size),
            "n_outliers": int(mask.sum()),
            "n_kept": int(kept.size),
        }
    return filtered, stats


def _build_well_count_rows(
    *,
    wells_summary: list[dict[str, Any]],
    all_units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one row per (dataset_key, well_id) for detected/reconstructed counts."""

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in wells_summary:
        dk = str(r.get("dataset_key", ""))
        wid = str(r.get("well_id", ""))
        if not dk or not wid:
            continue
        key = (dk, wid)
        rows_by_key[key] = {
            "dataset_key": dk,
            "well_id": wid,
            "condition": r.get("condition"),
            "plating_density_nbp": r.get("plating_density_nbp"),
            "div": r.get("div"),
            "n_units_detected": r.get("n_detected_units"),
            "n_units_reconstructed": int(r.get("n_reconstructed_units") or 0),
        }

    out = list(rows_by_key.values())
    out.sort(
        key=lambda r: (
            int(r.get("div")) if r.get("div") is not None else 10**9,
            str(r.get("well_id", "")),
        )
    )
    return out
