from __future__ import annotations

import numpy as np  # type: ignore[import-not-found]


def _round_up_step(x: float, step: float) -> float:
    if not np.isfinite(x):
        return float(max(1.0, step))
    s = max(1e-12, float(step))
    return float(s * np.ceil(float(x) / s))


def _round_up_nice(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(float(x)))
    base = float(x) / (10.0**exp)
    if base <= 1.0:
        nice = 1.0
    elif base <= 2.0:
        nice = 2.0
    elif base <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0**exp))


def compute_value_limits(
    *,
    values: np.ndarray,
    scale: str,
    percentile_low: float,
    percentile_high_linear: float,
    percentile_high_log: float,
    force_low_value: float | None = None,
    force_high_value: float | None = None,
    linear_cap_rounding_mode: str = "ceil_step",
    linear_cap_rounding_step: float = 10.0,
    linear_cap_min_vmax: float = 11.0,
    positive_only_for_percentiles: bool = False,
    linear_floor_zero: bool = False,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0

    mode = str(scale or "linear").strip().lower()
    use_log = bool(mode == "log")

    pct_low = float(np.clip(float(percentile_low), 0.0, 100.0))
    pct_high_linear = float(np.clip(float(percentile_high_linear), 0.0, 100.0))
    pct_high_log = float(np.clip(float(percentile_high_log), 0.0, 100.0))

    pct_source = arr
    if positive_only_for_percentiles:
        positive = arr[arr > 0.0]
        if positive.size > 0:
            pct_source = positive

    vmin = float(np.nanpercentile(pct_source, pct_low))
    high_pct = pct_high_log if use_log else pct_high_linear
    vmax = float(np.nanpercentile(pct_source, high_pct))

    if force_low_value is not None:
        try:
            vmin = float(force_low_value)
        except Exception:
            pass
    elif linear_floor_zero and not use_log:
        vmin = 0.0

    if force_high_value is not None:
        try:
            vmax = float(force_high_value)
        except Exception:
            pass
    elif not use_log:
        mode_round = str(linear_cap_rounding_mode or "ceil_step").strip().lower()
        if mode_round in {"none", "off", "disabled"}:
            rounded_vmax = float(vmax)
        elif mode_round in {"nice", "nice_1_2_5", "1-2-5"}:
            rounded_vmax = _round_up_nice(float(vmax))
        else:
            rounded_vmax = _round_up_step(float(vmax), float(linear_cap_rounding_step))
        vmax = float(max(float(linear_cap_min_vmax), rounded_vmax))

    if not np.isfinite(vmin):
        vmin = float(np.nanmin(arr))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(arr))

    if use_log:
        vmin = max(1e-12, float(vmin))

    if vmax <= vmin:
        vmax = vmin + (1e-6 if not use_log else max(1e-6, 0.01 * abs(vmin)))

    return float(vmin), float(vmax)


def prepare_linear_or_log_mapping(
    *,
    values: np.ndarray,
    scale: str,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, object | None, float, float]:
    """Prepare values and optional norm for matplotlib mappables.

    Returns (values_for_plot, norm_or_none, effective_vmin, effective_vmax).
    """

    arr = np.asarray(values, dtype=float)
    mode = str(scale or "linear").strip().lower()
    use_log = bool(mode == "log")
    lo = float(vmin)
    hi = float(vmax)

    if use_log:
        from matplotlib.colors import LogNorm  # type: ignore[import-not-found]

        lo_eff = max(1e-9, lo)
        hi_eff = max(lo_eff * 1.0001, hi)
        clipped = np.clip(arr, lo_eff, None)
        return clipped, LogNorm(vmin=lo_eff, vmax=hi_eff), float(lo_eff), float(hi_eff)

    return arr, None, float(lo), float(hi)


def build_footprint_norm_and_cmap(
    *,
    amp: np.ndarray,
    scale_amp: np.ndarray | None = None,
    use_log_norm: bool = True,
    scale_mode: str | None = None,
    low_color: str = "#1f4fff",
    mid_color: str = "#ffffff",
    high_color: str = "#ff0000",
    force_low_value: float | None = None,
    force_high_value: float | None = None,
    percentile_low: float = 5.0,
    percentile_high_linear: float = 99.0,
    percentile_high_log: float = 99.5,
    knot_anchor_values: tuple[float, float] = (1.0, 10.0),
    knot_y1_min: float = 0.02,
    knot_y1_max: float = 0.90,
    knot_y2_min: float = 0.07,
    knot_y2_max: float = 0.98,
    knot_min_gap: float = 0.05,
    linear_cap_rounding_mode: str = "ceil_step",
    linear_cap_rounding_step: float = 10.0,
    linear_cap_min_vmax: float = 11.0,
) -> tuple[object, object, np.ndarray, float, float]:
    """Return (norm, cmap, amp_render, vmin, vmax) for footprint rendering."""

    from matplotlib.colors import FuncNorm, LinearSegmentedColormap, LogNorm, Normalize

    mode = str(scale_mode or ("log" if bool(use_log_norm) else "linear")).strip().lower()
    if mode not in {"linear", "log"}:
        mode = "log" if bool(use_log_norm) else "linear"
    use_log = bool(mode == "log")

    a = np.asarray(amp, dtype=float).reshape(-1)
    scale_a = np.asarray(scale_amp, dtype=float).reshape(-1) if scale_amp is not None else np.asarray(a, dtype=float)

    amp_nonneg = np.maximum(a, 0.0)
    scale_nonneg = np.maximum(scale_a, 0.0)
    amp_render = np.maximum(amp_nonneg, 1e-12) if bool(use_log) else amp_nonneg
    scale_render = np.maximum(scale_nonneg, 1e-12) if bool(use_log) else scale_nonneg

    vmin, vmax = compute_value_limits(
        values=np.asarray(scale_render, dtype=float),
        scale=("log" if bool(use_log) else "linear"),
        percentile_low=float(percentile_low),
        percentile_high_linear=float(percentile_high_linear),
        percentile_high_log=float(percentile_high_log),
        force_low_value=force_low_value,
        force_high_value=force_high_value,
        linear_cap_rounding_mode=str(linear_cap_rounding_mode),
        linear_cap_rounding_step=float(linear_cap_rounding_step),
        linear_cap_min_vmax=float(linear_cap_min_vmax),
        positive_only_for_percentiles=True,
        linear_floor_zero=(not bool(use_log)),
    )

    if bool(use_log):
        norm = LogNorm(vmin=max(1e-12, float(vmin)), vmax=max(float(vmax), float(vmin) * 1.01))
    else:
        dyn_vmin = float(vmin)
        dyn_vmax = float(vmax)
        amp_sorted = np.sort(np.asarray(scale_render, dtype=float)) if scale_render.size else np.asarray([0.0, 1.0, 10.0, dyn_vmax], dtype=float)

        anchor_vals = list(knot_anchor_values) if knot_anchor_values is not None else [1.0, 10.0]
        if len(anchor_vals) < 2:
            anchor_vals = [1.0, 10.0]
        try:
            a1 = float(anchor_vals[0])
            a2 = float(anchor_vals[1])
        except Exception:
            a1, a2 = 1.0, 10.0
        if not np.isfinite(a1):
            a1 = 1.0
        if not np.isfinite(a2):
            a2 = 10.0
        if a1 > a2:
            a1, a2 = a2, a1
        a1 = float(np.clip(a1, dyn_vmin, dyn_vmax))
        a2 = float(np.clip(a2, dyn_vmin, dyn_vmax))
        if a2 <= a1:
            a2 = min(dyn_vmax, a1 + max(1e-6, 0.01 * max(1.0, dyn_vmax - dyn_vmin)))

        y1_min = float(np.clip(float(knot_y1_min), 0.0, 1.0))
        y1_max = float(np.clip(float(knot_y1_max), y1_min, 1.0))
        y2_min = float(np.clip(float(knot_y2_min), 0.0, 1.0))
        y2_max = float(np.clip(float(knot_y2_max), y2_min, 1.0))
        y_gap = float(max(0.0, float(knot_min_gap)))

        def _pct_rank(val: float) -> float:
            if amp_sorted.size == 0:
                return 0.5
            idx = int(np.searchsorted(amp_sorted, float(val), side="right"))
            return float(idx) / float(max(1, amp_sorted.size))

        y1 = float(np.clip(_pct_rank(a1), y1_min, y1_max))
        y2_lo = max(y2_min, y1 + y_gap)
        y2 = float(np.clip(_pct_rank(a2), y2_lo, y2_max))
        if y2 <= y1:
            y2 = min(1.0, y1 + max(1e-6, y_gap))

        x_knots = np.asarray([float(dyn_vmin), float(a1), float(a2), float(dyn_vmax)], dtype=float)
        y_knots = np.asarray([0.0, y1, y2, 1.0], dtype=float)

        def _piecewise_forward(x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x, dtype=float)
            arr = np.clip(arr, x_knots[0], x_knots[-1])
            return np.interp(arr, x_knots, y_knots)

        def _piecewise_inverse(y: np.ndarray) -> np.ndarray:
            arr = np.asarray(y, dtype=float)
            arr = np.clip(arr, y_knots[0], y_knots[-1])
            return np.interp(arr, y_knots, x_knots)

        if np.all(np.diff(x_knots) > 0) and np.all(np.diff(y_knots) > 0):
            norm = FuncNorm(
                (_piecewise_forward, _piecewise_inverse),
                vmin=float(dyn_vmin),
                vmax=float(dyn_vmax),
            )
        else:
            norm = Normalize(vmin=float(dyn_vmin), vmax=float(dyn_vmax))

    cmap = LinearSegmentedColormap.from_list(
        "black_low_mid_high",
        ["#000000", str(low_color), str(mid_color), str(high_color)],
        N=256,
    )
    return norm, cmap, amp_render, float(vmin), float(vmax)
