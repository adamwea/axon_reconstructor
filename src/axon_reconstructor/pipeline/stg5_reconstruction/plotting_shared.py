"""Shared callable plotting helpers for stg5 reconstruction."""

from __future__ import annotations

from typing import Any, Sequence


def normalize_minimap_linestyle(raw: Any) -> str | tuple[Any, ...]:
    """Normalize minimap inner-box linestyle tokens to Matplotlib styles."""

    v = str(raw or "dotted").strip().lower()
    if v in {"solid", "-"}:
        return "solid"
    if v in {"dotted", ":", "dot"}:
        return (0, (1.0, 1.0))
    if v in {"dashed", "--", "dash"}:
        return (0, (3.0, 2.0))
    return (0, (1.0, 1.0))


def normalize_corner_location(raw: Any, *, default: str = "bottomright") -> str:
    """Normalize corner-location tokens for overlays/annotations."""

    v = str(raw or default).strip().lower().replace("_", " ")
    aliases = {
        "topleft": "topleft",
        "top left": "topleft",
        "topright": "topright",
        "top right": "topright",
        "bottomleft": "bottomleft",
        "bottom left": "bottomleft",
        "bottomright": "bottomright",
        "bottom right": "bottomright",
    }
    return aliases.get(v, str(default or "bottomright"))


def parse_show_ticks_spec(raw: Any, *, default: Sequence[Any] = (1.0, 10.0, "dynamic_high")) -> list[Any]:
    """Parse show_ticks spec from YAML list or comma-separated string.

    Numeric values are kept as floats.
    Symbolic third-tick values are normalized to `dynamic_high`.
    """

    def _to_tokens(v: Any) -> list[Any]:
        if v is None:
            return list(default)
        if isinstance(v, (list, tuple)):
            return list(v)
        s = str(v).strip()
        if not s:
            return list(default)
        if s.lower() in {"none", "off", "false", "no"}:
            return []
        return [tok.strip() for tok in s.split(",") if tok.strip()]

    out: list[Any] = []
    for tok in _to_tokens(raw):
        if isinstance(tok, (int, float)):
            out.append(float(tok))
            continue
        sval = str(tok).strip().strip("{}[]()")
        low = sval.lower()
        if not low:
            continue
        try:
            out.append(float(sval))
            continue
        except Exception:
            pass
        if any(key in low for key in ("percentile", "p99", "99%", "top", "max", "high", ">")):
            out.append("dynamic_high")
            continue
        out.append(low)
    return out if out else list(default)


def resolve_colorbar_ticks(
    *,
    tick_spec: Sequence[Any],
    vmin: float,
    vmax: float,
    detected_amp_max: float | None = None,
) -> tuple[list[float], list[str] | None]:
    """Resolve tick positions/labels from mixed numeric/symbolic tick spec."""

    lo = float(min(vmin, vmax))
    hi = float(max(vmin, vmax))
    if hi <= lo:
        hi = lo + 1.0

    vals: list[float] = []
    labels: list[str] = []
    has_labels = False
    top_tick = hi
    for tok in list(tick_spec or []):
        if isinstance(tok, (int, float)):
            tv = float(tok)
            if lo <= tv <= hi:
                vals.append(tv)
                labels.append(_fmt_tick_plain(tv))
            continue

        t = str(tok).strip().lower()
        if t in {"dynamic_high", "top", "max", "high", "auto_high"}:
            vals.append(top_tick)
            if detected_amp_max is not None and float(detected_amp_max) > float(top_tick) + 1e-9:
                labels.append(f">{_fmt_tick_plain(top_tick)}")
                has_labels = True
            else:
                labels.append(_fmt_tick_plain(top_tick))
                has_labels = True
            continue

    if not vals:
        vals = [top_tick]
        labels = [_fmt_tick_plain(top_tick)]

    dedup: list[float] = []
    dedup_labels: list[str] = []
    for tv, lbl in sorted(zip(vals, labels), key=lambda x: x[0]):
        if dedup and abs(float(tv) - float(dedup[-1])) <= 1e-9:
            dedup_labels[-1] = lbl
            continue
        dedup.append(float(tv))
        dedup_labels.append(str(lbl))
    # Keep labels explicit when a symbolic token was used so tick display is deterministic.
    return dedup, (dedup_labels if has_labels else None)


def colorbar_axes_bounds(*, location: Any, length_fraction: float, pad_fraction: float) -> list[float]:
    """Return [left, bottom, width, height] for a figure-level vertical colorbar."""

    loc = normalize_corner_location(location, default="topright")
    length = min(0.95, max(0.05, float(length_fraction)))
    pad = min(0.25, max(0.0, float(pad_fraction)))
    width = 0.015

    if loc in {"topleft", "bottomleft"}:
        left = 0.06 + pad
    else:
        left = 0.93 - width - pad

    if loc in {"topleft", "topright"}:
        bottom = 0.97 - length - pad
    else:
        bottom = 0.03 + pad
    return [float(left), float(bottom), float(width), float(length)]


def _fmt_tick_plain(v: float) -> str:
    vv = float(v)
    if vv >= 100:
        return f"{vv:.0f}"
    if vv >= 10:
        return f"{vv:.1f}".rstrip("0").rstrip(".")
    return f"{vv:.2g}" if vv >= 1 else f"{vv:.2f}".rstrip("0").rstrip(".")


def build_footprint_norm_and_cmap(
    *,
    amp: Any,
    scale_amp: Any | None = None,
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
    knot_anchor_values: Sequence[float] = (1.0, 10.0),
    knot_y1_min: float = 0.02,
    knot_y1_max: float = 0.90,
    knot_y2_min: float = 0.07,
    knot_y2_max: float = 0.98,
    knot_min_gap: float = 0.05,
    linear_cap_rounding_mode: str = "ceil_step",
    linear_cap_rounding_step: float = 10.0,
    linear_cap_min_vmax: float = 11.0,
) -> tuple[Any, Any, Any, float, float]:
    """Return (norm, cmap, amp_render, vmin, vmax) for footprint rendering."""

    import numpy as np  # type: ignore[import-not-found]
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

    amp_detected_max = float(np.nanmax(scale_render)) if scale_render.size else 1.0
    vmax = float(amp_detected_max)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    try:
        pct_low = float(percentile_low)
    except Exception:
        pct_low = 5.0
    try:
        pct_high_linear = float(percentile_high_linear)
    except Exception:
        pct_high_linear = 99.0
    try:
        pct_high_log = float(percentile_high_log)
    except Exception:
        pct_high_log = 99.5
    pct_low = float(np.clip(pct_low, 0.0, 100.0))
    pct_high_linear = float(np.clip(pct_high_linear, 0.0, 100.0))
    pct_high_log = float(np.clip(pct_high_log, 0.0, 100.0))

    positive = scale_render[scale_render > 0.0]
    if positive.size > 0:
        vmin = float(np.nanpercentile(positive, pct_low))
    else:
        vmin = 1e-6
    vmin = max(1e-6, min(vmin, vmax)) if bool(use_log) else 0.0

    if force_low_value is not None:
        try:
            vmin = float(force_low_value)
        except Exception:
            pass
    if force_high_value is not None:
        try:
            vmax = float(force_high_value)
        except Exception:
            pass
    elif not bool(use_log):
        # Match the historical grid behavior: cap linear dynamic scale near p99,
        # then round up to a nice top tick so outliers can show as ">top".
        try:
            positive = scale_render[scale_render > 0.0]
            if positive.size:
                dyn_vmax_raw = float(np.percentile(positive, pct_high_linear))
                if not np.isfinite(dyn_vmax_raw) or dyn_vmax_raw <= 0.0:
                    dyn_vmax_raw = float(amp_detected_max)
            else:
                dyn_vmax_raw = float(amp_detected_max)

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

            mode_round = str(linear_cap_rounding_mode or "ceil_step").strip().lower()
            if mode_round in {"none", "off", "disabled"}:
                rounded_vmax = float(dyn_vmax_raw)
            elif mode_round in {"nice", "nice_1_2_5", "1-2-5"}:
                rounded_vmax = _round_up_nice(float(dyn_vmax_raw))
            else:
                rounded_vmax = _round_up_step(float(dyn_vmax_raw), float(linear_cap_rounding_step))

            vmax = float(max(float(linear_cap_min_vmax), rounded_vmax))
        except Exception:
            pass
    if vmax <= vmin:
        vmax = vmin + (1e-6 if not bool(use_log) else max(1e-6, 0.01 * abs(vmin)))

    if bool(use_log):
        if force_high_value is None and positive.size > 0:
            try:
                vmax = float(np.nanpercentile(positive, pct_high_log))
            except Exception:
                pass
        norm = LogNorm(vmin=max(1e-12, float(vmin)), vmax=max(float(vmax), float(vmin) * 1.01))
    else:
        # Preserve the grid's preferred dynamic linear distribution:
        # anchors [vmin, 1, 10, vmax] with knot positions driven by data percentiles.
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

        def _piecewise_forward(x: Any) -> Any:
            arr = np.asarray(x, dtype=float)
            arr = np.clip(arr, x_knots[0], x_knots[-1])
            return np.interp(arr, x_knots, y_knots)

        def _piecewise_inverse(y: Any) -> Any:
            arr = np.asarray(y, dtype=float)
            arr = np.clip(arr, y_knots[0], y_knots[-1])
            return np.interp(arr, y_knots, x_knots)

        norm = FuncNorm(
            (_piecewise_forward, _piecewise_inverse),
            vmin=float(dyn_vmin),
            vmax=float(dyn_vmax),
        )

    cmap = LinearSegmentedColormap.from_list(
        "black_low_mid_high",
        ["#000000", str(low_color), str(mid_color), str(high_color)],
        N=256,
    )
    return norm, cmap, amp_render, float(vmin), float(vmax)


def add_axes_scalebar(
    ax: Any,
    *,
    zoom_x0: float,
    zoom_x1: float,
    zoom_y0: float,
    zoom_y1: float,
    color: str = "white",
    bar_um: float | None = None,
    margin_frac: float = 0.06,
    text_offset_frac: float = 0.04,
    fontsize: float = 6.0,
    linewidth: float = 1.8,
) -> None:
    """Draw a horizontal scalebar in data coordinates for a zoomed panel."""

    try:
        span = max(1.0, float(zoom_x1) - float(zoom_x0))
        bar = (100.0 if span >= 180.0 else 50.0) if bar_um is None else max(1.0, float(bar_um))
        margin = float(margin_frac) * span

        x1 = float(zoom_x1) - margin
        x0 = x1 - float(bar)
        y0 = float(zoom_y0) + margin

        ax.plot(
            [x0, x1],
            [y0, y0],
            color=str(color),
            lw=max(0.1, float(linewidth)),
            solid_capstyle="butt",
            zorder=20,
        )
        ax.text(
            (x0 + x1) * 0.5,
            y0 + float(text_offset_frac) * span,
            f"{int(round(float(bar)))} um",
            color=str(color),
            fontsize=max(1.0, float(fontsize)),
            ha="center",
            va="bottom",
            zorder=21,
        )
    except Exception:
        return


def draw_footprint_squares(
    ax: Any,
    *,
    locs_xy: Any,
    amp: Any,
    scale_amp: Any | None = None,
    use_log_norm: bool = True,
    scale_mode: str | None = None,
    ch_pitch_um: float = 17.5,
    alpha: float = 0.95,
    low_color: str = "#1f4fff",
    mid_color: str = "#ffffff",
    high_color: str = "#ff0000",
    force_low_value: float | None = None,
    force_high_value: float | None = None,
    percentile_low: float = 5.0,
    percentile_high_linear: float = 99.0,
    percentile_high_log: float = 99.5,
    knot_anchor_values: Sequence[float] = (1.0, 10.0),
    knot_y1_min: float = 0.02,
    knot_y1_max: float = 0.90,
    knot_y2_min: float = 0.07,
    knot_y2_max: float = 0.98,
    knot_min_gap: float = 0.05,
    linear_cap_rounding_mode: str = "ceil_step",
    linear_cap_rounding_step: float = 10.0,
    linear_cap_min_vmax: float = 11.0,
) -> dict[str, Any] | None:
    """Draw MEA footprint as fixed-size channel squares with dynamic amplitude coloring."""

    import numpy as np  # type: ignore[import-not-found]
    from matplotlib.patches import Rectangle

    locs = np.asarray(locs_xy, dtype=float)
    a = np.asarray(amp, dtype=float).reshape(-1)
    if locs.ndim != 2 or locs.shape[1] < 2 or a.size != int(locs.shape[0]):
        return None

    norm, cmap, amp_render, vmin, vmax = build_footprint_norm_and_cmap(
        amp=a,
        scale_amp=scale_amp,
        use_log_norm=bool(use_log_norm),
        scale_mode=scale_mode,
        low_color=str(low_color),
        mid_color=str(mid_color),
        high_color=str(high_color),
        force_low_value=force_low_value,
        force_high_value=force_high_value,
        percentile_low=float(percentile_low),
        percentile_high_linear=float(percentile_high_linear),
        percentile_high_log=float(percentile_high_log),
        knot_anchor_values=list(knot_anchor_values),
        knot_y1_min=float(knot_y1_min),
        knot_y1_max=float(knot_y1_max),
        knot_y2_min=float(knot_y2_min),
        knot_y2_max=float(knot_y2_max),
        knot_min_gap=float(knot_min_gap),
        linear_cap_rounding_mode=str(linear_cap_rounding_mode),
        linear_cap_rounding_step=float(linear_cap_rounding_step),
        linear_cap_min_vmax=float(linear_cap_min_vmax),
    )

    facecolors = cmap(norm(amp_render))
    half = 0.5 * float(ch_pitch_um)
    for (x, y), fc in zip(locs[:, :2], facecolors):
        ax.add_patch(
            Rectangle(
                (float(x) - half, float(y) - half),
                float(ch_pitch_um),
                float(ch_pitch_um),
                facecolor=fc,
                edgecolor="none",
                alpha=float(alpha),
                zorder=1,
            )
        )
    return {
        "norm": norm,
        "cmap": cmap,
        "vmin": float(vmin),
        "vmax": float(vmax),
    }
