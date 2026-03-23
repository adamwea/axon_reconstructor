from __future__ import annotations

from typing import Any, Sequence

import numpy as np  # type: ignore[import-not-found]


def format_tick_plain(v: float) -> str:
    vv = float(v)
    if vv >= 100:
        return f"{vv:.0f}"
    if vv >= 10:
        return f"{vv:.1f}".rstrip("0").rstrip(".")
    return f"{vv:.2g}" if vv >= 1 else f"{vv:.2f}".rstrip("0").rstrip(".")


def parse_show_ticks_spec(raw: Any, *, default: Sequence[Any] = (1.0, 10.0, "dynamic_high")) -> list[Any]:
    """Parse show_ticks spec from YAML list or comma-separated string."""

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
    """Resolve tick positions/labels from mixed numeric/symbolic tick specs."""

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
                labels.append(format_tick_plain(tv))
            continue

        t = str(tok).strip().lower()
        if t in {"dynamic_high", "top", "max", "high", "auto_high"}:
            vals.append(top_tick)
            if detected_amp_max is not None and float(detected_amp_max) > float(top_tick) + 1e-9:
                labels.append(f">{format_tick_plain(top_tick)}")
                has_labels = True
            else:
                labels.append(format_tick_plain(top_tick))
                has_labels = True
            continue

    if not vals:
        vals = [top_tick]
        labels = [format_tick_plain(top_tick)]

    dedup: list[float] = []
    dedup_labels: list[str] = []
    for tv, lbl in sorted(zip(vals, labels), key=lambda x: x[0]):
        if dedup and abs(float(tv) - float(dedup[-1])) <= 1e-9:
            dedup_labels[-1] = lbl
            continue
        dedup.append(float(tv))
        dedup_labels.append(str(lbl))
    return dedup, (dedup_labels if has_labels else None)


def ticks_ending_in_0_or_5_with_max(
    *,
    vmin: float,
    vmax: float,
    decimal_places: int = 3,
    target_count: int | None = None,
) -> np.ndarray:
    vmin_f = float(vmin)
    vmax_f = float(vmax)
    if not np.isfinite(vmin_f) or not np.isfinite(vmax_f):
        return np.asarray([], dtype=float)
    if vmax_f <= vmin_f:
        return np.asarray([vmax_f], dtype=float)

    decimals = int(max(0, min(6, int(decimal_places))))
    if target_count is None:
        target = int(max(6, min(16, 6 + (decimals * 2))))
    else:
        target = int(max(3, min(24, int(target_count))))

    span = float(vmax_f - vmin_f)
    base_step = float(5.0 * (10.0 ** (-decimals)))
    if base_step <= float(np.finfo(float).eps):
        base_step = float(np.finfo(float).eps)

    multiplier = max(1, int(np.ceil(span / (base_step * float(max(1, target - 1))))))
    step = base_step * float(multiplier)

    start = float(np.ceil(vmin_f / step) * step)
    ticks = np.arange(start, vmax_f + (0.25 * step), step, dtype=float)
    ticks = ticks[np.isfinite(ticks)]
    ticks = ticks[(ticks >= (vmin_f - 1e-12)) & (ticks <= (vmax_f + 1e-12))]

    if ticks.size == 0:
        ticks = np.asarray([vmax_f], dtype=float)

    atol = max(1e-12, abs(step) * 1e-6)
    if not np.any(np.isclose(ticks, vmax_f, rtol=0.0, atol=atol)):
        ticks = np.append(ticks, vmax_f)

    ticks = np.unique(np.round(ticks, 12))
    ticks.sort()
    return ticks