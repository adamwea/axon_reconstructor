from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SharedHeatmapConfig:
    """Common heatmap rendering settings shared across pipeline_v2 stages."""

    background: str = "black"
    invert_y_axis: bool = True

    low_color: str = "#1f4fff"
    mid_color: str = "#ffffff"
    high_color: str = "#ff0000"

    show_colorbar: bool = True
    colorbar_location: str = "topright"
    colorbar_fontsize: float = 6.0
    colorbar_fontcolor: str = "white"
    colorbar_length_fraction: float = 0.30
    colorbar_pad_fraction: float = 0.02
    show_ticks: tuple[Any, ...] = (1, 10, "dynamic_high")

    force_low_value: float | None = 0.0
    force_high_value: float | None = None
    scale: str = "linear"
    percentile_low: float = 5.0
    percentile_high_linear: float = 99.0
    percentile_high_log: float = 99.5

    knot_anchor_values: tuple[float, float] = (1.0, 10.0)
    knot_y1_min: float = 0.02
    knot_y1_max: float = 0.90
    knot_y2_min: float = 0.07
    knot_y2_max: float = 0.98
    knot_min_gap: float = 0.05

    linear_cap_rounding_mode: str = "ceil_step"
    linear_cap_rounding_step: float = 10.0
    linear_cap_min_vmax: float = 11.0

    @classmethod
    def from_block(cls, block: Mapping[str, Any] | None) -> SharedHeatmapConfig:
        src = dict(block or {})
        color_bar_raw = src.get("color_bar")
        color_bar = color_bar_raw if isinstance(color_bar_raw, dict) else {}
        display_raw = src.get("display")
        display = display_raw if isinstance(display_raw, dict) else {}

        show_ticks_raw = color_bar.get("show_ticks", src.get("show_ticks", cls.show_ticks))
        show_ticks = _to_tuple(show_ticks_raw, default=cls.show_ticks)

        knot_raw = color_bar.get("knot_anchor_values", src.get("knot_anchor_values", cls.knot_anchor_values))
        knot_vals = _to_knot_anchor_values(knot_raw, default=cls.knot_anchor_values)

        return cls(
            background=str(src.get("panel_background_color", src.get("background", cls.background))),
            invert_y_axis=_as_bool(display.get("invert_y_axis", src.get("invert_y_axis", cls.invert_y_axis)), cls.invert_y_axis),
            low_color=str(color_bar.get("low_color", src.get("low_color", cls.low_color))),
            mid_color=str(color_bar.get("mid_color", src.get("mid_color", cls.mid_color))),
            high_color=str(color_bar.get("high_color", src.get("high_color", cls.high_color))),
            show_colorbar=_as_bool(color_bar.get("show", src.get("show_color_bar", cls.show_colorbar)), cls.show_colorbar),
            colorbar_location=str(color_bar.get("location", src.get("color_bar_location", cls.colorbar_location))),
            colorbar_fontsize=_as_float(color_bar.get("fontsize", src.get("color_bar_fontsize", cls.colorbar_fontsize)), cls.colorbar_fontsize),
            colorbar_fontcolor=str(color_bar.get("fontcolor", src.get("color_bar_fontcolor", cls.colorbar_fontcolor))),
            colorbar_length_fraction=_as_float(
                color_bar.get("length_fraction", src.get("color_bar_length_fraction", cls.colorbar_length_fraction)),
                cls.colorbar_length_fraction,
            ),
            colorbar_pad_fraction=_as_float(
                color_bar.get("pad_fraction", src.get("color_bar_pad_fraction", cls.colorbar_pad_fraction)),
                cls.colorbar_pad_fraction,
            ),
            show_ticks=show_ticks,
            force_low_value=_as_optional_float(color_bar.get("force_low_value", src.get("force_low_value", cls.force_low_value))),
            force_high_value=_as_optional_float(color_bar.get("force_high_value", src.get("force_high_value", cls.force_high_value))),
            scale=str(color_bar.get("scale", src.get("scale", cls.scale))),
            percentile_low=_as_float(color_bar.get("percentile_low", src.get("percentile_low", cls.percentile_low)), cls.percentile_low),
            percentile_high_linear=_as_float(
                color_bar.get("percentile_high_linear", src.get("percentile_high_linear", cls.percentile_high_linear)),
                cls.percentile_high_linear,
            ),
            percentile_high_log=_as_float(
                color_bar.get("percentile_high_log", src.get("percentile_high_log", cls.percentile_high_log)),
                cls.percentile_high_log,
            ),
            knot_anchor_values=knot_vals,
            knot_y1_min=_as_float(color_bar.get("knot_y1_min", src.get("knot_y1_min", cls.knot_y1_min)), cls.knot_y1_min),
            knot_y1_max=_as_float(color_bar.get("knot_y1_max", src.get("knot_y1_max", cls.knot_y1_max)), cls.knot_y1_max),
            knot_y2_min=_as_float(color_bar.get("knot_y2_min", src.get("knot_y2_min", cls.knot_y2_min)), cls.knot_y2_min),
            knot_y2_max=_as_float(color_bar.get("knot_y2_max", src.get("knot_y2_max", cls.knot_y2_max)), cls.knot_y2_max),
            knot_min_gap=_as_float(color_bar.get("knot_min_gap", src.get("knot_min_gap", cls.knot_min_gap)), cls.knot_min_gap),
            linear_cap_rounding_mode=str(
                color_bar.get("linear_cap_rounding_mode", src.get("linear_cap_rounding_mode", cls.linear_cap_rounding_mode))
            ),
            linear_cap_rounding_step=_as_float(
                color_bar.get("linear_cap_rounding_step", src.get("linear_cap_rounding_step", cls.linear_cap_rounding_step)),
                cls.linear_cap_rounding_step,
            ),
            linear_cap_min_vmax=_as_float(
                color_bar.get("linear_cap_min_vmax", src.get("linear_cap_min_vmax", cls.linear_cap_min_vmax)),
                cls.linear_cap_min_vmax,
            ),
        )


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_tuple(value: Any, *, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return default
        parts = [p.strip() for p in token.split(",") if p.strip()]
        return tuple(parts) if parts else default
    if value is None:
        return default
    return (value,)


def _to_knot_anchor_values(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return default
    return default
