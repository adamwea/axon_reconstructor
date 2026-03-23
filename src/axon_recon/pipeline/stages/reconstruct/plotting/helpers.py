from __future__ import annotations

from typing import Any, Sequence

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.shared.plotting import build_footprint_norm_and_cmap
from axon_recon.pipeline.shared.plotting import colorbar_axes_bounds
from axon_recon.pipeline.shared.plotting import compute_value_limits
from axon_recon.pipeline.shared.plotting import normalize_corner_location
from axon_recon.pipeline.shared.plotting import parse_show_ticks_spec
from axon_recon.pipeline.shared.plotting import prepare_linear_or_log_mapping
from axon_recon.pipeline.shared.plotting import resolve_colorbar_ticks


def reconstruct_tick_spec(raw: Any, *, default: Sequence[Any] = (1.0, 10.0, "dynamic_high")) -> list[Any]:
	"""Stage-local wrapper for parsing tick specifications."""
	return parse_show_ticks_spec(raw, default=default)


def reconstruct_colorbar_ticks(
	*,
	tick_spec: Sequence[Any],
	vmin: float,
	vmax: float,
	detected_max: float | None = None,
	detected_amp_max: float | None = None,
) -> tuple[list[float], list[str] | None]:
	"""Resolve reconstruct colorbar ticks/labels from mixed specs."""
	effective_detected = detected_amp_max if detected_amp_max is not None else detected_max
	return resolve_colorbar_ticks(
		tick_spec=tick_spec,
		vmin=vmin,
		vmax=vmax,
		detected_amp_max=effective_detected,
	)


def reconstruct_corner_location(raw: Any, *, default: str = "bottomright") -> str:
	"""Normalize location tokens for reconstruct overlays/annotations."""
	return normalize_corner_location(raw, default=default)


def reconstruct_colorbar_bounds(*, location: Any, length_fraction: float, pad_fraction: float) -> list[float]:
	"""Compute figure-relative colorbar bounds for reconstruct outputs."""
	return colorbar_axes_bounds(
		location=reconstruct_corner_location(location, default="topright"),
		length_fraction=length_fraction,
		pad_fraction=pad_fraction,
	)


def reconstruct_value_limits(
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
	"""Stage-local wrapper for shared percentile/cap-based value limits."""
	return compute_value_limits(
		values=values,
		scale=scale,
		percentile_low=percentile_low,
		percentile_high_linear=percentile_high_linear,
		percentile_high_log=percentile_high_log,
		force_low_value=force_low_value,
		force_high_value=force_high_value,
		linear_cap_rounding_mode=linear_cap_rounding_mode,
		linear_cap_rounding_step=linear_cap_rounding_step,
		linear_cap_min_vmax=linear_cap_min_vmax,
		positive_only_for_percentiles=positive_only_for_percentiles,
		linear_floor_zero=linear_floor_zero,
	)


def reconstruct_prepare_mapping(
	*,
	values: np.ndarray,
	scale: str,
	vmin: float,
	vmax: float,
) -> tuple[np.ndarray, object | None, float, float]:
	"""Prepare values + optional norm for reconstruct heatmap rendering."""
	return prepare_linear_or_log_mapping(values=values, scale=scale, vmin=vmin, vmax=vmax)


def reconstruct_draw_footprint_squares(
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
	"""Draw footprint squares using shared v2 normalization/cmap behavior."""

	from matplotlib.patches import Rectangle  # type: ignore[import-not-found]

	locs = np.asarray(locs_xy, dtype=float)
	a = np.asarray(amp, dtype=float).reshape(-1)
	if locs.ndim != 2 or locs.shape[1] < 2 or a.size != int(locs.shape[0]):
		return None

	norm, cmap, amp_render, vmin, vmax = build_footprint_norm_and_cmap(
		amp=a,
		scale_amp=None if scale_amp is None else np.asarray(scale_amp, dtype=float),
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
		knot_anchor_values=(float(knot_anchor_values[0]), float(knot_anchor_values[1])) if len(tuple(knot_anchor_values)) >= 2 else (1.0, 10.0),
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


__all__ = [
	"build_footprint_norm_and_cmap",
	"reconstruct_corner_location",
	"reconstruct_tick_spec",
	"reconstruct_colorbar_ticks",
	"reconstruct_colorbar_bounds",
	"reconstruct_value_limits",
	"reconstruct_prepare_mapping",
	"reconstruct_draw_footprint_squares",
]
