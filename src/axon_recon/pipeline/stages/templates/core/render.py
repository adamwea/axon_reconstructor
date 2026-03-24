from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]
from axon_recon.pipeline.shared.plotting import colorbar_axes_bounds
from axon_recon.pipeline.shared.plotting import compute_value_limits
from axon_recon.pipeline.shared.plotting import normalize_corner_location
from axon_recon.pipeline.shared.plotting import prepare_linear_or_log_mapping
from axon_recon.pipeline.shared.plotting import ticks_ending_in_0_or_5_with_max

from ..models.inputs import (
	FootprintMapGridReportConfig,
	FootprintMapConfig,
	ProbeGeometryConfig,
	PropagationPlotConfig,
	TemplateCirclesPlotConfig,
	TemplatePlotConfig,
	TemplateWaveformOverlayConfig,
	TopographicalFootprintConfig,
	TimeUpsampleConfig,
	WfOverlayGridReportConfig,
)


LOGGER = logging.getLogger("axon_recon.templates.render")


def _as_template_channels_by_time(template: Any, n_channels: int) -> np.ndarray:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError(f"Expected 2D template array, got shape={getattr(t, 'shape', None)}")
	if int(t.shape[0]) == int(n_channels):
		return t
	if int(t.shape[1]) == int(n_channels):
		return t.T
	return t


def _compute_plot_limits(points_xy: np.ndarray, *, pad_frac: float = 0.05, pad_abs: float = 10.0) -> tuple[float, float, float, float]:
	xs = points_xy[:, 0]
	ys = points_xy[:, 1]
	xmin = float(np.min(xs))
	xmax = float(np.max(xs))
	ymin = float(np.min(ys))
	ymax = float(np.max(ys))

	dx = max(float(xmax - xmin), 0.0)
	dy = max(float(ymax - ymin), 0.0)
	px = max(float(pad_abs), float(pad_frac * dx))
	py = max(float(pad_abs), float(pad_frac * dy))

	if dx == 0.0:
		px = max(px, float(pad_abs))
	if dy == 0.0:
		py = max(py, float(pad_abs))

	return xmin - px, xmax + px, ymin - py, ymax + py


def _make_square_limits(
	xmin: float,
	xmax: float,
	ymin: float,
	ymax: float,
	*,
	center_xy: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
	w = float(xmax - xmin)
	h = float(ymax - ymin)
	side = max(w, h)
	if center_xy is None:
		cx = float((xmin + xmax) / 2.0)
		cy = float((ymin + ymax) / 2.0)
	else:
		cx = float(center_xy[0])
		cy = float(center_xy[1])
	half = float(side / 2.0)
	return cx - half, cx + half, cy - half, cy + half


def _apply_style(fig: Any, ax: Any, *, config: TemplatePlotConfig) -> None:
	bg = str(config.background or "").strip().lower()
	if bg == "black":
		fig.patch.set_facecolor("black")
		ax.set_facecolor("black")
		ax.tick_params(colors="white")
		for spine in ax.spines.values():
			spine.set_color("white")
		ax.xaxis.label.set_color("white")
		ax.yaxis.label.set_color("white")
		ax.title.set_color("white")
	else:
		fig.patch.set_facecolor("white")
		ax.set_facecolor("white")


def _add_scale_bar(ax: Any, *, config: TemplatePlotConfig) -> None:
	if not bool(config.show_scale_bar):
		return
	x0, x1 = ax.get_xlim()
	y0, y1 = ax.get_ylim()
	span_x = max(1.0, float(abs(x1 - x0)))
	span_y = max(1.0, float(abs(y1 - y0)))

	if config.scale_bar_length_um is None:
		bar = 100.0 if span_x >= 180.0 else 50.0
	else:
		bar = max(1.0, float(config.scale_bar_length_um))

	margin = float(config.scale_bar_y_offset_frac) * span_x
	x_right = float(max(x0, x1)) - margin
	x_left = x_right - bar
	y_bar = float(min(y0, y1)) + float(config.scale_bar_y_offset_frac) * span_y

	if x_left <= float(min(x0, x1)):
		x_left = float(min(x0, x1)) + margin
		x_right = x_left + bar

	ax.plot([x_left, x_right], [y_bar, y_bar], color=str(config.scale_bar_color), lw=float(config.scale_bar_linewidth), solid_capstyle="butt")
	ax.text(
		(x_left + x_right) / 2.0,
		y_bar + float(config.scale_bar_text_offset_frac) * span_y,
		f"{int(round(bar))} um",
		color=str(config.scale_bar_color),
		horizontalalignment="center",
		verticalalignment="bottom",
		fontsize=float(config.scale_bar_fontsize),
	)


def _time_upsample_template(template: np.ndarray, upsample: TimeUpsampleConfig) -> np.ndarray:
	if not bool(upsample.enabled) or int(upsample.factor) <= 1:
		return template

	factor = int(max(1, int(upsample.factor)))
	n_channels, n_samples = int(template.shape[0]), int(template.shape[1])
	if n_samples <= 1:
		return template
	m = str(getattr(upsample, "method", "sinc") or "sinc").strip().lower()

	if m in {"", "sinc", "whittaker-shannon", "whittaker_shannon", "polyphase", "resample_poly"}:
		try:
			from scipy.signal import resample_poly  # type: ignore[import-not-found]

			# Our template convention is channels x time, so resample along axis=1.
			return np.asarray(resample_poly(template, up=int(factor), down=1, axis=1), dtype=float)
		except Exception:
			m = "linear"

	if m in {"nearest", "nn"}:
		x_old = np.arange(n_samples, dtype=float)
		x_new = np.linspace(0.0, float(n_samples - 1), int((n_samples - 1) * factor + 1), dtype=float)
		nearest_idx = np.rint(x_new).astype(int)
		nearest_idx = np.clip(nearest_idx, 0, n_samples - 1)
		return template[:, nearest_idx]

	if m in {"linear", "interp"}:
		x_old = np.arange(n_samples, dtype=float)
		x_new = np.linspace(0.0, float(n_samples - 1), int((n_samples - 1) * factor + 1), dtype=float)
		upsampled = np.empty((n_channels, int(x_new.shape[0])), dtype=float)
		for ch in range(n_channels):
			upsampled[ch, :] = np.interp(x_new, x_old, template[ch, :])
		return upsampled

	raise ValueError(f"Unsupported template time upsample method: {upsample.method!r}")


def _nice_scale_value(value: float) -> float:
	v = float(max(1e-12, abs(value)))
	exp = float(np.floor(np.log10(v)))
	base = v / float(10.0**exp)
	if base <= 1.0:
		nice = 1.0
	elif base <= 2.0:
		nice = 2.0
	elif base <= 5.0:
		nice = 5.0
	else:
		nice = 10.0
	return float(nice * (10.0**exp))


def _format_no_sci(value: float, *, max_decimals: int = 6) -> str:
	# Render plain decimal labels (no scientific notation) for plot annotations.
	decimals = max(0, int(max_decimals))
	text = f"{float(value):.{decimals}f}"
	if "." in text:
		text = text.rstrip("0").rstrip(".")
	if text in {"", "-0"}:
		return "0"
	return text


def _add_propagation_scale_bars(
	*,
	ax: Any,
	n_samples: int,
	trace_offset_step: float,
	trace_gain: float,
	probe_geometry: ProbeGeometryConfig | None,
	text_color: str,
	anchor_x_frac: float,
	anchor_y_frac: float,
	time_fraction: float,
	amp_fraction: float,
	linewidth: float,
	fontsize: float,
	time_label_offset_frac: float,
	amp_label_offset_frac: float,
	max_trace_amplitude_units: float | None = None,
	force_amp_frac_to_max_amp: bool = False,
) -> None:
	x0, x1 = ax.get_xlim()
	y0, y1 = ax.get_ylim()
	span_x = float(max(1.0, abs(x1 - x0)))
	span_y = float(max(1.0, abs(y1 - y0)))
	anchor_x_frac = float(min(1.0, max(0.0, anchor_x_frac)))
	anchor_y_frac = float(min(1.0, max(0.0, anchor_y_frac)))
	time_fraction = float(min(1.0, max(1e-6, time_fraction)))
	amp_fraction = float(min(1.0, max(1e-6, amp_fraction)))
	time_label_offset_frac = float(max(0.0, time_label_offset_frac))
	amp_label_offset_frac = float(max(0.0, amp_label_offset_frac))
	linewidth = float(max(0.1, linewidth))
	fontsize = float(max(1.0, fontsize))

	# Time bar length in samples, with optional ms label when sampling rate is known.
	target_time_samples = _nice_scale_value(max(1.0, float(n_samples) * time_fraction))
	time_bar_samples = float(min(max(1.0, target_time_samples), span_x * 0.30))
	sr_hz = None if probe_geometry is None else probe_geometry.sampling_rate_hz
	if sr_hz is not None and float(sr_hz) > 0.0:
		time_ms = (time_bar_samples / float(sr_hz)) * 1000.0
		time_label = f"{_format_no_sci(time_ms, max_decimals=3)} ms"
	else:
		time_label = f"{int(round(time_bar_samples))} samples"

	# Amplitude bar can be tied either to plotted y-span or to max trace amplitude in template units.
	max_units_by_span = float((span_y * 0.45) / max(1e-9, abs(trace_gain)))
	if bool(force_amp_frac_to_max_amp) and max_trace_amplitude_units is not None and float(max_trace_amplitude_units) > 0.0:
		desired_amp_units = float(max(1e-6, max_trace_amplitude_units))
		# In forced mode, use the actual pre-gain max amplitude (no round-up inflation).
		amp_bar_units = float(min(max(1e-6, desired_amp_units), max_units_by_span))
	else:
		desired_amp_plot = float(max(1e-6, span_y * amp_fraction * 0.35))
		amp_bar_plot = float(min(max(1e-6, _nice_scale_value(desired_amp_plot)), span_y * 0.45))
		amp_bar_units = float(amp_bar_plot / max(1e-9, abs(trace_gain)))
	amp_bar_plot = float(amp_bar_units * trace_gain)
	amp_uv = float(amp_bar_units)
	amp_label = f"{_format_no_sci(amp_uv, max_decimals=3)} uV"

	# Place an L-shaped scale bar with configurable anchor in axis-fraction units.
	anchor_x = float(min(x0, x1)) + (anchor_x_frac * span_x)
	anchor_y = float(min(y0, y1)) + (anchor_y_frac * span_y)
	anchor_x = float(min(max(anchor_x, min(x0, x1) + 1.0), max(x0, x1) - 1.0))
	anchor_y = float(min(max(anchor_y, min(y0, y1) + 1.0), max(y0, y1) - 1.0))
	x_left = anchor_x - time_bar_samples
	y_top = anchor_y + amp_bar_plot

	ax.plot([x_left, anchor_x], [anchor_y, anchor_y], color=text_color, lw=linewidth, solid_capstyle="butt")
	ax.plot([x_left, x_left], [anchor_y, y_top], color=text_color, lw=linewidth, solid_capstyle="butt")
	ax.text(
		(x_left + anchor_x) / 2.0,
		anchor_y - (time_label_offset_frac * span_y),
		time_label,
		color=text_color,
		horizontalalignment="center",
		verticalalignment="top",
		fontsize=fontsize,
	)
	ax.text(
		x_left - (amp_label_offset_frac * span_x),
		(anchor_y + y_top) / 2.0,
		amp_label,
		color=text_color,
		horizontalalignment="right",
		verticalalignment="center",
		rotation=90,
		fontsize=fontsize,
	)


def _probe_electrode_dims_um(probe_geometry: ProbeGeometryConfig | None) -> tuple[float, float] | None:
	if probe_geometry is None:
		return None
	if probe_geometry.pitch_um is not None:
		side = float(max(1e-6, float(probe_geometry.pitch_um)))
		return side, side
	dx = probe_geometry.electrode_size_um_x
	dy = probe_geometry.electrode_size_um_y
	if dx is None and dy is None:
		return None
	dx_val = float(max(1e-6, dx if dx is not None else dy))
	dy_val = float(max(1e-6, dy if dy is not None else dx))
	# Use square electrode glyphs/bases to match the probe's square contact shape.
	# If x/y differ in config, keep the larger side to avoid tiny dot-like rendering.
	side = float(max(dx_val, dy_val))
	return side, side


def _fallback_square_side_um(locs_xy: np.ndarray) -> float:
	# Reconstruct parity fallback for missing probe geometry.
	default_side = 17.5
	try:
		xs = np.unique(np.asarray(locs_xy[:, 0], dtype=float))
		ys = np.unique(np.asarray(locs_xy[:, 1], dtype=float))
		dx = np.diff(np.sort(xs)) if xs.size > 1 else np.asarray([], dtype=float)
		dy = np.diff(np.sort(ys)) if ys.size > 1 else np.asarray([], dtype=float)
		candidates = np.concatenate([dx, dy]) if (dx.size > 0 or dy.size > 0) else np.asarray([], dtype=float)
		candidates = candidates[np.isfinite(candidates)]
		candidates = candidates[candidates > 1e-6]
		if candidates.size > 0:
			return float(np.median(candidates))
	except Exception:
		pass
	return float(default_side)


def render_template_plot(
	*,
	template: Any,
	locations_xy: Any,
	config: TemplatePlotConfig,
	png_path: Path,
	svg_path: Path,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Expected locations shape (n,2+), got {getattr(locs, 'shape', None)}")
	locs = locs[:, :2]

	template_c_by_t = _as_template_channels_by_time(template, int(locs.shape[0]))
	if int(template_c_by_t.shape[0]) != int(locs.shape[0]):
		raise ValueError(
			f"Template/locations size mismatch: template_channels={template_c_by_t.shape[0]} locations={locs.shape[0]}"
		)

	signal_color = str(config.signal_color or "white")
	amp = np.ptp(template_c_by_t, axis=1)
	peak_idx = int(np.argmax(amp)) if amp.size > 0 else 0

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)

	# Draw each channel's waveform at its physical XY location.
	pitch_side = _fallback_square_side_um(locs)
	half_width = float(max(1.0, 0.42 * pitch_side))
	max_abs = float(np.max(np.abs(template_c_by_t))) if template_c_by_t.size > 0 else 0.0
	if max_abs <= float(np.finfo(float).eps):
		max_abs = 1.0
	vertical_scale = float(max(1.0, 0.42 * pitch_side) / max_abs)
	x_axis = np.linspace(-half_width, half_width, int(template_c_by_t.shape[1]), dtype=float)
	for ch in range(int(template_c_by_t.shape[0])):
		x_trace = float(locs[ch, 0]) + x_axis
		y_trace = float(locs[ch, 1]) + (template_c_by_t[ch, :] * vertical_scale)
		ax.plot(x_trace, y_trace, color=signal_color, linewidth=0.7, alpha=0.9)
	ax.set_xlabel("x (um)")
	ax.set_ylabel("y (um)")

	view_points = locs
	if str(config.channel_scope) == "recorded_channels":
		keep = np.where(amp > float(np.finfo(float).eps))[0]
		if keep.size > 0:
			view_points = locs[keep, :]

	xmin, xmax, ymin, ymax = _compute_plot_limits(view_points)
	if bool(config.force_square_aspect):
		center = None
		if bool(config.force_center_soma) and 0 <= peak_idx < int(locs.shape[0]):
			center = (float(locs[peak_idx, 0]), float(locs[peak_idx, 1]))
		xmin, xmax, ymin, ymax = _make_square_limits(xmin, xmax, ymin, ymax, center_xy=center)
	elif bool(config.force_center_soma) and 0 <= peak_idx < int(locs.shape[0]):
		cx, cy = float(locs[peak_idx, 0]), float(locs[peak_idx, 1])
		w = float(xmax - xmin)
		h = float(ymax - ymin)
		xmin, xmax = cx - (w / 2.0), cx + (w / 2.0)
		ymin, ymax = cy - (h / 2.0), cy + (h / 2.0)

	xmin, xmax, ymin, ymax = _expand_limits_for_glyph_half_size(
		xmin=xmin,
		xmax=xmax,
		ymin=ymin,
		ymax=ymax,
		half_dx=half_width,
		half_dy=max(1.0, 0.42 * pitch_side),
	)

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_aspect("equal", adjustable="box")
	_apply_style(fig, ax, config=config)
	_add_scale_bar(ax, config=config)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_png"] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_svg"] = str(svg_path)

	plt.close(fig)
	return outputs


def render_template_circles_plot(
	*,
	template: Any,
	locations_xy: Any,
	config: TemplateCirclesPlotConfig,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Expected locations shape (n,2+), got {getattr(locs, 'shape', None)}")
	locs = locs[:, :2]

	template_c_by_t = _as_template_channels_by_time(template, int(locs.shape[0]))
	if int(template_c_by_t.shape[0]) != int(locs.shape[0]):
		raise ValueError(
			f"Template/locations size mismatch: template_channels={template_c_by_t.shape[0]} locations={locs.shape[0]}"
		)

	amp = np.ptp(template_c_by_t, axis=1)
	min_idx = np.argmin(template_c_by_t, axis=1).astype(float)
	ref = float(min_idx[int(np.argmax(amp))]) if min_idx.size > 0 else 0.0
	lat_samples = min_idx - ref
	lat, latency_units_label = _convert_latency_samples_to_units(
		lat_samples,
		units=str(config.color_bar_units or ""),
		probe_geometry=probe_geometry,
	)

	size_metric = amp if str(config.size_by) == "amplitude" else np.abs(lat)
	color_metric = amp if str(config.color_by) == "amplitude" else lat
	color_values = np.asarray(color_metric, dtype=float)
	vmin = float(np.nanmin(color_values)) if color_values.size > 0 else 0.0
	vmax = float(np.nanmax(color_values)) if color_values.size > 0 else 1.0
	if not np.isfinite(vmin):
		vmin = 0.0
	if not np.isfinite(vmax):
		vmax = 1.0
	if vmax <= vmin:
		vmax = vmin + 1.0
	color_norm = plt.Normalize(vmin=vmin, vmax=vmax)

	size_norm = np.asarray(size_metric, dtype=float)
	size_norm = np.nan_to_num(size_norm, nan=0.0, posinf=0.0, neginf=0.0)
	if float(np.max(size_norm)) > 0.0:
		size_norm = size_norm / float(np.max(size_norm))
	size_scale = max(0.0, float(getattr(config, "circle_size_scale_factor", 1.0)))
	sizes = (8.0 + 42.0 * size_norm) * size_scale

	peak_idx = int(np.argmax(amp)) if amp.size > 0 else 0

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	circles_cmap = _maybe_reversed_colormap("viridis", reverse=(str(config.color_by) == "latency"))
	sc = ax.scatter(
		locs[:, 0],
		locs[:, 1],
		s=sizes,
		c=color_values,
		cmap=circles_cmap,
		norm=color_norm,
		alpha=0.92,
		linewidths=0.0,
	)
	ax.set_xlabel("x (um)")
	ax.set_ylabel("y (um)")

	xmin, xmax, ymin, ymax = _compute_plot_limits(locs)
	if bool(config.force_square_aspect):
		center = None
		if bool(config.force_center_soma) and 0 <= peak_idx < int(locs.shape[0]):
			center = (float(locs[peak_idx, 0]), float(locs[peak_idx, 1]))
		xmin, xmax, ymin, ymax = _make_square_limits(xmin, xmax, ymin, ymax, center_xy=center)
	elif bool(config.force_center_soma) and 0 <= peak_idx < int(locs.shape[0]):
		cx, cy = float(locs[peak_idx, 0]), float(locs[peak_idx, 1])
		w = float(xmax - xmin)
		h = float(ymax - ymin)
		xmin, xmax = cx - (w / 2.0), cx + (w / 2.0)
		ymin, ymax = cy - (h / 2.0), cy + (h / 2.0)

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_aspect("equal", adjustable="box")
	_apply_style(fig, ax, config=config)
	_add_scale_bar(ax, config=config)

	# Use Matplotlib-managed colorbar geometry so savefig tight-bbox and DPI scaling stay consistent.
	cbar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=plt.get_cmap(circles_cmap))
	cbar_mappable.set_array(color_values)
	cbar = fig.colorbar(cbar_mappable, ax=ax, fraction=0.04, pad=0.03, extend="neither")
	label_color = "white" if str(config.background or "").strip().lower() == "black" else "black"
	show_axes_title = bool(config.color_bar_show_axes_title)
	show_unit_labels = bool(config.color_bar_show_unit_labels)
	color_bar_title = str(config.color_bar_title or "").strip()
	unit_token = str(latency_units_label or "").strip()
	# Keep colorbar fully opaque and avoid edge seams at bin boundaries.
	try:
		if getattr(cbar, "solids", None) is not None:
			cbar.solids.set_alpha(1.0)
			cbar.solids.set_edgecolor("face")
	except Exception:
		pass
	if str(config.color_by) == "latency" and unit_token and unit_token != "samples":
		decimals = int(max(0, min(6, int(getattr(config, "color_bar_tick_decimal_places", 3)))))
		target_count_raw = getattr(config, "color_bar_tick_target_count", None)
		target_count = None if target_count_raw is None else int(target_count_raw)
		ticks = _ticks_ending_in_0_or_5_with_max(
			vmin=vmin,
			vmax=vmax,
			decimal_places=decimals,
			target_count=target_count,
		)
		if ticks is not None and len(ticks) > 0:
			labels = [f"{float(t):.{decimals}f} {unit_token}" for t in ticks] if show_unit_labels else [f"{float(t):.{decimals}f}" for t in ticks]
			cbar.set_ticks(ticks, labels=labels)
	if show_axes_title:
		if str(config.color_by) == "amplitude":
			cbar.set_label("", fontsize=7, color=label_color)
			units_token = str(config.color_bar_units or "").strip()
			if color_bar_title:
				cbar.ax.set_title(color_bar_title, fontsize=7, color=label_color, pad=4)
			elif units_token:
				cbar.ax.set_title(units_token, fontsize=7, color=label_color, pad=4)
			elif not units_token:
				cbar.ax.set_title("")
		else:
			cbar.set_label("", fontsize=7, color=label_color)
			if color_bar_title:
				cbar.ax.set_title(color_bar_title, fontsize=7, color=label_color, pad=4)
			else:
				fallback_title = f"Latency ({unit_token})" if (unit_token and unit_token != "samples") else "Latency"
				cbar.ax.set_title(fallback_title, fontsize=7, color=label_color, pad=4)
	else:
		# Hide axis-level colorbar title/label and keep only tick values.
		cbar.set_label("")
		cbar.ax.set_title("")
	if str(config.background or "").strip().lower() == "black":
		cbar.ax.tick_params(colors="white")
		cbar.outline.set_edgecolor("white")
		if show_axes_title:
			cbar.set_label(cbar.ax.get_ylabel(), color="white", fontsize=7)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_circles_png"] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_circles_svg"] = str(svg_path)

	plt.close(fig)
	return outputs


def render_template_circles_plot_v2(
	*,
	template: Any,
	locations_xy: Any,
	config: TemplateCirclesPlotConfig,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Expected locations shape (n,2+), got {getattr(locs, 'shape', None)}")
	locs = locs[:, :2]

	template_c_by_t = _as_template_channels_by_time(template, int(locs.shape[0]))
	if int(template_c_by_t.shape[0]) != int(locs.shape[0]):
		raise ValueError(
			f"Template/locations size mismatch: template_channels={template_c_by_t.shape[0]} locations={locs.shape[0]}"
		)

	amp = np.ptp(template_c_by_t, axis=1)
	min_idx = np.argmin(template_c_by_t, axis=1).astype(float)
	ref = float(min_idx[int(np.argmax(amp))]) if min_idx.size > 0 else 0.0
	lat_samples = min_idx - ref
	lat, latency_units_label = _convert_latency_samples_to_units(
		lat_samples,
		units=str(config.color_bar_units or ""),
		probe_geometry=probe_geometry,
	)

	size_metric = amp if str(config.size_by) == "amplitude" else np.abs(lat)
	color_metric = amp if str(config.color_by) == "amplitude" else lat
	color_values = np.asarray(color_metric, dtype=float)
	vmin = float(np.nanmin(color_values)) if color_values.size > 0 else 0.0
	vmax = float(np.nanmax(color_values)) if color_values.size > 0 else 1.0
	if not np.isfinite(vmin):
		vmin = 0.0
	if not np.isfinite(vmax):
		vmax = 1.0
	if vmax <= vmin:
		vmax = vmin + 1.0
	color_norm = plt.Normalize(vmin=vmin, vmax=vmax)

	size_norm = np.asarray(size_metric, dtype=float)
	size_norm = np.nan_to_num(size_norm, nan=0.0, posinf=0.0, neginf=0.0)
	if float(np.max(size_norm)) > 0.0:
		size_norm = size_norm / float(np.max(size_norm))
	size_scale = max(0.0, float(getattr(config, "circle_size_scale_factor", 1.0)))
	sizes = (8.0 + 42.0 * size_norm) * size_scale

	peak_idx = int(np.argmax(amp)) if amp.size > 0 else 0

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	circles_cmap = _maybe_reversed_colormap("viridis", reverse=(str(config.color_by) == "latency"))
	ax.scatter(
		locs[:, 0],
		locs[:, 1],
		s=sizes,
		c=color_values,
		cmap=circles_cmap,
		norm=color_norm,
		alpha=0.92,
		linewidths=0.0,
	)
	ax.set_xlabel("x (um)")
	ax.set_ylabel("y (um)")

	xmin, xmax, ymin, ymax = _compute_plot_limits(locs)
	if bool(config.force_square_aspect):
		center = None
		if bool(config.force_center_soma) and 0 <= peak_idx < int(locs.shape[0]):
			center = (float(locs[peak_idx, 0]), float(locs[peak_idx, 1]))
		xmin, xmax, ymin, ymax = _make_square_limits(xmin, xmax, ymin, ymax, center_xy=center)
	elif bool(config.force_center_soma) and 0 <= peak_idx < int(locs.shape[0]):
		cx, cy = float(locs[peak_idx, 0]), float(locs[peak_idx, 1])
		w = float(xmax - xmin)
		h = float(ymax - ymin)
		xmin, xmax = cx - (w / 2.0), cx + (w / 2.0)
		ymin, ymax = cy - (h / 2.0), cy + (h / 2.0)

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_aspect("equal", adjustable="box")
	_apply_style(fig, ax, config=config)
	_add_scale_bar(ax, config=config)

	# v2: build a boundary-based colorbar to avoid renderer interpolation seams/caps.
	# Keep the marker colormap continuous while forcing deterministic colorbar patch bounds.
	bounds = np.linspace(vmin, vmax, 257, dtype=float)
	boundary_norm = plt.matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=plt.get_cmap(circles_cmap).N, clip=True)
	cbar_mappable = plt.cm.ScalarMappable(norm=boundary_norm, cmap=plt.get_cmap(circles_cmap))
	cbar_mappable.set_array(color_values)
	cbar = fig.colorbar(
		cbar_mappable,
		ax=ax,
		fraction=0.04,
		pad=0.03,
		extend="neither",
		boundaries=bounds,
		spacing="proportional",
	)

	label_color = "white" if str(config.background or "").strip().lower() == "black" else "black"
	show_axes_title = bool(config.color_bar_show_axes_title)
	show_unit_labels = bool(config.color_bar_show_unit_labels)
	color_bar_title = str(config.color_bar_title or "").strip()
	unit_token = str(latency_units_label or "").strip()

	try:
		if getattr(cbar, "solids", None) is not None:
			cbar.solids.set_alpha(1.0)
			cbar.solids.set_edgecolor("face")
	except Exception:
		pass

	if str(config.color_by) == "latency" and unit_token and unit_token != "samples":
		decimals = int(max(0, min(6, int(getattr(config, "color_bar_tick_decimal_places", 3)))))
		target_count_raw = getattr(config, "color_bar_tick_target_count", None)
		target_count = None if target_count_raw is None else int(target_count_raw)
		ticks = _ticks_ending_in_0_or_5_with_max(
			vmin=vmin,
			vmax=vmax,
			decimal_places=decimals,
			target_count=target_count,
		)
		if ticks is not None and len(ticks) > 0:
			labels = [f"{float(t):.{decimals}f} {unit_token}" for t in ticks] if show_unit_labels else [f"{float(t):.{decimals}f}" for t in ticks]
			cbar.set_ticks(ticks, labels=labels)

	if show_axes_title:
		if str(config.color_by) == "amplitude":
			cbar.set_label("", fontsize=7, color=label_color)
			units_token = str(config.color_bar_units or "").strip()
			if color_bar_title:
				cbar.ax.set_title(color_bar_title, fontsize=7, color=label_color, pad=4)
			elif units_token:
				cbar.ax.set_title(units_token, fontsize=7, color=label_color, pad=4)
			else:
				cbar.ax.set_title("")
		else:
			cbar.set_label("", fontsize=7, color=label_color)
			if color_bar_title:
				cbar.ax.set_title(color_bar_title, fontsize=7, color=label_color, pad=4)
			else:
				fallback_title = f"Latency ({unit_token})" if (unit_token and unit_token != "samples") else "Latency"
				cbar.ax.set_title(fallback_title, fontsize=7, color=label_color, pad=4)
	else:
		cbar.set_label("")
		cbar.ax.set_title("")

	if str(config.background or "").strip().lower() == "black":
		cbar.ax.tick_params(colors="white")
		cbar.outline.set_edgecolor("white")
		if show_axes_title:
			cbar.set_label(cbar.ax.get_ylabel(), color="white", fontsize=7)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_circles_png"] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_circles_svg"] = str(svg_path)

	plt.close(fig)
	return outputs


def _convert_latency_samples_to_units(
	latency_samples: np.ndarray,
	*,
	units: str,
	probe_geometry: ProbeGeometryConfig | None,
) -> tuple[np.ndarray, str]:
	arr = np.asarray(latency_samples, dtype=float)
	unit = str(units or "").strip().lower()
	if unit in {"", "sample", "samples"}:
		return arr, "samples"

	fs_hz = None if probe_geometry is None else probe_geometry.sampling_rate_hz
	if fs_hz is None or fs_hz <= 0:
		# Fall back to samples if no valid sample rate is available.
		return arr, "samples"

	if unit in {"s", "sec", "second", "seconds"}:
		return arr / float(fs_hz), "s"
	if unit in {"ms", "millisecond", "milliseconds"}:
		return (arr / float(fs_hz)) * 1_000.0, "ms"
	if unit in {"us", "microsecond", "microseconds"}:
		return (arr / float(fs_hz)) * 1_000_000.0, "us"

	# Unknown token: keep values in samples and reflect fallback in label.
	return arr, "samples"


def _ticks_ending_in_0_or_5_with_max(
	*,
	vmin: float,
	vmax: float,
	decimal_places: int = 3,
	target_count: int | None = None,
) -> np.ndarray:
	return ticks_ending_in_0_or_5_with_max(
		vmin=vmin,
		vmax=vmax,
		decimal_places=decimal_places,
		target_count=target_count,
	)


def _add_location_aware_colorbar(
	*,
	fig: Any,
	ax: Any,
	mappable: Any,
	location: str,
	length_fraction: float,
	pad_fraction: float,
	default_fraction: float,
	default_pad: float,
) -> Any:
	loc = normalize_corner_location(location, default="topright")
	if loc == "topright":
		return fig.colorbar(
			mappable,
			ax=ax,
			fraction=float(default_fraction),
			pad=float(default_pad),
		)
	cax = fig.add_axes(
		colorbar_axes_bounds(
			location=loc,
			length_fraction=float(length_fraction),
			pad_fraction=float(pad_fraction),
		)
	)
	return fig.colorbar(mappable, cax=cax)


def _render_topographical_footprint(
	*,
	locations_xy: Any,
	values: np.ndarray,
	config: TopographicalFootprintConfig,
	probe_geometry: ProbeGeometryConfig | None,
	title: str,
	png_path: Path,
	svg_path: Path,
	output_key_png: str,
	output_key_svg: str,
	reverse_color_map: bool = False,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	locs = np.asarray(locations_xy, dtype=float)
	vals = np.asarray(values, dtype=float).reshape(-1)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError("Topographical footprint requires (n,2+) locations")
	if int(vals.shape[0]) != int(locs.shape[0]):
		raise ValueError("Topographical footprint values/locations length mismatch")

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection="3d")
	bg = str(config.background or "").strip().lower()
	if bg == "black":
		fig.patch.set_facecolor("black")
		ax.set_facecolor("black")
		ax.xaxis.label.set_color("white")
		ax.yaxis.label.set_color("white")
		ax.zaxis.label.set_color("white")
		ax.tick_params(colors="white")
		text_color = "white"
	else:
		fig.patch.set_facecolor("white")
		ax.set_facecolor("white")
		text_color = "black"

	cmap = plt.get_cmap(_maybe_reversed_colormap(str(config.color_map), reverse=bool(reverse_color_map)))
	vmin, vmax = compute_value_limits(
		values=np.asarray(vals, dtype=float),
		scale=str(config.scale),
		percentile_low=float(config.percentile_low),
		percentile_high_linear=float(config.percentile_high_linear),
		percentile_high_log=float(config.percentile_high_log),
		force_low_value=config.force_low_value,
		force_high_value=config.force_high_value,
		linear_cap_rounding_mode=str(config.linear_cap_rounding_mode),
		linear_cap_rounding_step=float(config.linear_cap_rounding_step),
		linear_cap_min_vmax=float(config.linear_cap_min_vmax),
	)
	vals_plot, norm, _, _ = prepare_linear_or_log_mapping(
		values=vals,
		scale=str(config.scale),
		vmin=float(vmin),
		vmax=float(vmax),
	)
	if norm is None:
		from matplotlib.colors import Normalize  # type: ignore[import-not-found]

		mappable_norm = Normalize(vmin=float(vmin), vmax=float(vmax))
	else:
		mappable_norm = norm
	colors = cmap(mappable_norm(vals_plot))
	dims = _probe_electrode_dims_um(probe_geometry)
	if dims is None:
		side = _fallback_square_side_um(locs[:, :2])
		dx = dy = float(side)
	else:
		dx, dy = dims
	z0 = np.minimum(vals, 0.0)
	dz = np.abs(vals)
	ax.bar3d(
		locs[:, 0] - (dx / 2.0),
		locs[:, 1] - (dy / 2.0),
		z0,
		np.full(locs.shape[0], dx, dtype=float),
		np.full(locs.shape[0], dy, dtype=float),
		dz,
		color=colors,
		shade=True,
		zsort="average",
	)
	xmin, xmax, ymin, ymax = _limits_for_template_shape(
		locs[:, :2],
		template_shape=str(config.template_shape),
	)
	xmin, xmax, ymin, ymax = _expand_limits_for_glyph_half_size(
		xmin=xmin,
		xmax=xmax,
		ymin=ymin,
		ymax=ymax,
		half_dx=float(dx) / 2.0,
		half_dy=float(dy) / 2.0,
	)
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.view_init(elev=float(config.elevation_deg), azim=float(config.azimuth_deg))
	ax.set_xlabel("x (um)", color=text_color)
	ax.set_ylabel("y (um)", color=text_color)
	ax.set_zlabel("value", color=text_color)
	ax.set_title(title, color=text_color)

	if bool(config.show_color_bar):
		sm = plt.cm.ScalarMappable(norm=mappable_norm, cmap=cmap)
		sm.set_array(vals_plot)
		cbar = _add_location_aware_colorbar(
			fig=fig,
			ax=ax,
			mappable=sm,
			location=str(config.color_bar_location),
			length_fraction=float(config.color_bar_length_fraction),
			pad_fraction=float(config.color_bar_pad_fraction),
			default_fraction=0.035,
			default_pad=0.08,
		)
		cbar.ax.tick_params(colors=text_color)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs[output_key_png] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs[output_key_svg] = str(svg_path)

	plt.close(fig)
	return outputs


def render_topographical_amplitude_footprint(
	*,
	template: Any,
	locations_xy: Any,
	config: TopographicalFootprintConfig,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
) -> dict[str, str]:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError("Topographical amplitude footprint requires 2D template")
	amp = np.ptp(t, axis=1)
	return _render_topographical_footprint(
		locations_xy=locations_xy,
		values=amp,
		config=config,
		probe_geometry=probe_geometry,
		title="Topographical footprint amplitude",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="topographical_amplitude_footprint_png",
		output_key_svg="topographical_amplitude_footprint_svg",
		reverse_color_map=False,
	)


def render_topographical_latency_footprint(
	*,
	template: Any,
	locations_xy: Any,
	config: TopographicalFootprintConfig,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
) -> dict[str, str]:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError("Topographical latency footprint requires 2D template")
	min_idx = np.argmin(t, axis=1).astype(float)
	ref = float(min_idx[int(np.argmax(np.ptp(t, axis=1)))]) if min_idx.size > 0 else 0.0
	lat = min_idx - ref
	return _render_topographical_footprint(
		locations_xy=locations_xy,
		values=lat,
		config=config,
		probe_geometry=probe_geometry,
		title="Topographical footprint latency",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="topographical_latency_footprint_png",
		output_key_svg="topographical_latency_footprint_svg",
		reverse_color_map=True,
	)


def render_propagation_plot(
	*,
	template: Any,
	locations_xy: Any,
	config: PropagationPlotConfig,
	pdf_path: Path,
	png_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	t = np.asarray(template)
	locs = np.asarray(locations_xy, dtype=float)
	if t.ndim != 2:
		raise ValueError("Propagation plot requires 2D template")
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError("Propagation plot requires (n,2+) locations")
	if int(t.shape[0]) != int(locs.shape[0]):
		raise ValueError("Propagation plot template/locations mismatch")

	ptp = np.ptp(t, axis=1)
	top_n = max(1, min(int(config.top_channels), int(t.shape[0])))
	selected = np.argsort(-ptp)[:top_n]
	selected_abs_max = np.max(np.abs(t[selected, :]), axis=1)
	max_amp_channel = int(selected[int(np.argmax(selected_abs_max))]) if selected.size > 0 else None
	if bool(getattr(config, "debug_max_amps_at_each_channel", False)):
		amps_by_channel = {
			int(ch): float(amp) for ch, amp in zip(selected.tolist(), selected_abs_max.tolist(), strict=False)
		}
		debug_msg = f"Propagation plot debug: max amplitude at each plotted channel before gain (uV): {amps_by_channel}"
		print(debug_msg)
		LOGGER.info(debug_msg)
	lat_idx = np.argmax(np.abs(t[selected, :]), axis=1).astype(float)
	order = np.argsort(lat_idx)
	selected = selected[order]
	lat_idx = lat_idx[order]

	channels_per_panel = max(1, int(config.channels_per_panel))
	overlap = max(0, int(config.channel_overlap))
	if overlap >= channels_per_panel:
		overlap = channels_per_panel - 1
	stride = max(1, channels_per_panel - overlap)

	panels: list[np.ndarray] = []
	start = 0
	n_selected = int(selected.shape[0])
	while start < n_selected:
		end = min(n_selected, start + channels_per_panel)
		panels.append(np.arange(start, end, dtype=int))
		if end >= n_selected:
			break
		start += stride

	n_panels = max(1, len(panels))
	fig = plt.figure(figsize=(13, 2.8 * n_panels + 1.0))
	gs = fig.add_gridspec(nrows=n_panels, ncols=1, hspace=0.35)
	trace_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_panels)]

	bg = str(config.background or "").strip().lower()
	if bg == "black":
		fig.patch.set_facecolor("black")
		for ax in trace_axes:
			ax.set_facecolor("black")
			ax.tick_params(colors="white")
			for spine in ax.spines.values():
				spine.set_color("white")
		trace_color = "white"
		text_color = "white"
	else:
		fig.patch.set_facecolor("white")
		for ax in trace_axes:
			ax.set_facecolor("white")
		trace_color = "black"
		text_color = "black"

	x = np.arange(int(t.shape[1]), dtype=float)
	base_step = float(max(1e-6, np.max(np.ptp(t[selected, :], axis=1))))
	offset_step = base_step * max(0.2, float(config.trace_spacing))
	trace_gain = float(max(1e-9, float(config.trace_gain)))
	peak_marker_height_frac = float(max(1e-6, float(getattr(config, "peak_marker_height_frac", 0.24))))
	peak_marker_linewidth = float(max(0.2, float(getattr(config, "peak_marker_linewidth", 1.4))))
	label_alignment = str(getattr(config, "channel_label_alignment", "left") or "left").strip().lower()
	if label_alignment not in {"left", "center", "right"}:
		label_alignment = "left"
	for panel_i, panel_inds in enumerate(panels):
		ax = trace_axes[panel_i]
		label_x_offset = float(config.channel_label_x_offset_frac) * float(max(1, x.shape[0]))
		label_y_offset = float(config.channel_label_y_offset_frac) * float(offset_step)
		min_label_x: float | None = None
		for i_local, idx in enumerate(panel_inds):
			ch = int(selected[int(idx)])
			off = float(i_local) * offset_step
			y = (t[ch, :] * trace_gain) + off
			ax.plot(x, y, color=trace_color, linewidth=0.9, alpha=0.95)
			pk = float(np.argmax(np.abs(t[ch, :])))
			peak_y = float(y[int(pk)])
			marker_height = float(max(1.2, peak_marker_height_frac * offset_step))
			marker_half = float(0.5 * marker_height)
			ax.plot(
				[pk, pk],
				[peak_y - marker_half, peak_y + marker_half],
				color="black",
				linewidth=peak_marker_linewidth,
				solid_capstyle="butt",
			)
			ax.text(
				x[0] + label_x_offset,
				off + label_y_offset,
				f"ch {int(ch)}",
				color=text_color,
				fontsize=float(config.channel_label_fontsize),
				fontweight=(
					"bold"
					if bool(getattr(config, "bold_max_amp_channel_label", False)) and max_amp_channel is not None and int(ch) == max_amp_channel
					else "normal"
				),
				horizontalalignment=label_alignment,
				verticalalignment="center",
			)
			if min_label_x is None:
				min_label_x = float(x[0] + label_x_offset)
			else:
				min_label_x = min(min_label_x, float(x[0] + label_x_offset))

		if min_label_x is not None:
			x_left, x_right = ax.get_xlim()
			if float(min_label_x) < float(x_left):
				pad = max(1.0, float(0.02 * abs(x_right - x_left)))
				ax.set_xlim(float(min_label_x) - pad, float(x_right))

		start_idx = int(panel_inds[0])
		end_idx = int(panel_inds[-1])
		if bool(config.show_title):
			title_template = str(config.title_template or "Propagation traces {start}-{end} / {total}")
			try:
				title = title_template.format(start=start_idx + 1, end=end_idx + 1, total=n_selected)
			except Exception:
				title = f"Propagation traces {start_idx + 1}-{end_idx + 1} / {n_selected}"
			ax.set_title(
				title,
				color=text_color,
				fontsize=float(config.title_fontsize),
			)
		ax.set_xticks([])
		ax.set_yticks([])
		for spine in ax.spines.values():
			spine.set_visible(False)

	if bool(config.show_scale_bar):
		max_trace_amp_units = float(np.max(np.abs(t[selected, :])))
		_add_propagation_scale_bars(
			ax=trace_axes[-1],
			n_samples=int(t.shape[1]),
			trace_offset_step=offset_step,
			trace_gain=trace_gain,
			max_trace_amplitude_units=max_trace_amp_units,
			force_amp_frac_to_max_amp=bool(config.force_amp_frac_to_max_amp),
			probe_geometry=probe_geometry,
			text_color=text_color,
			anchor_x_frac=float(config.scale_bar_anchor_x_frac),
			anchor_y_frac=float(config.scale_bar_anchor_y_frac),
			time_fraction=float(config.scale_bar_time_fraction),
			amp_fraction=float(config.scale_bar_amp_fraction),
			linewidth=float(config.scale_bar_linewidth),
			fontsize=float(config.scale_bar_fontsize),
			time_label_offset_frac=float(config.scale_bar_time_label_offset_frac),
			amp_label_offset_frac=float(config.scale_bar_amp_label_offset_frac),
		)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["propagation_plot_png"] = str(png_path)
	if bool(config.write_pdf):
		pdf_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["propagation_plot_pdf"] = str(pdf_path)

	plt.close(fig)
	return outputs


def render_template_wf_overlay(
	*,
	template: Any,
	config: TemplateWaveformOverlayConfig,
	time_upsample: TimeUpsampleConfig,
	pdf_path: Path,
	png_path: Path,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError(f"Expected 2D template array for overlay, got shape={getattr(t, 'shape', None)}")
	t = _time_upsample_template(t, time_upsample)

	n_channels, n_samples = int(t.shape[0]), int(t.shape[1])
	if n_channels <= 0 or n_samples <= 0:
		raise ValueError("Template overlay received empty template")

	ptp = np.ptp(t, axis=1)
	top_n = max(1, int(config.top_channels_per_template))
	order = np.argsort(-ptp)
	selected = order[: min(top_n, n_channels)]

	x = np.arange(n_samples, dtype=float)
	selected_templates = t[selected, :]
	offset_step = float(max(1e-6, np.max(np.ptp(selected_templates, axis=1)) * 1.4))

	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111)

	bg = str(config.background or "").strip().lower()
	if bg == "black":
		fig.patch.set_facecolor("black")
		ax.set_facecolor("black")
		ax.tick_params(colors="white")
		for spine in ax.spines.values():
			spine.set_color("white")
		trace_color = "white"
		mean_color = "cyan"
	else:
		fig.patch.set_facecolor("white")
		ax.set_facecolor("white")
		trace_color = "black"
		mean_color = "red"

	style = str(getattr(config, "style", "overlay") or "overlay").strip().lower()
	if style in {"stack", "stacked"}:
		for idx, ch in enumerate(selected):
			offset = float(idx) * offset_step
			ax.plot(x, selected_templates[idx, :] + offset, color=trace_color, linewidth=0.9, alpha=0.9)
			ax.text(
				x[0],
				offset,
				f"ch {int(ch)}",
				fontsize=6,
				color=trace_color,
				verticalalignment="bottom",
				horizontalalignment="left",
			)
	else:
		for idx, ch in enumerate(selected):
			ax.plot(x, selected_templates[idx, :], color=trace_color, linewidth=0.9, alpha=0.4)
			ax.text(
				x[0],
				selected_templates[idx, 0],
				f"ch {int(ch)}",
				fontsize=6,
				color=trace_color,
				verticalalignment="bottom",
				horizontalalignment="left",
			)

	if bool(config.include_mean):
		mean_t = np.mean(selected_templates, axis=0)
		mean_offset = float(len(selected)) * offset_step if style in {"stack", "stacked"} else 0.0
		ax.plot(x, mean_t + mean_offset, color=mean_color, linewidth=1.4, alpha=0.95)
		ax.text(
			x[0],
			float(mean_t[0]) + mean_offset,
			"mean",
			fontsize=6,
			color=mean_color,
			verticalalignment="bottom",
			horizontalalignment="left",
		)

	ax.set_xlabel("sample")
	ax.set_ylabel("amplitude + offset" if style in {"stack", "stacked"} else "amplitude")
	ax.set_title(f"Template waveforms (top channels, style={style})")

	if bool(config.include_scale_bar):
		x0, x1 = ax.get_xlim()
		y0, y1 = ax.get_ylim()
		span_x = max(1.0, float(abs(x1 - x0)))
		span_y = max(1.0, float(abs(y1 - y0)))
		bar_x = max(5.0, span_x * 0.1)
		bar_y = max(1e-6, span_y * 0.1)
		x_left = float(min(x0, x1)) + 0.05 * span_x
		y_bot = float(min(y0, y1)) + 0.08 * span_y
		color = str(config.scale_bar_color)
		lw = float(config.scale_bar_linewidth)
		ax.plot([x_left, x_left + bar_x], [y_bot, y_bot], color=color, lw=lw)
		ax.plot([x_left, x_left], [y_bot, y_bot + bar_y], color=color, lw=lw)
		ax.text(
			x_left + bar_x,
			y_bot,
			f" {int(round(bar_x))} samples",
			fontsize=float(config.scale_bar_fontsize),
			color=color,
			horizontalalignment="left",
			verticalalignment="center",
		)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_wf_overlay_png"] = str(png_path)
	if bool(config.write_pdf):
		pdf_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_wf_overlay_pdf"] = str(pdf_path)

	plt.close(fig)
	return outputs


def render_multi_source_pdf(
	*,
	units: list[dict[str, Any]],
	pdf_path: Path,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	from matplotlib.backends.backend_pdf import PdfPages  # type: ignore[import-not-found]

	pdf_path.parent.mkdir(parents=True, exist_ok=True)
	with PdfPages(pdf_path) as pdf:
		for unit in units:
			unit_id = unit.get("unit_id")
			outputs = unit.get("outputs", {}) if isinstance(unit.get("outputs", {}), dict) else {}
			image_candidates = [
				("template", outputs.get("template_png")),
				("overlay", outputs.get("template_wf_overlay_png")),
				("amp", outputs.get("footprint_amplitude_map_png")),
				("lat", outputs.get("footprint_latency_map_png")),
			]
			paths = [(label, Path(str(p))) for label, p in image_candidates if p]
			paths = [(label, p) for label, p in paths if p.exists()]
			if not paths:
				continue

			n = len(paths)
			ncols = min(3, max(1, int(np.ceil(np.sqrt(n)))))
			nrows = int(np.ceil(float(n) / float(ncols)))
			fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.2 * nrows))
			if not isinstance(axes, np.ndarray):
				axes = np.asarray([axes])
			ax_list = list(axes.ravel())

			for idx, ax in enumerate(ax_list):
				if idx >= n:
					ax.axis("off")
					continue
				label, img_path = paths[idx]
				img = plt.imread(img_path)
				ax.imshow(img)
				ax.set_title(label, fontsize=8)
				ax.axis("off")

			fig.suptitle(f"Unit {unit_id} templates summary", fontsize=11)
			pdf.savefig(fig, bbox_inches="tight")
			plt.close(fig)

	return {"multi_source_pdf": str(pdf_path)}


def render_wf_overlay_grid(
	*,
	overlay_png_paths: list[Path],
	config: WfOverlayGridReportConfig,
	pdf_path: Path,
	png_path: Path,
) -> dict[str, str]:
	return render_image_grid(
		image_paths=overlay_png_paths,
		write_pdf=bool(config.write_pdf),
		pdf_path=pdf_path,
		pdf_output_key="wf_overlay_grid_pdf",
		write_png=bool(config.write_png),
		png_path=png_path,
		png_output_key="wf_overlay_grid_png",
		title="Template waveform overlay grid",
	)


def render_footprint_map_grid(
	*,
	image_paths: list[Path],
	config: FootprintMapGridReportConfig,
	pdf_path: Path,
	png_path: Path,
	pdf_output_key: str,
	png_output_key: str,
	title: str,
) -> dict[str, str]:
	return render_image_grid(
		image_paths=image_paths,
		write_pdf=bool(config.write_pdf),
		pdf_path=pdf_path,
		pdf_output_key=pdf_output_key,
		write_png=bool(config.write_png),
		png_path=png_path,
		png_output_key=png_output_key,
		title=title,
	)


def render_image_grid(
	*,
	image_paths: list[Path],
	write_pdf: bool,
	pdf_path: Path,
	pdf_output_key: str,
	write_png: bool,
	png_path: Path,
	png_output_key: str,
	title: str,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	paths = [p for p in image_paths if p.exists()]
	if not paths:
		return {}

	n = len(paths)
	ncols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
	nrows = int(np.ceil(float(n) / float(ncols)))

	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0 * ncols, 3.0 * nrows))
	if not isinstance(axes, np.ndarray):
		axes = np.asarray([axes])
	ax_list = list(axes.ravel())

	for i, ax in enumerate(ax_list):
		if i >= n:
			ax.axis("off")
			continue
		img = plt.imread(paths[i])
		ax.imshow(img)
		ax.set_title(paths[i].parent.name, fontsize=7)
		ax.axis("off")

	fig.suptitle(title, fontsize=10)

	outputs: dict[str, str] = {}
	if bool(write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=200, bbox_inches="tight")
		outputs[png_output_key] = str(png_path)
	if bool(write_pdf):
		pdf_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
		outputs[pdf_output_key] = str(pdf_path)

	plt.close(fig)
	return outputs


def _map_values_to_limits(values: np.ndarray, config: FootprintMapConfig) -> tuple[float, float]:
	return compute_value_limits(
		values=np.asarray(values, dtype=float),
		scale=str(config.scale),
		percentile_low=float(config.percentile_low),
		percentile_high_linear=float(config.percentile_high_linear),
		percentile_high_log=float(config.percentile_high_log),
		force_low_value=config.force_low_value,
		force_high_value=config.force_high_value,
		linear_cap_rounding_mode=str(config.linear_cap_rounding_mode),
		linear_cap_rounding_step=float(config.linear_cap_rounding_step),
		linear_cap_min_vmax=float(config.linear_cap_min_vmax),
	)


def _limits_for_template_shape(
	points_xy: np.ndarray,
	*,
	template_shape: str,
	pad_frac: float = 0.01,
	pad_abs: float = 1.0,
) -> tuple[float, float, float, float]:
	xmin, xmax, ymin, ymax = _compute_plot_limits(points_xy, pad_frac=float(pad_frac), pad_abs=float(pad_abs))
	shape = str(template_shape or "square").strip().lower().replace("-", "_").replace(" ", "_")
	if shape == "square":
		return _make_square_limits(xmin, xmax, ymin, ymax)
	return xmin, xmax, ymin, ymax


def _expand_limits_for_glyph_half_size(
	*,
	xmin: float,
	xmax: float,
	ymin: float,
	ymax: float,
	half_dx: float,
	half_dy: float,
) -> tuple[float, float, float, float]:
	return (
		float(xmin) - float(max(0.0, half_dx)),
		float(xmax) + float(max(0.0, half_dx)),
		float(ymin) - float(max(0.0, half_dy)),
		float(ymax) + float(max(0.0, half_dy)),
	)


def _render_footprint_map(
	*,
	locations_xy: np.ndarray,
	values: np.ndarray,
	config: FootprintMapConfig,
	probe_geometry: ProbeGeometryConfig | None,
	title: str,
	png_path: Path,
	svg_path: Path,
	output_key_png: str,
	output_key_svg: str,
	reverse_color_map: bool = False,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Expected locations shape (n,2+), got {getattr(locs, 'shape', None)}")
	locs = locs[:, :2]
	vals = np.asarray(values, dtype=float).reshape(-1)
	if int(vals.shape[0]) != int(locs.shape[0]):
		raise ValueError("Footprint map values/locations length mismatch")

	vmin, vmax = _map_values_to_limits(vals, config)

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111)
	bg = str(config.background or "").strip().lower()
	if bg == "black":
		fig.patch.set_facecolor("black")
		ax.set_facecolor("black")
		ax.tick_params(colors="white")
		for spine in ax.spines.values():
			spine.set_color("white")
		text_color = "white"
	else:
		fig.patch.set_facecolor("white")
		ax.set_facecolor("white")
		text_color = "black"

	vals_plot, norm, vmin_eff, vmax_eff = prepare_linear_or_log_mapping(
		values=vals,
		scale=str(config.scale),
		vmin=float(vmin),
		vmax=float(vmax),
	)

	dims = _probe_electrode_dims_um(probe_geometry)
	from matplotlib.collections import PatchCollection  # type: ignore[import-not-found]
	from matplotlib.patches import Rectangle  # type: ignore[import-not-found]

	if dims is None:
		side = _fallback_square_side_um(locs[:, :2])
		dx = dy = float(side)
	else:
		dx, dy = dims
	patches = [
		Rectangle((float(x) - (dx / 2.0), float(y) - (dy / 2.0)), width=float(dx), height=float(dy))
		for x, y in locs[:, :2]
	]
	edge_color = "white" if bg == "black" else "black"
	sc = PatchCollection(
		patches,
		cmap=_maybe_reversed_colormap(str(config.color_map), reverse=bool(reverse_color_map)),
		linewidths=0.25,
		edgecolors=edge_color,
		antialiaseds=False,
	)
	sc.set_array(np.asarray(vals_plot, dtype=float))
	if norm is not None:
		sc.set_norm(norm)
	else:
		sc.set_clim(vmin_eff, vmax_eff)
	ax.add_collection(sc)
	xmin, xmax, ymin, ymax = _limits_for_template_shape(
		locs,
		template_shape=str(config.template_shape),
	)
	xmin, xmax, ymin, ymax = _expand_limits_for_glyph_half_size(
		xmin=xmin,
		xmax=xmax,
		ymin=ymin,
		ymax=ymax,
		half_dx=float(dx) / 2.0,
		half_dy=float(dy) / 2.0,
	)
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel("x (um)", color=text_color)
	ax.set_ylabel("y (um)", color=text_color)
	ax.set_title(title, color=text_color)

	if bool(config.show_color_bar):
		cbar = _add_location_aware_colorbar(
			fig=fig,
			ax=ax,
			mappable=sc,
			location=str(config.color_bar_location),
			length_fraction=float(config.color_bar_length_fraction),
			pad_fraction=float(config.color_bar_pad_fraction),
			default_fraction=float(config.color_bar_length_fraction),
			default_pad=float(config.color_bar_pad_fraction),
		)
		cbar.ax.tick_params(labelsize=float(config.color_bar_fontsize), colors=text_color)
		try:
			cbar.outline.set_edgecolor(text_color)
		except Exception:
			pass

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs[output_key_png] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs[output_key_svg] = str(svg_path)

	plt.close(fig)
	return outputs


def render_footprint_amplitude_map(
	*,
	template: Any,
	locations_xy: Any,
	config: FootprintMapConfig,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
) -> dict[str, str]:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError("Amplitude map requires 2D template")
	amp = np.ptp(t, axis=1)
	return _render_footprint_map(
		locations_xy=np.asarray(locations_xy),
		values=amp,
		config=config,
		probe_geometry=probe_geometry,
		title="Template footprint amplitude",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="footprint_amplitude_map_png",
		output_key_svg="footprint_amplitude_map_svg",
		reverse_color_map=False,
	)


def render_footprint_latency_map(
	*,
	template: Any,
	locations_xy: Any,
	config: FootprintMapConfig,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
) -> dict[str, str]:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError("Latency map requires 2D template")
	min_idx = np.argmin(t, axis=1).astype(float)
	ref = float(min_idx[int(np.argmax(np.ptp(t, axis=1)))]) if min_idx.size > 0 else 0.0
	lat = min_idx - ref
	return _render_footprint_map(
		locations_xy=np.asarray(locations_xy),
		values=lat,
		config=config,
		probe_geometry=probe_geometry,
		title="Template latency",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="footprint_latency_map_png",
		output_key_svg="footprint_latency_map_svg",
		reverse_color_map=True,
	)


def _maybe_reversed_colormap(cmap_name: str, *, reverse: bool) -> str:
	name = str(cmap_name or "viridis").strip() or "viridis"
	if not reverse or name.endswith("_r"):
		return name
	return f"{name}_r"
