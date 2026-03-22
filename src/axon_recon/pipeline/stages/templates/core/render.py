from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]

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
	sizes = 8.0 + 42.0 * size_norm

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
	sizes = 8.0 + 42.0 * size_norm

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
	vmin = float(np.nanmin(vals)) if vals.size > 0 else 0.0
	vmax = float(np.nanmax(vals)) if vals.size > 0 else 1.0
	if not np.isfinite(vmin):
		vmin = 0.0
	if not np.isfinite(vmax):
		vmax = 1.0
	if vmax <= vmin:
		vmax = vmin + 1.0
	norm = plt.Normalize(vmin=vmin, vmax=vmax)
	colors = cmap(norm(vals))
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
		sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array(vals)
		cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08)
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
	gs = fig.add_gridspec(nrows=n_panels, ncols=2, width_ratios=[2.2, 1.0], hspace=0.35, wspace=0.2)
	trace_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_panels)]
	map_ax = fig.add_subplot(gs[:, 1])

	bg = str(config.background or "").strip().lower()
	if bg == "black":
		fig.patch.set_facecolor("black")
		for ax in trace_axes + [map_ax]:
			ax.set_facecolor("black")
			ax.tick_params(colors="white")
			for spine in ax.spines.values():
				spine.set_color("white")
		trace_color = "white"
		text_color = "white"
	else:
		fig.patch.set_facecolor("white")
		for ax in trace_axes + [map_ax]:
			ax.set_facecolor("white")
		trace_color = "black"
		text_color = "black"

	x = np.arange(int(t.shape[1]), dtype=float)
	base_step = float(max(1e-6, np.max(np.ptp(t[selected, :], axis=1))))
	offset_step = base_step * max(0.2, float(config.trace_spacing))
	trace_gain = float(max(1e-9, float(config.trace_gain)))
	for panel_i, panel_inds in enumerate(panels):
		ax = trace_axes[panel_i]
		for i_local, idx in enumerate(panel_inds):
			ch = int(selected[int(idx)])
			off = float(i_local) * offset_step
			y = (t[ch, :] * trace_gain) + off
			ax.plot(x, y, color=trace_color, linewidth=0.9, alpha=0.95)
			pk = float(np.argmax(np.abs(t[ch, :])))
			ax.scatter([pk], [y[int(pk)]], color="red", s=10)
			if bool(config.show_electrode_ids):
				ax.text(x[0], off, f"ch {int(ch)}", color=text_color, fontsize=6)

		start_idx = int(panel_inds[0])
		end_idx = int(panel_inds[-1])
		ax.set_title(
			f"Propagation traces {start_idx + 1}-{end_idx + 1} / {n_selected}",
			color=text_color,
		)
		ax.set_ylabel("amplitude + offset", color=text_color)
		if panel_i == n_panels - 1:
			ax.set_xlabel("sample", color=text_color)

	if bool(config.latency_map.show):
		from matplotlib.collections import PatchCollection  # type: ignore[import-not-found]
		from matplotlib.patches import Rectangle  # type: ignore[import-not-found]

		map_locs = locs[selected, :2]
		dims = _probe_electrode_dims_um(probe_geometry)
		if dims is None:
			side = _fallback_square_side_um(map_locs)
			dx = dy = float(side)
		else:
			dx, dy = dims
		patches = [
			Rectangle((float(x) - (dx / 2.0), float(y) - (dy / 2.0)), width=float(dx), height=float(dy))
			for x, y in map_locs
		]
		edge_color = "white" if bg == "black" else "black"
		sc = PatchCollection(
			patches,
			cmap=_maybe_reversed_colormap(str(config.latency_map.color_map), reverse=True),
			linewidths=0.25,
			edgecolors=edge_color,
			antialiaseds=False,
		)
		lat_values = np.asarray(lat_idx, dtype=float)
		vmin, vmax = _map_values_to_limits(
			lat_values,
			FootprintMapConfig(
				force_low_value=config.latency_map.force_low_value,
				force_high_value=config.latency_map.force_high_value,
				scale=str(config.latency_map.scale),
				percentile_low=float(config.latency_map.percentile_low),
				percentile_high_linear=float(config.latency_map.percentile_high_linear),
				percentile_high_log=float(config.latency_map.percentile_high_log),
				linear_cap_rounding_mode=str(config.latency_map.linear_cap_rounding_mode),
				linear_cap_rounding_step=float(config.latency_map.linear_cap_rounding_step),
				linear_cap_min_vmax=float(config.latency_map.linear_cap_min_vmax),
			),
		)
		if str(config.latency_map.scale).lower() == "log":
			from matplotlib.colors import LogNorm  # type: ignore[import-not-found]

			vmin_eff = max(1e-9, float(vmin))
			lat_plot = np.clip(lat_values, vmin_eff, None)
			sc.set_norm(LogNorm(vmin=vmin_eff, vmax=max(vmin_eff * 1.0001, float(vmax))))
			sc.set_array(lat_plot)
		else:
			sc.set_array(lat_values)
			sc.set_clim(float(vmin), float(vmax))
		map_ax.add_collection(sc)
		xmin, xmax, ymin, ymax = _compute_plot_limits(map_locs, pad_frac=0.01, pad_abs=max(1.0, float(dx * 0.2)))
		xmin, xmax, ymin, ymax = _expand_limits_for_glyph_half_size(
			xmin=xmin,
			xmax=xmax,
			ymin=ymin,
			ymax=ymax,
			half_dx=float(dx) / 2.0,
			half_dy=float(dy) / 2.0,
		)
		map_ax.set_xlim(xmin, xmax)
		map_ax.set_ylim(ymin, ymax)
		map_ax.set_title(str(config.latency_map.title), color=text_color, fontsize=float(config.latency_map.fontsize))
		if bool(config.latency_map.axes.show):
			map_ax.set_xlabel(
				str(config.latency_map.axes.xlabel),
				color=text_color,
				fontsize=float(config.latency_map.axes.label_fontsize),
			)
			map_ax.set_ylabel(
				str(config.latency_map.axes.ylabel),
				color=text_color,
				fontsize=float(config.latency_map.axes.label_fontsize),
			)
			map_ax.tick_params(labelsize=float(config.latency_map.axes.tick_fontsize), colors=text_color)
		else:
			map_ax.set_xlabel("")
			map_ax.set_ylabel("")
			map_ax.set_xticks([])
			map_ax.set_yticks([])
		if bool(config.latency_map.force_square_aspect):
			map_ax.set_aspect("equal", adjustable="box")
		if bool(config.latency_map.show_color_bar):
			cbar = fig.colorbar(
				sc,
				ax=map_ax,
				fraction=float(config.latency_map.color_bar_length_fraction),
				pad=float(config.latency_map.color_bar_pad_fraction),
			)
			cbar.ax.tick_params(labelsize=float(config.latency_map.color_bar_fontsize), colors=text_color)
			try:
				cbar.outline.set_edgecolor(text_color)
			except Exception:
				pass
	else:
		map_ax.axis("off")

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
	v = np.asarray(values, dtype=float)
	v = v[np.isfinite(v)]
	if v.size == 0:
		return 0.0, 1.0
	vmin = float(np.percentile(v, float(config.percentile_low)))
	high_pct = float(config.percentile_high_log if str(config.scale).lower() == "log" else config.percentile_high_linear)
	vmax = float(np.percentile(v, high_pct))
	if config.force_low_value is not None:
		vmin = float(config.force_low_value)
	if config.force_high_value is not None:
		vmax = float(config.force_high_value)
	if str(config.linear_cap_rounding_mode).lower() == "ceil_step":
		step = max(1e-9, float(config.linear_cap_rounding_step))
		vmax = np.ceil(vmax / step) * step
		vmax = max(vmax, float(config.linear_cap_min_vmax))
	if not np.isfinite(vmin):
		vmin = float(np.min(v))
	if not np.isfinite(vmax):
		vmax = float(np.max(v))
	if vmax <= vmin:
		vmax = vmin + 1.0
	return float(vmin), float(vmax)


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

	norm = None
	if str(config.scale).lower() == "log":
		from matplotlib.colors import LogNorm  # type: ignore[import-not-found]

		vmin_eff = max(1e-9, vmin)
		vals_plot = np.clip(vals, vmin_eff, None)
		norm = LogNorm(vmin=vmin_eff, vmax=max(vmin_eff * 1.0001, vmax))
	else:
		vals_plot = vals

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
		sc.set_clim(vmin, vmax)
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
		cbar = fig.colorbar(sc, ax=ax, fraction=float(config.color_bar_length_fraction), pad=float(config.color_bar_pad_fraction))
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
