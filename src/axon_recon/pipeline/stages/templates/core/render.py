from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from ..models.inputs import (
	FootprintMapGridReportConfig,
	FootprintMapConfig,
	PropagationPlotConfig,
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

	size = np.ones_like(amp, dtype=float)
	if amp.size > 0 and float(np.max(amp)) > 0:
		size = 8.0 + 40.0 * (amp / float(np.max(amp)))
	ax.scatter(locs[:, 0], locs[:, 1], s=size, c=signal_color, alpha=0.9, linewidths=0.0)
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


def _render_topographical_footprint(
	*,
	locations_xy: Any,
	values: np.ndarray,
	config: TopographicalFootprintConfig,
	title: str,
	png_path: Path,
	svg_path: Path,
	output_key_png: str,
	output_key_svg: str,
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

	sc = ax.scatter(
		locs[:, 0],
		locs[:, 1],
		vals,
		c=vals,
		cmap=str(config.color_map),
		s=max(1.0, float(config.marker_size)),
		depthshade=True,
	)
	ax.view_init(elev=float(config.elevation_deg), azim=float(config.azimuth_deg))
	ax.set_xlabel("x (um)", color=text_color)
	ax.set_ylabel("y (um)", color=text_color)
	ax.set_zlabel("value", color=text_color)
	ax.set_title(title, color=text_color)

	if bool(config.show_color_bar):
		cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.08)
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
) -> dict[str, str]:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError("Topographical amplitude footprint requires 2D template")
	amp = np.ptp(t, axis=1)
	return _render_topographical_footprint(
		locations_xy=locations_xy,
		values=amp,
		config=config,
		title="Topographical footprint amplitude",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="topographical_amplitude_footprint_png",
		output_key_svg="topographical_amplitude_footprint_svg",
	)


def render_topographical_latency_footprint(
	*,
	template: Any,
	locations_xy: Any,
	config: TopographicalFootprintConfig,
	png_path: Path,
	svg_path: Path,
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
		title="Topographical footprint latency",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="topographical_latency_footprint_png",
		output_key_svg="topographical_latency_footprint_svg",
	)


def render_propagation_plot(
	*,
	template: Any,
	locations_xy: Any,
	config: PropagationPlotConfig,
	pdf_path: Path,
	png_path: Path,
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

	sc = map_ax.scatter(locs[selected, 0], locs[selected, 1], c=lat_idx, cmap="viridis", s=25)
	map_ax.set_title("Selected channels by peak latency", color=text_color)
	map_ax.set_xlabel("x (um)", color=text_color)
	map_ax.set_ylabel("y (um)", color=text_color)
	map_ax.set_aspect("equal", adjustable="box")
	fig.colorbar(sc, ax=map_ax, fraction=0.046, pad=0.04)

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

	if bool(config.include_mean):
		mean_t = np.mean(selected_templates, axis=0)
		mean_offset = float(len(selected)) * offset_step
		ax.plot(x, mean_t + mean_offset, color=mean_color, linewidth=1.4, alpha=0.95)
		ax.text(
			x[0],
			mean_offset,
			"mean",
			fontsize=6,
			color=mean_color,
			verticalalignment="bottom",
			horizontalalignment="left",
		)

	ax.set_xlabel("sample")
	ax.set_ylabel("amplitude + offset")
	ax.set_title("Template waveforms (top channels)")

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
				("peak-lat", outputs.get("footprint_peak_latency_map_png")),
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


def _render_footprint_map(
	*,
	locations_xy: np.ndarray,
	values: np.ndarray,
	config: FootprintMapConfig,
	title: str,
	png_path: Path,
	svg_path: Path,
	output_key_png: str,
	output_key_svg: str,
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

	sc = ax.scatter(
		locs[:, 0],
		locs[:, 1],
		c=vals_plot,
		cmap=str(config.color_map),
		norm=norm,
		vmin=(None if norm is not None else vmin),
		vmax=(None if norm is not None else vmax),
		s=20,
		linewidths=0.0,
	)
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
) -> dict[str, str]:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError("Amplitude map requires 2D template")
	amp = np.ptp(t, axis=1)
	return _render_footprint_map(
		locations_xy=np.asarray(locations_xy),
		values=amp,
		config=config,
		title="Template footprint amplitude",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="footprint_amplitude_map_png",
		output_key_svg="footprint_amplitude_map_svg",
	)


def render_footprint_peak_latency_map(
	*,
	template: Any,
	locations_xy: Any,
	config: FootprintMapConfig,
	png_path: Path,
	svg_path: Path,
) -> dict[str, str]:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError("Peak latency map requires 2D template")
	peak_idx = np.argmax(np.abs(t), axis=1).astype(float)
	ref = float(peak_idx[int(np.argmax(np.ptp(t, axis=1)))]) if peak_idx.size > 0 else 0.0
	peak_latency = peak_idx - ref
	return _render_footprint_map(
		locations_xy=np.asarray(locations_xy),
		values=peak_latency,
		config=config,
		title="Template peak latency",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="footprint_peak_latency_map_png",
		output_key_svg="footprint_peak_latency_map_svg",
	)


def render_footprint_latency_map(
	*,
	template: Any,
	locations_xy: Any,
	config: FootprintMapConfig,
	png_path: Path,
	svg_path: Path,
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
		title="Template latency",
		png_path=png_path,
		svg_path=svg_path,
		output_key_png="footprint_latency_map_png",
		output_key_svg="footprint_latency_map_svg",
	)
