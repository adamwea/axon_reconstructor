from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from ..models.inputs import TemplatePlotConfig


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
	used_av_plot = False

	try:
		from axon_velocity.plotting import plot_template as av_plot_template  # type: ignore[import-not-found]

		_ = av_plot_template(template=template_c_by_t, locations=locs, ax=ax)
		used_av_plot = True
	except Exception:
		pass

	if not used_av_plot:
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
