from __future__ import annotations

import copy
from dataclasses import replace
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any
import xml.etree.ElementTree as ET

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
	TemplateCirclesOverlapControlsConfig,
	TemplateCirclesPlotConfig,
	TemplatePlotTemplatesV2PhaseConfig,
	TemplatePlotV2TextConfig,
	TemplatePlotConfig,
	TemplateWaveformOverlayConfig,
	TopographicalFootprintConfig,
	TimeUpsampleConfig,
	UnitLocationsReportConfig,
	WfOverlayGridReportConfig,
)


LOGGER = logging.getLogger("axon_recon.templates.render")


FAST_RENDER_TEMPLATE_CIRCLES_MAX_DPI = 220.0


def _as_template_channels_by_time(template: Any, n_channels: int) -> np.ndarray:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError(f"Expected 2D template array, got shape={getattr(t, 'shape', None)}")
	if int(t.shape[0]) == int(n_channels):
		return t
	if int(t.shape[1]) == int(n_channels):
		return t.T
	return t


def compute_propagation_channel_order(
	*,
	template_c_by_t: np.ndarray,
	config: PropagationPlotConfig,
	channel_indices: list[int] | np.ndarray | None = None,
) -> dict[str, Any]:
	t = np.asarray(template_c_by_t)
	if t.ndim != 2:
		raise ValueError(f"Propagation ordering requires 2D template array, got shape={getattr(t, 'shape', None)}")

	if channel_indices is not None:
		requested = np.asarray(channel_indices, dtype=int).reshape(-1)
		selected = np.asarray(
			sorted({int(ch) for ch in requested.tolist() if 0 <= int(ch) < int(t.shape[0])}),
			dtype=int,
		)
		if int(selected.shape[0]) == 0:
			raise ValueError("Propagation ordering channel_indices produced an empty selection")
	else:
		ptp_all = np.ptp(t, axis=1)
		top_n = max(1, min(int(config.top_channels), int(t.shape[0])))
		selected = np.argsort(-ptp_all)[:top_n]

	selected_ptp = np.ptp(t[selected, :], axis=1)
	max_ptp_channel = int(selected[int(np.argmax(selected_ptp))]) if selected.size > 0 else None
	selected_negative_peak = np.min(t[selected, :], axis=1)
	max_negative_peak_channel = int(selected[int(np.argmin(selected_negative_peak))]) if selected.size > 0 else None
	selected_abs_max = np.max(np.abs(t[selected, :]), axis=1)
	max_abs_channel = int(selected[int(np.argmax(selected_abs_max))]) if selected.size > 0 else None

	ordering_latency_mode = str(getattr(config, "ordering_latency_mode", "abs_peak") or "abs_peak").strip().lower()
	if ordering_latency_mode == "negative_peak":
		lat_idx = np.argmin(t[selected, :], axis=1).astype(float)
	else:
		lat_idx = np.argmax(np.abs(t[selected, :]), axis=1).astype(float)
	tie_breaker = str(getattr(config, "latency_tie_breaker", "channel_index") or "channel_index").strip().lower()
	if tie_breaker in {"selected_order", "input_order", "stable"}:
		# Stable sort keeps existing selected order for equal-latency ties.
		order = np.argsort(lat_idx, kind="stable")
	else:
		# Default deterministic rule: sort by latency then by channel index.
		order = np.lexsort((selected.astype(float), lat_idx))
	selected = selected[order]
	lat_idx = lat_idx[order]

	anchor_channel: int | None = None
	anchor_shift = 0
	if bool(getattr(config, "force_start_with_max_negative_peak", False)) and max_negative_peak_channel is not None:
		anchor_channel = int(max_negative_peak_channel)
	elif bool(getattr(config, "force_start_with_max_ptp", True)) and max_ptp_channel is not None:
		anchor_channel = int(max_ptp_channel)
	if selected.size > 0 and anchor_channel is not None:
		anchor_pos = np.flatnonzero(selected == int(anchor_channel))
		if anchor_pos.size > 0 and int(anchor_pos[0]) != 0:
			shift = int(anchor_pos[0])
			anchor_shift = int(shift)
			selected = np.roll(selected, -shift)
			lat_idx = np.roll(lat_idx, -shift)

	rank_by_channel = {int(ch): int(i + 1) for i, ch in enumerate(selected.tolist())}
	relative_order_by_channel: dict[int, int] = {}
	if int(selected.shape[0]) > 0:
		n_ch = int(selected.shape[0])
		for i, ch in enumerate(selected.tolist()):
			rel = int(i)
			if anchor_shift > 0 and i >= (n_ch - anchor_shift):
				rel = int(i - n_ch)
			relative_order_by_channel[int(ch)] = rel
	return {
		"ordered_channel_indices": selected,
		"latency_indices": lat_idx,
		"rank_by_channel": rank_by_channel,
		"relative_order_by_channel": relative_order_by_channel,
		"anchor_shift": int(anchor_shift),
		"max_abs_channel": max_abs_channel,
		"max_ptp_channel": max_ptp_channel,
		"max_negative_peak_channel": max_negative_peak_channel,
	}


def _parse_svg_number(raw: str | None) -> float | None:
	if raw is None:
		return None
	m = re.match(r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)", str(raw))
	if m is None:
		return None
	try:
		return float(m.group(1))
	except Exception:
		return None


def _svg_canvas_size(root: ET.Element) -> tuple[float, float]:
	vb = root.attrib.get("viewBox", "")
	parts = [p for p in str(vb).replace(",", " ").split() if p]
	if len(parts) == 4:
		try:
			return float(parts[2]), float(parts[3])
		except Exception:
			pass
	width = _parse_svg_number(root.attrib.get("width", None))
	height = _parse_svg_number(root.attrib.get("height", None))
	if width is None or height is None:
		raise ValueError("Could not infer SVG canvas size from viewBox/width/height")
	return float(width), float(height)


def compose_svg_side_by_side(
	*,
	left_svg_path: Path,
	right_svg_path: Path,
	output_svg_path: Path,
	gap_fraction: float = 0.04,
	right_width_scale: float = 1.0,
) -> Path:
	left_tree = ET.parse(left_svg_path)
	right_tree = ET.parse(right_svg_path)
	left_root = left_tree.getroot()
	right_root = right_tree.getroot()

	left_w, left_h = _svg_canvas_size(left_root)
	right_w, right_h = _svg_canvas_size(right_root)
	if left_w <= 0 or left_h <= 0 or right_w <= 0 or right_h <= 0:
		raise ValueError("SVG canvas dimensions must be positive")

	gap = float(max(0.0, float(gap_fraction)) * left_w)
	scale = (left_h / right_h) * float(max(0.01, float(right_width_scale)))
	right_w_scaled = right_w * scale
	right_h_scaled = right_h * scale

	total_w = left_w + gap + right_w_scaled
	total_h = max(left_h, right_h_scaled)
	left_y = 0.5 * (total_h - left_h)
	right_y = 0.5 * (total_h - right_h_scaled)
	right_x = left_w + gap

	svg_ns = "http://www.w3.org/2000/svg"
	ET.register_namespace("", svg_ns)
	composed = ET.Element(f"{{{svg_ns}}}svg", {
		"version": "1.1",
		"width": str(total_w),
		"height": str(total_h),
		"viewBox": f"0 0 {total_w} {total_h}",
	})

	left_group = ET.SubElement(composed, f"{{{svg_ns}}}g", {"transform": f"translate(0,{left_y})"})
	for child in list(left_root):
		if str(child.tag).endswith("defs"):
			continue
		left_group.append(copy.deepcopy(child))

	right_group = ET.SubElement(
		composed,
		f"{{{svg_ns}}}g",
		{"transform": f"translate({right_x},{right_y}) scale({scale})"},
	)
	for child in list(right_root):
		if str(child.tag).endswith("defs"):
			continue
		right_group.append(copy.deepcopy(child))

	output_svg_path.parent.mkdir(parents=True, exist_ok=True)
	ET.ElementTree(composed).write(output_svg_path, encoding="utf-8", xml_declaration=True)
	return output_svg_path


def compose_svg_grid(
	*,
	panel_svg_paths: list[Path],
	output_svg_path: Path,
	ncols: int,
	show_title: bool = True,
	title: str = "",
) -> Path:
	if not panel_svg_paths:
		raise ValueError("compose_svg_grid requires at least one panel SVG")

	paths = [Path(p) for p in panel_svg_paths]
	n = len(paths)
	ncols_eff = max(1, int(ncols))
	nrows_eff = int(np.ceil(float(n) / float(ncols_eff)))

	panels: list[tuple[ET.Element, float, float]] = []
	for path in paths:
		tree = ET.parse(path)
		root = tree.getroot()
		w, h = _svg_canvas_size(root)
		if w <= 0 or h <= 0:
			raise ValueError(f"SVG panel dimensions must be positive: {path}")
		panels.append((root, float(w), float(h)))

	cell_w = float(max(p[1] for p in panels))
	cell_h = float(max(p[2] for p in panels))
	title_h = 24.0 if bool(show_title) and bool(str(title).strip()) else 0.0
	total_w = cell_w * float(ncols_eff)
	total_h = title_h + (cell_h * float(nrows_eff))

	svg_ns = "http://www.w3.org/2000/svg"
	ET.register_namespace("", svg_ns)
	composed = ET.Element(
		f"{{{svg_ns}}}svg",
		{
			"version": "1.1",
			"width": str(total_w),
			"height": str(total_h),
			"viewBox": f"0 0 {total_w} {total_h}",
		},
	)

	ET.SubElement(
		composed,
		f"{{{svg_ns}}}rect",
		{
			"x": "0",
			"y": "0",
			"width": str(total_w),
			"height": str(total_h),
			"fill": "white",
		},
	)

	if title_h > 0.0:
		title_node = ET.SubElement(
			composed,
			f"{{{svg_ns}}}text",
			{
				"x": str(0.5 * total_w),
				"y": "16",
				"text-anchor": "middle",
				"font-size": "12",
				"fill": "black",
			},
		)
		title_node.text = str(title)

	for i, (panel_root, panel_w, panel_h) in enumerate(panels):
		row = int(i // ncols_eff)
		col = int(i % ncols_eff)
		x = (float(col) * cell_w) + ((cell_w - panel_w) * 0.5)
		y = title_h + (float(row) * cell_h) + ((cell_h - panel_h) * 0.5)
		group = ET.SubElement(composed, f"{{{svg_ns}}}g", {"transform": f"translate({x},{y})"})
		for child in list(panel_root):
			if str(child.tag).endswith("defs"):
				continue
			group.append(copy.deepcopy(child))

	output_svg_path.parent.mkdir(parents=True, exist_ok=True)
	ET.ElementTree(composed).write(output_svg_path, encoding="utf-8", xml_declaration=True)
	return output_svg_path


def compose_png_side_by_side(
	*,
	left_png_path: Path,
	right_png_path: Path,
	output_png_path: Path,
	gap_fraction: float = 0.04,
	right_width_scale: float = 1.0,
	output_dpi: float | None = None,
) -> Path:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	left_img = np.asarray(plt.imread(left_png_path))
	right_img = np.asarray(plt.imread(right_png_path))
	if left_img.ndim != 3 or right_img.ndim != 3:
		raise ValueError("Expected RGB/RGBA PNG images for side-by-side composition")

	left_h, left_w = int(left_img.shape[0]), int(left_img.shape[1])
	right_h, right_w = int(right_img.shape[0]), int(right_img.shape[1])
	if left_h <= 0 or left_w <= 0 or right_h <= 0 or right_w <= 0:
		raise ValueError("Invalid PNG dimensions for side-by-side composition")

	right_display_w = (float(right_w) * (float(left_h) / float(right_h))) * float(max(0.01, float(right_width_scale)))
	gap_px = float(max(0.0, float(gap_fraction)) * float(left_w))

	total_w = float(left_w) + gap_px + right_display_w
	total_h = float(left_h)
	fig_h_in = 3.0
	fig_w_in = max(1.0, fig_h_in * (total_w / max(1.0, total_h)))

	left_frac = float(left_w) / total_w
	gap_frac = gap_px / total_w
	right_frac = max(1e-6, 1.0 - left_frac - gap_frac)

	render_dpi = max(72.0, float(left_h) / float(fig_h_in))
	if output_dpi is not None:
		render_dpi = max(72.0, float(output_dpi))
	fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=render_dpi)
	gs = fig.add_gridspec(1, 3, width_ratios=[left_frac, gap_frac, right_frac], wspace=0.0)
	ax_left = fig.add_subplot(gs[0, 0])
	ax_gap = fig.add_subplot(gs[0, 1])
	ax_right = fig.add_subplot(gs[0, 2])

	fig.patch.set_facecolor("white")
	ax_left.imshow(left_img, interpolation="none")
	ax_right.imshow(right_img, interpolation="none", aspect="auto")
	ax_gap.set_facecolor("white")
	for ax in (ax_left, ax_gap, ax_right):
		ax.set_axis_off()

	output_png_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_png_path, dpi=render_dpi, bbox_inches="tight", pad_inches=0.0, facecolor=fig.get_facecolor())
	plt.close(fig)
	return output_png_path


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


def _compute_plot_limits_with_padding_percent(
	points_xy: np.ndarray,
	*,
	padding_percent: float,
	flat_pad_abs: float = 10.0,
) -> tuple[float, float, float, float]:
	xs = points_xy[:, 0]
	ys = points_xy[:, 1]
	xmin = float(np.min(xs))
	xmax = float(np.max(xs))
	ymin = float(np.min(ys))
	ymax = float(np.max(ys))

	dx = max(float(xmax - xmin), 0.0)
	dy = max(float(ymax - ymin), 0.0)
	pad_frac = float(max(0.0, padding_percent) / 100.0)
	px = float(pad_frac * dx)
	py = float(pad_frac * dy)

	if dx == 0.0:
		px = max(px, float(flat_pad_abs))
	if dy == 0.0:
		py = max(py, float(flat_pad_abs))

	return xmin - px, xmax + px, ymin - py, ymax + py


def _recenter_limits_around_point(
	xmin: float,
	xmax: float,
	ymin: float,
	ymax: float,
	*,
	center_xy: tuple[float, float],
) -> tuple[float, float, float, float]:
	cx = float(center_xy[0])
	cy = float(center_xy[1])
	half_x = max(abs(cx - float(xmin)), abs(float(xmax) - cx))
	half_y = max(abs(cy - float(ymin)), abs(float(ymax) - cy))
	return cx - half_x, cx + half_x, cy - half_y, cy + half_y


def _compute_max_non_overlapping_circle_areas(
	*,
	centers_display_pt: np.ndarray,
	base_areas_pt2: np.ndarray,
	axis_x_limits_pt: tuple[float, float],
	axis_y_limits_pt: tuple[float, float],
	overlap_tolerance_pt: float = 0.0,
) -> np.ndarray:
	"""Scale circle areas to maximize size while preventing overlaps."""
	centers = np.asarray(centers_display_pt, dtype=float)
	areas = np.asarray(base_areas_pt2, dtype=float)
	if int(centers.shape[0]) == 0 or int(areas.size) == 0:
		return np.asarray(areas, dtype=float)

	areas = np.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
	areas = np.clip(areas, 0.0, None)
	if centers.ndim != 2 or int(centers.shape[1]) < 2:
		raise ValueError(f"Expected centers_display_pt shape (n,2+), got {getattr(centers, 'shape', None)}")
	centers = centers[:, :2]

	xmin_pt, xmax_pt = float(min(axis_x_limits_pt)), float(max(axis_x_limits_pt))
	ymin_pt, ymax_pt = float(min(axis_y_limits_pt)), float(max(axis_y_limits_pt))

	r_base_pt = np.sqrt(np.clip(areas, 0.0, None) / np.pi)
	tol = float(max(0.0, overlap_tolerance_pt))
	positive_mask = r_base_pt > 0.0
	min_constraint = float("inf")

	if np.any(positive_mask):
		centers_pos = centers[positive_mask, :]
		r_base_pos = r_base_pt[positive_mask]

		edge_clearances = np.min(
			np.column_stack(
				[
					centers_pos[:, 0] - xmin_pt,
					xmax_pt - centers_pos[:, 0],
					centers_pos[:, 1] - ymin_pt,
					ymax_pt - centers_pos[:, 1],
				]
			),
			axis=1,
		)
		valid_edge = np.isfinite(edge_clearances) & (edge_clearances > 0.0)
		if np.any(valid_edge):
			min_constraint = min(
				min_constraint,
				float(np.min(edge_clearances[valid_edge] / r_base_pos[valid_edge])),
			)

		n_pos = int(centers_pos.shape[0])
		if n_pos > 1:
			# Vectorize the common case to avoid Python-loop overhead on dense units,
			# but keep a loop fallback for unusually large channel counts.
			if n_pos <= 2048:
				dx = np.subtract.outer(centers_pos[:, 0], centers_pos[:, 0])
				dy = np.subtract.outer(centers_pos[:, 1], centers_pos[:, 1])
				distances = np.hypot(dx, dy)
				r_sum = np.add.outer(r_base_pos, r_base_pos)
				upper = np.triu_indices(n_pos, k=1)
				pair_constraints = (distances[upper] - tol) / r_sum[upper]
				finite_pairs = pair_constraints[np.isfinite(pair_constraints)]
				if finite_pairs.size > 0:
					min_constraint = min(min_constraint, float(np.min(finite_pairs)))
			else:
				for i in range(n_pos):
					xi = float(centers_pos[i, 0])
					yi = float(centers_pos[i, 1])
					ri = float(r_base_pos[i])
					for j in range(i + 1, n_pos):
						r_sum = ri + float(r_base_pos[j])
						if r_sum <= 0.0:
							continue
						d = float(np.hypot(float(centers_pos[j, 0]) - xi, float(centers_pos[j, 1]) - yi)) - tol
						if np.isfinite(d):
							min_constraint = min(min_constraint, float(d / r_sum))

	s_radius = 1.0 if not np.isfinite(min_constraint) else float(min_constraint)
	s_radius = float(max(0.0, s_radius))

	return np.asarray(areas * (s_radius ** 2), dtype=float)


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
		half = float(side / 2.0)
	else:
		xmin, xmax, ymin, ymax = _recenter_limits_around_point(
			xmin,
			xmax,
			ymin,
			ymax,
			center_xy=center_xy,
		)
		cx = float((xmin + xmax) / 2.0)
		cy = float((ymin + ymax) / 2.0)
		half = float(max(float(xmax - xmin), float(ymax - ymin)) / 2.0)
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
	if bool(getattr(config, "invert_y_axis", True)):
		ax.invert_yaxis()


def _add_scale_bar(ax: Any, *, config: TemplatePlotConfig) -> None:
	if not bool(config.show_scale_bar):
		return
	x0, x1 = ax.get_xlim()
	y0, y1 = ax.get_ylim()
	span_x = max(1.0, float(abs(x1 - x0)))
	span_y = max(1.0, float(abs(y1 - y0)))
	x_dir = 1.0 if float(x1) >= float(x0) else -1.0

	if config.scale_bar_length_um is None:
		bar = 100.0 if span_x >= 180.0 else 50.0
	else:
		bar = max(1.0, float(config.scale_bar_length_um))
	label = f"{int(round(bar))} um"

	x_offset_frac = getattr(config, "scale_bar_x_offset_frac", None)
	h_align = str(getattr(config, "scale_bar_horizontal_alignment", "right") or "right").strip().lower()
	v_align = str(getattr(config, "scale_bar_vertical_alignment", "bottom") or "bottom").strip().lower()
	if h_align not in {"left", "center", "right"}:
		h_align = "right"
	if v_align not in {"bottom", "center", "top"}:
		v_align = "bottom"

	extra_pad_x = 0.0
	if bool(getattr(config, "scale_bar_x_offset_considers_fontsize", False)):
		try:
			fig = ax.figure
			if fig is not None and fig.canvas is not None:
				fig.canvas.draw()
				renderer = fig.canvas.get_renderer()
				tmp_text = ax.text(
					0.0,
					0.0,
					label,
					fontsize=float(config.scale_bar_fontsize),
					alpha=0.0,
				)
				bbox = tmp_text.get_window_extent(renderer=renderer)
				tmp_text.remove()
				x_min = float(min(x0, x1))
				x_max = float(max(x0, x1))
				dx = float(max(1e-9, x_max - x_min))
				px0 = float(ax.transData.transform((x_min, 0.0))[0])
				px1 = float(ax.transData.transform((x_max, 0.0))[0])
				px_per_um = float(max(1e-9, abs(px1 - px0) / dx))
				half_label_w_um = 0.5 * float(max(0.0, bbox.width)) / px_per_um
				extra_pad_x = float(max(0.0, half_label_w_um - (0.5 * bar)))
		except Exception:
			extra_pad_x = 0.0

	if x_offset_frac is None:
		margin = float(config.scale_bar_y_offset_frac) * span_x
		if h_align == "left":
			x_left = float(x0) + (x_dir * margin)
			x_right = x_left + (x_dir * bar)
		elif h_align == "center":
			x_center = 0.5 * (float(x0) + float(x1))
			x_left = x_center - (0.5 * x_dir * bar)
			x_right = x_center + (0.5 * x_dir * bar)
		else:
			x_right = float(x1) - (x_dir * margin)
			x_left = x_right - (x_dir * bar)
	else:
		x_margin = float(max(0.0, float(x_offset_frac))) * span_x
		if h_align == "right":
			x_right = float(x1) - (x_dir * x_margin)
			x_left = x_right - (x_dir * bar)
		elif h_align == "center":
			x_center = 0.5 * (float(x0) + float(x1))
			x_left = x_center - (0.5 * x_dir * bar)
			x_right = x_center + (0.5 * x_dir * bar)
		else:
			x_left = float(x0) + (x_dir * x_margin)
			x_right = x_left + (x_dir * bar)
	if extra_pad_x > 0.0:
		if h_align == "right":
			x_left -= (x_dir * extra_pad_x)
		elif h_align == "left":
			x_left += (x_dir * extra_pad_x)
			x_right = x_left + (x_dir * bar)
	x_low = float(min(x_left, x_right))
	x_high = float(max(x_left, x_right))
	x_min = float(min(x0, x1))
	x_max = float(max(x0, x1))
	if x_low < x_min:
		delta = x_min - x_low
		x_left += delta
		x_right += delta
	if x_high > x_max:
		delta = x_high - x_max
		x_left -= delta
		x_right -= delta

	from matplotlib.transforms import blended_transform_factory  # type: ignore[import-not-found]

	y_margin_frac = float(min(1.0, max(0.0, float(config.scale_bar_y_offset_frac))))
	text_offset_frac = float(min(1.0, max(0.0, float(config.scale_bar_text_offset_frac))))
	if v_align == "top":
		y_bar = 1.0 - y_margin_frac
		label_y = max(0.0, y_bar - text_offset_frac)
		label_va = "top"
	elif v_align == "center":
		y_bar = 0.5
		label_y = min(1.0, y_bar + text_offset_frac)
		label_va = "bottom"
	else:
		y_bar = y_margin_frac
		label_y = min(1.0, y_bar + text_offset_frac)
		label_va = "bottom"

	if min(float(x_left), float(x_right)) <= x_min:
		fallback_margin = float(config.scale_bar_y_offset_frac) * span_x
		x_left = float(x0) + (x_dir * fallback_margin)
		x_right = x_left + (x_dir * bar)
	if max(float(x_left), float(x_right)) >= x_max:
		x_right = float(x1) - (x_dir * 1.0)
		x_left = x_right - (x_dir * bar)

	transform = blended_transform_factory(ax.transData, ax.transAxes)
	(line,) = ax.plot(
		[x_left, x_right],
		[y_bar, y_bar],
		color=str(config.scale_bar_color),
		lw=float(config.scale_bar_linewidth),
		solid_capstyle="butt",
		transform=transform,
	)
	line.set_gid("template_scale_bar_line")
	text = ax.text(
		(x_left + x_right) / 2.0,
		label_y,
		label,
		transform=transform,
		color=str(config.scale_bar_color),
		horizontalalignment="center",
		verticalalignment=label_va,
		fontsize=float(config.scale_bar_fontsize),
	)
	text.set_gid("template_scale_bar_text")


def _add_scale_circle(
	ax: Any,
	*,
	config: TemplatePlotConfig,
	reference_area_pt2: float,
	reference_value: float,
) -> None:
	if not bool(getattr(config, "show_scale_circle", False)):
		return
	scale_cfg = getattr(config, "scale_circle", None)
	if scale_cfg is None:
		return

	area_pt2 = float(max(0.0, reference_area_pt2))
	if area_pt2 <= 0.0:
		return

	fig = ax.figure
	if fig is None:
		return
	try:
		fig.canvas.draw()
	except Exception:
		pass

	bbox = ax.get_window_extent()
	bbox_w = float(max(1.0, bbox.width))
	bbox_h = float(max(1.0, bbox.height))
	radius_pt = float(np.sqrt(area_pt2 / np.pi))
	radius_px = radius_pt * float(fig.dpi / 72.0)
	rx_axes = float(max(1e-6, radius_px / bbox_w))
	ry_axes = float(max(1e-6, radius_px / bbox_h))

	ha = str(getattr(scale_cfg, "horizontal_alignment", "left") or "left").strip().lower()
	va = str(getattr(scale_cfg, "vertical_alignment", "top") or "top").strip().lower()
	x_off = float(max(0.0, float(getattr(scale_cfg, "x_offset_frac", 0.02))))
	y_off = float(max(0.0, float(getattr(scale_cfg, "y_offset_frac", 0.02))))

	if ha == "right":
		cx = 1.0 - x_off - rx_axes
	elif ha == "center":
		cx = 0.5
	else:
		cx = x_off + rx_axes

	if va == "bottom":
		cy = y_off + ry_axes
	elif va == "center":
		cy = 0.5
	else:
		cy = 1.0 - y_off - ry_axes

	cx = float(min(1.0 - rx_axes, max(rx_axes, cx)))
	cy = float(min(1.0 - ry_axes, max(ry_axes, cy)))

	from matplotlib.patches import Ellipse  # type: ignore[import-not-found]
	line_color = str(getattr(config, "scale_circle_color", "white") or "white")
	raw_linestyle = getattr(scale_cfg, "linestyle", "solid")
	linestyle = None if raw_linestyle is None else str(raw_linestyle).strip()
	if linestyle is not None and linestyle.lower() in {"", "none", "null"}:
		linestyle = None
	linewidth = float(max(0.1, float(getattr(scale_cfg, "linewidth", 1.8))))
	fill_enabled = bool(getattr(scale_cfg, "fill", False))
	raw_fill_color = getattr(scale_cfg, "fill_color", None)
	fill_color = None if raw_fill_color is None else str(raw_fill_color).strip()
	if not fill_color:
		fill_color = line_color

	patch = Ellipse(
		(cx, cy),
		width=2.0 * rx_axes,
		height=2.0 * ry_axes,
		transform=ax.transAxes,
		fill=fill_enabled,
		facecolor=(str(fill_color) if fill_enabled else "none"),
		edgecolor=(line_color if linestyle is not None else "none"),
		linewidth=(linewidth if linestyle is not None else 0.0),
		linestyle=(linestyle if linestyle is not None else "none"),
	)
	patch.set_gid("template_scale_circle_patch")
	ax.add_patch(patch)

	units = str(getattr(scale_cfg, "units", "uV") or "").strip()
	digits = int(max(0, int(getattr(scale_cfg, "digits_after_decimal", 0))))
	if digits <= 0:
		value_text = _format_no_sci(reference_value, max_decimals=0)
	else:
		value_text = f"{float(reference_value):.{digits}f}"
	label = f"{value_text} {units}" if units else value_text

	font_loc = str(getattr(scale_cfg, "font_location", "inside") or "inside").strip().lower()
	fallback_loc = str(getattr(scale_cfg, "font_location_circle_too_small", "below") or "below").strip().lower()
	if font_loc == "outside":
		font_loc = "right" if ha in {"left", "center"} else "left"
	circle_width_frac = float(2.0 * rx_axes)
	if font_loc == "inside" and circle_width_frac < 0.08:
		font_loc = fallback_loc

	pad = 0.012
	text_ha = "center"
	if font_loc == "above":
		tx, ty, text_va = cx, cy + ry_axes + pad, "bottom"
	elif font_loc == "below":
		tx, ty, text_va = cx, cy - ry_axes - pad, "top"
	elif font_loc == "left":
		tx, ty, text_va = cx - rx_axes - pad, cy, "center"
		text_ha = "right"
	elif font_loc == "right":
		tx, ty, text_va = cx + rx_axes + pad, cy, "center"
		text_ha = "left"
	else:
		tx, ty, text_va = cx, cy, "center"

	text = ax.text(
		tx,
		ty,
		label,
		transform=ax.transAxes,
		horizontalalignment=text_ha,
		verticalalignment=text_va,
		fontsize=float(max(1.0, float(getattr(scale_cfg, "fontsize", 6.0)))),
		color=line_color,
	)
	text.set_gid("template_scale_circle_text")


def _resolve_fast_render_template_circles_config(config: TemplateCirclesPlotConfig) -> TemplateCirclesPlotConfig:
	if not bool(getattr(config, "fast_render", False)):
		return config
	overlap_cfg = getattr(config, "overlap_controls", None)
	if overlap_cfg is None:
		overlap_cfg = TemplateCirclesOverlapControlsConfig()
	return replace(
		config,
		dpi=min(float(getattr(config, "dpi", 300.0)), FAST_RENDER_TEMPLATE_CIRCLES_MAX_DPI),
		show_scale_circle=False,
		overlap_controls=replace(
			overlap_cfg,
			scalebar_coords_overlap_detect=False,
			scalebar_colorbar_overlap_detect=False,
			unitid_label_channel_overlap_detect=False,
			coords_channel_overlap_detect=False,
			scalebar_channel_overlap_detect=False,
			scalecircle_channel_overlap_detect=False,
			max_overlap_check_iterations=0,
		),
	)


def _normalize_branch_morphology_payload(branch_morphology: Any) -> list[dict[str, Any]]:
	if branch_morphology is None:
		return []
	payload = branch_morphology
	if isinstance(payload, dict):
		payload = payload.get("branches", [])
	if not isinstance(payload, (list, tuple)):
		return []
	normalized: list[dict[str, Any]] = []
	for idx, item in enumerate(payload):
		if not isinstance(item, dict):
			continue
		raw_nodes = item.get("node_indices", item.get("channels", item.get("nodes", [])))
		if not isinstance(raw_nodes, (list, tuple, np.ndarray)):
			continue
		nodes: list[int] = []
		for v in raw_nodes:
			try:
				nodes.append(int(v))
			except Exception:
				continue
		if len(nodes) == 0:
			continue
		raw_edges = item.get("edges", None)
		edges: list[tuple[int, int]] = []
		if isinstance(raw_edges, (list, tuple)):
			for e in raw_edges:
				if not isinstance(e, (list, tuple)) or len(e) < 2:
					continue
				try:
					edges.append((int(e[0]), int(e[1])))
				except Exception:
					continue
		if len(edges) == 0:
			edges = [(int(nodes[i]), int(nodes[i + 1])) for i in range(max(0, len(nodes) - 1))]
		normalized.append(
			{
				"branch_index": int(item.get("branch_index", item.get("id", idx))),
				"node_indices": nodes,
				"edges": edges,
				"label": item.get("label", item.get("branch_index", item.get("id", idx))),
				"color": item.get("color", None),
			}
		)
	return normalized


def _as_int_list(values: Any) -> list[int]:
	if not isinstance(values, (list, tuple, np.ndarray)):
		return []
	out: list[int] = []
	for v in values:
		try:
			out.append(int(v))
		except Exception:
			continue
	return out


def _preferred_branch_ids(branch_like: Any) -> list[int]:
	if not isinstance(branch_like, dict):
		return []
	for key in ("electrode_ids", "channels", "node_indices", "nodes"):
		vals = _as_int_list(branch_like.get(key, []))
		if vals:
			return vals
	return []


def _branch_morphology_from_gtr(gtr: Any) -> list[dict[str, Any]]:
	if gtr is None:
		return []

	out: list[dict[str, Any]] = []
	branches = getattr(gtr, "branches", None)
	if isinstance(branches, (list, tuple)):
		for bi, branch in enumerate(branches):
			if not isinstance(branch, dict):
				continue
			channels = _preferred_branch_ids(branch)
			if len(channels) < 2:
				continue
			out.append(
				{
					"branch_index": int(branch.get("branch_index", bi)),
					"channels": channels,
					"label": branch.get("branch_index", bi),
				}
			)
		if out:
			return out

	raw_paths = getattr(gtr, "_paths_raw", None)
	if isinstance(raw_paths, (list, tuple)):
		for raw_idx, raw_path in enumerate(raw_paths):
			channels = _as_int_list(list(raw_path)[::-1])
			if len(channels) < 2:
				continue
			out.append({"branch_index": int(raw_idx), "channels": channels, "label": int(raw_idx)})

	return out


def _draw_branch_morphology_overlay(
	*,
	ax: Any,
	fig: Any,
	sc: Any,
	locations_xy: np.ndarray,
	branch_morphology: Any,
	branch_cfg: Any,
) -> None:
	branches = _normalize_branch_morphology_payload(branch_morphology)
	if len(branches) == 0:
		return

	offsets = np.asarray(sc.get_offsets(), dtype=float)
	sizes_pt2 = np.asarray(sc.get_sizes(), dtype=float)
	if offsets.ndim != 2 or sizes_pt2.ndim != 1 or int(offsets.shape[0]) != int(sizes_pt2.shape[0]):
		return

	centers_px = np.asarray(ax.transData.transform(offsets), dtype=float)
	radii_pt = np.sqrt(np.clip(sizes_pt2, 0.0, None) / np.pi)
	radii_px = radii_pt * float(fig.dpi / 72.0)
	n_channels = int(offsets.shape[0])

	unique_color_per_branch = bool(getattr(branch_cfg, "unique_color_per_branch", True))
	show_branch_labels = bool(getattr(branch_cfg, "show_branch_labels", False))
	show_branch_legend = bool(getattr(branch_cfg, "show_branch_legend", False))
	edge_lw = float(max(0.0, float(getattr(branch_cfg, "edge_linewidth", 0.8))))
	node_lw = float(max(0.0, float(getattr(branch_cfg, "node_border_linewidth", 0.35))))
	node_outline_color_raw = str(getattr(branch_cfg, "node_outline_color", "") or "").strip()
	node_outline_color = node_outline_color_raw if node_outline_color_raw else None
	node_outline_lw = float(max(0.0, float(getattr(branch_cfg, "node_outline_linewidth", 0.0))))
	branch_outline_color_raw = str(getattr(branch_cfg, "branch_outline_color", "") or "").strip()
	branch_outline_color = branch_outline_color_raw if branch_outline_color_raw else None
	branch_outline_lw = float(max(0.0, float(getattr(branch_cfg, "branch_outline_linewidth", 0.0))))
	color_scheme = str(getattr(branch_cfg, "color_scheme", "tab20") or "tab20")
	from matplotlib import colormaps  # type: ignore[import-not-found]

	cmap = colormaps.get_cmap(color_scheme)

	branch_colors: list[Any] = []
	for i, branch in enumerate(branches):
		override = branch.get("color", None)
		if override is not None:
			branch_colors.append(override)
			continue
		if unique_color_per_branch:
			den = max(1, len(branches) - 1)
			branch_colors.append(cmap(float(i) / float(den)))
		else:
			branch_colors.append(cmap(0.0))

	node_color_by_index: dict[int, Any] = {}
	for i, branch in enumerate(branches):
		color = branch_colors[i]
		for ch in branch["node_indices"]:
			if 0 <= int(ch) < n_channels and int(ch) not in node_color_by_index:
				node_color_by_index[int(ch)] = color

	if node_lw > 0.0 and len(node_color_by_index) > 0:
		node_inds = np.asarray(sorted(node_color_by_index.keys()), dtype=int)
		inner_r_pt = np.maximum(np.sqrt(np.clip(sizes_pt2[node_inds], 0.0, None) / np.pi) - node_lw, 0.0)
		inner_sizes = np.pi * np.square(inner_r_pt)
		node_colors = [node_color_by_index[int(ch)] for ch in node_inds.tolist()]
		if node_outline_color is not None and node_outline_lw > 0.0:
			outline_coll = ax.scatter(
				offsets[node_inds, 0],
				offsets[node_inds, 1],
				s=inner_sizes,
				facecolors="none",
				edgecolors=[node_outline_color for _ in node_inds.tolist()],
				linewidths=(float(node_lw) + (2.0 * float(node_outline_lw))),
				zorder=7.15,
			)
			outline_coll.set_gid("template_branch_node_outline")
		node_coll = ax.scatter(
			offsets[node_inds, 0],
			offsets[node_inds, 1],
			s=inner_sizes,
			facecolors="none",
			edgecolors=node_colors,
			linewidths=node_lw,
			zorder=7.2,
		)
		node_coll.set_gid("template_branch_node_borders")

	if edge_lw > 0.0:
		inv = ax.transData.inverted()
		for i, branch in enumerate(branches):
			color = branch_colors[i]
			for a, b in branch["edges"]:
				a_idx = int(a)
				b_idx = int(b)
				if not (0 <= a_idx < n_channels and 0 <= b_idx < n_channels):
					continue
				if a_idx == b_idx:
					continue
				ca = centers_px[a_idx, :2].astype(float)
				cb = centers_px[b_idx, :2].astype(float)
				vec = cb - ca
				d = float(np.hypot(float(vec[0]), float(vec[1])))
				if d <= 1e-6:
					continue
				ra = float(max(0.0, radii_px[a_idx]))
				rb = float(max(0.0, radii_px[b_idx]))
				if d <= float(ra + rb):
					continue
				u = vec / d
				start_px = ca + (u * ra)
				end_px = cb - (u * rb)
				start_xy = inv.transform(start_px)
				end_xy = inv.transform(end_px)
				if branch_outline_color is not None and branch_outline_lw > 0.0:
					(outline_line,) = ax.plot(
						[float(start_xy[0]), float(end_xy[0])],
						[float(start_xy[1]), float(end_xy[1])],
						color=branch_outline_color,
						linewidth=(float(edge_lw) + (2.0 * float(branch_outline_lw))),
						solid_capstyle="round",
						zorder=7.05,
					)
					outline_line.set_gid("template_branch_edge_outline")
				(line,) = ax.plot(
					[float(start_xy[0]), float(end_xy[0])],
					[float(start_xy[1]), float(end_xy[1])],
					color=color,
					linewidth=edge_lw,
					solid_capstyle="round",
					zorder=7.1,
				)
				line.set_gid("template_branch_edge")

	if show_branch_labels:
		for i, branch in enumerate(branches):
			pts = [int(ch) for ch in branch["node_indices"] if 0 <= int(ch) < n_channels]
			if len(pts) == 0:
				continue
			xy = np.asarray(locations_xy[pts, :2], dtype=float)
			x = float(np.nanmean(xy[:, 0]))
			y = float(np.nanmean(xy[:, 1]))
			text = ax.text(
				x,
				y,
				str(branch.get("label", branch.get("branch_index", i))),
				color=branch_colors[i],
				fontsize=7.0,
				horizontalalignment="center",
				verticalalignment="center",
				zorder=7.3,
			)
			text.set_gid("template_branch_label")

	if show_branch_legend:
		from matplotlib.lines import Line2D  # type: ignore[import-not-found]

		face_rgba = ax.get_facecolor()
		luminance = (0.2126 * float(face_rgba[0])) + (0.7152 * float(face_rgba[1])) + (0.0722 * float(face_rgba[2]))
		text_color = "black" if luminance > 0.5 else "white"
		frame_facecolor = (1.0, 1.0, 1.0, 0.9) if luminance > 0.5 else (0.0, 0.0, 0.0, 0.85)
		handles: list[Any] = []
		for i, branch in enumerate(branches):
			branch_label = branch.get("label", branch.get("branch_index", i))
			handles.append(
				Line2D(
					[0.0],
					[0.0],
					color=branch_colors[i],
					linewidth=max(1.0, edge_lw),
					marker="o",
					markersize=max(4.0, 4.0 + (2.0 * node_lw)),
					markerfacecolor="none",
					markeredgewidth=max(0.75, node_lw),
					markeredgecolor=branch_colors[i],
					label=str(branch_label),
				)
			)
		if handles:
			legend = ax.legend(
				handles=handles,
				title="Branches",
				loc="center right",
				frameon=True,
				fancybox=True,
				framealpha=0.9,
				fontsize=6.0,
				title_fontsize=7.0,
				borderpad=0.35,
				handlelength=1.6,
				handletextpad=0.5,
			)
			legend.set_zorder(7.4)
			legend.get_frame().set_facecolor(frame_facecolor)
			legend.get_frame().set_edgecolor(text_color)
			legend.get_frame().set_linewidth(0.6)
			legend.get_title().set_color(text_color)
			for text in legend.get_texts():
				text.set_color(text_color)
			legend.set_gid("template_branch_legend")


def _axes_anchor_pos(
	*,
	x_offset_frac: float,
	y_offset_frac: float,
	horizontal_alignment: str,
	vertical_alignment: str,
) -> tuple[float, float]:
	ha = str(horizontal_alignment or "right").strip().lower()
	va = str(vertical_alignment or "top").strip().lower()
	x = 1.0 - float(x_offset_frac) if ha == "right" else (0.5 if ha == "center" else float(x_offset_frac))
	y = 1.0 - float(y_offset_frac) if va == "top" else (0.5 if va == "center" else float(y_offset_frac))
	return float(x), float(y)


def _add_unit_id_label(ax: Any, *, config: TemplatePlotConfig, unit_id: Any | None) -> None:
	label_cfg = getattr(config, "unit_id_label", None)
	if not bool(getattr(label_cfg, "show", False)):
		return
	if unit_id is None:
		return
	ha = str(getattr(label_cfg, "horizontal_alignment", "right") or "right").strip().lower()
	va = str(getattr(label_cfg, "vertical_alignment", "top") or "top").strip().lower()
	x, y = _axes_anchor_pos(
		x_offset_frac=float(getattr(label_cfg, "x_offset_frac", 0.02)),
		y_offset_frac=float(getattr(label_cfg, "y_offset_frac", 0.02)),
		horizontal_alignment=ha,
		vertical_alignment=va,
	)
	text = ax.text(
		x,
		y,
		f"unit {unit_id}",
		transform=ax.transAxes,
		ha=ha,
		va=va,
		fontsize=float(getattr(label_cfg, "fontsize", 12.0)),
		color=str(getattr(label_cfg, "color", "white")),
	)
	text.set_gid("template_unit_id_label")


def _add_center_most_channel_coords(ax: Any, *, config: TemplatePlotConfig, locations_xy: np.ndarray) -> None:
	coords_cfg = getattr(config, "center_most_channel_coords", None)
	if not bool(getattr(coords_cfg, "show", False)):
		return
	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[0]) <= 0 or int(locs.shape[1]) < 2:
		return
	xy = locs[:, :2]
	cx = float(np.nanmean(xy[:, 0]))
	cy = float(np.nanmean(xy[:, 1]))
	d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
	idx = int(np.nanargmin(d2)) if np.isfinite(d2).any() else 0
	x0 = float(xy[idx, 0])
	y0 = float(xy[idx, 1])

	xmin, xmax = ax.get_xlim()
	ymin, ymax = ax.get_ylim()
	span_x = float(max(1e-9, abs(xmax - xmin)))
	span_y = float(max(1e-9, abs(ymax - ymin)))
	ha = str(getattr(coords_cfg, "horizontal_alignment", "left") or "left").strip().lower()
	va = str(getattr(coords_cfg, "vertical_alignment", "top") or "top").strip().lower()
	off_x = float(getattr(coords_cfg, "x_offset_frac", 0.02)) * span_x
	off_y = float(getattr(coords_cfg, "y_offset_frac", 0.01)) * span_y
	if ha == "right":
		x = xmax - off_x
	elif ha == "center":
		x = 0.5 * (xmin + xmax)
	else:
		x = xmin + off_x
	if va == "top":
		y = ymax - off_y
	elif va == "center":
		y = 0.5 * (ymin + ymax)
	else:
		y = ymin + off_y
	text = ax.text(
		x,
		y,
		f"({x0:.1f}, {y0:.1f})",
		ha=ha,
		va=va,
		fontsize=float(getattr(coords_cfg, "fontsize", 10.0)),
		color=str(getattr(coords_cfg, "color", "white")),
	)
	text.set_gid("template_center_coords_label")


def _apply_template_plot_overlays(
	ax: Any,
	*,
	config: TemplatePlotConfig,
	unit_id: Any | None,
	locations_xy: np.ndarray,
) -> None:
	_add_unit_id_label(ax, config=config, unit_id=unit_id)
	_add_center_most_channel_coords(ax, config=config, locations_xy=locations_xy)
	if not bool(getattr(config, "show_axes", True)):
		ax.set_axis_off()


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
	anchor_x_frac = float(anchor_x_frac)
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
	# For propagation, anchor_x_frac controls the LEFT edge of the time bar so
	# small negative values can nudge it slightly left of the plotting window.
	x_left = float(min(x0, x1)) + (anchor_x_frac * span_x)
	x_right = x_left + time_bar_samples
	x_min = float(min(x0, x1))
	x_max = float(max(x0, x1))
	if x_right > (x_max - 1.0):
		x_left -= float(x_right - (x_max - 1.0))
		x_right = x_left + time_bar_samples
	if x_left < x_min:
		pad = float(max(1.0, 0.02 * span_x))
		ax.set_xlim(float(x_left - pad), float(x_max))
		x0, x1 = ax.get_xlim()
		x_min = float(min(x0, x1))
		x_max = float(max(x0, x1))

	anchor_y = float(min(y0, y1)) + (anchor_y_frac * span_y)
	anchor_y = float(min(max(anchor_y, min(y0, y1) + 1.0), max(y0, y1) - 1.0))
	y_top = anchor_y + amp_bar_plot

	ax.plot([x_left, x_right], [anchor_y, anchor_y], color=text_color, lw=linewidth, solid_capstyle="butt")
	ax.plot([x_left, x_left], [anchor_y, y_top], color=text_color, lw=linewidth, solid_capstyle="butt")
	ax.text(
		(x_left + x_right) / 2.0,
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
	unit_id: Any | None = None,
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
		xmin, xmax, ymin, ymax = _recenter_limits_around_point(
			xmin,
			xmax,
			ymin,
			ymax,
			center_xy=(float(locs[peak_idx, 0]), float(locs[peak_idx, 1])),
		)

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
	_apply_template_plot_overlays(ax, config=config, unit_id=unit_id, locations_xy=locs)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(
			png_path,
			dpi=max(72.0, float(getattr(config, "dpi", 300.0))),
			bbox_inches="tight",
			facecolor=fig.get_facecolor(),
		)
		outputs["template_png"] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_svg"] = str(svg_path)

	plt.close(fig)
	return outputs


def _template_plot_v2_metric(
	metric: str,
	*,
	amplitude: np.ndarray,
	negative_peak: np.ndarray,
	latency: np.ndarray,
) -> np.ndarray | None:
	token = str(metric or "").strip().lower().replace("-", "_").replace(" ", "_")
	if token in {"", "fixed", "none", "constant"}:
		return None
	if token in {"latency", "delay"}:
		return np.asarray(latency, dtype=float)
	if token in {"negative_peak", "neg_peak", "peak"}:
		return np.asarray(negative_peak, dtype=float)
	return np.asarray(amplitude, dtype=float)


def _template_plot_v2_marker_sizes_pt2(
	*,
	size_values: np.ndarray | None,
	marker_min_size: float,
	marker_max_size: float,
	count: int,
) -> np.ndarray:
	min_diameter_pt = float(max(0.0, float(marker_min_size)))
	max_diameter_pt = float(max(min_diameter_pt, float(marker_max_size)))
	if size_values is None:
		diameters_pt = np.full((int(count),), max_diameter_pt, dtype=float)
	else:
		raw_values = np.asarray(size_values, dtype=float)
		raw_values = np.abs(np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0))
		finite_values = raw_values[np.isfinite(raw_values)]
		if int(finite_values.size) == 0:
			diameters_pt = np.full((int(count),), min_diameter_pt, dtype=float)
		else:
			value_min = float(np.min(finite_values))
			value_max = float(np.max(finite_values))
			if value_max > value_min:
				normalized = np.clip((raw_values - value_min) / float(value_max - value_min), 0.0, 1.0)
				diameters_pt = min_diameter_pt + ((max_diameter_pt - min_diameter_pt) * normalized)
			else:
				diameters_pt = np.full((int(count),), max_diameter_pt, dtype=float)
	return np.asarray(np.pi * np.square(np.maximum(diameters_pt * 0.5, 0.0)), dtype=float)


def _format_template_plot_v2_text(text: str, values: dict[str, Any]) -> str:
	try:
		return str(text).format(**values)
	except Exception:
		return str(text)


def _draw_template_plot_v2_text(
	ax: Any,
	config: TemplatePlotV2TextConfig,
	*,
	default_text: str,
	values: dict[str, Any],
) -> None:
	if not bool(config.show):
		return
	text = str(config.text or default_text)
	if not text.strip():
		return
	ax.text(
		float(config.x),
		float(config.y),
		_format_template_plot_v2_text(text, values),
		transform=ax.transAxes,
		fontsize=float(config.fontsize),
		color=str(config.color),
		horizontalalignment=str(config.horizontal_alignment),
		verticalalignment=str(config.vertical_alignment),
	)


def _template_plot_v2_limits(
	locs: np.ndarray,
	config: TemplatePlotTemplatesV2PhaseConfig,
	*,
	peak_xy: tuple[float, float] | None,
) -> tuple[float, float, float, float]:
	finite_locs = np.asarray(locs, dtype=float)
	finite_locs = finite_locs[np.isfinite(finite_locs).all(axis=1)]
	if int(finite_locs.shape[0]) == 0:
		raise ValueError("No finite channel locations available for plot_templates_v2")
	xmin = float(np.min(finite_locs[:, 0]))
	xmax = float(np.max(finite_locs[:, 0]))
	ymin = float(np.min(finite_locs[:, 1]))
	ymax = float(np.max(finite_locs[:, 1]))
	x_span = float(max(abs(xmax - xmin), 1.0))
	y_span = float(max(abs(ymax - ymin), 1.0))
	if config.padding_um is not None:
		x_pad = float(max(0.0, float(config.padding_um)))
		y_pad = float(max(0.0, float(config.padding_um)))
	else:
		padding_fraction = float(max(0.0, float(config.padding_fraction)))
		x_pad = x_span * padding_fraction
		y_pad = y_span * padding_fraction
	xmin -= x_pad
	xmax += x_pad
	ymin -= y_pad
	ymax += y_pad
	if config.x_min is not None:
		xmin = float(config.x_min)
	if config.x_max is not None:
		xmax = float(config.x_max)
	if config.y_min is not None:
		ymin = float(config.y_min)
	if config.y_max is not None:
		ymax = float(config.y_max)
	if float(xmax) <= float(xmin):
		xmax = float(xmin) + 1.0
	if float(ymax) <= float(ymin):
		ymax = float(ymin) + 1.0
	if bool(config.force_square_aspect):
		center = peak_xy if bool(config.center_on_peak) else None
		xmin, xmax, ymin, ymax = _make_square_limits(xmin, xmax, ymin, ymax, center_xy=center)
	return float(xmin), float(xmax), float(ymin), float(ymax)


def _draw_template_plot_v2_scale_bar(ax: Any, config: TemplatePlotTemplatesV2PhaseConfig) -> None:
	scale_bar = config.scale_bar
	if not bool(scale_bar.show):
		return
	x_limits = ax.get_xlim()
	x_span = float(abs(float(x_limits[1]) - float(x_limits[0])))
	if x_span <= 0.0:
		return
	length_um = float(scale_bar.length_um) if scale_bar.length_um is not None else x_span * 0.20
	length_frac = float(max(0.0, min(1.0, length_um / x_span)))
	x_anchor = float(scale_bar.x)
	alignment = str(scale_bar.horizontal_alignment or "right").strip().lower()
	if alignment == "center":
		x0 = x_anchor - (0.5 * length_frac)
		x1 = x_anchor + (0.5 * length_frac)
	elif alignment == "left":
		x0 = x_anchor
		x1 = x_anchor + length_frac
	else:
		x0 = x_anchor - length_frac
		x1 = x_anchor
	y_position = float(scale_bar.y)
	ax.plot(
		[x0, x1],
		[y_position, y_position],
		transform=ax.transAxes,
		color=str(scale_bar.color),
		linewidth=float(scale_bar.linewidth),
		solid_capstyle="butt",
		clip_on=False,
	)
	label = _format_template_plot_v2_text(str(scale_bar.label), {"length_um": length_um})
	if label.strip():
		ax.text(
			0.5 * (x0 + x1),
			y_position + float(scale_bar.text_y_offset),
			label,
			transform=ax.transAxes,
			color=str(scale_bar.color),
			fontsize=float(scale_bar.fontsize),
			horizontalalignment="center",
			verticalalignment="bottom",
		)


def render_template_circles_plot_v2(
	*,
	template: Any,
	locations_xy: Any,
	config: TemplatePlotTemplatesV2PhaseConfig,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
	unit_id: Any | None = None,
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

	resolved_probe_geometry = _resolve_probe_geometry_sampling_rate(
		probe_geometry=probe_geometry,
		config=config,
		png_path=png_path,
		svg_path=svg_path,
		unit_id=unit_id,
	)
	amplitude = np.ptp(template_c_by_t, axis=1)
	negative_peak = np.abs(np.min(template_c_by_t, axis=1))
	min_indices = np.argmin(template_c_by_t, axis=1).astype(float)
	peak_index = int(np.argmax(negative_peak)) if negative_peak.size > 0 else 0
	reference_index = float(min_indices[peak_index]) if min_indices.size > 0 else 0.0
	latency_samples = min_indices - reference_index
	latency, latency_units_label = _convert_latency_samples_to_units(
		latency_samples,
		units=str(config.color_bar_units or ""),
		probe_geometry=resolved_probe_geometry,
	)

	size_values = _template_plot_v2_metric(
		str(config.size_by),
		amplitude=amplitude,
		negative_peak=negative_peak,
		latency=latency,
	)
	marker_sizes = _template_plot_v2_marker_sizes_pt2(
		size_values=size_values,
		marker_min_size=float(config.marker_min_size),
		marker_max_size=float(config.marker_max_size),
		count=int(locs.shape[0]),
	)

	color_values = _template_plot_v2_metric(
		str(config.color_by),
		amplitude=amplitude,
		negative_peak=negative_peak,
		latency=latency,
	)
	peak_xy = None
	if 0 <= peak_index < int(locs.shape[0]) and bool(np.isfinite(locs[peak_index, :]).all()):
		peak_xy = (float(locs[peak_index, 0]), float(locs[peak_index, 1]))
	xmin, xmax, ymin, ymax = _template_plot_v2_limits(locs, config, peak_xy=peak_xy)

	fig, ax = plt.subplots(figsize=tuple(config.figsize))
	fig.patch.set_facecolor(str(config.background))
	ax.set_facecolor(str(config.background))
	fig.subplots_adjust(
		left=float(config.figure_left),
		right=float(config.figure_right),
		bottom=float(config.figure_bottom),
		top=float(config.figure_top),
	)
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	if bool(config.invert_y_axis):
		ax.invert_yaxis()
	ax.set_aspect("equal", adjustable="box")

	scatter_kwargs: dict[str, Any] = {
		"s": marker_sizes,
		"alpha": float(config.marker_alpha),
		"linewidths": float(config.marker_linewidth),
	}
	if str(config.edge_color).strip().lower() not in {"", "none"}:
		scatter_kwargs["edgecolors"] = str(config.edge_color)
	else:
		scatter_kwargs["edgecolors"] = "none"
	if color_values is None:
		scatter = ax.scatter(locs[:, 0], locs[:, 1], color=str(config.marker_color), **scatter_kwargs)
	else:
		color_values = np.asarray(color_values, dtype=float)
		finite_colors = color_values[np.isfinite(color_values)]
		if int(finite_colors.size) == 0:
			finite_colors = np.asarray([0.0, 1.0], dtype=float)
		vmin = float(np.min(finite_colors))
		vmax = float(np.max(finite_colors))
		if vmax <= vmin:
			vmax = vmin + 1.0
		scatter = ax.scatter(
			locs[:, 0],
			locs[:, 1],
			c=color_values,
			cmap=_maybe_reversed_colormap(
				str(config.cmap),
				reverse=bool(getattr(config.colorbar, "reverse", False) or str(config.color_by or "").strip().lower() == "latency"),
			),
			vmin=vmin,
			vmax=vmax,
			**scatter_kwargs,
		)

	axes_color = str(config.axis_label_color)
	for spine in ax.spines.values():
		spine.set_color(axes_color)
	if bool(config.show_axis_labels):
		ax.set_xlabel("x (um)", color=axes_color)
		ax.set_ylabel("y (um)", color=axes_color)
	ax.tick_params(colors=axes_color, labelsize=6)
	if not bool(config.show_axes):
		ax.set_axis_off()

	text_values = {
		"unit_id": unit_id,
		"x_um": None if peak_xy is None else float(peak_xy[0]),
		"y_um": None if peak_xy is None else float(peak_xy[1]),
		"latency_units": latency_units_label,
	}
	_draw_template_plot_v2_text(ax, config.title, default_text="Unit {unit_id}", values=text_values)
	_draw_template_plot_v2_text(ax, config.unit_id_label, default_text="{unit_id}", values=text_values)
	if peak_xy is not None:
		_draw_template_plot_v2_text(
			ax,
			config.coords,
			default_text="x={x_um:.1f} um, y={y_um:.1f} um",
			values=text_values,
		)
	_draw_template_plot_v2_scale_bar(ax, config)
	scale_circle_ref_value = float(np.nanmax(negative_peak)) if negative_peak.size > 0 else 0.0
	_add_scale_circle(
		ax,
		config=config,
		reference_area_pt2=float(np.nanmax(marker_sizes)) if marker_sizes.size > 0 else 0.0,
		reference_value=scale_circle_ref_value,
	)

	colorbar = config.colorbar
	if color_values is not None and bool(colorbar.show):
		colorbar_axes = None
		if None not in {colorbar.x, colorbar.y, colorbar.width, colorbar.height}:
			colorbar_axes = fig.add_axes([
				float(colorbar.x),
				float(colorbar.y),
				float(colorbar.width),
				float(colorbar.height),
			])
		if colorbar_axes is None:
			cbar = fig.colorbar(scatter, ax=ax, fraction=float(colorbar.fraction), pad=float(colorbar.pad))
		else:
			cbar = fig.colorbar(scatter, cax=colorbar_axes)
		label = str(colorbar.label or "").strip()
		if not label:
			metric = str(config.color_by or "amplitude").strip().lower()
			if metric == "latency":
				label = f"Latency ({latency_units_label})" if latency_units_label else "Latency"
			else:
				label = "Amplitude"
		cbar.set_label(label, color=str(colorbar.color), fontsize=float(colorbar.fontsize))
		cbar.ax.tick_params(labelsize=float(colorbar.tick_fontsize), colors=str(colorbar.color))
		cbar.outline.set_edgecolor(str(colorbar.color))

	outputs: dict[str, str] = {}
	savefig_kwargs: dict[str, Any] = {
		"dpi": max(72.0, float(config.dpi)),
		"facecolor": fig.get_facecolor(),
	}
	if config.bbox_inches is not None:
		savefig_kwargs["bbox_inches"] = str(config.bbox_inches)
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, **savefig_kwargs)
		outputs["template_circles_v2_png"] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", **savefig_kwargs)
		outputs["template_circles_v2_svg"] = str(svg_path)
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
	unit_id: Any | None = None,
	propagation_order_rank_by_channel: dict[int, int] | None = None,
	branch_morphology: Any | None = None,
	gtr: Any | None = None,
	plot_scope_points_xy: Any | None = None,
	zoom_padding_percent: float | None = None,
	allow_scope_expansion: bool = True,
	fig: Any | None = None,
	ax: Any | None = None,
	close_figure: bool = True,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	config = _resolve_fast_render_template_circles_config(config)

	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Expected locations shape (n,2+), got {getattr(locs, 'shape', None)}")
	locs = locs[:, :2]

	template_c_by_t = _as_template_channels_by_time(template, int(locs.shape[0]))
	if int(template_c_by_t.shape[0]) != int(locs.shape[0]):
		raise ValueError(
			f"Template/locations size mismatch: template_channels={template_c_by_t.shape[0]} locations={locs.shape[0]}"
		)

	resolved_probe_geometry = _resolve_probe_geometry_sampling_rate(
		probe_geometry=probe_geometry,
		png_path=png_path,
		svg_path=svg_path,
		unit_id=unit_id,
	)

	amp = np.ptp(template_c_by_t, axis=1)
	abs_negative_peak = np.abs(np.min(template_c_by_t, axis=1))
	min_idx = np.argmin(template_c_by_t, axis=1).astype(float)
	ref = float(min_idx[int(np.argmax(amp))]) if min_idx.size > 0 else 0.0
	lat_samples = min_idx - ref
	lat, latency_units_label = _convert_latency_samples_to_units(
		lat_samples,
		units=str(config.color_bar_units or ""),
		probe_geometry=resolved_probe_geometry,
	)

	size_metric = abs_negative_peak if str(config.size_by) == "amplitude" else np.abs(lat)
	color_metric = amp if str(config.color_by) == "amplitude" else lat
	color_values = np.asarray(color_metric, dtype=float)
	force_first_range_for_nonpositive = bool(
		getattr(config, "color_bar_force_zero_and_neg_values_first_color_range", False)
	)
	vmin = float(np.nanmin(color_values)) if color_values.size > 0 else 0.0
	vmax = float(np.nanmax(color_values)) if color_values.size > 0 else 1.0
	if not np.isfinite(vmin):
		vmin = 0.0
	if not np.isfinite(vmax):
		vmax = 1.0
	if vmax <= vmin:
		vmax = vmin + 1.0
	circles_cmap = _maybe_reversed_colormap("viridis", reverse=(str(config.color_by) == "latency"))
	cmap_obj = plt.get_cmap(circles_cmap)
	use_piecewise_zero_boundary = bool(
		str(config.color_by) == "latency"
		and force_first_range_for_nonpositive
		and float(vmin) < 0.0
		and float(vmax) > 0.0
	)
	if use_piecewise_zero_boundary:
		# Reserve exactly one colormap range for non-positive latency values, regardless of colormap choice.
		# This keeps negatives visible while placing zero at the boundary between first and second ranges.
		n_colors = int(max(2, int(getattr(cmap_obj, "N", 256))))
		first_range_frac = float(1.0 / float(n_colors))
		zero_contrast = float(max(1.0, float(getattr(config, "color_bar_zero_transition_contrast", 1.0))))
		inv_gamma = float(1.0 / zero_contrast)

		def _forward(vals: Any) -> np.ndarray:
			a = np.asarray(vals, dtype=float)
			out = np.empty_like(a, dtype=float)
			neg = a <= 0.0
			neg_t = (a[neg] - float(vmin)) / float(0.0 - float(vmin))
			neg_t = np.clip(neg_t, 0.0, 1.0)
			if zero_contrast > 1.0:
				# Move values away from the zero boundary on both sides to sharpen local contrast.
				dist_from_zero_neg = np.clip(1.0 - neg_t, 0.0, 1.0)
				dist_from_zero_neg = np.power(dist_from_zero_neg, inv_gamma)
				neg_t = 1.0 - dist_from_zero_neg
			out[neg] = neg_t * first_range_frac

			pos_t = (a[~neg] - 0.0) / float(float(vmax) - 0.0)
			pos_t = np.clip(pos_t, 0.0, 1.0)
			if zero_contrast > 1.0:
				pos_t = np.power(pos_t, inv_gamma)
			out[~neg] = first_range_frac + (pos_t * (1.0 - first_range_frac))
			return np.clip(out, 0.0, 1.0)

		def _inverse(fracs: Any) -> np.ndarray:
			u = np.asarray(fracs, dtype=float)
			out = np.empty_like(u, dtype=float)
			neg = u <= first_range_frac
			neg_t = np.clip(u[neg] / first_range_frac, 0.0, 1.0)
			if zero_contrast > 1.0:
				dist_from_zero_neg = np.clip(1.0 - neg_t, 0.0, 1.0)
				dist_from_zero_neg = np.power(dist_from_zero_neg, zero_contrast)
				neg_t = 1.0 - dist_from_zero_neg
			out[neg] = float(vmin) + (neg_t * float(0.0 - float(vmin)))

			pos_t = np.clip((u[~neg] - first_range_frac) / float(1.0 - first_range_frac), 0.0, 1.0)
			if zero_contrast > 1.0:
				pos_t = np.power(pos_t, zero_contrast)
			out[~neg] = pos_t * float(float(vmax) - 0.0)
			return np.clip(out, float(vmin), float(vmax))

		color_norm = plt.matplotlib.colors.FuncNorm((lambda v: _forward(v), lambda u: _inverse(u)), vmin=vmin, vmax=vmax, clip=True)
	else:
		color_norm = plt.Normalize(vmin=vmin, vmax=vmax)

	size_norm = np.asarray(size_metric, dtype=float)
	size_norm = np.nan_to_num(size_norm, nan=0.0, posinf=0.0, neginf=0.0)
	if float(np.max(size_norm)) > 0.0:
		size_norm = size_norm / float(np.max(size_norm))
	sizes_base = 8.0 + 42.0 * size_norm

	peak_idx = int(np.argmax(abs_negative_peak)) if abs_negative_peak.size > 0 else 0

	if (fig is None) != (ax is None):
		raise ValueError("render_template_circles_plot requires both fig and ax when reusing an existing host")
	if fig is None or ax is None:
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111)
	else:
		ax.clear()
	ax.set_xlabel("x (um)")
	ax.set_ylabel("y (um)")

	plot_scope_points = np.asarray(locs, dtype=float)
	if plot_scope_points_xy is not None:
		plot_scope_points = np.asarray(plot_scope_points_xy, dtype=float)
		if plot_scope_points.ndim != 2 or int(plot_scope_points.shape[0]) == 0 or int(plot_scope_points.shape[1]) < 2:
			raise ValueError(
				f"Expected plot_scope_points_xy shape (n,2+), got {getattr(plot_scope_points, 'shape', None)}"
			)
		plot_scope_points = plot_scope_points[:, :2]

	if zoom_padding_percent is None:
		xmin, xmax, ymin, ymax = _compute_plot_limits(plot_scope_points)
	else:
		xmin, xmax, ymin, ymax = _compute_plot_limits_with_padding_percent(
			plot_scope_points,
			padding_percent=float(zoom_padding_percent),
		)
	if bool(config.force_square_aspect):
		center = None
		if bool(config.force_center_soma) and 0 <= peak_idx < int(locs.shape[0]):
			center = (float(locs[peak_idx, 0]), float(locs[peak_idx, 1]))
		xmin, xmax, ymin, ymax = _make_square_limits(xmin, xmax, ymin, ymax, center_xy=center)
	elif bool(config.force_center_soma) and 0 <= peak_idx < int(locs.shape[0]):
		xmin, xmax, ymin, ymax = _recenter_limits_around_point(
			xmin,
			xmax,
			ymin,
			ymax,
			center_xy=(float(locs[peak_idx, 0]), float(locs[peak_idx, 1])),
		)

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_aspect("equal", adjustable="box")
	sc = ax.scatter(
		locs[:, 0],
		locs[:, 1],
		s=sizes_base,
		c=color_values,
		cmap=circles_cmap,
		norm=color_norm,
		alpha=0.92,
		linewidths=0.0,
	)
	_apply_style(fig, ax, config=config)
	_add_scale_bar(ax, config=config)
	scale_circle_ref_value = float(np.nanmax(abs_negative_peak)) if abs_negative_peak.size > 0 else 0.0
	_add_scale_circle(
		ax,
		config=config,
		reference_area_pt2=float(np.nanmax(sizes_base)) if sizes_base.size > 0 else 0.0,
		reference_value=scale_circle_ref_value,
	)

	# Build deterministic colorbar patch bounds in color-space. For piecewise zero-boundary
	# mode this keeps zero exactly at the first/second color-range transition.
	bounds = np.asarray(color_norm.inverse(np.linspace(0.0, 1.0, 257, dtype=float)), dtype=float)
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

	tick_fontsize = float(max(1.0, float(getattr(config, "color_bar_tick_fontsize", 6.0))))
	tick_color = "white" if str(config.background or "").strip().lower() == "black" else "black"
	cbar.ax.tick_params(labelsize=tick_fontsize, colors=tick_color)

	if str(config.background or "").strip().lower() == "black":
		cbar.outline.set_edgecolor("white")
		if show_axes_title:
			cbar.set_label(cbar.ax.get_ylabel(), color="white", fontsize=7)

	_apply_template_plot_overlays(ax, config=config, unit_id=unit_id, locations_xy=locs)

	def _update_non_overlapping_sizes_for_current_axes() -> None:
		bbox = ax.get_window_extent()
		axis_scale = float(72.0 / float(fig.dpi))
		centers_display_pt = np.asarray(ax.transData.transform(locs), dtype=float) * axis_scale
		bbox_x_limits_pt = (float(bbox.x0) * axis_scale, float(bbox.x1) * axis_scale)
		bbox_y_limits_pt = (float(bbox.y0) * axis_scale, float(bbox.y1) * axis_scale)
		sizes = _compute_max_non_overlapping_circle_areas(
			centers_display_pt=centers_display_pt,
			base_areas_pt2=sizes_base,
			axis_x_limits_pt=bbox_x_limits_pt,
			axis_y_limits_pt=bbox_y_limits_pt,
			overlap_tolerance_pt=0.0,
		)
		sc.set_sizes(sizes)

	def _expand_axes_limits(frac: float = 0.10) -> None:
		x0, x1 = ax.get_xlim()
		y0, y1 = ax.get_ylim()
		cx = 0.5 * (float(x0) + float(x1))
		cy = 0.5 * (float(y0) + float(y1))
		hx = 0.5 * abs(float(x1) - float(x0))
		hy = 0.5 * abs(float(y1) - float(y0))
		scale = 1.0 + float(max(0.01, frac))
		ax.set_xlim(cx - (hx * scale), cx + (hx * scale))
		ax.set_ylim(cy - (hy * scale), cy + (hy * scale))

	def _refresh_overlay_artists() -> None:
		for patch in list(ax.patches):
			if str(getattr(patch, "get_gid", lambda: "")() or "") == "template_scale_circle_patch":
				patch.remove()
		for line in list(ax.lines):
			if str(getattr(line, "get_gid", lambda: "")() or "") == "template_scale_bar_line":
				line.remove()
		for text in list(ax.texts):
			gid = str(getattr(text, "get_gid", lambda: "")() or "")
			if gid in {
				"template_scale_bar_text",
				"template_scale_circle_text",
				"template_unit_id_label",
				"template_center_coords_label",
			}:
				text.remove()
		_add_scale_bar(ax, config=config)
		_add_scale_circle(
			ax,
			config=config,
			reference_area_pt2=float(np.nanmax(np.asarray(sc.get_sizes(), dtype=float))),
			reference_value=scale_circle_ref_value,
		)
		_apply_template_plot_overlays(ax, config=config, unit_id=unit_id, locations_xy=locs)

	def _bbox_overlap(a: Any, b: Any) -> bool:
		try:
			return bool(a.overlaps(b))
		except Exception:
			return False

	def _any_overlap(group_a: list[Any], group_b: list[Any]) -> bool:
		for a in group_a:
			for b in group_b:
				if _bbox_overlap(a, b):
					return True
		return False

	def _collect_circle_bboxes() -> list[Any]:
		from matplotlib.transforms import Bbox  # type: ignore[import-not-found]

		offsets = np.asarray(sc.get_offsets(), dtype=float)
		sizes = np.asarray(sc.get_sizes(), dtype=float)
		if offsets.ndim != 2 or sizes.ndim != 1 or int(offsets.shape[0]) != int(sizes.shape[0]):
			return []
		centers_px = np.asarray(ax.transData.transform(offsets), dtype=float)
		radii_pt = np.sqrt(np.clip(sizes, 0.0, None) / np.pi)
		radii_px = radii_pt * float(fig.dpi / 72.0)
		bboxes: list[Any] = []
		for i in range(int(centers_px.shape[0])):
			cx, cy = float(centers_px[i, 0]), float(centers_px[i, 1])
			r = float(radii_px[i])
			bboxes.append(Bbox.from_extents(cx - r, cy - r, cx + r, cy + r))
		return bboxes

	overlap_cfg = getattr(config, "overlap_controls", None)
	max_overlap_iters = max(0, int(getattr(overlap_cfg, "max_overlap_check_iterations", 0) if overlap_cfg is not None else 0))
	check_scalebar_coords = bool(getattr(overlap_cfg, "scalebar_coords_overlap_detect", False)) if overlap_cfg is not None else False
	check_scalebar_colorbar = bool(getattr(overlap_cfg, "scalebar_colorbar_overlap_detect", False)) if overlap_cfg is not None else False
	check_unitid_channel = bool(getattr(overlap_cfg, "unitid_label_channel_overlap_detect", False)) if overlap_cfg is not None else False
	check_coords_channel = bool(getattr(overlap_cfg, "coords_channel_overlap_detect", False)) if overlap_cfg is not None else False
	check_scalebar_channel = bool(getattr(overlap_cfg, "scalebar_channel_overlap_detect", False)) if overlap_cfg is not None else False
	check_scalecircle_channel = bool(getattr(overlap_cfg, "scalecircle_channel_overlap_detect", False)) if overlap_cfg is not None else False
	enable_overlap_checks = any(
		[
			check_scalebar_coords,
			check_scalebar_colorbar,
			check_unitid_channel,
			check_coords_channel,
			check_scalebar_channel,
			check_scalecircle_channel,
		]
	)

	if enable_overlap_checks and max_overlap_iters > 0 and bool(allow_scope_expansion):
		for _ in range(max_overlap_iters):
			fig.canvas.draw()
			renderer = fig.canvas.get_renderer()

			scale_bar_bboxes = None
			if check_scalebar_coords or check_scalebar_colorbar or check_scalebar_channel:
				scale_bar_bboxes = [
					artist.get_window_extent(renderer=renderer)
					for artist in list(ax.lines) + list(ax.texts)
					if str(getattr(artist, "get_gid", lambda: "")() or "") in {"template_scale_bar_line", "template_scale_bar_text"}
				]

			unit_id_bboxes = None
			if check_unitid_channel:
				unit_id_bboxes = [
					artist.get_window_extent(renderer=renderer)
					for artist in list(ax.texts)
					if str(getattr(artist, "get_gid", lambda: "")() or "") == "template_unit_id_label"
				]

			coords_bboxes = None
			if check_scalebar_coords or check_coords_channel:
				coords_bboxes = [
					artist.get_window_extent(renderer=renderer)
					for artist in list(ax.texts)
					if str(getattr(artist, "get_gid", lambda: "")() or "") == "template_center_coords_label"
				]

			circle_bboxes = None
			if check_unitid_channel or check_coords_channel or check_scalebar_channel or check_scalecircle_channel:
				circle_bboxes = _collect_circle_bboxes()

			colorbar_bboxes = None
			if check_scalebar_colorbar:
				colorbar_bboxes = [cbar.ax.get_window_extent(renderer=renderer)]

			scale_circle_bboxes = None
			if check_scalecircle_channel:
				scale_circle_bboxes = [
					artist.get_window_extent(renderer=renderer)
					for artist in list(ax.patches) + list(ax.texts)
					if str(getattr(artist, "get_gid", lambda: "")() or "") in {"template_scale_circle_patch", "template_scale_circle_text"}
				]

			has_overlap = False
			if check_scalebar_coords and _any_overlap(scale_bar_bboxes or [], coords_bboxes or []):
				has_overlap = True
			if check_scalebar_colorbar and _any_overlap(scale_bar_bboxes or [], colorbar_bboxes or []):
				has_overlap = True
			if check_unitid_channel and _any_overlap(unit_id_bboxes or [], circle_bboxes or []):
				has_overlap = True
			if check_coords_channel and _any_overlap(coords_bboxes or [], circle_bboxes or []):
				has_overlap = True
			if check_scalebar_channel and _any_overlap(scale_bar_bboxes or [], circle_bboxes or []):
				has_overlap = True
			if check_scalecircle_channel and _any_overlap(scale_circle_bboxes or [], circle_bboxes or []):
				has_overlap = True

			if not has_overlap:
				break
			_expand_axes_limits(0.10)
			_refresh_overlay_artists()

	# Compute final non-overlapping sizes after colorbar/layout has finalized axis dimensions.
	fig.canvas.draw()
	_update_non_overlapping_sizes_for_current_axes()
	if bool(getattr(config, "show_scale_circle", False)):
		_refresh_overlay_artists()

	if bool(getattr(config, "show_propagation_order_labels", False)):
		rank_map = dict(propagation_order_rank_by_channel or {})
		label_color = str(getattr(config, "propagation_order_label_color", "white") or "white")
		label_fontsize = float(max(1.0, float(getattr(config, "propagation_order_label_fontsize", 6.0))))
		bbox_alpha = float(min(1.0, max(0.0, float(getattr(config, "propagation_order_label_bbox_alpha", 0.35)))))
		for ch_idx in range(int(locs.shape[0])):
			rank = rank_map.get(int(ch_idx), None)
			if rank is None:
				continue
			ax.text(
				float(locs[ch_idx, 0]),
				float(locs[ch_idx, 1]),
				str(int(rank)),
				color=label_color,
				fontsize=label_fontsize,
				horizontalalignment="center",
				verticalalignment="center",
				fontweight="bold",
				bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "alpha": bbox_alpha, "linewidth": 0.0},
			)

	branch_cfg = getattr(config, "branch_morphology", None)
	if bool(getattr(branch_cfg, "enabled", False)):
		resolved_branch_payload = branch_morphology
		if resolved_branch_payload is None:
			resolved_branch_payload = _branch_morphology_from_gtr(gtr)
		_draw_branch_morphology_overlay(
			ax=ax,
			fig=fig,
			sc=sc,
			locations_xy=locs,
			branch_morphology=resolved_branch_payload,
			branch_cfg=branch_cfg,
		)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(
			png_path,
			dpi=max(72.0, float(getattr(config, "dpi", 300.0))),
			bbox_inches="tight",
			facecolor=fig.get_facecolor(),
		)
		outputs["template_circles_png"] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(
			svg_path,
			format="svg",
			dpi=max(72.0, float(getattr(config, "dpi", 300.0))),
			bbox_inches="tight",
			facecolor=fig.get_facecolor(),
		)
		outputs["template_circles_svg"] = str(svg_path)

	if bool(close_figure):
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


def _resolve_probe_geometry_sampling_rate(
	*,
	probe_geometry: ProbeGeometryConfig | None,
	config: Any | None = None,
	png_path: Path,
	svg_path: Path,
	unit_id: Any | None,
) -> ProbeGeometryConfig | None:
	sampling_rate_hz = _read_sampling_rate_hz_from_templates_metadata(
		png_path=png_path,
		svg_path=svg_path,
		unit_id=unit_id,
	)
	if sampling_rate_hz is None and probe_geometry is not None and probe_geometry.sampling_rate_hz is not None:
		try:
			candidate = float(probe_geometry.sampling_rate_hz)
			if candidate > 0.0:
				sampling_rate_hz = candidate
		except Exception:
			pass

	if sampling_rate_hz is None:
		return probe_geometry

	if probe_geometry is None:
		return ProbeGeometryConfig(sampling_rate_hz=float(sampling_rate_hz))

	return ProbeGeometryConfig(
		pitch_um=probe_geometry.pitch_um,
		electrode_size_um_x=probe_geometry.electrode_size_um_x,
		electrode_size_um_y=probe_geometry.electrode_size_um_y,
		sampling_rate_hz=float(sampling_rate_hz),
	)


def _read_sampling_rate_hz_from_templates_metadata(*, png_path: Path, svg_path: Path, unit_id: Any | None) -> float | None:
	for candidate in _candidate_templates_summary_paths(png_path=png_path, svg_path=svg_path, unit_id=unit_id):
		if not candidate.exists() or not candidate.is_file():
			continue
		try:
			with candidate.open("r", encoding="utf-8") as f:
				payload = json.load(f)
		except Exception:
			continue

		rate = _extract_sampling_rate_hz_from_summary_payload(payload)
		if rate is not None and float(rate) > 0:
			return float(rate)
	return None


def _candidate_templates_summary_paths(*, png_path: Path, svg_path: Path, unit_id: Any | None) -> list[Path]:
	candidates: list[Path] = []
	seen: set[str] = set()

	def _push(path: Path) -> None:
		key = str(path)
		if key not in seen:
			seen.add(key)
			candidates.append(path)

	for base in (Path(png_path).parent, Path(svg_path).parent):
		_push(base / "unit_templates_summary.json")

	unit_tokens: list[str] = []
	if unit_id is not None:
		try:
			ui = int(unit_id)
			unit_tokens.extend([f"{ui:04d}", str(ui)])
		except Exception:
			unit_tokens.append(str(unit_id))

	search_roots: list[Path] = []
	for p in (Path(png_path), Path(svg_path)):
		search_roots.extend([p.parent, *p.parents])

	for root in search_roots:
		templates_units_dir = root / "template_outputs" / "units"
		if templates_units_dir.exists() and templates_units_dir.is_dir():
			for token in unit_tokens:
				_push(templates_units_dir / token / "unit_templates_summary.json")
			_push(templates_units_dir / "unit_templates_summary.json")

	return candidates


def _extract_sampling_rate_hz_from_summary_payload(payload: Any) -> float | None:
	if not isinstance(payload, dict):
		return None

	for key in ("effective_sampling_rate_hz", "sampling_rate_hz", "sample_rate_hz"):
		val = payload.get(key)
		if val is not None:
			try:
				rate = float(val)
				if rate > 0:
					return rate
			except Exception:
				pass

	upsampling = payload.get("upsampling")
	if isinstance(upsampling, dict):
		for key in ("effective_sampling_rate_hz", "target_hz", "analyzer_hz", "raw_hz"):
			val = upsampling.get(key)
			if val is None:
				continue
			try:
				rate = float(val)
				if rate > 0:
					return rate
			except Exception:
				pass

	return None


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
	if bool(getattr(config, "invert_y_axis", True)):
		ax.invert_yaxis()
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
	svg_path: Path | None = None,
	write_svg: bool = False,
	probe_geometry: ProbeGeometryConfig | None = None,
	channel_labels_by_row: list[Any] | None = None,
	trace_order_label_by_channel: dict[int, int] | None = None,
	channel_indices: list[int] | np.ndarray | None = None,
	peak_indices_by_channel: dict[int, list[int] | np.ndarray] | None = None,
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

	order_payload = compute_propagation_channel_order(
		template_c_by_t=t,
		config=config,
		channel_indices=channel_indices,
	)
	selected = np.asarray(order_payload["ordered_channel_indices"], dtype=int)
	lat_idx = np.asarray(order_payload["latency_indices"], dtype=float)
	max_abs_electrode = order_payload.get("max_abs_channel", None)
	max_ptp_electrode = order_payload.get("max_ptp_channel", None)
	selected_abs_max = np.max(np.abs(t[selected, :]), axis=1)
	if bool(getattr(config, "debug_max_amps_at_each_channel", False)):
		amps_by_channel = {
			int(ch): float(amp) for ch, amp in zip(selected.tolist(), selected_abs_max.tolist(), strict=False)
		}
		debug_msg = f"Propagation plot debug: max amplitude at each plotted electrode before gain (uV): {amps_by_channel}"
		print(debug_msg)
		LOGGER.info(debug_msg)
	ref_ch = int(max_abs_electrode) if max_abs_electrode is not None else int(selected[0])
	ref_neg_peak_idx = int(np.argmin(t[ref_ch, :]))

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
	plot_width_in = float(max(4.0, float(getattr(config, "plot_width_in", 13.0))))
	plot_panel_height_in = float(max(0.8, float(getattr(config, "plot_panel_height_in", 2.8))))
	plot_extra_height_in = float(max(0.0, float(getattr(config, "plot_extra_height_in", 1.0))))
	plot_hspace = float(max(0.0, float(getattr(config, "plot_hspace", 0.35))))
	plot_area_aspect_ratio = getattr(config, "plot_area_aspect_ratio", None)
	if plot_area_aspect_ratio is not None:
		try:
			ratio_val = float(plot_area_aspect_ratio)
			if ratio_val > 0.0:
				plot_panel_height_in = float(max(0.8, plot_width_in / ratio_val))
		except Exception:
			pass
	fig = plt.figure(figsize=(plot_width_in, (plot_panel_height_in * n_panels) + plot_extra_height_in))
	gs = fig.add_gridspec(nrows=n_panels, ncols=1, hspace=plot_hspace)
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

	if bool(getattr(config, "show_duration_info", False)):
		sr_hz_for_duration = None if probe_geometry is None else probe_geometry.sampling_rate_hz
		before_samples = int(max(0, ref_neg_peak_idx))
		after_samples = int(max(0, int(t.shape[1]) - ref_neg_peak_idx - 1))
		total_samples = int(max(0, int(t.shape[1])))
		if sr_hz_for_duration is not None and float(sr_hz_for_duration) > 0.0:
			before_ms = float(before_samples / float(sr_hz_for_duration) * 1000.0)
			after_ms = float(after_samples / float(sr_hz_for_duration) * 1000.0)
			total_ms = float(total_samples / float(sr_hz_for_duration) * 1000.0)
			duration_label = (
				f"before: {_format_no_sci(before_ms, max_decimals=3)} ms | "
				f"after: {_format_no_sci(after_ms, max_decimals=3)} ms | "
				f"total: {_format_no_sci(total_ms, max_decimals=3)} ms"
			)
		else:
			duration_label = f"before: {before_samples} samples | after: {after_samples} samples | total: {total_samples} samples"
		duration_ha = str(getattr(config, "duration_info_horizontal_alignment", "left") or "left").strip().lower()
		if duration_ha not in {"left", "center", "right"}:
			duration_ha = "left"
		duration_va = str(getattr(config, "duration_info_vertical_alignment", "top") or "top").strip().lower()
		if duration_va not in {"top", "center", "bottom"}:
			duration_va = "top"
		trace_axes[0].text(
			float(getattr(config, "duration_info_x_frac", 0.01)),
			float(getattr(config, "duration_info_y_frac", 0.99)),
			duration_label,
			transform=trace_axes[0].transAxes,
			fontsize=float(max(1.0, float(getattr(config, "duration_info_fontsize", 6.0)))),
			color=text_color,
			horizontalalignment=duration_ha,
			verticalalignment=duration_va,
		)

	x = np.arange(int(t.shape[1]), dtype=float)
	n_samples = int(t.shape[1])
	cut_start_idx: int | None = None
	cut_end_idx: int | None = None
	cut_len_samples = 0
	gap_samples = int(max(0, int(getattr(config, "post_ap_abbrev_gap_samples", 8))))
	shift_samples = 0
	marker_center_x: float | None = None
	if bool(getattr(config, "abbreviate_post_ap_signal", False)) and n_samples > 6:
		post_peak_remaining = int(max(0, n_samples - (ref_neg_peak_idx + 1)))
		sr_hz = None if probe_geometry is None else probe_geometry.sampling_rate_hz
		if sr_hz is not None and float(sr_hz) > 0.0:
			start_ms = float(max(0.0, float(getattr(config, "post_ap_abbrev_start_ms", 1.0))))
			start_after_peak_ms_samples = int(max(0, int(round((start_ms / 1000.0) * float(sr_hz)))))
			start_after_peak_samples = int(start_after_peak_ms_samples)
		else:
			start_after_peak_samples = int(max(0, int(getattr(config, "post_ap_abbrev_start_samples", 10))))
		# If ms-based start would land beyond the available post-peak tail,
		# fall back to sample-based knob to avoid a near no-op cut.
		fallback_start_samples = int(max(0, int(getattr(config, "post_ap_abbrev_start_samples", 10))))
		max_start_offset = int(max(0, post_peak_remaining - 3))
		if max_start_offset > 0 and start_after_peak_samples > max_start_offset and fallback_start_samples > 0:
			start_after_peak_samples = int(min(fallback_start_samples, max_start_offset))
		cut_start_candidate = int(ref_neg_peak_idx + start_after_peak_samples)
		cut_start_candidate = int(min(max(1, cut_start_candidate), n_samples - 3))
		cut_fraction = float(min(0.95, max(0.0, float(getattr(config, "post_ap_abbrev_cut_fraction", 0.5)))))
		desired_cut_len = int(np.floor(float(post_peak_remaining) * cut_fraction))
		available_after_start = int(max(0, (n_samples - 2) - cut_start_candidate))
		cut_len_candidate = int(min(desired_cut_len, available_after_start))
		min_cut = int(max(1, int(getattr(config, "post_ap_abbrev_min_samples_to_cut", 5))))
		if cut_len_candidate >= min_cut and (available_after_start - cut_len_candidate) >= 0:
			cut_start_idx = cut_start_candidate
			cut_end_idx = int(cut_start_candidate + cut_len_candidate)
			cut_len_samples = int(cut_len_candidate)
			gap_samples = int(min(max(0, gap_samples), max(0, cut_len_samples - 1)))
			shift_samples = int(max(0, cut_len_samples - gap_samples))
			marker_center_x = float(cut_start_idx) + (0.5 * float(gap_samples))

	def _map_sample_to_plot_x(sample_idx: float) -> float:
		if cut_start_idx is None or cut_end_idx is None or cut_len_samples <= 0:
			return float(sample_idx)
		if float(sample_idx) <= float(cut_start_idx):
			return float(sample_idx)
		if float(sample_idx) >= float(cut_end_idx):
			return float(sample_idx) - float(shift_samples)
		return float(sample_idx)

	base_step = float(max(1e-6, np.max(np.ptp(t[selected, :], axis=1))))
	offset_step = base_step * max(0.2, float(config.trace_spacing))
	trace_gain = float(max(1e-9, float(config.trace_gain)))
	peak_marker_height_frac = float(max(1e-6, float(getattr(config, "peak_marker_height_frac", 0.24))))
	peak_marker_linewidth = float(max(0.2, float(getattr(config, "peak_marker_linewidth", 1.4))))
	delay_peak_marker_color = str(getattr(config, "delay_peak_marker_color", "black") or "black")
	show_multiple_peak_markers = bool(getattr(config, "show_multiple_peak_markers", False))
	label_alignment = str(getattr(config, "electrode_label_alignment", getattr(config, "channel_label_alignment", "left")) or "left").strip().lower()
	if label_alignment not in {"left", "center", "right"}:
		label_alignment = "left"
	label_mode = str(getattr(config, "trace_label_mode", "electrode_id") or "electrode_id").strip().lower()
	order_index_label_by_channel: dict[int, int] | None = None
	if label_mode == "order_index" and int(selected.shape[0]) > 0:
		if trace_order_label_by_channel is not None:
			order_index_label_by_channel = {
				int(ch): int(val)
				for ch, val in trace_order_label_by_channel.items()
			}
		elif max_ptp_electrode is not None:
			anchor_hits = np.flatnonzero(selected == int(max_ptp_electrode))
			if anchor_hits.size > 0:
				anchor_pos = int(anchor_hits[0])
				order_index_label_by_channel = {
					int(ch): int(i - anchor_pos)
					for i, ch in enumerate(selected.tolist())
				}

	highlight_label_channel = max_abs_electrode
	if label_mode == "order_index":
		if order_index_label_by_channel is not None:
			zero_channels = [int(ch) for ch, v in order_index_label_by_channel.items() if int(v) == 0]
			for ch in zero_channels:
				if np.any(selected == int(ch)):
					highlight_label_channel = int(ch)
					break
		elif max_ptp_electrode is not None:
			highlight_label_channel = int(max_ptp_electrode)
	for panel_i, panel_inds in enumerate(panels):
		ax = trace_axes[panel_i]
		label_x_offset = float(getattr(config, "electrode_label_x_offset_frac", getattr(config, "channel_label_x_offset_frac", 0.01))) * float(max(1, x.shape[0] - shift_samples))
		label_y_offset = float(getattr(config, "electrode_label_y_offset_frac", getattr(config, "channel_label_y_offset_frac", 0.0))) * float(offset_step)
		min_label_x: float | None = None
		for i_local, idx in enumerate(panel_inds):
			ch = int(selected[int(idx)])
			label_id: Any = ch
			if channel_labels_by_row is not None and 0 <= int(ch) < int(len(channel_labels_by_row)):
				candidate = channel_labels_by_row[int(ch)]
				if candidate is not None:
					label_id = candidate
			label_text: str | None = None
			if label_mode == "order_index":
				if order_index_label_by_channel is not None and int(ch) in order_index_label_by_channel:
					label_text = str(int(order_index_label_by_channel[int(ch)]))
			elif bool(getattr(config, "show_electrode_ids", False)):
				label_text = f"eid {label_id}"
			off = float(i_local) * offset_step
			y = (t[ch, :] * trace_gain) + off
			if cut_start_idx is not None and cut_end_idx is not None and shift_samples > 0:
				left_slice = slice(0, cut_start_idx + 1)
				right_slice = slice(cut_end_idx, n_samples)
				x_left = x[left_slice]
				y_left = y[left_slice]
				if x_left.size > 1:
					ax.plot(x_left, y_left, color=trace_color, linewidth=0.9, alpha=0.95)
				x_right_raw = x[right_slice]
				y_right = y[right_slice]
				if x_right_raw.size > 1:
					x_right = x_right_raw - float(shift_samples)
					ax.plot(x_right, y_right, color=trace_color, linewidth=0.9, alpha=0.95)
				marker_text = str(getattr(config, "post_ap_abbrev_marker_text", "/.../") or "").strip()
				if marker_text and marker_center_x is not None:
					y_l = float(y[int(cut_start_idx)])
					y_r = float(y[int(cut_end_idx)])
					marker_y = 0.5 * (y_l + y_r)
					marker_y += float(getattr(config, "post_ap_abbrev_marker_y_offset_frac", 0.0)) * float(offset_step)
					ax.text(
						float(marker_center_x),
						marker_y,
						marker_text,
						color=text_color,
						fontsize=float(max(1.0, float(getattr(config, "post_ap_abbrev_marker_fontsize", 7.0)))),
						horizontalalignment="center",
						verticalalignment="center",
					)
			else:
				ax.plot(x, y, color=trace_color, linewidth=0.9, alpha=0.95)
			delay_pk = int(np.argmax(np.abs(t[ch, :])))
			if 0 <= int(idx) < int(lat_idx.shape[0]):
				try:
					latency_pk = int(round(float(lat_idx[int(idx)])))
					if 0 <= latency_pk < int(t.shape[1]):
						delay_pk = latency_pk
				except Exception:
					pass
			marker_height = float(max(1.2, peak_marker_height_frac * offset_step))
			marker_half = float(0.5 * marker_height)

			if show_multiple_peak_markers and peak_indices_by_channel is not None:
				raw_indices = peak_indices_by_channel.get(int(ch), None)
				if raw_indices is not None:
					for extra_pk in np.asarray(raw_indices, dtype=int).reshape(-1):
						extra_pk_i = int(extra_pk)
						if extra_pk_i < 0 or extra_pk_i >= int(t.shape[1]):
							continue
						if extra_pk_i == int(delay_pk):
							continue
						extra_pk_plot = _map_sample_to_plot_x(float(extra_pk_i))
						extra_peak_y = float(y[extra_pk_i])
						ax.plot(
							[extra_pk_plot, extra_pk_plot],
							[extra_peak_y - marker_half, extra_peak_y + marker_half],
							color="black",
							linewidth=peak_marker_linewidth,
							solid_capstyle="butt",
						)

			delay_pk_plot = _map_sample_to_plot_x(float(delay_pk))
			delay_peak_y = float(y[delay_pk])
			ax.plot(
				[delay_pk_plot, delay_pk_plot],
				[delay_peak_y - marker_half, delay_peak_y + marker_half],
				color=delay_peak_marker_color,
				linewidth=peak_marker_linewidth,
				solid_capstyle="butt",
			)
			if label_text is not None:
				ax.text(
					_map_sample_to_plot_x(x[0]) + label_x_offset,
					off + label_y_offset,
					label_text,
					color=text_color,
					fontsize=float(getattr(config, "electrode_label_fontsize", getattr(config, "channel_label_fontsize", 6.0))),
					fontweight=(
						"bold"
						if bool(getattr(config, "bold_max_amp_electrode_label", getattr(config, "bold_max_amp_channel_label", False))) and highlight_label_channel is not None and int(ch) == int(highlight_label_channel)
						else "normal"
					),
					horizontalalignment=label_alignment,
					verticalalignment="center",
				)
				label_anchor_x = float(_map_sample_to_plot_x(x[0]) + label_x_offset)
				if min_label_x is None:
					min_label_x = label_anchor_x
				else:
					min_label_x = min(min_label_x, label_anchor_x)

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
		# Marker is drawn per-trace above to align with each abbreviated waveform.

	if bool(config.show_scale_bar):
		max_trace_amp_units = float(np.max(np.abs(t[selected, :])))
		n_display_samples = int(max(2, int(t.shape[1]) - int(shift_samples)))
		_add_propagation_scale_bars(
			ax=trace_axes[-1],
			n_samples=n_display_samples,
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
		png_dpi = 220.0
		if bool(getattr(config, "show_right_panel", False)):
			left_cfg = getattr(config, "left_panel_png_dpi", None)
			if left_cfg is None:
				left_cfg = getattr(config, "right_panel_png_dpi", 300.0)
			png_dpi = float(max(72.0, float(left_cfg)))
		fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["propagation_plot_png"] = str(png_path)
	if bool(config.write_pdf):
		pdf_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["propagation_plot_pdf"] = str(pdf_path)
	if bool(write_svg) and svg_path is not None:
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["propagation_plot_svg"] = str(svg_path)

	plt.close(fig)
	return outputs


def render_template_wf_overlay(
	*,
	template: Any,
	config: TemplateWaveformOverlayConfig,
	time_upsample: TimeUpsampleConfig,
	pdf_path: Path,
	png_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
	waveform_traces: Any | None = None,
	top_electrode_id: Any | None = None,
	top_channel_id: Any | None = None,
	total_waveforms_at_channel: int | None = None,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	debug_mode = bool(getattr(config, "debug_mode", False))

	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111)
	if str(config.background or "").strip().lower() == "black":
		fig.patch.set_facecolor("black")
	else:
		fig.patch.set_facecolor("white")
	_draw_template_wf_overlay_panel(
		ax=ax,
		template=template,
		config=config,
		time_upsample=time_upsample,
		probe_geometry=probe_geometry,
		waveform_traces=waveform_traces,
		top_electrode_id=top_electrode_id,
		top_channel_id=top_channel_id,
		total_waveforms_at_channel=total_waveforms_at_channel,
	)

	outputs: dict[str, str] = {}
	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_wf_overlay_png"] = str(png_path)
		outputs["extremum_ch_wf_overlay_png"] = str(png_path)
	if bool(config.write_pdf):
		pdf_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
		outputs["template_wf_overlay_pdf"] = str(pdf_path)
		outputs["extremum_ch_wf_overlay_pdf"] = str(pdf_path)

	if debug_mode:
		print(
			"[template_wf_overlay][debug] "
			f"wrote_png={outputs.get('template_wf_overlay_png')} "
			f"wrote_pdf={outputs.get('template_wf_overlay_pdf')}",
			flush=True,
		)

	plt.close(fig)
	return outputs


def _draw_template_wf_overlay_panel(
	*,
	ax: Any,
	template: Any,
	config: TemplateWaveformOverlayConfig,
	time_upsample: TimeUpsampleConfig,
	probe_geometry: ProbeGeometryConfig | None = None,
	waveform_traces: Any | None = None,
	top_electrode_id: Any | None = None,
	top_channel_id: Any | None = None,
	total_waveforms_at_channel: int | None = None,
) -> None:
	t = np.asarray(template)
	if t.ndim != 2:
		raise ValueError(f"Expected 2D template array for overlay, got shape={getattr(t, 'shape', None)}")
	t = _time_upsample_template(t, time_upsample)

	n_channels, n_samples = int(t.shape[0]), int(t.shape[1])
	if n_channels <= 0 or n_samples <= 0:
		raise ValueError("Template overlay received empty template")

	effective_sr_hz: float | None = None
	if probe_geometry is not None and probe_geometry.sampling_rate_hz is not None and float(probe_geometry.sampling_rate_hz) > 0.0:
		effective_sr_hz = float(probe_geometry.sampling_rate_hz)
	if top_electrode_id is None and top_channel_id is not None:
		top_electrode_id = top_channel_id

	wf_all = None
	if waveform_traces is not None:
		wf_arr = np.asarray(waveform_traces, dtype=float)
		if wf_arr.ndim == 2 and int(wf_arr.shape[0]) > 0 and int(wf_arr.shape[1]) > 1:
			wf_all = _time_upsample_template(wf_arr, time_upsample)

	if wf_all is not None:
		n_total = int(wf_all.shape[0])
		n_show_target = int(max(1, int(getattr(config, "max_waveforms_to_show", 100))))
		n_show = int(min(n_total, n_show_target))
		mode = str(getattr(config, "waveform_sampling_mode", "uniform") or "uniform").strip().lower()
		if n_show >= n_total:
			show_idx = np.arange(n_total, dtype=int)
		elif mode in {"random", "rand"}:
			seed = getattr(config, "random_seed", 0)
			rng = np.random.default_rng(None if seed is None else int(seed))
			show_idx = np.sort(rng.choice(n_total, size=n_show, replace=False).astype(int))
		elif mode in {"first", "head"}:
			show_idx = np.arange(n_show, dtype=int)
		else:
			show_idx = np.linspace(0, n_total - 1, n_show, dtype=int)
		wf_show = wf_all[show_idx, :]
		mean_wave = np.mean(wf_all, axis=0) if bool(config.include_mean) else None
		offset_step = float(max(1e-6, np.max(np.ptp(wf_show, axis=1)) * 1.2))
	else:
		ptp = np.ptp(t, axis=1)
		top_n = max(1, int(config.top_channels_per_template))
		order = np.argsort(-ptp)
		selected = order[: min(top_n, n_channels)]
		wf_show = np.asarray(t[selected, :], dtype=float)
		n_total = int(wf_show.shape[0])
		mean_wave = np.mean(wf_show, axis=0) if bool(config.include_mean) else None
		offset_step = float(max(1e-6, np.max(np.ptp(wf_show, axis=1)) * 1.2))

	x = np.arange(int(wf_show.shape[1]), dtype=float)

	bg = str(config.background or "").strip().lower()
	if bg == "black":
		ax.set_facecolor("black")
		for spine in ax.spines.values():
			spine.set_color("white")
		trace_color = "white"
		mean_color = "cyan"
	else:
		ax.set_facecolor("white")
		trace_color = "black"
		mean_color = "red"

	style = str(getattr(config, "style", "overlay") or "overlay").strip().lower()
	if style in {"stack", "stacked"}:
		for idx in range(int(wf_show.shape[0])):
			offset = float(idx) * offset_step
			ax.plot(x, wf_show[idx, :] + offset, color=trace_color, linewidth=0.9, alpha=0.7)
	else:
		for idx in range(int(wf_show.shape[0])):
			ax.plot(x, wf_show[idx, :], color=trace_color, linewidth=0.9, alpha=0.25)

	if mean_wave is not None:
		mean_offset = float(wf_show.shape[0]) * offset_step if style in {"stack", "stacked"} else 0.0
		ax.plot(x, mean_wave + mean_offset, color=mean_color, linewidth=1.5, alpha=0.95)
		if bool(getattr(config, "show_channel_labels", False)):
			ax.text(
				x[0],
				float(mean_wave[0]) + mean_offset,
				"mean",
				fontsize=6,
				color=mean_color,
				verticalalignment="bottom",
				horizontalalignment="left",
			)

	info_lines: list[str] = []
	if bool(getattr(config, "show_top_channel_info", True)):
		label = "unknown" if top_electrode_id is None else str(top_electrode_id)
		info_lines.append(f"extremum eid: {label}")
	if bool(getattr(config, "show_waveform_count_info", True)):
		total_count = int(n_total if total_waveforms_at_channel is None else max(0, int(total_waveforms_at_channel)))
		info_lines.append(f"wfs at eid: {total_count}")
		info_lines.append(f"wfs shown: {int(wf_show.shape[0])}")
	if info_lines:
		ax.text(
			0.01,
			0.99,
			"\n".join(info_lines),
			transform=ax.transAxes,
			fontsize=6,
			color=trace_color,
			horizontalalignment="left",
			verticalalignment="top",
		)

	if bool(getattr(config, "show_axes", False)):
		ax.set_xlabel("sample")
		ax.set_ylabel("amplitude + offset" if style in {"stack", "stacked"} else "amplitude")
	else:
		ax.set_xticks([])
		ax.set_yticks([])
		for spine in ax.spines.values():
			spine.set_visible(False)

	if bool(getattr(config, "show_title", False)):
		ax.set_title(f"Extremum-channel waveforms (style={style})")

	if bool(config.include_scale_bar):
		x0, x1 = ax.get_xlim()
		y0, y1 = ax.get_ylim()
		span_x = max(1.0, float(abs(x1 - x0)))
		span_y = max(1.0, float(abs(y1 - y0)))
		time_fraction = float(min(1.0, max(1e-6, float(getattr(config, "scale_bar_time_fraction", 0.10)))))
		amp_fraction = float(min(1.0, max(1e-6, float(getattr(config, "scale_bar_amp_fraction", 0.10)))))
		time_label_offset_frac = float(max(0.0, float(getattr(config, "scale_bar_time_label_offset_frac", 0.03))))
		amp_label_offset_frac = float(max(0.0, float(getattr(config, "scale_bar_amp_label_offset_frac", 0.02))))
		bar_x = float(min(max(1.0, _nice_scale_value(span_x * time_fraction)), span_x * 0.30))
		bar_y = float(min(max(1e-6, _nice_scale_value(span_y * amp_fraction)), span_y * 0.30))
		x_left = float(min(x0, x1)) + 0.05 * span_x
		y_bot = float(min(y0, y1)) + 0.08 * span_y
		color = str(config.scale_bar_color)
		lw = float(config.scale_bar_linewidth)
		ax.plot([x_left, x_left + bar_x], [y_bot, y_bot], color=color, lw=lw, solid_capstyle="butt")
		ax.plot([x_left, x_left], [y_bot, y_bot + bar_y], color=color, lw=lw, solid_capstyle="butt")
		if effective_sr_hz is not None and float(effective_sr_hz) > 0.0:
			time_ms = float((bar_x / float(effective_sr_hz)) * 1000.0)
			time_label = f"{_format_no_sci(time_ms, max_decimals=3)} ms"
		else:
			time_label = f"{int(round(bar_x))} samples"
		amp_label = f"{_format_no_sci(float(bar_y), max_decimals=3)} uV"
		ax.text(
			x_left + (bar_x * 0.5),
			y_bot - (time_label_offset_frac * span_y),
			time_label,
			fontsize=float(config.scale_bar_fontsize),
			color=color,
			horizontalalignment="center",
			verticalalignment="top",
		)
		ax.text(
			x_left - (amp_label_offset_frac * span_x),
			y_bot + (bar_y * 0.5),
			amp_label,
			fontsize=float(config.scale_bar_fontsize),
			color=color,
			horizontalalignment="right",
			verticalalignment="center",
			rotation=90,
		)


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


def render_template_report_pdf(
	*,
	units: list[dict[str, Any]],
	pdf_path: Path,
	output_key: str = "template_report_pdf",
	title_suffix: str = "circle template",
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	from matplotlib.backends.backend_pdf import PdfPages  # type: ignore[import-not-found]

	pdf_path.parent.mkdir(parents=True, exist_ok=True)
	with PdfPages(pdf_path) as pdf:
		for unit in units:
			unit_id = unit.get("unit_id")
			image_path = Path(str(unit.get("image_path")))
			if not image_path.exists():
				continue

			fig, ax = plt.subplots(figsize=(11.0, 8.5))
			ax.imshow(plt.imread(image_path))
			ax.axis("off")
			fig.suptitle(f"Unit {unit_id} {str(title_suffix).strip()}", fontsize=16)
			pdf.savefig(fig, bbox_inches="tight")
			plt.close(fig)

	return {str(output_key): str(pdf_path)}


def render_unit_locations_report(
	*,
	unit_location_rows: list[dict[str, Any]],
	config: UnitLocationsReportConfig,
	json_path: Path,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
	concat_channel_locations_xy: Any | None = None,
	template_channel_locations_by_unit: dict[str, Any] | None = None,
	original_unit_locations_by_unit: dict[str, Any] | None = None,
) -> dict[str, str]:
	def _coerce_xy_points(raw: Any) -> np.ndarray:
		arr = np.asarray(raw, dtype=float)
		if arr.ndim != 2 or int(arr.shape[1]) < 2:
			return np.zeros((0, 2), dtype=float)
		arr = np.asarray(arr[:, :2], dtype=float)
		if int(arr.shape[0]) == 0:
			return np.zeros((0, 2), dtype=float)
		mask = np.isfinite(arr).all(axis=1)
		if not bool(np.any(mask)):
			return np.zeros((0, 2), dtype=float)
		return np.asarray(arr[mask, :], dtype=float)

	rows_clean: list[dict[str, Any]] = []
	for row in list(unit_location_rows):
		if not isinstance(row, dict):
			continue
		x = row.get("x_um", None)
		y = row.get("y_um", None)
		try:
			x_f = float(x)
			y_f = float(y)
		except Exception:
			continue
		if (not np.isfinite(x_f)) or (not np.isfinite(y_f)):
			continue
		rows_clean.append(
			{
				"unit_id": row.get("unit_id", None),
				"x_um": float(x_f),
				"y_um": float(y_f),
				"source": row.get("source", None),
				"channel_index": row.get("channel_index", None),
			}
		)

	concat_locs_clean = _coerce_xy_points(concat_channel_locations_xy)
	template_locs_clean: dict[str, np.ndarray] = {}
	if isinstance(template_channel_locations_by_unit, dict):
		for unit_key, locs_raw in template_channel_locations_by_unit.items():
			locs = _coerce_xy_points(locs_raw)
			if int(locs.shape[0]) == 0:
				continue
			template_locs_clean[str(unit_key)] = locs
	original_locations_clean: dict[str, tuple[float, float]] = {}
	if isinstance(original_unit_locations_by_unit, dict):
		for unit_key, loc_raw in original_unit_locations_by_unit.items():
			if not isinstance(loc_raw, dict):
				continue
			try:
				x = float(loc_raw.get("x_um", None))
				y = float(loc_raw.get("y_um", None))
			except Exception:
				continue
			if (not np.isfinite(x)) or (not np.isfinite(y)):
				continue
			key = str(unit_key).strip()
			original_locations_clean[key] = (x, y)
			try:
				original_locations_clean[str(int(unit_key))] = (x, y)
			except Exception:
				pass

	outputs: dict[str, str] = {}
	if bool(config.write_json):
		json_path.parent.mkdir(parents=True, exist_ok=True)
		with open(json_path, "w", encoding="utf-8") as f:
			json.dump(rows_clean, f, indent=2)
		outputs["unit_locations_json"] = str(json_path)

	if not (bool(config.write_png) or bool(config.write_svg)):
		return outputs

	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	from matplotlib.patches import Rectangle  # type: ignore[import-not-found]

	chip_width_um = 3850.0
	chip_height_um = 2100.0
	if bool(getattr(config, "use_probe_active_area", True)) and probe_geometry is not None:
		try:
			px = probe_geometry.active_area_um_x
			py = probe_geometry.active_area_um_y
			if px is not None and float(px) > 0.0 and np.isfinite(float(px)):
				chip_width_um = float(px)
			if py is not None and float(py) > 0.0 and np.isfinite(float(py)):
				chip_height_um = float(py)
		except Exception:
			pass

	chip_ratio = float(max(1e-9, chip_width_um) / max(1e-9, chip_height_um))
	fig_height_in = 4.8
	fig_width_in = float(np.clip(fig_height_in * chip_ratio, 6.0, 14.0))
	fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
	fig.patch.set_facecolor(str(config.background))
	ax.set_facecolor(str(config.background))

	chip_origin_x = 0.0
	chip_origin_y = 0.0
	if int(concat_locs_clean.shape[0]) > 0:
		chip_origin_x = float(np.min(concat_locs_clean[:, 0]))
		chip_origin_y = float(np.min(concat_locs_clean[:, 1]))
	elif template_locs_clean:
		all_template_locs = np.vstack(list(template_locs_clean.values()))
		chip_origin_x = float(np.min(all_template_locs[:, 0]))
		chip_origin_y = float(np.min(all_template_locs[:, 1]))
	pitch = None if probe_geometry is None else getattr(probe_geometry, "pitch_um", None)
	try:
		pitch_f = float(pitch) if pitch is not None else None
		if pitch_f is not None and np.isfinite(pitch_f) and pitch_f > 0.0:
			chip_origin_x = float(np.floor(chip_origin_x / pitch_f) * pitch_f)
			chip_origin_y = float(np.floor(chip_origin_y / pitch_f) * pitch_f)
	except Exception:
		pass

	ax.add_patch(
		Rectangle(
			(float(chip_origin_x), float(chip_origin_y)),
			float(chip_width_um),
			float(chip_height_um),
			fill=False,
			edgecolor="#666666",
			linewidth=0.8,
			alpha=0.7,
			zorder=0,
		)
	)

	if bool(config.underlay_concat_channels) and int(concat_locs_clean.shape[0]) > 0:
		ax.scatter(
			concat_locs_clean[:, 0],
			concat_locs_clean[:, 1],
			s=float(max(0.0, float(config.concat_channel_scatter_size))),
			c=str(config.concat_channel_scatter_color),
			alpha=float(min(1.0, max(0.0, float(config.concat_channel_scatter_alpha)))),
			linewidths=0.0,
			zorder=1,
		)

	if bool(config.underlay_template_channels) and template_locs_clean:
		unit_items = sorted(template_locs_clean.items(), key=lambda item: item[0])
		cmap = plt.get_cmap(str(config.template_channel_colormap or "tab20"))
		den = max(1, len(unit_items) - 1)
		for idx, (_, locs) in enumerate(unit_items):
			color = cmap(float(idx) / float(den))
			ax.scatter(
				locs[:, 0],
				locs[:, 1],
				s=float(max(0.0, float(config.template_channel_scatter_size))),
				c=[color],
				alpha=float(min(1.0, max(0.0, float(config.template_channel_scatter_alpha)))),
				linewidths=0.0,
				zorder=2,
			)

	if rows_clean:
		xs = np.asarray([float(row["x_um"]) for row in rows_clean], dtype=float)
		ys = np.asarray([float(row["y_um"]) for row in rows_clean], dtype=float)
		if bool(config.show_original_to_current_redlines) and original_locations_clean:
			for row in rows_clean:
				uid_raw = row.get("unit_id", "")
				uid_key = str(uid_raw).strip()
				orig = original_locations_clean.get(uid_key, None)
				if orig is None:
					try:
						orig = original_locations_clean.get(str(int(uid_raw)), None)
					except Exception:
						orig = None
				if orig is None:
					continue
				ax.plot(
					[float(orig[0]), float(row["x_um"])],
					[float(orig[1]), float(row["y_um"])],
					color=str(config.redline_color),
					alpha=float(min(1.0, max(0.0, float(config.redline_alpha)))),
					linewidth=float(max(0.1, float(config.redline_linewidth))),
					zorder=2,
				)

		ax.scatter(
			xs,
			ys,
			s=float(max(0.0, float(config.chip_scatter_size))),
			c=str(config.chip_scatter_color),
			alpha=float(min(1.0, max(0.0, float(config.chip_scatter_alpha)))),
			linewidths=0.0,
			zorder=3,
		)

		dx = float(max(chip_width_um, (np.max(xs) - np.min(xs)))) if int(xs.shape[0]) > 1 else float(chip_width_um)
		dy = float(max(chip_height_um, (np.max(ys) - np.min(ys)))) if int(ys.shape[0]) > 1 else float(chip_height_um)
		x_offset = float(config.unit_id_label_x_offset_frac) * float(max(1e-9, dx))
		y_offset = float(config.unit_id_label_y_offset_frac) * float(max(1e-9, dy))
		if bool(config.show_unit_id_labels):
			for row in rows_clean:
				ax.text(
					float(row["x_um"]) + x_offset,
					float(row["y_um"]) + y_offset,
					str(row.get("unit_id", "")),
					fontsize=float(max(1.0, float(config.unit_id_label_fontsize))),
					color=str(config.unit_id_label_color),
					ha=str(config.unit_id_label_horizontal_alignment),
					va=str(config.unit_id_label_vertical_alignment),
					zorder=4,
				)

	if (not rows_clean) and int(concat_locs_clean.shape[0]) == 0 and (not template_locs_clean):
		ax.text(0.5, 0.5, "No unit locations", color="white", ha="center", va="center", transform=ax.transAxes)

	reference_sets: list[np.ndarray] = []
	reference_sets.append(
		np.asarray(
			[
				[float(chip_origin_x), float(chip_origin_y)],
				[float(chip_origin_x + chip_width_um), float(chip_origin_y + chip_height_um)],
			],
			dtype=float,
		)
	)
	if int(concat_locs_clean.shape[0]) > 0:
		reference_sets.append(concat_locs_clean)
	for locs in template_locs_clean.values():
		reference_sets.append(locs)
	if original_locations_clean:
		reference_sets.append(np.asarray(list(original_locations_clean.values()), dtype=float))
	if rows_clean:
		reference_sets.append(
			np.asarray(
				[[float(row["x_um"]), float(row["y_um"])] for row in rows_clean],
				dtype=float,
			)
		)

	reference_xy = np.vstack(reference_sets)
	x_ref_min = float(np.min(reference_xy[:, 0]))
	x_ref_max = float(np.max(reference_xy[:, 0]))
	y_ref_min = float(np.min(reference_xy[:, 1]))
	y_ref_max = float(np.max(reference_xy[:, 1]))

	xpad = float(max(5.0, 0.02 * float(chip_width_um)))
	ypad = float(max(5.0, 0.02 * float(chip_height_um)))
	ax.set_xlim(float(x_ref_min - xpad), float(x_ref_max + xpad))
	ax.set_ylim(float(y_ref_min - ypad), float(y_ref_max + ypad))

	ax.set_xlabel("x (um)")
	ax.set_ylabel("y (um)")
	ax.set_title("Unit Locations")
	ax.set_aspect("equal", adjustable="box")
	if bool(config.invert_y_axis):
		ax.invert_yaxis()

	if bool(config.write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=300, bbox_inches="tight")
		outputs["unit_locations_png"] = str(png_path)
	if bool(config.write_svg):
		svg_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(svg_path, format="svg", bbox_inches="tight")
		outputs["unit_locations_svg"] = str(svg_path)
	plt.close(fig)
	return outputs


def render_wf_overlay_grid_from_assets(
	*,
	overlay_png_paths: list[Path],
	config: WfOverlayGridReportConfig,
	pdf_path: Path,
	png_path: Path,
	write_svg: bool = False,
	svg_path: Path | None = None,
	svg_output_key: str = "wf_overlay_grid_svg",
) -> dict[str, str]:
	"""Compose waveform overlay grid outputs from pre-rendered per-unit assets."""
	return render_image_grid(
		image_paths=overlay_png_paths,
		write_pdf=bool(config.write_pdf),
		pdf_path=pdf_path,
		pdf_output_key="wf_overlay_grid_pdf",
		write_png=bool(config.write_png),
		png_path=png_path,
		png_output_key="wf_overlay_grid_png",
		write_svg=bool(write_svg),
		svg_path=svg_path,
		svg_output_key=svg_output_key,
		title="Template waveform overlay grid",
		dpi=max(72.0, float(getattr(config, "dpi", 300.0))),
	)


def render_footprint_map_grid_from_assets(
	*,
	image_paths: list[Path],
	config: FootprintMapGridReportConfig,
	pdf_path: Path,
	png_path: Path,
	write_svg: bool = False,
	svg_path: Path | None = None,
	svg_output_key: str = "footprint_map_grid_svg",
	pdf_output_key: str,
	png_output_key: str,
	title: str,
) -> dict[str, str]:
	"""Compose footprint grid outputs from pre-rendered per-unit assets."""
	return render_image_grid(
		image_paths=image_paths,
		write_pdf=bool(config.write_pdf),
		pdf_path=pdf_path,
		pdf_output_key=pdf_output_key,
		write_png=bool(config.write_png),
		png_path=png_path,
		png_output_key=png_output_key,
		write_svg=bool(write_svg),
		svg_path=svg_path,
		svg_output_key=svg_output_key,
		title=title,
		show_title=bool(getattr(config, "show_title", True)),
		dpi=max(72.0, float(getattr(config, "dpi", 300.0))),
	)


def finalize_grid_svg_output(
	*,
	raw_outputs: dict[str, str],
	write_svg: bool,
	keep_temp_svg: bool,
	temp_svg_output_key: str,
	final_svg_output_key: str,
	temp_svg_path: Path,
	final_svg_path: Path,
	report_name: str,
	logger: logging.Logger | None = None,
) -> dict[str, str]:
	active_logger = LOGGER if logger is None else logger
	if not bool(write_svg):
		raw_outputs.pop(temp_svg_output_key, None)
		return raw_outputs
	if temp_svg_output_key not in raw_outputs:
		return raw_outputs
	try:
		final_svg_path.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(temp_svg_path, final_svg_path)
		raw_outputs[final_svg_output_key] = str(final_svg_path)
	except Exception as exc:
		active_logger.warning("Failed to finalize %s SVG output: %s", report_name, exc)
	if not bool(keep_temp_svg):
		raw_outputs.pop(temp_svg_output_key, None)
		try:
			if temp_svg_path.exists():
				temp_svg_path.unlink()
		except Exception:
			pass
	return raw_outputs


def render_image_grid(
	*,
	image_paths: list[Path],
	write_pdf: bool,
	pdf_path: Path,
	pdf_output_key: str,
	write_png: bool,
	png_path: Path,
	png_output_key: str,
	write_svg: bool,
	svg_path: Path | None,
	svg_output_key: str,
	title: str,
	show_title: bool = True,
	dpi: float = 200.0,
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

	if bool(show_title):
		fig.suptitle(title, fontsize=10)

	outputs: dict[str, str] = {}
	if bool(write_png):
		png_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(png_path, dpi=max(72.0, float(dpi)), bbox_inches="tight")
		outputs[png_output_key] = str(png_path)
	if bool(write_pdf):
		pdf_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
		outputs[pdf_output_key] = str(pdf_path)
	if bool(write_svg) and svg_path is not None:
		panel_svg_paths = [Path(str(p)).with_suffix(".svg") for p in paths]
		if panel_svg_paths and all(p.exists() for p in panel_svg_paths):
			try:
				compose_svg_grid(
					panel_svg_paths=panel_svg_paths,
					output_svg_path=svg_path,
					ncols=ncols,
					show_title=bool(show_title),
					title=str(title),
				)
				outputs[svg_output_key] = str(svg_path)
			except Exception:
				svg_path.parent.mkdir(parents=True, exist_ok=True)
				fig.savefig(svg_path, format="svg", bbox_inches="tight")
				outputs[svg_output_key] = str(svg_path)
		else:
			svg_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(svg_path, format="svg", bbox_inches="tight")
			outputs[svg_output_key] = str(svg_path)

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
	branch_morphology: Any | None = None,
	branch_cfg: Any | None = None,
	fig: Any | None = None,
	ax: Any | None = None,
	close_figure: bool = True,
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

	if (fig is None) != (ax is None):
		raise ValueError("_render_footprint_map requires both fig and ax when reusing an existing host")
	if fig is None or ax is None:
		fig = plt.figure(figsize=(8, 6))
		ax = fig.add_subplot(111)
	else:
		ax.clear()
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
	if bool(getattr(config, "invert_y_axis", True)):
		ax.invert_yaxis()
	ax.set_xlabel("x (um)", color=text_color)
	ax.set_ylabel("y (um)", color=text_color)
	ax.set_title(title, color=text_color)

	if branch_cfg is not None and bool(getattr(branch_cfg, "enabled", False)):
		try:
			fig.canvas.draw()
			origin_px = np.asarray(ax.transData.transform((0.0, 0.0)), dtype=float)
			x_px = np.asarray(ax.transData.transform((float(dx), 0.0)), dtype=float)
			y_px = np.asarray(ax.transData.transform((0.0, float(dy))), dtype=float)
			radius_px = float(max(0.0, min(abs(float(x_px[0] - origin_px[0])), abs(float(y_px[1] - origin_px[1]))) / 2.0))
			radius_pt = float(radius_px * (72.0 / float(fig.dpi)))
			size_pt2 = float(np.pi * (radius_pt**2))
			overlay_sc = ax.scatter(
				locs[:, 0],
				locs[:, 1],
				s=np.full((int(locs.shape[0]),), max(1e-6, size_pt2), dtype=float),
				facecolors="none",
				edgecolors="none",
				linewidths=0.0,
				alpha=0.0,
				zorder=0.0,
			)
			_draw_branch_morphology_overlay(
				ax=ax,
				fig=fig,
				sc=overlay_sc,
				locations_xy=locs,
				branch_morphology=branch_morphology,
				branch_cfg=branch_cfg,
			)
		except Exception:
			pass

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

	if bool(close_figure):
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
	branch_morphology: Any | None = None,
	branch_cfg: Any | None = None,
	fig: Any | None = None,
	ax: Any | None = None,
	close_figure: bool = True,
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
		branch_morphology=branch_morphology,
		branch_cfg=branch_cfg,
		fig=fig,
		ax=ax,
		close_figure=close_figure,
	)


def render_footprint_latency_map(
	*,
	template: Any,
	locations_xy: Any,
	config: FootprintMapConfig,
	png_path: Path,
	svg_path: Path,
	probe_geometry: ProbeGeometryConfig | None = None,
	branch_morphology: Any | None = None,
	branch_cfg: Any | None = None,
	fig: Any | None = None,
	ax: Any | None = None,
	close_figure: bool = True,
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
		branch_morphology=branch_morphology,
		branch_cfg=branch_cfg,
		fig=fig,
		ax=ax,
		close_figure=close_figure,
	)


def _maybe_reversed_colormap(cmap_name: str, *, reverse: bool) -> str:
	name = str(cmap_name or "viridis").strip() or "viridis"
	if not reverse or name.endswith("_r"):
		return name
	return f"{name}_r"
