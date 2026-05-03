from __future__ import annotations

from pathlib import Path
from typing import Any


def _use_agg_backend() -> Any:
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	return plt


def _coerce_template_and_locations(*, template_ch_by_t: Any, locs_xy: Any, gtr: Any) -> tuple[Any, Any, float]:
	import numpy as np  # type: ignore[import-not-found]

	template_source = getattr(gtr, "template", None)
	if template_source is None:
		template_source = template_ch_by_t
	locations_source = getattr(gtr, "locations", None)
	if locations_source is None:
		locations_source = locs_xy
	template = np.asarray(template_source, dtype=float)
	locations = np.asarray(locations_source, dtype=float)
	if locations.ndim != 2 or int(locations.shape[1]) < 2:
		raise ValueError(f"Expected plotting locations to be [N,2+], got shape={locations.shape}")
	locations = np.asarray(locations[:, :2], dtype=float)
	if template.ndim != 2:
		raise ValueError(f"Expected plotting template to be 2D, got shape={template.shape}")
	if int(template.shape[0]) != int(locations.shape[0]) and int(template.shape[1]) == int(locations.shape[0]):
		template = template.T
	if int(template.shape[0]) != int(locations.shape[0]):
		raise ValueError(
			"Diagnostic plotting template channels do not match locations rows: "
			f"{template.shape[0]} vs {locations.shape[0]}"
		)
	try:
		fs_hz = float(getattr(gtr, "fs", 1.0) or 1.0)
	except Exception:
		fs_hz = 1.0
	return template, locations, fs_hz


def _save_figure(*, fig: Any, output_png: Path | None, output_svg: Path | None, dpi: float) -> None:
	if output_png is not None:
		output_png.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
	if output_svg is not None:
		output_svg.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_svg, dpi=dpi, bbox_inches="tight")


def _normalize_colorbar_limits(values: Any) -> tuple[float, float]:
	import numpy as np  # type: ignore[import-not-found]

	array = np.asarray(values, dtype=float)
	if array.size <= 0:
		return 0.0, 1.0
	vmin = float(np.min(array))
	vmax = float(np.max(array))
	if vmax <= vmin:
		return vmin, vmin + 1.0
	return vmin, vmax


def _plot_channel_selection_panels(*, fig: Any, gtr: Any, locations: Any, invert_y_axis: bool) -> None:
	import numpy as np  # type: ignore[import-not-found]

	filters: tuple[tuple[str, Any, Any, bool], ...] = (
		(
			"Detection threshold",
			getattr(gtr, "_detect_threshold", None),
			getattr(gtr, "_selected_channels_detect", ()),
			False,
		),
		(
			"Kurtosis threshold",
			getattr(gtr, "_kurt_threshold", None),
			getattr(gtr, "_selected_channels_kurt", ()),
			False,
		),
		(
			"Peak std threshold",
			getattr(gtr, "_peak_std_threhsold", getattr(gtr, "_peak_std_threshold", None)),
			getattr(gtr, "_selected_channels_peakstd", ()),
			False,
		),
		(
			"Init delay threshold",
			getattr(gtr, "_init_delay", None),
			getattr(gtr, "_selected_channels_init", ()),
			True,
		),
	)
	panel_specs: list[tuple[str, Any, bool]] = []
	for title, threshold, selected_channels, show_init in filters:
		if threshold is None:
			continue
		panel_specs.append((f"{title}: {threshold}", np.asarray(list(selected_channels), dtype=int), show_init))
	panel_specs.append(("All thresholds", np.asarray(getattr(gtr, "selected_channels", ()), dtype=int), True))

	selection_spec = fig.add_gridspec(
		1,
		max(1, len(panel_specs)),
		left=0.38,
		right=0.99,
		bottom=0.08,
		top=0.88,
		wspace=0.08,
	)
	for index, (title, selected_channels, show_init) in enumerate(panel_specs):
		ax = fig.add_subplot(selection_spec[0, index])
		ax.set_title(str(title), fontsize=11)
		ax.plot(locations[:, 0], locations[:, 1], marker=".", color="grey", ls="", alpha=0.2)
		if selected_channels.size > 0:
			ax.plot(
				locations[selected_channels, 0],
				locations[selected_channels, 1],
				marker=".",
				color="k",
				ls="",
				alpha=0.5,
			)
		if show_init:
			init_channel = getattr(gtr, "init_channel", None)
			if init_channel is not None:
				ax.plot(*locations[int(init_channel)], marker=".", color="r", ls="", alpha=0.5)
		ax.axis("off")
		ax.set_aspect("equal", adjustable="box")
		if bool(invert_y_axis):
			ax.invert_yaxis()


def _plot_graph_panels(*, fig: Any, gtr: Any, invert_y_axis: bool) -> None:
	import matplotlib as mpl  # type: ignore[import-not-found]

	graph_spec = fig.add_gridspec(
		1,
		4,
		left=0.05,
		right=0.95,
		bottom=0.56,
		top=0.92,
		width_ratios=(7.0, 0.6, 7.0, 0.6),
		wspace=0.18,
	)
	ax_nodes = fig.add_subplot(graph_spec[0, 0])
	ax_nodes_cb = fig.add_subplot(graph_spec[0, 1])
	ax_edges = fig.add_subplot(graph_spec[0, 2])
	ax_edges_cb = fig.add_subplot(graph_spec[0, 3])

	gtr._plot_nodes(cmap_nodes="viridis", node_searched_labels=False, ax=ax_nodes)
	node_vmin, node_vmax = _normalize_colorbar_limits(getattr(gtr, "_node_heuristic", (1.0,)))
	node_norm = mpl.colors.Normalize(vmin=node_vmin, vmax=node_vmax)
	mpl.colorbar.ColorbarBase(
		ax_nodes_cb,
		cmap=mpl.colormaps["viridis"],
		norm=node_norm,
		orientation="vertical",
	).set_label("heuristic init (a.u.)")

	gtr._plot_edges(cmap_edges="YlGn", ax=ax_edges)
	edge_vmin, edge_vmax = _normalize_colorbar_limits([data.get("heur", 0.0) for _, _, data in gtr.graph.edges.data()])
	edge_norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
	mpl.colorbar.ColorbarBase(
		ax_edges_cb,
		cmap=mpl.colormaps["YlGn"],
		norm=edge_norm,
		orientation="vertical",
	).set_label("heuristic (a.u.)")

	ax_nodes.set_title("Nodes\n(by node heuristic)")
	ax_edges.set_title("Edges\n(by edge heuristic)")
	if bool(invert_y_axis):
		ax_nodes.invert_yaxis()
		ax_edges.invert_yaxis()


def _plot_branch_velocity_panel(*, ax: Any, gtr: Any) -> None:
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	import numpy as np  # type: ignore[import-not-found]
	from axon_velocity.plotting import plot_velocity  # type: ignore[import-not-found]

	paths_raw = list(getattr(gtr, "_paths_raw", ()))
	cm = plt.get_cmap("tab20")
	branch_colors = [cm(index / max(1, len(paths_raw))) for index, _ in enumerate(paths_raw)]
	for branch in getattr(gtr, "branches", ()):
		raw_idx = int(branch.get("raw_path_idx", 0))
		if raw_idx >= len(paths_raw):
			continue
		color = branch_colors[raw_idx] if branch_colors else cm(0.0)
		path = np.asarray(paths_raw[raw_idx])[::-1][1:]
		peaks, dists = gtr._estimate_peaks_and_dists(path)
		_path_clean, velocity, offset, r2, _p_value, dists_clean, peaks_clean, inlier_mask = gtr.robust_velocity_estimator(
			path,
			peaks,
			dists,
			True,
		)
		if velocity is None or offset is None or inlier_mask is None or dists_clean is None or peaks_clean is None:
			continue
		plot_velocity(
			peaks_clean,
			dists_clean,
			velocity,
			offset,
			color=color,
			r2=r2,
			ax=ax,
			markeredgecolor="k",
			alpha_markers=0.3,
			lw=2,
			markersize=12,
			fs=18,
			plot_markers=True,
		)
		outlier_idxs = np.where(inlier_mask == False)
		if outlier_idxs[0].size > 0:
			ax.plot(
				peaks[outlier_idxs],
				dists[outlier_idxs],
				marker="d",
				ls="",
				color=color,
				markersize=18,
				markeredgecolor="k",
				zorder=10,
				alpha=0.7,
			)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)


def write_unit_channel_selection_diagnostic_figure(
	*,
	av: Any,
	output_png: Path | None,
	output_svg: Path | None,
	template_ch_by_t: Any,
	locs_xy: Any,
	gtr: Any,
	dpi: float = 300.0,
	invert_y_axis: bool = True,
) -> None:
	if output_png is None and output_svg is None:
		return
	plt = _use_agg_backend()
	template, locations, fs_hz = _coerce_template_and_locations(template_ch_by_t=template_ch_by_t, locs_xy=locs_xy, gtr=gtr)
	fig = plt.figure(figsize=(18.0, 8.0))
	try:
		left_spec = fig.add_gridspec(2, 1, left=0.04, right=0.34, bottom=0.08, top=0.88, hspace=0.18)
		ax_amp = fig.add_subplot(left_spec[0, 0])
		av.plot_amplitude_map(
			template,
			locations,
			log=True,
			ax=ax_amp,
			cmap="PRGn",
			colorbar=True,
			colorbar_orientation="horizontal",
		)
		ax_amp.set_title("Amplitude", fontsize=16)
		if bool(invert_y_axis):
			ax_amp.invert_yaxis()

		ax_latency = fig.add_subplot(left_spec[1, 0])
		av.plot_peak_latency_map(
			template,
			locations,
			fs=fs_hz,
			log=False,
			ax=ax_latency,
			colorbar=True,
			colorbar_orientation="horizontal",
		)
		ax_latency.set_title("Peak latency", fontsize=16)
		if bool(invert_y_axis):
			ax_latency.invert_yaxis()

		_plot_channel_selection_panels(fig=fig, gtr=gtr, locations=locations, invert_y_axis=bool(invert_y_axis))
		fig.suptitle("Channel Selection Procedure", fontsize=18, y=0.97)
		_save_figure(fig=fig, output_png=output_png, output_svg=output_svg, dpi=float(max(72.0, dpi)))
	finally:
		plt.close(fig)


def write_unit_axon_reconstruction_diagnostic_figure(
	*,
	output_png: Path | None,
	output_svg: Path | None,
	gtr: Any,
	dpi: float = 300.0,
	invert_y_axis: bool = True,
) -> None:
	if output_png is None and output_svg is None:
		return
	plt = _use_agg_backend()
	fig = plt.figure(figsize=(15.0, 12.0))
	try:
		_plot_graph_panels(fig=fig, gtr=gtr, invert_y_axis=bool(invert_y_axis))
		lower_spec = fig.add_gridspec(1, 2, left=0.05, right=0.95, bottom=0.07, top=0.47, wspace=0.14)
		ax_raw = fig.add_subplot(lower_spec[0, 0])
		gtr.plot_raw_branches(
			cmap="tab20",
			plot_bp=True,
			plot_neighbors=True,
			plot_full_template=True,
			ax=ax_raw,
		)
		ax_raw.set_title("Raw branches", fontsize=16)
		if bool(invert_y_axis):
			ax_raw.invert_yaxis()
		handles, labels = ax_raw.get_legend_handles_labels()
		if handles and labels:
			ax_raw.legend(fontsize=9, loc="best")

		ax_velocity = fig.add_subplot(lower_spec[0, 1])
		_plot_branch_velocity_panel(ax=ax_velocity, gtr=gtr)
		ax_velocity.set_title("Branch velocities", fontsize=16)
		fig.suptitle("Axonal Reconstruction Method", fontsize=18, y=0.97)
		_save_figure(fig=fig, output_png=output_png, output_svg=output_svg, dpi=float(max(72.0, dpi)))
	finally:
		plt.close(fig)


__all__ = [
	"write_unit_channel_selection_diagnostic_figure",
	"write_unit_axon_reconstruction_diagnostic_figure",
]
