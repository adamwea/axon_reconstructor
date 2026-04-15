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


def _figure_to_rgba_image(*, fig: Any) -> Any:
	import numpy as np  # type: ignore[import-not-found]

	fig.canvas.draw()
	width, height = fig.canvas.get_width_height()
	buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
	return buffer.reshape(height, width, 4).copy()


def _save_figure(*, fig: Any, output_png: Path | None, output_svg: Path | None, dpi: float) -> None:
	if output_png is not None:
		output_png.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
	if output_svg is not None:
		output_svg.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_svg, dpi=dpi, bbox_inches="tight")


def _clear_suptitle(fig: Any) -> None:
	suptitle = getattr(fig, "_suptitle", None)
	if suptitle is not None:
		suptitle.set_text("")


def write_unit_channel_selection_diagnostic_figure(
	*,
	av: Any,
	output_png: Path | None,
	output_svg: Path | None,
	template_ch_by_t: Any,
	locs_xy: Any,
	gtr: Any,
	dpi: float = 300.0,
) -> None:
	if output_png is None and output_svg is None:
		return
	plt = _use_agg_backend()
	template, locations, fs_hz = _coerce_template_and_locations(template_ch_by_t=template_ch_by_t, locs_xy=locs_xy, gtr=gtr)

	fig_amp = plt.figure(figsize=(7.5, 4.6))
	ax_amp = fig_amp.add_subplot(111)
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

	fig_latency = plt.figure(figsize=(7.5, 4.6))
	ax_latency = fig_latency.add_subplot(111)
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

	fig_selection = gtr.plot_channel_selection()
	fig_selection.set_size_inches(16.0, 4.8, forward=True)
	_clear_suptitle(fig_selection)

	amp_img = _figure_to_rgba_image(fig=fig_amp)
	latency_img = _figure_to_rgba_image(fig=fig_latency)
	selection_img = _figure_to_rgba_image(fig=fig_selection)

	plt.close(fig_amp)
	plt.close(fig_latency)
	plt.close(fig_selection)

	fig = plt.figure(figsize=(18.0, 8.0))
	gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 2.6), height_ratios=(1.0, 1.0), wspace=0.03, hspace=0.06)
	ax_amp_img = fig.add_subplot(gs[0, 0])
	ax_latency_img = fig.add_subplot(gs[1, 0])
	ax_selection_img = fig.add_subplot(gs[:, 1])
	for ax, image in ((ax_amp_img, amp_img), (ax_latency_img, latency_img), (ax_selection_img, selection_img)):
		ax.imshow(image)
		ax.axis("off")
	fig.suptitle("Channel Selection Procedure", fontsize=18, y=0.99)
	fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)
	_save_figure(fig=fig, output_png=output_png, output_svg=output_svg, dpi=float(max(72.0, dpi)))
	plt.close(fig)


def write_unit_axon_reconstruction_diagnostic_figure(
	*,
	output_png: Path | None,
	output_svg: Path | None,
	gtr: Any,
	dpi: float = 300.0,
) -> None:
	if output_png is None and output_svg is None:
		return
	plt = _use_agg_backend()

	fig_graph = plt.figure(figsize=(10.0, 7.0))
	gtr.plot_graph(node_search_labels=False, fig=fig_graph, cmap_nodes="viridis", cmap_edges="YlGn")
	_clear_suptitle(fig_graph)

	fig_raw, ax_raw = plt.subplots(figsize=(7.0, 8.0))
	gtr.plot_raw_branches(
		cmap="tab20",
		plot_bp=True,
		plot_neighbors=True,
		plot_full_template=True,
		ax=ax_raw,
	)
	handles, labels = ax_raw.get_legend_handles_labels()
	if handles and labels:
		ax_raw.legend(fontsize=9, loc="best")

	fig_velocity = plt.figure(figsize=(8.0, 8.0))
	gtr.plot_velocities(
		fig=fig_velocity,
		cmap="tab20",
		plot_outliers=True,
		markersize=12,
		markersize_out=18,
		fs=18,
	)
	_clear_suptitle(fig_velocity)

	graph_img = _figure_to_rgba_image(fig=fig_graph)
	raw_img = _figure_to_rgba_image(fig=fig_raw)
	velocity_img = _figure_to_rgba_image(fig=fig_velocity)

	plt.close(fig_graph)
	plt.close(fig_raw)
	plt.close(fig_velocity)

	fig = plt.figure(figsize=(15.0, 12.0))
	gs = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0), width_ratios=(1.0, 1.0), hspace=0.06, wspace=0.04)
	ax_graph = fig.add_subplot(gs[0, :])
	ax_raw_img = fig.add_subplot(gs[1, 0])
	ax_velocity_img = fig.add_subplot(gs[1, 1])
	for ax, image in ((ax_graph, graph_img), (ax_raw_img, raw_img), (ax_velocity_img, velocity_img)):
		ax.imshow(image)
		ax.axis("off")
	fig.suptitle("Axonal Reconstruction Method", fontsize=18, y=0.99)
	fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)
	_save_figure(fig=fig, output_png=output_png, output_svg=output_svg, dpi=float(max(72.0, dpi)))
	plt.close(fig)


__all__ = [
	"write_unit_channel_selection_diagnostic_figure",
	"write_unit_axon_reconstruction_diagnostic_figure",
]