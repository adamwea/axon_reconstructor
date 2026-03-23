from __future__ import annotations

from pathlib import Path
from typing import Any

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig


def write_unit_amplitude_map_png(
	*,
	output_png: Path,
	template_ch_by_t: Any,
	locs_xy: Any,
	panel_background_color: str = "black",
	high_color: str = "red",
	mid_color: str = "white",
	low_color: str = "blue",
	show_ticks: Any = (1, 10, "dynamic_high"),
	show_colorbar: bool = True,
	heatmap_config: SharedHeatmapConfig | None = None,
) -> None:
	import matplotlib
	import numpy as np  # type: ignore[import-not-found]

	from ..plotting.helpers import reconstruct_colorbar_bounds
	from ..plotting.helpers import reconstruct_colorbar_ticks
	from ..plotting.helpers import reconstruct_draw_footprint_squares
	from ..plotting.helpers import reconstruct_tick_spec

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	template = np.asarray(template_ch_by_t, dtype=float)
	locs = np.asarray(locs_xy, dtype=float)
	if template.ndim != 2:
		raise ValueError(f"Expected template_ch_by_t to be 2D, got shape={template.shape}")
	if locs.ndim != 2 or locs.shape[1] < 2:
		raise ValueError(f"Expected locs_xy to be [N,2+], got shape={locs.shape}")
	if template.shape[0] != locs.shape[0]:
		raise ValueError(
			"template_ch_by_t channel count does not match locs_xy rows: "
			f"{template.shape[0]} vs {locs.shape[0]}"
		)

	amp_values = np.max(np.abs(template), axis=1)
	resolved_cfg = heatmap_config if heatmap_config is not None else SharedHeatmapConfig(
		background=panel_background_color,
		high_color=high_color,
		mid_color=mid_color,
		low_color=low_color,
		show_ticks=tuple(show_ticks) if isinstance(show_ticks, (list, tuple)) else (show_ticks,),
		show_colorbar=bool(show_colorbar),
	)

	fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200)
	fig.patch.set_facecolor(resolved_cfg.background)
	ax.set_facecolor(resolved_cfg.background)
	ax.set_aspect("equal")
	ax.axis("off")

	bundle = reconstruct_draw_footprint_squares(
		ax=ax,
		locs_xy=locs,
		amp=amp_values,
		high_color=resolved_cfg.high_color,
		mid_color=resolved_cfg.mid_color,
		low_color=resolved_cfg.low_color,
		force_low_value=resolved_cfg.force_low_value,
		force_high_value=resolved_cfg.force_high_value,
		scale_mode=resolved_cfg.scale,
		percentile_low=resolved_cfg.percentile_low,
		percentile_high_linear=resolved_cfg.percentile_high_linear,
		percentile_high_log=resolved_cfg.percentile_high_log,
		knot_anchor_values=resolved_cfg.knot_anchor_values,
		knot_y1_min=resolved_cfg.knot_y1_min,
		knot_y1_max=resolved_cfg.knot_y1_max,
		knot_y2_min=resolved_cfg.knot_y2_min,
		knot_y2_max=resolved_cfg.knot_y2_max,
		knot_min_gap=resolved_cfg.knot_min_gap,
		linear_cap_rounding_mode=resolved_cfg.linear_cap_rounding_mode,
		linear_cap_rounding_step=resolved_cfg.linear_cap_rounding_step,
		linear_cap_min_vmax=resolved_cfg.linear_cap_min_vmax,
	)

	if bool(resolved_cfg.show_colorbar):
		from matplotlib.cm import ScalarMappable  # type: ignore[import-not-found]

		bounds = reconstruct_colorbar_bounds(
			location=resolved_cfg.colorbar_location,
			length_fraction=resolved_cfg.colorbar_length_fraction,
			pad_fraction=resolved_cfg.colorbar_pad_fraction,
		)
		cax = fig.add_axes(bounds)
		sm = ScalarMappable(norm=bundle["norm"], cmap=bundle["cmap"])
		sm.set_array([])
		cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
		tick_spec = reconstruct_tick_spec(resolved_cfg.show_ticks)
		ticks, labels = reconstruct_colorbar_ticks(
			tick_spec=tick_spec,
			vmin=float(bundle["vmin"]),
			vmax=float(bundle["vmax"]),
		)
		if ticks:
			cbar.set_ticks(ticks)
			cbar.set_ticklabels(labels)
		cbar.ax.tick_params(labelsize=resolved_cfg.colorbar_fontsize, colors=resolved_cfg.colorbar_fontcolor)

	output_png = Path(output_png)
	output_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
	plt.close(fig)


__all__ = ["write_unit_amplitude_map_png"]
