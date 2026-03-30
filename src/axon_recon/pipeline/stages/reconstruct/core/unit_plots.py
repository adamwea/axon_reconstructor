from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import replace

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconConfig


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
	if template.shape[0] != locs.shape[0] and template.shape[1] == locs.shape[0]:
		template = template.T
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


def _normalize_template_channels_by_time(template_ch_by_t: Any, n_channels: int) -> Any:
	import numpy as np  # type: ignore[import-not-found]

	tpl = np.asarray(template_ch_by_t, dtype=float)
	if tpl.ndim != 2:
		raise ValueError(f"Expected template_ch_by_t to be 2D, got shape={tpl.shape}")
	if int(tpl.shape[0]) == int(n_channels):
		return tpl
	if int(tpl.shape[1]) == int(n_channels):
		return tpl.T
	raise ValueError(f"Template channels do not match locations rows: {tpl.shape} vs n_channels={n_channels}")


def _raw_branch_payload_from_gtr(gtr: Any) -> list[dict[str, Any]]:
	from axon_recon.pipeline.stages.reconstruct.io import as_list

	raw_paths = as_list(getattr(gtr, "_paths_raw", None))
	out: list[dict[str, Any]] = []
	for raw_idx, raw_path in enumerate(raw_paths):
		try:
			channels = [int(x) for x in list(raw_path)[::-1]]
		except Exception:
			continue
		if len(channels) < 2:
			continue
		out.append({"branch_index": int(raw_idx), "channels": channels, "label": int(raw_idx)})
	return out


def _preferred_branch_ids(branch_like: Any) -> list[int]:
	from axon_recon.pipeline.stages.reconstruct.io import as_int_list

	if not isinstance(branch_like, dict):
		return []
	for key in ("electrode_ids", "channels", "node_indices", "nodes"):
		vals = as_int_list(branch_like.get(key, []))
		if vals:
			return vals
	return []


def _clean_branch_payload_from_gtr(gtr: Any) -> list[dict[str, Any]]:
	from axon_recon.pipeline.stages.reconstruct.io import as_list

	out: list[dict[str, Any]] = []
	for bi, branch in enumerate(as_list(getattr(gtr, "branches", None))):
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
	return out


def write_unit_circle_recon_plot(
	*,
	output_png: Path,
	output_svg: Path,
	template_ch_by_t: Any,
	locs_xy: Any,
	gtr: Any,
	circle_config: CircleReconConfig,
	unit_id: Any,
) -> dict[str, str]:
	import numpy as np  # type: ignore[import-not-found]

	from axon_recon.pipeline.stages.templates.core.render import render_template_circles_plot
	from axon_recon.pipeline.stages.templates.models.inputs import TemplateCirclesPlotConfig

	# Step 1: keep reconstruct circle_recon as a thin wrapper around templates-stage circles rendering.

	locs = np.asarray(locs_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Expected locs_xy to be [N,2+], got shape={locs.shape}")
	locs = locs[:, :2]
	tpl = template_ch_by_t

	output_cfg = circle_config.output

	base_cfg = getattr(circle_config, "base_template_circles", None)
	if not isinstance(base_cfg, TemplateCirclesPlotConfig):
		base_cfg = TemplateCirclesPlotConfig()

	cfg = replace(
		base_cfg,
		write_png=bool(output_cfg.write_png),
		write_svg=bool(output_cfg.write_svg),
		dpi=float(max(72.0, float(output_cfg.dpi))),
		relpath=str(output_cfg.relpath),
	)

	return render_template_circles_plot(
		template=tpl,
		locations_xy=locs,
		config=cfg,
		png_path=Path(output_png),
		svg_path=Path(output_svg),
		unit_id=unit_id,
		gtr=gtr,
	)


__all__ = ["write_unit_amplitude_map_png", "write_unit_circle_recon_plot"]
