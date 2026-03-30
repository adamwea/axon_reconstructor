from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from dataclasses import replace

from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig
from axon_recon.pipeline.stages.reconstruct.models.inputs import CircleReconConfig


LOGGER = logging.getLogger("axon_recon.reconstruct")


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
				"color": branch.get("color", None),
			}
		)
	return out


def _clean_path_payload_from_gtr(gtr: Any) -> list[dict[str, Any]]:
	from axon_recon.pipeline.stages.reconstruct.io import as_list

	out: list[dict[str, Any]] = []
	for bi, path in enumerate(as_list(getattr(gtr, "_paths_clean", None))):
		try:
			channels = [int(x) for x in list(path)]
		except Exception:
			continue
		if len(channels) < 2:
			continue
		out.append(
			{
				"branch_index": int(bi),
				"channels": channels,
				"label": int(bi),
				"color": None,
			}
		)
	return out


def _branch_item_class_name(branches: Any) -> str:
	from axon_recon.pipeline.stages.reconstruct.io import as_list

	items = as_list(branches)
	if len(items) == 0:
		return "None"
	return str(type(items[0]).__name__)


def _select_branch_payload_from_gtr(
	*,
	gtr: Any,
	branch_scope: str,
) -> tuple[list[dict[str, Any]], str, str, str, int, int, int]:
	from axon_recon.pipeline.stages.reconstruct.io import as_list

	raw_paths = as_list(getattr(gtr, "_paths_raw", None))
	clean_branch_records = as_list(getattr(gtr, "branches", None))
	clean_paths = as_list(getattr(gtr, "_paths_clean", None))

	scope = str(branch_scope or "raw").strip().lower()
	if scope == "clean":
		payload = _clean_branch_payload_from_gtr(gtr)
		source_name = "gtr.branches"
		source_collection = getattr(gtr, "branches", None)
		if len(payload) == 0 and len(clean_paths) > 0:
			payload = _clean_path_payload_from_gtr(gtr)
			source_name = "gtr._paths_clean"
			source_collection = getattr(gtr, "_paths_clean", None)
	else:
		payload = _raw_branch_payload_from_gtr(gtr)
		source_name = "gtr._paths_raw"
		source_collection = getattr(gtr, "_paths_raw", None)

	return (
		payload,
		source_name,
		str(type(source_collection).__name__),
		_branch_item_class_name(source_collection),
		len(raw_paths),
		len(clean_branch_records),
		len(clean_paths),
	)


def _gtr_node_indices(gtr: Any) -> set[int]:
	graph = getattr(gtr, "graph", None)
	if graph is None:
		return set()
	nodes_fn = getattr(graph, "nodes", None)
	if not callable(nodes_fn):
		return set()
	try:
		return {int(node) for node in list(nodes_fn())}
	except Exception:
		return set()


def _branch_channel_set(branch_payload: list[dict[str, Any]]) -> set[int]:
	out: set[int] = set()
	for branch in branch_payload:
		if not isinstance(branch, dict):
			continue
		for ch in list(branch.get("channels", [])):
			try:
				out.add(int(ch))
			except Exception:
				continue
	return out


def _remap_branch_payload(
	branch_payload: list[dict[str, Any]],
	old_to_new: dict[int, int],
) -> list[dict[str, Any]]:
	out: list[dict[str, Any]] = []
	for i, branch in enumerate(branch_payload):
		if not isinstance(branch, dict):
			continue
		mapped_channels: list[int] = []
		for ch in list(branch.get("channels", [])):
			try:
				mapped = old_to_new.get(int(ch), None)
			except Exception:
				mapped = None
			if mapped is not None:
				mapped_channels.append(int(mapped))
		if len(mapped_channels) < 2:
			continue
		branch_index = branch.get("branch_index", i)
		out.append(
			{
				"branch_index": int(branch_index),
				"channels": mapped_channels,
				"label": branch.get("label", branch_index),
				"color": branch.get("color", None),
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

	from axon_recon.pipeline.stages.templates.core.render import render_footprint_amplitude_map
	from axon_recon.pipeline.stages.templates.core.render import render_footprint_latency_map
	from axon_recon.pipeline.stages.templates.core.render import render_template_circles_plot
	from axon_recon.pipeline.stages.templates.models.inputs import FootprintMapConfig
	from axon_recon.pipeline.stages.templates.models.inputs import TemplateCirclesBranchMorphologyConfig
	from axon_recon.pipeline.stages.templates.models.inputs import TemplateCirclesPlotConfig

	# Step 1: keep reconstruct circle_recon as a thin wrapper around templates-stage circles rendering.

	locs = np.asarray(locs_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Expected locs_xy to be [N,2+], got shape={locs.shape}")
	locs = locs[:, :2]
	tpl = _normalize_template_channels_by_time(template_ch_by_t, n_channels=int(locs.shape[0]))

	output_cfg = circle_config.output
	display_cfg = getattr(circle_config, "display", None)
	base_mode = str(getattr(display_cfg, "base", "template_circles") or "template_circles").strip().lower()
	if base_mode not in {"template_circles", "amplitude_map", "latency_map"}:
		base_mode = "template_circles"

	base_cfg = getattr(circle_config, "base_template_circles", None)
	if not isinstance(base_cfg, TemplateCirclesPlotConfig):
		base_cfg = TemplateCirclesPlotConfig()

	branch_scope = str(getattr(display_cfg, "branch_scope", "raw") or "raw").strip().lower()
	if branch_scope not in {"raw", "clean"}:
		branch_scope = "raw"
	(
		branch_payload,
		branch_source_name,
		branch_collection_class,
		selected_branch_class,
		raw_branch_count,
		clean_branch_count,
		clean_path_count,
	) = _select_branch_payload_from_gtr(gtr=gtr, branch_scope=branch_scope)
	LOGGER.info(
		"Unit %s circle_recon branch_scope=%s source=%s collection_class=%s selected_branch_class=%s selected_count=%d raw_count=%d clean_branch_count=%d clean_path_count=%d",
		unit_id,
		branch_scope,
		branch_source_name,
		branch_collection_class,
		selected_branch_class,
		len(branch_payload),
		raw_branch_count,
		clean_branch_count,
		clean_path_count,
	)

	n_channels = int(locs.shape[0])
	node_channels = {int(ch) for ch in _gtr_node_indices(gtr) if 0 <= int(ch) < n_channels}
	branch_channels = {int(ch) for ch in _branch_channel_set(branch_payload) if 0 <= int(ch) < n_channels}

	channel_scope = str(getattr(display_cfg, "channel_scope", "nodes_and_branches") or "nodes_and_branches").strip().lower()
	if channel_scope not in {"nodes_and_branches", "branches_only", "nodes_only"}:
		channel_scope = "nodes_and_branches"

	if channel_scope == "branches_only":
		selected_channels = sorted(branch_channels)
	elif channel_scope == "nodes_only":
		selected_channels = sorted(node_channels if len(node_channels) > 0 else branch_channels)
	else:
		selected_union = node_channels | branch_channels
		selected_channels = sorted(selected_union if len(selected_union) > 0 else set(range(n_channels)))

	if len(selected_channels) == 0:
		selected_channels = list(range(n_channels))

	if len(selected_channels) < n_channels:
		selected_idx = np.asarray(selected_channels, dtype=int)
		tpl = np.asarray(tpl, dtype=float)[selected_idx, :]
		locs = np.asarray(locs, dtype=float)[selected_idx, :]
		old_to_new = {int(old): int(new) for new, old in enumerate(selected_channels)}
		branch_payload = _remap_branch_payload(branch_payload, old_to_new)

	base_branch_cfg = getattr(base_cfg, "branch_morphology", None)
	if not isinstance(base_branch_cfg, TemplateCirclesBranchMorphologyConfig):
		base_branch_cfg = TemplateCirclesBranchMorphologyConfig()

	branch_cfg = replace(
		base_branch_cfg,
		enabled=True,
		node_outline_color=(
			str(getattr(display_cfg, "node_outline_color", getattr(base_branch_cfg, "node_outline_color", None)) or "").strip()
			or None
		),
		node_outline_linewidth=float(
			max(
				0.0,
				float(getattr(display_cfg, "node_outline_linewidth", getattr(base_branch_cfg, "node_outline_linewidth", 0.0))),
			)
		),
		branch_outline_color=(
			str(getattr(display_cfg, "branch_outline_color", getattr(base_branch_cfg, "branch_outline_color", None)) or "").strip()
			or None
		),
		branch_outline_linewidth=float(
			max(
				0.0,
				float(getattr(display_cfg, "branch_outline_linewidth", getattr(base_branch_cfg, "branch_outline_linewidth", 0.0))),
			)
		),
		node_border_linewidth=float(
			max(
				0.0,
				float(getattr(display_cfg, "node_border_linewidth", base_branch_cfg.node_border_linewidth)),
			)
		),
		edge_linewidth=float(
			max(
				0.0,
				float(getattr(display_cfg, "edge_linewidth", base_branch_cfg.edge_linewidth)),
			)
		),
		show_branch_labels=bool(getattr(display_cfg, "show_branch_labels", base_branch_cfg.show_branch_labels)),
		unique_color_per_branch=bool(getattr(display_cfg, "unique_color_per_branch", base_branch_cfg.unique_color_per_branch)),
		color_scheme=str(getattr(display_cfg, "color_scheme", base_branch_cfg.color_scheme) or base_branch_cfg.color_scheme),
	)

	cfg = replace(
		base_cfg,
		write_png=bool(output_cfg.write_png),
		write_svg=bool(output_cfg.write_svg),
		dpi=float(max(72.0, float(output_cfg.dpi))),
		relpath=str(output_cfg.relpath),
		force_center_soma=bool(getattr(display_cfg, "force_center_soma", base_cfg.force_center_soma)),
		branch_morphology=branch_cfg,
	)

	if base_mode in {"amplitude_map", "latency_map"}:
		footprint_base_cfg = (
			getattr(circle_config, "base_footprint_amplitude", None)
			if base_mode == "amplitude_map"
			else getattr(circle_config, "base_footprint_latency", None)
		)
		if not isinstance(footprint_base_cfg, FootprintMapConfig):
			footprint_base_cfg = FootprintMapConfig()

		foot_cfg = replace(
			footprint_base_cfg,
			write_png=bool(output_cfg.write_png),
			write_svg=bool(output_cfg.write_svg),
			relpath=str(output_cfg.relpath),
			background=str(getattr(base_cfg, "background", footprint_base_cfg.background) or footprint_base_cfg.background),
			template_shape=str(
				getattr(base_cfg, "template_shape", footprint_base_cfg.template_shape) or footprint_base_cfg.template_shape
			),
			template_padding_value=str(
				getattr(base_cfg, "template_padding_value", footprint_base_cfg.template_padding_value)
				or footprint_base_cfg.template_padding_value
			),
		)

		if base_mode == "amplitude_map":
			return render_footprint_amplitude_map(
				template=tpl,
				locations_xy=locs,
				config=foot_cfg,
				png_path=Path(output_png),
				svg_path=Path(output_svg),
				branch_morphology={"branches": branch_payload},
				branch_cfg=branch_cfg,
			)

		return render_footprint_latency_map(
			template=tpl,
			locations_xy=locs,
			config=foot_cfg,
			png_path=Path(output_png),
			svg_path=Path(output_svg),
			branch_morphology={"branches": branch_payload},
			branch_cfg=branch_cfg,
		)

	return render_template_circles_plot(
		template=tpl,
		locations_xy=locs,
		config=cfg,
		png_path=Path(output_png),
		svg_path=Path(output_svg),
		unit_id=unit_id,
		branch_morphology={"branches": branch_payload},
		gtr=gtr,
		plot_scope_points_xy=locs,
		zoom_padding_percent=float(max(0.0, float(getattr(display_cfg, "zoom_padding_percent", 20.0)))),
		allow_scope_expansion=False,
	)


__all__ = ["write_unit_amplitude_map_png", "write_unit_circle_recon_plot"]
