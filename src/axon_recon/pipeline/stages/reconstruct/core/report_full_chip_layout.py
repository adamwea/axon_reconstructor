from __future__ import annotations

import colorsys
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from typing import Any, Callable

from .branch_styles import ReconstructBranchRecord
from .branch_styles import select_reconstruct_branch_records
from ..models.inputs import ReconstructionBranchColorsConfig
from ..models.inputs import ReconstructionInputs
from ..models.results import UnitReconstructionResult


@dataclass(frozen=True)
class FullChipLayoutBranchRecord:
	branch_id: int
	branch_index: int
	label: Any
	points_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FullChipLayoutUnitRecord:
	unit_id: Any
	color: str
	branches: tuple[FullChipLayoutBranchRecord, ...]


def _unit_sort_key(unit_id: Any) -> tuple[int, str]:
	try:
		return (0, f"{int(unit_id):08d}")
	except Exception:
		return (1, str(unit_id))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
	return "#{0:02x}{1:02x}{2:02x}".format(
		int(max(0, min(255, round(float(rgb[0]) * 255.0)))),
		int(max(0, min(255, round(float(rgb[1]) * 255.0)))),
		int(max(0, min(255, round(float(rgb[2]) * 255.0)))),
	)


def _resolve_unit_palette(*, unit_count: int, color_config: Any) -> list[str]:
	if unit_count <= 0:
		return []
	strategy = str(getattr(color_config, "strategy", "distinct_hsv") or "distinct_hsv").strip().lower()
	color_scheme = str(getattr(color_config, "color_scheme", "nipy_spectral") or "nipy_spectral")
	if strategy == "colormap":
		try:
			from matplotlib import colormaps  # type: ignore[import-not-found]
			from matplotlib import colors as mcolors  # type: ignore[import-not-found]

			cmap = colormaps.get_cmap(color_scheme)
			if unit_count == 1:
				return [str(mcolors.to_hex(cmap(0.5), keep_alpha=False))]
			den = float(max(1, unit_count - 1))
			return [str(mcolors.to_hex(cmap(float(idx) / den), keep_alpha=False)) for idx in range(unit_count)]
		except Exception:
			strategy = "distinct_hsv"
	if strategy != "distinct_hsv":
		strategy = "distinct_hsv"
	colors: list[str] = []
	golden_ratio = 0.618033988749895
	for idx in range(unit_count):
		tier = int(idx // 24)
		hue = (0.17 + (float(idx) * golden_ratio)) % 1.0
		saturation = max(0.55, 0.86 - (0.10 * float(tier % 3)))
		value = max(0.62, 0.95 - (0.12 * float((tier // 3) % 2)))
		colors.append(_rgb_to_hex(colorsys.hsv_to_rgb(hue, saturation, value)))
	return colors


def _extract_branch_points(*, locs_xy: Any, branch_record: ReconstructBranchRecord) -> tuple[tuple[float, float], ...]:
	import numpy as np  # type: ignore[import-not-found]

	locs = np.asarray(locs_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		return ()
	selected_channels = [int(ch) for ch in branch_record.selected_channels if 0 <= int(ch) < int(locs.shape[0])]
	if len(selected_channels) < 2:
		return ()
	points = np.asarray(locs[selected_channels, :2], dtype=float)
	if points.ndim != 2 or int(points.shape[0]) < 2:
		return ()
	return tuple((float(row[0]), float(row[1])) for row in points)


def _probe_geometry_payload(probe_geometry: Any) -> dict[str, float | None]:
	return {
		"pitch_um": getattr(probe_geometry, "pitch_um", None),
		"electrode_size_um_x": getattr(probe_geometry, "electrode_size_um_x", None),
		"electrode_size_um_y": getattr(probe_geometry, "electrode_size_um_y", None),
		"active_area_um_x": getattr(probe_geometry, "active_area_um_x", None),
		"active_area_um_y": getattr(probe_geometry, "active_area_um_y", None),
		"sampling_rate_hz": getattr(probe_geometry, "sampling_rate_hz", None),
	}


def _resolve_plot_extents(*, unit_records: list[FullChipLayoutUnitRecord], probe_geometry: Any) -> tuple[float, float, float, float]:
	points = [point for unit_record in unit_records for branch in unit_record.branches for point in branch.points_xy]
	if not points:
		return (0.0, 1.0, 0.0, 1.0)
	xs = [float(point[0]) for point in points]
	ys = [float(point[1]) for point in points]
	active_area_um_x = getattr(probe_geometry, "active_area_um_x", None)
	active_area_um_y = getattr(probe_geometry, "active_area_um_y", None)
	try:
		x_max = float(active_area_um_x)
	except Exception:
		x_max = max(xs)
	try:
		y_max = float(active_area_um_y)
	except Exception:
		y_max = max(ys)
	if x_max <= 0.0:
		x_max = max(xs)
	if y_max <= 0.0:
		y_max = max(ys)
	x_min = 0.0 if getattr(probe_geometry, "active_area_um_x", None) is not None else min(xs)
	y_min = 0.0 if getattr(probe_geometry, "active_area_um_y", None) is not None else min(ys)
	if x_max <= x_min:
		x_max = x_min + 1.0
	if y_max <= y_min:
		y_max = y_min + 1.0
	return (float(x_min), float(x_max), float(y_min), float(y_max))


def _resolve_text_color(background_color: str) -> str:
	try:
		from matplotlib import colors as mcolors  # type: ignore[import-not-found]

		rgb = mcolors.to_rgb(background_color)
		luminance = (0.2126 * float(rgb[0])) + (0.7152 * float(rgb[1])) + (0.0722 * float(rgb[2]))
		return "black" if luminance >= 0.55 else "white"
	except Exception:
		return "black"


def write_full_chip_layout_plot(
	*,
	output_png: Path,
	output_svg: Path,
	unit_records: list[FullChipLayoutUnitRecord],
	probe_geometry: Any,
	display_config: Any,
	output_config: Any,
) -> dict[str, str]:
	import matplotlib
	import numpy as np  # type: ignore[import-not-found]

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	from matplotlib.lines import Line2D  # type: ignore[import-not-found]
	from matplotlib.patches import Rectangle  # type: ignore[import-not-found]

	figsize = tuple(getattr(display_config, "figsize", (11.0, 6.0)) or (11.0, 6.0))
	background_color = str(getattr(display_config, "background_color", "white") or "white")
	text_color = _resolve_text_color(background_color)
	fig, ax = plt.subplots(figsize=figsize, dpi=float(max(72.0, float(getattr(output_config, "dpi", 300.0) or 300.0))))
	fig.patch.set_facecolor(background_color)
	ax.set_facecolor(background_color)

	alpha = float(min(1.0, max(0.0, float(getattr(display_config, "alpha", 0.8) or 0.8))))
	linewidth = float(max(0.1, float(getattr(display_config, "linewidth", 1.25) or 1.25)))
	for unit_record in unit_records:
		for branch in unit_record.branches:
			points = np.asarray(branch.points_xy, dtype=float)
			if points.ndim != 2 or int(points.shape[0]) < 2:
				continue
			ax.plot(
				points[:, 0],
				points[:, 1],
				color=str(unit_record.color),
				alpha=alpha,
				linewidth=linewidth,
				solid_capstyle="round",
			)

	x_min, x_max, y_min, y_max = _resolve_plot_extents(unit_records=unit_records, probe_geometry=probe_geometry)
	ax.set_xlim(float(x_min), float(x_max))
	ax.set_ylim(float(y_min), float(y_max))
	if bool(getattr(display_config, "invert_y_axis", True)):
		ax.invert_yaxis()
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel("x (um)", color=text_color)
	ax.set_ylabel("y (um)", color=text_color)
	for spine in ax.spines.values():
		spine.set_color(text_color)
	ax.tick_params(colors=text_color)

	if bool(getattr(display_config, "draw_chip_outline", True)):
		chip_outline_color = str(getattr(display_config, "chip_outline_color", "#666666") or "#666666")
		chip_outline_linewidth = float(max(0.1, float(getattr(display_config, "chip_outline_linewidth", 1.0) or 1.0)))
		ax.add_patch(
			Rectangle(
				(x_min, y_min),
				float(x_max - x_min),
				float(y_max - y_min),
				fill=False,
				edgecolor=chip_outline_color,
				linewidth=chip_outline_linewidth,
			)
		)

	if bool(getattr(display_config, "show_title", True)):
		title = str(getattr(display_config, "title", "Full-chip reconstructed branch layout") or "Full-chip reconstructed branch layout")
		ax.set_title(title, color=text_color)

	if bool(getattr(display_config, "show_legend", False)):
		legend_handles = [
			Line2D([0], [0], color=str(unit_record.color), linewidth=linewidth, label=f"unit {unit_record.unit_id}")
			for unit_record in unit_records
		]
		if legend_handles:
			legend = ax.legend(
				handles=legend_handles,
				fontsize=float(max(1.0, float(getattr(display_config, "legend_fontsize", 6.0) or 6.0))),
				ncol=max(1, int(getattr(display_config, "legend_ncols", 1) or 1)),
				frameon=False,
			)
			for text in legend.get_texts():
				text.set_color(text_color)

	fig.tight_layout()
	outputs: dict[str, str] = {}
	if bool(getattr(output_config, "write_png", True)):
		output_png.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_png, dpi=float(max(72.0, float(getattr(output_config, "dpi", 300.0) or 300.0))), bbox_inches="tight")
		outputs["full_chip_layout_png"] = str(output_png)
	if bool(getattr(output_config, "write_svg", False)):
		output_svg.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_svg, bbox_inches="tight")
		outputs["full_chip_layout_svg"] = str(output_svg)
	plt.close(fig)
	return outputs


def run_report_full_chip_layout_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_results: list[UnitReconstructionResult],
	preserve_stage_reports: bool,
	existing_stage_outputs: dict[str, str],
	load_templates_for_unit_fn: Callable[..., Any],
	write_full_chip_layout_plot_fn: Callable[..., dict[str, str]],
	write_json_fn: Callable[[Path, Any], None],
	resolve_unit_output_paths_fn: Callable[..., dict[str, Path]],
	resolve_full_chip_layout_output_paths_fn: Callable[..., dict[str, Path]],
	logger: logging.Logger | None = None,
) -> dict[str, str]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.report_full_chip_layout")
	phase_cfg = inputs.phases.report_full_chip_layout
	stage_outputs: dict[str, str] = dict(existing_stage_outputs)
	output_paths = resolve_full_chip_layout_output_paths_fn(
		reconstruction_out_dir=reconstruction_out_dir,
		phase_output=phase_cfg.output,
	)
	existing_png = bool(output_paths["png_path"].exists())
	existing_svg = bool(output_paths["svg_path"].exists())
	existing_manifest = bool(output_paths["manifest_json"].exists())
	if preserve_stage_reports:
		active_logger.info(
			"reconstruct.report_full_chip_layout preserving existing outputs branch_scope=%s force_restart=%s force_replot=%s existing_png=%s existing_svg=%s existing_manifest=%s",
			str(phase_cfg.branch_scope),
			bool(inputs.force_restart),
			bool(inputs.force_replot),
			existing_png,
			existing_svg,
			existing_manifest,
		)
		if output_paths["manifest_json"].exists():
			stage_outputs["full_chip_layout_manifest_json"] = str(output_paths["manifest_json"])
		if bool(phase_cfg.output.write_png) and output_paths["png_path"].exists():
			stage_outputs["full_chip_layout_png"] = str(output_paths["png_path"])
		if bool(phase_cfg.output.write_svg) and output_paths["svg_path"].exists():
			stage_outputs["full_chip_layout_svg"] = str(output_paths["svg_path"])
		return stage_outputs

	active_logger.info(
		"reconstruct.report_full_chip_layout rewriting outputs branch_scope=%s force_restart=%s force_replot=%s existing_png=%s existing_svg=%s existing_manifest=%s units_total=%d",
		str(phase_cfg.branch_scope),
		bool(inputs.force_restart),
		bool(inputs.force_replot),
		existing_png,
		existing_svg,
		existing_manifest,
		len(unit_results),
	)

	successful_units = sorted(
		[item for item in unit_results if str(getattr(item, "status", "")).strip().lower() == "ok"],
		key=lambda item: _unit_sort_key(item.unit_id),
	)
	if not successful_units:
		raise FileNotFoundError(
			"No successful reconstructed units available for reconstruct.report_full_chip_layout"
		)

	unit_palette = _resolve_unit_palette(unit_count=len(successful_units), color_config=phase_cfg.unit_colors)
	unit_records: list[FullChipLayoutUnitRecord] = []
	manifest_units: list[dict[str, Any]] = []
	branches_total = 0
	units_plotted = 0
	units_error = 0
	units_skipped = 0
	selection_colors = ReconstructionBranchColorsConfig(unique_color_per_branch=False, color_scheme="tab20")

	for idx, unit_result in enumerate(successful_units):
		unit_id = unit_result.unit_id
		unit_color = unit_palette[idx] if idx < len(unit_palette) else "#1f77b4"
		unit_entry: dict[str, Any] = {
			"unit_id": unit_id,
			"status": "skipped",
			"error": None,
			"unit_color": unit_color,
			"branch_count": 0,
			"branches": [],
		}
		try:
			unit_paths = resolve_unit_output_paths_fn(
				reconstruction_out_dir=reconstruction_out_dir,
				unit_id=unit_id,
				per_unit_outputs=inputs.per_unit_outputs,
			)
			gtr_pkl = unit_paths["gtr_pkl"]
			unit_entry["unit_summary_json"] = str(unit_paths["unit_summary_json"])
			if not gtr_pkl.exists():
				raise FileNotFoundError(f"Missing gtr.pkl for unit {unit_id}")
			_, _, _gtr_template_ch_by_t, gtr_locs_xy, _fs, _source_name = load_templates_for_unit_fn(
				unit_id=unit_id,
				merged_units_dir=merged_units_dir,
				full_channels_templates_dir=full_channels_templates_dir,
				template_source=str(inputs.per_unit_outputs.template_source),
				use_full_channels_templates=inputs.use_full_channels_templates,
				require_full_channels_templates=inputs.require_full_channels_templates,
				probe_geometry=inputs.probe_geometry,
			)
			with open(gtr_pkl, "rb") as handle:
				gtr = pickle.load(handle)
			branch_selection = select_reconstruct_branch_records(
				gtr=gtr,
				branch_scope=str(phase_cfg.branch_scope),
				branch_colors=selection_colors,
			)
			unit_entry["source_name"] = str(branch_selection.source_name)
			branches: list[FullChipLayoutBranchRecord] = []
			for branch_record in branch_selection.records:
				points_xy = _extract_branch_points(locs_xy=gtr_locs_xy, branch_record=branch_record)
				if len(points_xy) < 2:
					continue
				branches.append(
					FullChipLayoutBranchRecord(
						branch_id=int(branch_record.branch_id),
						branch_index=int(branch_record.branch_index),
						label=branch_record.label,
						points_xy=points_xy,
					)
				)
				unit_entry["branches"].append(
					{
						"branch_id": int(branch_record.branch_id),
						"branch_index": int(branch_record.branch_index),
						"label": branch_record.label,
						"point_count": len(points_xy),
					}
				)
			if branches:
				unit_records.append(
					FullChipLayoutUnitRecord(
						unit_id=unit_id,
						color=unit_color,
						branches=tuple(branches),
					)
				)
				unit_entry["status"] = "ok"
				unit_entry["branch_count"] = len(branches)
				units_plotted += 1
				branches_total += len(branches)
			else:
				unit_entry["status"] = "skipped"
				unit_entry["error"] = f"No {phase_cfg.branch_scope} branches available for unit {unit_id}"
				units_skipped += 1
		except Exception as exc:
			unit_entry["status"] = "error"
			unit_entry["error"] = str(exc)
			units_error += 1
			active_logger.exception("Failed full-chip layout branch collection for unit %s", unit_id)
		manifest_units.append(unit_entry)

	manifest: dict[str, Any] = {
		"phase": "report_full_chip_layout",
		"branch_scope": str(phase_cfg.branch_scope),
		"probe_geometry": _probe_geometry_payload(inputs.probe_geometry),
		"unit_color_strategy": str(phase_cfg.unit_colors.strategy),
		"unit_color_scheme": str(phase_cfg.unit_colors.color_scheme),
		"units_total": len(unit_results),
		"units_successful": len(successful_units),
		"units_plotted": int(units_plotted),
		"units_error": int(units_error),
		"units_skipped": int(units_skipped),
		"branches_total": int(branches_total),
		"units": manifest_units,
	}
	if branches_total <= 0:
		active_logger.warning(
			"reconstruct.report_full_chip_layout no plottable branches found branch_scope=%s units_successful=%d units_error=%d units_skipped=%d manifest=%s",
			str(phase_cfg.branch_scope),
			len(successful_units),
			int(units_error),
			int(units_skipped),
			output_paths["manifest_json"],
		)
		write_json_fn(output_paths["manifest_json"], manifest)
		raise FileNotFoundError(
			f"No successful reconstructed units with {phase_cfg.branch_scope} branches available for reconstruct.report_full_chip_layout"
		)

	stage_outputs.update(
		write_full_chip_layout_plot_fn(
			output_png=output_paths["png_path"],
			output_svg=output_paths["svg_path"],
			unit_records=unit_records,
			probe_geometry=inputs.probe_geometry,
			display_config=phase_cfg.display,
			output_config=phase_cfg.output,
		)
	)
	stage_outputs["full_chip_layout_manifest_json"] = str(output_paths["manifest_json"])
	manifest["outputs"] = {
		key: value
		for key, value in stage_outputs.items()
		if key in {"full_chip_layout_png", "full_chip_layout_svg", "full_chip_layout_manifest_json"}
	}
	write_json_fn(output_paths["manifest_json"], manifest)
	active_logger.info(
		"reconstruct.report_full_chip_layout wrote outputs manifest=%s png=%s svg=%s units_successful=%d units_plotted=%d branches_total=%d",
		output_paths["manifest_json"],
		stage_outputs.get("full_chip_layout_png"),
		stage_outputs.get("full_chip_layout_svg"),
		len(successful_units),
		int(units_plotted),
		int(branches_total),
	)
	return stage_outputs


__all__ = [
	"FullChipLayoutBranchRecord",
	"FullChipLayoutUnitRecord",
	"run_report_full_chip_layout_phase",
	"write_full_chip_layout_plot",
]