from __future__ import annotations

import concurrent.futures
from dataclasses import replace
import logging
from pathlib import Path
import pickle
from typing import Any, Callable

from .branch_styles import select_reconstruct_branch_records
from .plot_branch_propagations import write_unit_branch_propagation_plot
from .plot_branch_velocities import prepare_branch_velocity_plot_data
from .plot_branch_velocities import write_unit_branch_velocity_plot
from .unit_plots import write_unit_circle_recon_plot
from ..models.inputs import ReconstructionInputs
from ..models.results import UnitReconstructionResult


def _unit_result_from_summary(*, unit_id: Any, payload: Any) -> UnitReconstructionResult:
	data = payload if isinstance(payload, dict) else {}
	outputs = dict(data.get("outputs", {})) if isinstance(data.get("outputs", {}), dict) else {}
	return UnitReconstructionResult(
		unit_id=unit_id,
		status=str(data.get("status", "ok")),
		outputs={str(key): str(value) for key, value in outputs.items() if value is not None},
		error=(None if not data.get("error") else str(data.get("error"))),
	)


def _estimate_circle_panel_size(*, velocity_figsize: tuple[float, float], circle_config: Any) -> tuple[float, float]:
	base_mode = str(getattr(getattr(circle_config, "display", None), "base", "template_circles") or "template_circles")
	base_mode = base_mode.strip().lower()
	velocity_width = float(max(1.0, float(velocity_figsize[0])))
	velocity_height = float(max(1.0, float(velocity_figsize[1])))
	if base_mode in {"amplitude_map", "latency_map"}:
		return (max(4.5, velocity_width), max(4.5, velocity_height))
	side = max(5.0, velocity_height)
	return (side, side)


def _estimate_embedded_velocity_panel_width(
	*,
	velocity_figsize: tuple[float, float],
	show_velocity_legend: bool,
	reserve_velocity_legend_space: bool,
) -> float:
	standalone_figure_width = float(max(1.0, float(velocity_figsize[0])))
	# Match the effective standalone plot-area width after Matplotlib margins and the
	# branch-velocity legend gutter are applied in write_unit_branch_velocity_plot.
	if bool(show_velocity_legend) and bool(reserve_velocity_legend_space):
		return float(max(1.0, standalone_figure_width * (0.72 - 0.125)))
	return float(max(1.0, standalone_figure_width * (0.90 - 0.125)))


def _resolve_optional_positive_float(value: Any, fallback: float) -> float:
	try:
		if value is None:
			raise ValueError()
		parsed = float(value)
		if parsed <= 0.0:
			raise ValueError()
		return parsed
	except Exception:
		return float(max(1.0, fallback))


def write_unit_summary_plot(
	*,
	output_png: Path,
	output_svg: Path,
	template_ch_by_t: Any,
	locs_xy: Any,
	gtr: Any,
	circle_config: Any,
	branch_propagation_phase_config: Any,
	branch_velocity_phase_config: Any,
	branch_colors: Any,
	display_config: Any,
	output_config: Any,
	unit_id: Any,
	logger: logging.Logger | None = None,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	active_logger = logger or logging.getLogger("axon_recon.reconstruct.plot_unit_summary")
	propagation_selection = select_reconstruct_branch_records(
		gtr=gtr,
		branch_scope=str(getattr(branch_propagation_phase_config, "branch_scope", "raw") or "raw"),
		branch_colors=branch_colors,
	)
	if len(propagation_selection.records) <= 0:
		raise ValueError(
			f"No {getattr(branch_propagation_phase_config, 'branch_scope', 'raw')} branches available for reconstruct.plot_unit_summary"
		)

	velocity_plot_data = prepare_branch_velocity_plot_data(
		gtr=gtr,
		branch_scope=str(getattr(branch_velocity_phase_config, "branch_scope", "raw") or "raw"),
		branch_colors=branch_colors,
		logger=active_logger,
		unit_id=unit_id,
	)
	velocity_branch_records = tuple(velocity_plot_data["valid_branch_records"])
	velocity_fit_payloads = tuple(velocity_plot_data["valid_fit_payloads"])
	if len(velocity_branch_records) <= 0:
		raise ValueError("No branch velocity figure could be prepared for reconstruct.plot_unit_summary")

	velocity_figsize = tuple(getattr(branch_velocity_phase_config.display, "figsize", (6.0, 4.0)) or (6.0, 4.0))
	propagation_figsize = tuple(
		getattr(branch_propagation_phase_config.display, "figsize", (2.75, 6.0)) or (2.75, 6.0)
	)
	estimated_circle_width, estimated_circle_height = _estimate_circle_panel_size(
		velocity_figsize=velocity_figsize,
		circle_config=circle_config,
	)
	velocity_height = float(max(1.0, float(velocity_figsize[1])))
	propagation_panel_width = _resolve_optional_positive_float(
		getattr(display_config, "propagation_panel_width", None),
		float(max(1.0, float(propagation_figsize[0]))),
	)
	propagation_panel_height = _resolve_optional_positive_float(
		getattr(display_config, "propagation_row_height", None),
		float(max(1.0, float(propagation_figsize[1]))),
	)
	top_row_height = _resolve_optional_positive_float(
		getattr(display_config, "top_row_height", None),
		float(max(estimated_circle_height, velocity_height)),
	)
	circle_width = _resolve_optional_positive_float(
		getattr(display_config, "circle_panel_width", None),
		float(max(estimated_circle_width, top_row_height if estimated_circle_width == estimated_circle_height else estimated_circle_width)),
	)
	legend_width = _resolve_optional_positive_float(
		getattr(display_config, "velocity_legend_width", None),
		float(getattr(display_config, "velocity_legend_width", 2.25) or 2.25),
	)
	top_row_panel_gap_width_raw = getattr(display_config, "top_row_panel_gap_width", None)
	top_row_panel_gap_width = 0.0
	if top_row_panel_gap_width_raw is not None:
		try:
			resolved_gap = float(top_row_panel_gap_width_raw)
		except Exception:
			resolved_gap = 0.0
		top_row_panel_gap_width = float(max(0.0, resolved_gap))
	branch_count = max(1, int(len(propagation_selection.records)))
	show_velocity_legend_cfg = getattr(display_config, "show_velocity_legend", None)
	show_velocity_legend = (
		bool(getattr(branch_velocity_phase_config.display, "show_legend", True))
		if show_velocity_legend_cfg is None
		else bool(show_velocity_legend_cfg)
	)
	reserve_velocity_legend_space_cfg = getattr(display_config, "reserve_velocity_legend_space", None)
	reserve_velocity_legend_space = (
		bool(show_velocity_legend)
		if reserve_velocity_legend_space_cfg is None
		else bool(reserve_velocity_legend_space_cfg)
	)
	velocity_width = _resolve_optional_positive_float(
		getattr(display_config, "velocity_panel_width", None),
		_estimate_embedded_velocity_panel_width(
			velocity_figsize=velocity_figsize,
			show_velocity_legend=show_velocity_legend,
			reserve_velocity_legend_space=reserve_velocity_legend_space,
		),
	)
	reserved_legend_width = legend_width if (show_velocity_legend and reserve_velocity_legend_space) else 0.0
	top_row_width = float(circle_width + top_row_panel_gap_width + velocity_width + reserved_legend_width)
	bottom_row_width = float(propagation_panel_width * float(branch_count))
	fig_width = max(top_row_width, bottom_row_width)
	fig_height = float(top_row_height + propagation_panel_height)
	dpi = float(max(72.0, float(getattr(output_config, "dpi", 300.0) or 300.0)))
	fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
	try:
		fig.patch.set_facecolor("black")
		outer_grid = fig.add_gridspec(
			2,
			1,
			height_ratios=[top_row_height, propagation_panel_height],
			hspace=0.16,
		)
		if top_row_panel_gap_width > 0.0:
			if reserved_legend_width > 0.0:
				top_grid = outer_grid[0].subgridspec(
					1,
					4,
					width_ratios=[circle_width, top_row_panel_gap_width, velocity_width, reserved_legend_width],
					wspace=0.0,
				)
				ax_circle = fig.add_subplot(top_grid[0, 0])
				ax_velocity = fig.add_subplot(top_grid[0, 2])
			else:
				top_grid = outer_grid[0].subgridspec(
					1,
					3,
					width_ratios=[circle_width, top_row_panel_gap_width, velocity_width],
					wspace=0.0,
				)
				ax_circle = fig.add_subplot(top_grid[0, 0])
				ax_velocity = fig.add_subplot(top_grid[0, 2])
		else:
			if reserved_legend_width > 0.0:
				top_grid = outer_grid[0].subgridspec(
					1,
					3,
					width_ratios=[circle_width, velocity_width, reserved_legend_width],
					wspace=0.08,
				)
				ax_circle = fig.add_subplot(top_grid[0, 0])
				ax_velocity = fig.add_subplot(top_grid[0, 1])
			else:
				top_grid = outer_grid[0].subgridspec(
					1,
					2,
					width_ratios=[circle_width, velocity_width],
					wspace=0.08,
				)
				ax_circle = fig.add_subplot(top_grid[0, 0])
				ax_velocity = fig.add_subplot(top_grid[0, 1])
		bottom_grid = outer_grid[1].subgridspec(1, branch_count, wspace=0.06)
		propagation_axes = [fig.add_subplot(bottom_grid[0, idx]) for idx in range(branch_count)]

		summary_circle_config = replace(
			circle_config,
			output=replace(circle_config.output, write_png=False, write_svg=False),
		)
		summary_propagation_output = replace(
			branch_propagation_phase_config.output,
			write_png=False,
			write_svg=False,
		)
		summary_velocity_output = replace(
			branch_velocity_phase_config.output,
			write_png=False,
			write_svg=False,
		)

		write_unit_circle_recon_plot(
			output_png=output_png,
			output_svg=output_svg,
			template_ch_by_t=template_ch_by_t,
			locs_xy=locs_xy,
			gtr=gtr,
			circle_config=summary_circle_config,
			unit_id=unit_id,
			fig=fig,
			ax=ax_circle,
			close_figure=False,
		)
		write_unit_branch_velocity_plot(
			output_png=output_png,
			output_svg=output_svg,
			branch_records=velocity_branch_records,
			fit_payloads=velocity_fit_payloads,
			display_config=branch_velocity_phase_config.display,
			output_config=summary_velocity_output,
			unit_id=unit_id,
			fig=fig,
			ax=ax_velocity,
			close_figure=False,
			manage_layout=False,
			show_legend=show_velocity_legend,
			reserve_legend_space=False,
		)
		write_unit_branch_propagation_plot(
			output_png=output_png,
			output_svg=output_svg,
			template_ch_by_t=template_ch_by_t,
			locs_xy=locs_xy,
			branch_records=propagation_selection.records,
			display_config=branch_propagation_phase_config.display,
			output_config=summary_propagation_output,
			unit_id=unit_id,
			fig=fig,
			axes=propagation_axes,
			close_figure=False,
			manage_layout=False,
			show_figure_title=False,
		)

		if bool(getattr(display_config, "show_title", False)):
			fig.suptitle(f"Unit {unit_id} summary", color="white")
		fig.subplots_adjust(
			left=0.02,
			right=0.98,
			bottom=0.02,
			top=(0.94 if bool(getattr(display_config, "show_title", False)) else 0.985),
		)

		outputs: dict[str, str] = {}
		if bool(getattr(output_config, "write_png", False)):
			output_png.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
			outputs["png_path"] = str(output_png)
		if bool(getattr(output_config, "write_svg", False)):
			output_svg.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(output_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
			outputs["svg_path"] = str(output_svg)
		return outputs
	finally:
		plt.close(fig)


def run_plot_unit_summary_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_ids: list[Any],
	load_templates_for_unit_fn: Callable[..., Any],
	write_unit_summary_plot_fn: Callable[..., dict[str, str]],
	read_json_fn: Callable[[Path], Any],
	write_json_fn: Callable[[Path, Any], None],
	resolve_unit_output_paths_fn: Callable[..., dict[str, Path]],
	resolve_unit_summary_phase_output_paths_fn: Callable[..., dict[str, Path]],
	logger: logging.Logger | None = None,
) -> list[UnitReconstructionResult]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.plot_unit_summary")
	force_replot = bool(inputs.force_restart) or bool(inputs.force_replot)
	phase_cfg = inputs.phases.plot_unit_summary
	phase_name = "plot_unit_summary"

	def _process_unit(unit_id: Any) -> UnitReconstructionResult:
		paths = resolve_unit_output_paths_fn(
			reconstruction_out_dir=reconstruction_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		unit_summary_json = paths["unit_summary_json"]
		if not unit_summary_json.exists():
			payload = {
				"unit_id": unit_id,
				"status": "error",
				"error": "Missing generate_gtrs unit summary; run reconstruct.generate_gtrs first",
				"outputs": {},
			}
			write_json_fn(unit_summary_json, payload)
			return _unit_result_from_summary(unit_id=unit_id, payload=payload)

		payload = read_json_fn(unit_summary_json)
		unit_summary = dict(payload) if isinstance(payload, dict) else {"unit_id": unit_id, "outputs": {}}
		unit_summary.setdefault("unit_id", unit_id)
		unit_summary.setdefault("outputs", {})
		if not isinstance(unit_summary["outputs"], dict):
			unit_summary["outputs"] = {}

		if not (bool(phase_cfg.output.write_png) or bool(phase_cfg.output.write_svg)):
			unit_summary["status"] = "error"
			unit_summary["error"] = "No unit summary outputs enabled"
			write_json_fn(unit_summary_json, unit_summary)
			return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

		gtr_path = paths["gtr_pkl"]
		if not gtr_path.exists():
			unit_summary["status"] = "error"
			unit_summary["error"] = "Missing generate_gtrs artifact gtr.pkl; run reconstruct.generate_gtrs first"
			write_json_fn(unit_summary_json, unit_summary)
			return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

		try:
			_, _, gtr_template_ch_by_t, gtr_locs_xy, _, _ = load_templates_for_unit_fn(
				unit_id=unit_id,
				merged_units_dir=merged_units_dir,
				full_channels_templates_dir=full_channels_templates_dir,
				template_source=str(inputs.per_unit_outputs.template_source),
				use_full_channels_templates=inputs.use_full_channels_templates,
				require_full_channels_templates=inputs.require_full_channels_templates,
				probe_geometry=inputs.probe_geometry,
			)
			with open(gtr_path, "rb") as handle:
				gtr = pickle.load(handle)

			phase_paths = resolve_unit_summary_phase_output_paths_fn(
				reconstruction_out_dir=reconstruction_out_dir,
				unit_id=unit_id,
				per_unit_outputs=inputs.per_unit_outputs,
				phase_output=phase_cfg.output,
			)
			needs_plot = bool(force_replot)
			if not needs_plot:
				if bool(phase_cfg.output.write_png) and (not phase_paths["png_path"].exists()):
					needs_plot = True
				if bool(phase_cfg.output.write_svg) and (not phase_paths["svg_path"].exists()):
					needs_plot = True
			if needs_plot:
				write_unit_summary_plot_fn(
					output_png=phase_paths["png_path"],
					output_svg=phase_paths["svg_path"],
					template_ch_by_t=gtr_template_ch_by_t,
					locs_xy=gtr_locs_xy,
					gtr=gtr,
					circle_config=inputs.per_unit_outputs.circle_recon,
					branch_propagation_phase_config=inputs.phases.plot_branch_propagations,
					branch_velocity_phase_config=inputs.phases.plot_branch_velocities,
					branch_colors=inputs.branch_colors,
					display_config=phase_cfg.display,
					output_config=phase_cfg.output,
					unit_id=unit_id,
					logger=active_logger,
				)

			has_figure_output = False
			if bool(phase_cfg.output.write_png) and phase_paths["png_path"].exists():
				unit_summary["outputs"]["plot_unit_summary_png"] = str(phase_paths["png_path"])
				has_figure_output = True
			if bool(phase_cfg.output.write_svg) and phase_paths["svg_path"].exists():
				unit_summary["outputs"]["plot_unit_summary_svg"] = str(phase_paths["svg_path"])
				has_figure_output = True
			if has_figure_output:
				unit_summary["status"] = "ok"
				unit_summary["error"] = None
			else:
				unit_summary["status"] = "error"
				unit_summary["error"] = "No unit summary figure was written"
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			active_logger.exception("Failed reconstruct.plot_unit_summary for unit %s", unit_id)

		write_json_fn(unit_summary_json, unit_summary)
		return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

	worker_count = int(max(1, int(inputs.n_jobs)))
	unit_results: list[UnitReconstructionResult] = []
	if worker_count <= 1 or len(unit_ids) <= 1:
		for unit_id in unit_ids:
			unit_results.append(_process_unit(unit_id))
	else:
		with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
			futures = {pool.submit(_process_unit, unit_id): unit_id for unit_id in unit_ids}
			for future in concurrent.futures.as_completed(futures):
				unit_results.append(future.result())

	unit_results.sort(key=lambda item: str(item.unit_id))
	return unit_results


__all__ = ["run_plot_unit_summary_phase", "write_unit_summary_plot"]