from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from ..models.inputs import ReconstructionInputs
from ..models.results import UnitReconstructionResult


def run_report_recons_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	unit_results: list[UnitReconstructionResult],
	unit_results_for_reports: list[UnitReconstructionResult],
	preserve_stage_reports: bool,
	existing_stage_outputs: dict[str, str],
	resolve_report_output_paths_fn: Callable[..., dict[str, Path]],
	write_amplitude_map_summary_png_fn: Callable[..., bool],
	render_footprint_map_grid_from_assets_fn: Callable[..., dict[str, str]],
	finalize_grid_svg_output_fn: Callable[..., dict[str, str]],
	render_template_report_pdf_fn: Callable[..., dict[str, str]],
	write_reconstruct_report_markdown_fn: Callable[..., Path],
	logger: logging.Logger | None = None,
) -> dict[str, str]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.report_recons")
	phase_cfg = inputs.phases.report_recons

	stage_outputs: dict[str, str] = dict(existing_stage_outputs)
	if bool(phase_cfg.av_recons.write_pdf) and not preserve_stage_reports:
		report_paths = resolve_report_output_paths_fn(
			reconstruction_out_dir=reconstruction_out_dir,
			reports=inputs.reports,
			report_recons_phase=phase_cfg,
		)
		render_units: list[dict[str, Any]] = []
		missing_units: list[Any] = []
		for item in unit_results_for_reports:
			circle_png = item.outputs.get("circle_recon_png") if isinstance(item.outputs, dict) else None
			if circle_png is None:
				missing_units.append(item.unit_id)
				continue
			circle_png_path = Path(str(circle_png))
			if not circle_png_path.exists():
				missing_units.append(item.unit_id)
				continue
			render_units.append({"unit_id": item.unit_id, "image_path": str(circle_png_path)})
		if not render_units:
			raise FileNotFoundError(
				"Missing circle_recon plot assets required for reconstruct.report_recons av_recons.pdf; run reconstruct.plot_recons first"
			)
		active_logger.info(
			"Reconstruct reports av_recons inputs=%d missing_units=%d",
			len(render_units),
			len(missing_units),
		)
		stage_outputs.update(
			render_template_report_pdf_fn(
				units=render_units,
				pdf_path=report_paths["av_recons_pdf"],
				output_key="av_recons_pdf",
				title_suffix="axon reconstruction",
			)
		)

	circle_grid_cfg = inputs.reports.grids.circle_recon_grid
	write_circle_grid = bool(circle_grid_cfg.write_png) or bool(circle_grid_cfg.write_pdf) or bool(circle_grid_cfg.write_svg)
	if write_circle_grid and not preserve_stage_reports:
		report_paths = resolve_report_output_paths_fn(
			reconstruction_out_dir=reconstruction_out_dir,
			reports=inputs.reports,
			report_recons_phase=phase_cfg,
		)
		circle_entries = [
			Path(item.outputs["circle_recon_png"])
			for item in unit_results_for_reports
			if isinstance(item.outputs, dict) and "circle_recon_png" in item.outputs
		]
		active_logger.info("Reconstruct reports circle_recon_grid inputs=%d", len(circle_entries))
		circle_grid_outputs = render_footprint_map_grid_from_assets_fn(
			image_paths=circle_entries,
			config=circle_grid_cfg,
			pdf_path=report_paths["circle_recon_grid_pdf"],
			png_path=report_paths["circle_recon_grid_png"],
			write_svg=bool(circle_grid_cfg.write_svg),
			svg_path=report_paths["circle_recon_grid_temp_svg"],
			svg_output_key="circle_recon_grid_temp_svg",
			pdf_output_key="circle_recon_grid_pdf",
			png_output_key="circle_recon_grid_png",
			title="Reconstruct circle recon grid",
		)
		circle_grid_outputs = finalize_grid_svg_output_fn(
			raw_outputs=circle_grid_outputs,
			write_svg=bool(circle_grid_cfg.write_svg),
			keep_temp_svg=bool(circle_grid_cfg.keep_temp_svg),
			temp_svg_output_key="circle_recon_grid_temp_svg",
			final_svg_output_key="circle_recon_grid_svg",
			temp_svg_path=report_paths["circle_recon_grid_temp_svg"],
			final_svg_path=report_paths["circle_recon_grid_svg"],
			report_name="circle_recon_grid",
			logger=active_logger,
		)
		stage_outputs.update(circle_grid_outputs)

	if bool(inputs.write_summary_png) and not preserve_stage_reports:
		summary_png = reconstruction_out_dir / Path(str(inputs.summary_png_relpath)).expanduser()
		entries: list[tuple[Any, Path]] = []
		for item in unit_results:
			path_like = item.outputs.get("amplitude_map_png") if isinstance(item.outputs, dict) else None
			if path_like:
				entries.append((item.unit_id, Path(str(path_like))))
		if entries:
			wrote = write_amplitude_map_summary_png_fn(
				entries=entries,
				output_png=summary_png,
				ncols=int(max(1, int(inputs.summary_grid_ncols))),
			)
			if wrote and summary_png.exists():
				stage_outputs["summary_png"] = str(summary_png)

	if bool(inputs.write_report_md) and not preserve_stage_reports:
		report_md = reconstruction_out_dir / Path(str(inputs.report_md_relpath)).expanduser()
		unit_rows = [
			{
				"unit_id": item.unit_id,
				"status": item.status,
				"outputs": item.outputs,
				"error": item.error,
			}
			for item in unit_results
		]
		write_reconstruct_report_markdown_fn(
			output_md=report_md,
			h5_path=inputs.h5_path,
			stream_id=inputs.stream_id,
			reconstruction_out_dir=reconstruction_out_dir,
			stage_outputs=stage_outputs,
			unit_rows=unit_rows,
		)
		if report_md.exists():
			stage_outputs["report_md"] = str(report_md)

	return stage_outputs


__all__ = ["run_report_recons_phase"]