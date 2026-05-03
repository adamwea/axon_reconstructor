from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from ..models.inputs import ReconstructionInputs
from ..models.results import UnitReconstructionResult


def run_report_recon_grid_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	unit_results_for_reports: list[UnitReconstructionResult],
	preserve_stage_reports: bool,
	existing_stage_outputs: dict[str, str],
	resolve_report_output_paths_fn: Callable[..., dict[str, Path]],
	render_footprint_map_grid_from_assets_fn: Callable[..., dict[str, str]],
	finalize_grid_svg_output_fn: Callable[..., dict[str, str]],
	logger: logging.Logger | None = None,
) -> dict[str, str]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.report_recon_grid")
	phase_cfg = inputs.phases.report_recon_grid
	grid_cfg = phase_cfg.output
	stage_outputs: dict[str, str] = dict(existing_stage_outputs)
	write_circle_grid = bool(grid_cfg.write_png) or bool(grid_cfg.write_pdf) or bool(grid_cfg.write_svg)
	if (not write_circle_grid) or bool(preserve_stage_reports):
		return stage_outputs

	report_paths = resolve_report_output_paths_fn(
		reconstruction_out_dir=reconstruction_out_dir,
		report_recon_grid_phase=phase_cfg,
	)
	circle_entries = [
		Path(item.outputs["circle_recon_png"])
		for item in unit_results_for_reports
		if isinstance(item.outputs, dict) and "circle_recon_png" in item.outputs
	]
	active_logger.info("Reconstruct reports circle_recon_grid inputs=%d", len(circle_entries))
	render_cfg = SimpleNamespace(
		write_pdf=bool(grid_cfg.write_pdf),
		write_png=bool(grid_cfg.write_png),
		show_title=bool(phase_cfg.display.show_title),
		dpi=float(phase_cfg.render.dpi),
	)
	circle_grid_outputs = render_footprint_map_grid_from_assets_fn(
		image_paths=circle_entries,
		config=render_cfg,
		pdf_path=report_paths["circle_recon_grid_pdf"],
		png_path=report_paths["circle_recon_grid_png"],
		write_svg=bool(grid_cfg.write_svg),
		svg_path=report_paths["circle_recon_grid_temp_svg"],
		svg_output_key="circle_recon_grid_temp_svg",
		pdf_output_key="circle_recon_grid_pdf",
		png_output_key="circle_recon_grid_png",
		title="Reconstruct circle recon grid",
	)
	circle_grid_outputs = finalize_grid_svg_output_fn(
		raw_outputs=circle_grid_outputs,
		write_svg=bool(grid_cfg.write_svg),
		keep_temp_svg=bool(grid_cfg.keep_temp_svg),
		temp_svg_output_key="circle_recon_grid_temp_svg",
		final_svg_output_key="circle_recon_grid_svg",
		temp_svg_path=report_paths["circle_recon_grid_temp_svg"],
		final_svg_path=report_paths["circle_recon_grid_svg"],
		report_name="circle_recon_grid",
		logger=active_logger,
	)
	stage_outputs.update(circle_grid_outputs)
	return stage_outputs


__all__ = ["run_report_recon_grid_phase"]
