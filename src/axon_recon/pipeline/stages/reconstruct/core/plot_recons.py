from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
import pickle
from typing import Any, Callable

from ..models.inputs import ReconstructionInputs
from ..models.results import UnitReconstructionResult
from axon_recon.pipeline.cpu_allocation import current_phase_budget, resolve_inner_worker_count


def _unit_result_from_summary(*, unit_id: Any, payload: Any) -> UnitReconstructionResult:
	data = payload if isinstance(payload, dict) else {}
	outputs = dict(data.get("outputs", {})) if isinstance(data.get("outputs", {}), dict) else {}
	return UnitReconstructionResult(
		unit_id=unit_id,
		status=str(data.get("status", "ok")),
		outputs={str(key): str(value) for key, value in outputs.items() if value is not None},
		error=(None if not data.get("error") else str(data.get("error"))),
	)


def run_plot_recons_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_ids: list[Any],
	load_templates_for_unit_fn: Callable[..., Any],
	write_unit_amplitude_map_png_fn: Callable[..., Any],
	write_unit_circle_recon_plot_fn: Callable[..., Any],
	read_json_fn: Callable[[Path], Any],
	write_json_fn: Callable[[Path, Any], None],
	resolve_unit_output_paths_fn: Callable[..., dict[str, Path]],
	logger: logging.Logger | None = None,
) -> list[UnitReconstructionResult]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.plot_recons")
	force_replot = bool(inputs.force_restart) or bool(inputs.force_replot)
	phase_outputs = inputs.phases.plot_recons.outputs

	def _process_unit(unit_id: Any) -> UnitReconstructionResult:
		paths = resolve_unit_output_paths_fn(
			reconstruction_out_dir=reconstruction_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
			unit_reldir=inputs.unit_reldir,
		)
		unit_summary_json = paths["unit_summary_json"]
		if not unit_summary_json.exists():
			payload = {
				"unit_id": unit_id,
				"status": "error",
				"error": "Missing axon_velocity_gtrs unit summary; run reconstruct.axon_velocity_gtrs first",
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

		if str(unit_summary.get("status", "ok")).strip().lower() != "ok":
			return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

		gtr_path = paths["gtr_pkl"]
		if not gtr_path.exists():
			unit_summary["status"] = "error"
			unit_summary["error"] = "Missing axon_velocity_gtrs artifact gtr.pkl; run reconstruct.axon_velocity_gtrs first"
			write_json_fn(unit_summary_json, unit_summary)
			return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

		try:
			plot_template_ch_by_t, plot_locs_xy, gtr_template_ch_by_t, gtr_locs_xy, _, _ = load_templates_for_unit_fn(
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

			if bool(phase_outputs.amplitude_map.write_png):
				amplitude_map_path = paths["amplitude_map_png"]
				if force_replot or (not amplitude_map_path.exists()):
					write_unit_amplitude_map_png_fn(
						output_png=amplitude_map_path,
						template_ch_by_t=plot_template_ch_by_t,
						locs_xy=plot_locs_xy,
						heatmap_config=phase_outputs.amplitude_map.heatmap,
					)
				if amplitude_map_path.exists():
					unit_summary["outputs"]["amplitude_map_png"] = str(amplitude_map_path)

			circle_output_cfg = phase_outputs.circle_recon.output
			write_circle_recon = bool(circle_output_cfg.write_png) or bool(circle_output_cfg.write_svg)
			if write_circle_recon:
				circle_png_path = paths["circle_recon_png"]
				circle_svg_path = paths["circle_recon_svg"]
				needs_plot = bool(force_replot)
				if not needs_plot:
					if bool(circle_output_cfg.write_png) and (not circle_png_path.exists()):
						needs_plot = True
					if bool(circle_output_cfg.write_svg) and (not circle_svg_path.exists()):
						needs_plot = True
				if needs_plot:
					write_unit_circle_recon_plot_fn(
						output_png=circle_png_path,
						output_svg=circle_svg_path,
						template_ch_by_t=gtr_template_ch_by_t,
						locs_xy=gtr_locs_xy,
						gtr=gtr,
						circle_config=phase_outputs.circle_recon,
						unit_id=unit_id,
					)
				if bool(circle_output_cfg.write_png) and circle_png_path.exists():
					unit_summary["outputs"]["circle_recon_png"] = str(circle_png_path)
				if bool(circle_output_cfg.write_svg) and circle_svg_path.exists():
					unit_summary["outputs"]["circle_recon_svg"] = str(circle_svg_path)
			unit_summary["error"] = None
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			active_logger.exception("Failed reconstruct.plot_recons for unit %s", unit_id)

		write_json_fn(unit_summary_json, unit_summary)
		return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

	_phase_budget = current_phase_budget("reconstruct", "plot_recons")
	worker_count = resolve_inner_worker_count(
		nested_shape=str(getattr(_phase_budget, "nested_shape", "unit_workers") or "unit_workers"),
		phase_cpus_per_task=getattr(_phase_budget, "cpus_per_task", None) if _phase_budget else None,
		yaml_n_jobs_override=int(inputs.n_jobs) if getattr(inputs, "n_jobs", None) is not None else None,
		work_item_count=int(len(unit_ids)) if unit_ids is not None else None,
	)
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


__all__ = ["run_plot_recons_phase"]