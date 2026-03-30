from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
import pickle
from typing import Any

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

from .core.reconstruct import (
	compute_branches_with_polyline,
	compute_gtr_json_payload,
	compute_heuristics_payload,
	compute_raw_branches_payload,
	load_templates_for_unit,
)
from .core.summary_plots import write_amplitude_map_summary_png
from .core.unit_plots import write_unit_amplitude_map_png
from .core.unit_plots import write_unit_circle_recon_plot
from .integrations.axon_velocity import compute_graph_tracking, import_axon_velocity
from .io import read_json, resolve_unit_output_paths, write_json
from .models.inputs import ReconstructionInputs
from .models.results import ReconstructionResult, UnitReconstructionResult
from .reporting.slides import write_reconstruct_report_markdown


LOGGER = logging.getLogger("axon_recon.reconstruct")


def _is_empty_signal_selection_error(exc: Exception) -> bool:
	msg = str(exc).strip().lower()
	if "zero-size array to reduction operation maximum which has no identity" in msg:
		return True
	if "zero-size" in msg and "maximum" in msg:
		return True
	if "no branches found" in msg:
		return True
	return False


def _normalize_template_for_tracking(template: Any, locs_xy: Any) -> Any:
	import numpy as np  # type: ignore[import-not-found]

	tpl = np.asarray(template, dtype=float)
	locs = np.asarray(locs_xy, dtype=float)
	if tpl.ndim != 2 or locs.ndim != 2:
		return template
	n_channels = int(locs.shape[0])
	if int(tpl.shape[0]) == n_channels:
		return tpl
	if int(tpl.shape[1]) == n_channels:
		return np.asarray(tpl.T, dtype=float)
	return tpl


def _discover_unit_ids(merged_units_dir: Path) -> list[Any]:
	unit_ids: list[Any] = []
	for p in sorted(merged_units_dir.iterdir() if merged_units_dir.exists() else []):
		if not p.is_dir():
			continue
		token = p.name
		if token.startswith("unit_"):
			token = token.split("unit_", 1)[1]
		try:
			unit_ids.append(int(token))
		except Exception:
			# Ignore non-unit directories (for example reports/ or cache subfolders).
			continue
	return unit_ids


def _resolve_templates_dirs(
	well_out_dir: Path,
	*,
	load_assets_from_v2pipeline_templates_stage: bool = False,
) -> tuple[Path, Path, Path]:
	if bool(load_assets_from_v2pipeline_templates_stage):
		templates_out_dir = well_out_dir / "template_outputs"
		merged_units_dir = templates_out_dir / "units"
		full_channels_templates_dir = merged_units_dir
		if merged_units_dir.exists():
			return templates_out_dir, merged_units_dir, full_channels_templates_dir

	templates_out_dir = well_out_dir / "stg4_templates_outputs"
	templates_dir = templates_out_dir / "templates"

	merged_units_dir = templates_dir / "merged"
	full_channels_templates_dir = templates_dir / "full"

	if not merged_units_dir.exists():
		legacy = templates_out_dir / "merged_units"
		if legacy.exists():
			merged_units_dir = legacy

	if not full_channels_templates_dir.exists():
		legacy = templates_out_dir / "full_channels_templates"
		if legacy.exists():
			full_channels_templates_dir = legacy

	if not merged_units_dir.exists():
		raise FileNotFoundError(f"Missing merged templates directory: {merged_units_dir}")

	return templates_out_dir, merged_units_dir, full_channels_templates_dir


def _build_unit_ids(inputs: ReconstructionInputs, merged_units_dir: Path) -> list[Any]:
	discovered = _discover_unit_ids(merged_units_dir)
	unit_ids = list(inputs.unit_ids) if inputs.unit_ids is not None else discovered
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	return unit_ids


def run_reconstruct_stage(inputs: ReconstructionInputs) -> ReconstructionResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	reconstruction_out_dir = well_out_dir / str(inputs.output_rel_root)
	reconstruction_out_dir.mkdir(parents=True, exist_ok=True)

	_, merged_units_dir, full_channels_templates_dir = _resolve_templates_dirs(
		well_out_dir,
		load_assets_from_v2pipeline_templates_stage=inputs.load_assets_from_v2pipeline_templates_stage,
	)
	unit_ids = _build_unit_ids(inputs, merged_units_dir)

	av = import_axon_velocity(repo_root=inputs.axon_velocity_repo_root)

	def _process_unit(unit_id: Any) -> UnitReconstructionResult:
		paths = resolve_unit_output_paths(
			reconstruction_out_dir=reconstruction_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		paths["unit_dir"].mkdir(parents=True, exist_ok=True)

		if (not bool(inputs.force_restart)) and (not bool(inputs.force_replot)) and paths["unit_summary_json"].exists():
			try:
				existing = read_json(paths["unit_summary_json"])
				outputs = dict((existing or {}).get("outputs", {})) if isinstance(existing, dict) else {}
				return UnitReconstructionResult(
					unit_id=unit_id,
					status=str((existing or {}).get("status", "ok")),
					outputs={str(k): str(v) for k, v in outputs.items() if v is not None},
					error=(None if not isinstance(existing, dict) else existing.get("error")),
				)
			except Exception:
				pass

		unit_summary: dict[str, Any] = {
			"unit_id": unit_id,
			"status": "ok",
			"error": None,
			"outputs": {},
		}
		try:
			plot_template_ch_by_t, plot_locs_xy, gtr_template_ch_by_t, gtr_locs_xy, fs_hz, selected_template_source = load_templates_for_unit(
				unit_id=unit_id,
				merged_units_dir=merged_units_dir,
				full_channels_templates_dir=full_channels_templates_dir,
				template_source=str(inputs.per_unit_outputs.template_source),
				use_full_channels_templates=inputs.use_full_channels_templates,
				require_full_channels_templates=inputs.require_full_channels_templates,
				probe_geometry=inputs.probe_geometry,
			)
			unit_summary["selected_template_source"] = selected_template_source
			unit_summary["graph_tracking_source"] = selected_template_source

			gtr = None
			primary_exc: Exception | None = None
			primary_template_for_tracking = _normalize_template_for_tracking(gtr_template_ch_by_t, gtr_locs_xy)
			try:
				gtr = compute_graph_tracking(
					av=av,
					template_ch_by_t=primary_template_for_tracking,
					locs_xy=gtr_locs_xy,
					sampling_frequency_hz=float(fs_hz),
					params=dict(inputs.axon_velocity_params),
				)
			except Exception as exc:
				primary_exc = exc
				if (
					_is_empty_signal_selection_error(exc)
					and str(selected_template_source) not in {"merged_contributing", "merged_per_unit_output"}
				):
					raise RuntimeError(
						"Graph tracking failed for requested template source "
						f"{selected_template_source} (unit={unit_id}): {exc}. "
						"Fallback to merged template source is disabled."
					) from exc
				else:
					raise

			if gtr is None:
				if primary_exc is not None:
					raise primary_exc
				raise RuntimeError(f"Graph tracking did not return a result for unit {unit_id}")

			if bool(inputs.per_unit_outputs.write_branches_raw_json):
				payload = compute_raw_branches_payload(unit_id=unit_id, gtr=gtr)
				write_json(paths["branches_raw_json"], payload)
				unit_summary["outputs"]["branches_raw_json"] = str(paths["branches_raw_json"])

			if bool(inputs.per_unit_outputs.write_branches_json):
				payload = compute_branches_with_polyline(unit_id=unit_id, gtr=gtr, locs_xy=plot_locs_xy)
				write_json(paths["branches_json"], payload)
				unit_summary["outputs"]["branches_json"] = str(paths["branches_json"])

			if bool(inputs.per_unit_outputs.write_heuristics_json):
				payload = compute_heuristics_payload(unit_id=unit_id, gtr=gtr, locs_xy=plot_locs_xy)
				write_json(paths["heuristics_json"], payload)
				unit_summary["outputs"]["heuristics_json"] = str(paths["heuristics_json"])

			if bool(inputs.per_unit_outputs.write_gtr_pkl):
				paths["gtr_pkl"].parent.mkdir(parents=True, exist_ok=True)
				with open(paths["gtr_pkl"], "wb") as f:
					pickle.dump(gtr, f)
				unit_summary["outputs"]["gtr_pkl"] = str(paths["gtr_pkl"])

			if bool(inputs.per_unit_outputs.write_gtr_json):
				payload = compute_gtr_json_payload(unit_id=unit_id, gtr=gtr, locs_xy=plot_locs_xy)
				write_json(paths["gtr_json"], payload)
				unit_summary["outputs"]["gtr_json"] = str(paths["gtr_json"])

			if bool(inputs.per_unit_outputs.write_amplitude_map_png):
				amplitude_map_path = paths["amplitude_map_png"]
				if bool(inputs.force_replot) or (not amplitude_map_path.exists()):
					write_unit_amplitude_map_png(
						output_png=amplitude_map_path,
						template_ch_by_t=plot_template_ch_by_t,
						locs_xy=plot_locs_xy,
						heatmap_config=inputs.per_unit_outputs.amplitude_map_heatmap,
					)
				if amplitude_map_path.exists():
					unit_summary["outputs"]["amplitude_map_png"] = str(amplitude_map_path)

			circle_output_cfg = inputs.per_unit_outputs.circle_recon.output
			write_circle_recon = bool(circle_output_cfg.write_png) or bool(circle_output_cfg.write_svg)
			if write_circle_recon:
				circle_png_path = paths["circle_recon_png"]
				circle_svg_path = paths["circle_recon_svg"]
				needs_plot = bool(inputs.force_replot)
				if not needs_plot:
					if bool(circle_output_cfg.write_png) and (not circle_png_path.exists()):
						needs_plot = True
					if bool(circle_output_cfg.write_svg) and (not circle_svg_path.exists()):
						needs_plot = True
				if needs_plot:
					write_unit_circle_recon_plot(
						output_png=circle_png_path,
						output_svg=circle_svg_path,
						template_ch_by_t=plot_template_ch_by_t,
						locs_xy=plot_locs_xy,
						gtr=gtr,
						circle_config=inputs.per_unit_outputs.circle_recon,
						unit_id=unit_id,
					)
				if bool(circle_output_cfg.write_png) and circle_png_path.exists():
					unit_summary["outputs"]["circle_recon_png"] = str(circle_png_path)
				if bool(circle_output_cfg.write_svg) and circle_svg_path.exists():
					unit_summary["outputs"]["circle_recon_svg"] = str(circle_svg_path)

		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			LOGGER.exception("Failed reconstruct for unit %s", unit_id)

		write_json(paths["unit_summary_json"], unit_summary)
		return UnitReconstructionResult(
			unit_id=unit_id,
			status=str(unit_summary.get("status", "ok")),
			outputs={str(k): str(v) for k, v in dict(unit_summary.get("outputs", {})).items()},
			error=(unit_summary.get("error") if unit_summary.get("error") else None),
		)

	unit_results: list[UnitReconstructionResult] = []
	worker_count = int(max(1, int(inputs.n_jobs)))
	if worker_count <= 1 or len(unit_ids) <= 1:
		for unit_id in unit_ids:
			unit_results.append(_process_unit(unit_id))
	else:
		futures: dict[concurrent.futures.Future[UnitReconstructionResult], Any] = {}
		with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
			for unit_id in unit_ids:
				fut = pool.submit(_process_unit, unit_id)
				futures[fut] = unit_id

			for fut in concurrent.futures.as_completed(futures):
				unit_results.append(fut.result())

	# Keep summary output deterministic across serial/threaded modes.
	unit_results.sort(key=lambda r: str(r.unit_id))

	stage_outputs: dict[str, str] = {}
	if bool(inputs.write_summary_png):
		summary_png = reconstruction_out_dir / Path(str(inputs.summary_png_relpath)).expanduser()
		entries: list[tuple[Any, Path]] = []
		for item in unit_results:
			p = item.outputs.get("amplitude_map_png") if isinstance(item.outputs, dict) else None
			if p:
				entries.append((item.unit_id, Path(str(p))))
		if entries:
			wrote = write_amplitude_map_summary_png(
				entries=entries,
				output_png=summary_png,
				ncols=int(max(1, int(inputs.summary_grid_ncols))),
			)
			if wrote and summary_png.exists():
				stage_outputs["summary_png"] = str(summary_png)

	unit_rows = [
		{
			"unit_id": u.unit_id,
			"status": u.status,
			"outputs": u.outputs,
			"error": u.error,
		}
		for u in unit_results
	]

	if bool(inputs.write_report_md):
		report_md = reconstruction_out_dir / Path(str(inputs.report_md_relpath)).expanduser()
		write_reconstruct_report_markdown(
			output_md=report_md,
			h5_path=inputs.h5_path,
			stream_id=inputs.stream_id,
			reconstruction_out_dir=reconstruction_out_dir,
			stage_outputs=stage_outputs,
			unit_rows=unit_rows,
		)
		if report_md.exists():
			stage_outputs["report_md"] = str(report_md)

	summary_json = reconstruction_out_dir / "reconstruction_summary.json"
	summary_payload = {
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"n_jobs": int(max(1, int(inputs.n_jobs))),
		"well_out_dir": str(well_out_dir),
		"reconstruction_out_dir": str(reconstruction_out_dir),
		"outputs": stage_outputs,
		"units": unit_rows,
	}
	write_json(summary_json, summary_payload)

	return ReconstructionResult(
		well_out_dir=well_out_dir,
		reconstruction_out_dir=reconstruction_out_dir,
		summary_json=summary_json,
		units=unit_results,
	)
