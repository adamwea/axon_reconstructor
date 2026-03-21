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
from .integrations.axon_velocity import compute_graph_tracking, import_axon_velocity
from .io import read_json, resolve_unit_output_paths, write_json
from .models.inputs import ReconstructionInputs
from .models.results import ReconstructionResult, UnitReconstructionResult


LOGGER = logging.getLogger("axon_recon.reconstruct")


def _discover_unit_ids(merged_units_dir: Path) -> list[Any]:
	unit_ids: list[Any] = []
	for p in sorted(merged_units_dir.glob("unit_*")):
		if not p.is_dir():
			continue
		token = p.name.split("unit_", 1)[1]
		try:
			unit_ids.append(int(token))
		except Exception:
			unit_ids.append(token)
	return unit_ids


def _resolve_templates_dirs(well_out_dir: Path) -> tuple[Path, Path, Path]:
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

	_, merged_units_dir, full_channels_templates_dir = _resolve_templates_dirs(well_out_dir)
	unit_ids = _build_unit_ids(inputs, merged_units_dir)

	av = import_axon_velocity(repo_root=inputs.axon_velocity_repo_root)

	def _process_unit(unit_id: Any) -> UnitReconstructionResult:
		paths = resolve_unit_output_paths(
			reconstruction_out_dir=reconstruction_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		paths["unit_dir"].mkdir(parents=True, exist_ok=True)

		if (not bool(inputs.force_restart)) and paths["unit_summary_json"].exists():
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
			template_ch_by_t, locs_xy, fs_hz, selected_template_source = load_templates_for_unit(
				unit_id=unit_id,
				merged_units_dir=merged_units_dir,
				full_channels_templates_dir=full_channels_templates_dir,
				use_full_channels_templates=inputs.use_full_channels_templates,
				require_full_channels_templates=inputs.require_full_channels_templates,
			)
			unit_summary["selected_template_source"] = selected_template_source

			gtr = compute_graph_tracking(
				av=av,
				template_ch_by_t=template_ch_by_t,
				locs_xy=locs_xy,
				sampling_frequency_hz=float(fs_hz),
				params=dict(inputs.axon_velocity_params),
			)

			if bool(inputs.per_unit_outputs.write_branches_raw_json):
				payload = compute_raw_branches_payload(unit_id=unit_id, gtr=gtr)
				write_json(paths["branches_raw_json"], payload)
				unit_summary["outputs"]["branches_raw_json"] = str(paths["branches_raw_json"])

			if bool(inputs.per_unit_outputs.write_branches_json):
				payload = compute_branches_with_polyline(unit_id=unit_id, gtr=gtr, locs_xy=locs_xy)
				write_json(paths["branches_json"], payload)
				unit_summary["outputs"]["branches_json"] = str(paths["branches_json"])

			if bool(inputs.per_unit_outputs.write_heuristics_json):
				payload = compute_heuristics_payload(unit_id=unit_id, gtr=gtr, locs_xy=locs_xy)
				write_json(paths["heuristics_json"], payload)
				unit_summary["outputs"]["heuristics_json"] = str(paths["heuristics_json"])

			if bool(inputs.per_unit_outputs.write_gtr_pkl):
				paths["gtr_pkl"].parent.mkdir(parents=True, exist_ok=True)
				with open(paths["gtr_pkl"], "wb") as f:
					pickle.dump(gtr, f)
				unit_summary["outputs"]["gtr_pkl"] = str(paths["gtr_pkl"])

			if bool(inputs.per_unit_outputs.write_gtr_json):
				payload = compute_gtr_json_payload(unit_id=unit_id, gtr=gtr, locs_xy=locs_xy)
				write_json(paths["gtr_json"], payload)
				unit_summary["outputs"]["gtr_json"] = str(paths["gtr_json"])

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

	summary_json = reconstruction_out_dir / "reconstruction_summary.json"
	summary_payload = {
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"n_jobs": int(max(1, int(inputs.n_jobs))),
		"well_out_dir": str(well_out_dir),
		"reconstruction_out_dir": str(reconstruction_out_dir),
		"units": [
			{
				"unit_id": u.unit_id,
				"status": u.status,
				"outputs": u.outputs,
				"error": u.error,
			}
			for u in unit_results
		],
	}
	write_json(summary_json, summary_payload)

	return ReconstructionResult(
		well_out_dir=well_out_dir,
		reconstruction_out_dir=reconstruction_out_dir,
		summary_json=summary_json,
		units=unit_results,
	)

