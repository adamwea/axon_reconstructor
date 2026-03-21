from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

from .core.render import render_template_plot
from .io import read_json, resolve_unit_output_paths, write_json
from .models.inputs import TemplatesInputs
from .models.results import TemplatesResult, UnitTemplatesResult


LOGGER = logging.getLogger("axon_recon.templates")


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


def _load_merged_unit(unit_dir: Path) -> tuple[np.ndarray, np.ndarray]:
	tmpl = np.load(unit_dir / "merged_contributing_template.npy")
	locs = np.load(unit_dir / "merged_contributing_channel_locations.npy")
	locs = np.asarray(locs, dtype=float)[:, :2]
	tmpl = np.asarray(tmpl)

	if tmpl.ndim != 2:
		raise ValueError(f"Unexpected merged template shape: {tmpl.shape}")
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Unexpected merged locations shape: {locs.shape}")

	if int(tmpl.shape[0]) == int(locs.shape[0]):
		return tmpl, locs
	if int(tmpl.shape[1]) == int(locs.shape[0]):
		return tmpl.T, locs
	return tmpl, locs


def _load_full_unit(full_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
	tmpl_path = full_dir / "full_template.npy"
	locs_path = full_dir / "full_channel_locations_xy.npy"
	if not tmpl_path.exists() or not locs_path.exists():
		return None

	tmpl = np.load(tmpl_path)
	locs = np.load(locs_path)
	locs = np.asarray(locs, dtype=float)[:, :2]
	tmpl = np.asarray(tmpl)

	if tmpl.ndim != 2:
		raise ValueError(f"Unexpected full template shape: {tmpl.shape}")
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Unexpected full locations shape: {locs.shape}")

	if int(tmpl.shape[0]) == int(locs.shape[0]):
		return tmpl, locs
	if int(tmpl.shape[1]) == int(locs.shape[0]):
		return tmpl.T, locs
	return tmpl, locs


def _select_template_for_scope(
	*,
	merged_template: np.ndarray,
	merged_locs: np.ndarray,
	full_payload: tuple[np.ndarray, np.ndarray] | None,
	channel_scope: str,
) -> tuple[np.ndarray, np.ndarray, str]:
	scope = str(channel_scope or "contributing_channels")
	if scope == "contributing_channels":
		return merged_template, merged_locs, "merged_contributing"

	if full_payload is None:
		return merged_template, merged_locs, "merged_contributing"

	full_template, full_locs = full_payload
	if scope == "all_channels":
		return full_template, full_locs, "full_channels"

	if scope == "recorded_channels":
		if int(full_template.shape[0]) == 0:
			return full_template, full_locs, "full_channels"
		per_ch_max = np.nanmax(np.abs(full_template), axis=1)
		keep = np.where(per_ch_max > float(np.finfo(float).eps))[0]
		if keep.size > 0:
			return full_template[keep, :], full_locs[keep, :], "full_channels_recorded_only"
		return full_template, full_locs, "full_channels"

	return merged_template, merged_locs, "merged_contributing"


def _build_unit_ids(inputs: TemplatesInputs, merged_units_dir: Path) -> list[Any]:
	discovered = _discover_unit_ids(merged_units_dir)
	unit_ids = list(inputs.unit_ids) if inputs.unit_ids is not None else discovered
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	return unit_ids


def run_templates_stage(inputs: TemplatesInputs) -> TemplatesResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	templates_out_dir = well_out_dir / str(inputs.output_rel_root)
	templates_out_dir.mkdir(parents=True, exist_ok=True)

	_, merged_units_dir, full_channels_templates_dir = _resolve_templates_dirs(well_out_dir)
	unit_ids = _build_unit_ids(inputs, merged_units_dir)

	def _process_unit(unit_id: Any) -> UnitTemplatesResult:
		paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		paths["unit_dir"].mkdir(parents=True, exist_ok=True)

		if (not bool(inputs.force_restart)) and paths["unit_summary_json"].exists():
			try:
				existing = read_json(paths["unit_summary_json"])
				outputs = dict((existing or {}).get("outputs", {})) if isinstance(existing, dict) else {}
				return UnitTemplatesResult(
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
			merged_dir = merged_units_dir / f"unit_{unit_id}"
			full_dir = full_channels_templates_dir / f"unit_{unit_id}"

			merged_template, merged_locs = _load_merged_unit(merged_dir)
			full_payload = _load_full_unit(full_dir) if full_channels_templates_dir.exists() else None

			template_plot, locs_plot, source = _select_template_for_scope(
				merged_template=merged_template,
				merged_locs=merged_locs,
				full_payload=full_payload,
				channel_scope=inputs.per_unit_outputs.template.channel_scope,
			)
			unit_summary["selected_template_source"] = source

			outputs = render_template_plot(
				template=template_plot,
				locations_xy=locs_plot,
				config=inputs.per_unit_outputs.template,
				png_path=paths["template_png"],
				svg_path=paths["template_svg"],
			)
			unit_summary["outputs"].update(outputs)
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			LOGGER.exception("Failed templates for unit %s", unit_id)

		write_json(paths["unit_summary_json"], unit_summary)
		return UnitTemplatesResult(
			unit_id=unit_id,
			status=str(unit_summary.get("status", "ok")),
			outputs={str(k): str(v) for k, v in dict(unit_summary.get("outputs", {})).items()},
			error=(unit_summary.get("error") if unit_summary.get("error") else None),
		)

	unit_results: list[UnitTemplatesResult] = []
	worker_count = int(max(1, int(inputs.n_jobs)))
	if worker_count <= 1 or len(unit_ids) <= 1:
		for unit_id in unit_ids:
			unit_results.append(_process_unit(unit_id))
	else:
		futures: dict[concurrent.futures.Future[UnitTemplatesResult], Any] = {}
		with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
			for unit_id in unit_ids:
				fut = pool.submit(_process_unit, unit_id)
				futures[fut] = unit_id

			for fut in concurrent.futures.as_completed(futures):
				unit_results.append(fut.result())

	unit_results.sort(key=lambda r: str(r.unit_id))

	summary_json = templates_out_dir / "templates_summary.json"
	summary_payload = {
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"n_jobs": int(max(1, int(inputs.n_jobs))),
		"well_out_dir": str(well_out_dir),
		"templates_out_dir": str(templates_out_dir),
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

	return TemplatesResult(
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
		summary_json=summary_json,
		units=unit_results,
	)
