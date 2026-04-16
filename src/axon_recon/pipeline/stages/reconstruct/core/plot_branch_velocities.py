from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
import pickle
from typing import Any, Callable

from .branch_styles import ReconstructBranchRecord
from .branch_styles import select_reconstruct_branch_records
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


def _fit_branch_velocity(gtr: Any, branch_record: ReconstructBranchRecord) -> dict[str, Any]:
	import numpy as np  # type: ignore[import-not-found]

	if (
		branch_record.velocity is not None
		and branch_record.offset is not None
		and len(branch_record.peak_times) >= 2
		and len(branch_record.peak_times) == len(branch_record.distances)
	):
		return {
			"velocity": float(branch_record.velocity),
			"offset": float(branch_record.offset),
			"r2": (None if branch_record.r2 is None else float(branch_record.r2)),
			"peak_times": [float(value) for value in branch_record.peak_times],
			"distances": [float(value) for value in branch_record.distances],
		}

	est = getattr(gtr, "_estimate_peaks_and_dists", None)
	rve = getattr(gtr, "robust_velocity_estimator", None)
	if not callable(est) or not callable(rve):
		raise ValueError("Graph tracking result does not expose raw branch velocity fitting helpers")

	path = [int(ch) for ch in branch_record.selected_channels[1:]]
	if len(path) < 2:
		raise ValueError(f"Branch {branch_record.branch_id} does not have enough path points for velocity fitting")

	peaks, dists = est(path)
	peaks = np.asarray(peaks, dtype=float)
	dists = np.asarray(dists, dtype=float)
	if peaks.size < 2 or dists.size != peaks.size:
		raise ValueError(f"Branch {branch_record.branch_id} did not yield enough peak-distance samples")

	result = rve(path, peaks, dists, True)
	try:
		_, velocity, offset, r2, _p_value, dists_clean, peaks_clean, inlier_mask = result
	except Exception as exc:
		raise ValueError(f"Unexpected velocity fit payload for branch {branch_record.branch_id}") from exc

	clean_peaks = np.asarray(peaks_clean, dtype=float) if peaks_clean is not None else np.asarray([], dtype=float)
	clean_dists = np.asarray(dists_clean, dtype=float) if dists_clean is not None else np.asarray([], dtype=float)
	if clean_peaks.size < 2 or clean_dists.size != clean_peaks.size:
		try:
			mask = np.asarray(inlier_mask, dtype=bool)
		except Exception:
			mask = np.ones_like(peaks, dtype=bool)
		clean_peaks = peaks[mask]
		clean_dists = dists[mask]
	if clean_peaks.size < 2 or clean_dists.size != clean_peaks.size:
		raise ValueError(f"Branch {branch_record.branch_id} did not retain enough inlier samples for velocity plotting")

	return {
		"velocity": float(velocity),
		"offset": float(offset),
		"r2": (None if r2 is None else float(r2)),
		"peak_times": [float(value) for value in clean_peaks.tolist()],
		"distances": [float(value) for value in clean_dists.tolist()],
	}


def write_unit_branch_velocity_plot(
	*,
	output_png: Path,
	output_svg: Path,
	branch_record: ReconstructBranchRecord,
	fit_payload: dict[str, Any],
	display_config: Any,
	output_config: Any,
	unit_id: Any,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	from axon_velocity.plotting import plot_velocity  # type: ignore[import-not-found]

	figsize = tuple(getattr(display_config, "figsize", (6.0, 4.0)) or (6.0, 4.0))
	fig, ax = plt.subplots(figsize=figsize, dpi=float(max(72.0, float(getattr(output_config, "dpi", 300.0) or 300.0))))
	plot_velocity(
		peak_times=fit_payload["peak_times"],
		distances=fit_payload["distances"],
		velocity=float(fit_payload["velocity"]),
		offset=float(fit_payload["offset"]),
		color=str(branch_record.color),
		r2=fit_payload.get("r2", None),
		ax=ax,
	)
	legend = ax.get_legend()
	if legend is not None:
		if not bool(getattr(display_config, "show_legend", True)):
			legend.remove()
		else:
			fontsize = float(max(1.0, float(getattr(display_config, "legend_fontsize", 8.0) or 8.0)))
			for text in legend.get_texts():
				text.set_fontsize(fontsize)
	if bool(getattr(display_config, "show_title", True)):
		ax.set_title(f"Unit {unit_id} branch {branch_record.label} velocity")
	outputs: dict[str, str] = {}
	if bool(getattr(output_config, "write_png", False)):
		output_png.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_png, dpi=float(max(72.0, float(getattr(output_config, "dpi", 300.0) or 300.0))), bbox_inches="tight")
		outputs["png_path"] = str(output_png)
	if bool(getattr(output_config, "write_svg", False)):
		output_svg.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_svg, bbox_inches="tight")
		outputs["svg_path"] = str(output_svg)
	plt.close(fig)
	return outputs


def run_plot_branch_velocities_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_ids: list[Any],
	load_templates_for_unit_fn: Callable[..., Any],
	write_unit_branch_velocity_plot_fn: Callable[..., dict[str, str]],
	read_json_fn: Callable[[Path], Any],
	write_json_fn: Callable[[Path, Any], None],
	resolve_unit_output_paths_fn: Callable[..., dict[str, Path]],
	resolve_branch_phase_output_paths_fn: Callable[..., dict[str, Path]],
	resolve_branch_phase_branch_output_paths_fn: Callable[..., dict[str, Path]],
	logger: logging.Logger | None = None,
) -> list[UnitReconstructionResult]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.plot_branch_velocities")
	force_replot = bool(inputs.force_restart) or bool(inputs.force_replot)
	phase_cfg = inputs.phases.plot_branch_velocities
	phase_name = "plot_branch_velocities"

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
			unit_summary["error"] = "No branch velocity outputs enabled"
			write_json_fn(unit_summary_json, unit_summary)
			return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

		gtr_path = paths["gtr_pkl"]
		if not gtr_path.exists():
			unit_summary["status"] = "error"
			unit_summary["error"] = "Missing generate_gtrs artifact gtr.pkl; run reconstruct.generate_gtrs first"
			write_json_fn(unit_summary_json, unit_summary)
			return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

		try:
			load_templates_for_unit_fn(
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

			branch_selection = select_reconstruct_branch_records(
				gtr=gtr,
				branch_scope=str(phase_cfg.branch_scope),
				branch_colors=inputs.branch_colors,
			)
			phase_paths = resolve_branch_phase_output_paths_fn(
				reconstruction_out_dir=reconstruction_out_dir,
				unit_id=unit_id,
				per_unit_outputs=inputs.per_unit_outputs,
				phase_output=phase_cfg.output,
				branch_scope=phase_cfg.branch_scope,
			)
			phase_paths["output_dir"].mkdir(parents=True, exist_ok=True)

			manifest: dict[str, Any] = {
				"phase": phase_name,
				"unit_id": unit_id,
				"branch_scope": str(phase_cfg.branch_scope),
				"source_name": branch_selection.source_name,
				"output_dir": str(phase_paths["output_dir"]),
				"branch_count": len(branch_selection.records),
				"branches": [],
			}
			branches_ok = 0
			branches_error = 0
			branches_skipped = 0
			for branch_record in branch_selection.records:
				branch_paths = resolve_branch_phase_branch_output_paths_fn(
					reconstruction_out_dir=reconstruction_out_dir,
					unit_id=unit_id,
					per_unit_outputs=inputs.per_unit_outputs,
					phase_output=phase_cfg.output,
					branch_scope=phase_cfg.branch_scope,
					branch_id=branch_record.branch_id,
				)
				entry = {
					"branch_id": int(branch_record.branch_id),
					"branch_index": int(branch_record.branch_index),
					"label": branch_record.label,
					"scope": str(branch_record.scope),
					"selected_channels": [int(ch) for ch in branch_record.selected_channels],
					"color": str(branch_record.color),
					"status": "skipped",
					"error": None,
				}
				try:
					fit_payload = _fit_branch_velocity(gtr, branch_record)
					entry["velocity"] = float(fit_payload["velocity"])
					entry["offset"] = float(fit_payload["offset"])
					entry["r2"] = fit_payload.get("r2", None)
					entry["point_count"] = len(fit_payload.get("peak_times", []))
					needs_plot = bool(force_replot)
					if not needs_plot:
						if bool(phase_cfg.output.write_png) and (not branch_paths["png_path"].exists()):
							needs_plot = True
						if bool(phase_cfg.output.write_svg) and (not branch_paths["svg_path"].exists()):
							needs_plot = True
					if needs_plot:
						write_unit_branch_velocity_plot_fn(
							output_png=branch_paths["png_path"],
							output_svg=branch_paths["svg_path"],
							branch_record=branch_record,
							fit_payload=fit_payload,
							display_config=phase_cfg.display,
							output_config=phase_cfg.output,
							unit_id=unit_id,
						)
					if bool(phase_cfg.output.write_png) and branch_paths["png_path"].exists():
						entry["png_path"] = str(branch_paths["png_path"])
					if bool(phase_cfg.output.write_svg) and branch_paths["svg_path"].exists():
						entry["svg_path"] = str(branch_paths["svg_path"])
					if "png_path" in entry or "svg_path" in entry:
						entry["status"] = "ok"
						branches_ok += 1
					else:
						entry["status"] = "skipped"
						entry["error"] = "No branch velocity artifacts were written"
						branches_skipped += 1
				except Exception as exc:
					entry["status"] = "error"
					entry["error"] = str(exc)
					branches_error += 1
					active_logger.exception("Failed branch velocity plot for unit %s branch %s", unit_id, branch_record.branch_id)
				manifest["branches"].append(entry)

			manifest["branches_ok"] = int(branches_ok)
			manifest["branches_error"] = int(branches_error)
			manifest["branches_skipped"] = int(branches_skipped)
			write_json_fn(phase_paths["manifest_json"], manifest)
			unit_summary["outputs"]["branch_velocities_manifest_json"] = str(phase_paths["manifest_json"])
			unit_summary["outputs"]["branch_velocities_dir"] = str(phase_paths["output_dir"])
			if branches_ok <= 0:
				unit_summary["status"] = "error"
				if len(branch_selection.records) == 0:
					unit_summary["error"] = f"No {phase_cfg.branch_scope} branches available for reconstruct.plot_branch_velocities"
				else:
					unit_summary["error"] = "No branch velocity artifacts were written"
			else:
				unit_summary["error"] = None
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			active_logger.exception("Failed reconstruct.plot_branch_velocities for unit %s", unit_id)

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


__all__ = ["run_plot_branch_velocities_phase", "write_unit_branch_velocity_plot"]
