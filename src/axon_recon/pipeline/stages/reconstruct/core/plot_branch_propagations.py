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


def write_unit_branch_propagation_plot(
	*,
	output_png: Path,
	output_svg: Path,
	template_ch_by_t: Any,
	locs_xy: Any,
	branch_record: ReconstructBranchRecord,
	display_config: Any,
	output_config: Any,
	unit_id: Any,
) -> dict[str, str]:
	import matplotlib
	import numpy as np  # type: ignore[import-not-found]

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	from axon_velocity.plotting import plot_template_propagation  # type: ignore[import-not-found]

	locs = np.asarray(locs_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Expected locs_xy to be [N,2+], got shape={locs.shape}")
	locs = locs[:, :2]
	tpl = _normalize_template_channels_by_time(template_ch_by_t, n_channels=int(locs.shape[0]))
	selected_channels = [int(ch) for ch in branch_record.selected_channels if 0 <= int(ch) < int(locs.shape[0])]
	if len(selected_channels) < 2:
		raise ValueError(f"Branch {branch_record.branch_id} has fewer than two channels after bounds filtering")

	figsize = tuple(getattr(display_config, "figsize", (6.0, 4.0)) or (6.0, 4.0))
	fig, ax = plt.subplots(figsize=figsize, dpi=float(max(72.0, float(getattr(output_config, "dpi", 300.0) or 300.0))))
	plot_template_propagation(
		tpl,
		locs,
		selected_channels,
		sort_templates=bool(getattr(display_config, "sort_templates", False)),
		color="k",
		color_marker=str(branch_record.color),
		ax=ax,
	)
	if bool(getattr(display_config, "invert_y_axis", True)):
		ax.invert_yaxis()
	if bool(getattr(display_config, "show_title", True)):
		ax.set_title(f"Unit {unit_id} branch {branch_record.label} propagation")
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


def run_plot_branch_propagations_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_ids: list[Any],
	load_templates_for_unit_fn: Callable[..., Any],
	write_unit_branch_propagation_plot_fn: Callable[..., dict[str, str]],
	read_json_fn: Callable[[Path], Any],
	write_json_fn: Callable[[Path, Any], None],
	resolve_unit_output_paths_fn: Callable[..., dict[str, Path]],
	resolve_branch_phase_output_paths_fn: Callable[..., dict[str, Path]],
	resolve_branch_phase_branch_output_paths_fn: Callable[..., dict[str, Path]],
	logger: logging.Logger | None = None,
) -> list[UnitReconstructionResult]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.plot_branch_propagations")
	force_replot = bool(inputs.force_restart) or bool(inputs.force_replot)
	phase_cfg = inputs.phases.plot_branch_propagations
	phase_name = "plot_branch_propagations"

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
			unit_summary["error"] = "No branch propagation outputs enabled"
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
					needs_plot = bool(force_replot)
					if not needs_plot:
						if bool(phase_cfg.output.write_png) and (not branch_paths["png_path"].exists()):
							needs_plot = True
						if bool(phase_cfg.output.write_svg) and (not branch_paths["svg_path"].exists()):
							needs_plot = True
					if needs_plot:
						write_unit_branch_propagation_plot_fn(
							output_png=branch_paths["png_path"],
							output_svg=branch_paths["svg_path"],
							template_ch_by_t=gtr_template_ch_by_t,
							locs_xy=gtr_locs_xy,
							branch_record=branch_record,
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
						entry["error"] = "No branch propagation artifacts were written"
						branches_skipped += 1
				except Exception as exc:
					entry["status"] = "error"
					entry["error"] = str(exc)
					branches_error += 1
					active_logger.exception("Failed branch propagation plot for unit %s branch %s", unit_id, branch_record.branch_id)
				manifest["branches"].append(entry)

			manifest["branches_ok"] = int(branches_ok)
			manifest["branches_error"] = int(branches_error)
			manifest["branches_skipped"] = int(branches_skipped)
			write_json_fn(phase_paths["manifest_json"], manifest)
			unit_summary["outputs"]["branch_propagations_manifest_json"] = str(phase_paths["manifest_json"])
			unit_summary["outputs"]["branch_propagations_dir"] = str(phase_paths["output_dir"])
			if branches_ok <= 0:
				unit_summary["status"] = "error"
				if len(branch_selection.records) == 0:
					unit_summary["error"] = (
						f"No {phase_cfg.branch_scope} branches available for reconstruct.plot_branch_propagations"
					)
				else:
					unit_summary["error"] = "No branch propagation artifacts were written"
			else:
				unit_summary["error"] = None
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			active_logger.exception("Failed reconstruct.plot_branch_propagations for unit %s", unit_id)

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


__all__ = ["run_plot_branch_propagations_phase", "write_unit_branch_propagation_plot"]
