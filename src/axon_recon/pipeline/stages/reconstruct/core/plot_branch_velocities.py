from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
import pickle
from typing import Any, Callable

from .branch_styles import ReconstructBranchRecord
from .branch_styles import format_branch_short_label
from .branch_styles import select_reconstruct_branch_records
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


def _normalize_velocity_plot_items(
	*,
	branch_records: Any | None = None,
	fit_payloads: Any | None = None,
	branch_record: ReconstructBranchRecord | None = None,
	fit_payload: dict[str, Any] | None = None,
) -> tuple[tuple[ReconstructBranchRecord, dict[str, Any]], ...]:
	if branch_records is None:
		if branch_record is None or fit_payload is None:
			return ()
		return ((branch_record, dict(fit_payload)),)
	if isinstance(branch_records, ReconstructBranchRecord):
		if fit_payloads is None:
			return ()
		return ((branch_records, dict(fit_payloads)),)
	branch_items = tuple(item for item in branch_records if isinstance(item, ReconstructBranchRecord))
	fit_items = tuple(dict(item) for item in (fit_payloads or ()))
	if len(branch_items) != len(fit_items):
		raise ValueError("branch_records and fit_payloads must have the same length")
	return tuple(zip(branch_items, fit_items, strict=False))


def _format_branch_velocity_label(branch_record: ReconstructBranchRecord, fit_payload: dict[str, Any]) -> str:
	velocity = float(fit_payload["velocity"])
	r2_raw = fit_payload.get("r2", None)
	parts = [f"{format_branch_short_label(branch_record)}: {velocity:.1f} mm/s"]
	if r2_raw is not None:
		r2 = float(r2_raw)
		parts.append(f"r^2={r2:.2f}")
	return ", ".join(parts)


def _resolve_branch_velocity_fontsizes(display_config: Any) -> dict[str, float]:
	return {
		"title": float(max(1.0, float(getattr(display_config, "title_fontsize", 12.0) or 12.0))),
		"axis_label": float(max(1.0, float(getattr(display_config, "axis_label_fontsize", 10.0) or 10.0))),
		"tick_label": float(max(1.0, float(getattr(display_config, "tick_label_fontsize", 10.0) or 10.0))),
		"legend": float(max(1.0, float(getattr(display_config, "legend_fontsize", 8.0) or 8.0))),
	}


def _remove_legacy_branch_velocity_artifacts(*, output_dir: Path) -> None:
	for pattern in ("branch_*.png", "branch_*.svg"):
		for legacy_path in output_dir.glob(pattern):
			try:
				legacy_path.unlink()
			except FileNotFoundError:
				continue


def prepare_branch_velocity_plot_data(
	*,
	gtr: Any,
	branch_scope: str,
	branch_colors: Any,
	logger: logging.Logger | None = None,
	unit_id: Any | None = None,
) -> dict[str, Any]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.plot_branch_velocities")
	branch_selection = select_reconstruct_branch_records(
		gtr=gtr,
		branch_scope=str(branch_scope),
		branch_colors=branch_colors,
	)
	manifest_entries: list[dict[str, Any]] = []
	valid_branch_records: list[ReconstructBranchRecord] = []
	valid_fit_payloads: list[dict[str, Any]] = []
	branches_error = 0
	for branch_record in branch_selection.records:
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
			entry["status"] = "pending"
			valid_branch_records.append(branch_record)
			valid_fit_payloads.append(dict(fit_payload))
		except Exception as exc:
			entry["status"] = "error"
			entry["error"] = str(exc)
			branches_error += 1
			active_logger.exception(
				"Failed branch velocity fit for unit %s branch %s",
				unit_id,
				branch_record.branch_id,
			)
		manifest_entries.append(entry)
	return {
		"branch_selection": branch_selection,
		"manifest_entries": manifest_entries,
		"valid_branch_records": tuple(valid_branch_records),
		"valid_fit_payloads": tuple(valid_fit_payloads),
		"branches_error": int(branches_error),
	}


def write_unit_branch_velocity_plot(
	*,
	output_png: Path,
	output_svg: Path,
	branch_records: Any | None = None,
	fit_payloads: Any | None = None,
	branch_record: ReconstructBranchRecord | None = None,
	fit_payload: dict[str, Any] | None = None,
	display_config: Any,
	output_config: Any,
	unit_id: Any,
	fig: Any | None = None,
	ax: Any | None = None,
	close_figure: bool = True,
	manage_layout: bool = True,
	show_legend: bool | None = None,
	reserve_legend_space: bool = True,
) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	from axon_velocity.plotting import plot_velocity  # type: ignore[import-not-found]

	plot_items = _normalize_velocity_plot_items(
		branch_records=branch_records,
		fit_payloads=fit_payloads,
		branch_record=branch_record,
		fit_payload=fit_payload,
	)
	if len(plot_items) <= 0:
		raise ValueError("At least one branch velocity plot item is required")

	figsize = tuple(getattr(display_config, "figsize", (6.0, 4.0)) or (6.0, 4.0))
	dpi = float(max(72.0, float(getattr(output_config, "dpi", 300.0) or 300.0)))
	if (fig is None) != (ax is None):
		raise ValueError("write_unit_branch_velocity_plot requires both fig and ax when reusing an existing host")
	if fig is None or ax is None:
		fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
	try:
		ax.clear()
		fig.patch.set_facecolor("black")
		ax.set_facecolor("black")
		fonts = _resolve_branch_velocity_fontsizes(display_config)
		for item, item_fit_payload in plot_items:
			line_count_before = len(ax.lines)
			plot_velocity(
				peak_times=item_fit_payload["peak_times"],
				distances=item_fit_payload["distances"],
				velocity=float(item_fit_payload["velocity"]),
				offset=float(item_fit_payload["offset"]),
				color=str(item.color),
				r2=item_fit_payload.get("r2", None),
				ax=ax,
			)
			new_lines = list(ax.lines[line_count_before:])
			if len(new_lines) >= 1:
				new_lines[0].set_label("_nolegend_")
			if len(new_lines) >= 2:
				new_lines[-1].set_label(_format_branch_velocity_label(item, item_fit_payload))

		units_only_axis_labels = bool(getattr(display_config, "units_only_axis_labels", True))
		x_label = "ms" if units_only_axis_labels else "Peak time (ms)"
		y_label = "$\\mu$m" if units_only_axis_labels else "Distance ($\\mu$m)"
		ax.set_xlabel(x_label, color="white", fontsize=fonts["axis_label"])
		ax.set_ylabel(y_label, color="white", fontsize=fonts["axis_label"])
		for spine in ax.spines.values():
			spine.set_color("white")
		ax.tick_params(colors="white", labelsize=fonts["tick_label"])
		if bool(getattr(display_config, "show_title", True)):
			ax.set_title(f"Unit {unit_id} branch velocities", color="white", fontsize=fonts["title"])

		legend = ax.get_legend()
		if legend is not None:
			legend.remove()
		resolved_show_legend = bool(getattr(display_config, "show_legend", True)) if show_legend is None else bool(show_legend)
		if resolved_show_legend:
			legend = ax.legend(
				fontsize=fonts["legend"],
				framealpha=0.9,
				facecolor="black",
				edgecolor="white",
				loc="center left",
				bbox_to_anchor=(1.02, 0.5),
				borderaxespad=0.0,
			)
			for text in legend.get_texts():
				text.set_color("white")
			for legend_line in legend.get_lines():
				legend_line.set_linewidth(max(1.5, legend_line.get_linewidth()))
		if resolved_show_legend and bool(reserve_legend_space) and bool(manage_layout):
			fig.subplots_adjust(right=0.72)

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
		if bool(close_figure):
			plt.close(fig)


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
	replot = bool(inputs.force_restart) or bool(inputs.replot)
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

		if not (bool(phase_cfg.output.write_png) or bool(phase_cfg.output.write_svg)):
			unit_summary["status"] = "error"
			unit_summary["error"] = "No branch velocity outputs enabled"
			write_json_fn(unit_summary_json, unit_summary)
			return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

		gtr_path = paths["gtr_pkl"]
		if not gtr_path.exists():
			unit_summary["status"] = "error"
			unit_summary["error"] = "Missing axon_velocity_gtrs artifact gtr.pkl; run reconstruct.axon_velocity_gtrs first"
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

			prepared_plot_data = prepare_branch_velocity_plot_data(
				gtr=gtr,
				branch_scope=str(phase_cfg.branch_scope),
				branch_colors=inputs.branch_colors,
				logger=active_logger,
				unit_id=unit_id,
			)
			branch_selection = prepared_plot_data["branch_selection"]
			phase_paths = resolve_branch_phase_output_paths_fn(
				reconstruction_out_dir=reconstruction_out_dir,
				unit_id=unit_id,
				per_unit_outputs=inputs.per_unit_outputs,
				phase_output=phase_cfg.output,
				branch_scope=phase_cfg.branch_scope,
			)
			phase_paths["output_dir"].mkdir(parents=True, exist_ok=True)
			_remove_legacy_branch_velocity_artifacts(output_dir=phase_paths["output_dir"])
			figure_png_path = phase_paths["output_dir"] / "branch_velocities.png"
			figure_svg_path = phase_paths["output_dir"] / "branch_velocities.svg"

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
			branches_error = int(prepared_plot_data["branches_error"])
			branches_skipped = 0
			valid_branch_records = list(prepared_plot_data["valid_branch_records"])
			valid_fit_payloads = list(prepared_plot_data["valid_fit_payloads"])
			manifest["branches"].extend(list(prepared_plot_data["manifest_entries"]))

			if len(valid_branch_records) > 0:
				needs_plot = bool(replot)
				if not needs_plot:
					if bool(phase_cfg.output.write_png) and (not figure_png_path.exists()):
						needs_plot = True
					if bool(phase_cfg.output.write_svg) and (not figure_svg_path.exists()):
						needs_plot = True
				if needs_plot:
					write_unit_branch_velocity_plot_fn(
						output_png=figure_png_path,
						output_svg=figure_svg_path,
						branch_records=tuple(valid_branch_records),
						fit_payloads=tuple(valid_fit_payloads),
						display_config=phase_cfg.display,
						output_config=phase_cfg.output,
						unit_id=unit_id,
					)

			has_figure_output = False
			if bool(phase_cfg.output.write_png) and figure_png_path.exists():
				manifest["png_path"] = str(figure_png_path)
				has_figure_output = True
			if bool(phase_cfg.output.write_svg) and figure_svg_path.exists():
				manifest["svg_path"] = str(figure_svg_path)
				has_figure_output = True

			for entry in manifest["branches"]:
				if str(entry.get("status", "")).strip().lower() != "pending":
					continue
				if has_figure_output:
					if "png_path" in manifest:
						entry["png_path"] = str(manifest["png_path"])
					if "svg_path" in manifest:
						entry["svg_path"] = str(manifest["svg_path"])
					entry["status"] = "ok"
					branches_ok += 1
				else:
					entry["status"] = "skipped"
					entry["error"] = "No branch velocity figure was written"
					branches_skipped += 1

			manifest["branches_ok"] = int(branches_ok)
			manifest["branches_error"] = int(branches_error)
			manifest["branches_skipped"] = int(branches_skipped)
			write_json_fn(phase_paths["manifest_json"], manifest)
			unit_summary["outputs"]["branch_velocities_manifest_json"] = str(phase_paths["manifest_json"])
			unit_summary["outputs"]["branch_velocities_dir"] = str(phase_paths["output_dir"])
			if "png_path" in manifest:
				unit_summary["outputs"]["branch_velocities_png"] = str(manifest["png_path"])
			if "svg_path" in manifest:
				unit_summary["outputs"]["branch_velocities_svg"] = str(manifest["svg_path"])
			if branches_ok <= 0:
				unit_summary["status"] = "error"
				if len(branch_selection.records) == 0:
					unit_summary["error"] = f"No {phase_cfg.branch_scope} branches available for reconstruct.plot_branch_velocities"
				else:
					unit_summary["error"] = "No branch velocity figure was written"
			else:
				unit_summary["status"] = "ok"
				unit_summary["error"] = None
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			active_logger.exception("Failed reconstruct.plot_branch_velocities for unit %s", unit_id)

		write_json_fn(unit_summary_json, unit_summary)
		return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

	_phase_budget = current_phase_budget("reconstruct", "plot_branch_velocities")
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


__all__ = [
	"prepare_branch_velocity_plot_data",
	"run_plot_branch_velocities_phase",
	"write_unit_branch_velocity_plot",
]
