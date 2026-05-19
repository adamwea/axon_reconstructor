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


def _normalize_branch_records(
	*,
	branch_records: Any | None = None,
	branch_record: ReconstructBranchRecord | None = None,
) -> tuple[ReconstructBranchRecord, ...]:
	if branch_records is None:
		if branch_record is None:
			return ()
		return (branch_record,)
	if isinstance(branch_records, ReconstructBranchRecord):
		return (branch_records,)
	return tuple(item for item in branch_records if isinstance(item, ReconstructBranchRecord))


def _filter_selected_channels(*, branch_record: ReconstructBranchRecord, n_channels: int) -> tuple[int, ...]:
	return tuple(
		int(channel)
		for channel in branch_record.selected_channels
		if 0 <= int(channel) < int(n_channels)
	)


def _remove_legacy_branch_artifacts(*, output_dir: Path) -> None:
	for pattern in ("branch_*.png", "branch_*.svg"):
		for legacy_path in output_dir.glob(pattern):
			try:
				legacy_path.unlink()
			except FileNotFoundError:
				continue


def resolve_branch_propagation_layout(*, branch_count: int, display_config: Any) -> dict[str, float]:
	resolved_branch_count = max(1, int(branch_count))
	panel_width, panel_height = tuple(getattr(display_config, "figsize", (2.75, 6.0)) or (2.75, 6.0))
	panel_width = float(max(0.5, panel_width))
	panel_height = float(max(1.0, panel_height))
	total_width_raw = getattr(display_config, "total_width", None)
	try:
		total_width = None if total_width_raw is None else float(total_width_raw)
	except Exception:
		total_width = None
	if total_width is not None and total_width > 0.0:
		effective_total_width = float(max(0.5 * float(resolved_branch_count), total_width))
		effective_panel_width = float(max(0.5, effective_total_width / float(resolved_branch_count)))
	else:
		effective_panel_width = panel_width
		effective_total_width = float(effective_panel_width * float(resolved_branch_count))
	return {
		"panel_width": effective_panel_width,
		"panel_height": panel_height,
		"total_width": effective_total_width,
	}


def write_unit_branch_propagation_plot(
	*,
	output_png: Path,
	output_svg: Path,
	template_ch_by_t: Any,
	locs_xy: Any,
	branch_records: Any | None = None,
	branch_record: ReconstructBranchRecord | None = None,
	display_config: Any,
	output_config: Any,
	unit_id: Any,
	fig: Any | None = None,
	axes: Any | None = None,
	close_figure: bool = True,
	manage_layout: bool = True,
	show_figure_title: bool | None = None,
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
	branch_items = _normalize_branch_records(branch_records=branch_records, branch_record=branch_record)
	if len(branch_items) <= 0:
		raise ValueError("At least one branch propagation record is required")

	layout = resolve_branch_propagation_layout(branch_count=len(branch_items), display_config=display_config)
	panel_height = float(layout["panel_height"])
	total_width = float(layout["total_width"])
	dpi = float(max(72.0, float(getattr(output_config, "dpi", 300.0) or 300.0)))
	if (fig is None) != (axes is None):
		raise ValueError("write_unit_branch_propagation_plot requires both fig and axes when reusing an existing host")
	if fig is None or axes is None:
		fig, axes = plt.subplots(
			1,
			len(branch_items),
			figsize=(total_width, panel_height),
			dpi=dpi,
			squeeze=False,
		)
	try:
		fig.patch.set_facecolor("black")
		try:
			axes_list = list(axes.ravel())
		except Exception:
			if isinstance(axes, (list, tuple)):
				axes_list = list(axes)
			else:
				axes_list = [axes]
		if len(axes_list) != len(branch_items):
			raise ValueError(
				f"Expected {len(branch_items)} propagation axes, received {len(axes_list)}"
			)
		show_title = bool(getattr(display_config, "show_title", True))
		if show_figure_title is None:
			show_figure_title = bool(show_title)
		for ax, item in zip(axes_list, branch_items):
			ax.clear()
			ax.set_facecolor("black")
			selected_channels = _filter_selected_channels(branch_record=item, n_channels=int(locs.shape[0]))
			if len(selected_channels) < 2:
				raise ValueError(f"Branch {item.branch_id} has fewer than two channels after bounds filtering")
			plot_template_propagation(
				tpl,
				locs,
				[int(channel) for channel in selected_channels],
				sort_templates=bool(getattr(display_config, "sort_templates", False)),
				color=str(item.color),
				color_marker=str(item.color),
				ax=ax,
			)
			if bool(getattr(display_config, "invert_y_axis", True)):
				ax.invert_yaxis()
			if show_title:
				ax.set_title(format_branch_short_label(item), color=str(item.color))
		if bool(show_figure_title):
			fig.suptitle(f"Unit {unit_id} branch propagations", color="white")
		if bool(manage_layout):
			fig.subplots_adjust(
				wspace=0.08,
				top=(0.84 if bool(show_figure_title) else 0.97),
				bottom=0.03,
				left=0.02,
				right=0.98,
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
		if bool(close_figure):
			plt.close(fig)


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
	replot = bool(inputs.force_restart) or bool(inputs.replot)
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
			unit_summary["error"] = "No branch propagation outputs enabled"
			write_json_fn(unit_summary_json, unit_summary)
			return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

		gtr_path = paths["gtr_pkl"]
		if not gtr_path.exists():
			unit_summary["status"] = "error"
			unit_summary["error"] = "Missing axon_velocity_gtrs artifact gtr.pkl; run reconstruct.axon_velocity_gtrs first"
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
			_remove_legacy_branch_artifacts(output_dir=phase_paths["output_dir"])
			figure_png_path = phase_paths["output_dir"] / "branch_propagations.png"
			figure_svg_path = phase_paths["output_dir"] / "branch_propagations.svg"

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
			valid_branch_records: list[ReconstructBranchRecord] = []
			for branch_record in branch_selection.records:
				selected_channels = _filter_selected_channels(
					branch_record=branch_record,
					n_channels=int(len(gtr_locs_xy)),
				)
				entry = {
					"branch_id": int(branch_record.branch_id),
					"branch_index": int(branch_record.branch_index),
					"label": branch_record.label,
					"scope": str(branch_record.scope),
					"selected_channels": [int(ch) for ch in selected_channels],
					"color": str(branch_record.color),
					"status": "pending",
					"error": None,
				}
				if len(selected_channels) < 2:
					entry["status"] = "error"
					entry["error"] = "Branch has fewer than two channels after bounds filtering"
					branches_error += 1
				else:
					entry["column_index"] = len(valid_branch_records)
					valid_branch_records.append(
						ReconstructBranchRecord(
							branch_id=int(branch_record.branch_id),
							branch_index=int(branch_record.branch_index),
							label=branch_record.label,
							selected_channels=selected_channels,
							color=str(branch_record.color),
							scope=str(branch_record.scope),
							velocity=branch_record.velocity,
							offset=branch_record.offset,
							r2=branch_record.r2,
							peak_times=tuple(branch_record.peak_times),
							distances=tuple(branch_record.distances),
						)
					)
				manifest["branches"].append(entry)

			if len(valid_branch_records) > 0:
				needs_plot = bool(replot)
				if not needs_plot:
					if bool(phase_cfg.output.write_png) and (not figure_png_path.exists()):
						needs_plot = True
					if bool(phase_cfg.output.write_svg) and (not figure_svg_path.exists()):
						needs_plot = True
				if needs_plot:
					write_unit_branch_propagation_plot_fn(
						output_png=figure_png_path,
						output_svg=figure_svg_path,
						template_ch_by_t=gtr_template_ch_by_t,
						locs_xy=gtr_locs_xy,
						branch_records=tuple(valid_branch_records),
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
					entry["error"] = "No branch propagation figure was written"
					branches_skipped += 1

			manifest["branches_ok"] = int(branches_ok)
			manifest["branches_error"] = int(branches_error)
			manifest["branches_skipped"] = int(branches_skipped)
			write_json_fn(phase_paths["manifest_json"], manifest)
			unit_summary["outputs"]["branch_propagations_manifest_json"] = str(phase_paths["manifest_json"])
			unit_summary["outputs"]["branch_propagations_dir"] = str(phase_paths["output_dir"])
			if "png_path" in manifest:
				unit_summary["outputs"]["branch_propagations_png"] = str(manifest["png_path"])
			if "svg_path" in manifest:
				unit_summary["outputs"]["branch_propagations_svg"] = str(manifest["svg_path"])
			if branches_ok <= 0:
				unit_summary["status"] = "error"
				if len(branch_selection.records) == 0:
					unit_summary["error"] = (
						f"No {phase_cfg.branch_scope} branches available for reconstruct.plot_branch_propagations"
					)
				else:
					unit_summary["error"] = "No branch propagation figure was written"
			else:
				unit_summary["status"] = "ok"
				unit_summary["error"] = None
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			active_logger.exception("Failed reconstruct.plot_branch_propagations for unit %s", unit_id)

		write_json_fn(unit_summary_json, unit_summary)
		return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

	_phase_budget = current_phase_budget("reconstruct", "plot_branch_propagations")
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


__all__ = ["resolve_branch_propagation_layout", "run_plot_branch_propagations_phase", "write_unit_branch_propagation_plot"]
