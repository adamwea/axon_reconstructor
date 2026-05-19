from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
import pickle
from typing import Any, Callable

from axon_recon.pipeline.execution.progress import add_current_progress_total, advance_current_progress
from axon_recon.pipeline.cpu_allocation import current_phase_budget, resolve_inner_worker_count

from ..models.inputs import ReconstructionGenerateGtrsOutputsConfig, ReconstructionInputs
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


def _should_persist_gtr(inputs: ReconstructionInputs) -> bool:
	_ = inputs
	return True


def _resolve_axon_velocity_gtrs_outputs(inputs: ReconstructionInputs) -> ReconstructionGenerateGtrsOutputsConfig:
	phase_outputs = inputs.phases.axon_velocity_gtrs.outputs
	if phase_outputs != ReconstructionGenerateGtrsOutputsConfig():
		return phase_outputs
	legacy = inputs.per_unit_outputs
	return ReconstructionGenerateGtrsOutputsConfig(
		write_branches_raw_json=bool(legacy.write_branches_raw_json),
		branches_raw_relpath=str(legacy.branches_raw_relpath),
		write_branches_json=bool(legacy.write_branches_json),
		branches_relpath=str(legacy.branches_relpath),
		write_detection_filter_json=bool(legacy.write_detection_filter_json),
		detection_filter_relpath=str(legacy.detection_filter_relpath),
		write_kurtosis_filter_json=bool(legacy.write_kurtosis_filter_json),
		kurtosis_filter_relpath=str(legacy.kurtosis_filter_relpath),
		write_peak_std_filter_json=bool(legacy.write_peak_std_filter_json),
		peak_std_filter_relpath=str(legacy.peak_std_filter_relpath),
		write_delay_filter_json=bool(legacy.write_delay_filter_json),
		delay_filter_relpath=str(legacy.delay_filter_relpath),
		write_all_filters_json=bool(legacy.write_all_filters_json),
		all_filters_relpath=str(legacy.all_filters_relpath),
		write_heuristics_json=bool(legacy.write_heuristics_json),
		heuristics_relpath=str(legacy.heuristics_relpath),
		write_gtr_pkl=bool(legacy.write_gtr_pkl),
		gtr_pkl_relpath=str(legacy.gtr_pkl_relpath),
		template_source=str(legacy.template_source),
		write_gtr_json=bool(legacy.write_gtr_json),
		gtr_json_relpath=str(legacy.gtr_json_relpath),
		channel_selection_figure=legacy.channel_selection_figure,
		axon_reconstruction_figure=legacy.axon_reconstruction_figure,
	)


def _generate_outputs_ready(*, inputs: ReconstructionInputs, paths: dict[str, Path]) -> bool:
	phase_outputs = _resolve_axon_velocity_gtrs_outputs(inputs)
	checks: list[bool] = []
	if bool(phase_outputs.write_branches_raw_json):
		checks.append(paths["branches_raw_json"].exists())
	if bool(phase_outputs.write_branches_json):
		checks.append(paths["branches_json"].exists())
	if bool(phase_outputs.write_detection_filter_json):
		checks.append(paths["detection_filter_json"].exists())
	if bool(phase_outputs.write_kurtosis_filter_json):
		checks.append(paths["kurtosis_filter_json"].exists())
	if bool(phase_outputs.write_peak_std_filter_json):
		checks.append(paths["peak_std_filter_json"].exists())
	if bool(phase_outputs.write_delay_filter_json):
		checks.append(paths["delay_filter_json"].exists())
	if bool(phase_outputs.write_all_filters_json):
		checks.append(paths["all_filters_json"].exists())
	if bool(phase_outputs.write_heuristics_json):
		checks.append(paths["heuristics_json"].exists())
	if _should_persist_gtr(inputs):
		checks.append(paths["gtr_pkl"].exists())
	if bool(phase_outputs.write_gtr_json):
		checks.append(paths["gtr_json"].exists())
	if bool(phase_outputs.channel_selection_figure.write_png):
		checks.append(paths["channel_selection_figure_png"].exists())
	if bool(phase_outputs.channel_selection_figure.write_svg):
		checks.append(paths["channel_selection_figure_svg"].exists())
	if bool(phase_outputs.axon_reconstruction_figure.write_png):
		checks.append(paths["axon_reconstruction_figure_png"].exists())
	if bool(phase_outputs.axon_reconstruction_figure.write_svg):
		checks.append(paths["axon_reconstruction_figure_svg"].exists())
	return all(checks) if checks else True


def _resolve_plotting_worker_count(*, inputs: ReconstructionInputs) -> int:
	return int(max(1, int(inputs.n_jobs)))


def run_axon_velocity_gtrs_phase(
	*,
	inputs: ReconstructionInputs,
	reconstruction_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_ids: list[Any],
	import_axon_velocity_fn: Callable[..., Any],
	load_templates_for_unit_fn: Callable[..., Any],
	compute_graph_tracking_fn: Callable[..., Any],
	compute_raw_branches_payload_fn: Callable[..., Any],
	compute_branches_with_polyline_fn: Callable[..., Any],
	compute_detection_filter_payload_fn: Callable[..., Any],
	compute_kurtosis_filter_payload_fn: Callable[..., Any],
	compute_peak_std_filter_payload_fn: Callable[..., Any],
	compute_delay_filter_payload_fn: Callable[..., Any],
	compute_all_filters_payload_fn: Callable[..., Any],
	compute_heuristics_payload_fn: Callable[..., Any],
	compute_gtr_json_payload_fn: Callable[..., Any],
	write_unit_channel_selection_diagnostic_figure_fn: Callable[..., None],
	write_unit_axon_reconstruction_diagnostic_figure_fn: Callable[..., None],
	read_json_fn: Callable[[Path], Any],
	write_json_fn: Callable[[Path, Any], None],
	resolve_unit_output_paths_fn: Callable[..., dict[str, Path]],
	is_empty_signal_selection_error_fn: Callable[[Exception], bool],
	is_expected_reconstruct_unit_failure_fn: Callable[[Exception], bool],
	normalize_template_for_tracking_fn: Callable[[Any, Any], Any],
	logger: logging.Logger | None = None,
	progress_total_already_added: bool = False,
) -> list[UnitReconstructionResult]:
	active_logger = logger or logging.getLogger("axon_recon.reconstruct.axon_velocity_gtrs")
	phase_cfg = inputs.phases.axon_velocity_gtrs
	phase_outputs = _resolve_axon_velocity_gtrs_outputs(inputs)
	_phase_budget = current_phase_budget("reconstruct", "axon_velocity_gtrs")
	worker_count = resolve_inner_worker_count(
		nested_shape=str(getattr(_phase_budget, "nested_shape", "unit_workers") or "unit_workers"),
		phase_cpus_per_task=getattr(_phase_budget, "cpus_per_task", None) if _phase_budget else None,
		yaml_n_jobs_override=int(inputs.n_jobs) if getattr(inputs, "n_jobs", None) is not None else None,
		work_item_count=int(len(unit_ids)) if unit_ids is not None else None,
	)
	plotting_worker_count = _resolve_plotting_worker_count(inputs=inputs)
	if not bool(phase_cfg.axon_velocity.enabled):
		raise RuntimeError("reconstruct.axon_velocity_gtrs currently requires phases.axon_velocity_gtrs.axon_velocity.enabled=true")
	active_logger.info(
		"reconstruct.axon_velocity_gtrs start: reconstruction_out_dir=%s units=%d force_restart=%s n_jobs=%d plotting_workers=%d template_source=%s gtr_relpath=%s",
		str(reconstruction_out_dir),
		len(unit_ids),
		bool(inputs.force_restart),
		worker_count,
		plotting_worker_count,
		str(phase_outputs.template_source),
		str(phase_outputs.gtr_pkl_relpath),
	)
	if not bool(phase_outputs.write_gtr_pkl):
		active_logger.info(
			"reconstruct.axon_velocity_gtrs forcing gtr.pkl persistence for phase contract even though configured write_gtr_pkl=false"
		)
	if not bool(progress_total_already_added):
		add_current_progress_total(len(unit_ids))

	av = import_axon_velocity_fn(repo_root=inputs.axon_velocity_repo_root)

	def _load_cached_result(unit_id: Any, paths: dict[str, Path]) -> UnitReconstructionResult | None:
		unit_summary_json = paths["unit_summary_json"]
		if not unit_summary_json.exists():
			return None
		try:
			payload = read_json_fn(unit_summary_json)
		except Exception:
			return None
		cached = _unit_result_from_summary(unit_id=unit_id, payload=payload)
		status = str(cached.status).strip().lower()
		if status != "ok":
			active_logger.info(
				"reconstruct.axon_velocity_gtrs unit %s reusing cached failed summary: %s",
				unit_id,
				str(unit_summary_json),
			)
			return cached
		if _generate_outputs_ready(inputs=inputs, paths=paths):
			active_logger.info(
				"reconstruct.axon_velocity_gtrs unit %s using cached outputs: gtr_pkl=%s summary=%s",
				unit_id,
				str(paths["gtr_pkl"]),
				str(unit_summary_json),
			)
			return cached
		return None

	def _process_unit(unit_id: Any) -> UnitReconstructionResult:
		paths = resolve_unit_output_paths_fn(
			reconstruction_out_dir=reconstruction_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		paths["unit_dir"].mkdir(parents=True, exist_ok=True)
		active_logger.info(
			"reconstruct.axon_velocity_gtrs unit %s start: unit_dir=%s gtr_pkl=%s",
			unit_id,
			str(paths["unit_dir"]),
			str(paths["gtr_pkl"]),
		)

		if not bool(inputs.force_restart):
			cached = _load_cached_result(unit_id=unit_id, paths=paths)
			if cached is not None:
				return cached

		unit_summary: dict[str, Any] = {
			"unit_id": unit_id,
			"status": "ok",
			"error": None,
			"outputs": {},
		}
		try:
			_, _, gtr_template_ch_by_t, gtr_locs_xy, fs_hz, selected_template_source = load_templates_for_unit_fn(
				unit_id=unit_id,
				merged_units_dir=merged_units_dir,
				full_channels_templates_dir=full_channels_templates_dir,
				template_source=str(phase_outputs.template_source),
				use_full_channels_templates=inputs.use_full_channels_templates,
				require_full_channels_templates=inputs.require_full_channels_templates,
				probe_geometry=inputs.probe_geometry,
			)
			unit_summary["selected_template_source"] = selected_template_source
			unit_summary["graph_tracking_source"] = selected_template_source
			try:
				from axon_recon.pipeline.shared.grid_sorting import compute_template_grid_sort_metrics

				unit_summary["grid_sort_metrics"] = compute_template_grid_sort_metrics(
					template_c_by_t=gtr_template_ch_by_t,
					locations_xy=gtr_locs_xy,
					sampling_rate_hz=fs_hz,
					probe_pitch_um=(
						None
						if inputs.probe_geometry is None
						else getattr(inputs.probe_geometry, "pitch_um", None)
					),
				)
			except Exception:
				unit_summary["grid_sort_metrics"] = {}

			try:
				gtr = compute_graph_tracking_fn(
					av=av,
					template_ch_by_t=normalize_template_for_tracking_fn(gtr_template_ch_by_t, gtr_locs_xy),
					locs_xy=gtr_locs_xy,
					sampling_frequency_hz=float(fs_hz),
					params=dict(inputs.axon_velocity_params),
				)
			except Exception as exc:
				if (
					is_empty_signal_selection_error_fn(exc)
					and str(selected_template_source) not in {"merged_contributing", "merged_per_unit_output"}
				):
					raise RuntimeError(
						"Graph tracking failed for requested template source "
						f"{selected_template_source} (unit={unit_id}): {exc}. "
						"Fallback to merged template source is disabled."
					) from exc
				raise

			if bool(phase_outputs.write_branches_raw_json):
				payload = compute_raw_branches_payload_fn(unit_id=unit_id, gtr=gtr)
				write_json_fn(paths["branches_raw_json"], payload)
				unit_summary["outputs"]["branches_raw_json"] = str(paths["branches_raw_json"])

			if bool(phase_outputs.write_branches_json):
				payload = compute_branches_with_polyline_fn(unit_id=unit_id, gtr=gtr, locs_xy=gtr_locs_xy)
				write_json_fn(paths["branches_json"], payload)
				unit_summary["outputs"]["branches_json"] = str(paths["branches_json"])

			if bool(phase_outputs.write_detection_filter_json):
				payload = compute_detection_filter_payload_fn(unit_id=unit_id, gtr=gtr, locs_xy=gtr_locs_xy)
				write_json_fn(paths["detection_filter_json"], payload)
				unit_summary["outputs"]["detection_filter_json"] = str(paths["detection_filter_json"])

			if bool(phase_outputs.write_kurtosis_filter_json):
				payload = compute_kurtosis_filter_payload_fn(unit_id=unit_id, gtr=gtr, locs_xy=gtr_locs_xy)
				write_json_fn(paths["kurtosis_filter_json"], payload)
				unit_summary["outputs"]["kurtosis_filter_json"] = str(paths["kurtosis_filter_json"])

			if bool(phase_outputs.write_peak_std_filter_json):
				payload = compute_peak_std_filter_payload_fn(unit_id=unit_id, gtr=gtr, locs_xy=gtr_locs_xy)
				write_json_fn(paths["peak_std_filter_json"], payload)
				unit_summary["outputs"]["peak_std_filter_json"] = str(paths["peak_std_filter_json"])

			if bool(phase_outputs.write_delay_filter_json):
				payload = compute_delay_filter_payload_fn(unit_id=unit_id, gtr=gtr, locs_xy=gtr_locs_xy)
				write_json_fn(paths["delay_filter_json"], payload)
				unit_summary["outputs"]["delay_filter_json"] = str(paths["delay_filter_json"])

			if bool(phase_outputs.write_all_filters_json):
				payload = compute_all_filters_payload_fn(unit_id=unit_id, gtr=gtr, locs_xy=gtr_locs_xy)
				write_json_fn(paths["all_filters_json"], payload)
				unit_summary["outputs"]["all_filters_json"] = str(paths["all_filters_json"])

			if bool(phase_outputs.write_heuristics_json):
				payload = compute_heuristics_payload_fn(unit_id=unit_id, gtr=gtr, locs_xy=gtr_locs_xy)
				write_json_fn(paths["heuristics_json"], payload)
				unit_summary["outputs"]["heuristics_json"] = str(paths["heuristics_json"])

			if _should_persist_gtr(inputs):
				paths["gtr_pkl"].parent.mkdir(parents=True, exist_ok=True)
				with open(paths["gtr_pkl"], "wb") as handle:
					pickle.dump(gtr, handle)
				unit_summary["outputs"]["gtr_pkl"] = str(paths["gtr_pkl"])
				active_logger.info(
					"reconstruct.axon_velocity_gtrs unit %s wrote gtr_pkl=%s",
					unit_id,
					str(paths["gtr_pkl"]),
				)

			if bool(phase_outputs.write_gtr_json):
				payload = compute_gtr_json_payload_fn(unit_id=unit_id, gtr=gtr, locs_xy=gtr_locs_xy)
				write_json_fn(paths["gtr_json"], payload)
				unit_summary["outputs"]["gtr_json"] = str(paths["gtr_json"])

			channel_selection_requested = bool(phase_outputs.channel_selection_figure.write_png) or bool(
				phase_outputs.channel_selection_figure.write_svg
			)
			axon_reconstruction_requested = bool(phase_outputs.axon_reconstruction_figure.write_png) or bool(
				phase_outputs.axon_reconstruction_figure.write_svg
			)
			if channel_selection_requested or axon_reconstruction_requested:
				if channel_selection_requested:
					write_unit_channel_selection_diagnostic_figure_fn(
						av=av,
						output_png=(paths["channel_selection_figure_png"] if bool(phase_outputs.channel_selection_figure.write_png) else None),
						output_svg=(paths["channel_selection_figure_svg"] if bool(phase_outputs.channel_selection_figure.write_svg) else None),
						template_ch_by_t=gtr_template_ch_by_t,
						locs_xy=gtr_locs_xy,
						gtr=gtr,
						dpi=float(phase_outputs.channel_selection_figure.dpi),
						invert_y_axis=bool(phase_outputs.channel_selection_figure.invert_y_axis),
					)
					if bool(phase_outputs.channel_selection_figure.write_png):
						unit_summary["outputs"]["channel_selection_figure_png"] = str(paths["channel_selection_figure_png"])
					if bool(phase_outputs.channel_selection_figure.write_svg):
						unit_summary["outputs"]["channel_selection_figure_svg"] = str(paths["channel_selection_figure_svg"])

					if axon_reconstruction_requested:
						write_unit_axon_reconstruction_diagnostic_figure_fn(
							output_png=(paths["axon_reconstruction_figure_png"] if bool(phase_outputs.axon_reconstruction_figure.write_png) else None),
							output_svg=(paths["axon_reconstruction_figure_svg"] if bool(phase_outputs.axon_reconstruction_figure.write_svg) else None),
							gtr=gtr,
							dpi=float(phase_outputs.axon_reconstruction_figure.dpi),
							invert_y_axis=bool(phase_outputs.axon_reconstruction_figure.invert_y_axis),
						)
						if bool(phase_outputs.axon_reconstruction_figure.write_png):
							unit_summary["outputs"]["axon_reconstruction_figure_png"] = str(paths["axon_reconstruction_figure_png"])
						if bool(phase_outputs.axon_reconstruction_figure.write_svg):
							unit_summary["outputs"]["axon_reconstruction_figure_svg"] = str(paths["axon_reconstruction_figure_svg"])

			active_logger.info(
				"reconstruct.axon_velocity_gtrs unit %s complete: status=%s outputs=%s",
				unit_id,
				str(unit_summary.get("status", "ok")),
				sorted(str(key) for key in dict(unit_summary.get("outputs", {})).keys()),
			)
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			if is_expected_reconstruct_unit_failure_fn(exc):
				active_logger.warning("Reconstruct unit %s failed: %s", unit_id, exc)
			else:
				active_logger.exception("Failed reconstruct for unit %s", unit_id)

		write_json_fn(paths["unit_summary_json"], unit_summary)
		active_logger.info(
			"reconstruct.axon_velocity_gtrs unit %s wrote summary=%s status=%s",
			unit_id,
			str(paths["unit_summary_json"]),
			str(unit_summary.get("status", "ok")),
		)
		return _unit_result_from_summary(unit_id=unit_id, payload=unit_summary)

	unit_results: list[UnitReconstructionResult] = []
	if worker_count <= 1 or len(unit_ids) <= 1:
		for unit_id in unit_ids:
			unit_results.append(_process_unit(unit_id))
			advance_current_progress()
	else:
		with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
			futures = {pool.submit(_process_unit, unit_id): unit_id for unit_id in unit_ids}
			for future in concurrent.futures.as_completed(futures):
				unit_results.append(future.result())
				advance_current_progress()

	unit_results.sort(key=lambda item: str(item.unit_id))
	units_ok = sum(1 for item in unit_results if str(item.status).strip().lower() == "ok")
	units_error = sum(1 for item in unit_results if str(item.status).strip().lower() != "ok")
	active_logger.info(
		"reconstruct.axon_velocity_gtrs complete: units_total=%d units_ok=%d units_error=%d reconstruction_out_dir=%s",
		len(unit_results),
		units_ok,
		units_error,
		str(reconstruction_out_dir),
	)
	return unit_results


__all__ = ["run_axon_velocity_gtrs_phase"]