from __future__ import annotations

import concurrent.futures
import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from axon_recon.pipeline.execution import install_linux_parent_death_signal
from axon_recon.pipeline.execution.phase_chain import PhaseDescriptor, run_phase_chain
from axon_recon.pipeline.execution.progress import (
	add_current_progress_total,
	advance_current_progress,
)
from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.resource_budget import current_phase_worker_allocation
from axon_recon.pipeline.shared.grid_sorting import (
	coerce_grid_sort_metrics,
	grid_sort_key_for_unit,
	normalize_grid_sort_by,
)

from .core.diagnostic_plots import (
	write_unit_axon_reconstruction_diagnostic_figure,
	write_unit_channel_selection_diagnostic_figure,
)
from .core.generate_gtrs import run_generate_gtrs_phase as run_generate_gtrs_core_phase
from .core.plot_branch_propagations import (
	run_plot_branch_propagations_phase as run_plot_branch_propagations_core_phase,
)
from .core.plot_branch_propagations import write_unit_branch_propagation_plot
from .core.plot_branch_velocities import (
	run_plot_branch_velocities_phase as run_plot_branch_velocities_core_phase,
)
from .core.plot_branch_velocities import write_unit_branch_velocity_plot
from .core.plot_recons import run_plot_recons_phase as run_plot_recons_core_phase
from .core.plot_unit_summary import run_plot_unit_summary_phase as run_plot_unit_summary_core_phase
from .core.plot_unit_summary import write_unit_summary_plot
from .core.reconstruct import (
	compute_all_filters_payload,
	compute_branches_with_polyline,
	compute_delay_filter_payload,
	compute_detection_filter_payload,
	compute_gtr_json_payload,
	compute_heuristics_payload,
	compute_kurtosis_filter_payload,
	compute_peak_std_filter_payload,
	compute_raw_branches_payload,
	load_templates_for_unit,
)
from .core.report_full_chip_layout import (
	run_report_full_chip_layout_phase as run_report_full_chip_layout_core_phase,
)
from .core.report_full_chip_layout import write_full_chip_layout_plot
from .core.report_recon_grid import run_report_recon_grid_phase as run_report_recon_grid_core_phase
from .core.report_recons import run_report_recons_phase as run_report_recons_core_phase
from .core.report_summaries import run_report_summaries_phase as run_report_summaries_core_phase
from .core.report_summaries import write_reconstruct_summary_slides_pdf
from .core.summary_plots import write_amplitude_map_summary_png
from .core.unit_plots import write_unit_amplitude_map_png, write_unit_circle_recon_plot
from .integrations.axon_velocity import compute_graph_tracking, import_axon_velocity
from .io import (
	read_json,
	resolve_branch_phase_branch_output_paths,
	resolve_branch_phase_output_paths,
	resolve_full_chip_layout_output_paths,
	resolve_report_output_paths,
	resolve_unit_output_paths,
	resolve_unit_summary_phase_output_paths,
	write_json,
)
from .models.inputs import ReconstructionInputs
from .models.results import ReconstructionResult, UnitReconstructionResult
from .reporting.slides import write_reconstruct_report_markdown
from .templates.core.render import (
	finalize_grid_svg_output,
	render_footprint_map_grid_from_assets,
	render_template_report_pdf,
)
from .templates.core.unit_labels import (
	count_labels,
	filter_unit_ids_by_labels,
	load_unit_labels_from_spikesorting,
)
from .templates.models.inputs import TemplatesInputs

LOGGER = logging.getLogger("axon_recon.reconstruct")

NOISY_PLOT_LOGGER_NAMES: tuple[str, ...] = (
	"matplotlib",
	"matplotlib.font_manager",
	"matplotlib.backends.backend_pdf",
	"PIL",
	"PIL.PngImagePlugin",
	"fontTools",
	"fontTools.subset",
	"h5py",
	"h5py._conv",
	"numcodecs",
	"numcodecs.registry",
	"zarr",
	"numba",
	"numba.core",
	"numba.core.byteflow",
	"numba.core.interpreter",
	"numba.core.ssa",
)

DEBUGGY_PROJECT_LOGGER_NAMES: tuple[str, ...] = (
	"axon_recon.templates",
	"axon_recon.templates.spikeinterface",
)

DEFAULT_INTERNAL_RECONSTRUCTION_PHASE_SEQUENCE: tuple[str, ...] = (
	"templates_resolve_sources",
	"templates_analyzers",
	"templates_extract_partial_templates",
	"templates_build_templates",
	"templates_compute_template_similarity",
	"templates_plot_templates",
	"templates_report_templates",
	"templates_reports",
	"generate_gtrs",
	"plot_recons",
	"plot_branch_propagations",
	"plot_branch_velocities",
	"plot_unit_summary",
	"report_recons",
	"report_recon_grid",
	"report_full_chip_layout",
	"report_summaries",
	"clear_templates_cache",
)


def _debug_prints_enabled(inputs: ReconstructionInputs) -> bool:
	return bool(getattr(inputs, "debug_prints", False))


@contextmanager
def _quiet_unexpected_plot_logs(inputs: ReconstructionInputs):
	if _debug_prints_enabled(inputs):
		yield
		return
	original_levels: dict[str, int] = {}
	for logger_name in NOISY_PLOT_LOGGER_NAMES:
		logger = logging.getLogger(logger_name)
		original_levels[logger_name] = int(logger.level)
		if int(logger.getEffectiveLevel()) < int(logging.WARNING):
			logger.setLevel(logging.WARNING)
	for logger_name in DEBUGGY_PROJECT_LOGGER_NAMES:
		logger = logging.getLogger(logger_name)
		original_levels[logger_name] = int(logger.level)
		if int(logger.getEffectiveLevel()) < int(logging.INFO):
			logger.setLevel(logging.INFO)
	try:
		yield
	finally:
		for logger_name, level in original_levels.items():
			logging.getLogger(logger_name).setLevel(level)


def _is_empty_signal_selection_error(exc: Exception) -> bool:
	msg = str(exc).strip().lower()
	if "zero-size array to reduction operation maximum which has no identity" in msg:
		return True
	if "zero-size" in msg and "maximum" in msg:
		return True
	if "no branches found" in msg:
		return True
	return False


def _is_expected_reconstruct_unit_failure(exc: Exception) -> bool:
	msg = str(exc).strip().lower()
	if _is_empty_signal_selection_error(exc):
		return True
	if "graph tracking failed for requested template source" in msg:
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
	templates_inputs: TemplatesInputs | None = None,
	output_rel_root: str = "recon_outputs",
) -> tuple[Path, Path, Path]:
	configured_rel_root = str(output_rel_root or "recon_outputs").strip() or "recon_outputs"
	if templates_inputs is not None:
		templates_rel_root = str(getattr(templates_inputs, "output_rel_root", "") or "").strip()
		if templates_rel_root:
			configured_rel_root = templates_rel_root
	configured_path = Path(configured_rel_root).expanduser()
	templates_out_dir = configured_path if configured_path.is_absolute() else well_out_dir / configured_path
	templates_dir = templates_out_dir / "cache" / "templates"
	merged_units_dir = templates_dir / "merged"
	full_channels_templates_dir = templates_dir / "full"
	if merged_units_dir.exists():
		return templates_out_dir, merged_units_dir, full_channels_templates_dir
	raise FileNotFoundError(f"Missing merged templates directory: {merged_units_dir}")


def _reconstruct_unit_label_filter_config(inputs: ReconstructionInputs) -> tuple[tuple[str, ...], bool]:
	templates_inputs = getattr(inputs, "templates_inputs", None)
	labels_raw = getattr(inputs, "unit_label_filter_labels", None)
	if labels_raw is None and templates_inputs is not None:
		labels_raw = getattr(templates_inputs, "unit_label_filter_labels", None)
	labels = tuple(str(label).strip().lower() for label in (labels_raw or ()) if str(label).strip())
	required_raw = getattr(inputs, "unit_label_filter_required", None)
	if required_raw is None and templates_inputs is not None:
		required_raw = getattr(templates_inputs, "unit_label_filter_required", None)
	return labels, bool(required_raw) if required_raw is not None else False


def _apply_reconstruct_unit_label_filter(
	inputs: ReconstructionInputs,
	unit_ids: list[Any],
	well_out_dir: Path,
	*,
	context: str,
) -> list[Any]:
	allowed_labels, required = _reconstruct_unit_label_filter_config(inputs)
	if not allowed_labels:
		return list(unit_ids)
	labels_by_unit = load_unit_labels_from_spikesorting(well_out_dir)
	if not labels_by_unit:
		if bool(required):
			raise RuntimeError(
				"Reconstruct unit label filter is enabled, but no Bombcell/Kilosort unit labels were found under "
				f"{well_out_dir}."
			)
		LOGGER.warning(
			"reconstruct %s: unit label filter skipped because no labels were found under %s",
			context,
			well_out_dir,
		)
		return list(unit_ids)
	filtered = filter_unit_ids_by_labels(unit_ids, labels_by_unit, allowed_labels)
	LOGGER.info(
		"reconstruct %s: unit label filter allowed=%s kept=%d/%d counts=%s",
		context,
		list(allowed_labels),
		len(filtered),
		len(unit_ids),
		count_labels(labels_by_unit),
	)
	return filtered


def _build_unit_ids(inputs: ReconstructionInputs, merged_units_dir: Path, *, well_out_dir: Path | None = None) -> list[Any]:
	discovered = _discover_unit_ids(merged_units_dir)
	unit_ids = list(inputs.unit_ids) if inputs.unit_ids is not None else discovered
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	if well_out_dir is not None:
		unit_ids = _apply_reconstruct_unit_label_filter(inputs, unit_ids, well_out_dir, context="unit selection")
	return unit_ids


def _is_unit_scoped_reconstruct_run(inputs: ReconstructionInputs) -> bool:
	return bool(inputs.unit_ids) and len(inputs.unit_ids) == 1


def _should_preserve_reconstruct_reports(inputs: ReconstructionInputs) -> bool:
	if not _is_unit_scoped_reconstruct_run(inputs):
		return False
	if not (bool(inputs.force_restart) or bool(inputs.force_replot)):
		return False
	return not bool(inputs.overwrite_report_outputs_on_unit_rerun)


def _positive_int_or_none(value: Any) -> int | None:
	if value is None:
		return None
	try:
		parsed = int(value)
	except (TypeError, ValueError):
		return None
	return parsed if parsed > 0 else None


def _reconstruct_applied_debug_limits(inputs: ReconstructionInputs) -> dict[str, Any]:
	limits = {
		"limit_datasets": _positive_int_or_none(getattr(inputs, "debug_limit_datasets", None)),
		"limit_wells": _positive_int_or_none(getattr(inputs, "debug_limit_wells", None)),
		"limit_wells_per_dataset": _positive_int_or_none(
			getattr(inputs, "debug_limit_wells_per_dataset", None)
		),
		"limit_units": _positive_int_or_none(getattr(inputs, "unit_limit", None)),
		"limit_segments": _positive_int_or_none(getattr(inputs, "limit_segments", None)),
	}
	return {
		"debug_mode_enabled": bool(getattr(inputs, "debug_mode_enabled", False))
		or any(value is not None for value in limits.values()),
		**limits,
	}


def _collect_existing_reconstruct_stage_outputs(
	*,
	reconstruction_out_dir: Path,
	inputs: ReconstructionInputs,
) -> dict[str, str]:
	stage_outputs: dict[str, str] = {}
	for key, path in resolve_report_output_paths(
		reconstruction_out_dir=reconstruction_out_dir,
		report_recons_phase=inputs.phases.report_recons,
		report_recon_grid_phase=inputs.phases.report_recon_grid,
		report_full_chip_layout_phase=inputs.phases.report_full_chip_layout,
		report_summaries_phase=inputs.phases.report_summaries,
	).items():
		if path.exists():
			stage_outputs[key] = str(path)
	if bool(inputs.phases.report_recons.summary_png.write):
		summary_png = reconstruction_out_dir / Path(str(inputs.phases.report_recons.summary_png.relpath)).expanduser()
		if summary_png.exists():
			stage_outputs["summary_png"] = str(summary_png)
	if bool(inputs.phases.report_recons.report_md.write):
		report_md = reconstruction_out_dir / Path(str(inputs.phases.report_recons.report_md.relpath)).expanduser()
		if report_md.exists():
			stage_outputs["report_md"] = str(report_md)
	return stage_outputs


def _load_reconstruct_unit_grid_sort_metrics(
	*,
	reconstruction_out_dir: Path,
	unit_id: Any,
	inputs: ReconstructionInputs,
) -> dict[str, float]:
	paths = resolve_unit_output_paths(
		reconstruction_out_dir=reconstruction_out_dir,
		unit_id=unit_id,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	unit_summary_json = paths["unit_summary_json"]
	if not unit_summary_json.exists():
		return {}
	try:
		payload = read_json(unit_summary_json)
	except Exception:
		return {}
	if not isinstance(payload, dict):
		return {}
	return coerce_grid_sort_metrics(payload.get("grid_sort_metrics", {}))


def _sort_reconstruct_units_for_reports(
	*,
	unit_results: list[UnitReconstructionResult],
	reconstruction_out_dir: Path,
	inputs: ReconstructionInputs,
	sort_by: str,
) -> list[UnitReconstructionResult]:
	sort_mode = normalize_grid_sort_by(sort_by, default="unit_id")
	metrics_by_unit: dict[str, dict[str, float]] | None = None
	if sort_mode != "unit_id":
		metrics_by_unit = {}
		for unit_result in unit_results:
			metrics_by_unit[str(unit_result.unit_id).strip()] = _load_reconstruct_unit_grid_sort_metrics(
				reconstruction_out_dir=reconstruction_out_dir,
				unit_id=unit_result.unit_id,
				inputs=inputs,
			)
	return sorted(
		list(unit_results),
		key=lambda result: grid_sort_key_for_unit(
			result.unit_id,
			sort_by=sort_mode,
			metrics_by_unit=metrics_by_unit,
		),
	)


def _remove_unit_outputs_preserving(*, unit_dir: Path, preserve_paths: list[Path]) -> list[str]:
	if not unit_dir.exists():
		return []
	preserve = {path.resolve() for path in preserve_paths if path.exists()}
	removed_paths: list[str] = []
	for path in sorted((candidate for candidate in unit_dir.rglob("*") if candidate.is_file()), key=lambda candidate: (-len(candidate.parts), str(candidate))):
		if path.resolve() in preserve:
			continue
		path.unlink(missing_ok=True)
		removed_paths.append(str(path))
	for directory in sorted((candidate for candidate in unit_dir.rglob("*") if candidate.is_dir()), key=lambda candidate: -len(candidate.parts)):
		try:
			directory.rmdir()
		except OSError:
			pass
	return removed_paths


def _cleanup_failed_reconstruct_unit_outputs(
	*,
	reconstruction_out_dir: Path,
	inputs: ReconstructionInputs,
	unit_results: list[UnitReconstructionResult],
) -> tuple[list[UnitReconstructionResult], Path | None]:
	cleanup_enabled = bool(inputs.cleanup_failed_unit_outputs)
	failed_rows: list[dict[str, Any]] = []
	updated_results: list[UnitReconstructionResult] = []
	for unit_result in unit_results:
		status = str(unit_result.status or "").strip().lower()
		if status == "ok":
			updated_results.append(unit_result)
			continue
		paths = resolve_unit_output_paths(
			reconstruction_out_dir=reconstruction_out_dir,
			unit_id=unit_result.unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		unit_summary_json = paths["unit_summary_json"]
		removed_output_paths: list[str] = []
		if cleanup_enabled:
			removed_output_paths = _remove_unit_outputs_preserving(
				unit_dir=paths["unit_dir"],
				preserve_paths=[unit_summary_json],
			)
			if unit_summary_json.exists():
				unit_summary_payload = read_json(unit_summary_json)
				if isinstance(unit_summary_payload, dict):
					unit_summary_payload["outputs"] = {}
					unit_summary_payload["cleanup_failed_outputs_applied"] = True
					unit_summary_payload["removed_output_paths"] = removed_output_paths
					write_json(unit_summary_json, unit_summary_payload)
		updated_results.append(
			UnitReconstructionResult(
				unit_id=unit_result.unit_id,
				status=unit_result.status,
				outputs={} if cleanup_enabled else dict(unit_result.outputs),
				error=unit_result.error,
			)
		)
		failed_rows.append(
			{
				"unit_id": unit_result.unit_id,
				"status": unit_result.status,
				"error": unit_result.error,
				"unit_dir": str(paths["unit_dir"]),
				"unit_summary_json": str(unit_summary_json),
				"cleanup_failed_outputs_applied": cleanup_enabled,
				"removed_output_paths": removed_output_paths,
			}
		)
	failed_summary_json: Path | None = None
	if failed_rows:
		failed_summary_json = reconstruction_out_dir / Path(str(inputs.failed_units_summary_relpath)).expanduser()
		write_json(
			failed_summary_json,
			{
				"stage": "reconstruct",
				"h5_path": str(inputs.h5_path),
				"stream_id": str(inputs.stream_id),
				"cleanup_failed_unit_outputs": cleanup_enabled,
				"failed_unit_count": len(failed_rows),
				"units": failed_rows,
			},
		)
	return updated_results, failed_summary_json


@dataclass(frozen=True)
class _ReconstructPhaseEnvironment:
	well_out_dir: Path
	reconstruction_out_dir: Path
	merged_units_dir: Path
	full_channels_templates_dir: Path
	unit_ids: list[Any]
	preserve_stage_reports: bool
	existing_stage_outputs: dict[str, str]


def _unit_result_from_summary_payload(*, unit_id: Any, payload: Any) -> UnitReconstructionResult:
	data = payload if isinstance(payload, dict) else {}
	outputs = dict(data.get("outputs", {})) if isinstance(data.get("outputs", {}), dict) else {}
	return UnitReconstructionResult(
		unit_id=unit_id,
		status=str(data.get("status", "ok")),
		outputs={str(key): str(value) for key, value in outputs.items() if value is not None},
		error=(None if not data.get("error") else str(data.get("error"))),
	)


def _load_reconstruct_unit_results(
	*,
	reconstruction_out_dir: Path,
	inputs: ReconstructionInputs,
	unit_ids: list[Any],
) -> list[UnitReconstructionResult]:
	results: list[UnitReconstructionResult] = []
	for unit_id in unit_ids:
		paths = resolve_unit_output_paths(
			reconstruction_out_dir=reconstruction_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		unit_summary_json = paths["unit_summary_json"]
		if not unit_summary_json.exists():
			results.append(
				UnitReconstructionResult(
					unit_id=unit_id,
					status="error",
					outputs={},
					error="Missing unit summary",
				)
			)
			continue
		try:
			payload = read_json(unit_summary_json)
		except Exception as exc:
			results.append(
				UnitReconstructionResult(
					unit_id=unit_id,
					status="error",
					outputs={},
					error=str(exc),
				)
			)
			continue
		results.append(_unit_result_from_summary_payload(unit_id=unit_id, payload=payload))
	results.sort(key=lambda item: str(item.unit_id))
	return results


def _load_full_chip_layout_unit_results(
	*,
	reconstruction_out_dir: Path,
	inputs: ReconstructionInputs,
	merged_units_dir: Path,
	unit_ids: list[Any] | None = None,
) -> list[UnitReconstructionResult]:
	unit_ids = list(unit_ids) if unit_ids is not None else _discover_unit_ids(merged_units_dir)
	if not unit_ids:
		unit_ids = list(inputs.unit_ids or [])
	return _load_reconstruct_unit_results(
		reconstruction_out_dir=reconstruction_out_dir,
		inputs=inputs,
		unit_ids=unit_ids,
	)


def _count_unit_statuses(unit_results: list[UnitReconstructionResult]) -> tuple[int, int]:
	units_ok = sum(1 for item in unit_results if str(item.status).strip().lower() == "ok")
	units_error = sum(1 for item in unit_results if str(item.status).strip().lower() != "ok")
	return units_ok, units_error


def _unit_rows(unit_results: list[UnitReconstructionResult]) -> list[dict[str, Any]]:
	return [
		{
			"unit_id": item.unit_id,
			"status": item.status,
			"outputs": item.outputs,
			"error": item.error,
		}
		for item in unit_results
	]


def _write_reconstruct_phase_summary(
	*,
	phase_name: str,
	summary_json: Path,
	inputs: ReconstructionInputs,
	well_out_dir: Path,
	reconstruction_out_dir: Path,
	unit_results: list[UnitReconstructionResult],
	stage_outputs: dict[str, str] | None = None,
	failed_units_summary_json: Path | None = None,
	preserve_stage_reports: bool = False,
	extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
	units_ok, units_error = _count_unit_statuses(unit_results)
	payload: dict[str, Any] = {
		"phase": phase_name,
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"n_jobs": int(max(1, int(inputs.n_jobs))),
		"well_out_dir": str(well_out_dir),
		"reconstruction_out_dir": str(reconstruction_out_dir),
		"applied_debug_limits": _reconstruct_applied_debug_limits(inputs),
		"unit_count": len(unit_results),
		"units_ok": units_ok,
		"units_error": units_error,
		"outputs": dict(stage_outputs or {}),
		"cleanup_failed_unit_outputs": bool(inputs.cleanup_failed_unit_outputs),
		"failed_units_summary_json": str(failed_units_summary_json) if failed_units_summary_json else None,
		"reports_overwrite_skipped": bool(preserve_stage_reports),
		"units": _unit_rows(unit_results),
	}
	if extra_fields:
		payload.update(dict(extra_fields))
	write_json(summary_json, payload)
	payload["summary_json"] = str(summary_json)
	return payload


def _prepare_reconstruct_phase_environment(
	*,
	inputs: ReconstructionInputs,
	clear_output_root: bool,
) -> _ReconstructPhaseEnvironment:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	full_restart = bool(inputs.force_restart) and (not bool(inputs.force_replot))
	reconstruction_out_dir = well_out_dir / str(inputs.output_rel_root)
	preserve_stage_reports = _should_preserve_reconstruct_reports(inputs)
	existing_stage_outputs = (
		_collect_existing_reconstruct_stage_outputs(
			reconstruction_out_dir=reconstruction_out_dir,
			inputs=inputs,
		)
		if preserve_stage_reports
		else {}
	)
	if clear_output_root and full_restart and reconstruction_out_dir.exists():
		if preserve_stage_reports:
			LOGGER.info(
				"Reconstruct unit-scoped restart preserving stage reports for unit_ids=%s",
				list(inputs.unit_ids or []),
			)
		else:
			preserved_templates_cache: Path | None = None
			templates_cache_dir = reconstruction_out_dir / "cache" / "templates"
			if templates_cache_dir.exists():
				preserved_templates_cache = reconstruction_out_dir.parent / f".{reconstruction_out_dir.name}_templates_cache_preserved"
				if preserved_templates_cache.exists():
					shutil.rmtree(preserved_templates_cache)
				shutil.move(str(templates_cache_dir), str(preserved_templates_cache))
			LOGGER.info("Reconstruct full restart: clearing output root %s", reconstruction_out_dir)
			shutil.rmtree(reconstruction_out_dir)
	reconstruction_out_dir.mkdir(parents=True, exist_ok=True)
	if clear_output_root and full_restart and not preserve_stage_reports:
		preserved_templates_cache = reconstruction_out_dir.parent / f".{reconstruction_out_dir.name}_templates_cache_preserved"
		if preserved_templates_cache.exists():
			restored_templates_cache = reconstruction_out_dir / "cache" / "templates"
			restored_templates_cache.parent.mkdir(parents=True, exist_ok=True)
			if restored_templates_cache.exists():
				shutil.rmtree(restored_templates_cache)
			shutil.move(str(preserved_templates_cache), str(restored_templates_cache))
	if clear_output_root and preserve_stage_reports and full_restart:
		for unit_id in inputs.unit_ids or []:
			unit_dir = resolve_unit_output_paths(
				reconstruction_out_dir=reconstruction_out_dir,
				unit_id=unit_id,
				per_unit_outputs=inputs.per_unit_outputs,
			)["unit_dir"]
			if unit_dir.exists():
				shutil.rmtree(unit_dir)

	try:
		_, merged_units_dir, full_channels_templates_dir = _resolve_templates_dirs(
			well_out_dir,
			templates_inputs=inputs.templates_inputs,
			output_rel_root=inputs.output_rel_root,
		)
	except TypeError as exc:
		if "templates_inputs" not in str(exc):
			raise
		_, merged_units_dir, full_channels_templates_dir = _resolve_templates_dirs(well_out_dir)
	unit_ids = _build_unit_ids(inputs, merged_units_dir, well_out_dir=well_out_dir)
	return _ReconstructPhaseEnvironment(
		well_out_dir=well_out_dir,
		reconstruction_out_dir=reconstruction_out_dir,
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_channels_templates_dir,
		unit_ids=unit_ids,
		preserve_stage_reports=preserve_stage_reports,
		existing_stage_outputs=dict(existing_stage_outputs),
	)


def _current_failed_units_summary_json(*, inputs: ReconstructionInputs, reconstruction_out_dir: Path) -> Path | None:
	candidate = reconstruction_out_dir / Path(str(inputs.failed_units_summary_relpath)).expanduser()
	return candidate if candidate.exists() else None


def _as_positive_int_or_none(value: Any) -> int | None:
	try:
		parsed = int(value)
	except Exception:
		return None
	if parsed <= 0:
		return None
	return int(parsed)


def _chunk_unit_ids(unit_ids: list[Any], *, batch_size: int) -> list[list[Any]]:
	resolved_batch_size = max(1, int(batch_size))
	return [list(unit_ids[idx : idx + resolved_batch_size]) for idx in range(0, len(unit_ids), resolved_batch_size)]


def _resolve_generate_gtrs_execution_plan(
	*,
	inputs: ReconstructionInputs,
	unit_ids: list[Any],
) -> tuple[int, int, int, list[list[Any]]]:
	unit_count = len(unit_ids)
	if unit_count <= 0:
		return 1, 1, 1, []
	derived_unit_workers = max(1, int(inputs.n_jobs))
	phase_cfg = inputs.phases.generate_gtrs
	unit_procs = _as_positive_int_or_none(getattr(phase_cfg, "unit_procs", None))
	if unit_procs is None:
		unit_procs = min(derived_unit_workers, 6)
	unit_procs = max(1, min(int(unit_procs), derived_unit_workers, unit_count))
	unit_batch_size = _as_positive_int_or_none(getattr(phase_cfg, "unit_batch_size", None))
	if unit_batch_size is None:
		unit_batch_size = max(1, (unit_count + unit_procs - 1) // unit_procs)
	batches = _chunk_unit_ids(unit_ids, batch_size=unit_batch_size)
	process_workers = max(1, min(unit_procs, len(batches)))
	return derived_unit_workers, process_workers, int(unit_batch_size), batches


@dataclass(frozen=True)
class _GenerateGtrsBatchInputs:
	inputs: ReconstructionInputs
	reconstruction_out_dir: Path
	merged_units_dir: Path
	full_channels_templates_dir: Path


def _run_generate_gtrs_batch(batch_inputs: _GenerateGtrsBatchInputs) -> list[UnitReconstructionResult]:
	return run_generate_gtrs_core_phase(
		inputs=batch_inputs.inputs,
		reconstruction_out_dir=batch_inputs.reconstruction_out_dir,
		merged_units_dir=batch_inputs.merged_units_dir,
		full_channels_templates_dir=batch_inputs.full_channels_templates_dir,
		unit_ids=list(batch_inputs.inputs.unit_ids or []),
		import_axon_velocity_fn=import_axon_velocity,
		load_templates_for_unit_fn=load_templates_for_unit,
		compute_graph_tracking_fn=compute_graph_tracking,
		compute_raw_branches_payload_fn=compute_raw_branches_payload,
		compute_branches_with_polyline_fn=compute_branches_with_polyline,
		compute_detection_filter_payload_fn=compute_detection_filter_payload,
		compute_kurtosis_filter_payload_fn=compute_kurtosis_filter_payload,
		compute_peak_std_filter_payload_fn=compute_peak_std_filter_payload,
		compute_delay_filter_payload_fn=compute_delay_filter_payload,
		compute_all_filters_payload_fn=compute_all_filters_payload,
		compute_heuristics_payload_fn=compute_heuristics_payload,
		compute_gtr_json_payload_fn=compute_gtr_json_payload,
		write_unit_channel_selection_diagnostic_figure_fn=write_unit_channel_selection_diagnostic_figure,
		write_unit_axon_reconstruction_diagnostic_figure_fn=write_unit_axon_reconstruction_diagnostic_figure,
		read_json_fn=read_json,
		write_json_fn=write_json,
		resolve_unit_output_paths_fn=resolve_unit_output_paths,
		is_empty_signal_selection_error_fn=_is_empty_signal_selection_error,
		is_expected_reconstruct_unit_failure_fn=_is_expected_reconstruct_unit_failure,
		normalize_template_for_tracking_fn=_normalize_template_for_tracking,
		logger=LOGGER,
	)


def _run_reconstruct_generate_gtrs_batches(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
) -> list[UnitReconstructionResult]:
	derived_unit_workers, unit_procs, unit_batch_size, batches = _resolve_generate_gtrs_execution_plan(
		inputs=inputs,
		unit_ids=env.unit_ids,
	)
	LOGGER.info(
		"reconstruct.generate_gtrs execution plan: requested_units=%d derived_unit_workers=%d unit_procs=%d unit_batch_size=%d unit_batches=%d",
		len(env.unit_ids),
		int(derived_unit_workers),
		int(unit_procs),
		int(unit_batch_size),
		len(batches),
	)
	if len(batches) <= 1 or unit_procs <= 1:
		return run_generate_gtrs_core_phase(
			inputs=inputs,
			reconstruction_out_dir=env.reconstruction_out_dir,
			merged_units_dir=env.merged_units_dir,
			full_channels_templates_dir=env.full_channels_templates_dir,
			unit_ids=env.unit_ids,
			import_axon_velocity_fn=import_axon_velocity,
			load_templates_for_unit_fn=load_templates_for_unit,
			compute_graph_tracking_fn=compute_graph_tracking,
			compute_raw_branches_payload_fn=compute_raw_branches_payload,
			compute_branches_with_polyline_fn=compute_branches_with_polyline,
			compute_detection_filter_payload_fn=compute_detection_filter_payload,
			compute_kurtosis_filter_payload_fn=compute_kurtosis_filter_payload,
			compute_peak_std_filter_payload_fn=compute_peak_std_filter_payload,
			compute_delay_filter_payload_fn=compute_delay_filter_payload,
			compute_all_filters_payload_fn=compute_all_filters_payload,
			compute_heuristics_payload_fn=compute_heuristics_payload,
			compute_gtr_json_payload_fn=compute_gtr_json_payload,
			write_unit_channel_selection_diagnostic_figure_fn=write_unit_channel_selection_diagnostic_figure,
			write_unit_axon_reconstruction_diagnostic_figure_fn=write_unit_axon_reconstruction_diagnostic_figure,
			read_json_fn=read_json,
			write_json_fn=write_json,
			resolve_unit_output_paths_fn=resolve_unit_output_paths,
			is_empty_signal_selection_error_fn=_is_empty_signal_selection_error,
			is_expected_reconstruct_unit_failure_fn=_is_expected_reconstruct_unit_failure,
			normalize_template_for_tracking_fn=_normalize_template_for_tracking,
			logger=LOGGER,
		)

	batch_inputs_list = [
		_GenerateGtrsBatchInputs(
			inputs=replace(inputs, n_jobs=1, unit_ids=list(batch_unit_ids)),
			reconstruction_out_dir=env.reconstruction_out_dir,
			merged_units_dir=env.merged_units_dir,
			full_channels_templates_dir=env.full_channels_templates_dir,
		)
		for batch_unit_ids in batches
	]
	batch_results: list[UnitReconstructionResult] = []
	try:
		add_current_progress_total(len(env.unit_ids))
		with concurrent.futures.ProcessPoolExecutor(
			max_workers=unit_procs,
			initializer=install_linux_parent_death_signal,
		) as pool:
			futures = {
				pool.submit(_run_generate_gtrs_batch, batch_inputs): list(batch_inputs.inputs.unit_ids or [])
				for batch_inputs in batch_inputs_list
			}
			completed = 0
			completed_units = 0
			total_batches = len(futures)
			for future in concurrent.futures.as_completed(futures):
				batch_result = future.result()
				batch_results.extend(batch_result)
				completed += 1
				completed_batch_units = len(batch_result)
				completed_units += completed_batch_units
				advance_current_progress(completed_batch_units)
				LOGGER.info(
					"reconstruct.generate_gtrs unified progress: %d/%d units completed (%d/%d batches)",
					completed_units,
					len(env.unit_ids),
					completed,
					total_batches,
				)
	except Exception as exc:
		LOGGER.warning(
			"reconstruct.generate_gtrs process pool execution failed, falling back to in-process execution: %s",
			exc,
		)
		return run_generate_gtrs_core_phase(
			inputs=inputs,
			reconstruction_out_dir=env.reconstruction_out_dir,
			merged_units_dir=env.merged_units_dir,
			full_channels_templates_dir=env.full_channels_templates_dir,
			unit_ids=env.unit_ids,
			import_axon_velocity_fn=import_axon_velocity,
			load_templates_for_unit_fn=load_templates_for_unit,
			compute_graph_tracking_fn=compute_graph_tracking,
			compute_raw_branches_payload_fn=compute_raw_branches_payload,
			compute_branches_with_polyline_fn=compute_branches_with_polyline,
			compute_detection_filter_payload_fn=compute_detection_filter_payload,
			compute_kurtosis_filter_payload_fn=compute_kurtosis_filter_payload,
			compute_peak_std_filter_payload_fn=compute_peak_std_filter_payload,
			compute_delay_filter_payload_fn=compute_delay_filter_payload,
			compute_all_filters_payload_fn=compute_all_filters_payload,
			compute_heuristics_payload_fn=compute_heuristics_payload,
			compute_gtr_json_payload_fn=compute_gtr_json_payload,
			write_unit_channel_selection_diagnostic_figure_fn=write_unit_channel_selection_diagnostic_figure,
			write_unit_axon_reconstruction_diagnostic_figure_fn=write_unit_axon_reconstruction_diagnostic_figure,
			read_json_fn=read_json,
			write_json_fn=write_json,
			resolve_unit_output_paths_fn=resolve_unit_output_paths,
			is_empty_signal_selection_error_fn=_is_empty_signal_selection_error,
			is_expected_reconstruct_unit_failure_fn=_is_expected_reconstruct_unit_failure,
			normalize_template_for_tracking_fn=_normalize_template_for_tracking,
			logger=LOGGER,
			progress_total_already_added=True,
		)

	batch_results.sort(key=lambda item: str(item.unit_id))
	return batch_results


def _run_reconstruct_generate_gtrs_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
) -> tuple[list[UnitReconstructionResult], Path | None]:
	unit_results = _run_reconstruct_generate_gtrs_batches(inputs=inputs, env=env)
	return _cleanup_failed_reconstruct_unit_outputs(
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		unit_results=unit_results,
	)


def _run_reconstruct_plot_recons_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
) -> tuple[list[UnitReconstructionResult], Path | None]:
	unit_results = run_plot_recons_core_phase(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
		merged_units_dir=env.merged_units_dir,
		full_channels_templates_dir=env.full_channels_templates_dir,
		unit_ids=env.unit_ids,
		load_templates_for_unit_fn=load_templates_for_unit,
		write_unit_amplitude_map_png_fn=write_unit_amplitude_map_png,
		write_unit_circle_recon_plot_fn=write_unit_circle_recon_plot,
		read_json_fn=read_json,
		write_json_fn=write_json,
		resolve_unit_output_paths_fn=resolve_unit_output_paths,
		logger=LOGGER,
	)
	return _cleanup_failed_reconstruct_unit_outputs(
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		unit_results=unit_results,
	)


def _run_reconstruct_plot_branch_propagations_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
) -> tuple[list[UnitReconstructionResult], Path | None]:
	unit_results = run_plot_branch_propagations_core_phase(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
		merged_units_dir=env.merged_units_dir,
		full_channels_templates_dir=env.full_channels_templates_dir,
		unit_ids=env.unit_ids,
		load_templates_for_unit_fn=load_templates_for_unit,
		write_unit_branch_propagation_plot_fn=write_unit_branch_propagation_plot,
		read_json_fn=read_json,
		write_json_fn=write_json,
		resolve_unit_output_paths_fn=resolve_unit_output_paths,
		resolve_branch_phase_output_paths_fn=resolve_branch_phase_output_paths,
		resolve_branch_phase_branch_output_paths_fn=resolve_branch_phase_branch_output_paths,
		logger=LOGGER,
	)
	return _cleanup_failed_reconstruct_unit_outputs(
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		unit_results=unit_results,
	)


def _run_reconstruct_plot_branch_velocities_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
) -> tuple[list[UnitReconstructionResult], Path | None]:
	unit_results = run_plot_branch_velocities_core_phase(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
		merged_units_dir=env.merged_units_dir,
		full_channels_templates_dir=env.full_channels_templates_dir,
		unit_ids=env.unit_ids,
		load_templates_for_unit_fn=load_templates_for_unit,
		write_unit_branch_velocity_plot_fn=write_unit_branch_velocity_plot,
		read_json_fn=read_json,
		write_json_fn=write_json,
		resolve_unit_output_paths_fn=resolve_unit_output_paths,
		resolve_branch_phase_output_paths_fn=resolve_branch_phase_output_paths,
		resolve_branch_phase_branch_output_paths_fn=resolve_branch_phase_branch_output_paths,
		logger=LOGGER,
	)
	return _cleanup_failed_reconstruct_unit_outputs(
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		unit_results=unit_results,
	)


def _run_reconstruct_plot_unit_summary_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
) -> tuple[list[UnitReconstructionResult], Path | None]:
	unit_results = run_plot_unit_summary_core_phase(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
		merged_units_dir=env.merged_units_dir,
		full_channels_templates_dir=env.full_channels_templates_dir,
		unit_ids=env.unit_ids,
		load_templates_for_unit_fn=load_templates_for_unit,
		write_unit_summary_plot_fn=write_unit_summary_plot,
		read_json_fn=read_json,
		write_json_fn=write_json,
		resolve_unit_output_paths_fn=resolve_unit_output_paths,
		resolve_unit_summary_phase_output_paths_fn=resolve_unit_summary_phase_output_paths,
		logger=LOGGER,
	)
	return _cleanup_failed_reconstruct_unit_outputs(
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		unit_results=unit_results,
	)


def _run_reconstruct_report_recons_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
	unit_results: list[UnitReconstructionResult],
) -> dict[str, str]:
	report_sort_by = normalize_grid_sort_by(inputs.report_sort_by, default="unit_id")
	unit_results_for_reports = _sort_reconstruct_units_for_reports(
		unit_results=unit_results,
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		sort_by=report_sort_by,
	)
	return run_report_recons_core_phase(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
		unit_results=unit_results,
		unit_results_for_reports=unit_results_for_reports,
		preserve_stage_reports=env.preserve_stage_reports,
		existing_stage_outputs=env.existing_stage_outputs,
		resolve_report_output_paths_fn=resolve_report_output_paths,
		write_amplitude_map_summary_png_fn=write_amplitude_map_summary_png,
		render_template_report_pdf_fn=render_template_report_pdf,
		write_reconstruct_report_markdown_fn=write_reconstruct_report_markdown,
		logger=LOGGER,
	)


def _run_reconstruct_report_recon_grid_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
	unit_results: list[UnitReconstructionResult],
	stage_outputs: dict[str, str],
) -> dict[str, str]:
	report_sort_by = normalize_grid_sort_by(inputs.report_sort_by, default="unit_id")
	unit_results_for_reports = _sort_reconstruct_units_for_reports(
		unit_results=unit_results,
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		sort_by=report_sort_by,
	)
	return run_report_recon_grid_core_phase(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
		unit_results_for_reports=unit_results_for_reports,
		preserve_stage_reports=env.preserve_stage_reports,
		existing_stage_outputs=stage_outputs,
		resolve_report_output_paths_fn=resolve_report_output_paths,
		render_footprint_map_grid_from_assets_fn=render_footprint_map_grid_from_assets,
		finalize_grid_svg_output_fn=finalize_grid_svg_output,
		logger=LOGGER,
	)


def _run_reconstruct_report_full_chip_layout_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
	unit_results: list[UnitReconstructionResult],
) -> dict[str, str]:
	LOGGER.info(
		"reconstruct.report_full_chip_layout overwrite policy: action=rewrite preserve_stage_reports_requested=%s force_restart=%s force_replot=%s selected_units=%d discovered_units=%d existing_full_chip_outputs=%s",
		bool(env.preserve_stage_reports),
		bool(inputs.force_restart),
		bool(inputs.force_replot),
		len(env.unit_ids),
		len(unit_results),
		sorted(
			key
			for key in env.existing_stage_outputs
			if str(key).startswith("full_chip_layout_")
		),
	)
	return run_report_full_chip_layout_core_phase(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
		merged_units_dir=env.merged_units_dir,
		full_channels_templates_dir=env.full_channels_templates_dir,
		unit_results=unit_results,
		preserve_stage_reports=False,
		existing_stage_outputs=env.existing_stage_outputs,
		load_templates_for_unit_fn=load_templates_for_unit,
		write_full_chip_layout_plot_fn=write_full_chip_layout_plot,
		write_json_fn=write_json,
		resolve_unit_output_paths_fn=resolve_unit_output_paths,
		resolve_full_chip_layout_output_paths_fn=resolve_full_chip_layout_output_paths,
		logger=LOGGER,
	)


def _run_reconstruct_report_summaries_phase_impl(
	*,
	inputs: ReconstructionInputs,
	env: _ReconstructPhaseEnvironment,
	unit_results: list[UnitReconstructionResult],
	stage_outputs: dict[str, str],
) -> dict[str, str]:
	report_sort_by = normalize_grid_sort_by(inputs.report_sort_by, default="unit_id")
	unit_results_for_reports = _sort_reconstruct_units_for_reports(
		unit_results=unit_results,
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		sort_by=report_sort_by,
	)
	return run_report_summaries_core_phase(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
		unit_results=unit_results,
		unit_results_for_reports=unit_results_for_reports,
		preserve_stage_reports=env.preserve_stage_reports,
		existing_stage_outputs=stage_outputs,
		resolve_report_output_paths_fn=resolve_report_output_paths,
		write_reconstruct_summary_slides_pdf_fn=write_reconstruct_summary_slides_pdf,
		logger=LOGGER,
	)


def run_reconstruct_templates_resolve_sources_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.resolve_sources import (
		run_reconstruct_templates_resolve_sources_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError("reconstruct.templates_resolve_sources requires templates_inputs to be populated on ReconstructionInputs")
	return run_reconstruct_templates_resolve_sources_phase(inputs.templates_inputs)


def run_reconstruct_templates_analyzers_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.analyzers import (
		run_reconstruct_templates_analyzers_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError("reconstruct.templates_analyzers requires templates_inputs to be populated on ReconstructionInputs")
	return run_reconstruct_templates_analyzers_phase(inputs.templates_inputs)


def run_reconstruct_templates_extract_partial_templates_phase(
	inputs: ReconstructionInputs,
) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.extract_partial_templates import (
		run_reconstruct_templates_extract_partial_templates_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError(
			"reconstruct.templates_extract_partial_templates requires templates_inputs to be populated on ReconstructionInputs"
		)
	return run_reconstruct_templates_extract_partial_templates_phase(inputs.templates_inputs)


def run_reconstruct_templates_build_templates_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.build_templates import (
		run_reconstruct_templates_build_templates_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError("reconstruct.templates_build_templates requires templates_inputs to be populated on ReconstructionInputs")
	return run_reconstruct_templates_build_templates_phase(inputs.templates_inputs)


# Short aliases used by the slice 2 acceptance check. These mirror the canonical
# `run_reconstruct_templates_<phase>_phase` shims above without the
# ``templates_`` infix, so external scripts can import either form.
run_reconstruct_extract_partial_templates_phase = (
	run_reconstruct_templates_extract_partial_templates_phase
)
run_reconstruct_build_templates_phase = run_reconstruct_templates_build_templates_phase


def run_reconstruct_templates_compute_template_similarity_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.compute_template_similarity import (
		run_reconstruct_templates_compute_template_similarity_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError("reconstruct.templates_compute_template_similarity requires templates_inputs to be populated on ReconstructionInputs")
	return run_reconstruct_templates_compute_template_similarity_phase(inputs.templates_inputs)


def run_reconstruct_templates_plot_templates_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.plot_templates import (
		run_reconstruct_templates_plot_templates_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError("reconstruct.templates_plot_templates requires templates_inputs to be populated on ReconstructionInputs")
	return run_reconstruct_templates_plot_templates_phase(inputs.templates_inputs)


def run_reconstruct_templates_plot_templates_v2_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.plot_templates_v2 import (
		run_reconstruct_templates_plot_templates_v2_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError("reconstruct.templates_plot_templates_v2 requires templates_inputs to be populated on ReconstructionInputs")
	return run_reconstruct_templates_plot_templates_v2_phase(inputs.templates_inputs)


def run_reconstruct_templates_report_templates_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.report_templates import (
		run_reconstruct_templates_report_templates_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError("reconstruct.templates_report_templates requires templates_inputs to be populated on ReconstructionInputs")
	return run_reconstruct_templates_report_templates_phase(inputs.templates_inputs)


def run_reconstruct_templates_reports_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.reports import (
		run_reconstruct_templates_reports_phase,
	)

	if inputs.templates_inputs is None:
		raise ValueError("reconstruct.templates_reports requires templates_inputs to be populated on ReconstructionInputs")
	return run_reconstruct_templates_reports_phase(inputs.templates_inputs)


def run_reconstruct_clear_templates_cache_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.clear_templates_cache import (
		run_reconstruct_clear_templates_cache_phase,
	)

	return run_reconstruct_clear_templates_cache_phase(inputs)


def run_reconstruct_generate_gtrs_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.generate_gtrs import (
		run_reconstruct_generate_gtrs_phase,
	)

	return run_reconstruct_generate_gtrs_phase(inputs)


def run_reconstruct_plot_recons_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.plot_recons import (
		run_reconstruct_plot_recons_phase,
	)

	return run_reconstruct_plot_recons_phase(inputs)


def run_reconstruct_plot_branch_propagations_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.plot_branch_propagations import (
		run_reconstruct_plot_branch_propagations_phase,
	)

	return run_reconstruct_plot_branch_propagations_phase(inputs)


def run_reconstruct_plot_branch_velocities_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.plot_branch_velocities import (
		run_reconstruct_plot_branch_velocities_phase,
	)

	return run_reconstruct_plot_branch_velocities_phase(inputs)


def run_reconstruct_plot_unit_summary_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.plot_unit_summary import (
		run_reconstruct_plot_unit_summary_phase,
	)

	return run_reconstruct_plot_unit_summary_phase(inputs)


def run_reconstruct_report_recons_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.report_recons import (
		run_reconstruct_report_recons_phase,
	)

	return run_reconstruct_report_recons_phase(inputs)


def run_reconstruct_report_recon_grid_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.report_recon_grid import (
		run_reconstruct_report_recon_grid_phase,
	)

	return run_reconstruct_report_recon_grid_phase(inputs)


def run_reconstruct_report_full_chip_layout_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.report_full_chip_layout import (
		run_reconstruct_report_full_chip_layout_phase,
	)

	return run_reconstruct_report_full_chip_layout_phase(inputs)


def run_reconstruct_report_summaries_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
	from axon_recon.pipeline.stages.reconstruct.phases.report_summaries import (
		run_reconstruct_report_summaries_phase,
	)

	return run_reconstruct_report_summaries_phase(inputs)


def _normalize_reconstruct_stage_phase_name(raw: Any) -> str:
	token = str(raw or "").strip().replace("-", "_").replace(" ", "_")
	aliases = {
		"generate": "generate_gtrs",
		"gtrs": "generate_gtrs",
		"resolve_sources": "templates_resolve_sources",
		"templates.resolve_sources": "templates_resolve_sources",
		"analyzers": "templates_analyzers",
		"templates.analyzers": "templates_analyzers",
		"extract_partial_templates": "templates_extract_partial_templates",
		"templates.extract_partial_templates": "templates_extract_partial_templates",
		"build_templates": "templates_build_templates",
		"templates.build_templates": "templates_build_templates",
		"compute_template_similarity": "templates_compute_template_similarity",
		"templates.compute_template_similarity": "templates_compute_template_similarity",
		"plot_templates": "templates_plot_templates",
		"templates.plot_templates": "templates_plot_templates",
		"plot_templates_v2": "templates_plot_templates_v2",
		"plots_v2": "templates_plot_templates_v2",
		"template_plots_v2": "templates_plot_templates_v2",
		"templates.plot_templates_v2": "templates_plot_templates_v2",
		"report_templates": "templates_report_templates",
		"templates.report_templates": "templates_report_templates",
		"reports": "templates_reports",
		"templates.reports": "templates_reports",
		"clear_cache": "clear_templates_cache",
		"plot_reconstructions": "plot_recons",
		"report_reconstructions": "report_recons",
		"report_recon_grid": "report_recon_grid",
	}
	return aliases.get(token, token)


def _reconstruct_stage_phase_enabled(inputs: ReconstructionInputs, phase_name: str) -> bool:
	phase = _normalize_reconstruct_stage_phase_name(phase_name)
	if phase == "templates_resolve_sources":
		return bool(inputs.templates_inputs is not None and inputs.templates_inputs.resolve_sources_phase.enabled)
	if phase == "templates_analyzers":
		return bool(inputs.templates_inputs is not None and inputs.templates_inputs.phases.analyzers.enabled)
	if phase == "templates_extract_partial_templates":
		return bool(
			inputs.templates_inputs is not None
			and inputs.templates_inputs.phases.extract_partial_templates.enabled
		)
	if phase == "templates_build_templates":
		return bool(inputs.templates_inputs is not None and inputs.templates_inputs.phases.build_templates.enabled)
	if phase == "templates_compute_template_similarity":
		return bool(inputs.templates_inputs is not None and inputs.templates_inputs.phases.compute_template_similarity.enabled)
	if phase == "templates_plot_templates":
		return bool(inputs.templates_inputs is not None and inputs.templates_inputs.phases.plot_templates.enabled)
	if phase == "templates_plot_templates_v2":
		phase_cfg = None if inputs.templates_inputs is None else getattr(inputs.templates_inputs.phases, "plot_templates_v2", None)
		return bool(False if phase_cfg is None else phase_cfg.enabled)
	if phase == "templates_report_templates":
		return bool(inputs.templates_inputs is not None and inputs.templates_inputs.phases.report_templates.enabled)
	if phase == "templates_reports":
		return bool(inputs.templates_inputs is not None and inputs.templates_inputs.phases.reports.enabled)
	if phase == "generate_gtrs":
		return bool(inputs.phases.generate_gtrs.enabled)
	if phase == "plot_recons":
		return bool(inputs.phases.plot_recons.enabled)
	if phase == "plot_branch_propagations":
		return bool(inputs.phases.plot_branch_propagations.enabled)
	if phase == "plot_branch_velocities":
		return bool(inputs.phases.plot_branch_velocities.enabled)
	if phase == "plot_unit_summary":
		return bool(inputs.phases.plot_unit_summary.enabled)
	if phase == "report_recons":
		return bool(inputs.phases.report_recons.enabled)
	if phase == "report_recon_grid":
		return bool(inputs.phases.report_recon_grid.enabled)
	if phase == "report_full_chip_layout":
		return bool(inputs.phases.report_full_chip_layout.enabled)
	if phase == "report_summaries":
		return bool(inputs.phases.report_summaries.enabled)
	if phase == "clear_templates_cache":
		return bool(inputs.phases.clear_templates_cache.enabled)
	return False


def _reconstruct_phase_selected(inputs: ReconstructionInputs, phase_name: str) -> bool:
	sequence = tuple(_normalize_reconstruct_stage_phase_name(phase) for phase in (inputs.phase_sequence or ()))
	return (not sequence) or _normalize_reconstruct_stage_phase_name(phase_name) in sequence


def _reconstruct_stage_phase_resource_class(inputs: ReconstructionInputs, phase_name: str) -> str | None:
	def _resource_class(value: Any) -> str | None:
		return getattr(value, "resource_class", None)

	phase = _normalize_reconstruct_stage_phase_name(phase_name)
	if phase == "templates_resolve_sources":
		return None if inputs.templates_inputs is None else _resource_class(inputs.templates_inputs.resolve_sources_phase)
	if phase == "templates_analyzers":
		return None if inputs.templates_inputs is None else _resource_class(inputs.templates_inputs.phases.analyzers)
	if phase == "templates_extract_partial_templates":
		return (
			None
			if inputs.templates_inputs is None
			else _resource_class(inputs.templates_inputs.phases.extract_partial_templates)
		)
	if phase == "templates_build_templates":
		return None if inputs.templates_inputs is None else _resource_class(inputs.templates_inputs.phases.build_templates)
	if phase == "templates_compute_template_similarity":
		return (
			None
			if inputs.templates_inputs is None
			else _resource_class(inputs.templates_inputs.phases.compute_template_similarity)
		)
	if phase == "templates_plot_templates":
		return None if inputs.templates_inputs is None else _resource_class(inputs.templates_inputs.phases.plot_templates)
	if phase == "templates_plot_templates_v2":
		phase_cfg = None if inputs.templates_inputs is None else getattr(inputs.templates_inputs.phases, "plot_templates_v2", None)
		return _resource_class(phase_cfg)
	if phase == "templates_report_templates":
		return None if inputs.templates_inputs is None else _resource_class(inputs.templates_inputs.phases.report_templates)
	if phase == "templates_reports":
		return None if inputs.templates_inputs is None else _resource_class(inputs.templates_inputs.phases.reports)
	if phase == "generate_gtrs":
		return _resource_class(inputs.phases.generate_gtrs)
	if phase == "plot_recons":
		return _resource_class(inputs.phases.plot_recons)
	if phase == "plot_branch_propagations":
		return _resource_class(inputs.phases.plot_branch_propagations)
	if phase == "plot_branch_velocities":
		return _resource_class(inputs.phases.plot_branch_velocities)
	if phase == "plot_unit_summary":
		return _resource_class(inputs.phases.plot_unit_summary)
	if phase == "report_recons":
		return _resource_class(inputs.phases.report_recons)
	if phase == "report_recon_grid":
		return _resource_class(inputs.phases.report_recon_grid)
	if phase == "report_full_chip_layout":
		return _resource_class(inputs.phases.report_full_chip_layout)
	if phase == "report_summaries":
		return _resource_class(inputs.phases.report_summaries)
	if phase == "clear_templates_cache":
		return _resource_class(inputs.phases.clear_templates_cache)
	return None


def _display_reconstruct_stage_phase_name(phase_name: str) -> str:
	phase = _normalize_reconstruct_stage_phase_name(phase_name)
	aliases = {
		"templates_resolve_sources": "resolve_sources",
		"templates_analyzers": "analyzers",
		"templates_extract_partial_templates": "extract_partial_templates",
		"templates_build_templates": "build_templates",
		"templates_compute_template_similarity": "compute_template_similarity",
		"templates_plot_templates": "plot_templates",
		"templates_plot_templates_v2": "plot_templates_v2",
		"templates_report_templates": "report_templates",
		"templates_reports": "reports",
	}
	return str(aliases.get(phase, phase))


def _reconstruct_phase_worker_allocation(
	inputs: ReconstructionInputs,
	phase_name: str,
) -> tuple[int, str, str | None]:
	resource_class = _reconstruct_stage_phase_resource_class(inputs, phase_name)
	workers, source = current_phase_worker_allocation(
		resource_class=resource_class,
		fallback_workers=max(1, int(inputs.n_jobs)),
	)
	return max(1, int(workers)), str(source), resource_class


def _reconstruct_inputs_for_phase_workers(
	inputs: ReconstructionInputs,
	phase_name: str,
) -> tuple[ReconstructionInputs, int, str, str | None]:
	workers, source, resource_class = _reconstruct_phase_worker_allocation(inputs, phase_name)
	phase_inputs = replace(inputs, n_jobs=int(workers))
	if _normalize_reconstruct_stage_phase_name(phase_name).startswith("templates_") and inputs.templates_inputs is not None:
		phase_inputs = replace(phase_inputs, templates_inputs=replace(inputs.templates_inputs, n_jobs=int(workers)))
	return phase_inputs, int(workers), str(source), resource_class


def _reconstruct_stage_phase_runner(phase_name: str):
	phase = _normalize_reconstruct_stage_phase_name(phase_name)
	if phase == "templates_resolve_sources":
		return run_reconstruct_templates_resolve_sources_phase
	if phase == "templates_analyzers":
		return run_reconstruct_templates_analyzers_phase
	if phase == "templates_extract_partial_templates":
		return run_reconstruct_templates_extract_partial_templates_phase
	if phase == "templates_build_templates":
		return run_reconstruct_templates_build_templates_phase
	if phase == "templates_compute_template_similarity":
		return run_reconstruct_templates_compute_template_similarity_phase
	if phase == "templates_plot_templates":
		return run_reconstruct_templates_plot_templates_phase
	if phase == "templates_plot_templates_v2":
		return run_reconstruct_templates_plot_templates_v2_phase
	if phase == "templates_report_templates":
		return run_reconstruct_templates_report_templates_phase
	if phase == "templates_reports":
		return run_reconstruct_templates_reports_phase
	if phase == "generate_gtrs":
		return run_reconstruct_generate_gtrs_phase
	if phase == "plot_recons":
		return run_reconstruct_plot_recons_phase
	if phase == "plot_branch_propagations":
		return run_reconstruct_plot_branch_propagations_phase
	if phase == "plot_branch_velocities":
		return run_reconstruct_plot_branch_velocities_phase
	if phase == "plot_unit_summary":
		return run_reconstruct_plot_unit_summary_phase
	if phase == "report_recons":
		return run_reconstruct_report_recons_phase
	if phase == "report_recon_grid":
		return run_reconstruct_report_recon_grid_phase
	if phase == "report_full_chip_layout":
		return run_reconstruct_report_full_chip_layout_phase
	if phase == "report_summaries":
		return run_reconstruct_report_summaries_phase
	if phase == "clear_templates_cache":
		return run_reconstruct_clear_templates_cache_phase
	raise ValueError(f"Unknown reconstruct phase: {phase_name!r}")


def collect_reconstruct_result_from_outputs(inputs: ReconstructionInputs) -> ReconstructionResult:
	env = _prepare_reconstruct_phase_environment(inputs=inputs, clear_output_root=False)
	unit_results = _load_reconstruct_unit_results(
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
		unit_ids=env.unit_ids,
	)
	units_ok, units_error = _count_unit_statuses(unit_results)
	report_sort_by = normalize_grid_sort_by(inputs.report_sort_by, default="unit_id")
	stage_outputs = _collect_existing_reconstruct_stage_outputs(
		reconstruction_out_dir=env.reconstruction_out_dir,
		inputs=inputs,
	)
	failed_units_summary_json = _current_failed_units_summary_json(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
	)
	summary_json = env.reconstruction_out_dir / "reconstruction_summary.json"
	write_json(
		summary_json,
		{
			"h5_path": str(inputs.h5_path),
			"stream_id": str(inputs.stream_id),
			"n_jobs": int(max(1, int(inputs.n_jobs))),
			"phase_sequence": list(inputs.phase_sequence or []),
			"well_out_dir": str(env.well_out_dir),
			"reconstruction_out_dir": str(env.reconstruction_out_dir),
			"applied_debug_limits": _reconstruct_applied_debug_limits(inputs),
			"units_ok": units_ok,
			"units_error": units_error,
			"outputs": stage_outputs,
			"cleanup_failed_unit_outputs": bool(inputs.cleanup_failed_unit_outputs),
			"failed_units_summary_json": str(failed_units_summary_json) if failed_units_summary_json else None,
			"reports_overwrite_skipped": env.preserve_stage_reports,
			"report_sort_by": str(report_sort_by),
			"units": _unit_rows(unit_results),
		},
	)
	return ReconstructionResult(
		well_out_dir=env.well_out_dir,
		reconstruction_out_dir=env.reconstruction_out_dir,
		summary_json=summary_json,
		units=unit_results,
	)


def run_reconstruct_stage(inputs: ReconstructionInputs) -> ReconstructionResult:
	phase_sequence = tuple(
		_normalize_reconstruct_stage_phase_name(phase)
		for phase in (inputs.phase_sequence or DEFAULT_INTERNAL_RECONSTRUCTION_PHASE_SEQUENCE)
	)
	phase_plan = [phase for phase in phase_sequence if _reconstruct_stage_phase_enabled(inputs, phase)]
	if not phase_plan:
		return collect_reconstruct_result_from_outputs(inputs)
	LOGGER.info(
		"reconstruct stage start: stream_id=%s phases=%s applied_debug_limits=%s",
		str(inputs.stream_id),
		[_display_reconstruct_stage_phase_name(phase) for phase in phase_plan],
		_reconstruct_applied_debug_limits(inputs),
	)

	def _descriptor_for_phase(phase_name: str) -> PhaseDescriptor:
		def _run_phase(phase_name: str = phase_name):
			phase_inputs, workers, source, resource_class = _reconstruct_inputs_for_phase_workers(inputs, phase_name)
			LOGGER.info(
				"reconstruct phase worker allocation: phase=%s resource_class=%s n_jobs=%d n_jobs_source=%s",
				_display_reconstruct_stage_phase_name(phase_name),
				str(resource_class or "none"),
				int(workers),
				str(source),
			)
			return _reconstruct_stage_phase_runner(phase_name)(phase_inputs)

		workers, _source, resource_class = _reconstruct_phase_worker_allocation(inputs, phase_name)

		return PhaseDescriptor(
			name=_display_reconstruct_stage_phase_name(phase_name),
			runner=_run_phase,
			resource_class=resource_class,
			pipeline_thread_count=int(workers),
		)

	run_phase_chain(
		phases=[_descriptor_for_phase(phase) for phase in phase_plan],
		logger=LOGGER,
		target_label=str(inputs.stream_id),
		resource_key_context=inputs,
	)
	return collect_reconstruct_result_from_outputs(inputs)


def _run_reconstruct_stage_default_order(inputs: ReconstructionInputs) -> ReconstructionResult:
	env = _prepare_reconstruct_phase_environment(inputs=inputs, clear_output_root=True)
	unit_results: list[UnitReconstructionResult] = []
	failed_units_summary_json: Path | None = _current_failed_units_summary_json(
		inputs=inputs,
		reconstruction_out_dir=env.reconstruction_out_dir,
	)

	if _reconstruct_phase_selected(inputs, "generate_gtrs") and bool(inputs.phases.generate_gtrs.enabled):
		unit_results, failed_units_summary_json = _run_reconstruct_generate_gtrs_phase_impl(inputs=inputs, env=env)
		_write_reconstruct_phase_summary(
			phase_name="generate_gtrs",
			summary_json=env.reconstruction_out_dir / Path(str(inputs.phases.generate_gtrs.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=unit_results,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=env.preserve_stage_reports,
		)
	else:
		unit_results = _load_reconstruct_unit_results(
			reconstruction_out_dir=env.reconstruction_out_dir,
			inputs=inputs,
			unit_ids=env.unit_ids,
		)

	if _reconstruct_phase_selected(inputs, "plot_recons") and bool(inputs.phases.plot_recons.enabled):
		unit_results, failed_units_summary_json = _run_reconstruct_plot_recons_phase_impl(inputs=inputs, env=env)
		_write_reconstruct_phase_summary(
			phase_name="plot_recons",
			summary_json=env.reconstruction_out_dir / Path(str(inputs.phases.plot_recons.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=unit_results,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=env.preserve_stage_reports,
		)
	elif not unit_results:
		unit_results = _load_reconstruct_unit_results(
			reconstruction_out_dir=env.reconstruction_out_dir,
			inputs=inputs,
			unit_ids=env.unit_ids,
		)

	if _reconstruct_phase_selected(inputs, "plot_branch_propagations") and bool(inputs.phases.plot_branch_propagations.enabled):
		unit_results, failed_units_summary_json = _run_reconstruct_plot_branch_propagations_phase_impl(inputs=inputs, env=env)
		_write_reconstruct_phase_summary(
			phase_name="plot_branch_propagations",
			summary_json=env.reconstruction_out_dir
			/ Path(str(inputs.phases.plot_branch_propagations.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=unit_results,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=env.preserve_stage_reports,
		)
	elif not unit_results:
		unit_results = _load_reconstruct_unit_results(
			reconstruction_out_dir=env.reconstruction_out_dir,
			inputs=inputs,
			unit_ids=env.unit_ids,
		)

	if _reconstruct_phase_selected(inputs, "plot_branch_velocities") and bool(inputs.phases.plot_branch_velocities.enabled):
		unit_results, failed_units_summary_json = _run_reconstruct_plot_branch_velocities_phase_impl(inputs=inputs, env=env)
		_write_reconstruct_phase_summary(
			phase_name="plot_branch_velocities",
			summary_json=env.reconstruction_out_dir
			/ Path(str(inputs.phases.plot_branch_velocities.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=unit_results,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=env.preserve_stage_reports,
		)
	elif not unit_results:
		unit_results = _load_reconstruct_unit_results(
			reconstruction_out_dir=env.reconstruction_out_dir,
			inputs=inputs,
			unit_ids=env.unit_ids,
		)

	if _reconstruct_phase_selected(inputs, "plot_unit_summary") and bool(inputs.phases.plot_unit_summary.enabled):
		unit_results, failed_units_summary_json = _run_reconstruct_plot_unit_summary_phase_impl(inputs=inputs, env=env)
		_write_reconstruct_phase_summary(
			phase_name="plot_unit_summary",
			summary_json=env.reconstruction_out_dir
			/ Path(str(inputs.phases.plot_unit_summary.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=unit_results,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=env.preserve_stage_reports,
		)
	elif not unit_results:
		unit_results = _load_reconstruct_unit_results(
			reconstruction_out_dir=env.reconstruction_out_dir,
			inputs=inputs,
			unit_ids=env.unit_ids,
		)

	if not unit_results:
		unit_results = _load_reconstruct_unit_results(
			reconstruction_out_dir=env.reconstruction_out_dir,
			inputs=inputs,
			unit_ids=env.unit_ids,
		)

	report_sort_by = normalize_grid_sort_by(inputs.report_sort_by, default="unit_id")
	stage_outputs: dict[str, str] = dict(env.existing_stage_outputs)
	if _reconstruct_phase_selected(inputs, "report_recons") and bool(inputs.phases.report_recons.enabled):
		stage_outputs = _run_reconstruct_report_recons_phase_impl(inputs=inputs, env=env, unit_results=unit_results)
		_write_reconstruct_phase_summary(
			phase_name="report_recons",
			summary_json=env.reconstruction_out_dir / Path(str(inputs.phases.report_recons.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=unit_results,
			stage_outputs=stage_outputs,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=env.preserve_stage_reports,
			extra_fields={"report_sort_by": str(report_sort_by)},
		)

	if _reconstruct_phase_selected(inputs, "report_recon_grid") and bool(inputs.phases.report_recon_grid.enabled):
		stage_outputs = _run_reconstruct_report_recon_grid_phase_impl(
			inputs=inputs,
			env=env,
			unit_results=unit_results,
			stage_outputs=stage_outputs,
		)
		_write_reconstruct_phase_summary(
			phase_name="report_recon_grid",
			summary_json=env.reconstruction_out_dir
			/ Path(str(inputs.phases.report_recon_grid.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=unit_results,
			stage_outputs=stage_outputs,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=env.preserve_stage_reports,
			extra_fields={"report_sort_by": str(report_sort_by)},
		)

	if _reconstruct_phase_selected(inputs, "report_full_chip_layout") and bool(inputs.phases.report_full_chip_layout.enabled):
		full_chip_unit_results = _load_full_chip_layout_unit_results(
			reconstruction_out_dir=env.reconstruction_out_dir,
			inputs=inputs,
			merged_units_dir=env.merged_units_dir,
			unit_ids=env.unit_ids,
		)
		full_chip_outputs = _run_reconstruct_report_full_chip_layout_phase_impl(
			inputs=inputs,
			env=env,
			unit_results=full_chip_unit_results,
		)
		stage_outputs.update(full_chip_outputs)
		_write_reconstruct_phase_summary(
			phase_name="report_full_chip_layout",
			summary_json=env.reconstruction_out_dir
			/ Path(str(inputs.phases.report_full_chip_layout.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=full_chip_unit_results,
			stage_outputs=stage_outputs,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=False,
		)

	if _reconstruct_phase_selected(inputs, "report_summaries") and bool(inputs.phases.report_summaries.enabled):
		stage_outputs = _run_reconstruct_report_summaries_phase_impl(
			inputs=inputs,
			env=env,
			unit_results=unit_results,
			stage_outputs=stage_outputs,
		)
		_write_reconstruct_phase_summary(
			phase_name="report_summaries",
			summary_json=env.reconstruction_out_dir
			/ Path(str(inputs.phases.report_summaries.summary_json_relpath)).expanduser(),
			inputs=inputs,
			well_out_dir=env.well_out_dir,
			reconstruction_out_dir=env.reconstruction_out_dir,
			unit_results=unit_results,
			stage_outputs=stage_outputs,
			failed_units_summary_json=failed_units_summary_json,
			preserve_stage_reports=env.preserve_stage_reports,
			extra_fields={"report_sort_by": str(report_sort_by)},
		)

	units_ok, units_error = _count_unit_statuses(unit_results)
	summary_json = env.reconstruction_out_dir / "reconstruction_summary.json"
	summary_payload = {
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"n_jobs": int(max(1, int(inputs.n_jobs))),
		"phase_sequence": list(inputs.phase_sequence or []),
		"well_out_dir": str(env.well_out_dir),
		"reconstruction_out_dir": str(env.reconstruction_out_dir),
		"applied_debug_limits": _reconstruct_applied_debug_limits(inputs),
		"units_ok": units_ok,
		"units_error": units_error,
		"outputs": stage_outputs,
		"cleanup_failed_unit_outputs": bool(inputs.cleanup_failed_unit_outputs),
		"failed_units_summary_json": str(failed_units_summary_json) if failed_units_summary_json else None,
		"reports_overwrite_skipped": env.preserve_stage_reports,
		"report_sort_by": str(report_sort_by),
		"units": _unit_rows(unit_results),
	}
	write_json(summary_json, summary_payload)

	return ReconstructionResult(
		well_out_dir=env.well_out_dir,
		reconstruction_out_dir=env.reconstruction_out_dir,
		summary_json=summary_json,
		units=unit_results,
	)
