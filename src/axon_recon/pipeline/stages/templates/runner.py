from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir #TODO: dont import this from v1

from .integrations import materialize_templates_from_spikeinterface
from .core.render import (
	render_footprint_amplitude_map,
	render_footprint_map_grid,
	render_footprint_latency_map,
	render_footprint_peak_latency_map,
	render_multi_source_pdf,
	render_propagation_plot,
	render_template_plot,
	render_template_wf_overlay,
	render_topographical_amplitude_footprint,
	render_topographical_latency_footprint,
	render_wf_overlay_grid,
)
from .io import read_json, resolve_report_output_paths, resolve_unit_output_paths, write_json
from .models.inputs import TemplatesInputs
from .models.results import TemplatesResult, UnitTemplatesResult


LOGGER = logging.getLogger("axon_recon.templates")


def _pad_value_from_mode(mode: str) -> float:
	m = str(mode or "zero").strip().lower()
	if m == "one":
		return 1.0
	if m == "nan":
		return float("nan")
	return 0.0


def _build_square_template(template_c_by_t: np.ndarray, *, padding_mode: str) -> np.ndarray:
	t = np.asarray(template_c_by_t, dtype=float)
	n_channels = int(t.shape[0])
	target = int(np.ceil(np.sqrt(max(1, n_channels))) ** 2)
	if target <= n_channels:
		return t
	pad_val = _pad_value_from_mode(padding_mode)
	out = np.full((target, int(t.shape[1])), pad_val, dtype=float)
	out[:n_channels, :] = t
	return out


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


def _discover_unit_ids_from_unit_summaries(templates_out_dir: Path) -> list[Any]:
	unit_ids: list[Any] = []
	for p in sorted(templates_out_dir.glob("units/**/unit_templates_summary.json")):
		if not p.is_file():
			continue
		uid: Any | None = None
		try:
			payload = read_json(p)
			if isinstance(payload, dict):
				uid = payload.get("unit_id")
		except Exception:
			uid = None
		if uid is None:
			try:
				uid = int(p.parent.name)
			except Exception:
				uid = p.parent.name
		unit_ids.append(uid)
	return unit_ids


def _resolve_templates_dirs(*, well_out_dir: Path, templates_out_dir: Path) -> tuple[Path, Path]:
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

	return merged_units_dir, full_channels_templates_dir


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


def _normalize_compare_token(value: Any) -> str:
	return str(value).strip()


def _load_curated_units_from_spikesorting(well_out_dir: Path) -> list[Any] | None:
	qm_xlsx = well_out_dir / "stg2_spikesorting_outputs" / "qm_unfiltered.xlsx"
	if not qm_xlsx.exists():
		return None
	try:
		import pandas as pd  # type: ignore[import-not-found]

		df = pd.read_excel(qm_xlsx, index_col=0)
		return list(df.index.values)
	except Exception:
		return None


def _apply_curated_filter(unit_ids: list[Any], curated_units: list[Any]) -> list[Any]:
	if not curated_units:
		return list(unit_ids)
	curated_tokens = {_normalize_compare_token(x) for x in curated_units}
	return [uid for uid in unit_ids if _normalize_compare_token(uid) in curated_tokens]


def _load_unit_result_from_summary(*, unit_id: Any, unit_summary_json: Path) -> UnitTemplatesResult | None:
	if not unit_summary_json.exists():
		return None
	try:
		existing = read_json(unit_summary_json)
		outputs = dict((existing or {}).get("outputs", {})) if isinstance(existing, dict) else {}
		return UnitTemplatesResult(
			unit_id=unit_id,
			status=str((existing or {}).get("status", "ok")),
			outputs={str(k): str(v) for k, v in outputs.items() if v is not None},
			error=(None if not isinstance(existing, dict) else existing.get("error")),
		)
	except Exception:
		return None


def run_templates_stage(inputs: TemplatesInputs) -> TemplatesResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	templates_out_dir = well_out_dir / str(inputs.output_rel_root)
	templates_out_dir.mkdir(parents=True, exist_ok=True)

	merged_units_dir: Path | None = None
	full_channels_templates_dir: Path | None = None

	def _ensure_templates_dirs() -> tuple[Path, Path]:
		nonlocal merged_units_dir, full_channels_templates_dir
		if merged_units_dir is None or full_channels_templates_dir is None:
			prefer_spikeinterface = bool(inputs.force_restart) and (not bool(inputs.reports.replot_from_disk))
			if prefer_spikeinterface:
				try:
					LOGGER.info(
						"force_restart enabled; materializing templates from SpikeInterface "
						"(include_concat=%s, include_segments=%s)",
						bool(inputs.include_concat),
						bool(inputs.include_segments),
					)
					merged_units_dir_resolved, full_channels_templates_dir_resolved = materialize_templates_from_spikeinterface(
						well_out_dir=well_out_dir,
						templates_out_dir=templates_out_dir,
						unit_ids=(list(inputs.unit_ids) if inputs.unit_ids is not None else None),
						include_concat=bool(inputs.include_concat),
						include_segments=bool(inputs.include_segments),
						enable_merge=bool(inputs.merge.enable),
						merge_method=str(inputs.merge.method),
						centering_method=str(inputs.merge.centering_method),
						max_waveforms_per_source_channel=int(max(1, int(inputs.merge.max_waveforms_per_source_channel))),
						overlap_match_priority=tuple(inputs.merge.overlap_match_priority),
						location_tolerance_um=float(inputs.merge.location_tolerance_um),
					)
					merged_units_dir = merged_units_dir_resolved
					full_channels_templates_dir = full_channels_templates_dir_resolved
					return merged_units_dir, full_channels_templates_dir
				except Exception:
					LOGGER.exception(
						"SpikeInterface materialization failed during force_restart; falling back to persisted templates artifacts"
					)

			try:
				merged_units_dir_resolved, full_channels_templates_dir_resolved = _resolve_templates_dirs(
					well_out_dir=well_out_dir,
					templates_out_dir=templates_out_dir,
				)
			except FileNotFoundError:
				LOGGER.info(
					"Missing templates artifacts under %s; attempting SpikeInterface materialization "
					"(include_concat=%s, include_segments=%s)",
					templates_out_dir / "templates",
					bool(inputs.include_concat),
					bool(inputs.include_segments),
				)
				merged_units_dir_resolved, full_channels_templates_dir_resolved = materialize_templates_from_spikeinterface(
					well_out_dir=well_out_dir,
					templates_out_dir=templates_out_dir,
					unit_ids=(list(inputs.unit_ids) if inputs.unit_ids is not None else None),
					include_concat=bool(inputs.include_concat),
					include_segments=bool(inputs.include_segments),
					enable_merge=bool(inputs.merge.enable),
					merge_method=str(inputs.merge.method),
					centering_method=str(inputs.merge.centering_method),
					max_waveforms_per_source_channel=int(max(1, int(inputs.merge.max_waveforms_per_source_channel))),
					overlap_match_priority=tuple(inputs.merge.overlap_match_priority),
					location_tolerance_um=float(inputs.merge.location_tolerance_um),
				)
			merged_units_dir = merged_units_dir_resolved
			full_channels_templates_dir = full_channels_templates_dir_resolved
		return merged_units_dir, full_channels_templates_dir

	if bool(inputs.reports.replot_from_disk) and (not bool(inputs.force_restart)):
		if inputs.unit_ids is not None:
			unit_ids = list(inputs.unit_ids)
		else:
			unit_ids = _discover_unit_ids_from_unit_summaries(templates_out_dir)
		if inputs.unit_limit is not None:
			unit_ids = unit_ids[: int(inputs.unit_limit)]
	else:
		merged_units_dir_resolved, _ = _ensure_templates_dirs()
		unit_ids = _build_unit_ids(inputs, merged_units_dir_resolved)
	if bool(inputs.require_curated_units) and inputs.unit_ids is None:
		curated = _load_curated_units_from_spikesorting(well_out_dir)
		if curated is None:
			raise RuntimeError(
				"Templates stage requires curated units, but curated units file was not found/readable at "
				f"{well_out_dir / 'stg2_spikesorting_outputs' / 'qm_unfiltered.xlsx'}"
			)
		unit_ids = _apply_curated_filter(unit_ids, curated)

	def _process_unit(unit_id: Any) -> UnitTemplatesResult:
		paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		paths["unit_dir"].mkdir(parents=True, exist_ok=True)

		if (not bool(inputs.force_restart)) and (not bool(inputs.force_replot_per_unit)):
			existing_result = _load_unit_result_from_summary(
				unit_id=unit_id,
				unit_summary_json=paths["unit_summary_json"],
			)
			if existing_result is not None:
				return existing_result

		unit_summary: dict[str, Any] = {
			"unit_id": unit_id,
			"status": "ok",
			"error": None,
			"outputs": {},
		}
		try:
			merged_units_dir_resolved, full_channels_templates_dir_resolved = _ensure_templates_dirs()
			merged_dir = merged_units_dir_resolved / f"unit_{unit_id}"
			full_dir = full_channels_templates_dir_resolved / f"unit_{unit_id}"

			merged_template, merged_locs = _load_merged_unit(merged_dir)
			full_payload = _load_full_unit(full_dir) if full_channels_templates_dir_resolved.exists() else None

			if bool(inputs.per_unit_outputs.merged_template.write_npy):
				paths["merged_template_npy"].parent.mkdir(parents=True, exist_ok=True)
				np.save(paths["merged_template_npy"], merged_template)
				unit_summary["outputs"]["merged_template_npy"] = str(paths["merged_template_npy"])

			if bool(inputs.per_unit_outputs.full_template.write_npy):
				full_template_to_write = merged_template if full_payload is None else full_payload[0]
				paths["full_template_npy"].parent.mkdir(parents=True, exist_ok=True)
				np.save(paths["full_template_npy"], full_template_to_write)
				unit_summary["outputs"]["full_template_npy"] = str(paths["full_template_npy"])

			if bool(inputs.per_unit_outputs.scan_template.write_npy):
				scan_template_to_write = merged_template if full_payload is None else full_payload[0]
				paths["scan_template_npy"].parent.mkdir(parents=True, exist_ok=True)
				np.save(paths["scan_template_npy"], scan_template_to_write)
				unit_summary["outputs"]["scan_template_npy"] = str(paths["scan_template_npy"])

			if bool(inputs.per_unit_outputs.square_template.write_npy):
				sq = _build_square_template(
					merged_template,
					padding_mode=str(inputs.per_unit_outputs.square_template.padding_value),
				)
				paths["square_template_npy"].parent.mkdir(parents=True, exist_ok=True)
				np.save(paths["square_template_npy"], sq)
				unit_summary["outputs"]["square_template_npy"] = str(paths["square_template_npy"])

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

			overlay_outputs = render_template_wf_overlay(
				template=template_plot,
				config=inputs.per_unit_outputs.template_wf_overlay,
				time_upsample=inputs.reports.time_upsample,
				pdf_path=paths["template_wf_overlay_pdf"],
				png_path=paths["template_wf_overlay_png"],
			)
			unit_summary["outputs"].update(overlay_outputs)

			amp_outputs = render_footprint_amplitude_map(
				template=template_plot,
				locations_xy=locs_plot,
				config=inputs.per_unit_outputs.footprint_plots.amplitude_map,
				png_path=paths["footprint_amplitude_map_png"],
				svg_path=paths["footprint_amplitude_map_svg"],
			)
			unit_summary["outputs"].update(amp_outputs)

			peak_lat_outputs = render_footprint_peak_latency_map(
				template=template_plot,
				locations_xy=locs_plot,
				config=inputs.per_unit_outputs.footprint_plots.peak_latency_map,
				png_path=paths["footprint_peak_latency_map_png"],
				svg_path=paths["footprint_peak_latency_map_svg"],
			)
			unit_summary["outputs"].update(peak_lat_outputs)

			lat_outputs = render_footprint_latency_map(
				template=template_plot,
				locations_xy=locs_plot,
				config=inputs.per_unit_outputs.footprint_plots.latency_map,
				png_path=paths["footprint_latency_map_png"],
				svg_path=paths["footprint_latency_map_svg"],
			)
			unit_summary["outputs"].update(lat_outputs)

			topo_amp_outputs = render_topographical_amplitude_footprint(
				template=template_plot,
				locations_xy=locs_plot,
				config=inputs.per_unit_outputs.topographical_footprints.amplitude,
				png_path=paths["topographical_amplitude_footprint_png"],
				svg_path=paths["topographical_amplitude_footprint_svg"],
			)
			unit_summary["outputs"].update(topo_amp_outputs)

			topo_lat_outputs = render_topographical_latency_footprint(
				template=template_plot,
				locations_xy=locs_plot,
				config=inputs.per_unit_outputs.topographical_footprints.latency,
				png_path=paths["topographical_latency_footprint_png"],
				svg_path=paths["topographical_latency_footprint_svg"],
			)
			unit_summary["outputs"].update(topo_lat_outputs)

			prop_outputs = render_propagation_plot(
				template=template_plot,
				locations_xy=locs_plot,
				config=inputs.per_unit_outputs.propagation_plots,
				pdf_path=paths["propagation_plot_pdf"],
				png_path=paths["propagation_plot_png"],
			)
			unit_summary["outputs"].update(prop_outputs)
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
	units_to_process = list(unit_ids)
	if bool(inputs.reports.replot_from_disk) and (not bool(inputs.force_restart)):
		units_to_process = []
		for unit_id in unit_ids:
			paths = resolve_unit_output_paths(
				templates_out_dir=templates_out_dir,
				unit_id=unit_id,
				per_unit_outputs=inputs.per_unit_outputs,
			)
			existing_result = _load_unit_result_from_summary(
				unit_id=unit_id,
				unit_summary_json=paths["unit_summary_json"],
			)
			if existing_result is not None:
				unit_results.append(existing_result)
			else:
				units_to_process.append(unit_id)
		if units_to_process:
			LOGGER.info(
				"Templates reports replot_from_disk: %d unit summaries missing; processing those units from templates artifacts",
				len(units_to_process),
			)

	worker_count = int(max(1, int(inputs.n_jobs)))
	if worker_count <= 1 or len(units_to_process) <= 1:
		for unit_id in units_to_process:
			unit_results.append(_process_unit(unit_id))
	else:
		futures: dict[concurrent.futures.Future[UnitTemplatesResult], Any] = {}
		with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
			for unit_id in units_to_process:
				fut = pool.submit(_process_unit, unit_id)
				futures[fut] = unit_id

			for fut in concurrent.futures.as_completed(futures):
				unit_results.append(fut.result())

	unit_results.sort(key=lambda r: str(r.unit_id))
	report_outputs: dict[str, str] = {}
	try:
		report_paths = resolve_report_output_paths(templates_out_dir=templates_out_dir, reports=inputs.reports)
		overlay_paths = [
			Path(u.outputs["template_wf_overlay_png"])
			for u in unit_results
			if "template_wf_overlay_png" in u.outputs
		]
		report_outputs.update(
			render_wf_overlay_grid(
				overlay_png_paths=overlay_paths,
				config=inputs.reports.wf_overlay_grid,
				pdf_path=report_paths["wf_overlay_grid_pdf"],
				png_path=report_paths["wf_overlay_grid_png"],
			)
		)
		amp_map_paths = [
			Path(u.outputs["footprint_amplitude_map_png"])
			for u in unit_results
			if "footprint_amplitude_map_png" in u.outputs
		]
		report_outputs.update(
			render_footprint_map_grid(
				image_paths=amp_map_paths,
				config=inputs.reports.footprint_grids.amplitude_map_grid,
				pdf_path=report_paths["footprint_amplitude_map_grid_pdf"],
				png_path=report_paths["footprint_amplitude_map_grid_png"],
				pdf_output_key="footprint_amplitude_map_grid_pdf",
				png_output_key="footprint_amplitude_map_grid_png",
				title="Template footprint amplitude map grid",
			)
		)
		lat_map_paths = [
			Path(u.outputs["footprint_latency_map_png"])
			for u in unit_results
			if "footprint_latency_map_png" in u.outputs
		]
		report_outputs.update(
			render_footprint_map_grid(
				image_paths=lat_map_paths,
				config=inputs.reports.footprint_grids.latency_map_grid,
				pdf_path=report_paths["footprint_latency_map_grid_pdf"],
				png_path=report_paths["footprint_latency_map_grid_png"],
				pdf_output_key="footprint_latency_map_grid_pdf",
				png_output_key="footprint_latency_map_grid_png",
				title="Template footprint latency map grid",
			)
		)
		if bool(inputs.reports.plot_multi_source_pdf.enabled):
			report_outputs.update(
				render_multi_source_pdf(
					units=[
						{"unit_id": u.unit_id, "outputs": u.outputs}
						for u in unit_results
						if str(u.status) == "ok"
					],
					pdf_path=report_paths["multi_source_pdf"],
				)
			)
	except Exception:
		LOGGER.exception("Failed writing templates reports for stream %s", inputs.stream_id)

	summary_json = templates_out_dir / "templates_summary.json"
	summary_payload = {
		"h5_path": str(inputs.h5_path),
		"stream_id": str(inputs.stream_id),
		"n_jobs": int(max(1, int(inputs.n_jobs))),
		"well_out_dir": str(well_out_dir),
		"templates_out_dir": str(templates_out_dir),
		"reports": report_outputs,
		"reports_replot_from_disk": bool(inputs.reports.replot_from_disk),
		"reports_time_upsample": {
			"enabled": bool(inputs.reports.time_upsample.enabled),
			"factor": int(max(1, int(inputs.reports.time_upsample.factor))),
			"method": str(inputs.reports.time_upsample.method),
		},
		"require_curated_units": bool(inputs.require_curated_units),
		"include_concat": bool(inputs.include_concat),
		"include_segments": bool(inputs.include_segments),
		"merge": {
			"enable": bool(inputs.merge.enable),
			"method": str(inputs.merge.method),
			"centering_method": str(inputs.merge.centering_method),
			"weighting_mode": str(inputs.merge.weighting_mode),
			"max_waveforms_per_source_channel": int(max(1, int(inputs.merge.max_waveforms_per_source_channel))),
			"overlap_match_priority": list(inputs.merge.overlap_match_priority),
			"location_tolerance_um": float(inputs.merge.location_tolerance_um),
		},
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
		report_outputs=report_outputs,
	)
