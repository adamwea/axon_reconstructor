from __future__ import annotations

import concurrent.futures
from dataclasses import replace
import logging
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir #TODO: dont import this from v1

from .core.merge import materialize_templates_from_spikeinterface
from .core.render import (
	render_footprint_amplitude_map,
	render_footprint_map_grid,
	render_footprint_latency_map,
	render_multi_source_pdf,
	render_propagation_plot,
	render_template_circles_plot,
	render_template_plot,
	render_template_wf_overlay,
	render_topographical_amplitude_footprint,
	render_topographical_latency_footprint,
	render_wf_overlay_grid,
)
from .io import (
	load_materialized_overlay_waveforms,
	load_materialized_merged_electrode_ids,
	read_json,
	resolve_report_output_paths,
	resolve_unit_output_paths,
	write_json,
)
from .models.inputs import ProbeGeometryConfig, TemplatesInputs, TimeUpsampleConfig
from .models.results import TemplatesResult, UnitTemplatesResult


LOGGER = logging.getLogger("axon_recon.templates")


def _as_positive_float_or_none(value: Any) -> float | None:
	try:
		parsed = float(value)
	except Exception:
		return None
	if not np.isfinite(parsed) or parsed <= 0.0:
		return None
	return float(parsed)


def _effective_probe_geometry_for_unit(
	*,
	base_probe_geometry: ProbeGeometryConfig | None,
	upsampling_decision: dict[str, Any] | None,
) -> ProbeGeometryConfig | None:
	if not isinstance(upsampling_decision, dict):
		return base_probe_geometry

	target_hz = _as_positive_float_or_none(upsampling_decision.get("target_hz", None))
	applied = bool(upsampling_decision.get("applied", False))
	analyzer_hz = _as_positive_float_or_none(upsampling_decision.get("analyzer_hz", None))
	raw_hz = _as_positive_float_or_none(upsampling_decision.get("raw_hz", None))

	effective_hz: float | None = None
	if applied and target_hz is not None:
		effective_hz = float(target_hz)
	elif analyzer_hz is not None:
		effective_hz = float(analyzer_hz)
	elif raw_hz is not None:
		effective_hz = float(raw_hz)

	if effective_hz is None:
		return base_probe_geometry

	if base_probe_geometry is None:
		return ProbeGeometryConfig(sampling_rate_hz=float(effective_hz))

	current_hz = _as_positive_float_or_none(base_probe_geometry.sampling_rate_hz)
	if current_hz is not None and abs(float(current_hz) - float(effective_hz)) <= 1e-9:
		return base_probe_geometry

	return replace(base_probe_geometry, sampling_rate_hz=float(effective_hz))


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


def _build_square_locations(locations_xy: np.ndarray, *, target_channels: int) -> np.ndarray:
	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Unexpected locations shape for square template: {locs.shape}")
	if target_channels <= int(locs.shape[0]):
		return np.asarray(locs[:target_channels, :2], dtype=float)
	out = np.full((target_channels, 2), np.nan, dtype=float)
	out[: int(locs.shape[0]), :] = locs[:, :2]
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


def _select_template_for_shape(
	*,
	merged_template: np.ndarray,
	merged_locs: np.ndarray,
	full_payload: tuple[np.ndarray, np.ndarray] | None,
	template_shape: str,
) -> tuple[np.ndarray, np.ndarray, str]:
	shape = str(template_shape or "square").strip().lower().replace("-", "_").replace(" ", "_")
	if shape in {"full", "scan"} and full_payload is not None:
		full_template, full_locs = full_payload
		return full_template, full_locs, "full_channels"
	# `square` and unknown values use merged contributing channels.
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


def _load_persisted_upsampling_decision(*, unit_summary_json: Path) -> dict[str, Any] | None:
	if not unit_summary_json.exists():
		return None
	try:
		existing = read_json(unit_summary_json)
		if not isinstance(existing, dict):
			return None
		upsampling = existing.get("upsampling", None)
		if isinstance(upsampling, dict) and upsampling:
			return dict(upsampling)
	except Exception:
		return None
	return None


def _infer_replot_upsampling_decision(*, inputs: TemplatesInputs) -> dict[str, Any] | None:
	base_hz = None if inputs.probe_geometry is None else _as_positive_float_or_none(inputs.probe_geometry.sampling_rate_hz)
	if base_hz is None:
		return None
	if (not bool(inputs.execution_upsampling.enabled)) or int(inputs.execution_upsampling.factor) <= 1:
		return None
	target_hz = float(base_hz) * float(max(1, int(inputs.execution_upsampling.factor)))
	if target_hz <= float(base_hz):
		return None
	return {
		"enabled": True,
		"method": str(inputs.execution_upsampling.method),
		"factor": int(max(1, int(inputs.execution_upsampling.factor))),
		"raw_hz": float(base_hz),
		"analyzer_hz": float(base_hz),
		"target_hz": float(target_hz),
		"applied": True,
		"skip_reason": None,
		"rate_source": "replot_inferred_from_execution",
	}


def run_templates_stage(inputs: TemplatesInputs) -> TemplatesResult:
	LOGGER.info(
		"Templates stage start: stream=%s force_restart=%s force_replot=%s force_replot_per_unit=%s reports_replot_from_disk=%s n_jobs=%d",
		str(inputs.stream_id),
		bool(inputs.force_restart),
		bool(inputs.force_replot),
		bool(inputs.force_replot_per_unit),
		bool(inputs.reports.replot_from_disk),
		int(max(1, int(inputs.n_jobs))),
	)
	LOGGER.info(
		"Templates stage merge settings: enable=%s method=%s centering=%s max_waveforms_per_source_channel=%s",
		bool(inputs.merge.enable),
		str(inputs.merge.method),
		str(inputs.merge.centering_method),
		("unlimited" if inputs.merge.max_waveforms_per_source_channel is None else str(int(inputs.merge.max_waveforms_per_source_channel))),
	)
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	templates_out_dir = well_out_dir / str(inputs.output_rel_root)
	templates_out_dir.mkdir(parents=True, exist_ok=True)

	merged_units_dir: Path | None = None
	full_channels_templates_dir: Path | None = None
	upsampling_decisions_by_unit: dict[Any, dict[str, Any]] = {}

	def _ensure_templates_dirs() -> tuple[Path, Path]:
		nonlocal merged_units_dir, full_channels_templates_dir, upsampling_decisions_by_unit
		if merged_units_dir is None or full_channels_templates_dir is None:
			overlay_debug_mode = bool(getattr(inputs.per_unit_outputs.template_wf_overlay, "debug_mode", False))
			prefer_spikeinterface = (
				bool(inputs.force_restart)
				and (not bool(inputs.force_replot))
				and (not bool(inputs.force_replot_per_unit))
				and (not bool(inputs.reports.replot_from_disk))
			)
			if prefer_spikeinterface:
				try:
					LOGGER.info(
						"force_restart enabled; materializing templates from SpikeInterface "
						"(include_concat=%s, include_segments=%s)",
						bool(inputs.include_concat),
						bool(inputs.include_segments),
					)
					materialize_out = materialize_templates_from_spikeinterface(
						well_out_dir=well_out_dir,
						templates_out_dir=templates_out_dir,
						raw_data_h5_path=inputs.h5_path,
						stream_id=str(inputs.stream_id),
						unit_ids=(list(inputs.unit_ids) if inputs.unit_ids is not None else None),
						include_concat=bool(inputs.include_concat),
						include_segments=bool(inputs.include_segments),
						execution_upsampling=inputs.execution_upsampling,
						enable_merge=bool(inputs.merge.enable),
						merge_method=str(inputs.merge.method),
						centering_method=str(inputs.merge.centering_method),
						max_waveforms_per_source_channel=(
							None
							if inputs.merge.max_waveforms_per_source_channel is None
							else int(inputs.merge.max_waveforms_per_source_channel)
						),
						overlap_match_priority=tuple(inputs.merge.overlap_match_priority),
						location_tolerance_um=float(inputs.merge.location_tolerance_um),
						debug_overlay=overlay_debug_mode,
					)
					if len(materialize_out) == 3:
						merged_units_dir_resolved, full_channels_templates_dir_resolved, upsampling_decisions_by_unit = materialize_out
					else:
						merged_units_dir_resolved, full_channels_templates_dir_resolved = materialize_out  # type: ignore[misc]
						upsampling_decisions_by_unit = {}
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
				materialize_out = materialize_templates_from_spikeinterface(
					well_out_dir=well_out_dir,
					templates_out_dir=templates_out_dir,
					raw_data_h5_path=inputs.h5_path,
					stream_id=str(inputs.stream_id),
					unit_ids=(list(inputs.unit_ids) if inputs.unit_ids is not None else None),
					include_concat=bool(inputs.include_concat),
					include_segments=bool(inputs.include_segments),
					execution_upsampling=inputs.execution_upsampling,
					enable_merge=bool(inputs.merge.enable),
					merge_method=str(inputs.merge.method),
					centering_method=str(inputs.merge.centering_method),
					max_waveforms_per_source_channel=(
						None
						if inputs.merge.max_waveforms_per_source_channel is None
						else int(inputs.merge.max_waveforms_per_source_channel)
					),
					overlap_match_priority=tuple(inputs.merge.overlap_match_priority),
					location_tolerance_um=float(inputs.merge.location_tolerance_um),
					debug_overlay=overlay_debug_mode,
				)
				if len(materialize_out) == 3:
					merged_units_dir_resolved, full_channels_templates_dir_resolved, upsampling_decisions_by_unit = materialize_out
				else:
					merged_units_dir_resolved, full_channels_templates_dir_resolved = materialize_out  # type: ignore[misc]
					upsampling_decisions_by_unit = {}
			merged_units_dir = merged_units_dir_resolved
			full_channels_templates_dir = full_channels_templates_dir_resolved
		return merged_units_dir, full_channels_templates_dir

	if (
		bool(inputs.reports.replot_from_disk)
		and (not bool(inputs.force_restart))
		and (not bool(inputs.force_replot))
		and (not bool(inputs.force_replot_per_unit))
	):
		LOGGER.info("Templates stage selecting units from existing unit summaries (reports_replot_from_disk=true)")
		if inputs.unit_ids is not None:
			unit_ids = list(inputs.unit_ids)
		else:
			unit_ids = _discover_unit_ids_from_unit_summaries(templates_out_dir)
		if inputs.unit_limit is not None:
			unit_ids = unit_ids[: int(inputs.unit_limit)]
	else:
		merged_units_dir_resolved, _ = _ensure_templates_dirs()
		unit_ids = _build_unit_ids(inputs, merged_units_dir_resolved)
	LOGGER.info("Templates stage discovered %d unit(s) before curated filtering", len(unit_ids))
	if bool(inputs.require_curated_units) and inputs.unit_ids is None:
		curated = _load_curated_units_from_spikesorting(well_out_dir)
		if curated is None:
			raise RuntimeError(
				"Templates stage requires curated units, but curated units file was not found/readable at "
				f"{well_out_dir / 'stg2_spikesorting_outputs' / 'qm_unfiltered.xlsx'}"
			)
		unit_ids = _apply_curated_filter(unit_ids, curated)
		LOGGER.info("Templates stage curated filter retained %d unit(s)", len(unit_ids))

	LOGGER.info("Templates stage will process %d unit(s)", len(unit_ids))

	def _process_unit(unit_id: Any) -> UnitTemplatesResult:
		LOGGER.info("Templates unit start: unit_id=%s", unit_id)
		paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		paths["unit_dir"].mkdir(parents=True, exist_ok=True)

		if (
			(not bool(inputs.force_restart))
			and (not bool(inputs.force_replot))
			and (not bool(inputs.force_replot_per_unit))
		):
			existing_result = _load_unit_result_from_summary(
				unit_id=unit_id,
				unit_summary_json=paths["unit_summary_json"],
			)
			if existing_result is not None:
				LOGGER.info("Templates unit skip: unit_id=%s (reusing existing summary)", unit_id)
				return existing_result

		unit_summary: dict[str, Any] = {
			"unit_id": unit_id,
			"status": "ok",
			"error": None,
			"outputs": {},
		}
		decision_for_unit = upsampling_decisions_by_unit.get(unit_id, None)
		if decision_for_unit is None:
			decision_for_unit = _load_persisted_upsampling_decision(unit_summary_json=paths["unit_summary_json"])
		if decision_for_unit is None and (bool(inputs.force_replot) or bool(inputs.force_replot_per_unit)):
			decision_for_unit = _infer_replot_upsampling_decision(inputs=inputs)
		if isinstance(decision_for_unit, dict):
			unit_summary["upsampling"] = dict(decision_for_unit)
		unit_probe_geometry = _effective_probe_geometry_for_unit(
			base_probe_geometry=inputs.probe_geometry,
			upsampling_decision=decision_for_unit,
		)
		unit_summary["effective_sampling_rate_hz"] = (
			None
			if unit_probe_geometry is None
			else _as_positive_float_or_none(unit_probe_geometry.sampling_rate_hz)
		)
		try:
			merged_units_dir_resolved, full_channels_templates_dir_resolved = _ensure_templates_dirs()
			merged_dir = merged_units_dir_resolved / f"unit_{unit_id}"
			full_dir = full_channels_templates_dir_resolved / f"unit_{unit_id}"
			overlay_debug_mode = bool(getattr(inputs.per_unit_outputs.template_wf_overlay, "debug_mode", False))
			LOGGER.info("Templates unit load artifacts: unit_id=%s merged_dir=%s", unit_id, merged_dir)

			merged_template, merged_locs = _load_merged_unit(merged_dir)
			merged_electrode_ids = load_materialized_merged_electrode_ids(merged_unit_dir=merged_dir)
			full_payload = _load_full_unit(full_dir) if full_channels_templates_dir_resolved.exists() else None

			if bool(inputs.per_unit_outputs.merged_template.write_npy):
				paths["merged_template_npy"].parent.mkdir(parents=True, exist_ok=True)
				np.save(paths["merged_template_npy"], merged_template)
				unit_summary["outputs"]["merged_template_npy"] = str(paths["merged_template_npy"])
				merged_locs_path = paths.get("merged_template_channel_locations_npy")
				if merged_locs_path is not None:
					merged_locs_path.parent.mkdir(parents=True, exist_ok=True)
					np.save(merged_locs_path, merged_locs)
					unit_summary["outputs"]["merged_template_channel_locations_npy"] = str(merged_locs_path)

			if bool(inputs.per_unit_outputs.full_template.write_npy):
				if full_payload is None:
					full_template_to_write = merged_template
					full_locations_to_write = merged_locs
				else:
					full_template_to_write = full_payload[0]
					full_locations_to_write = full_payload[1]
				paths["full_template_npy"].parent.mkdir(parents=True, exist_ok=True)
				np.save(paths["full_template_npy"], full_template_to_write)
				unit_summary["outputs"]["full_template_npy"] = str(paths["full_template_npy"])
				full_locs_path = paths.get("full_template_channel_locations_npy")
				if full_locs_path is not None:
					full_locs_path.parent.mkdir(parents=True, exist_ok=True)
					np.save(full_locs_path, full_locations_to_write)
					unit_summary["outputs"]["full_template_channel_locations_npy"] = str(full_locs_path)

			if bool(inputs.per_unit_outputs.scan_template.write_npy):
				scan_template_to_write = merged_template if full_payload is None else full_payload[0]
				scan_locations_to_write = merged_locs if full_payload is None else full_payload[1]
				paths["scan_template_npy"].parent.mkdir(parents=True, exist_ok=True)
				np.save(paths["scan_template_npy"], scan_template_to_write)
				unit_summary["outputs"]["scan_template_npy"] = str(paths["scan_template_npy"])
				scan_locs_path = paths.get("scan_template_channel_locations_npy")
				if scan_locs_path is not None:
					scan_locs_path.parent.mkdir(parents=True, exist_ok=True)
					np.save(scan_locs_path, scan_locations_to_write)
					unit_summary["outputs"]["scan_template_channel_locations_npy"] = str(scan_locs_path)

			if bool(inputs.per_unit_outputs.square_template.write_npy):
				sq = _build_square_template(
					merged_template,
					padding_mode=str(inputs.per_unit_outputs.square_template.padding_value),
				)
				sq_locs = _build_square_locations(merged_locs, target_channels=int(sq.shape[0]))
				paths["square_template_npy"].parent.mkdir(parents=True, exist_ok=True)
				np.save(paths["square_template_npy"], sq)
				unit_summary["outputs"]["square_template_npy"] = str(paths["square_template_npy"])
				square_locs_path = paths.get("square_template_channel_locations_npy")
				if square_locs_path is not None:
					square_locs_path.parent.mkdir(parents=True, exist_ok=True)
					np.save(square_locs_path, sq_locs)
					unit_summary["outputs"]["square_template_channel_locations_npy"] = str(square_locs_path)

			template_plot, locs_plot, source = _select_template_for_scope(
				merged_template=merged_template,
				merged_locs=merged_locs,
				full_payload=full_payload,
				channel_scope=inputs.per_unit_outputs.template.channel_scope,
			)
			template_circles, locs_circles, circles_source = _select_template_for_scope(
				merged_template=merged_template,
				merged_locs=merged_locs,
				full_payload=full_payload,
				channel_scope=inputs.per_unit_outputs.template_circles.channel_scope,
			)
			unit_summary["selected_template_source"] = source

			amp_template, amp_locs, amp_source = _select_template_for_shape(
				merged_template=merged_template,
				merged_locs=merged_locs,
				full_payload=full_payload,
				template_shape=inputs.per_unit_outputs.footprint_plots.amplitude_map.template_shape,
			)
			lat_template, lat_locs, lat_source = _select_template_for_shape(
				merged_template=merged_template,
				merged_locs=merged_locs,
				full_payload=full_payload,
				template_shape=inputs.per_unit_outputs.footprint_plots.latency_map.template_shape,
			)
			topo_amp_template, topo_amp_locs, topo_amp_source = _select_template_for_shape(
				merged_template=merged_template,
				merged_locs=merged_locs,
				full_payload=full_payload,
				template_shape=inputs.per_unit_outputs.topographical_footprints.amplitude.template_shape,
			)
			topo_lat_template, topo_lat_locs, topo_lat_source = _select_template_for_shape(
				merged_template=merged_template,
				merged_locs=merged_locs,
				full_payload=full_payload,
				template_shape=inputs.per_unit_outputs.topographical_footprints.latency.template_shape,
			)
			prop_shape = str(inputs.per_unit_outputs.propagation_plots.latency_map.template_shape or "top_channels_only")
			if prop_shape.strip().lower().replace("-", "_").replace(" ", "_") == "top_channels_only":
				prop_template, prop_locs, prop_source = template_plot, locs_plot, source
			else:
				prop_template, prop_locs, prop_source = _select_template_for_shape(
					merged_template=merged_template,
					merged_locs=merged_locs,
					full_payload=full_payload,
					template_shape=prop_shape,
				)
			unit_summary["selected_template_sources"] = {
				"template": source,
				"template_circles": circles_source,
				"footprint_amplitude_map": amp_source,
				"footprint_latency_map": lat_source,
				"topographical_amplitude": topo_amp_source,
				"topographical_latency": topo_lat_source,
				"propagation": prop_source,
			}
			LOGGER.info("Templates unit render plots: unit_id=%s", unit_id)

			outputs = render_template_plot(
				template=template_plot,
				locations_xy=locs_plot,
				config=inputs.per_unit_outputs.template,
				png_path=paths["template_png"],
				svg_path=paths["template_svg"],
				unit_id=unit_id,
			)
			unit_summary["outputs"].update(outputs)

			circles_outputs = render_template_circles_plot(
				template=template_circles,
				locations_xy=locs_circles,
				config=inputs.per_unit_outputs.template_circles,
				png_path=paths["template_circles_png"],
				svg_path=paths["template_circles_svg"],
				probe_geometry=unit_probe_geometry,
				unit_id=unit_id,
			)
			unit_summary["outputs"].update(circles_outputs)

			overlay_payload = load_materialized_overlay_waveforms(merged_unit_dir=merged_dir)
			if overlay_payload is None:
				if overlay_debug_mode:
					print(
						"[template_wf_overlay][debug] "
						f"unit={unit_id} missing overlay artifact under {merged_dir}",
						flush=True,
					)
				unit_summary["outputs"]["template_wf_overlay_error"] = (
					"Skipped overlay: missing top-electrode waveform artifact for unit"
				)
				unit_summary["outputs"]["extremum_ch_wf_overlay_error"] = unit_summary["outputs"]["template_wf_overlay_error"]
			else:
				overlay_waveforms, overlay_top_electrode_id, overlay_total_count = overlay_payload
				overlay_time_upsample = TimeUpsampleConfig(enabled=False, factor=1, method=str(inputs.reports.time_upsample.method))
				if isinstance(decision_for_unit, dict):
					if bool(decision_for_unit.get("applied", False)):
						overlay_time_upsample = TimeUpsampleConfig(
							enabled=True,
							factor=max(1, int(decision_for_unit.get("factor", 1))),
							method=str(decision_for_unit.get("method", inputs.reports.time_upsample.method)),
						)
				elif bool(inputs.reports.time_upsample.enabled):
					# Legacy fallback when no per-unit decision is available.
					overlay_time_upsample = inputs.reports.time_upsample
				# Prefer merged-template electrode-id label for consistent cross-plot channel identity.
				if merged_electrode_ids is not None and int(template_plot.shape[0]) > 0 and int(len(merged_electrode_ids)) >= int(template_plot.shape[0]):
					try:
						overlay_top_idx = int(np.argmax(np.ptp(template_plot, axis=1)))
						overlay_top_electrode_id = merged_electrode_ids[overlay_top_idx]
					except Exception:
						pass
				if overlay_debug_mode:
					print(
						"[template_wf_overlay][debug] "
						f"unit={unit_id} loaded overlay payload "
						f"waveforms_shape={tuple(np.asarray(overlay_waveforms).shape)} "
						f"top_electrode_id={overlay_top_electrode_id} total={overlay_total_count}",
						flush=True,
					)
				overlay_outputs = render_template_wf_overlay(
					template=template_plot,
					config=inputs.per_unit_outputs.template_wf_overlay,
					time_upsample=overlay_time_upsample,
					pdf_path=paths["template_wf_overlay_pdf"],
					png_path=paths["template_wf_overlay_png"],
					probe_geometry=unit_probe_geometry,
					waveform_traces=overlay_waveforms,
					top_electrode_id=overlay_top_electrode_id,
					total_waveforms_at_channel=overlay_total_count,
				)
				if overlay_debug_mode:
					print(
						"[template_wf_overlay][debug] "
						f"unit={unit_id} overlay outputs={overlay_outputs}",
						flush=True,
					)
				unit_summary["outputs"].update(overlay_outputs)

			amp_outputs = render_footprint_amplitude_map(
				template=amp_template,
				locations_xy=amp_locs,
				config=inputs.per_unit_outputs.footprint_plots.amplitude_map,
				probe_geometry=unit_probe_geometry,
				png_path=paths["footprint_amplitude_map_png"],
				svg_path=paths["footprint_amplitude_map_svg"],
			)
			unit_summary["outputs"].update(amp_outputs)

			lat_outputs = render_footprint_latency_map(
				template=lat_template,
				locations_xy=lat_locs,
				config=inputs.per_unit_outputs.footprint_plots.latency_map,
				probe_geometry=unit_probe_geometry,
				png_path=paths["footprint_latency_map_png"],
				svg_path=paths["footprint_latency_map_svg"],
			)
			unit_summary["outputs"].update(lat_outputs)

			topo_amp_outputs = render_topographical_amplitude_footprint(
				template=topo_amp_template,
				locations_xy=topo_amp_locs,
				config=inputs.per_unit_outputs.topographical_footprints.amplitude,
				probe_geometry=unit_probe_geometry,
				png_path=paths["topographical_amplitude_footprint_png"],
				svg_path=paths["topographical_amplitude_footprint_svg"],
			)
			unit_summary["outputs"].update(topo_amp_outputs)

			topo_lat_outputs = render_topographical_latency_footprint(
				template=topo_lat_template,
				locations_xy=topo_lat_locs,
				config=inputs.per_unit_outputs.topographical_footprints.latency,
				probe_geometry=unit_probe_geometry,
				png_path=paths["topographical_latency_footprint_png"],
				svg_path=paths["topographical_latency_footprint_svg"],
			)
			unit_summary["outputs"].update(topo_lat_outputs)

			prop_outputs = render_propagation_plot(
				template=prop_template,
				locations_xy=prop_locs,
				config=inputs.per_unit_outputs.propagation_plots,
				pdf_path=paths["propagation_plot_pdf"],
				png_path=paths["propagation_plot_png"],
				probe_geometry=unit_probe_geometry,
				channel_labels_by_row=(
					merged_electrode_ids
					if (prop_source == "merged_contributing" and merged_electrode_ids is not None and int(len(merged_electrode_ids)) == int(prop_template.shape[0]))
					else None
				),
			)
			unit_summary["outputs"].update(prop_outputs)
			LOGGER.info("Templates unit render complete: unit_id=%s outputs=%d", unit_id, len(unit_summary["outputs"]))
		except Exception as exc:
			unit_summary["status"] = "error"
			unit_summary["error"] = str(exc)
			LOGGER.exception("Failed templates for unit %s", unit_id)

		write_json(paths["unit_summary_json"], unit_summary)
		LOGGER.info("Templates unit done: unit_id=%s status=%s", unit_id, str(unit_summary.get("status", "ok")))
		return UnitTemplatesResult(
			unit_id=unit_id,
			status=str(unit_summary.get("status", "ok")),
			outputs={str(k): str(v) for k, v in dict(unit_summary.get("outputs", {})).items()},
			error=(unit_summary.get("error") if unit_summary.get("error") else None),
		)

	unit_results: list[UnitTemplatesResult] = []
	units_to_process = list(unit_ids)
	if (
		bool(inputs.reports.replot_from_disk)
		and (not bool(inputs.force_restart))
		and (not bool(inputs.force_replot))
		and (not bool(inputs.force_replot_per_unit))
	):
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
	LOGGER.info(
		"Templates unit execution start: units_to_process=%d reused_units=%d worker_count=%d",
		len(units_to_process),
		len(unit_results),
		worker_count,
	)
	if worker_count <= 1 or len(units_to_process) <= 1:
		total_to_run = len(units_to_process)
		for idx, unit_id in enumerate(units_to_process, start=1):
			unit_results.append(_process_unit(unit_id))
			LOGGER.info("Templates unit progress: %d/%d completed", idx, total_to_run)
	else:
		futures: dict[concurrent.futures.Future[UnitTemplatesResult], Any] = {}
		with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
			for unit_id in units_to_process:
				fut = pool.submit(_process_unit, unit_id)
				futures[fut] = unit_id

			total_to_run = len(futures)
			completed = 0
			for fut in concurrent.futures.as_completed(futures):
				unit_results.append(fut.result())
				completed += 1
				LOGGER.info("Templates unit progress: %d/%d completed", completed, total_to_run)

	unit_results.sort(key=lambda r: str(r.unit_id))
	LOGGER.info("Templates unit execution complete: total_results=%d", len(unit_results))
	report_outputs: dict[str, str] = {}
	try:
		LOGGER.info("Templates reports start: stream=%s", inputs.stream_id)
		report_paths = resolve_report_output_paths(templates_out_dir=templates_out_dir, reports=inputs.reports)

		def _collect_grid_unit_payloads(*, kind: str, template_shape: str | None = None) -> list[dict[str, Any]]:
			payloads: list[dict[str, Any]] = []
			try:
				merged_units_dir_resolved, full_channels_templates_dir_resolved = _ensure_templates_dirs()
			except Exception:
				return payloads
			for u in unit_results:
				if str(u.status) != "ok":
					continue
				uid = u.unit_id
				try:
					merged_dir = merged_units_dir_resolved / f"unit_{uid}"
					full_dir = full_channels_templates_dir_resolved / f"unit_{uid}"
					merged_template, merged_locs = _load_merged_unit(merged_dir)
					full_payload = _load_full_unit(full_dir) if full_channels_templates_dir_resolved.exists() else None
					if str(kind) == "circles":
						t, locs, _ = _select_template_for_scope(
							merged_template=merged_template,
							merged_locs=merged_locs,
							full_payload=full_payload,
							channel_scope=inputs.per_unit_outputs.template_circles.channel_scope,
						)
					else:
						t, locs, _ = _select_template_for_shape(
							merged_template=merged_template,
							merged_locs=merged_locs,
							full_payload=full_payload,
							template_shape=str(template_shape or "square"),
						)
					payloads.append({"unit_id": uid, "template": t, "locations_xy": locs})
				except Exception:
					continue
			return payloads

		def _collect_overlay_unit_payloads() -> list[dict[str, Any]]:
			payloads: list[dict[str, Any]] = []
			try:
				merged_units_dir_resolved, _ = _ensure_templates_dirs()
			except Exception:
				return payloads
			for u in unit_results:
				if str(u.status) != "ok":
					continue
				uid = u.unit_id
				try:
					merged_dir = merged_units_dir_resolved / f"unit_{uid}"
					merged_template, _ = _load_merged_unit(merged_dir)
					overlay_payload = load_materialized_overlay_waveforms(merged_unit_dir=merged_dir)
					if overlay_payload is None:
						continue
					waveforms, top_electrode_id, total_count = overlay_payload
					payloads.append(
						{
							"unit_id": uid,
							"template": merged_template,
							"waveform_traces": waveforms,
							"top_electrode_id": top_electrode_id,
							"total_waveforms_at_channel": total_count,
						}
					)
				except Exception:
					continue
			return payloads

		overlay_paths = [
			Path(u.outputs["template_wf_overlay_png"])
			for u in unit_results
			if "template_wf_overlay_png" in u.outputs
		]
		LOGGER.info("Templates reports wf_overlay_grid inputs=%d", len(overlay_paths))
		report_outputs.update(
			render_wf_overlay_grid(
				overlay_png_paths=overlay_paths,
				unit_payloads=_collect_overlay_unit_payloads(),
				config=inputs.reports.wf_overlay_grid,
				pdf_path=report_paths["wf_overlay_grid_pdf"],
				png_path=report_paths["wf_overlay_grid_png"],
				overlay_config=inputs.per_unit_outputs.template_wf_overlay,
				report_time_upsample=inputs.reports.time_upsample,
				probe_geometry=inputs.probe_geometry,
			)
		)
		circles_map_paths = [
			Path(u.outputs["template_circles_png"])
			for u in unit_results
			if "template_circles_png" in u.outputs
		]
		LOGGER.info("Templates reports circles_map_grid inputs=%d", len(circles_map_paths))
		report_outputs.update(
			render_footprint_map_grid(
				image_paths=circles_map_paths,
				unit_payloads=_collect_grid_unit_payloads(kind="circles"),
				config=inputs.reports.footprint_grids.circles_map_grid,
				pdf_path=report_paths["template_circles_map_grid_pdf"],
				png_path=report_paths["template_circles_map_grid_png"],
				pdf_output_key="template_circles_map_grid_pdf",
				png_output_key="template_circles_map_grid_png",
				title="Template circles map grid",
				panel_kind="circles",
				circles_config=inputs.per_unit_outputs.template_circles,
				probe_geometry=inputs.probe_geometry,
			)
		)
		amp_map_paths = [
			Path(u.outputs["footprint_amplitude_map_png"])
			for u in unit_results
			if "footprint_amplitude_map_png" in u.outputs
		]
		LOGGER.info("Templates reports amplitude_map_grid inputs=%d", len(amp_map_paths))
		report_outputs.update(
			render_footprint_map_grid(
				image_paths=amp_map_paths,
				unit_payloads=_collect_grid_unit_payloads(
					kind="amplitude",
					template_shape=inputs.reports.footprint_grids.amplitude_map_grid.template_shape,
				),
				config=inputs.reports.footprint_grids.amplitude_map_grid,
				pdf_path=report_paths["footprint_amplitude_map_grid_pdf"],
				png_path=report_paths["footprint_amplitude_map_grid_png"],
				pdf_output_key="footprint_amplitude_map_grid_pdf",
				png_output_key="footprint_amplitude_map_grid_png",
				title="Template footprint amplitude map grid",
				panel_kind="amplitude",
				footprint_config=inputs.per_unit_outputs.footprint_plots.amplitude_map,
				probe_geometry=inputs.probe_geometry,
			)
		)
		lat_map_paths = [
			Path(u.outputs["footprint_latency_map_png"])
			for u in unit_results
			if "footprint_latency_map_png" in u.outputs
		]
		LOGGER.info("Templates reports latency_map_grid inputs=%d", len(lat_map_paths))
		report_outputs.update(
			render_footprint_map_grid(
				image_paths=lat_map_paths,
				unit_payloads=_collect_grid_unit_payloads(
					kind="latency",
					template_shape=inputs.reports.footprint_grids.latency_map_grid.template_shape,
				),
				config=inputs.reports.footprint_grids.latency_map_grid,
				pdf_path=report_paths["footprint_latency_map_grid_pdf"],
				png_path=report_paths["footprint_latency_map_grid_png"],
				pdf_output_key="footprint_latency_map_grid_pdf",
				png_output_key="footprint_latency_map_grid_png",
				title="Template footprint latency map grid",
				panel_kind="latency",
				footprint_config=inputs.per_unit_outputs.footprint_plots.latency_map,
				probe_geometry=inputs.probe_geometry,
			)
		)
		if bool(inputs.reports.plot_multi_source_pdf.enabled):
			LOGGER.info("Templates reports multi_source_pdf enabled; rendering")
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
	else:
		LOGGER.info("Templates reports complete: generated=%d", len(report_outputs))

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
			"max_waveforms_per_source_channel": (
				None if inputs.merge.max_waveforms_per_source_channel is None else int(inputs.merge.max_waveforms_per_source_channel)
			),
			"overlap_match_priority": list(inputs.merge.overlap_match_priority),
			"location_tolerance_um": float(inputs.merge.location_tolerance_um),
		},
		"execution_upsampling": {
			"enabled": bool(inputs.execution_upsampling.enabled),
			"factor": int(max(1, int(inputs.execution_upsampling.factor))),
			"method": str(inputs.execution_upsampling.method),
			"mismatch_tolerance_hz": float(max(0.0, float(inputs.execution_upsampling.mismatch_tolerance_hz))),
			"raw_rate_fallback_hz": (
				None
				if inputs.execution_upsampling.raw_rate_fallback_hz is None
				else float(inputs.execution_upsampling.raw_rate_fallback_hz)
			),
		},
		"upsampling_decisions_by_unit": {str(k): v for k, v in upsampling_decisions_by_unit.items()},
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
	ok_count = sum(1 for u in unit_results if str(u.status) == "ok")
	err_count = sum(1 for u in unit_results if str(u.status) != "ok")
	LOGGER.info(
		"Templates stage done: stream=%s units_ok=%d units_error=%d summary=%s",
		str(inputs.stream_id),
		ok_count,
		err_count,
		summary_json,
	)

	return TemplatesResult(
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
		summary_json=summary_json,
		units=unit_results,
		report_outputs=report_outputs,
	)
