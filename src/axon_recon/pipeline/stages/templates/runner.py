from __future__ import annotations

import concurrent.futures
from dataclasses import replace
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir #TODO: dont import this from v1

from .core.merge import materialize_templates_from_spikeinterface
from .core.quality_checks import detect_multiple_negative_peaks
from .core.render import (
	compose_png_side_by_side,
	compose_svg_side_by_side,
	compute_propagation_channel_order,
	render_footprint_amplitude_map,
	render_footprint_map_grid_from_assets,
	render_footprint_latency_map,
	render_multi_source_pdf,
	render_propagation_plot,
	render_template_circles_plot,
	render_template_plot,
	render_template_wf_overlay,
	render_topographical_amplitude_footprint,
	render_topographical_latency_footprint,
	render_wf_overlay_grid_from_assets,
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


def _order_index_labels_from_anchor(order_payload: dict[str, Any], *, anchor_channel: int | None) -> dict[int, int] | None:
	ordered = np.asarray(order_payload.get("ordered_channel_indices", []), dtype=int).reshape(-1)
	if int(ordered.shape[0]) == 0:
		return None
	if anchor_channel is None:
		return None
	hits = np.flatnonzero(ordered == int(anchor_channel))
	if hits.size == 0:
		return None
	anchor_pos = int(hits[0])
	return {int(ch): int(i - anchor_pos) for i, ch in enumerate(ordered.tolist())}


def _channel_order_labels_for_mode(
	order_payload: dict[str, Any],
	*,
	trace_label_mode: str,
	use_relative_signed_numbers: bool,
	order_index_anchor_channel: int | None = None,
) -> dict[int, int]:
	mode = str(trace_label_mode or "electrode_id").strip().lower()
	if mode == "order_index":
		order_index_labels = _order_index_labels_from_anchor(
			order_payload,
			anchor_channel=order_index_anchor_channel,
		)
		if order_index_labels is not None:
			return order_index_labels
	rank_by_channel = dict(order_payload.get("rank_by_channel", {}))
	relative_order_by_channel = dict(order_payload.get("relative_order_by_channel", {}))
	if bool(use_relative_signed_numbers):
		return relative_order_by_channel
	return rank_by_channel


def _select_max_amplitude_window_from_order(
	*,
	template_c_by_t: np.ndarray,
	ordered_channel_indices: np.ndarray,
	window_size: int,
) -> np.ndarray:
	ordered = np.asarray(ordered_channel_indices, dtype=int).reshape(-1)
	n_channels = int(ordered.shape[0])
	if n_channels == 0:
		return ordered
	k = int(max(1, window_size))
	if k >= n_channels:
		return ordered

	ptp_all = np.ptp(np.asarray(template_c_by_t), axis=1)
	ordered_ptp = np.asarray([float(ptp_all[int(ch)]) for ch in ordered.tolist()], dtype=float)

	window_sum = float(np.sum(ordered_ptp[:k]))
	best_sum = float(window_sum)
	best_start = 0
	for start in range(1, n_channels - k + 1):
		window_sum += float(ordered_ptp[start + k - 1]) - float(ordered_ptp[start - 1])
		if window_sum > best_sum:
			best_sum = float(window_sum)
			best_start = int(start)
	return np.asarray(ordered[best_start : best_start + k], dtype=int)


def _select_window_from_order_with_strategy(
	*,
	template_c_by_t: np.ndarray,
	ordered_channel_indices: np.ndarray,
	window_size: int,
	window_strategy: str,
) -> np.ndarray:
	ordered = np.asarray(ordered_channel_indices, dtype=int).reshape(-1)
	n_channels = int(ordered.shape[0])
	if n_channels == 0:
		return ordered
	k = int(max(1, window_size))
	if k >= n_channels:
		return ordered

	strategy = str(window_strategy or "max_ptp_sum").strip().lower()
	if strategy == "first_k":
		return np.asarray(ordered[:k], dtype=int)
	# default
	return _select_max_amplitude_window_from_order(
		template_c_by_t=template_c_by_t,
		ordered_channel_indices=ordered,
		window_size=k,
	)


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


def _run_template_quality_checks(
	*,
	unit_id: Any,
	merged_template: np.ndarray,
	merged_electrode_ids: list[Any] | None,
	inputs: TemplatesInputs,
) -> dict[str, Any]:
	qc_cfg = inputs.quality_checks
	multi_cfg = qc_cfg.check_for_multiple_peaks_at_channel_templates
	if (not bool(qc_cfg.enable)) or (not bool(multi_cfg.enable)):
		return {
			"enabled": False,
			"check_for_multiple_peaks_at_channel_templates": {
				"enabled": False,
				"detected": False,
				"violation_count": 0,
				"violations": [],
			},
		}

	result = detect_multiple_negative_peaks(
		template_c_by_t=merged_template,
		channel_labels=merged_electrode_ids,
		prominence_fraction=float(multi_cfg.prominence_fraction),
		min_separation_samples=int(multi_cfg.min_separation_samples),
		max_peaks_per_channel=int(multi_cfg.max_peaks_per_channel),
	)
	emit_quality_check_warnings = not bool(getattr(qc_cfg, "suppress_warnings", False))

	if emit_quality_check_warnings:
		for violation in result.get("violations", []):
			ch_index = int(violation.get("channel_index", -1))
			ch_label = violation.get("channel_label", None)
			peak_count = int(violation.get("peak_count", 0))
			peak_indices = violation.get("peak_indices", [])
			if ch_label is None or str(ch_label).strip() == "":
				channel_repr = f"idx={ch_index}"
			else:
				channel_repr = f"eid={ch_label} idx={ch_index}"
			LOGGER.warning(
				"quality_check multiple_negative_peaks: unit_id=%s channel=%s peaks=%d peak_indices=%s",
				unit_id,
				channel_repr,
				peak_count,
				peak_indices,
			)

	if emit_quality_check_warnings and bool(result.get("detected", False)):
		LOGGER.warning(
			"quality_check multiple_negative_peaks summary: unit_id=%s violating_channels=%d/%d",
			unit_id,
			int(result.get("violation_count", 0)),
			int(result.get("channel_count", 0)),
		)

	return {
		"enabled": True,
		"check_for_multiple_peaks_at_channel_templates": {
			"enabled": True,
			**result,
		},
	}


def _build_quality_check_aggregate(*, templates_out_dir: Path, inputs: TemplatesInputs) -> tuple[dict[str, Any], Path | None]:
	agg_cfg = inputs.quality_checks_outputs.check_for_multiple_peaks_at_channel_templates
	aggregate_json = templates_out_dir / str(agg_cfg.json_relpath)
	units_payload: list[dict[str, Any]] = []
	all_violations: list[dict[str, Any]] = []
	for p in sorted(templates_out_dir.glob("units/**/unit_templates_summary.json")):
		if not p.is_file():
			continue
		try:
			payload = read_json(p)
		except Exception:
			continue
		if not isinstance(payload, dict):
			continue
		unit_id = payload.get("unit_id", None)
		qc_payload = payload.get("quality_checks", {})
		if not isinstance(qc_payload, dict):
			continue
		multiple_payload = qc_payload.get("check_for_multiple_peaks_at_channel_templates", {})
		if not isinstance(multiple_payload, dict):
			continue
		unit_record = {
			"unit_id": unit_id,
			"status": payload.get("status", "ok"),
			"detected": bool(multiple_payload.get("detected", False)),
			"violation_count": int(multiple_payload.get("violation_count", 0)),
			"channel_count": int(multiple_payload.get("channel_count", 0)),
			"violations": list(multiple_payload.get("violations", [])),
		}
		units_payload.append(unit_record)
		for violation in unit_record["violations"]:
			if not isinstance(violation, dict):
				continue
			all_violations.append({"unit_id": unit_id, **violation})

	payload = {
		"check": "check_for_multiple_peaks_at_channel_templates",
		"units_scanned": int(len(units_payload)),
		"units_with_violations": int(sum(1 for u in units_payload if bool(u.get("detected", False)))),
		"total_violations": int(len(all_violations)),
		"violations": all_violations,
		"units": units_payload,
	}
	if bool(agg_cfg.write_json):
		write_json(aggregate_json, payload)
		return payload, aggregate_json
	return payload, None


def _pad_value_from_mode(mode: str) -> float:
	m = str(mode or "zero").strip().lower()
	if m == "one":
		return 1.0
	if m == "nan":
		return float("nan")
	return 0.0


def _infer_axis_pitch(values: np.ndarray) -> float | None:
	v = np.asarray(values, dtype=float).reshape(-1)
	v = v[np.isfinite(v)]
	if int(v.size) < 2:
		return None
	v_unique = np.unique(np.round(v, decimals=9))
	diffs = np.diff(v_unique)
	positive = diffs[diffs > 1e-9]
	if int(positive.size) == 0:
		return None
	return float(np.min(positive))


def _build_square_grid(locations_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		raise ValueError(f"Unexpected locations shape for square template: {locs.shape}")

	locs2 = np.asarray(locs[:, :2], dtype=float)
	valid_mask = np.isfinite(locs2).all(axis=1)
	valid_locs = locs2[valid_mask]
	if int(valid_locs.shape[0]) == 0:
		raise ValueError("No finite channel locations available for square template construction")

	x0 = float(np.min(valid_locs[:, 0]))
	y0 = float(np.min(valid_locs[:, 1]))
	x_pitch = _infer_axis_pitch(valid_locs[:, 0])
	y_pitch = _infer_axis_pitch(valid_locs[:, 1])
	if x_pitch is None and y_pitch is None:
		x_pitch = 1.0
		y_pitch = 1.0
	elif x_pitch is None:
		x_pitch = float(y_pitch)
	elif y_pitch is None:
		y_pitch = float(x_pitch)

	x_span = float(np.max(valid_locs[:, 0]) - x0)
	y_span = float(np.max(valid_locs[:, 1]) - y0)
	nx = int(np.round(x_span / float(x_pitch))) + 1
	ny = int(np.round(y_span / float(y_pitch))) + 1
	n_side = int(max(1, nx, ny, int(np.ceil(np.sqrt(max(1, locs2.shape[0]))))))

	x_vals = x0 + np.arange(n_side, dtype=float) * float(x_pitch)
	y_vals = y0 + np.arange(n_side, dtype=float) * float(y_pitch)
	grid = np.asarray([[float(x), float(y)] for y in y_vals for x in x_vals], dtype=float)

	mapping = np.full((int(locs2.shape[0]),), -1, dtype=int)
	occupied = np.zeros((int(grid.shape[0]),), dtype=bool)
	for idx, loc in enumerate(locs2):
		if not bool(np.isfinite(loc).all()):
			continue
		x_idx = int(np.clip(np.round((float(loc[0]) - x0) / float(x_pitch)), 0, n_side - 1))
		y_idx = int(np.clip(np.round((float(loc[1]) - y0) / float(y_pitch)), 0, n_side - 1))
		candidate = int(y_idx * n_side + x_idx)
		if not occupied[candidate]:
			mapping[idx] = candidate
			occupied[candidate] = True
			continue
		available = np.flatnonzero(~occupied)
		if int(available.size) == 0:
			continue
		dists = np.sum((grid[available, :] - loc[None, :]) ** 2, axis=1)
		best = int(available[int(np.argmin(dists))])
		mapping[idx] = best
		occupied[best] = True

	return grid, mapping


def _build_square_template(
	template_c_by_t: np.ndarray,
	*,
	padding_mode: str,
	locations_xy: np.ndarray | None = None,
) -> np.ndarray:
	t = np.asarray(template_c_by_t, dtype=float)
	if t.ndim != 2:
		raise ValueError(f"Unexpected template shape for square template: {t.shape}")

	if locations_xy is None:
		n_channels = int(t.shape[0])
		target = int(np.ceil(np.sqrt(max(1, n_channels))) ** 2)
		if target <= n_channels:
			return t
		pad_val = _pad_value_from_mode(padding_mode)
		out = np.full((target, int(t.shape[1])), pad_val, dtype=float)
		out[:n_channels, :] = t
		return out

	grid, mapping = _build_square_grid(locations_xy)
	pad_val = _pad_value_from_mode(padding_mode)
	out = np.full((int(grid.shape[0]), int(t.shape[1])), pad_val, dtype=float)
	source_channels = min(int(t.shape[0]), int(mapping.shape[0]))
	for src_idx in range(source_channels):
		dst_idx = int(mapping[src_idx])
		if dst_idx < 0 or dst_idx >= int(out.shape[0]):
			continue
		out[dst_idx, :] = t[src_idx, :]
	return out


def _build_square_locations(locations_xy: np.ndarray, *, target_channels: int | None = None) -> np.ndarray:
	grid, _ = _build_square_grid(locations_xy)
	if target_channels is None:
		return np.asarray(grid, dtype=float)
	target = int(target_channels)
	if target <= int(grid.shape[0]):
		return np.asarray(grid[:target, :2], dtype=float)
	out = np.full((target, 2), np.nan, dtype=float)
	out[: int(grid.shape[0]), :] = grid[:, :2]
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
						concat_analyzer_relpath=inputs.concat_analyzer_relpath,
						preproc_seg_sources_reldir=inputs.preproc_seg_sources_reldir,
						raw_data_h5_path=inputs.h5_path,
						stream_id=str(inputs.stream_id),
						unit_ids=(list(inputs.unit_ids) if inputs.unit_ids is not None else None),
						include_concat=bool(inputs.include_concat),
						include_segments=bool(inputs.include_segments),
						waveform_ms_before=inputs.waveform_extraction.ms_before,
						waveform_ms_after=inputs.waveform_extraction.ms_after,
						waveform_max_spikes_per_unit=inputs.waveform_extraction.max_spikes_per_unit,
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
					concat_analyzer_relpath=inputs.concat_analyzer_relpath,
					preproc_seg_sources_reldir=inputs.preproc_seg_sources_reldir,
					raw_data_h5_path=inputs.h5_path,
					stream_id=str(inputs.stream_id),
					unit_ids=(list(inputs.unit_ids) if inputs.unit_ids is not None else None),
					include_concat=bool(inputs.include_concat),
					include_segments=bool(inputs.include_segments),
					waveform_ms_before=inputs.waveform_extraction.ms_before,
					waveform_ms_after=inputs.waveform_extraction.ms_after,
					waveform_max_spikes_per_unit=inputs.waveform_extraction.max_spikes_per_unit,
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
			unit_quality_checks = _run_template_quality_checks(
				unit_id=unit_id,
				merged_template=merged_template,
				merged_electrode_ids=merged_electrode_ids,
				inputs=inputs,
			)
			unit_summary["quality_checks"] = unit_quality_checks
			unit_qc_cfg = inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates
			if bool(unit_qc_cfg.write_json):
				quality_checks_json = paths["quality_checks_multiple_negative_peaks_json"]
				write_json(quality_checks_json, {
					"unit_id": unit_id,
					**unit_quality_checks,
				})
				unit_summary["outputs"]["quality_checks_multiple_negative_peaks_json"] = str(quality_checks_json)

			qc_violations = unit_quality_checks.get("check_for_multiple_peaks_at_channel_templates", {}).get("violations", [])
			plot_cfg = unit_qc_cfg.plot
			if (
				(bool(plot_cfg.write_png) or bool(plot_cfg.write_svg))
				and isinstance(qc_violations, list)
				and len(qc_violations) > 0
			):
				peak_indices_by_channel: dict[int, list[int]] = {}
				for violation in qc_violations:
					if not isinstance(violation, dict):
						continue
					ch_idx = int(violation.get("channel_index", -1))
					if ch_idx < 0 or ch_idx >= int(merged_template.shape[0]):
						continue
					raw_peaks = violation.get("peak_indices", [])
					peaks = [
						int(pk)
						for pk in list(raw_peaks)
						if isinstance(pk, (int, float, np.integer, np.floating))
						and int(pk) >= 0
						and int(pk) < int(merged_template.shape[1])
					]
					if peaks:
						peak_indices_by_channel[ch_idx] = sorted(set(peaks))

				violation_channel_indices = sorted(
					{
						int(v.get("channel_index", -1))
						for v in qc_violations
						if isinstance(v, dict)
						and int(v.get("channel_index", -1)) >= 0
						and int(v.get("channel_index", -1)) < int(merged_template.shape[0])
					}
				)
				if len(violation_channel_indices) > 0:
					qc_prop_config = replace(
						inputs.per_unit_outputs.propagation_plots,
						write_pdf=False,
						write_png=bool(plot_cfg.write_png),
						show_multiple_peak_markers=bool(plot_cfg.show_multiple_peak_markers),
						delay_peak_marker_color=str(plot_cfg.delay_peak_marker_color),
					)
					qc_plot_outputs = render_propagation_plot(
						template=merged_template,
						locations_xy=merged_locs,
						config=qc_prop_config,
						pdf_path=paths["quality_checks_multiple_negative_peaks_plot_png"].with_suffix(".pdf"),
						png_path=paths["quality_checks_multiple_negative_peaks_plot_png"],
						svg_path=paths["quality_checks_multiple_negative_peaks_plot_svg"],
						write_svg=bool(plot_cfg.write_svg),
						probe_geometry=unit_probe_geometry,
						channel_labels_by_row=(
							merged_electrode_ids
							if (merged_electrode_ids is not None and int(len(merged_electrode_ids)) == int(merged_template.shape[0]))
							else None
						),
						channel_indices=violation_channel_indices,
						peak_indices_by_channel=peak_indices_by_channel,
					)
					if "propagation_plot_png" in qc_plot_outputs:
						unit_summary["outputs"]["quality_checks_multiple_negative_peaks_plot_png"] = str(qc_plot_outputs["propagation_plot_png"])
					if "propagation_plot_svg" in qc_plot_outputs:
						unit_summary["outputs"]["quality_checks_multiple_negative_peaks_plot_svg"] = str(qc_plot_outputs["propagation_plot_svg"])
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
					locations_xy=merged_locs,
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

			circles_rank_map: dict[int, int] | None = None
			if bool(inputs.per_unit_outputs.template_circles.show_propagation_order_labels):
				circles_order_payload = compute_propagation_channel_order(
					template_c_by_t=np.asarray(template_circles),
					config=inputs.per_unit_outputs.propagation_plots,
					channel_indices=np.arange(int(template_circles.shape[0]), dtype=int),
				)
				trace_label_mode = str(getattr(inputs.per_unit_outputs.propagation_plots, "trace_label_mode", "electrode_id") or "electrode_id").strip().lower()
				circles_rank_map = _channel_order_labels_for_mode(
					circles_order_payload,
					trace_label_mode=trace_label_mode,
					use_relative_signed_numbers=bool(inputs.per_unit_outputs.propagation_plots.relative_signed_order_numbers),
				)

			circles_outputs = render_template_circles_plot(
				template=template_circles,
				locations_xy=locs_circles,
				config=inputs.per_unit_outputs.template_circles,
				png_path=paths["template_circles_png"],
				svg_path=paths["template_circles_svg"],
				probe_geometry=unit_probe_geometry,
				unit_id=unit_id,
				propagation_order_rank_by_channel=circles_rank_map,
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

			prop_cfg = inputs.per_unit_outputs.propagation_plots
			prop_order_payload = compute_propagation_channel_order(
				template_c_by_t=np.asarray(prop_template),
				config=prop_cfg,
				channel_indices=np.arange(int(prop_template.shape[0]), dtype=int),
			)
			trace_label_mode = str(getattr(prop_cfg, "trace_label_mode", "electrode_id") or "electrode_id").strip().lower()
			canonical_prop_order = np.asarray(prop_order_payload.get("ordered_channel_indices", []), dtype=int)
			prop_window_channel_indices = _select_window_from_order_with_strategy(
				template_c_by_t=np.asarray(prop_template),
				ordered_channel_indices=canonical_prop_order,
				window_size=int(prop_cfg.top_channels),
				window_strategy=str(getattr(prop_cfg, "window_strategy", "max_ptp_sum") or "max_ptp_sum"),
			)
			order_anchor_channel: int | None = None
			if bool(getattr(prop_cfg, "force_min_neg_peak_index_zero", False)):
				anchor_candidate = prop_order_payload.get("max_ptp_channel", None)
				if anchor_candidate is not None:
					order_anchor_channel = int(anchor_candidate)
			if order_anchor_channel is None and int(prop_window_channel_indices.shape[0]) > 0:
				order_anchor_channel = int(prop_window_channel_indices[0])

			order_number_by_channel = _channel_order_labels_for_mode(
				prop_order_payload,
				trace_label_mode=trace_label_mode,
				use_relative_signed_numbers=bool(prop_cfg.relative_signed_order_numbers),
				order_index_anchor_channel=order_anchor_channel,
			)

			circles_order_payload = compute_propagation_channel_order(
				template_c_by_t=np.asarray(template_circles),
				config=prop_cfg,
				channel_indices=np.arange(int(template_circles.shape[0]), dtype=int),
			)
			circles_anchor_channel = order_anchor_channel
			if bool(getattr(prop_cfg, "force_min_neg_peak_index_zero", False)):
				circles_anchor = circles_order_payload.get("max_ptp_channel", None)
				if circles_anchor is not None:
					circles_anchor_channel = int(circles_anchor)

			circles_order_number_by_channel = _channel_order_labels_for_mode(
				circles_order_payload,
				trace_label_mode=trace_label_mode,
				use_relative_signed_numbers=bool(prop_cfg.relative_signed_order_numbers),
				order_index_anchor_channel=circles_anchor_channel,
			)

			if bool(getattr(prop_cfg, "debug_ordering", False)):
				window_start = None
				window_end = None
				if int(prop_window_channel_indices.shape[0]) > 0 and int(canonical_prop_order.shape[0]) > 0:
					first = int(prop_window_channel_indices[0])
					hits = np.flatnonzero(canonical_prop_order == first)
					if hits.size > 0:
						window_start = int(hits[0])
						window_end = int(window_start + int(prop_window_channel_indices.shape[0]))
				prop_vals = list(order_number_by_channel.values())
				circles_vals = list(circles_order_number_by_channel.values())
				LOGGER.debug(
					"Propagation ordering debug: unit_id=%s canonical_n=%d selected_n=%d window_strategy=%s window=[%s,%s) "
					"trace_mode=%s ordering_latency_mode=%s anchor_channel=%s prop_label_minmax=(%s,%s) circles_label_minmax=(%s,%s)",
					unit_id,
					int(canonical_prop_order.shape[0]),
					int(prop_window_channel_indices.shape[0]),
					str(getattr(prop_cfg, "window_strategy", "max_ptp_sum")),
					window_start,
					window_end,
					trace_label_mode,
					str(getattr(prop_cfg, "ordering_latency_mode", "abs_peak")),
					order_anchor_channel,
					(min(prop_vals) if prop_vals else None),
					(max(prop_vals) if prop_vals else None),
					(min(circles_vals) if circles_vals else None),
					(max(circles_vals) if circles_vals else None),
				)

			prop_outputs = render_propagation_plot(
				template=prop_template,
				locations_xy=prop_locs,
				config=prop_cfg,
				pdf_path=paths["propagation_plot_pdf"],
				png_path=paths["propagation_plot_png"],
				svg_path=paths["propagation_plot_svg"],
				write_svg=bool(prop_cfg.write_svg),
				probe_geometry=unit_probe_geometry,
				channel_labels_by_row=(
					merged_electrode_ids
					if (prop_source == "merged_contributing" and merged_electrode_ids is not None and int(len(merged_electrode_ids)) == int(prop_template.shape[0]))
					else None
				),
				trace_order_label_by_channel=order_number_by_channel,
				channel_indices=prop_window_channel_indices,
			)
			need_numbered_png = bool(prop_cfg.write_circles_template_numbered_png) or bool(prop_cfg.write_propagation_2panel_png)
			need_numbered_svg = bool(prop_cfg.write_circles_template_numbered_svg) or bool(prop_cfg.write_propagation_2panel_svg)
			right_png_path = paths["circles_template_numbered_png"] if bool(prop_cfg.write_circles_template_numbered_png) else paths["propagation_plot_right_temp_png"]
			right_svg_path = paths["circles_template_numbered_svg"] if bool(prop_cfg.write_circles_template_numbered_svg) else paths["propagation_plot_right_temp_svg"]

			circles_numbered_cfg = replace(
				inputs.per_unit_outputs.template_circles,
				write_png=bool(need_numbered_png),
				write_svg=bool(need_numbered_svg),
				show_propagation_order_labels=True,
				dpi=float(prop_cfg.right_panel_png_dpi),
			)
			circles_numbered_outputs = render_template_circles_plot(
				template=template_circles,
				locations_xy=locs_circles,
				config=circles_numbered_cfg,
				png_path=right_png_path,
				svg_path=right_svg_path,
				probe_geometry=unit_probe_geometry,
				unit_id=unit_id,
				propagation_order_rank_by_channel=circles_order_number_by_channel,
			)

			if bool(prop_cfg.write_circles_template_numbered_png) and circles_numbered_outputs.get("template_circles_png"):
				prop_outputs["circles_template_numbered_png"] = str(paths["circles_template_numbered_png"])
			if bool(prop_cfg.write_circles_template_numbered_svg) and circles_numbered_outputs.get("template_circles_svg"):
				prop_outputs["circles_template_numbered_svg"] = str(paths["circles_template_numbered_svg"])

			if bool(prop_cfg.write_propagation_2panel_png) and circles_numbered_outputs.get("template_circles_png") and prop_outputs.get("propagation_plot_png"):
				try:
					compose_dpi = prop_cfg.composed_png_dpi
					if compose_dpi is None:
						compose_dpi = prop_cfg.right_panel_png_dpi
					compose_png_side_by_side(
						left_png_path=paths["propagation_plot_png"],
						right_png_path=right_png_path,
						output_png_path=paths["propagation_2panel_png"],
						gap_fraction=float(prop_cfg.right_panel_gap_fraction),
						right_width_scale=float(prop_cfg.right_panel_width_scale),
						output_dpi=float(compose_dpi),
					)
					prop_outputs["propagation_2panel_png"] = str(paths["propagation_2panel_png"])
				except Exception as compose_png_exc:
					LOGGER.warning(
						"Failed to compose propagation_2panel PNG for unit %s: %s",
						unit_id,
						compose_png_exc,
					)

			if bool(prop_cfg.write_propagation_2panel_svg) and circles_numbered_outputs.get("template_circles_svg") and prop_outputs.get("propagation_plot_svg"):
				try:
					compose_svg_side_by_side(
						left_svg_path=paths["propagation_plot_svg"],
						right_svg_path=right_svg_path,
						output_svg_path=paths["propagation_2panel_svg"],
						gap_fraction=float(prop_cfg.right_panel_gap_fraction),
						right_width_scale=float(prop_cfg.right_panel_width_scale),
					)
					prop_outputs["propagation_2panel_svg"] = str(paths["propagation_2panel_svg"])
				except Exception as compose_exc:
					LOGGER.warning(
						"Failed to compose propagation_2panel SVG for unit %s: %s",
						unit_id,
						compose_exc,
					)

			if not bool(prop_cfg.write_circles_template_numbered_png):
				try:
					if right_png_path.exists():
						right_png_path.unlink()
				except Exception:
					pass
			if not bool(prop_cfg.write_circles_template_numbered_svg):
				try:
					if right_svg_path.exists():
						right_svg_path.unlink()
				except Exception:
					pass
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

		def _finalize_grid_svg_output(
			*,
			raw_outputs: dict[str, str],
			write_svg: bool,
			keep_temp_svg: bool,
			temp_svg_output_key: str,
			final_svg_output_key: str,
			temp_svg_path: Path,
			final_svg_path: Path,
			report_name: str,
		) -> dict[str, str]:
			if not bool(write_svg):
				raw_outputs.pop(temp_svg_output_key, None)
				return raw_outputs
			if temp_svg_output_key not in raw_outputs:
				return raw_outputs
			try:
				final_svg_path.parent.mkdir(parents=True, exist_ok=True)
				shutil.copyfile(temp_svg_path, final_svg_path)
				raw_outputs[final_svg_output_key] = str(final_svg_path)
			except Exception as exc:
				LOGGER.warning("Failed to finalize %s SVG output: %s", report_name, exc)
			if not bool(keep_temp_svg):
				raw_outputs.pop(temp_svg_output_key, None)
				try:
					if temp_svg_path.exists():
						temp_svg_path.unlink()
				except Exception:
					pass
			return raw_outputs

		overlay_paths = [
			Path(u.outputs["template_wf_overlay_png"])
			for u in unit_results
			if "template_wf_overlay_png" in u.outputs
		]
		LOGGER.info("Templates reports wf_overlay_grid inputs=%d", len(overlay_paths))
		wf_grid_outputs = render_wf_overlay_grid_from_assets(
			overlay_png_paths=overlay_paths,
			config=inputs.reports.wf_overlay_grid,
			pdf_path=report_paths["wf_overlay_grid_pdf"],
			png_path=report_paths["wf_overlay_grid_png"],
			write_svg=bool(inputs.reports.wf_overlay_grid.write_svg),
			svg_path=report_paths["wf_overlay_grid_temp_svg"],
			svg_output_key="wf_overlay_grid_temp_svg",
		)
		wf_grid_outputs = _finalize_grid_svg_output(
			raw_outputs=wf_grid_outputs,
			write_svg=bool(inputs.reports.wf_overlay_grid.write_svg),
			keep_temp_svg=bool(inputs.reports.wf_overlay_grid.keep_temp_svg),
			temp_svg_output_key="wf_overlay_grid_temp_svg",
			final_svg_output_key="wf_overlay_grid_svg",
			temp_svg_path=report_paths["wf_overlay_grid_temp_svg"],
			final_svg_path=report_paths["wf_overlay_grid_svg"],
			report_name="wf_overlay_grid",
		)
		report_outputs.update(wf_grid_outputs)
		circles_map_paths = [
			Path(u.outputs["template_circles_png"])
			for u in unit_results
			if "template_circles_png" in u.outputs
		]
		LOGGER.info("Templates reports circles_map_grid inputs=%d", len(circles_map_paths))
		circles_grid_outputs = render_footprint_map_grid_from_assets(
			image_paths=circles_map_paths,
			config=inputs.reports.footprint_grids.circles_map_grid,
			pdf_path=report_paths["template_circles_map_grid_pdf"],
			png_path=report_paths["template_circles_map_grid_png"],
			write_svg=bool(inputs.reports.footprint_grids.circles_map_grid.write_svg),
			svg_path=report_paths["template_circles_map_grid_temp_svg"],
			svg_output_key="template_circles_map_grid_temp_svg",
			pdf_output_key="template_circles_map_grid_pdf",
			png_output_key="template_circles_map_grid_png",
			title="Template circles map grid",
		)
		circles_grid_outputs = _finalize_grid_svg_output(
			raw_outputs=circles_grid_outputs,
			write_svg=bool(inputs.reports.footprint_grids.circles_map_grid.write_svg),
			keep_temp_svg=bool(inputs.reports.footprint_grids.circles_map_grid.keep_temp_svg),
			temp_svg_output_key="template_circles_map_grid_temp_svg",
			final_svg_output_key="template_circles_map_grid_svg",
			temp_svg_path=report_paths["template_circles_map_grid_temp_svg"],
			final_svg_path=report_paths["template_circles_map_grid_svg"],
			report_name="circles_map_grid",
		)
		report_outputs.update(circles_grid_outputs)
		amp_map_paths = [
			Path(u.outputs["footprint_amplitude_map_png"])
			for u in unit_results
			if "footprint_amplitude_map_png" in u.outputs
		]
		LOGGER.info("Templates reports amplitude_map_grid inputs=%d", len(amp_map_paths))
		amp_grid_outputs = render_footprint_map_grid_from_assets(
			image_paths=amp_map_paths,
			config=inputs.reports.footprint_grids.amplitude_map_grid,
			pdf_path=report_paths["footprint_amplitude_map_grid_pdf"],
			png_path=report_paths["footprint_amplitude_map_grid_png"],
			write_svg=bool(inputs.reports.footprint_grids.amplitude_map_grid.write_svg),
			svg_path=report_paths["footprint_amplitude_map_grid_temp_svg"],
			svg_output_key="footprint_amplitude_map_grid_temp_svg",
			pdf_output_key="footprint_amplitude_map_grid_pdf",
			png_output_key="footprint_amplitude_map_grid_png",
			title="Template footprint amplitude map grid",
		)
		amp_grid_outputs = _finalize_grid_svg_output(
			raw_outputs=amp_grid_outputs,
			write_svg=bool(inputs.reports.footprint_grids.amplitude_map_grid.write_svg),
			keep_temp_svg=bool(inputs.reports.footprint_grids.amplitude_map_grid.keep_temp_svg),
			temp_svg_output_key="footprint_amplitude_map_grid_temp_svg",
			final_svg_output_key="footprint_amplitude_map_grid_svg",
			temp_svg_path=report_paths["footprint_amplitude_map_grid_temp_svg"],
			final_svg_path=report_paths["footprint_amplitude_map_grid_svg"],
			report_name="amplitude_map_grid",
		)
		report_outputs.update(amp_grid_outputs)
		lat_map_paths = [
			Path(u.outputs["footprint_latency_map_png"])
			for u in unit_results
			if "footprint_latency_map_png" in u.outputs
		]
		LOGGER.info("Templates reports latency_map_grid inputs=%d", len(lat_map_paths))
		lat_grid_outputs = render_footprint_map_grid_from_assets(
			image_paths=lat_map_paths,
			config=inputs.reports.footprint_grids.latency_map_grid,
			pdf_path=report_paths["footprint_latency_map_grid_pdf"],
			png_path=report_paths["footprint_latency_map_grid_png"],
			write_svg=bool(inputs.reports.footprint_grids.latency_map_grid.write_svg),
			svg_path=report_paths["footprint_latency_map_grid_temp_svg"],
			svg_output_key="footprint_latency_map_grid_temp_svg",
			pdf_output_key="footprint_latency_map_grid_pdf",
			png_output_key="footprint_latency_map_grid_png",
			title="Template footprint latency map grid",
		)
		lat_grid_outputs = _finalize_grid_svg_output(
			raw_outputs=lat_grid_outputs,
			write_svg=bool(inputs.reports.footprint_grids.latency_map_grid.write_svg),
			keep_temp_svg=bool(inputs.reports.footprint_grids.latency_map_grid.keep_temp_svg),
			temp_svg_output_key="footprint_latency_map_grid_temp_svg",
			final_svg_output_key="footprint_latency_map_grid_svg",
			temp_svg_path=report_paths["footprint_latency_map_grid_temp_svg"],
			final_svg_path=report_paths["footprint_latency_map_grid_svg"],
			report_name="latency_map_grid",
		)
		report_outputs.update(lat_grid_outputs)
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
	quality_check_aggregate, quality_check_aggregate_json = _build_quality_check_aggregate(
		templates_out_dir=templates_out_dir,
		inputs=inputs,
	)
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
		"execution_inputs": {
			"concat_analyzer_relpath": inputs.concat_analyzer_relpath,
			"preproc_seg_sources_reldir": inputs.preproc_seg_sources_reldir,
		},
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
		"quality_checks": {
			"enabled": bool(inputs.quality_checks.enable),
			"outputs": {
				"run_level": {
					"check_for_multiple_peaks_at_channel_templates": {
						"write_json": bool(inputs.quality_checks_outputs.check_for_multiple_peaks_at_channel_templates.write_json),
						"json_relpath": str(inputs.quality_checks_outputs.check_for_multiple_peaks_at_channel_templates.json_relpath),
					},
				},
				"per_unit": {
					"check_for_multiple_peaks_at_channel_templates": {
						"write_json": bool(inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates.write_json),
						"json_relpath": str(inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates.json_relpath),
						"plot": {
							"write_png": bool(inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates.plot.write_png),
							"write_svg": bool(inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates.plot.write_svg),
							"relpath": str(inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates.plot.relpath),
						},
					},
				},
			},
			"check_for_multiple_peaks_at_channel_templates": {
				"enabled": bool(inputs.quality_checks.check_for_multiple_peaks_at_channel_templates.enable),
				"prominence_fraction": float(inputs.quality_checks.check_for_multiple_peaks_at_channel_templates.prominence_fraction),
				"min_separation_samples": int(inputs.quality_checks.check_for_multiple_peaks_at_channel_templates.min_separation_samples),
				"max_peaks_per_channel": int(inputs.quality_checks.check_for_multiple_peaks_at_channel_templates.max_peaks_per_channel),
			},
			"aggregate_json": (None if quality_check_aggregate_json is None else str(quality_check_aggregate_json)),
			"aggregate": quality_check_aggregate,
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
