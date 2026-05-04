from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
from dataclasses import replace
import gc
import logging
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Callable

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.shared.grid_sorting import (
	coerce_grid_sort_metrics,
	compute_template_grid_sort_metrics,
	grid_sort_key_for_unit,
	normalize_grid_sort_by,
)
from axon_recon.pipeline.execution.phase_chain import PhaseDescriptor, run_phase_chain
from axon_recon.pipeline.execution.progress import add_current_progress_total, advance_current_progress

from .core.build_templates import build_templates_phase_from_payloads, build_templates_phase_from_unit_payloads
from .core.compute_template_similarity import (
	TemplateSimilarityUnitInput,
	build_template_similarity_phase_summary,
)
from .core.merge import materialize_templates_from_spikeinterface
from .core.plot_templates import (
	build_plot_templates_phase_inputs,
	build_plot_templates_phase_summary,
	excluded_plot_output_keys,
	propagation_outputs_requested,
	requested_plot_output_keys,
)
from .core.report_templates import (
	build_report_templates_phase_summary,
	report_templates_pdf_requested,
)
from .core.unit_labels import count_labels, filter_unit_ids_by_labels, load_unit_labels_from_spikesorting
from .core.quality_checks import detect_multiple_negative_peaks
from .core.render import (
	compose_png_side_by_side,
	compose_svg_side_by_side,
	compute_propagation_channel_order,
	finalize_grid_svg_output,
	render_footprint_amplitude_map,
	render_footprint_map_grid_from_assets,
	render_footprint_latency_map,
	render_multi_source_pdf,
	render_template_report_pdf,
	render_propagation_plot,
	render_template_circles_plot,
	render_unit_locations_report,
	render_template_plot,
	render_template_wf_overlay,
	render_topographical_amplitude_footprint,
	render_topographical_latency_footprint,
	render_wf_overlay_grid_from_assets,
)
from .io import (
	MATERIALIZED_TEMPLATES_CACHE_RELPATH,
	load_materialized_source_payload,
	load_materialized_overlay_waveforms,
	load_materialized_merged_electrode_ids,
	read_json,
	resolve_materialized_source_payload_unit_dir,
	resolve_materialized_templates_dirs,
	resolve_report_output_paths,
	resolve_similarity_output_paths,
	resolve_unit_output_paths,
	write_materialized_merged_electrode_ids,
	write_materialized_source_payload,
	write_materialized_unit_templates,
	write_json,
)
from .integrations.spikeinterface_extract import (
	build_unit_source_payload,
	discover_cached_spikeinterface_analyzer_source_names,
	iter_spikeinterface_analyzers,
	load_cached_spikeinterface_analyzers,
	load_spikeinterface_analyzers,
)
from .models.inputs import ProbeGeometryConfig, TemplatesInputs, TimeUpsampleConfig
from .models.results import TemplatesResult, UnitTemplatesResult


LOGGER = logging.getLogger("axon_recon.templates")

NOISY_PLOT_LOGGER_NAMES: tuple[str, ...] = (
	"matplotlib",
	"matplotlib.font_manager",
	"PIL",
	"PIL.PngImagePlugin",
	"fontTools",
)

DEFAULT_INTERNAL_TEMPLATES_PHASE_SEQUENCE: tuple[str, ...] = (
	"resolve_sources",
	"analyzers",
	"extract_template_segments",
	"build_templates",
	"compute_template_similarity",
	"plot_templates",
	"report_templates",
	"reports",
)


def _as_positive_float_or_none(value: Any) -> float | None:
	try:
		parsed = float(value)
	except Exception:
		return None
	if not np.isfinite(parsed) or parsed <= 0.0:
		return None
	return float(parsed)


def _positive_int_or_none(value: Any) -> int | None:
	if value is None:
		return None
	try:
		parsed = int(value)
	except (TypeError, ValueError):
		return None
	return parsed if parsed > 0 else None


def _templates_applied_debug_limits(inputs: TemplatesInputs) -> dict[str, Any]:
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


def _compute_unit_location_from_template(
	*,
	unit_id: Any,
	template_c_by_t: np.ndarray,
	locations_xy: np.ndarray,
	source: str,
) -> dict[str, Any] | None:
	template = np.asarray(template_c_by_t, dtype=float)
	locs = np.asarray(locations_xy, dtype=float)
	if template.ndim != 2 or locs.ndim != 2:
		return None
	if int(locs.shape[1]) < 2:
		return None
	if int(template.shape[0]) != int(locs.shape[0]):
		if int(template.shape[1]) == int(locs.shape[0]):
			template = np.asarray(template.T, dtype=float)
		else:
			return None
	if int(template.shape[0]) == 0:
		return None
	try:
		neg_peaks = np.min(template, axis=1)
		channel_index = int(np.argmin(neg_peaks))
		x_um = float(locs[channel_index, 0])
		y_um = float(locs[channel_index, 1])
	except Exception:
		return None
	if (not np.isfinite(x_um)) or (not np.isfinite(y_um)):
		return None
	return {
		"unit_id": unit_id,
		"x_um": float(x_um),
		"y_um": float(y_um),
		"channel_index": int(channel_index),
		"method": "max_negative_peak",
		"source": str(source),
	}


def _load_templates_unit_grid_sort_metrics(
	*,
	templates_out_dir: Path,
	unit_id: Any,
	inputs: TemplatesInputs,
) -> dict[str, float]:
	paths = resolve_unit_output_paths(
		templates_out_dir=templates_out_dir,
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


def _load_templates_unit_location_row(
	*,
	templates_out_dir: Path,
	unit_id: Any,
	inputs: TemplatesInputs,
) -> dict[str, Any] | None:
	paths = resolve_unit_output_paths(
		templates_out_dir=templates_out_dir,
		unit_id=unit_id,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	unit_summary_json = paths["unit_summary_json"]
	if unit_summary_json.exists():
		try:
			payload = read_json(unit_summary_json)
		except Exception:
			payload = None
		if isinstance(payload, dict):
			unit_location = payload.get("unit_location", None)
			if isinstance(unit_location, dict):
				try:
					x = float(unit_location.get("x_um", None))
					y = float(unit_location.get("y_um", None))
				except Exception:
					x, y = None, None
				if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
					return {
						"unit_id": unit_id,
						"x_um": float(x),
						"y_um": float(y),
						"channel_index": unit_location.get("channel_index", None),
						"source": unit_location.get("source", "merged_contributing"),
					}

	for merged_root in _materialized_merged_root_candidates(templates_out_dir):
		merged_unit_dir = merged_root / f"unit_{unit_id}"
		if merged_unit_dir.exists():
			try:
				merged_template, merged_locs = _load_merged_unit(merged_unit_dir)
			except Exception:
				return None
			return _compute_unit_location_from_template(
				unit_id=unit_id,
				template_c_by_t=merged_template,
				locations_xy=merged_locs,
				source="merged_contributing",
			)
	return None


def _materialized_templates_root_candidates(templates_out_dir: Path) -> list[Path]:
	return [
		templates_out_dir / MATERIALIZED_TEMPLATES_CACHE_RELPATH,
		templates_out_dir / "templates",
	]


def _materialized_merged_root_candidates(templates_out_dir: Path) -> list[Path]:
	return [
		*[root / "merged" for root in _materialized_templates_root_candidates(templates_out_dir)],
		templates_out_dir / "merged_units",
	]


def _materialized_full_root_candidates(templates_out_dir: Path) -> list[Path]:
	return [
		*[root / "full" for root in _materialized_templates_root_candidates(templates_out_dir)],
		templates_out_dir / "full_channels_templates",
	]


def _materialized_concat_channel_locations_candidates(templates_out_dir: Path) -> list[Path]:
	return [root / "concat_channel_locations_xy.npy" for root in _materialized_templates_root_candidates(templates_out_dir)]


def _materialized_concat_unit_locations_candidates(templates_out_dir: Path) -> list[Path]:
	return [
		templates_out_dir / "units" / "concat_unit_locations.json",
		templates_out_dir / "templates" / "concat_unit_locations.json",
	]


def _unit_dir_candidates_for_id(root_dir: Path, unit_id: Any) -> list[Path]:
	candidates: list[Path] = []
	try:
		uid_int = int(unit_id)
		candidates.extend([root_dir / f"unit_{uid_int}", root_dir / f"{uid_int:04d}", root_dir / str(uid_int)])
	except Exception:
		candidates.extend([root_dir / f"unit_{unit_id}", root_dir / str(unit_id)])
	seen: set[Path] = set()
	ordered: list[Path] = []
	for path in candidates:
		if path in seen:
			continue
		seen.add(path)
		ordered.append(path)
	return ordered


def _load_templates_concat_channel_locations(
	*,
	templates_out_dir: Path,
	unit_ids_for_priority: list[Any],
) -> np.ndarray | None:
	global_concat_locs = next(
		(path for path in _materialized_concat_channel_locations_candidates(templates_out_dir) if path.exists()),
		None,
	)
	if global_concat_locs is not None:
		try:
			locs = np.asarray(np.load(global_concat_locs), dtype=float)
		except Exception:
			locs = np.zeros((0, 2), dtype=float)
		if locs.ndim == 2 and int(locs.shape[1]) >= 2 and int(locs.shape[0]) > 0:
			locs = np.asarray(locs[:, :2], dtype=float)
			finite = np.isfinite(locs).all(axis=1)
			if bool(np.any(finite)):
				return np.asarray(locs[finite, :], dtype=float)

	best_locs: np.ndarray | None = None
	full_roots = _materialized_full_root_candidates(templates_out_dir)
	for full_root in full_roots:
		if not full_root.exists():
			continue
		candidate_dirs: list[Path] = []
		for unit_id in list(unit_ids_for_priority):
			candidate_dirs.extend(_unit_dir_candidates_for_id(full_root, unit_id))
		candidate_dirs.extend(sorted(full_root.glob("unit_*")))
		seen: set[Path] = set()
		for unit_dir in candidate_dirs:
			if unit_dir in seen:
				continue
			seen.add(unit_dir)
			locs_path = unit_dir / "full_channel_locations_xy.npy"
			if not locs_path.exists():
				continue
			try:
				locs = np.asarray(np.load(locs_path), dtype=float)
			except Exception:
				continue
			if locs.ndim != 2 or int(locs.shape[1]) < 2 or int(locs.shape[0]) == 0:
				continue
			locs = np.asarray(locs[:, :2], dtype=float)
			finite = np.isfinite(locs).all(axis=1)
			if not bool(np.any(finite)):
				continue
			clean = np.asarray(locs[finite, :], dtype=float)
			if best_locs is None or int(clean.shape[0]) > int(best_locs.shape[0]):
				best_locs = clean
	return best_locs


def _infer_chip_grid_from_probe_geometry(probe_geometry: ProbeGeometryConfig | None) -> np.ndarray | None:
	if probe_geometry is None:
		return None
	pitch = _as_positive_float_or_none(getattr(probe_geometry, "pitch_um", None))
	active_x = _as_positive_float_or_none(getattr(probe_geometry, "active_area_um_x", None))
	active_y = _as_positive_float_or_none(getattr(probe_geometry, "active_area_um_y", None))
	if pitch is None or active_x is None or active_y is None:
		return None
	nx = int(max(1, np.round(float(active_x) / float(pitch))))
	ny = int(max(1, np.round(float(active_y) / float(pitch))))
	x_vals = np.arange(nx, dtype=float) * float(pitch)
	y_vals = np.arange(ny, dtype=float) * float(pitch)
	return np.asarray([[float(x), float(y)] for y in y_vals for x in x_vals], dtype=float)


def _expected_chip_channel_count(probe_geometry: ProbeGeometryConfig | None) -> int | None:
	if probe_geometry is None:
		return None
	pitch = _as_positive_float_or_none(getattr(probe_geometry, "pitch_um", None))
	active_x = _as_positive_float_or_none(getattr(probe_geometry, "active_area_um_x", None))
	active_y = _as_positive_float_or_none(getattr(probe_geometry, "active_area_um_y", None))
	if pitch is None or active_x is None or active_y is None:
		return None
	nx = int(max(1, np.round(float(active_x) / float(pitch))))
	ny = int(max(1, np.round(float(active_y) / float(pitch))))
	return int(max(1, nx * ny))


def _resolve_well_relative_path_candidates(*, well_dirs: list[Path], relpath_tokens: list[str]) -> list[Path]:
	seen: set[Path] = set()
	resolved: list[Path] = []
	for raw_token in relpath_tokens:
		token = str(raw_token or "").strip()
		if not token:
			continue
		abs_path = Path(token).expanduser()
		if abs_path.is_absolute() and abs_path.exists():
			candidate = abs_path.resolve()
			if candidate not in seen:
				seen.add(candidate)
				resolved.append(candidate)
	for well_dir in well_dirs:
		for raw_token in relpath_tokens:
			token = str(raw_token or "").strip()
			if not token:
				continue
			path_token = Path(token).expanduser()
			if path_token.is_absolute():
				path_token = Path(str(path_token).lstrip("/"))
			candidate = (well_dir / path_token).resolve()
			if candidate in seen:
				continue
			seen.add(candidate)
			resolved.append(candidate)
	return resolved


def _load_spikesort_locations_metadata(
	*,
	well_out_dir: Path,
	alternate_well_out_dirs: list[Path],
	inputs: TemplatesInputs,
) -> tuple[np.ndarray | None, dict[str, dict[str, float]]]:
	well_dirs = [well_out_dir, *list(alternate_well_out_dirs)]
	relpath_candidates: list[str] = []
	if inputs.concat_analyzer_relpath is not None:
		relpath_candidates.append(str(inputs.concat_analyzer_relpath))
	relpath_candidates.extend(
		[
			"/spikesort_outputs/analyzer_output",
			"/stg2_spikesorting_outputs/analyzer_output",
			"spikesort_outputs/analyzer_output",
			"stg2_spikesorting_outputs/analyzer_output",
		]
	)
	analyzer_dirs = _resolve_well_relative_path_candidates(
		well_dirs=well_dirs,
		relpath_tokens=relpath_candidates,
	)

	for analyzer_dir in analyzer_dirs:
		if not analyzer_dir.exists():
			continue
		try:
			import spikeinterface.full as si  # type: ignore[import-not-found]
			analyzer = si.load_sorting_analyzer(analyzer_dir)
		except Exception:
			continue

		concat_locs: np.ndarray | None = None
		try:
			locs = np.asarray(analyzer.recording.get_channel_locations(), dtype=float)
			if locs.ndim == 2 and int(locs.shape[1]) >= 2 and int(locs.shape[0]) > 0:
				locs = np.asarray(locs[:, :2], dtype=float)
				finite = np.isfinite(locs).all(axis=1)
				if bool(np.any(finite)):
					concat_locs = np.asarray(locs[finite, :], dtype=float)
		except Exception:
			concat_locs = None

		original_by_unit: dict[str, dict[str, float]] = {}
		try:
			if analyzer.has_extension("unit_locations"):
				unit_locations = analyzer.get_extension("unit_locations").get_data()
				if hasattr(unit_locations, "to_numpy"):
					unit_locations = unit_locations.to_numpy()
				loc_arr = np.asarray(unit_locations, dtype=float)
				unit_ids_raw = getattr(getattr(analyzer, "sorting", None), "unit_ids", None)
				if unit_ids_raw is None:
					unit_ids_raw = getattr(analyzer, "unit_ids", [])
				unit_ids = list(unit_ids_raw or [])
				if loc_arr.ndim == 2 and int(loc_arr.shape[1]) >= 2 and int(loc_arr.shape[0]) == int(len(unit_ids)):
					for uid, xy in zip(unit_ids, loc_arr[:, :2], strict=False):
						x = float(xy[0])
						y = float(xy[1])
						if (not np.isfinite(x)) or (not np.isfinite(y)):
							continue
						uid_key = str(uid).strip()
						original_by_unit[uid_key] = {"x_um": x, "y_um": y}
						try:
							original_by_unit[str(int(uid))] = {"x_um": x, "y_um": y}
						except Exception:
							pass
		except Exception:
			original_by_unit = {}

		if concat_locs is not None or bool(original_by_unit):
			return concat_locs, original_by_unit

	return None, {}


def _load_templates_original_unit_locations(
	*,
	templates_out_dir: Path,
) -> dict[str, dict[str, float]]:
	original_locations_path = next(
		(path for path in _materialized_concat_unit_locations_candidates(templates_out_dir) if path.exists()),
		None,
	)
	if original_locations_path is None:
		return {}
	try:
		payload = read_json(original_locations_path)
	except Exception:
		return {}
	if not isinstance(payload, list):
		return {}
	rows: dict[str, dict[str, float]] = {}
	for row in payload:
		if not isinstance(row, dict):
			continue
		uid_raw = row.get("unit_id", None)
		if uid_raw is None:
			continue
		try:
			x = float(row.get("x_um", None))
			y = float(row.get("y_um", None))
		except Exception:
			continue
		if (not np.isfinite(x)) or (not np.isfinite(y)):
			continue
		uid_key = str(uid_raw).strip()
		rows[uid_key] = {"x_um": x, "y_um": y}
		try:
			rows[str(int(uid_raw))] = {"x_um": x, "y_um": y}
		except Exception:
			pass
	return rows


def _load_templates_channel_locations_by_unit(
	*,
	templates_out_dir: Path,
	unit_ids: list[Any],
) -> dict[str, np.ndarray]:
	merged_roots = _materialized_merged_root_candidates(templates_out_dir)
	loaded: dict[str, np.ndarray] = {}
	for unit_id in list(unit_ids):
		unit_key = str(unit_id)
		for merged_root in merged_roots:
			if not merged_root.exists():
				continue
			for unit_dir in _unit_dir_candidates_for_id(merged_root, unit_id):
				if not unit_dir.exists():
					continue
				loc_candidates = [
					unit_dir / "merged_contributing_channel_locations.npy",
					unit_dir / "merged_channel_locations.npy",
				]
				for locs_path in loc_candidates:
					if not locs_path.exists():
						continue
					try:
						locs = np.asarray(np.load(locs_path), dtype=float)
					except Exception:
						continue
					if locs.ndim != 2 or int(locs.shape[1]) < 2 or int(locs.shape[0]) == 0:
						continue
					locs = np.asarray(locs[:, :2], dtype=float)
					finite = np.isfinite(locs).all(axis=1)
					if not bool(np.any(finite)):
						continue
					loaded[unit_key] = np.asarray(locs[finite, :], dtype=float)
					break
				if unit_key in loaded:
					break
			if unit_key in loaded:
				break
	return loaded


def _sort_template_units_for_reports(
	*,
	unit_results: list[UnitTemplatesResult],
	templates_out_dir: Path,
	inputs: TemplatesInputs,
	sort_by: str,
) -> list[UnitTemplatesResult]:
	sort_mode = normalize_grid_sort_by(sort_by, default="unit_id")
	metrics_by_unit: dict[str, dict[str, float]] | None = None
	if sort_mode != "unit_id":
		metrics_by_unit = {}
		for unit_result in unit_results:
			metrics_by_unit[str(unit_result.unit_id).strip()] = _load_templates_unit_grid_sort_metrics(
				templates_out_dir=templates_out_dir,
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


def _is_unit_scoped_templates_run(inputs: TemplatesInputs) -> bool:
	return bool(inputs.unit_ids) and len(inputs.unit_ids) == 1


def _reports_replot_requested(inputs: TemplatesInputs) -> bool:
	return bool(inputs.reports.replot_from_disk) or bool(inputs.force_rereport)


def _should_preserve_templates_reports(inputs: TemplatesInputs) -> bool:
	if not _is_unit_scoped_templates_run(inputs):
		return False
	if not (bool(inputs.force_restart) or bool(inputs.force_replot) or bool(inputs.force_replot_per_unit)):
		return False
	if _reports_replot_requested(inputs):
		return False
	return not bool(inputs.reports.overwrite_on_unit_rerun)


def _collect_existing_templates_report_outputs(
	*,
	templates_out_dir: Path,
	inputs: TemplatesInputs,
) -> dict[str, str]:
	report_outputs: dict[str, str] = {}
	for key, path in resolve_report_output_paths(templates_out_dir=templates_out_dir, reports=inputs.reports).items():
		if path.exists():
			report_outputs[key] = str(path)
	if bool(inputs.phases.compute_template_similarity.enabled):
		for key, path in resolve_similarity_output_paths(
			templates_out_dir=templates_out_dir,
			similarity=inputs.phases.compute_template_similarity,
		).items():
			if path.exists():
				report_outputs[key] = str(path)
	return report_outputs


def collect_templates_result_from_outputs(inputs: TemplatesInputs) -> TemplatesResult:
	well_out_dir, _, templates_out_dir, _ = _resolve_templates_phase_environment(inputs)
	unit_ids = list(inputs.unit_ids) if inputs.unit_ids is not None else _discover_unit_ids_from_unit_summaries(templates_out_dir)
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	unit_ids = _apply_unit_label_filter(inputs, unit_ids, well_out_dir, context="collect_result")
	unit_results: list[UnitTemplatesResult] = []
	for unit_id in unit_ids:
		paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		unit_result = _load_unit_result_from_summary(unit_id=unit_id, unit_summary_json=paths["unit_summary_json"])
		if unit_result is not None:
			unit_results.append(unit_result)
	report_outputs = _collect_existing_templates_report_outputs(templates_out_dir=templates_out_dir, inputs=inputs)
	summary_json = _write_reconstruct_templates_summary(
		inputs=inputs,
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
		unit_results=unit_results,
		report_outputs=report_outputs,
		analyzer_cache_dir=None,
		upsampling_decisions_by_unit={},
		reports_replot_requested=_reports_replot_requested(inputs),
		report_only_rerun=False,
		preserve_stage_reports=_should_preserve_templates_reports(inputs),
	)
	return TemplatesResult(
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
		summary_json=summary_json,
		units=unit_results,
		report_outputs=report_outputs,
	)


def run_reconstruct_templates_pipeline(inputs: TemplatesInputs) -> TemplatesResult:
	if inputs.phase_sequence is None:
		return _run_reconstruct_templates_pipeline_monolithic(inputs)
	phase_sequence = tuple(
		_normalize_reconstruct_templates_phase_name(phase)
		for phase in inputs.phase_sequence
	)
	phase_plan = [phase for phase in phase_sequence if _reconstruct_templates_phase_enabled(inputs, phase)]
	if not phase_plan:
		return collect_templates_result_from_outputs(inputs)

	def _templates_phase_resource_class(phase_name: str) -> str | None:
		def _resource_class(value: Any) -> str | None:
			return getattr(value, "resource_class", None)

		phase = _normalize_reconstruct_templates_phase_name(phase_name)
		if phase == "resolve_sources":
			return _resource_class(inputs.resolve_sources_phase)
		if phase == "analyzers":
			return _resource_class(inputs.phases.analyzers)
		if phase == "extract_template_segments":
			return _resource_class(inputs.phases.per_unit_processing.extract_template_segments)
		if phase == "build_templates":
			return _resource_class(inputs.phases.build_templates)
		if phase == "compute_template_similarity":
			return _resource_class(inputs.phases.compute_template_similarity)
		if phase == "plot_templates":
			return _resource_class(inputs.phases.plot_templates)
		if phase == "report_templates":
			return _resource_class(inputs.phases.report_templates)
		if phase == "reports":
			return _resource_class(inputs.phases.reports)
		if phase == "per_unit_processing":
			return _resource_class(inputs.phases.per_unit_processing)
		return None

	def _descriptor_for_phase(phase_name: str) -> PhaseDescriptor:
		def _run_phase(phase_name: str = phase_name):
			return _reconstruct_templates_phase_runner(phase_name)(inputs)

		return PhaseDescriptor(
			name=str(phase_name),
			runner=_run_phase,
			resource_class=_templates_phase_resource_class(phase_name),
		)

	run_phase_chain(
		phases=[_descriptor_for_phase(phase) for phase in phase_plan],
		logger=LOGGER,
		target_label=str(inputs.stream_id),
		resource_key_context=inputs,
	)
	return collect_templates_result_from_outputs(inputs)


def _normalize_reconstruct_templates_phase_name(raw: Any) -> str:
	token = str(raw or "").strip().replace("-", "_").replace(" ", "_")
	aliases = {
		"resolve": "resolve_sources",
		"analyzer": "analyzers",
		"extract": "extract_template_segments",
		"build": "build_templates",
		"similarity": "compute_template_similarity",
		"compute_similarity": "compute_template_similarity",
		"plot": "plot_templates",
		"plots": "plot_templates",
		"template_report": "report_templates",
		"per_unit_processing.extract_template_segments": "extract_template_segments",
		"per_unit_processing.build_templates": "build_templates",
		"per_unit_processing.plots": "plot_templates",
	}
	return aliases.get(token, token)


def _reconstruct_templates_phase_enabled(inputs: TemplatesInputs, phase_name: str) -> bool:
	phase = _normalize_reconstruct_templates_phase_name(phase_name)
	phases = inputs.phases
	if phase == "resolve_sources":
		return bool(inputs.resolve_sources_phase.enabled)
	if phase == "analyzers":
		return bool(phases.analyzers.enabled)
	if phase == "extract_template_segments":
		return bool(phases.per_unit_processing.extract_template_segments.enabled)
	if phase == "build_templates":
		return bool(phases.build_templates.enabled)
	if phase == "compute_template_similarity":
		return bool(phases.compute_template_similarity.enabled)
	if phase == "plot_templates":
		return bool(phases.plot_templates.enabled)
	if phase == "report_templates":
		return bool(phases.report_templates.enabled)
	if phase == "reports":
		return bool(phases.reports.enabled)
	if phase == "per_unit_processing":
		return bool(phases.per_unit_processing.enabled)
	return False


def _reconstruct_templates_phase_runner(phase_name: str) -> Callable[[TemplatesInputs], Any]:
	phase = _normalize_reconstruct_templates_phase_name(phase_name)
	if phase == "resolve_sources":
		return run_reconstruct_templates_resolve_sources_phase
	if phase == "analyzers":
		return run_reconstruct_templates_analyzers_phase
	if phase == "extract_template_segments":
		return run_reconstruct_templates_extract_template_segments_phase
	if phase == "build_templates":
		return run_reconstruct_templates_build_templates_phase
	if phase == "compute_template_similarity":
		return run_reconstruct_templates_compute_template_similarity_phase
	if phase == "plot_templates":
		return run_reconstruct_templates_plot_templates_phase
	if phase == "report_templates":
		return run_reconstruct_templates_report_templates_phase
	if phase == "reports":
		return run_reconstruct_templates_reports_phase
	if phase == "per_unit_processing":
		return run_reconstruct_templates_per_unit_processing_phase
	raise ValueError(f"Unknown templates phase: {phase_name!r}")


def _path_is_within_preserved_set(path: Path, preserve_paths: list[Path]) -> bool:
	resolved = path.resolve()
	for preserve_path in preserve_paths:
		preserve_resolved = preserve_path.resolve()
		if resolved == preserve_resolved:
			return True
		if preserve_resolved in resolved.parents:
			return True
		if resolved in preserve_resolved.parents:
			return True
	return False


def _clear_directory_contents_preserving(*, root_dir: Path, preserve_paths: list[Path]) -> None:
	if not root_dir.exists():
		return
	preserved_existing = [path for path in preserve_paths if path.exists()]
	if not preserved_existing:
		shutil.rmtree(root_dir)
		return
	for child in list(root_dir.iterdir()):
		if _path_is_within_preserved_set(child, preserved_existing):
			if child.is_dir():
				_clear_directory_contents_preserving(root_dir=child, preserve_paths=preserved_existing)
			continue
		if child.is_dir():
			shutil.rmtree(child)
		else:
			child.unlink(missing_ok=True)


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
	merged_units_dir = next(
		(path for path in _materialized_merged_root_candidates(templates_out_dir) if path.exists()),
		templates_out_dir / MATERIALIZED_TEMPLATES_CACHE_RELPATH / "merged",
	)
	full_channels_templates_dir = next(
		(path for path in _materialized_full_root_candidates(templates_out_dir) if path.exists()),
		templates_out_dir / MATERIALIZED_TEMPLATES_CACHE_RELPATH / "full",
	)

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


def _apply_unit_label_filter(inputs: TemplatesInputs, unit_ids: list[Any], well_out_dir: Path, *, context: str) -> list[Any]:
	allowed_labels = tuple(str(label).strip().lower() for label in inputs.unit_label_filter_labels if str(label).strip())
	if not allowed_labels:
		return list(unit_ids)
	labels_by_unit = load_unit_labels_from_spikesorting(well_out_dir)
	if not labels_by_unit:
		if bool(inputs.unit_label_filter_required):
			raise RuntimeError(
				"Templates unit label filter is enabled, but no Bombcell/Kilosort unit labels were found under "
				f"{well_out_dir}."
			)
		LOGGER.warning(
			"Templates %s: unit label filter skipped because no labels were found under %s",
			context,
			well_out_dir,
		)
		return list(unit_ids)
	filtered = filter_unit_ids_by_labels(unit_ids, labels_by_unit, allowed_labels)
	LOGGER.info(
		"Templates %s: unit label filter allowed=%s kept=%d/%d counts=%s",
		context,
		list(allowed_labels),
		len(filtered),
		len(unit_ids),
		count_labels(labels_by_unit),
	)
	return filtered


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


def _cleanup_unit_output_artifacts(
	*,
	templates_out_dir: Path,
	unit_ids: list[Any],
	per_unit_outputs: Any,
	output_keys: tuple[str, ...],
) -> None:
	for unit_id in unit_ids:
		paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=per_unit_outputs,
		)
		for output_key in output_keys:
			artifact_path = paths.get(output_key)
			if artifact_path is None:
				continue
			try:
				if artifact_path.exists():
					artifact_path.unlink()
			except Exception:
				LOGGER.debug(
					"Templates cleanup skipped artifact removal: unit_id=%s key=%s path=%s",
					unit_id,
					output_key,
					artifact_path,
					exc_info=True,
				)


def _collect_existing_unit_output_paths(
	*,
	templates_out_dir: Path,
	unit_id: Any,
	per_unit_outputs: Any,
	output_keys: tuple[str, ...],
) -> dict[str, str]:
	paths = resolve_unit_output_paths(
		templates_out_dir=templates_out_dir,
		unit_id=unit_id,
		per_unit_outputs=per_unit_outputs,
	)
	existing_outputs: dict[str, str] = {}
	for output_key in output_keys:
		artifact_path = paths.get(output_key)
		if artifact_path is not None and artifact_path.exists():
			existing_outputs[str(output_key)] = str(artifact_path)
	return existing_outputs


def _persist_unit_summary_output_paths(
	*,
	templates_out_dir: Path,
	unit_id: Any,
	per_unit_outputs: Any,
	output_paths: dict[str, str],
) -> None:
	if not output_paths:
		return
	paths = resolve_unit_output_paths(
		templates_out_dir=templates_out_dir,
		unit_id=unit_id,
		per_unit_outputs=per_unit_outputs,
	)
	unit_summary_json = paths["unit_summary_json"]
	payload: dict[str, Any] = {}
	if unit_summary_json.exists():
		try:
			existing = read_json(unit_summary_json)
			if isinstance(existing, dict):
				payload = dict(existing)
		except Exception:
			payload = {}
	existing_outputs = payload.get("outputs", {})
	if not isinstance(existing_outputs, dict):
		existing_outputs = {}
	serialized_outputs = {
		str(key): str(value)
		for key, value in dict(existing_outputs).items()
		if value is not None
	}
	changed = not unit_summary_json.exists()
	for output_key, output_path in output_paths.items():
		if serialized_outputs.get(str(output_key)) != str(output_path):
			serialized_outputs[str(output_key)] = str(output_path)
			changed = True
	payload.setdefault("unit_id", unit_id)
	payload.setdefault("status", "ok")
	payload.setdefault("error", None)
	payload["outputs"] = serialized_outputs
	if changed:
		write_json(unit_summary_json, payload)


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


def _resolve_alternate_well_out_dirs(*, inputs: TemplatesInputs, primary_well_out_dir: Path) -> list[Path]:
	roots_to_probe: list[Path] = []
	if inputs.final_output_root is not None:
		roots_to_probe.append(Path(inputs.final_output_root).expanduser().resolve())
	for root in list(inputs.artifact_lookup_roots or ()):  # model-level alternates from data config
		try:
			roots_to_probe.append(Path(root).expanduser().resolve())
		except Exception:
			continue

	resolved_primary = primary_well_out_dir.resolve()
	seen: set[Path] = {resolved_primary}
	alternate_well_out_dirs: list[Path] = []
	for candidate_root in roots_to_probe:
		try:
			candidate_well_out_dir = compute_mea_analysis_output_dir(
				output_root=candidate_root,
				data_file=inputs.h5_path,
				well=inputs.stream_id,
			)
			resolved_candidate = candidate_well_out_dir.resolve()
		except Exception:
			continue
		if resolved_candidate in seen:
			continue
		seen.add(resolved_candidate)
		alternate_well_out_dirs.append(candidate_well_out_dir)
	return alternate_well_out_dirs


def _dedupe_string_tokens(tokens: list[str]) -> list[str]:
	seen: set[str] = set()
	ordered: list[str] = []
	for raw in list(tokens):
		token = str(raw or "").strip()
		if not token or token in seen:
			continue
		seen.add(token)
		ordered.append(token)
	return ordered


def _summarize_source_candidates(
	*,
	name: str,
	tokens: list[str],
	candidate_paths: list[Path],
	check_path_exists: bool,
	max_candidates_per_source: int,
) -> dict[str, Any]:
	first_existing: str | None = None
	for path in candidate_paths:
		if path.exists():
			first_existing = str(path)
			break

	limit = int(max(1, max_candidates_per_source))
	entries: list[dict[str, Any]] = []
	for candidate in list(candidate_paths)[:limit]:
		row: dict[str, Any] = {"path": str(candidate)}
		if check_path_exists:
			row["exists"] = bool(candidate.exists())
		entries.append(row)

	truncated_count = int(max(0, len(candidate_paths) - len(entries)))
	return {
		"name": name,
		"tokens": list(tokens),
		"candidate_count": int(len(candidate_paths)),
		"first_existing": first_existing,
		"candidates": entries,
		"candidates_truncated": truncated_count,
	}


def run_reconstruct_templates_resolve_sources_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	phase_cfg = inputs.resolve_sources_phase
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	alternate_well_out_dirs = (
		_resolve_alternate_well_out_dirs(inputs=inputs, primary_well_out_dir=well_out_dir)
		if bool(phase_cfg.include_alternate_well_dirs)
		else []
	)
	well_dirs = [well_out_dir, *list(alternate_well_out_dirs)]

	concat_analyzer_tokens = _dedupe_string_tokens(
		[
			str(inputs.concat_analyzer_relpath or ""),
			"/spikesort_outputs/analyzer_output",
			"/stg2_spikesorting_outputs/analyzer_output",
			"spikesort_outputs/analyzer_output",
			"stg2_spikesorting_outputs/analyzer_output",
		]
	)
	concat_sorting_tokens = _dedupe_string_tokens(
		[
			str(inputs.concat_sorting_relpath or ""),
			"/spikesort_outputs/sorter_output",
			"/stg2_spikesorting_outputs/sorter_output",
			"spikesort_outputs/sorter_output",
			"stg2_spikesorting_outputs/sorter_output",
		]
	)
	preprocessed_concat_tokens = _dedupe_string_tokens(
		[
			str(inputs.preprocessed_concat_reldir or ""),
			"/preprocess_outputs/preprocessed_recording",
			"preprocess_outputs/preprocessed_recording",
		]
	)
	preprocessed_segments_tokens = _dedupe_string_tokens(
		[
			str(inputs.preprocessed_segments_reldir or ""),
			str(inputs.preproc_seg_sources_reldir or ""),
			"/preprocess_outputs/per_segment_preprocessed",
			"/preprocess_outputs/per_segment_recordings",
			"preprocess_outputs/per_segment_preprocessed",
			"preprocess_outputs/per_segment_recordings",
		]
	)

	concat_analyzer_candidates = _resolve_well_relative_path_candidates(
		well_dirs=well_dirs,
		relpath_tokens=concat_analyzer_tokens,
	)
	concat_sorting_candidates = _resolve_well_relative_path_candidates(
		well_dirs=well_dirs,
		relpath_tokens=concat_sorting_tokens,
	)
	preprocessed_concat_candidates = _resolve_well_relative_path_candidates(
		well_dirs=well_dirs,
		relpath_tokens=preprocessed_concat_tokens,
	)
	preprocessed_segments_candidates = _resolve_well_relative_path_candidates(
		well_dirs=well_dirs,
		relpath_tokens=preprocessed_segments_tokens,
	)

	source_summaries = {
		"concat_analyzer": _summarize_source_candidates(
			name="concat_analyzer",
			tokens=concat_analyzer_tokens,
			candidate_paths=concat_analyzer_candidates,
			check_path_exists=bool(phase_cfg.check_path_exists),
			max_candidates_per_source=int(phase_cfg.max_candidates_per_source),
		),
		"concat_sorting": _summarize_source_candidates(
			name="concat_sorting",
			tokens=concat_sorting_tokens,
			candidate_paths=concat_sorting_candidates,
			check_path_exists=bool(phase_cfg.check_path_exists),
			max_candidates_per_source=int(phase_cfg.max_candidates_per_source),
		),
		"preprocessed_concat": _summarize_source_candidates(
			name="preprocessed_concat",
			tokens=preprocessed_concat_tokens,
			candidate_paths=preprocessed_concat_candidates,
			check_path_exists=bool(phase_cfg.check_path_exists),
			max_candidates_per_source=int(phase_cfg.max_candidates_per_source),
		),
		"preprocessed_segments": _summarize_source_candidates(
			name="preprocessed_segments",
			tokens=preprocessed_segments_tokens,
			candidate_paths=preprocessed_segments_candidates,
			check_path_exists=bool(phase_cfg.check_path_exists),
			max_candidates_per_source=int(phase_cfg.max_candidates_per_source),
		),
	}

	unit_label_probe: dict[str, Any] = {
		"allowed_labels": list(inputs.unit_label_filter_labels),
		"required": bool(inputs.unit_label_filter_required),
		"probe_attempted": False,
		"available": None,
		"counts_by_label": {},
	}
	if bool(phase_cfg.probe_unit_labels) and inputs.unit_label_filter_labels:
		unit_label_probe["probe_attempted"] = True
		labels_by_unit = load_unit_labels_from_spikesorting(well_out_dir)
		if labels_by_unit is not None:
			unit_label_probe["available"] = True
			unit_label_probe["counts_by_label"] = count_labels(labels_by_unit)
		else:
			unit_label_probe["available"] = False

	summary: dict[str, Any] = {
		"phase": "resolve_sources",
		"stream_id": str(inputs.stream_id),
		"h5_path": str(inputs.h5_path),
		"well_out_dir": str(well_out_dir),
		"alternate_well_out_dirs": [str(path) for path in alternate_well_out_dirs],
		"run_intent": {
			"force_restart": bool(inputs.force_restart),
			"force_replot": bool(inputs.force_replot),
			"force_replot_per_unit": bool(inputs.force_replot_per_unit),
			"force_rereport": bool(inputs.force_rereport),
		},
		"applied_debug_limits": _templates_applied_debug_limits(inputs),
		"unit_scope": {
			"unit_ids": (None if inputs.unit_ids is None else list(inputs.unit_ids)),
			"unit_limit": inputs.unit_limit,
			"unit_label_filter": unit_label_probe,
		},
		"source_requirements": {
			"include_concat": bool(inputs.include_concat),
			"include_segments": bool(inputs.include_segments),
			"require_concat_analyzer": bool(inputs.require_concat_analyzer),
			"require_segment_analyzers": bool(inputs.require_segment_analyzers),
		},
		"sources": source_summaries,
	}

	if bool(phase_cfg.fail_if_required_sources_missing):
		missing_required: list[str] = []
		if bool(inputs.include_concat) and bool(inputs.require_concat_analyzer):
			if source_summaries["concat_analyzer"].get("first_existing", None) is None:
				missing_required.append("concat_analyzer")
		if bool(inputs.include_segments) and bool(inputs.require_segment_analyzers):
			if source_summaries["preprocessed_segments"].get("first_existing", None) is None:
				missing_required.append("preprocessed_segments")
		if missing_required:
			raise RuntimeError(
				"resolve_sources required inputs missing: "
				+ ", ".join(missing_required)
			)

	if bool(phase_cfg.enabled):
		if bool(phase_cfg.show_header):
			header_title = f"templates.resolve_sources [{inputs.stream_id}]"
			header_line = "=" * max(24, len(header_title))
			LOGGER.info(header_line)
			LOGGER.info(header_title)
			LOGGER.info(header_line)
		LOGGER.info(
			"resolve_sources: stream=%s run_intent=%s",
			str(inputs.stream_id),
			summary["run_intent"],
		)
		LOGGER.info(
			"resolve_sources: well_out_dir=%s alternate_well_out_dirs=%s",
			str(well_out_dir),
			["%s" % path for path in alternate_well_out_dirs],
		)
		for source_name, payload in source_summaries.items():
			LOGGER.info(
				"resolve_sources: %s first_existing=%s candidates=%d",
				source_name,
				payload.get("first_existing", None),
				int(payload.get("candidate_count", 0)),
			)
			if bool(phase_cfg.log_candidates):
				for row in list(payload.get("candidates", [])):
					LOGGER.info(
						"resolve_sources: %s candidate path=%s exists=%s",
						source_name,
						row.get("path", None),
						row.get("exists", None),
					)
		LOGGER.info("resolve_sources: unit_scope=%s", summary["unit_scope"])

	if bool(phase_cfg.write_json):
		templates_out_dir = well_out_dir / str(inputs.output_rel_root)
		json_path = templates_out_dir / str(phase_cfg.json_relpath)
		json_path.parent.mkdir(parents=True, exist_ok=True)
		write_json(json_path, summary)
		summary["summary_json"] = str(json_path)

	return summary


def _resolve_templates_phase_environment(
	inputs: TemplatesInputs,
) -> tuple[Path, list[Path], Path, Path | None]:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	alternate_well_out_dirs = _resolve_alternate_well_out_dirs(
		inputs=inputs,
		primary_well_out_dir=well_out_dir,
	)
	templates_out_dir = well_out_dir / str(inputs.output_rel_root)
	templates_out_dir.mkdir(parents=True, exist_ok=True)
	analyzer_cache_dir = _resolve_templates_analyzer_cache_dir(inputs=inputs, well_out_dir=well_out_dir)
	return well_out_dir, alternate_well_out_dirs, templates_out_dir, analyzer_cache_dir


def _resolve_templates_analyzer_cache_dir(*, inputs: TemplatesInputs, well_out_dir: Path) -> Path | None:
	if not bool(inputs.analyzer_cache.enabled):
		return None
	cache_rel = Path(str(inputs.analyzer_cache.relpath or "analyzers")).expanduser()
	if cache_rel.is_absolute():
		cache_rel = Path(str(cache_rel).lstrip("/"))
	return (well_out_dir / str(inputs.output_rel_root) / cache_rel)


def _format_templates_log_value(value: Any) -> str:
	if isinstance(value, float):
		try:
			if np.isfinite(value):
				return f"{float(value):.4g}"
		except Exception:
			pass
	return str(value)


def _format_templates_log_fields(fields: dict[str, Any]) -> str:
	parts: list[str] = []
	for key, value in fields.items():
		if value is None:
			continue
		parts.append(f"{key}={_format_templates_log_value(value)}")
	return " ".join(parts)


def _templates_analyzer_policy_log_fields(policy: Any) -> dict[str, Any]:
	percentage = None
	if getattr(policy, "random_spikes_percentage", None) is not None:
		try:
			percentage = f"{float(policy.random_spikes_percentage) * 100.0:.1f}%"
		except Exception:
			percentage = policy.random_spikes_percentage
	return {
		"ms_before": getattr(policy, "ms_before", None),
		"ms_after": getattr(policy, "ms_after", None),
		"dtype": getattr(policy, "dtype", None),
		"max_spikes_per_unit": getattr(policy, "max_spikes_per_unit", None),
		"min_spikes_per_unit": getattr(policy, "min_spikes_per_unit", None),
		"random_spikes_method": getattr(policy, "random_spikes_method", None),
		"random_spikes_percentage": percentage,
		"random_seed": getattr(policy, "random_seed", None),
		"log_before_after_spike_counts": bool(getattr(policy, "log_before_after_spike_counts", False)),
		"margin_size": getattr(policy, "margin_size", None),
		"compute_sparsity": bool(getattr(policy, "compute_sparsity", True)),
		"sparsity_mode": getattr(policy, "sparsity_mode", None),
		"sparsity_method": getattr(policy, "sparsity_method", None),
		"sparsity_radius_um": getattr(policy, "sparsity_radius_um", None),
		"sparsity_num_channels": getattr(policy, "sparsity_num_channels", None),
		"sparsity_threshold": getattr(policy, "sparsity_threshold", None),
		"sparsity_peak_sign": getattr(policy, "sparsity_peak_sign", None),
		"sparsity_num_spikes_for_sparsity": getattr(policy, "sparsity_num_spikes_for_sparsity", None),
		"n_jobs": getattr(policy, "n_jobs", None),
		"chunk_duration": getattr(policy, "chunk_duration", None),
	}


def _templates_runtime_n_jobs(inputs: TemplatesInputs) -> int:
	return max(1, int(getattr(inputs, "n_jobs", 1) or 1))


def _resolve_analyzer_policy_runtime_n_jobs(inputs: TemplatesInputs, policy: Any) -> Any:
	if policy is None:
		return policy
	if getattr(policy, "n_jobs", None) is not None:
		return policy
	try:
		return replace(policy, n_jobs=_templates_runtime_n_jobs(inputs))
	except Exception:
		return policy


def _templates_analyzer_policy_for_source(inputs: TemplatesInputs, source_name: str) -> Any:
	if str(source_name) == "concat":
		policy = inputs.phases.analyzers.concat.policy
	else:
		policy = inputs.phases.analyzers.segments.policy
	return _resolve_analyzer_policy_runtime_n_jobs(inputs, policy)


def _iter_templates_phase_analyzers(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	alternate_well_out_dirs: list[Path],
	analyzer_cache_dir: Path | None,
	source_scope: str | None = None,
	requested_source_names: list[str] | tuple[str, ...] | set[str] | None = None,
	load_stats: dict[str, Any] | None = None,
) -> Any:
	include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
	include_segments = bool(inputs.include_segments) and bool(inputs.phases.analyzers.segments.enabled)
	if source_scope == "concat":
		include_segments = False
	elif source_scope == "segments":
		include_concat = False
	return iter_spikeinterface_analyzers(
		well_out_dir=well_out_dir,
		concat_analyzer_relpath=(inputs.phases.analyzers.concat.analyzer_relpath or inputs.concat_analyzer_relpath),
		concat_sorting_relpath=(inputs.phases.analyzers.concat.sorting_relpath or inputs.concat_sorting_relpath),
		preprocessed_concat_reldir=(
			inputs.phases.analyzers.concat.preprocessed_recording_reldir or inputs.preprocessed_concat_reldir
		),
		preprocessed_segments_reldir=(
			inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preprocessed_segments_reldir
		),
		preproc_seg_sources_reldir=(
			inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preproc_seg_sources_reldir
		),
		analyzer_cache_dir=analyzer_cache_dir,
		analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
		analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
		alternate_well_out_dirs=alternate_well_out_dirs,
		stream_id=str(inputs.stream_id),
		include_concat=include_concat,
		include_segments=include_segments,
		require_concat=(bool(inputs.require_concat_analyzer) and include_concat),
		require_segments=(bool(inputs.require_segment_analyzers) and include_segments),
		waveform_ms_before=inputs.waveform_extraction.ms_before,
		waveform_ms_after=inputs.waveform_extraction.ms_after,
		waveform_max_spikes_per_unit=inputs.waveform_extraction.max_spikes_per_unit,
		concat_policy=_resolve_analyzer_policy_runtime_n_jobs(inputs, inputs.phases.analyzers.concat.policy),
		segments_policy=_resolve_analyzer_policy_runtime_n_jobs(inputs, inputs.phases.analyzers.segments.policy),
		concat_use_existing_analyzer=bool(inputs.phases.analyzers.concat.use_existing_analyzer),
		concat_build_if_missing=bool(inputs.phases.analyzers.concat.build_if_missing),
		segments_use_existing_analyzer=bool(inputs.phases.analyzers.segments.use_existing_analyzer),
		segments_build_if_missing=bool(inputs.phases.analyzers.segments.build_if_missing),
		requested_source_names=requested_source_names,
		limit_segments=inputs.limit_segments,
		load_stats=load_stats,
	)


def _load_templates_phase_analyzers(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	alternate_well_out_dirs: list[Path],
	analyzer_cache_dir: Path | None,
	source_scope: str | None = None,
	requested_source_names: list[str] | tuple[str, ...] | set[str] | None = None,
	return_stats: bool = False,
) -> Any:
	include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
	include_segments = bool(inputs.include_segments) and bool(inputs.phases.analyzers.segments.enabled)
	if source_scope == "concat":
		include_segments = False
	elif source_scope == "segments":
		include_concat = False
	return load_spikeinterface_analyzers(
		well_out_dir=well_out_dir,
		concat_analyzer_relpath=(inputs.phases.analyzers.concat.analyzer_relpath or inputs.concat_analyzer_relpath),
		concat_sorting_relpath=(inputs.phases.analyzers.concat.sorting_relpath or inputs.concat_sorting_relpath),
		preprocessed_concat_reldir=(
			inputs.phases.analyzers.concat.preprocessed_recording_reldir or inputs.preprocessed_concat_reldir
		),
		preprocessed_segments_reldir=(
			inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preprocessed_segments_reldir
		),
		preproc_seg_sources_reldir=(
			inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preproc_seg_sources_reldir
		),
		analyzer_cache_dir=analyzer_cache_dir,
		analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
		analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
		alternate_well_out_dirs=alternate_well_out_dirs,
		stream_id=str(inputs.stream_id),
		include_concat=include_concat,
		include_segments=include_segments,
		require_concat=(bool(inputs.require_concat_analyzer) and include_concat),
		require_segments=(bool(inputs.require_segment_analyzers) and include_segments),
		waveform_ms_before=inputs.waveform_extraction.ms_before,
		waveform_ms_after=inputs.waveform_extraction.ms_after,
		waveform_max_spikes_per_unit=inputs.waveform_extraction.max_spikes_per_unit,
		concat_policy=_resolve_analyzer_policy_runtime_n_jobs(inputs, inputs.phases.analyzers.concat.policy),
		segments_policy=_resolve_analyzer_policy_runtime_n_jobs(inputs, inputs.phases.analyzers.segments.policy),
		concat_use_existing_analyzer=bool(inputs.phases.analyzers.concat.use_existing_analyzer),
		concat_build_if_missing=bool(inputs.phases.analyzers.concat.build_if_missing),
		segments_use_existing_analyzer=bool(inputs.phases.analyzers.segments.use_existing_analyzer),
		segments_build_if_missing=bool(inputs.phases.analyzers.segments.build_if_missing),
		requested_source_names=requested_source_names,
		limit_segments=inputs.limit_segments,
		return_stats=return_stats,
	)


def _discover_templates_cached_build_sources(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	alternate_well_out_dirs: list[Path],
) -> tuple[Path, Path | None, list[str]]:
	include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
	include_segments = bool(inputs.include_segments) and bool(inputs.phases.analyzers.segments.enabled)
	primary_cache_dir = _resolve_templates_analyzer_cache_dir(inputs=inputs, well_out_dir=well_out_dir)
	candidate_well_out_dirs = [well_out_dir, *list(alternate_well_out_dirs)]
	for candidate_well_out_dir in candidate_well_out_dirs:
		candidate_cache_dir = _resolve_templates_analyzer_cache_dir(inputs=inputs, well_out_dir=candidate_well_out_dir)
		source_names = discover_cached_spikeinterface_analyzer_source_names(
			analyzer_cache_dir=candidate_cache_dir,
			analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
			analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
			include_concat=include_concat,
			include_segments=include_segments,
			limit_segments=inputs.limit_segments,
		)
		if source_names:
			if candidate_well_out_dir != well_out_dir:
				LOGGER.info(
					"templates.build_templates using fallback analyzer cache for cached build bootstrap: primary=%s fallback=%s source_count=%d",
					str(well_out_dir),
					str(candidate_well_out_dir),
					int(len(source_names)),
				)
			return candidate_well_out_dir, candidate_cache_dir, source_names
	return well_out_dir, primary_cache_dir, []


def _build_templates_phase_from_cached_analyzers(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	alternate_well_out_dirs: list[Path],
	templates_out_dir: Path,
	payload_root: Path,
) -> dict[str, Any]:
	analyzer_well_out_dir, analyzer_cache_dir, source_names = _discover_templates_cached_build_sources(
		inputs=inputs,
		well_out_dir=well_out_dir,
		alternate_well_out_dirs=alternate_well_out_dirs,
	)
	if analyzer_cache_dir is None or not source_names:
		raise FileNotFoundError(
			"No cached templates analyzers found for build_templates. "
			f"checked analyzer_cache_dir={analyzer_cache_dir}; run templates.analyzers before templates.build_templates."
		)

	LOGGER.info(
		"templates.build_templates loading cached analyzers for build bootstrap: analyzer_well_out_dir=%s analyzer_cache_dir=%s source_count=%d",
		str(analyzer_well_out_dir),
		str(analyzer_cache_dir),
		int(len(source_names)),
	)

	output_rel_root = str(inputs.phases.per_unit_processing.extract_template_segments.output_rel_root)
	streamed_sources_summary: dict[str, Any] = {}
	unit_ids: list[Any] | None = (None if inputs.unit_ids is None else list(inputs.unit_ids))
	if unit_ids is not None and inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]

	for requested_source_name in source_names:
		analyzers = load_cached_spikeinterface_analyzers(
			well_out_dir=analyzer_well_out_dir,
			preprocessed_concat_reldir=(
				inputs.phases.analyzers.concat.preprocessed_recording_reldir or inputs.preprocessed_concat_reldir
			),
			preprocessed_segments_reldir=(
				inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preprocessed_segments_reldir
			),
			preproc_seg_sources_reldir=(
				inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preproc_seg_sources_reldir
			),
			analyzer_cache_dir=analyzer_cache_dir,
			analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
			analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
			include_concat=bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled),
			include_segments=bool(inputs.include_segments) and bool(inputs.phases.analyzers.segments.enabled),
			requested_source_names=[str(requested_source_name)],
			limit_segments=inputs.limit_segments,
		)
		source_match = next(
			((name, analyzer) for name, analyzer in analyzers if str(name) == str(requested_source_name)),
			None,
		)
		if source_match is None:
			raise FileNotFoundError(
				f"Failed loading requested templates analyzer source {requested_source_name!r} under {analyzer_well_out_dir}"
			)

		source_name, analyzer = source_match
		if unit_ids is None:
			unit_ids = _collect_templates_phase_unit_ids(
				inputs=inputs,
				analyzers=[source_match],
				well_out_dir=well_out_dir,
			)
		materialized_units: list[Any] = []
		for unit_id in list(unit_ids or []):
			payload = build_unit_source_payload(
				analyzer=analyzer,
				unit_id=unit_id,
				include_overlay_waveforms=False,
				allow_prepare=False,
			)
			if payload is None:
				continue
			# Spool per-unit per-source payload to disk so the build phase can
			# stream from disk one unit at a time instead of holding every
			# (unit, source) tensor coresident in memory.
			write_materialized_source_payload(
				templates_out_dir=templates_out_dir,
				output_rel_root=output_rel_root,
				source_name=str(source_name),
				unit_id=unit_id,
				template_c_by_t=payload[0],
				locations_xy=payload[1],
				electrode_ids=payload[2],
				channel_ids=payload[3],
				waveform_count=payload[4],
				sampling_rate_hz=payload[5],
				overlay_waveforms=None,
				top_electrode_id=None,
				total_waveforms_at_channel=None,
			)
			materialized_units.append(unit_id)
			del payload
		streamed_sources_summary[str(source_name)] = {
			"units_materialized": [unit for unit in materialized_units],
			"unit_count": int(len(materialized_units)),
		}
		LOGGER.info(
			"templates.build_templates materialized cached analyzer payloads: source=%s unit_count=%d",
			str(source_name),
			int(len(materialized_units)),
		)
		del analyzer
		del analyzers
		gc.collect()

	if unit_ids is None:
		unit_ids = []

	def _payload_loader(unit_id: Any) -> list[tuple[str, tuple[Any, ...]]]:
		loaded: list[tuple[str, tuple[Any, ...]]] = []
		for source_name in source_names:
			payload = load_materialized_source_payload(
				source_payload_unit_dir=resolve_materialized_source_payload_unit_dir(
					templates_out_dir=templates_out_dir,
					output_rel_root=output_rel_root,
					source_name=str(source_name),
					unit_id=unit_id,
				),
			)
			if payload is None:
				continue
			loaded.append((str(source_name), payload))
		return loaded

	summary = build_templates_phase_from_unit_payloads(
		inputs=inputs,
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
		unit_ids=list(unit_ids),
		source_names=[str(name) for name in source_names],
		payload_root=payload_root,
		payload_materialization_mode="analyzer_cache",
		payload_loader=_payload_loader,
	)
	summary["source_payload_well_out_dir"] = str(analyzer_well_out_dir)
	summary["analyzer_cache_dir"] = str(analyzer_cache_dir)
	summary["source_payload_sources"] = streamed_sources_summary
	return summary


def _collect_templates_phase_unit_ids(
	*,
	inputs: TemplatesInputs,
	analyzers: list[tuple[str, Any]],
	well_out_dir: Path,
) -> list[Any]:
	if inputs.unit_ids is not None:
		unit_ids = list(inputs.unit_ids)
	elif analyzers:
		unit_ids = list(getattr(analyzers[0][1].sorting, "unit_ids", []))
	else:
		unit_ids = []
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	return _apply_unit_label_filter(inputs, unit_ids, well_out_dir, context="collect_unit_ids")


def _report_scope_config(reports: Any, scope: str | None) -> Any:
	if scope is None:
		return reports
	disabled_locations = replace(reports.locations, write_json=False, write_png=False, write_svg=False)
	disabled_wf_grid = replace(reports.wf_overlay_grid, write_pdf=False, write_png=False, write_svg=False)
	disabled_footprint_grids = replace(
		reports.footprint_grids,
		circles_map_grid=replace(reports.footprint_grids.circles_map_grid, write_pdf=False, write_png=False, write_svg=False),
		amplitude_map_grid=replace(reports.footprint_grids.amplitude_map_grid, write_pdf=False, write_png=False, write_svg=False),
		latency_map_grid=replace(reports.footprint_grids.latency_map_grid, write_pdf=False, write_png=False, write_svg=False),
	)
	disabled_multi_source = replace(reports.plot_multi_source_pdf, enabled=False)
	if scope == "locations":
		return replace(
			reports,
			locations=reports.locations,
			wf_overlay_grid=disabled_wf_grid,
			footprint_grids=disabled_footprint_grids,
			plot_multi_source_pdf=disabled_multi_source,
		)
	if scope == "wf_overlay_grid":
		return replace(
			reports,
			locations=disabled_locations,
			wf_overlay_grid=reports.wf_overlay_grid,
			footprint_grids=disabled_footprint_grids,
			plot_multi_source_pdf=disabled_multi_source,
		)
	if scope == "footprint_grids":
		return replace(
			reports,
			locations=disabled_locations,
			wf_overlay_grid=disabled_wf_grid,
			footprint_grids=reports.footprint_grids,
			plot_multi_source_pdf=disabled_multi_source,
		)
	if scope == "multi_source_pdf":
		return replace(
			reports,
			locations=disabled_locations,
			wf_overlay_grid=disabled_wf_grid,
			footprint_grids=disabled_footprint_grids,
			plot_multi_source_pdf=reports.plot_multi_source_pdf,
		)
	if scope == "none":
		return replace(
			reports,
			locations=disabled_locations,
			wf_overlay_grid=disabled_wf_grid,
			footprint_grids=disabled_footprint_grids,
			plot_multi_source_pdf=disabled_multi_source,
		)
	return reports


def _disable_reports_config(reports: Any) -> Any:
	return _report_scope_config(reports, "none")


def run_reconstruct_templates_analyzers_phase(inputs: TemplatesInputs, *, source_scope: str | None = None) -> dict[str, Any]:
	phase_started = perf_counter()
	well_out_dir, alternate_well_out_dirs, templates_out_dir, analyzer_cache_dir = _resolve_templates_phase_environment(inputs)
	include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
	include_segments = bool(inputs.include_segments) and bool(inputs.phases.analyzers.segments.enabled)
	require_concat = bool(inputs.require_concat_analyzer) and include_concat
	require_segments = bool(inputs.require_segment_analyzers) and include_segments
	if source_scope == "concat":
		include_segments = False
		require_segments = False
	elif source_scope == "segments":
		include_concat = False
		require_concat = False
	LOGGER.info(
		"templates.analyzers start: source_scope=%s well_out_dir=%s templates_out_dir=%s analyzer_cache_dir=%s include_concat=%s include_segments=%s require_concat=%s require_segments=%s force_restart=%s",
		source_scope,
		str(well_out_dir),
		str(templates_out_dir),
		(None if analyzer_cache_dir is None else str(analyzer_cache_dir)),
		bool(include_concat),
		bool(include_segments),
		bool(require_concat),
		bool(require_segments),
		bool(inputs.force_restart),
	)
	LOGGER.info(
		"templates.analyzers concat settings: use_existing=%s build_if_missing=%s analyzer_relpath=%s sorting_relpath=%s preprocessed_recording_reldir=%s settings=%s",
		bool(inputs.phases.analyzers.concat.use_existing_analyzer),
		bool(inputs.phases.analyzers.concat.build_if_missing),
		(inputs.phases.analyzers.concat.analyzer_relpath or inputs.concat_analyzer_relpath),
		(inputs.phases.analyzers.concat.sorting_relpath or inputs.concat_sorting_relpath),
		(inputs.phases.analyzers.concat.preprocessed_recording_reldir or inputs.preprocessed_concat_reldir),
		_format_templates_log_fields(_templates_analyzer_policy_log_fields(_resolve_analyzer_policy_runtime_n_jobs(inputs, inputs.phases.analyzers.concat.policy))),
	)
	LOGGER.info(
		"templates.analyzers segments settings: use_existing=%s build_if_missing=%s preprocessed_sources_reldir=%s settings=%s",
		bool(inputs.phases.analyzers.segments.use_existing_analyzer),
		bool(inputs.phases.analyzers.segments.build_if_missing),
		(inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preprocessed_segments_reldir or inputs.preproc_seg_sources_reldir),
		_format_templates_log_fields(_templates_analyzer_policy_log_fields(_resolve_analyzer_policy_runtime_n_jobs(inputs, inputs.phases.analyzers.segments.policy))),
	)
	if bool(inputs.force_restart) and analyzer_cache_dir is not None and analyzer_cache_dir.exists():
		LOGGER.info("templates.analyzers clearing analyzer cache on force_restart: %s", str(analyzer_cache_dir))
		shutil.rmtree(analyzer_cache_dir)
	LOGGER.info("templates.analyzers streaming analyzer sources (one at a time)")
	load_stats: dict[str, Any] = {}
	sources_summary: dict[str, Any] = {}
	source_count = 0
	concat_count = 0
	segment_count = 0
	for source_name, analyzer in _iter_templates_phase_analyzers(
		inputs=inputs,
		well_out_dir=well_out_dir,
		alternate_well_out_dirs=alternate_well_out_dirs,
		analyzer_cache_dir=analyzer_cache_dir,
		source_scope=source_scope,
		load_stats=load_stats,
	):
		policy = _templates_analyzer_policy_for_source(inputs, source_name)
		num_channels: int | None = None
		get_num_channels = getattr(analyzer, "get_num_channels", None)
		if callable(get_num_channels):
			try:
				num_channels = int(get_num_channels())
			except Exception:
				num_channels = None
		elif hasattr(getattr(analyzer, "recording", None), "get_num_channels"):
			try:
				num_channels = int(analyzer.recording.get_num_channels())
			except Exception:
				num_channels = None
		sources_summary[str(source_name)] = {
			"has_sparsity": bool(getattr(analyzer, "sparsity", None) is not None),
			"num_channels": num_channels,
			"num_units": int(len(list(getattr(analyzer.sorting, "unit_ids", [])))) if hasattr(analyzer, "sorting") else None,
			"policy": {
				"sparsity_mode": str(policy.sparsity_mode),
				"compute_sparsity": bool(policy.compute_sparsity),
				"sparsity_method": str(policy.sparsity_method),
				"sparsity_radius_um": policy.sparsity_radius_um,
				"sparsity_num_channels": policy.sparsity_num_channels,
				"sparsity_threshold": policy.sparsity_threshold,
				"sparsity_peak_sign": str(policy.sparsity_peak_sign),
				"sparsity_num_spikes_for_sparsity": policy.sparsity_num_spikes_for_sparsity,
				"sparsity_by_property": policy.sparsity_by_property,
				"random_spikes_method": str(policy.random_spikes_method),
				"random_spikes_percentage": policy.random_spikes_percentage,
				"min_spikes_per_unit": policy.min_spikes_per_unit,
				"random_seed": policy.random_seed,
				"log_before_after_spike_counts": bool(policy.log_before_after_spike_counts),
				"margin_size": policy.margin_size,
				"ms_before": policy.ms_before,
				"ms_after": policy.ms_after,
				"dtype": policy.dtype,
				"max_spikes_per_unit": policy.max_spikes_per_unit,
				"n_jobs": policy.n_jobs,
				"chunk_duration": policy.chunk_duration,
			},
		}
		source_count += 1
		if str(source_name) == "concat":
			concat_count += 1
		else:
			segment_count += 1
		# Drop the analyzer reference before advancing so its waveform tensors can be freed.
		del analyzer
		gc.collect()
	summary = {
		"phase": ("analyzers" if source_scope is None else f"analyzers.{source_scope}"),
		"stream_id": str(inputs.stream_id),
		"well_out_dir": str(well_out_dir),
		"templates_out_dir": str(templates_out_dir),
		"applied_debug_limits": _templates_applied_debug_limits(inputs),
		"analyzer_cache_dir": (None if analyzer_cache_dir is None else str(analyzer_cache_dir)),
		"source_scope": source_scope,
		"source_count": int(source_count),
		"load_stats": load_stats,
		"sources": sources_summary,
	}
	summary["timing"] = {"duration_seconds": float(perf_counter() - phase_started)}
	summary_path = templates_out_dir / str(inputs.phases.analyzers.summary_json_relpath)
	LOGGER.info("templates.analyzers generating outputs: summary_json=%s", str(summary_path))
	write_json(summary_path, summary)
	summary["summary_json"] = str(summary_path)
	LOGGER.info("templates.analyzers wrote summary output: %s", str(summary_path))
	LOGGER.info(
		"templates.analyzers run stats: duration_seconds=%.3f source_count=%d concat_count=%d segment_count=%d load_stats=%s",
		float(summary["timing"]["duration_seconds"]),
		int(source_count),
		int(concat_count),
		int(segment_count),
		load_stats,
	)
	return summary


def run_reconstruct_templates_extract_template_segments_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	well_out_dir, alternate_well_out_dirs, templates_out_dir, analyzer_cache_dir = _resolve_templates_phase_environment(inputs)
	payload_root = templates_out_dir / Path(str(inputs.phases.per_unit_processing.extract_template_segments.output_rel_root)).expanduser()
	if bool(inputs.force_restart) and payload_root.exists():
		shutil.rmtree(payload_root)
	payload_root.mkdir(parents=True, exist_ok=True)
	analyzers = _load_templates_phase_analyzers(
		inputs=inputs,
		well_out_dir=well_out_dir,
		alternate_well_out_dirs=alternate_well_out_dirs,
		analyzer_cache_dir=analyzer_cache_dir,
	)
	unit_ids = _collect_templates_phase_unit_ids(inputs=inputs, analyzers=analyzers, well_out_dir=well_out_dir)
	sources_summary: dict[str, Any] = {}
	for source_name, analyzer in analyzers:
		policy = _templates_analyzer_policy_for_source(inputs, source_name)
		written_units: list[Any] = []
		for unit_id in unit_ids:
			payload = build_unit_source_payload(
				analyzer=analyzer,
				unit_id=unit_id,
				max_spikes_per_unit=policy.max_spikes_per_unit,
				min_spikes_per_unit=policy.min_spikes_per_unit,
				waveform_ms_before=policy.ms_before,
				waveform_ms_after=policy.ms_after,
				waveform_dtype=policy.dtype,
				random_spikes_method=policy.random_spikes_method,
				random_spikes_percentage=policy.random_spikes_percentage,
				random_seed=policy.random_seed,
				log_before_after_spike_counts=policy.log_before_after_spike_counts,
				margin_size=policy.margin_size,
				compute_n_jobs=policy.n_jobs,
				compute_chunk_duration=policy.chunk_duration,
			)
			if payload is None:
				continue
			write_materialized_source_payload(
				templates_out_dir=templates_out_dir,
				output_rel_root=str(inputs.phases.per_unit_processing.extract_template_segments.output_rel_root),
				source_name=str(source_name),
				unit_id=unit_id,
				template_c_by_t=payload[0],
				locations_xy=payload[1],
				electrode_ids=payload[2],
				channel_ids=payload[3],
				waveform_count=payload[4],
				sampling_rate_hz=payload[5],
				overlay_waveforms=payload[6],
				top_electrode_id=payload[7],
				total_waveforms_at_channel=payload[8],
			)
			written_units.append(unit_id)
		sources_summary[str(source_name)] = {
			"units_written": [unit for unit in written_units],
			"unit_count": int(len(written_units)),
		}
	summary = {
		"phase": "per_unit_processing.extract_template_segments",
		"stream_id": str(inputs.stream_id),
		"well_out_dir": str(well_out_dir),
		"templates_out_dir": str(templates_out_dir),
		"payload_root": str(payload_root),
		"applied_debug_limits": _templates_applied_debug_limits(inputs),
		"unit_ids": [unit for unit in unit_ids],
		"sources": sources_summary,
	}
	summary_path = templates_out_dir / str(inputs.phases.per_unit_processing.extract_template_segments.summary_json_relpath)
	write_json(summary_path, summary)
	summary["summary_json"] = str(summary_path)
	return summary


def _templates_payload_root_status(payload_root: Path) -> str:
	if not payload_root.exists():
		return "missing"
	try:
		if any(path.is_dir() for path in payload_root.iterdir()):
			return "ready"
	except Exception:
		return "unreadable"
	return "empty"


def run_reconstruct_templates_build_templates_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	phase_started = perf_counter()
	well_out_dir, _, templates_out_dir, analyzer_cache_dir = _resolve_templates_phase_environment(inputs)
	alternate_well_out_dirs = _resolve_alternate_well_out_dirs(inputs=inputs, primary_well_out_dir=well_out_dir)
	payload_root = templates_out_dir / Path(str(inputs.phases.per_unit_processing.extract_template_segments.output_rel_root)).expanduser()
	payload_status = _templates_payload_root_status(payload_root)
	LOGGER.info(
		"templates.build_templates start: well_out_dir=%s templates_out_dir=%s payload_root=%s force_restart=%s",
		str(well_out_dir),
		str(templates_out_dir),
		str(payload_root),
		bool(inputs.force_restart),
	)
	LOGGER.info(
		"templates.build_templates settings: merge_enable=%s merge_method=%s centering_method=%s max_waveforms_per_source_channel=%s upsampling_enabled=%s upsampling_factor=%d upsampling_method=%s",
		bool(inputs.phases.build_templates.merge.enable),
		str(inputs.phases.build_templates.merge.method),
		str(inputs.phases.build_templates.merge.centering_method),
		(
			"unlimited"
			if inputs.phases.build_templates.merge.max_waveforms_per_source_channel is None
			else str(int(inputs.phases.build_templates.merge.max_waveforms_per_source_channel))
		),
		bool(inputs.phases.build_templates.execution_upsampling.enabled),
		int(max(1, int(inputs.phases.build_templates.execution_upsampling.factor))),
		str(inputs.phases.build_templates.execution_upsampling.method),
	)
	if bool(inputs.force_restart) and payload_root.exists():
		LOGGER.info("templates.build_templates clearing persisted source payloads on force_restart: %s", str(payload_root))
		shutil.rmtree(payload_root)
		payload_status = "missing"
	if bool(inputs.force_restart) or payload_status != "ready":
		bootstrap_reason = ("force_restart" if bool(inputs.force_restart) else payload_status)
		LOGGER.info(
			"templates.build_templates loading source payloads from cached analyzers: payload_root=%s reason=%s",
			str(payload_root),
			bootstrap_reason,
		)
		summary = _build_templates_phase_from_cached_analyzers(
			inputs=inputs,
			well_out_dir=well_out_dir,
			alternate_well_out_dirs=alternate_well_out_dirs,
			templates_out_dir=templates_out_dir,
			payload_root=payload_root,
		)
		LOGGER.info(
			"templates.build_templates loaded source payloads from cached analyzers: payload_root=%s source_count=%d analyzer_well_out_dir=%s analyzer_cache_dir=%s",
			str(payload_root),
			int(summary.get("source_count", 0)),
			str(summary.get("source_payload_well_out_dir", "")),
			str(summary.get("analyzer_cache_dir", "")),
		)
	else:
		summary = build_templates_phase_from_payloads(
			inputs=inputs,
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
		)
	summary["timing"] = {"duration_seconds": float(perf_counter() - phase_started)}
	summary["applied_debug_limits"] = _templates_applied_debug_limits(inputs)
	summary_path = templates_out_dir / str(inputs.phases.build_templates.summary_json_relpath)
	LOGGER.info("templates.build_templates generating outputs: summary_json=%s", str(summary_path))
	write_json(summary_path, summary)
	summary["summary_json"] = str(summary_path)
	LOGGER.info("templates.build_templates wrote summary output: %s", str(summary_path))
	LOGGER.info(
		"templates.build_templates run stats: duration_seconds=%.3f unit_count=%d built_units=%d skipped_units=%d",
		float(summary["timing"]["duration_seconds"]),
		int(summary.get("unit_count", 0)),
		int(len(summary.get("built_units", []))),
		int(len(summary.get("skipped_units", []))),
	)
	return summary


def run_reconstruct_templates_compute_template_similarity_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	phase_started = perf_counter()
	well_out_dir, _, templates_out_dir, _ = _resolve_templates_phase_environment(inputs)
	try:
		merged_units_dir, _ = _resolve_templates_dirs(
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
		)
	except FileNotFoundError as exc:
		raise FileNotFoundError(
			"Missing built template artifacts for compute_template_similarity; run templates.build_templates first"
		) from exc
	unit_ids = _build_unit_ids(inputs, merged_units_dir)
	unit_ids = _apply_unit_label_filter(inputs, unit_ids, well_out_dir, context="compute_template_similarity")
	if not unit_ids:
		raise FileNotFoundError(
			f"No built template artifacts found under {merged_units_dir}; run templates.build_templates first"
		)
	phase_cfg = inputs.phases.compute_template_similarity
	output_paths = resolve_similarity_output_paths(
		templates_out_dir=templates_out_dir,
		similarity=phase_cfg,
	)
	worker_count = int(max(1, min(len(unit_ids), int(max(1, int(inputs.n_jobs))))))
	pair_plot_dir = output_paths["template_similarity_candidate_pair_plots_dir"]
	if pair_plot_dir.exists():
		shutil.rmtree(pair_plot_dir)
	for output_key in (
		"template_similarity_matrix_png",
		"template_similarity_matrix_svg",
		"template_similarity_scores_json",
		"template_similarity_candidate_pairs_json",
	):
		artifact_path = output_paths[output_key]
		if artifact_path.exists():
			artifact_path.unlink()
	summary_path = templates_out_dir / str(phase_cfg.summary_json_relpath)
	if summary_path.exists():
		summary_path.unlink()
	unit_payloads_by_key: dict[str, TemplateSimilarityUnitInput] = {}
	missing_units: list[dict[str, Any]] = []
	progress_interval = max(1, len(unit_ids) // 10)

	def _load_unit_payload(unit_id: Any) -> tuple[Any, TemplateSimilarityUnitInput | None, dict[str, Any] | None]:
		merged_dir = merged_units_dir / f"unit_{unit_id}"
		try:
			merged_template, merged_locs = _load_merged_unit(merged_dir)
		except Exception as exc:
			return (
				unit_id,
				None,
				{
					"unit_id": unit_id,
					"reason": "failed_to_load_merged_template",
					"error": str(exc),
					"path": str(merged_dir),
				},
			)
		return (
			unit_id,
			TemplateSimilarityUnitInput(
				unit_id=unit_id,
				template_c_by_t=merged_template,
				locations_xy=merged_locs,
			),
			None,
		)

	LOGGER.info(
		"templates.compute_template_similarity load start: units=%d worker_count=%d",
		int(len(unit_ids)),
		int(worker_count),
	)
	if worker_count <= 1 or len(unit_ids) <= 1:
		for idx, unit_id in enumerate(unit_ids, start=1):
			loaded_unit_id, payload, missing = _load_unit_payload(unit_id)
			if payload is not None:
				unit_payloads_by_key[str(loaded_unit_id)] = payload
			if missing is not None:
				missing_units.append(missing)
			if (idx % progress_interval == 0) or (idx == len(unit_ids)):
				LOGGER.info(
					"templates.compute_template_similarity load progress: %d/%d units scanned",
					int(idx),
					int(len(unit_ids)),
				)
	else:
		with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
			futures = {pool.submit(_load_unit_payload, unit_id): unit_id for unit_id in unit_ids}
			completed = 0
			for future in concurrent.futures.as_completed(futures):
				loaded_unit_id, payload, missing = future.result()
				if payload is not None:
					unit_payloads_by_key[str(loaded_unit_id)] = payload
				if missing is not None:
					missing_units.append(missing)
				completed += 1
				if (completed % progress_interval == 0) or (completed == len(unit_ids)):
					LOGGER.info(
						"templates.compute_template_similarity load progress: %d/%d units scanned",
						int(completed),
						int(len(unit_ids)),
					)
	unit_payloads = [unit_payloads_by_key[str(unit_id)] for unit_id in unit_ids if str(unit_id) in unit_payloads_by_key]
	if not unit_payloads:
		raise FileNotFoundError(
			"Missing built template artifacts for compute_template_similarity; run templates.build_templates first"
		)
	LOGGER.info(
		"templates.compute_template_similarity start: templates_out_dir=%s loaded_units=%d missing_units=%d method=%s worker_count=%d",
		str(templates_out_dir),
		len(unit_payloads),
		len(missing_units),
		str(phase_cfg.method),
		int(worker_count),
	)
	summary = build_template_similarity_phase_summary(
		unit_payloads=unit_payloads,
		templates_out_dir=templates_out_dir,
		config=phase_cfg,
		output_paths=output_paths,
		per_unit_outputs=inputs.per_unit_outputs,
		probe_geometry=inputs.probe_geometry,
		missing_units=missing_units,
	)
	summary["stream_id"] = str(inputs.stream_id)
	summary["well_out_dir"] = str(well_out_dir)
	summary["applied_debug_limits"] = _templates_applied_debug_limits(inputs)
	summary["duration_seconds"] = float(perf_counter() - phase_started)
	summary_path = templates_out_dir / str(phase_cfg.summary_json_relpath)
	write_json(summary_path, summary)
	summary["summary_json"] = str(summary_path)
	LOGGER.info("templates.compute_template_similarity wrote summary output: %s", str(summary_path))
	LOGGER.info(
		"templates.compute_template_similarity run stats: duration_seconds=%.3f unit_count=%d pair_count=%d candidate_pairs=%d missing_units=%d",
		float(summary["duration_seconds"]),
		int(summary.get("unit_count", 0)),
		int(summary.get("pair_count", 0)),
		int(summary.get("candidate_pair_count", 0)),
		int(len(summary.get("missing_units", []))),
	)
	return summary


def _resolve_plot_templates_execution_plan(
	*,
	inputs: TemplatesInputs,
	unit_ids: list[Any],
) -> tuple[int, int, int, list[list[Any]]]:
	unit_count = len(unit_ids)
	if unit_count <= 0:
		return 1, 1, 1, []
	return 1, 1, int(unit_count), [list(unit_ids)]


def _debug_prints_enabled(inputs: TemplatesInputs) -> bool:
	return bool(getattr(inputs.phases.plot_templates, "debug_prints", False))


@contextmanager
def _quiet_unexpected_plot_logs(inputs: TemplatesInputs):
	if _debug_prints_enabled(inputs):
		yield
		return
	original_levels: dict[str, int] = {}
	for logger_name in NOISY_PLOT_LOGGER_NAMES:
		logger = logging.getLogger(logger_name)
		original_levels[logger_name] = int(logger.level)
		if int(logger.getEffectiveLevel()) < int(logging.WARNING):
			logger.setLevel(logging.WARNING)
	try:
		yield
	finally:
		for logger_name, level in original_levels.items():
			logging.getLogger(logger_name).setLevel(level)


def _plot_safe_template_wf_overlay_config(inputs: TemplatesInputs) -> Any:
	config = inputs.per_unit_outputs.template_wf_overlay
	if _debug_prints_enabled(inputs):
		return config
	return replace(config, debug_mode=False)


def _plot_safe_propagation_config(inputs: TemplatesInputs, config: Any) -> Any:
	if _debug_prints_enabled(inputs):
		return config
	return replace(config, debug_max_amps_at_each_channel=False)


def _write_reconstruct_templates_summary(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	templates_out_dir: Path,
	unit_results: list[UnitTemplatesResult],
	report_outputs: dict[str, str],
	analyzer_cache_dir: Path | None,
	upsampling_decisions_by_unit: dict[Any, dict[str, Any]],
	reports_replot_requested: bool,
	report_only_rerun: bool,
	preserve_stage_reports: bool,
) -> Path:
	summary_json = templates_out_dir / "templates_summary.json"
	report_grid_sort_by = normalize_grid_sort_by(inputs.reports.grid_sort_by, default="unit_id")
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
		"applied_debug_limits": _templates_applied_debug_limits(inputs),
		"analyzer_cache": {
			"enabled": bool(inputs.analyzer_cache.enabled),
			"relpath": str(inputs.analyzer_cache.relpath),
			"concat_analyzer_subdir": str(inputs.analyzer_cache.concat_analyzer_subdir),
			"segment_analyzers_subdir": str(inputs.analyzer_cache.segment_analyzers_subdir),
			"cleanup_on_success": bool(inputs.analyzer_cache.cleanup_on_success),
			"reuse_on_force_restart": bool(inputs.analyzer_cache.reuse_on_force_restart),
			"resolved_dir": (None if analyzer_cache_dir is None else str(analyzer_cache_dir)),
		},
		"reports": report_outputs,
		"reports_replot_from_disk": reports_replot_requested,
		"force_rereport": report_only_rerun,
		"reports_overwrite_skipped": preserve_stage_reports,
		"reports_grid_sort_by": str(report_grid_sort_by),
		"reports_time_upsample": {
			"enabled": bool(inputs.reports.time_upsample.enabled),
			"factor": int(max(1, int(inputs.reports.time_upsample.factor))),
			"method": str(inputs.reports.time_upsample.method),
		},
		"unit_label_filter": {
			"allowed_labels": list(inputs.unit_label_filter_labels),
			"required": bool(inputs.unit_label_filter_required),
		},
		"execution_inputs": {
			"concat_analyzer_relpath": inputs.concat_analyzer_relpath,
			"concat_sorting_relpath": inputs.concat_sorting_relpath,
			"preprocessed_concat_reldir": inputs.preprocessed_concat_reldir,
			"preprocessed_segments_reldir": inputs.preprocessed_segments_reldir,
			"preproc_seg_sources_reldir": inputs.preproc_seg_sources_reldir,
		},
		"include_concat": bool(inputs.include_concat),
		"include_segments": bool(inputs.include_segments),
		"require_concat_analyzer": bool(inputs.require_concat_analyzer),
		"require_segment_analyzers": bool(inputs.require_segment_analyzers),
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
	return summary_json


def _run_reconstruct_templates_plot_batches(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	templates_out_dir: Path,
	unit_ids: list[Any],
) -> TemplatesResult:
	plot_unit_workers, unit_procs, unit_batch_size, batches = _resolve_plot_templates_execution_plan(
		inputs=inputs,
		unit_ids=unit_ids,
	)
	LOGGER.info(
		"templates.plot_templates execution plan: requested_units=%d derived_unit_workers=%d plot_unit_workers=%d unit_procs=%d unit_batch_size=%d unit_batches=%d parallel=false",
		len(unit_ids),
		int(max(1, int(inputs.n_jobs))),
		int(plot_unit_workers),
		int(unit_procs),
		int(unit_batch_size),
		len(batches),
	)
	with _quiet_unexpected_plot_logs(inputs):
		return _run_reconstruct_templates_pipeline_monolithic(replace(inputs, unit_ids=list(unit_ids), n_jobs=1))


def run_reconstruct_templates_plot_templates_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	phase_started = perf_counter()
	well_out_dir, _, templates_out_dir, _ = _resolve_templates_phase_environment(inputs)
	try:
		merged_units_dir, _ = _resolve_templates_dirs(
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
		)
	except FileNotFoundError as exc:
		raise FileNotFoundError(
			"Missing built template artifacts for plot_templates; run templates.build_templates first"
		) from exc
	unit_ids = _build_unit_ids(inputs, merged_units_dir)
	unit_ids = _apply_unit_label_filter(inputs, unit_ids, well_out_dir, context="plot_templates")
	if not unit_ids:
		raise FileNotFoundError(
			f"No built template artifacts found under {merged_units_dir}; run templates.build_templates first"
		)
	phase_inputs = build_plot_templates_phase_inputs(inputs)
	requested_outputs = requested_plot_output_keys(phase_inputs.per_unit_outputs)
	force_replot_requested = (
		bool(inputs.force_restart)
		or bool(inputs.force_replot)
		or bool(inputs.force_replot_per_unit)
	)
	_cleanup_unit_output_artifacts(
		templates_out_dir=templates_out_dir,
		unit_ids=unit_ids,
		per_unit_outputs=phase_inputs.per_unit_outputs,
		output_keys=excluded_plot_output_keys(),
	)
	units_to_render: list[Any] = []
	skipped_units: list[Any] = []
	for unit_id in unit_ids:
		existing_outputs = _collect_existing_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=phase_inputs.per_unit_outputs,
			output_keys=requested_outputs,
		)
		if (not force_replot_requested) and len(existing_outputs) == len(requested_outputs):
			_persist_unit_summary_output_paths(
				templates_out_dir=templates_out_dir,
				unit_id=unit_id,
				per_unit_outputs=phase_inputs.per_unit_outputs,
				output_paths=existing_outputs,
			)
			skipped_units.append(unit_id)
		else:
			units_to_render.append(unit_id)
	LOGGER.info(
		"templates.plot_templates start: templates_out_dir=%s units=%d units_to_render=%d skipped_units=%d force_restart=%s",
		str(templates_out_dir),
		len(unit_ids),
		len(units_to_render),
		len(skipped_units),
		bool(inputs.force_restart),
	)
	result = TemplatesResult(
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
		summary_json=templates_out_dir / "templates_summary.json",
		units=[],
	)
	if units_to_render:
		result = _run_reconstruct_templates_plot_batches(
			inputs=phase_inputs,
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			unit_ids=units_to_render,
		)
	summary = build_plot_templates_phase_summary(
		inputs=phase_inputs,
		result=result,
		skipped_units=skipped_units,
		duration_seconds=float(perf_counter() - phase_started),
	)
	summary["applied_debug_limits"] = _templates_applied_debug_limits(phase_inputs)
	summary_path = templates_out_dir / str(inputs.phases.plot_templates.summary_json_relpath)
	write_json(summary_path, summary)
	summary["summary_json"] = str(summary_path)
	LOGGER.info("templates.plot_templates wrote summary output: %s", str(summary_path))
	LOGGER.info(
		"templates.plot_templates run stats: duration_seconds=%.3f unit_count=%d rendered_units=%d skipped_units=%d failed_units=%d",
		float(summary["duration_seconds"]),
		int(summary.get("unit_count", 0)),
		int(len(summary.get("rendered_units", []))),
		int(len(summary.get("skipped_units", []))),
		int(len(summary.get("failed_units", []))),
	)
	return summary


def run_reconstruct_templates_per_unit_processing_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	_, _, templates_out_dir, _ = _resolve_templates_phase_environment(inputs)
	run_reconstruct_templates_extract_template_segments_phase(inputs)
	run_reconstruct_templates_build_templates_phase(inputs)
	try:
		_resolve_templates_dirs(
			well_out_dir=compute_mea_analysis_output_dir(
				output_root=inputs.mea_output_root,
				data_file=inputs.h5_path,
				well=inputs.stream_id,
			),
			templates_out_dir=templates_out_dir,
		)
	except FileNotFoundError as exc:
		raise FileNotFoundError("Missing built template artifacts for per-unit processing") from exc
	unit_inputs = replace(
		inputs,
		force_restart=False,
		force_replot=True,
		force_replot_per_unit=False,
		force_rereport=False,
		reports=_disable_reports_config(inputs.reports),
	)
	_run_reconstruct_templates_pipeline_monolithic(unit_inputs)
	summary_json = templates_out_dir / "templates_summary.json"
	if summary_json.exists():
		summary = read_json(summary_json)
		if isinstance(summary, dict):
			summary["phase"] = "per_unit_processing"
			return summary
	return {"phase": "per_unit_processing", "templates_out_dir": str(templates_out_dir)}


def run_reconstruct_templates_reports_phase(inputs: TemplatesInputs, *, report_scope: str | None = None) -> dict[str, Any]:
	_, _, templates_out_dir, _ = _resolve_templates_phase_environment(inputs)
	unit_ids = list(inputs.unit_ids) if inputs.unit_ids is not None else _discover_unit_ids_from_unit_summaries(templates_out_dir)
	if not unit_ids:
		raise FileNotFoundError(
			f"No unit summaries found under {templates_out_dir}; run templates.per_unit_processing first"
		)
	missing_unit_summaries: list[Any] = []
	for unit_id in unit_ids:
		paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		if not paths["unit_summary_json"].exists():
			missing_unit_summaries.append(unit_id)
	if missing_unit_summaries:
		raise FileNotFoundError(
			"Missing unit summaries required for reports phase; rerun per_unit_processing first for unit_ids="
			+ str(missing_unit_summaries)
		)
	report_inputs = replace(
		inputs,
		force_restart=False,
		force_replot=False,
		force_replot_per_unit=False,
		force_rereport=True,
		reports=_report_scope_config(inputs.reports, report_scope),
	)
	_run_reconstruct_templates_pipeline_monolithic(report_inputs)
	summary_json = templates_out_dir / "templates_summary.json"
	if summary_json.exists():
		summary = read_json(summary_json)
		if isinstance(summary, dict):
			summary["phase"] = ("reports" if report_scope is None else f"reports.{report_scope}")
			return summary
	return {
		"phase": ("reports" if report_scope is None else f"reports.{report_scope}"),
		"templates_out_dir": str(templates_out_dir),
	}


def run_reconstruct_templates_report_templates_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	phase_started = perf_counter()
	well_out_dir, _, templates_out_dir, _ = _resolve_templates_phase_environment(inputs)
	unit_ids = list(inputs.unit_ids) if inputs.unit_ids is not None else _discover_unit_ids_from_unit_summaries(templates_out_dir)
	unit_ids = _apply_unit_label_filter(inputs, unit_ids, well_out_dir, context="report_templates")
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	if not unit_ids:
		raise FileNotFoundError(
			f"No unit summaries found under {templates_out_dir}; run templates.plot_templates first"
		)

	render_units: list[dict[str, Any]] = []
	missing_units: list[dict[str, Any]] = []
	for unit_id in unit_ids:
		paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		unit_result = _load_unit_result_from_summary(unit_id=unit_id, unit_summary_json=paths["unit_summary_json"])
		if unit_result is None:
			missing_units.append({"unit_id": unit_id, "reason": "missing_unit_summary"})
			continue
		circle_png = unit_result.outputs.get("template_circles_png")
		if circle_png is None:
			missing_units.append({"unit_id": unit_id, "reason": "missing_template_circles_png"})
			continue
		circle_png_path = Path(str(circle_png))
		if not circle_png_path.exists():
			missing_units.append({"unit_id": unit_id, "reason": "missing_circle_png_file", "path": str(circle_png_path)})
			continue
		render_units.append({"unit_id": unit_id, "image_path": str(circle_png_path)})

	if not render_units:
		raise FileNotFoundError(
			"Missing circle plot assets required for report_templates; run templates.plot_templates first"
		)

	report_outputs: dict[str, str] = {}
	if report_templates_pdf_requested(inputs):
		report_path = templates_out_dir / str(inputs.phases.report_templates.relpath)
		LOGGER.info(
			"templates.report_templates start: templates_out_dir=%s units=%d",
			str(templates_out_dir),
			len(render_units),
		)
		report_outputs.update(render_template_report_pdf(units=render_units, pdf_path=report_path))

	summary = build_report_templates_phase_summary(
		inputs=inputs,
		templates_out_dir=templates_out_dir,
		rendered_units=[unit["unit_id"] for unit in render_units],
		missing_units=missing_units,
		report_outputs=report_outputs,
		duration_seconds=float(perf_counter() - phase_started),
	)
	summary["applied_debug_limits"] = _templates_applied_debug_limits(inputs)
	summary_path = templates_out_dir / str(inputs.phases.report_templates.summary_json_relpath)
	write_json(summary_path, summary)
	summary["summary_json"] = str(summary_path)
	LOGGER.info("templates.report_templates wrote summary output: %s", str(summary_path))
	LOGGER.info(
		"templates.report_templates run stats: duration_seconds=%.3f unit_count=%d rendered_units=%d missing_units=%d",
		float(summary["duration_seconds"]),
		int(summary.get("unit_count", 0)),
		int(len(summary.get("rendered_units", []))),
		int(len(summary.get("missing_units", []))),
	)
	return summary


def _run_reconstruct_templates_pipeline_monolithic(inputs: TemplatesInputs) -> TemplatesResult:
	reports_replot_requested = _reports_replot_requested(inputs)
	report_only_rerun = bool(inputs.force_rereport)
	LOGGER.info(
		"Templates stage start: stream=%s force_restart=%s force_replot=%s force_replot_per_unit=%s force_rereport=%s reports_replot_requested=%s reports_replot_from_disk=%s n_jobs=%d",
		str(inputs.stream_id),
		bool(inputs.force_restart),
		bool(inputs.force_replot),
		bool(inputs.force_replot_per_unit),
		report_only_rerun,
		reports_replot_requested,
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
	alternate_well_out_dirs = _resolve_alternate_well_out_dirs(
		inputs=inputs,
		primary_well_out_dir=well_out_dir,
	)
	if alternate_well_out_dirs:
		LOGGER.info(
			"Templates artifact lookup fallbacks enabled: %s",
			[str(path) for path in alternate_well_out_dirs],
		)
	full_restart = (
		bool(inputs.force_restart)
		and (not bool(inputs.force_replot))
		and (not bool(inputs.force_replot_per_unit))
		and (not reports_replot_requested)
	)
	templates_out_dir = well_out_dir / str(inputs.output_rel_root)
	preserve_stage_reports = _should_preserve_templates_reports(inputs)
	existing_report_outputs = (
		_collect_existing_templates_report_outputs(
			templates_out_dir=templates_out_dir,
			inputs=inputs,
		)
		if preserve_stage_reports
		else {}
	)
	cache_rel = Path(str(inputs.analyzer_cache.relpath or "analyzers")).expanduser()
	if cache_rel.is_absolute():
		cache_rel = Path(str(cache_rel).lstrip("/"))
	analyzer_cache_dir = (
		templates_out_dir / cache_rel
		if bool(inputs.analyzer_cache.enabled)
		else None
	)
	unit_scoped_force_restart = full_restart and _is_unit_scoped_templates_run(inputs)
	preserve_analyzer_cache = (
		full_restart
		and (not unit_scoped_force_restart)
		and bool(inputs.analyzer_cache.enabled)
		and bool(inputs.analyzer_cache.reuse_on_force_restart)
		and analyzer_cache_dir is not None
		and analyzer_cache_dir.exists()
	)
	if full_restart and templates_out_dir.exists():
		if unit_scoped_force_restart:
			LOGGER.info(
				"Templates unit-scoped restart: preserving stage root for unit_ids=%s",
				list(inputs.unit_ids or []),
			)
		elif preserve_analyzer_cache:
			LOGGER.info(
				"Templates full restart: clearing output root %s while preserving analyzer cache %s",
				templates_out_dir,
				analyzer_cache_dir,
			)
			_clear_directory_contents_preserving(root_dir=templates_out_dir, preserve_paths=[analyzer_cache_dir])
		else:
			LOGGER.info("Templates full restart: clearing output root %s", templates_out_dir)
			shutil.rmtree(templates_out_dir)
	templates_out_dir.mkdir(parents=True, exist_ok=True)

	merged_units_dir: Path | None = None
	full_channels_templates_dir: Path | None = None
	upsampling_decisions_by_unit: dict[Any, dict[str, Any]] = {}

	def _ensure_templates_dirs() -> tuple[Path, Path]:
		nonlocal merged_units_dir, full_channels_templates_dir, upsampling_decisions_by_unit
		if merged_units_dir is None or full_channels_templates_dir is None:
			overlay_debug_mode = _debug_prints_enabled(inputs) and bool(getattr(inputs.per_unit_outputs.template_wf_overlay, "debug_mode", False))
			prefer_spikeinterface = bool(full_restart)
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
						concat_sorting_relpath=inputs.concat_sorting_relpath,
						preprocessed_concat_reldir=inputs.preprocessed_concat_reldir,
						preprocessed_segments_reldir=inputs.preprocessed_segments_reldir,
						preproc_seg_sources_reldir=inputs.preproc_seg_sources_reldir,
						analyzer_cache_dir=analyzer_cache_dir,
						analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
						analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
						alternate_well_out_dirs=alternate_well_out_dirs,
						raw_data_h5_path=inputs.h5_path,
						stream_id=str(inputs.stream_id),
						unit_ids=(list(inputs.unit_ids) if inputs.unit_ids is not None else None),
						include_concat=bool(inputs.include_concat),
						include_segments=bool(inputs.include_segments),
						require_concat=bool(inputs.require_concat_analyzer),
						require_segments=bool(inputs.require_segment_analyzers),
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
						unit_reldir=str(inputs.per_unit_outputs.unit_reldir),
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
					templates_out_dir / MATERIALIZED_TEMPLATES_CACHE_RELPATH,
					bool(inputs.include_concat),
					bool(inputs.include_segments),
				)
				materialize_out = materialize_templates_from_spikeinterface(
					well_out_dir=well_out_dir,
					templates_out_dir=templates_out_dir,
					concat_analyzer_relpath=inputs.concat_analyzer_relpath,
					concat_sorting_relpath=inputs.concat_sorting_relpath,
					preprocessed_concat_reldir=inputs.preprocessed_concat_reldir,
					preprocessed_segments_reldir=inputs.preprocessed_segments_reldir,
					preproc_seg_sources_reldir=inputs.preproc_seg_sources_reldir,
					analyzer_cache_dir=analyzer_cache_dir,
					analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
					analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
					alternate_well_out_dirs=alternate_well_out_dirs,
					raw_data_h5_path=inputs.h5_path,
					stream_id=str(inputs.stream_id),
					unit_ids=(list(inputs.unit_ids) if inputs.unit_ids is not None else None),
					include_concat=bool(inputs.include_concat),
					include_segments=bool(inputs.include_segments),
					require_concat=bool(inputs.require_concat_analyzer),
					require_segments=bool(inputs.require_segment_analyzers),
					concat_policy=_resolve_analyzer_policy_runtime_n_jobs(inputs, inputs.phases.analyzers.concat.policy),
					segments_policy=_resolve_analyzer_policy_runtime_n_jobs(inputs, inputs.phases.analyzers.segments.policy),
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
					unit_reldir=str(inputs.per_unit_outputs.unit_reldir),
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
		reports_replot_requested
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
	if bool(inputs.log_stage_unit_counts):
		LOGGER.info("Templates stage discovered %d unit(s) before label filtering", len(unit_ids))
	unit_ids = _apply_unit_label_filter(inputs, unit_ids, well_out_dir, context="stage")

	if bool(inputs.log_stage_unit_counts):
		LOGGER.info("Templates stage will process %d unit(s)", len(unit_ids))

	similarity_outputs: dict[str, str] = {}
	if preserve_stage_reports:
		LOGGER.info(
			"Templates stage-level similarity skip: preserving existing stage outputs during unit-scoped rerun for unit_ids=%s",
			list(inputs.unit_ids or []),
		)
	elif bool(inputs.phases.compute_template_similarity.enabled):
		try:
			similarity_summary = run_reconstruct_templates_compute_template_similarity_phase(inputs)
			if isinstance(similarity_summary, dict):
				summary_outputs = similarity_summary.get("outputs", {})
				if isinstance(summary_outputs, dict):
					similarity_outputs.update(
						{
							str(key): str(value)
							for key, value in summary_outputs.items()
							if value is not None
						}
					)
		except Exception:
			LOGGER.exception("Failed writing template similarity outputs for stream %s", inputs.stream_id)

	def _process_unit(unit_id: Any) -> UnitTemplatesResult:
		LOGGER.info("Templates unit start: unit_id=%s", unit_id)
		paths = resolve_unit_output_paths(
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			per_unit_outputs=inputs.per_unit_outputs,
		)
		if unit_scoped_force_restart and any(str(unit_id) == str(target_id) for target_id in (inputs.unit_ids or [])) and paths["unit_dir"].exists():
			_clear_directory_contents_preserving(
				root_dir=paths["unit_dir"],
				preserve_paths=[
					paths["merged_contributing_electrode_ids_json"],
					paths["overlay_top_channel_meta_json"],
				],
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
			overlay_cfg = _plot_safe_template_wf_overlay_config(inputs)
			overlay_debug_mode = _debug_prints_enabled(inputs) and bool(getattr(overlay_cfg, "debug_mode", False))
			LOGGER.info("Templates unit load artifacts: unit_id=%s merged_dir=%s", unit_id, merged_dir)

			merged_template, merged_locs = _load_merged_unit(merged_dir)
			merged_electrode_ids = load_materialized_merged_electrode_ids(merged_unit_dir=merged_dir, metadata_dir=paths["unit_dir"])
			unit_summary["grid_sort_metrics"] = compute_template_grid_sort_metrics(
				template_c_by_t=merged_template,
				locations_xy=merged_locs,
				sampling_rate_hz=unit_summary.get("effective_sampling_rate_hz", None),
				probe_pitch_um=(None if unit_probe_geometry is None else unit_probe_geometry.pitch_um),
			)
			unit_summary["unit_location"] = _compute_unit_location_from_template(
				unit_id=unit_id,
				template_c_by_t=merged_template,
				locations_xy=merged_locs,
				source="merged_contributing",
			)
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
					qc_prop_config = _plot_safe_propagation_config(inputs, qc_prop_config)
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

			for stale_npy_key in (
				"merged_template_npy",
				"merged_template_channel_locations_npy",
				"full_template_npy",
				"full_template_channel_locations_npy",
				"scan_template_npy",
				"scan_template_channel_locations_npy",
				"square_template_npy",
				"square_template_channel_locations_npy",
			):
				stale_path = paths.get(stale_npy_key)
				if stale_path is not None and stale_path.exists():
					stale_path.unlink()
				unit_summary["outputs"].pop(stale_npy_key, None)

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

			overlay_payload = load_materialized_overlay_waveforms(merged_unit_dir=merged_dir, metadata_dir=paths["unit_dir"])
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
					config=overlay_cfg,
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

			if propagation_outputs_requested(inputs.per_unit_outputs.propagation_plots):
				prop_cfg = _plot_safe_propagation_config(inputs, inputs.per_unit_outputs.propagation_plots)
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
		reports_replot_requested
		and (not bool(inputs.force_restart))
		and (not bool(inputs.force_replot))
		and (not bool(inputs.force_replot_per_unit))
	):
		units_to_process = []
		missing_unit_summaries: list[Any] = []
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
			elif report_only_rerun:
				missing_unit_summaries.append(unit_id)
			else:
				units_to_process.append(unit_id)
		if units_to_process:
			LOGGER.info(
				"Templates reports replot_from_disk: %d unit summaries missing; processing those units from templates artifacts",
				len(units_to_process),
			)
		if missing_unit_summaries:
			LOGGER.warning(
				"Templates force_rereport: skipping %d unit(s) without existing unit summaries: %s",
				len(missing_unit_summaries),
				[unit for unit in missing_unit_summaries],
			)

	worker_count = int(max(1, int(inputs.n_jobs)))
	if bool(inputs.log_stage_unit_counts):
		LOGGER.info(
			"Templates unit execution start: units_to_process=%d reused_units=%d worker_count=%d",
			len(units_to_process),
			len(unit_results),
			worker_count,
		)
	add_current_progress_total(len(units_to_process))
	if worker_count <= 1 or len(units_to_process) <= 1:
		total_to_run = len(units_to_process)
		for idx, unit_id in enumerate(units_to_process, start=1):
			unit_results.append(_process_unit(unit_id))
			advance_current_progress()
			if bool(inputs.log_unit_progress):
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
				advance_current_progress()
				if bool(inputs.log_unit_progress):
					LOGGER.info("Templates unit progress: %d/%d completed", completed, total_to_run)

	unit_results.sort(key=lambda r: str(r.unit_id))
	LOGGER.info("Templates unit execution complete: total_results=%d", len(unit_results))
	report_grid_sort_by = normalize_grid_sort_by(inputs.reports.grid_sort_by, default="unit_id")
	unit_results_for_reports = _sort_template_units_for_reports(
		unit_results=unit_results,
		templates_out_dir=templates_out_dir,
		inputs=inputs,
		sort_by=report_grid_sort_by,
	)
	report_outputs: dict[str, str] = dict(existing_report_outputs)
	report_outputs.update(similarity_outputs)
	if preserve_stage_reports:
		LOGGER.info(
			"Templates reports skip: preserving existing stage reports during unit-scoped rerun for unit_ids=%s",
			list(inputs.unit_ids or []),
		)
	else:
		try:
			LOGGER.info("Templates reports start: stream=%s", inputs.stream_id)
			report_paths = resolve_report_output_paths(templates_out_dir=templates_out_dir, reports=inputs.reports)

			location_rows: list[dict[str, Any]] = []
			for unit_result in unit_results_for_reports:
				row = _load_templates_unit_location_row(
					templates_out_dir=templates_out_dir,
					unit_id=unit_result.unit_id,
					inputs=inputs,
				)
				if row is not None:
					location_rows.append(row)
			need_concat_underlay = bool(inputs.reports.locations.underlay_concat_channels)
			need_redlines = bool(inputs.reports.locations.show_original_to_current_redlines)
			concat_channel_locs = (
				_load_templates_concat_channel_locations(
					templates_out_dir=templates_out_dir,
					unit_ids_for_priority=[u.unit_id for u in unit_results_for_reports],
				)
				if need_concat_underlay
				else None
			)
			expected_chip_channels = _expected_chip_channel_count(inputs.probe_geometry)
			concat_underlay_is_sparse = (
				concat_channel_locs is not None
				and expected_chip_channels is not None
				and int(concat_channel_locs.shape[0]) < int(max(64, int(0.25 * float(expected_chip_channels))))
			)
			original_unit_locations_by_unit = (
				_load_templates_original_unit_locations(templates_out_dir=templates_out_dir)
				if need_redlines
				else {}
			)
			if (need_concat_underlay and (concat_channel_locs is None or concat_underlay_is_sparse)) or (need_redlines and (not original_unit_locations_by_unit)):
				spikesort_concat_locs, spikesort_original_locations = _load_spikesort_locations_metadata(
					well_out_dir=well_out_dir,
					alternate_well_out_dirs=alternate_well_out_dirs,
					inputs=inputs,
				)
				if need_concat_underlay and spikesort_concat_locs is not None:
					if concat_channel_locs is None or int(spikesort_concat_locs.shape[0]) > int(concat_channel_locs.shape[0]):
						concat_channel_locs = spikesort_concat_locs
				if need_redlines and (not original_unit_locations_by_unit):
					original_unit_locations_by_unit = spikesort_original_locations
			if need_concat_underlay and concat_channel_locs is None:
				concat_channel_locs = _infer_chip_grid_from_probe_geometry(inputs.probe_geometry)
			template_channel_locs_by_unit = (
				_load_templates_channel_locations_by_unit(
					templates_out_dir=templates_out_dir,
					unit_ids=[u.unit_id for u in unit_results_for_reports],
				)
				if bool(inputs.reports.locations.underlay_template_channels)
				else None
			)
			LOGGER.info("Templates reports locations inputs=%d", len(location_rows))
			report_outputs.update(
				render_unit_locations_report(
					unit_location_rows=location_rows,
					config=inputs.reports.locations,
					json_path=report_paths["unit_locations_json"],
					png_path=report_paths["unit_locations_png"],
					svg_path=report_paths["unit_locations_svg"],
					probe_geometry=inputs.probe_geometry,
					concat_channel_locations_xy=concat_channel_locs,
					template_channel_locations_by_unit=template_channel_locs_by_unit,
					original_unit_locations_by_unit=original_unit_locations_by_unit,
				)
			)

			overlay_paths = [
				Path(u.outputs["template_wf_overlay_png"])
				for u in unit_results_for_reports
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
			wf_grid_outputs = finalize_grid_svg_output(
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
				for u in unit_results_for_reports
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
			circles_grid_outputs = finalize_grid_svg_output(
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
				for u in unit_results_for_reports
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
			amp_grid_outputs = finalize_grid_svg_output(
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
				for u in unit_results_for_reports
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
			lat_grid_outputs = finalize_grid_svg_output(
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
	if bool(inputs.write_stage_summary):
		summary_json = _write_reconstruct_templates_summary(
			inputs=inputs,
			well_out_dir=well_out_dir,
			templates_out_dir=templates_out_dir,
			unit_results=unit_results,
			report_outputs=report_outputs,
			analyzer_cache_dir=analyzer_cache_dir,
			upsampling_decisions_by_unit=upsampling_decisions_by_unit,
			reports_replot_requested=reports_replot_requested,
			report_only_rerun=report_only_rerun,
			preserve_stage_reports=preserve_stage_reports,
		)
	ok_count = sum(1 for u in unit_results if str(u.status) == "ok")
	err_count = sum(1 for u in unit_results if str(u.status) != "ok")
	if (
		bool(inputs.analyzer_cache.enabled)
		and bool(inputs.analyzer_cache.cleanup_on_success)
		and analyzer_cache_dir is not None
		and analyzer_cache_dir.exists()
		and err_count == 0
	):
		try:
			LOGGER.info("Templates analyzer cache cleanup_on_success: removing %s", analyzer_cache_dir)
			shutil.rmtree(analyzer_cache_dir)
		except Exception:
			LOGGER.warning("Failed to remove analyzer cache dir: %s", analyzer_cache_dir, exc_info=True)
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
