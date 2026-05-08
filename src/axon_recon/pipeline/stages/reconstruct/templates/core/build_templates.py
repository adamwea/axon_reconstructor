from __future__ import annotations

import concurrent.futures
import gc
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.cpu_allocation import current_phase_budget, resolve_inner_worker_count
from axon_recon.pipeline.execution import install_linux_parent_death_signal
from axon_recon.pipeline.shared.grid_sorting import compute_template_grid_sort_metrics
from axon_recon.pipeline.shared.sampling import read_maxwell_sampling_frequency_hz

from ..io import (
	SOURCE_PAYLOADS_CACHE_RELPATH,
	load_materialized_source_payload,
	read_json,
	resolve_materialized_source_payload_unit_dir,
	resolve_materialized_templates_dirs,
	resolve_unit_output_paths,
	write_json,
	write_materialized_merged_electrode_ids,
	write_materialized_unit_templates,
)
from ..models.inputs import TemplatesInputs
from .merge import materialize_unit_templates_from_sources_with_meta, normalize_overlap_priorities
from .unit_labels import count_labels, filter_unit_ids_by_labels, load_unit_labels_from_spikesorting

LOGGER = logging.getLogger("axon_recon.templates.build_templates")


@dataclass(frozen=True)
class _BuildTemplatesUnitJob:
	inputs: TemplatesInputs
	templates_out_dir: Path
	merged_units_dir: Path
	full_channels_templates_dir: Path
	unit_id: Any
	source_names: tuple[str, ...]
	payload_output_rel_root: str
	raw_sampling_rate_hz: float


def _payload_output_rel_root_from_payload_root(*, templates_out_dir: Path, payload_root: Path | None) -> str | None:
	if payload_root is None:
		return None
	try:
		return str(Path(payload_root).relative_to(Path(templates_out_dir)))
	except ValueError:
		return None


def _load_source_payloads_from_disk(
	*,
	templates_out_dir: Path,
	output_rel_root: str,
	source_names: list[str] | tuple[str, ...],
	unit_id: Any,
) -> list[tuple[str, tuple[Any, ...]]]:
	loaded: list[tuple[str, tuple[Any, ...]]] = []
	for source_name in source_names:
		payload = load_materialized_source_payload(
			source_payload_unit_dir=resolve_materialized_source_payload_unit_dir(
				templates_out_dir=templates_out_dir,
				output_rel_root=str(output_rel_root),
				source_name=str(source_name),
				unit_id=unit_id,
			),
		)
		if payload is None:
			continue
		loaded.append((str(source_name), payload))
	return loaded


def _apply_unit_label_filter(inputs: TemplatesInputs, unit_ids: list[Any], well_out_dir: Path) -> list[Any]:
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
		LOGGER.warning("Templates build phase: unit label filter skipped because no labels were found under %s", well_out_dir)
		return list(unit_ids)
	filtered = filter_unit_ids_by_labels(unit_ids, labels_by_unit, allowed_labels)
	LOGGER.info(
		"Templates build phase: unit label filter allowed=%s kept=%d/%d counts=%s",
		list(allowed_labels),
		len(filtered),
		len(unit_ids),
		count_labels(labels_by_unit),
	)
	return filtered


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


def _as_positive_float_or_none(value: Any) -> float | None:
	try:
		parsed = float(value)
	except Exception:
		return None
	if not np.isfinite(parsed) or parsed <= 0.0:
		return None
	return float(parsed)


def _jsonish_channel_identity(value: Any) -> str | None:
	if value is None:
		return None
	if isinstance(value, np.generic):
		value = value.item()
	try:
		if isinstance(value, float) and not np.isfinite(value):
			return None
	except Exception:
		pass
	text = str(value).strip()
	if not text or text.lower() in {"none", "nan"}:
		return None
	return text


def _channel_location_key(location_xy: Any, *, location_tolerance_um: float) -> str | None:
	try:
		loc = np.asarray(location_xy, dtype=float).reshape(-1)
	except Exception:
		return None
	if int(loc.size) < 2:
		return None
	if (not np.isfinite(loc[0])) or (not np.isfinite(loc[1])):
		return None
	tolerance = max(1e-6, float(location_tolerance_um))
	qx = int(np.rint(float(loc[0]) / tolerance))
	qy = int(np.rint(float(loc[1]) / tolerance))
	return f"loc:{qx}:{qy}"


def _source_payload_channel_candidate_keys(
	*,
	electrode_id: Any,
	channel_id: Any,
	location_xy: Any,
	overlap_match_priority: tuple[str, ...],
	location_tolerance_um: float,
) -> list[str]:
	eid = _jsonish_channel_identity(electrode_id)
	cid = _jsonish_channel_identity(channel_id)
	loc_key = _channel_location_key(location_xy, location_tolerance_um=location_tolerance_um)
	keys: list[str] = []
	for priority in normalize_overlap_priorities(overlap_match_priority):
		if priority == "electrode_id" and eid is not None:
			keys.append(f"eid:{eid}")
		elif priority == "channel_id" and cid is not None:
			keys.append(f"cid:{cid}")
		elif priority == "location" and loc_key is not None:
			keys.append(str(loc_key))
	return keys


def _payload_item(values: Any, index: int) -> Any:
	if values is None:
		return None
	try:
		if int(index) < len(values):
			return values[int(index)]
	except Exception:
		return None
	return None


def _source_payload_channel_scope_summary(
	*,
	unit_id: Any,
	source_payloads: list[tuple[str, tuple[Any, ...]]],
	overlap_match_priority: tuple[str, ...],
	location_tolerance_um: float,
) -> dict[str, Any]:
	source_channel_counts: dict[str, int] = {}
	total_source_channels = 0
	unique_channel_keys: set[str] = set()
	for source_index, (source_name_raw, payload) in enumerate(source_payloads):
		source_name = str(source_name_raw)
		source_key = source_name if source_name not in source_channel_counts else f"{source_name}#{source_index}"
		try:
			template = np.asarray(payload[0], dtype=float)
		except Exception:
			template = np.empty((0, 0), dtype=float)
		channel_count = int(template.shape[0]) if template.ndim >= 1 else 0
		source_channel_counts[source_key] = int(channel_count)
		total_source_channels += int(channel_count)
		try:
			locations = np.asarray(payload[1], dtype=float)
		except Exception:
			locations = np.empty((0, 2), dtype=float)
		electrode_ids = payload[2] if len(payload) >= 3 else None
		channel_ids = payload[3] if len(payload) >= 4 else None
		for channel_index in range(channel_count):
			location_xy = (
				locations[channel_index, :2]
				if locations.ndim == 2 and int(locations.shape[0]) > channel_index and int(locations.shape[1]) >= 2
				else None
			)
			candidate_keys = _source_payload_channel_candidate_keys(
				electrode_id=_payload_item(electrode_ids, channel_index),
				channel_id=_payload_item(channel_ids, channel_index),
				location_xy=location_xy,
				overlap_match_priority=overlap_match_priority,
				location_tolerance_um=location_tolerance_um,
			)
			canonical_key = candidate_keys[0] if candidate_keys else f"source:{source_name}:channel_index:{int(channel_index)}"
			for candidate_key in candidate_keys:
				if candidate_key in unique_channel_keys:
					canonical_key = candidate_key
					break
			unique_channel_keys.add(str(canonical_key))
	max_source_channels = max(source_channel_counts.values(), default=0)
	return {
		"unit_id": unit_id,
		"source_count": int(len(source_payloads)),
		"source_channel_counts": source_channel_counts,
		"total_source_channel_count": int(total_source_channels),
		"max_source_channel_count": int(max_source_channels),
		"total_unique_channel_count": int(len(unique_channel_keys)),
	}


def _log_source_channel_scope_if_requested(*, inputs: TemplatesInputs, scope_summary: dict[str, Any]) -> None:
	if not bool(inputs.phases.analyzers.emit_total_unique_channel_count_per_unit_log):
		return
	unique_channels = int(scope_summary.get("total_unique_channel_count", 0))
	level = logging.WARNING if unique_channels <= 0 else logging.INFO
	LOGGER.log(
		level,
		"templates.build_templates source channel scope: unit_id=%s source_count=%d total_source_channels=%d total_unique_channels=%d max_source_channels=%d per_source=%s",
		scope_summary.get("unit_id"),
		int(scope_summary.get("source_count", 0)),
		int(scope_summary.get("total_source_channel_count", 0)),
		unique_channels,
		int(scope_summary.get("max_source_channel_count", 0)),
		dict(scope_summary.get("source_channel_counts", {})),
	)


def _merged_channel_scope_summary(
	*,
	source_scope_summary: dict[str, Any],
	merged_template: np.ndarray,
	full_template: np.ndarray,
) -> dict[str, Any]:
	summary = dict(source_scope_summary)
	merged_channel_count = int(np.asarray(merged_template).shape[0])
	full_channel_count = int(np.asarray(full_template).shape[0])
	unique_channel_count = int(summary.get("total_unique_channel_count", 0))
	max_source_channels = int(summary.get("max_source_channel_count", 0))
	source_count = int(summary.get("source_count", 0))
	warning_reasons: list[str] = []
	if merged_channel_count <= 0:
		warning_reasons.append("merged_template_has_no_channels")
	if unique_channel_count > 0 and merged_channel_count != unique_channel_count:
		warning_reasons.append("merged_channel_count_differs_from_unique_source_channel_count")
	if source_count > 1 and unique_channel_count > max_source_channels and merged_channel_count <= max_source_channels:
		warning_reasons.append("merged_template_did_not_expand_beyond_largest_source")
	summary.update(
		{
			"merged_channel_count": int(merged_channel_count),
			"full_channel_count": int(full_channel_count),
			"warning_reasons": warning_reasons,
			"ok": not warning_reasons,
		}
	)
	return summary


def _log_merged_channel_scope_if_requested(*, inputs: TemplatesInputs, scope_summary: dict[str, Any]) -> None:
	if not bool(inputs.phases.build_templates.emit_channel_count_per_unit_after_merge_log):
		return
	warning_reasons = list(scope_summary.get("warning_reasons", []))
	level = logging.WARNING if warning_reasons else logging.INFO
	LOGGER.log(
		level,
		"templates.build_templates merged channel scope: unit_id=%s source_unique_channels=%d merged_channels=%d full_channels=%d max_source_channels=%d reasons=%s per_source=%s",
		scope_summary.get("unit_id"),
		int(scope_summary.get("total_unique_channel_count", 0)),
		int(scope_summary.get("merged_channel_count", 0)),
		int(scope_summary.get("full_channel_count", 0)),
		int(scope_summary.get("max_source_channel_count", 0)),
		warning_reasons,
		dict(scope_summary.get("source_channel_counts", {})),
	)


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


def _discover_source_dirs(payload_root: Path) -> list[Path]:
	return sorted(
		[path for path in payload_root.iterdir() if path.is_dir()],
		key=lambda path: (0 if path.name == "concat" else 1, path.name),
	)


def _apply_source_dir_segment_limit(source_dirs: list[Path], *, limit_segments: int | None) -> list[Path]:
	try:
		limit = int(limit_segments) if limit_segments is not None else None
	except Exception:
		limit = None
	if limit is None or limit <= 0:
		return list(source_dirs)
	limited: list[Path] = []
	segment_count = 0
	for source_dir in source_dirs:
		if source_dir.name == "concat":
			limited.append(source_dir)
			continue
		if segment_count >= limit:
			continue
		limited.append(source_dir)
		segment_count += 1
	return limited


def _discover_unit_ids_from_payloads(source_dirs: list[Path]) -> list[Any]:
	unit_tokens: list[Any] = []
	for source_dir in source_dirs:
		for unit_dir in sorted(source_dir.glob("unit_*")):
			token = unit_dir.name.split("unit_", 1)[1]
			try:
				value: Any = int(token)
			except Exception:
				value = token
			if value not in unit_tokens:
				unit_tokens.append(value)
	return unit_tokens


def _artifact_file_complete(path: Path | None) -> bool:
	if path is None:
		return True
	try:
		return path.exists() and path.stat().st_size > 0
	except OSError:
		return False


def _unit_summary_complete(path: Path) -> bool:
	if not _artifact_file_complete(path):
		return False
	try:
		payload = read_json(path)
	except Exception:
		return False
	if not isinstance(payload, dict):
		return False
	return str(payload.get("status", "")).strip().lower() == "ok"


def _materialized_unit_templates_complete(
	*,
	inputs: TemplatesInputs,
	templates_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_id: Any,
) -> bool:
	merged_dir = merged_units_dir / f"unit_{unit_id}"
	required = [
		merged_dir / "merged_contributing_template.npy",
		merged_dir / "merged_contributing_channel_locations.npy",
	]
	if bool(inputs.per_unit_outputs.full_template.write_npy):
		full_dir = full_channels_templates_dir / f"unit_{unit_id}"
		required.extend(
			[
				full_dir / "full_template.npy",
				full_dir / "full_channel_locations_xy.npy",
			]
		)
	if not all(_artifact_file_complete(path) for path in required):
		return False
	unit_paths = resolve_unit_output_paths(
		templates_out_dir=templates_out_dir,
		unit_id=unit_id,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	return _unit_summary_complete(unit_paths["unit_summary_json"])


def _split_units_by_build_artifact_resume(
	*,
	inputs: TemplatesInputs,
	templates_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_ids: list[Any],
) -> tuple[list[Any], list[Any]]:
	if bool(inputs.force_restart):
		return [], list(unit_ids)
	reused_units: list[Any] = []
	pending_units: list[Any] = []
	for unit_id in unit_ids:
		if _materialized_unit_templates_complete(
			inputs=inputs,
			templates_out_dir=templates_out_dir,
			merged_units_dir=merged_units_dir,
			full_channels_templates_dir=full_channels_templates_dir,
			unit_id=unit_id,
		):
			reused_units.append(unit_id)
		else:
			pending_units.append(unit_id)
	return reused_units, pending_units


def _effective_sampling_rate_hz(*, decision: dict[str, Any] | None, inputs: TemplatesInputs) -> float | None:
	if isinstance(decision, dict):
		target_hz = _as_positive_float_or_none(decision.get("target_hz", None))
		applied = bool(decision.get("applied", False))
		analyzer_hz = _as_positive_float_or_none(decision.get("analyzer_hz", None))
		raw_hz = _as_positive_float_or_none(decision.get("raw_hz", None))
		if applied and target_hz is not None:
			return float(target_hz)
		if analyzer_hz is not None:
			return float(analyzer_hz)
		if raw_hz is not None:
			return float(raw_hz)
	if inputs.probe_geometry is None:
		return None
	return _as_positive_float_or_none(inputs.probe_geometry.sampling_rate_hz)


def _remove_disabled_output(path: Path | None) -> None:
	if path is None:
		return
	try:
		if path.exists():
			path.unlink()
	except FileNotFoundError:
		return


def _remove_redundant_per_unit_npy_outputs(paths: dict[str, Path | None]) -> None:
	for key in (
		"merged_template_npy",
		"merged_template_channel_locations_npy",
		"full_template_npy",
		"full_template_channel_locations_npy",
		"scan_template_npy",
		"scan_template_channel_locations_npy",
		"square_template_npy",
		"square_template_channel_locations_npy",
	):
		_remove_disabled_output(paths.get(key))


def _write_per_unit_data_outputs(
	*,
	inputs: TemplatesInputs,
	templates_out_dir: Path,
	unit_id: Any,
	merged_template: np.ndarray,
	merged_locs: np.ndarray,
	full_template: np.ndarray,
	full_locs: np.ndarray,
	decision: dict[str, Any] | None,
	channel_scope_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
	paths = resolve_unit_output_paths(
		templates_out_dir=templates_out_dir,
		unit_id=unit_id,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	paths["unit_dir"].mkdir(parents=True, exist_ok=True)
	outputs: dict[str, str] = {}
	_remove_redundant_per_unit_npy_outputs(paths)
	for key in ("merged_contributing_electrode_ids_json", "overlay_top_channel_meta_json"):
		metadata_path = paths.get(key)
		if metadata_path is not None and metadata_path.exists():
			outputs[key] = str(metadata_path)

	effective_sampling_rate_hz = _effective_sampling_rate_hz(decision=decision, inputs=inputs)
	unit_summary = {
		"unit_id": unit_id,
		"status": "ok",
		"error": None,
		"outputs": outputs,
		"upsampling": (dict(decision) if isinstance(decision, dict) else None),
		"effective_sampling_rate_hz": effective_sampling_rate_hz,
		"grid_sort_metrics": compute_template_grid_sort_metrics(
			template_c_by_t=np.asarray(merged_template, dtype=float),
			locations_xy=np.asarray(merged_locs, dtype=float),
			sampling_rate_hz=effective_sampling_rate_hz,
			probe_pitch_um=(None if inputs.probe_geometry is None else inputs.probe_geometry.pitch_um),
		),
		"unit_location": _compute_unit_location_from_template(
			unit_id=unit_id,
			template_c_by_t=np.asarray(merged_template, dtype=float),
			locations_xy=np.asarray(merged_locs, dtype=float),
			source="merged_contributing",
		),
		"selected_template_source": "merged_contributing",
	}
	if channel_scope_summary is not None:
		unit_summary["channel_scope"] = dict(channel_scope_summary)
	write_json(paths["unit_summary_json"], unit_summary)
	outputs["unit_summary_json"] = str(paths["unit_summary_json"])
	return unit_summary


def _build_templates_unit_from_source_payloads(
	*,
	inputs: TemplatesInputs,
	templates_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_id: Any,
	source_payloads: list[tuple[str, tuple[Any, ...]]],
	raw_sampling_rate_hz: float,
) -> dict[str, Any]:
	merge_cfg = inputs.phases.build_templates.merge
	upsampling_cfg = inputs.phases.build_templates.execution_upsampling
	LOGGER.info("build_templates unit start: unit_id=%s", unit_id)
	if not source_payloads:
		LOGGER.info("build_templates unit skipped: unit_id=%s reason=no_source_payloads", unit_id)
		return {"unit_id": unit_id, "status": "skipped", "reason": "no_source_payloads"}
	source_scope_summary = _source_payload_channel_scope_summary(
		unit_id=unit_id,
		source_payloads=source_payloads,
		overlap_match_priority=tuple(merge_cfg.overlap_match_priority),
		location_tolerance_um=float(merge_cfg.location_tolerance_um),
	)
	_log_source_channel_scope_if_requested(inputs=inputs, scope_summary=source_scope_summary)

	materialized, decision = materialize_unit_templates_from_sources_with_meta(
		source_payloads=source_payloads,
		enable_merge=bool(merge_cfg.enable),
		merge_method=str(merge_cfg.method),
		centering_method=str(merge_cfg.centering_method),
		max_waveforms_per_source_channel=merge_cfg.max_waveforms_per_source_channel,
		overlap_match_priority=tuple(merge_cfg.overlap_match_priority),
		location_tolerance_um=float(merge_cfg.location_tolerance_um),
		execution_upsampling=upsampling_cfg,
		raw_sampling_rate_hz=raw_sampling_rate_hz,
		log_context=f"unit_id={unit_id}",
	)
	if materialized is None:
		LOGGER.info("build_templates unit skipped: unit_id=%s reason=materialization_returned_none", unit_id)
		return {
			"unit_id": unit_id,
			"status": "skipped",
			"reason": "materialization_returned_none",
			"upsampling": dict(decision) if isinstance(decision, dict) else None,
		}

	merged_template, merged_locs, full_template, full_locs, merged_electrode_ids = materialized
	channel_scope_summary = _merged_channel_scope_summary(
		source_scope_summary=source_scope_summary,
		merged_template=merged_template,
		full_template=full_template,
	)
	_log_merged_channel_scope_if_requested(inputs=inputs, scope_summary=channel_scope_summary)
	write_materialized_unit_templates(
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_channels_templates_dir,
		unit_id=unit_id,
		merged_template=merged_template,
		merged_locations_xy=merged_locs,
		full_template=full_template,
		full_locations_xy=full_locs,
		write_full_template=bool(inputs.per_unit_outputs.full_template.write_npy),
	)
	unit_paths = resolve_unit_output_paths(
		templates_out_dir=templates_out_dir,
		unit_id=unit_id,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	if bool(inputs.force_restart):
		unit_dir = unit_paths["unit_dir"]
		if unit_dir.exists():
			shutil.rmtree(unit_dir)
	write_materialized_merged_electrode_ids(
		merged_units_dir=merged_units_dir,
		unit_id=unit_id,
		electrode_ids=merged_electrode_ids,
		metadata_json_path=unit_paths["merged_contributing_electrode_ids_json"],
	)
	_write_per_unit_data_outputs(
		inputs=inputs,
		templates_out_dir=templates_out_dir,
		unit_id=unit_id,
		merged_template=merged_template,
		merged_locs=merged_locs,
		full_template=full_template,
		full_locs=full_locs,
		decision=decision,
		channel_scope_summary=channel_scope_summary,
	)
	LOGGER.info("build_templates unit done: unit_id=%s sources=%d", unit_id, len(source_payloads))
	result = {
		"unit_id": unit_id,
		"status": "built",
		"upsampling": dict(decision) if isinstance(decision, dict) else None,
		"channel_scope": dict(channel_scope_summary),
	}
	del source_payloads
	del materialized
	del merged_template
	del merged_locs
	del full_template
	del full_locs
	gc.collect()
	return result


def _build_templates_unit_with_loader(
	*,
	inputs: TemplatesInputs,
	templates_out_dir: Path,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	unit_id: Any,
	raw_sampling_rate_hz: float,
	payload_loader: Callable[[Any], list[tuple[str, tuple[Any, ...]]]],
) -> dict[str, Any]:
	return _build_templates_unit_from_source_payloads(
		inputs=inputs,
		templates_out_dir=templates_out_dir,
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_channels_templates_dir,
		unit_id=unit_id,
		source_payloads=list(payload_loader(unit_id)),
		raw_sampling_rate_hz=raw_sampling_rate_hz,
	)


def _run_build_templates_unit_job(job: _BuildTemplatesUnitJob) -> dict[str, Any]:
	source_payloads = _load_source_payloads_from_disk(
		templates_out_dir=job.templates_out_dir,
		output_rel_root=job.payload_output_rel_root,
		source_names=job.source_names,
		unit_id=job.unit_id,
	)
	return _build_templates_unit_from_source_payloads(
		inputs=job.inputs,
		templates_out_dir=job.templates_out_dir,
		merged_units_dir=job.merged_units_dir,
		full_channels_templates_dir=job.full_channels_templates_dir,
		unit_id=job.unit_id,
		source_payloads=source_payloads,
		raw_sampling_rate_hz=job.raw_sampling_rate_hz,
	)


def _run_build_templates_unit_jobs_with_processes(
	*,
	jobs: list[_BuildTemplatesUnitJob],
	worker_count: int,
) -> list[dict[str, Any]]:
	results: list[dict[str, Any]] = []
	with concurrent.futures.ProcessPoolExecutor(
		max_workers=max(1, int(worker_count)),
		initializer=install_linux_parent_death_signal,
	) as pool:
		futures = {pool.submit(_run_build_templates_unit_job, job): job.unit_id for job in jobs}
		for future in concurrent.futures.as_completed(futures):
			results.append(future.result())
	return results


def build_templates_phase_from_unit_payloads(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	templates_out_dir: Path,
	source_payloads_by_unit: dict[Any, list[tuple[str, tuple[Any, ...]]]] | None = None,
	unit_ids: list[Any],
	source_names: list[str],
	payload_root: Path | None = None,
	payload_materialization_mode: str = "disk",
	payload_loader: Callable[[Any], list[tuple[str, tuple[Any, ...]]]] | None = None,
) -> dict[str, Any]:
	merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(templates_out_dir=templates_out_dir)
	if bool(inputs.force_restart):
		if merged_units_dir.exists():
			shutil.rmtree(merged_units_dir)
		if full_channels_templates_dir.exists():
			shutil.rmtree(full_channels_templates_dir)
		merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(templates_out_dir=templates_out_dir)

	reused_units, pending_unit_ids = _split_units_by_build_artifact_resume(
		inputs=inputs,
		templates_out_dir=templates_out_dir,
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_channels_templates_dir,
		unit_ids=list(unit_ids),
	)
	if reused_units:
		LOGGER.info(
			"build_templates artifact resume: requested_units=%d reused_units=%d pending_units=%d",
			int(len(unit_ids)),
			int(len(reused_units)),
			int(len(pending_unit_ids)),
		)
	_phase_budget = current_phase_budget("reconstruct", "build_templates")
	worker_count = resolve_inner_worker_count(
		nested_shape=str(getattr(_phase_budget, "nested_shape", "unit_workers") or "unit_workers"),
		phase_cpus_per_task=getattr(_phase_budget, "cpus_per_task", None) if _phase_budget else None,
		yaml_n_jobs_override=int(inputs.n_jobs) if getattr(inputs, "n_jobs", None) is not None else None,
		work_item_count=int(len(pending_unit_ids)) if pending_unit_ids else None,
	) if pending_unit_ids else 0
	payload_output_rel_root = _payload_output_rel_root_from_payload_root(
		templates_out_dir=templates_out_dir,
		payload_root=payload_root,
	)
	can_use_process_workers = bool(payload_output_rel_root and source_names)
	executor_kind = "none" if not pending_unit_ids else "serial"
	if worker_count > 1 and can_use_process_workers:
		executor_kind = "process"
	elif worker_count > 1:
		executor_kind = "thread"
	LOGGER.info(
		"build_templates unit execution start: requested_units=%d worker_count=%d executor=%s payload_materialization_mode=%s",
		len(pending_unit_ids),
		int(worker_count),
		str(executor_kind),
		str(payload_materialization_mode),
	)
	upsampling_decisions_by_unit: dict[Any, dict[str, Any]] = {}
	channel_scope_by_unit: dict[str, dict[str, Any]] = {}
	results: list[dict[str, Any]] = [{"unit_id": unit_id, "status": "reused"} for unit_id in reused_units]
	if pending_unit_ids:
		raw_sampling_rate_hz = read_maxwell_sampling_frequency_hz(
			h5_path=Path(inputs.h5_path),
			stream_id=str(inputs.stream_id),
		)
	else:
		raw_sampling_rate_hz = 0.0
	if pending_unit_ids and (worker_count <= 1 or len(pending_unit_ids) <= 1):
		for unit_id in pending_unit_ids:
			unit_payload_loader = payload_loader
			if unit_payload_loader is None:
				unit_payload_loader = lambda unit, _source_payloads_by_unit=source_payloads_by_unit: list(
					(_source_payloads_by_unit or {}).get(unit, [])
				)
			results.append(
				_build_templates_unit_with_loader(
					inputs=inputs,
					templates_out_dir=templates_out_dir,
					merged_units_dir=merged_units_dir,
					full_channels_templates_dir=full_channels_templates_dir,
					unit_id=unit_id,
					raw_sampling_rate_hz=float(raw_sampling_rate_hz),
					payload_loader=unit_payload_loader,
				)
			)
	elif pending_unit_ids and can_use_process_workers and payload_output_rel_root is not None:
		jobs = [
			_BuildTemplatesUnitJob(
				inputs=inputs,
				templates_out_dir=templates_out_dir,
				merged_units_dir=merged_units_dir,
				full_channels_templates_dir=full_channels_templates_dir,
				unit_id=unit_id,
				source_names=tuple(str(name) for name in source_names),
				payload_output_rel_root=str(payload_output_rel_root),
				raw_sampling_rate_hz=float(raw_sampling_rate_hz),
			)
			for unit_id in pending_unit_ids
		]
		try:
			results = _run_build_templates_unit_jobs_with_processes(jobs=jobs, worker_count=worker_count)
			results = [{"unit_id": unit_id, "status": "reused"} for unit_id in reused_units] + results
		except Exception as exc:
			LOGGER.warning(
				"build_templates process unit workers failed; falling back to thread workers: %s",
				exc,
			)
			unit_payload_loader = lambda unit: _load_source_payloads_from_disk(
				templates_out_dir=templates_out_dir,
				output_rel_root=str(payload_output_rel_root),
				source_names=source_names,
				unit_id=unit,
			)
			with concurrent.futures.ThreadPoolExecutor(max_workers=int(worker_count)) as pool:
				futures = {
					pool.submit(
						_build_templates_unit_with_loader,
						inputs=inputs,
						templates_out_dir=templates_out_dir,
						merged_units_dir=merged_units_dir,
						full_channels_templates_dir=full_channels_templates_dir,
						unit_id=unit_id,
						raw_sampling_rate_hz=float(raw_sampling_rate_hz),
						payload_loader=unit_payload_loader,
					): unit_id
					for unit_id in pending_unit_ids
				}
				for future in concurrent.futures.as_completed(futures):
					results.append(future.result())
	elif pending_unit_ids:
		unit_payload_loader = lambda unit: list((source_payloads_by_unit or {}).get(unit, []))
		with concurrent.futures.ThreadPoolExecutor(max_workers=int(worker_count)) as pool:
			futures = {
				pool.submit(
					_build_templates_unit_with_loader,
					inputs=inputs,
					templates_out_dir=templates_out_dir,
					merged_units_dir=merged_units_dir,
					full_channels_templates_dir=full_channels_templates_dir,
					unit_id=unit_id,
					raw_sampling_rate_hz=float(raw_sampling_rate_hz),
					payload_loader=unit_payload_loader,
				): unit_id
				for unit_id in pending_unit_ids
			}
			for future in concurrent.futures.as_completed(futures):
				results.append(future.result())

	results_by_unit = {str(result.get("unit_id")): result for result in results}
	built_units: list[Any] = []
	reused_built_units: list[Any] = []
	skipped_units: list[Any] = []
	for unit_id in unit_ids:
		result = results_by_unit.get(str(unit_id), {"unit_id": unit_id, "status": "skipped"})
		if result.get("upsampling") is not None:
			upsampling_decisions_by_unit[unit_id] = dict(result.get("upsampling") or {})
		if result.get("channel_scope") is not None:
			channel_scope_by_unit[str(unit_id)] = dict(result.get("channel_scope") or {})
		status = str(result.get("status", "")).strip().lower()
		if status in {"built", "reused"}:
			built_units.append(unit_id)
			if status == "reused":
				reused_built_units.append(unit_id)
		else:
			skipped_units.append(unit_id)
	LOGGER.info(
		"build_templates unit execution complete: requested_units=%d built_units=%d skipped_units=%d worker_count=%d executor=%s",
		len(unit_ids),
		len(built_units),
		len(skipped_units),
		int(worker_count),
		str(executor_kind),
	)

	return {
		"phase": "build_templates",
		"stream_id": str(inputs.stream_id),
		"well_out_dir": str(well_out_dir),
		"templates_out_dir": str(templates_out_dir),
		"payload_root": (None if payload_root is None else str(payload_root)),
		"payload_materialization_mode": str(payload_materialization_mode),
		"source_names": [str(name) for name in source_names],
		"source_count": int(len(source_names)),
		"requested_units": [unit for unit in unit_ids],
		"built_units": [unit for unit in built_units],
		"reused_units": [unit for unit in reused_built_units],
		"skipped_units": [unit for unit in skipped_units],
		"unit_count": int(len(built_units)),
		"unit_workers": int(worker_count),
		"unit_executor": str(executor_kind),
		"upsampling_decisions_by_unit": {str(k): v for k, v in upsampling_decisions_by_unit.items()},
		"channel_scope_by_unit": channel_scope_by_unit,
	}


def build_templates_phase_from_payloads(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	templates_out_dir: Path,
) -> dict[str, Any]:
	payload_root = templates_out_dir / SOURCE_PAYLOADS_CACHE_RELPATH
	if not payload_root.exists():
		raise FileNotFoundError(
			f"Missing materialized source payloads at {payload_root}; run templates.analyzers before templates.build_templates"
		)

	merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(templates_out_dir=templates_out_dir)
	if bool(inputs.force_restart):
		if merged_units_dir.exists():
			shutil.rmtree(merged_units_dir)
		if full_channels_templates_dir.exists():
			shutil.rmtree(full_channels_templates_dir)
		merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(templates_out_dir=templates_out_dir)

	source_dirs = _apply_source_dir_segment_limit(
		_discover_source_dirs(payload_root),
		limit_segments=inputs.limit_segments,
	)
	if not source_dirs:
		raise FileNotFoundError(
			f"No source payload directories found under {payload_root}; run templates.analyzers before templates.build_templates"
		)

	if inputs.unit_ids is not None:
		unit_ids = list(inputs.unit_ids)
	else:
		unit_ids = _discover_unit_ids_from_payloads(source_dirs)
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	unit_ids = _apply_unit_label_filter(inputs, unit_ids, well_out_dir)
	output_rel_root = str(SOURCE_PAYLOADS_CACHE_RELPATH)

	def _payload_loader(unit_id: Any) -> list[tuple[str, tuple[Any, ...]]]:
		loaded: list[tuple[str, tuple[Any, ...]]] = []
		for source_dir in source_dirs:
			payload = load_materialized_source_payload(
				source_payload_unit_dir=resolve_materialized_source_payload_unit_dir(
					templates_out_dir=templates_out_dir,
					output_rel_root=output_rel_root,
					source_name=source_dir.name,
					unit_id=unit_id,
				),
			)
			if payload is None:
				continue
			loaded.append((str(source_dir.name), payload))
		return loaded

	return build_templates_phase_from_unit_payloads(
		inputs=inputs,
		well_out_dir=well_out_dir,
		templates_out_dir=templates_out_dir,
		unit_ids=unit_ids,
		source_names=[str(path.name) for path in source_dirs],
		payload_root=payload_root,
		payload_materialization_mode="disk",
		payload_loader=_payload_loader,
	)
