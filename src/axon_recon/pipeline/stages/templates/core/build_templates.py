from __future__ import annotations

import logging
from pathlib import Path
import shutil
import gc
from typing import Any, Callable

import numpy as np  # type: ignore[import-not-found]

from axon_recon.pipeline.shared.grid_sorting import compute_template_grid_sort_metrics
from axon_recon.pipeline.shared.sampling import read_maxwell_sampling_frequency_hz

from ..io import (
	load_materialized_source_payload,
	resolve_materialized_source_payload_unit_dir,
	resolve_materialized_templates_dirs,
	resolve_unit_output_paths,
	write_json,
	write_materialized_merged_electrode_ids,
	write_materialized_unit_templates,
)

from ..models.inputs import TemplatesInputs
from .merge import materialize_unit_templates_from_sources_with_meta
from .unit_labels import count_labels, filter_unit_ids_by_labels, load_unit_labels_from_spikesorting


LOGGER = logging.getLogger("axon_recon.templates.build_templates")


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
) -> dict[str, Any]:
	paths = resolve_unit_output_paths(
		templates_out_dir=templates_out_dir,
		unit_id=unit_id,
		per_unit_outputs=inputs.per_unit_outputs,
	)
	paths["unit_dir"].mkdir(parents=True, exist_ok=True)
	outputs: dict[str, str] = {}

	merged_locs_path = paths["merged_template_channel_locations_npy"]
	if bool(inputs.per_unit_outputs.merged_template.write_npy):
		paths["merged_template_npy"].parent.mkdir(parents=True, exist_ok=True)
		np.save(paths["merged_template_npy"], np.asarray(merged_template, dtype=float))
		outputs["merged_template_npy"] = str(paths["merged_template_npy"])
		merged_locs_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(merged_locs_path, np.asarray(merged_locs, dtype=float))
		outputs["merged_template_channel_locations_npy"] = str(merged_locs_path)
	else:
		_remove_disabled_output(paths["merged_template_npy"])
		_remove_disabled_output(merged_locs_path)

	full_locs_path = paths["full_template_channel_locations_npy"]
	if bool(inputs.per_unit_outputs.full_template.write_npy):
		paths["full_template_npy"].parent.mkdir(parents=True, exist_ok=True)
		np.save(paths["full_template_npy"], np.asarray(full_template, dtype=float))
		outputs["full_template_npy"] = str(paths["full_template_npy"])
		full_locs_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(full_locs_path, np.asarray(full_locs, dtype=float))
		outputs["full_template_channel_locations_npy"] = str(full_locs_path)
	else:
		_remove_disabled_output(paths["full_template_npy"])
		_remove_disabled_output(full_locs_path)

	scan_locs_path = paths["scan_template_channel_locations_npy"]
	if bool(inputs.per_unit_outputs.scan_template.write_npy):
		paths["scan_template_npy"].parent.mkdir(parents=True, exist_ok=True)
		np.save(paths["scan_template_npy"], np.asarray(full_template, dtype=float))
		outputs["scan_template_npy"] = str(paths["scan_template_npy"])
		scan_locs_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(scan_locs_path, np.asarray(full_locs, dtype=float))
		outputs["scan_template_channel_locations_npy"] = str(scan_locs_path)
	else:
		_remove_disabled_output(paths["scan_template_npy"])
		_remove_disabled_output(scan_locs_path)

	square_locs_path = paths["square_template_channel_locations_npy"]
	if bool(inputs.per_unit_outputs.square_template.write_npy):
		square_template = _build_square_template(
			merged_template,
			padding_mode=str(inputs.per_unit_outputs.square_template.padding_value),
			locations_xy=merged_locs,
		)
		square_locations = _build_square_locations(merged_locs, target_channels=int(square_template.shape[0]))
		paths["square_template_npy"].parent.mkdir(parents=True, exist_ok=True)
		np.save(paths["square_template_npy"], np.asarray(square_template, dtype=float))
		outputs["square_template_npy"] = str(paths["square_template_npy"])
		square_locs_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(square_locs_path, np.asarray(square_locations, dtype=float))
		outputs["square_template_channel_locations_npy"] = str(square_locs_path)
	else:
		_remove_disabled_output(paths["square_template_npy"])
		_remove_disabled_output(square_locs_path)

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
	write_json(paths["unit_summary_json"], unit_summary)
	outputs["unit_summary_json"] = str(paths["unit_summary_json"])
	return unit_summary


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

	raw_sampling_rate_hz = read_maxwell_sampling_frequency_hz(
		h5_path=Path(inputs.h5_path),
		stream_id=str(inputs.stream_id),
	)
	merge_cfg = inputs.phases.build_templates.merge
	upsampling_cfg = inputs.phases.build_templates.execution_upsampling
	upsampling_decisions_by_unit: dict[Any, dict[str, Any]] = {}
	built_units: list[Any] = []
	skipped_units: list[Any] = []

	for unit_id in unit_ids:
		LOGGER.info("build_templates unit start: unit_id=%s", unit_id)
		if payload_loader is not None:
			source_payloads = list(payload_loader(unit_id))
		else:
			source_payloads = list((source_payloads_by_unit or {}).get(unit_id, []))
		if not source_payloads:
			skipped_units.append(unit_id)
			LOGGER.info("build_templates unit skipped: unit_id=%s reason=no_source_payloads", unit_id)
			continue

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
		upsampling_decisions_by_unit[unit_id] = dict(decision)
		if materialized is None:
			skipped_units.append(unit_id)
			LOGGER.info("build_templates unit skipped: unit_id=%s reason=materialization_returned_none", unit_id)
			continue

		merged_template, merged_locs, full_template, full_locs, merged_electrode_ids = materialized
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
		write_materialized_merged_electrode_ids(
			merged_units_dir=merged_units_dir,
			unit_id=unit_id,
			electrode_ids=merged_electrode_ids,
		)
		if bool(inputs.force_restart):
			unit_dir = resolve_unit_output_paths(
				templates_out_dir=templates_out_dir,
				unit_id=unit_id,
				per_unit_outputs=inputs.per_unit_outputs,
			)["unit_dir"]
			if unit_dir.exists():
				shutil.rmtree(unit_dir)
		_write_per_unit_data_outputs(
			inputs=inputs,
			templates_out_dir=templates_out_dir,
			unit_id=unit_id,
			merged_template=merged_template,
			merged_locs=merged_locs,
			full_template=full_template,
			full_locs=full_locs,
			decision=decision,
		)
		built_units.append(unit_id)
		LOGGER.info("build_templates unit done: unit_id=%s sources=%d", unit_id, len(source_payloads))
		# Drop per-unit payloads/templates so the next iteration starts clean.
		del source_payloads
		del materialized
		del merged_template
		del merged_locs
		del full_template
		del full_locs
		gc.collect()

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
		"skipped_units": [unit for unit in skipped_units],
		"unit_count": int(len(built_units)),
		"upsampling_decisions_by_unit": {str(k): v for k, v in upsampling_decisions_by_unit.items()},
	}


def build_templates_phase_from_payloads(
	*,
	inputs: TemplatesInputs,
	well_out_dir: Path,
	templates_out_dir: Path,
) -> dict[str, Any]:
	payload_root = templates_out_dir / Path(str(inputs.phases.per_unit_processing.extract_template_segments.output_rel_root)).expanduser()
	if not payload_root.exists():
		raise FileNotFoundError(
			f"Missing extracted source payloads at {payload_root}; run templates.extract_template_segments first"
		)

	merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(templates_out_dir=templates_out_dir)
	if bool(inputs.force_restart):
		if merged_units_dir.exists():
			shutil.rmtree(merged_units_dir)
		if full_channels_templates_dir.exists():
			shutil.rmtree(full_channels_templates_dir)
		merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(templates_out_dir=templates_out_dir)

	source_dirs = _discover_source_dirs(payload_root)
	if not source_dirs:
		raise FileNotFoundError(
			f"No source payload directories found under {payload_root}; run templates.extract_template_segments first"
		)

	if inputs.unit_ids is not None:
		unit_ids = list(inputs.unit_ids)
	else:
		unit_ids = _discover_unit_ids_from_payloads(source_dirs)
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	unit_ids = _apply_unit_label_filter(inputs, unit_ids, well_out_dir)
	output_rel_root = str(inputs.phases.per_unit_processing.extract_template_segments.output_rel_root)

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
