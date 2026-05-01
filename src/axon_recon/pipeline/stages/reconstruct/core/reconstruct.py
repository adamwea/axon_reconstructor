from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import as_float_list, as_int_list, as_list, jsonable, read_json


def load_templates_for_unit(
	*,
	unit_id: Any,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	template_source: str,
	use_full_channels_templates: bool,
	require_full_channels_templates: bool,
	probe_geometry: Any | None = None,
) -> tuple[Any, Any, Any, Any, float, str]:
	import numpy as np  # type: ignore[import-not-found]
	from axon_recon.pipeline.stages.reconstruct.templates.runner import _build_square_locations
	from axon_recon.pipeline.stages.reconstruct.templates.runner import _build_square_template

	def _project_template_to_target_locations(
		template_c_by_t: np.ndarray,
		source_locs_xy: np.ndarray,
		target_locs_xy: np.ndarray,
		*,
		padding_mode: str = "zero",
	) -> np.ndarray:
		pad = 0.0 if str(padding_mode).strip().lower() != "nan" else float("nan")
		out = np.full((int(target_locs_xy.shape[0]), int(template_c_by_t.shape[1])), pad, dtype=float)
		key_to_target: dict[tuple[float, float], int] = {}
		for ti in range(int(target_locs_xy.shape[0])):
			k = (float(np.round(float(target_locs_xy[ti, 0]), 6)), float(np.round(float(target_locs_xy[ti, 1]), 6)))
			if k not in key_to_target:
				key_to_target[k] = int(ti)
		for si in range(min(int(template_c_by_t.shape[0]), int(source_locs_xy.shape[0]))):
			sloc = np.asarray(source_locs_xy[si, :2], dtype=float)
			if not bool(np.isfinite(sloc).all()):
				continue
			k = (float(np.round(float(sloc[0]), 6)), float(np.round(float(sloc[1]), 6)))
			ti = key_to_target.get(k, None)
			if ti is None:
				d = np.sum((np.asarray(target_locs_xy[:, :2], dtype=float) - sloc[None, :]) ** 2, axis=1)
				ti = int(np.argmin(d))
			out[int(ti), :] = np.asarray(template_c_by_t[si, :], dtype=float)
		return out

	def _infer_full_locations_from_probe(probe: Any | None) -> np.ndarray | None:
		if probe is None:
			return None
		pitch = getattr(probe, "pitch_um", None)
		ax = getattr(probe, "active_area_um_x", None)
		ay = getattr(probe, "active_area_um_y", None)
		try:
			pitch_um = float(pitch) if pitch is not None else None
			area_x = float(ax) if ax is not None else None
			area_y = float(ay) if ay is not None else None
		except Exception:
			return None
		if pitch_um is None or pitch_um <= 0.0 or area_x is None or area_y is None or area_x <= 0.0 or area_y <= 0.0:
			return None
		nx = int(max(1, np.round(area_x / pitch_um)))
		ny = int(max(1, np.round(area_y / pitch_um)))
		x_vals = np.arange(nx, dtype=float) * float(pitch_um)
		y_vals = np.arange(ny, dtype=float) * float(pitch_um)
		return np.asarray([[float(x), float(y)] for y in y_vals for x in x_vals], dtype=float)

	def _coerce_template_to_ch_by_t(template_any: np.ndarray, *, n_channels: int, template_label: str) -> np.ndarray:
		tpl = np.asarray(template_any, dtype=float)
		if tpl.ndim != 2:
			raise ValueError(f"{template_label} must be 2D, got shape={tpl.shape}")
		if int(tpl.shape[0]) == int(n_channels):
			return tpl
		if int(tpl.shape[1]) == int(n_channels):
			return np.asarray(tpl.T, dtype=float)
		raise ValueError(
			f"Cannot orient {template_label} to channel-by-time for unit {unit_id}: "
			f"shape={tpl.shape}, n_channels={n_channels}"
		)

	unit_tokens: list[str] = []
	try:
		uid = int(unit_id)
		unit_tokens = [f"unit_{uid}", f"{uid:04d}", str(uid)]
	except Exception:
		unit_tokens = [f"unit_{unit_id}", str(unit_id)]

	merged_unit_dir = next((merged_units_dir / tok for tok in unit_tokens if (merged_units_dir / tok).exists()), merged_units_dir / unit_tokens[0])
	full_unit_dir = next((full_channels_templates_dir / tok for tok in unit_tokens if (full_channels_templates_dir / tok).exists()), full_channels_templates_dir / unit_tokens[0])

	legacy_merged_tmpl_npy = merged_unit_dir / "merged_contributing_template.npy"
	legacy_merged_locs_npy = merged_unit_dir / "merged_contributing_channel_locations.npy"
	legacy_merged_meta_json = merged_unit_dir / "merged_contributing_template_meta.json"

	v2_merged_tmpl_npy = merged_unit_dir / "merged_template.npy"
	v2_merged_locs_npy = merged_unit_dir / "merged_channel_locations.npy"
	v2_unit_summary_json = merged_unit_dir / "unit_templates_summary.json"

	if v2_merged_tmpl_npy.exists() and v2_merged_locs_npy.exists():
		merged_tmpl_npy = v2_merged_tmpl_npy
		merged_locs_npy = v2_merged_locs_npy
		merged_meta_json = v2_unit_summary_json
		selected_merged_source = "merged_per_unit_output"
	else:
		merged_tmpl_npy = legacy_merged_tmpl_npy
		merged_locs_npy = legacy_merged_locs_npy
		merged_meta_json = legacy_merged_meta_json
		selected_merged_source = "merged_contributing"

	full_tmpl_npy = full_unit_dir / "full_template.npy"
	full_locs_npy = full_unit_dir / "full_channel_locations_xy.npy"
	full_meta_json = full_unit_dir / "full_template_meta.json"

	if not merged_tmpl_npy.exists() or not merged_locs_npy.exists():
		raise FileNotFoundError(f"Missing merged templates for unit {unit_id}")

	merged_tmpl = np.load(merged_tmpl_npy)
	merged_locs = np.load(merged_locs_npy)
	if merged_tmpl.ndim != 2:
		raise ValueError(f"Unexpected merged template shape for unit {unit_id}: {merged_tmpl.shape}")
	if merged_locs.ndim != 2 or merged_locs.shape[1] < 2:
		raise ValueError(f"Unexpected merged locations shape for unit {unit_id}: {merged_locs.shape}")
	merged_locs_xy = np.asarray(merged_locs[:, :2], dtype=float)
	merged_template_ch_by_t = _coerce_template_to_ch_by_t(
		merged_tmpl,
		n_channels=int(merged_locs_xy.shape[0]),
		template_label="merged template",
	)

	source = str(template_source or "square").strip().lower()
	if source not in {"square", "merged", "full", "full_from_merged"}:
		source = "square"

	use_full = bool(use_full_channels_templates)
	if source == "full" and use_full and (not full_tmpl_npy.exists() or not full_locs_npy.exists()):
		if bool(require_full_channels_templates):
			raise FileNotFoundError(
				f"Missing full-channel templates for unit {unit_id}: {full_tmpl_npy} and {full_locs_npy}"
			)
		use_full = False

	gtr_tmpl: np.ndarray
	gtr_locs: np.ndarray
	selected_source: str
	if source == "full":
		if not use_full:
			raise FileNotFoundError(f"template_source=full requested but full templates unavailable for unit {unit_id}")
		full_tmpl = np.load(full_tmpl_npy)
		full_locs = np.load(full_locs_npy)
		if full_tmpl.ndim != 2:
			raise ValueError(f"Unexpected full template shape for unit {unit_id}: {full_tmpl.shape}")
		if full_locs.ndim != 2 or full_locs.shape[1] < 2:
			raise ValueError(f"Unexpected full locations shape for unit {unit_id}: {full_locs.shape}")
		gtr_tmpl = np.asarray(full_tmpl, dtype=float)
		gtr_locs = np.asarray(full_locs[:, :2], dtype=float)
		selected_source = "full_channels_templates"
	elif source == "merged":
		gtr_tmpl = np.asarray(merged_tmpl, dtype=float)
		gtr_locs = np.asarray(merged_locs[:, :2], dtype=float)
		selected_source = selected_merged_source
	elif source == "square":
		merged_c_by_t = np.asarray(merged_template_ch_by_t, dtype=float)
		square_c_by_t = _build_square_template(
			merged_c_by_t,
			padding_mode="zero",
			locations_xy=np.asarray(merged_locs_xy)[:, :2],
		)
		square_locs = _build_square_locations(np.asarray(merged_locs_xy)[:, :2], target_channels=int(square_c_by_t.shape[0]))
		gtr_tmpl = np.asarray(square_c_by_t.T, dtype=float)
		gtr_locs = np.asarray(square_locs[:, :2], dtype=float)
		selected_source = "square_from_merged_per_unit" if selected_merged_source == "merged_per_unit_output" else "square_from_merged"
	elif source == "full_from_merged":
		merged_c_by_t = np.asarray(merged_template_ch_by_t, dtype=float)
		target_full_locs: np.ndarray | None = None
		if full_locs_npy.exists():
			loaded_full_locs = np.load(full_locs_npy)
			if loaded_full_locs.ndim == 2 and int(loaded_full_locs.shape[1]) >= 2:
				target_full_locs = np.asarray(loaded_full_locs[:, :2], dtype=float)
		if target_full_locs is None:
			target_full_locs = _infer_full_locations_from_probe(probe_geometry)
		if target_full_locs is None:
			raise FileNotFoundError(
				f"template_source=full_from_merged requested but no full-channel locations/probe geometry available for unit {unit_id}"
			)
		full_c_by_t = _project_template_to_target_locations(
			template_c_by_t=merged_c_by_t,
			source_locs_xy=np.asarray(merged_locs_xy)[:, :2],
			target_locs_xy=np.asarray(target_full_locs)[:, :2],
			padding_mode="zero",
		)
		gtr_tmpl = np.asarray(full_c_by_t.T, dtype=float)
		gtr_locs = np.asarray(target_full_locs[:, :2], dtype=float)
		selected_source = "full_from_merged_per_unit" if selected_merged_source == "merged_per_unit_output" else "full_from_merged"
	else:
		raise ValueError(f"Unsupported template_source for unit {unit_id}: {source}")

	fs_hz = 10_000.0
	if merged_meta_json.exists():
		try:
			meta = read_json(merged_meta_json)
			if isinstance(meta, dict):
				if meta.get("sampling_frequency_hz") is not None:
					fs_hz = float(meta.get("sampling_frequency_hz"))
				elif meta.get("effective_sampling_rate_hz") is not None:
					fs_hz = float(meta.get("effective_sampling_rate_hz"))
				elif isinstance(meta.get("upsampling"), dict):
					ups = meta.get("upsampling")
					if ups.get("target_hz") is not None:
						fs_hz = float(ups.get("target_hz"))
					elif ups.get("analyzer_hz") is not None:
						fs_hz = float(ups.get("analyzer_hz"))
					elif ups.get("raw_hz") is not None:
						fs_hz = float(ups.get("raw_hz"))
		except Exception:
			fs_hz = 10_000.0

	plot_template_ch_by_t = np.asarray(merged_tmpl, dtype=float).T
	plot_locs_xy = np.asarray(merged_locs, dtype=float)[:, :2]
	gtr_template_ch_by_t = np.asarray(gtr_tmpl, dtype=float).T
	gtr_locs_xy = np.asarray(gtr_locs, dtype=float)[:, :2]
	return (
		plot_template_ch_by_t,
		plot_locs_xy,
		gtr_template_ch_by_t,
		gtr_locs_xy,
		float(fs_hz),
		selected_source,
	)


def compute_raw_branches_payload(*, unit_id: Any, gtr: Any) -> dict[str, Any]:
	branches_out: list[dict[str, Any]] = []
	for bi, branch in enumerate(as_list(getattr(gtr, "branches", None))):
		branch_dict = branch if isinstance(branch, dict) else {}
		branches_out.append(
			{
				"unit_id": jsonable(unit_id),
				"branch_index": int(branch_dict.get("branch_index", bi)),
				"channels": as_int_list(branch_dict.get("channels")),
				"velocity": jsonable(branch_dict.get("velocity")),
				"offset": jsonable(branch_dict.get("offset")),
				"r2": jsonable(branch_dict.get("r2")),
				"pval": jsonable(branch_dict.get("pval")),
				"distances": as_float_list(branch_dict.get("distances")),
				"peak_times": as_float_list(branch_dict.get("peak_times")),
			}
		)
	return {"unit_id": jsonable(unit_id), "branches": branches_out}


def compute_branches_with_polyline(*, unit_id: Any, gtr: Any, locs_xy: Any) -> dict[str, Any]:
	branches_out: list[dict[str, Any]] = []
	for bi, branch in enumerate(as_list(getattr(gtr, "branches", None))):
		branch_dict = branch if isinstance(branch, dict) else {}
		channels = as_int_list(branch_dict.get("channels"))

		polyline_xy: list[list[float]] = []
		try:
			for ch in channels:
				polyline_xy.append([float(locs_xy[ch, 0]), float(locs_xy[ch, 1])])
		except Exception:
			polyline_xy = []

		branches_out.append(
			{
				"branch_index": int(bi),
				"channels": channels,
				"polyline_xy": polyline_xy,
				"velocity": jsonable(branch_dict.get("velocity")),
				"offset": jsonable(branch_dict.get("offset")),
				"r2": jsonable(branch_dict.get("r2")),
				"pval": jsonable(branch_dict.get("pval")),
				"distances": as_float_list(branch_dict.get("distances")),
				"peak_times": as_float_list(branch_dict.get("peak_times")),
			}
		)

	return {"unit_id": jsonable(unit_id), "branches": branches_out}


def _coerce_filter_locations(*, locs_xy: Any, gtr: Any) -> Any:
	import numpy as np  # type: ignore[import-not-found]

	locs_source = locs_xy if locs_xy is not None else getattr(gtr, "locations", None)
	if locs_source is None:
		return np.zeros((0, 2), dtype=float)
	try:
		locs = np.asarray(locs_source, dtype=float)
	except Exception:
		return np.zeros((0, 2), dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		return np.zeros((0, 2), dtype=float)
	return np.asarray(locs[:, :2], dtype=float)


def _sorted_unique_channels(raw_channels: Any) -> list[int]:
	return sorted(set(as_int_list(raw_channels)))



def _summarize_filter_payload(payload: dict[str, Any]) -> dict[str, Any]:
	return {
		"selected_channel_count": int(payload.get("selected_channel_count", 0)),
		"selected_channels": list(payload.get("selected_channels", [])),
		"selected_channels_preview": list(payload.get("selected_channels_preview", [])),
		"filter_params": dict(payload.get("filter_params", {})),
	}


def _build_filter_payload(
	*,
	unit_id: Any,
	gtr: Any,
	locs_xy: Any,
	filter_name: str,
	selected_channels: Any,
	filter_params: dict[str, Any],
	extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
	locs = _coerce_filter_locations(locs_xy=locs_xy, gtr=gtr)
	channels = _sorted_unique_channels(selected_channels)
	payload: dict[str, Any] = {
		"schema_version": 1,
		"unit_id": jsonable(unit_id),
		"filter_name": str(filter_name),
		"init_channel": jsonable(getattr(gtr, "init_channel", None)),
		"n_channels_total": int(locs.shape[0]),
		"selected_channel_count": len(channels),
		"selected_channels": channels,
		"selected_channels_preview": list(channels[:32]),
		"filter_params": {str(key): jsonable(value) for key, value in dict(filter_params).items() if value is not None},
	}
	if extra_fields:
		payload.update({str(key): jsonable(value) for key, value in dict(extra_fields).items()})
	return payload


def compute_detection_filter_payload(*, unit_id: Any, gtr: Any, locs_xy: Any) -> dict[str, Any]:
	amplitudes = as_float_list(getattr(gtr, "amplitudes", None))
	max_amplitude = max(amplitudes) if amplitudes else None
	detection_type = getattr(gtr, "_detection_type", None)
	detect_threshold = getattr(gtr, "_detect_threshold", None)
	effective_threshold = None
	try:
		threshold_value = float(detect_threshold)
		if str(detection_type).strip().lower() == "relative" and max_amplitude is not None:
			effective_threshold = threshold_value * float(max_amplitude)
		else:
			effective_threshold = threshold_value
	except Exception:
		effective_threshold = None
	return _build_filter_payload(
		unit_id=unit_id,
		gtr=gtr,
		locs_xy=locs_xy,
		filter_name="detection",
		selected_channels=getattr(gtr, "_selected_channels_detect", None),
		filter_params={
			"detect_threshold": detect_threshold,
			"detection_type": detection_type,
			"effective_amplitude_threshold": effective_threshold,
			"max_amplitude": max_amplitude,
		},
	)


def compute_kurtosis_filter_payload(*, unit_id: Any, gtr: Any, locs_xy: Any) -> dict[str, Any]:
	return _build_filter_payload(
		unit_id=unit_id,
		gtr=gtr,
		locs_xy=locs_xy,
		filter_name="kurtosis",
		selected_channels=getattr(gtr, "_selected_channels_kurt", None),
		filter_params={
			"kurtosis_threshold": getattr(gtr, "_kurt_threshold", None),
		},
	)


def compute_peak_std_filter_payload(*, unit_id: Any, gtr: Any, locs_xy: Any) -> dict[str, Any]:
	return _build_filter_payload(
		unit_id=unit_id,
		gtr=gtr,
		locs_xy=locs_xy,
		filter_name="peak_std",
		selected_channels=getattr(gtr, "_selected_channels_peakstd", None),
		filter_params={
			"peak_std_threshold": getattr(gtr, "_peak_std_threhsold", None),
			"peak_std_distance_um": getattr(gtr, "_peak_std_distance", None),
		},
	)


def compute_delay_filter_payload(*, unit_id: Any, gtr: Any, locs_xy: Any) -> dict[str, Any]:
	init_channel_peak_frame = None
	peak_times = as_float_list(getattr(gtr, "peak_times", None))
	try:
		init_channel = int(getattr(gtr, "init_channel", -1))
		if 0 <= init_channel < len(peak_times):
			init_channel_peak_frame = float(peak_times[init_channel])
	except Exception:
		init_channel_peak_frame = None
	return _build_filter_payload(
		unit_id=unit_id,
		gtr=gtr,
		locs_xy=locs_xy,
		filter_name="delay",
		selected_channels=getattr(gtr, "_selected_channels_init", None),
		filter_params={
			"init_delay_ms": getattr(gtr, "_init_delay", None),
			"init_delay_frames": getattr(gtr, "_init_frames", None),
			"init_channel_peak_frame": init_channel_peak_frame,
		},
	)


def compute_all_filters_payload(*, unit_id: Any, gtr: Any, locs_xy: Any) -> dict[str, Any]:
	final_payload = _build_filter_payload(
		unit_id=unit_id,
		gtr=gtr,
		locs_xy=locs_xy,
		filter_name="all_filters",
		selected_channels=getattr(gtr, "selected_channels", None),
		filter_params={
			"remove_isolated": getattr(gtr, "_remove_isolated", None),
			"min_selected_points": getattr(gtr, "_min_selected_points", None),
			"compute_velocity": getattr(gtr, "compute_velocity", None),
		},
	)
	return {
		"schema_version": 1,
		"unit_id": jsonable(unit_id),
		"init_channel": jsonable(getattr(gtr, "init_channel", None)),
		"n_channels_total": int(final_payload.get("n_channels_total", 0)),
		"selected_channel_count": int(final_payload.get("selected_channel_count", 0)),
		"selected_channels": list(final_payload.get("selected_channels", [])),
		"selected_channels_preview": list(final_payload.get("selected_channels_preview", [])),
		"compute_velocity": bool(getattr(gtr, "compute_velocity", False)),
		"filters": {
			"detection": _summarize_filter_payload(
				compute_detection_filter_payload(unit_id=unit_id, gtr=gtr, locs_xy=locs_xy)
			),
			"kurtosis": _summarize_filter_payload(
				compute_kurtosis_filter_payload(unit_id=unit_id, gtr=gtr, locs_xy=locs_xy)
			),
			"peak_std": _summarize_filter_payload(
				compute_peak_std_filter_payload(unit_id=unit_id, gtr=gtr, locs_xy=locs_xy)
			),
			"delay": _summarize_filter_payload(
				compute_delay_filter_payload(unit_id=unit_id, gtr=gtr, locs_xy=locs_xy)
			),
			"final": _summarize_filter_payload(final_payload),
		},
	}


def compute_heuristics_payload(*, unit_id: Any, gtr: Any, locs_xy: Any) -> dict[str, Any]:
	heuristics: dict[str, Any] = {
		"init_channel": int(getattr(gtr, "init_channel", 0)),
		"n_channels_total": int(locs_xy.shape[0]),
		"n_selected_channels": int(len(as_list(getattr(gtr, "selected_channels", None)))),
		"selected_channels": as_int_list(getattr(gtr, "selected_channels", None)),
		"node_heuristic": None,
	}
	try:
		node_h = getattr(gtr, "_node_heuristic", None)
		if node_h is not None:
			heuristics["node_heuristic"] = [float(x) for x in list(node_h)]
	except Exception:
		pass
	return {"unit_id": jsonable(unit_id), "heuristics": heuristics}


def compute_gtr_json_payload(*, unit_id: Any, gtr: Any, locs_xy: Any) -> dict[str, Any]:
	payload = {
		"schema_version": 1,
		"unit_id": jsonable(unit_id),
		"init_channel": jsonable(getattr(gtr, "init_channel", None)),
		"selected_channels": as_int_list(getattr(gtr, "selected_channels", None)),
		"node_heuristic": as_float_list(getattr(gtr, "_node_heuristic", None)),
		"branches": compute_branches_with_polyline(unit_id=unit_id, gtr=gtr, locs_xy=locs_xy).get("branches", []),
	}
	return payload
