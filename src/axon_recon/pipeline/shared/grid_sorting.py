from __future__ import annotations

from typing import Any

import numpy as np


GRID_SORT_KEYS: tuple[str, ...] = (
	"unit_id",
	"max_amplitude",
	"max_ptp",
	"max_negative_peak",
	"max_delay",
	"template_density",
)


_GRID_SORT_ALIASES: dict[str, str] = {
	"unit": "unit_id",
	"unitid": "unit_id",
	"id": "unit_id",
	"max_amp": "max_amplitude",
	"amplitude": "max_amplitude",
	"peak_amplitude": "max_amplitude",
	"max_peak_amplitude": "max_amplitude",
	"ptp": "max_ptp",
	"peak_to_peak": "max_ptp",
	"max_peak_to_peak": "max_ptp",
	"max_negative": "max_negative_peak",
	"negative_peak": "max_negative_peak",
	"most_negative_peak": "max_negative_peak",
	"delay": "max_delay",
	"latency": "max_delay",
	"max_latency": "max_delay",
	"density": "template_density",
	"templatedensity": "template_density",
	"template_channel_density": "template_density",
}


def _as_finite_float(value: Any) -> float | None:
	try:
		parsed = float(value)
	except Exception:
		return None
	if not np.isfinite(parsed):
		return None
	return float(parsed)


def normalize_grid_sort_by(value: Any, *, default: str = "unit_id") -> str:
	token = str(value if value is not None else default).strip().lower()
	token = token.replace("-", "_").replace(" ", "_")
	if not token:
		token = str(default).strip().lower().replace("-", "_").replace(" ", "_")
	token = _GRID_SORT_ALIASES.get(token, token)
	if token in GRID_SORT_KEYS:
		return token
	return "unit_id"


def coerce_grid_sort_metrics(raw: Any) -> dict[str, float]:
	if not isinstance(raw, dict):
		return {}
	out: dict[str, float] = {}
	for key in GRID_SORT_KEYS:
		if key == "unit_id":
			continue
		val = _as_finite_float(raw.get(key, None))
		if val is not None:
			out[key] = val
	if "template_density" not in out:
		density = _as_finite_float(raw.get("density", None))
		if density is not None:
			out["template_density"] = density
	return out


def unit_id_sort_key(unit_id: Any) -> tuple[int, int, str]:
	token = str(unit_id).strip()
	try:
		return (0, int(token), token)
	except Exception:
		return (1, 0, token)


def grid_sort_key_for_unit(
	unit_id: Any,
	*,
	sort_by: str,
	metrics_by_unit: dict[str, dict[str, float]] | None,
) -> tuple[float, int, int, str]:
	sort_mode = normalize_grid_sort_by(sort_by)
	unit_sort = unit_id_sort_key(unit_id)
	if sort_mode == "unit_id":
		return (0.0, unit_sort[0], unit_sort[1], unit_sort[2])

	unit_token = str(unit_id).strip()
	metric = None
	if isinstance(metrics_by_unit, dict):
		metric = _as_finite_float((metrics_by_unit.get(unit_token, {}) if isinstance(metrics_by_unit.get(unit_token, {}), dict) else {}).get(sort_mode, None))
	metric_sort = float("inf") if metric is None else -float(metric)
	return (metric_sort, unit_sort[0], unit_sort[1], unit_sort[2])


def compute_template_grid_sort_metrics(
	*,
	template_c_by_t: Any,
	locations_xy: Any,
	sampling_rate_hz: float | None = None,
	probe_pitch_um: float | None = None,
) -> dict[str, float]:
	template = np.asarray(template_c_by_t, dtype=float)
	locs = np.asarray(locations_xy, dtype=float)
	if template.ndim != 2:
		return {}
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		return {}

	if int(template.shape[0]) != int(locs.shape[0]) and int(template.shape[1]) == int(locs.shape[0]):
		template = np.asarray(template.T, dtype=float)
	if int(template.shape[0]) != int(locs.shape[0]):
		return {}

	try:
		ptp_by_channel = np.ptp(template, axis=1)
	except Exception:
		return {}
	if int(ptp_by_channel.shape[0]) == 0:
		return {}

	finite_ptp = np.isfinite(ptp_by_channel)
	selected_idx = np.where(finite_ptp & (ptp_by_channel > 0.0))[0]
	if int(selected_idx.shape[0]) == 0:
		selected_idx = np.where(finite_ptp)[0]
	if int(selected_idx.shape[0]) == 0:
		return {}

	selected_template = np.asarray(template[selected_idx, :], dtype=float)
	max_amplitude = _as_finite_float(np.nanmax(np.abs(selected_template)))
	max_ptp = _as_finite_float(np.nanmax(ptp_by_channel[selected_idx]))
	min_peak = _as_finite_float(np.nanmin(selected_template))
	max_negative_peak = None if min_peak is None else float(max(0.0, -min_peak))

	delay_indices: list[int] = []
	for row in selected_template:
		finite_row = np.isfinite(row)
		if not bool(np.any(finite_row)):
			continue
		row_for_argmin = np.asarray(np.where(finite_row, row, np.inf), dtype=float)
		delay_indices.append(int(np.argmin(row_for_argmin)))
	if delay_indices:
		delay_span_samples = int(max(delay_indices) - min(delay_indices))
	else:
		delay_span_samples = 0
	sampling_rate = _as_finite_float(sampling_rate_hz)
	if sampling_rate is not None and sampling_rate > 0.0:
		max_delay = float((1000.0 * delay_span_samples) / sampling_rate)
	else:
		max_delay = float(delay_span_samples)

	selected_locs = np.asarray(locs[selected_idx, :2], dtype=float)
	finite_xy = np.all(np.isfinite(selected_locs), axis=1)
	selected_locs = selected_locs[finite_xy, :]
	n_density_channels = int(selected_locs.shape[0]) if int(selected_locs.shape[0]) > 0 else int(selected_idx.shape[0])

	template_area = None
	if int(selected_locs.shape[0]) >= 2:
		extent_x = float(np.max(selected_locs[:, 0]) - np.min(selected_locs[:, 0]))
		extent_y = float(np.max(selected_locs[:, 1]) - np.min(selected_locs[:, 1]))
		if extent_x > 0.0 and extent_y > 0.0:
			template_area = float(extent_x * extent_y)
	if template_area is None:
		pitch = _as_finite_float(probe_pitch_um)
		if pitch is not None and pitch > 0.0:
			template_area = float(max(1, n_density_channels)) * float(pitch * pitch)

	if template_area is None or template_area <= 0.0:
		template_density = float(max(1, n_density_channels))
	else:
		template_density = float(max(1, n_density_channels)) / float(template_area)

	metrics: dict[str, float] = {
		"max_delay": float(max_delay),
		"template_density": float(template_density),
	}
	if max_amplitude is not None:
		metrics["max_amplitude"] = float(max_amplitude)
	if max_ptp is not None:
		metrics["max_ptp"] = float(max_ptp)
	if max_negative_peak is not None:
		metrics["max_negative_peak"] = float(max_negative_peak)
	return metrics
