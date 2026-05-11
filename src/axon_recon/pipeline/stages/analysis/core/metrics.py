"""Pure metric helpers for the analysis stage.

Each helper takes plain dicts (the parsed JSON contents) and returns a
metric value or NaN when the input is missing/empty. No filesystem reads
here — callers do the IO and feed payloads in.
"""

from __future__ import annotations

import math
from typing import Any


def _nan() -> float:
	return float("nan")


def branch_count(branches_payload: dict[str, Any] | None) -> float:
	"""`len(branches.json["branches"])`, or NaN if payload is missing."""
	if not branches_payload:
		return _nan()
	branches = branches_payload.get("branches", None)
	if not isinstance(branches, list):
		return _nan()
	return float(len(branches))


def total_branch_length_um(branches_payload: dict[str, Any] | None) -> float:
	"""Sum of per-branch path lengths.

	Primary: `sum(sum(branch["distances"]) for branch in branches)`.
	Fallback when `distances` is missing/empty for a branch:
	`sum(euclidean distances between successive polyline_xy nodes)`.
	NaN if the payload is missing or the branches list is empty.
	"""
	if not branches_payload:
		return _nan()
	branches = branches_payload.get("branches", None)
	if not isinstance(branches, list) or not branches:
		return _nan()
	total = 0.0
	for branch in branches:
		if not isinstance(branch, dict):
			continue
		total += _branch_path_length_um(branch)
	return float(total)


def _branch_path_length_um(branch: dict[str, Any]) -> float:
	distances = branch.get("distances", None)
	if isinstance(distances, list) and distances:
		try:
			return float(sum(float(d) for d in distances))
		except (TypeError, ValueError):
			pass  # fall through to polyline_xy fallback
	polyline = branch.get("polyline_xy", None)
	if not isinstance(polyline, list) or len(polyline) < 2:
		return 0.0
	total = 0.0
	prev = polyline[0]
	for current in polyline[1:]:
		try:
			dx = float(current[0]) - float(prev[0])
			dy = float(current[1]) - float(prev[1])
		except (TypeError, ValueError, IndexError):
			prev = current
			continue
		total += math.sqrt(dx * dx + dy * dy)
		prev = current
	return float(total)


def template_density(unit_summary_payload: dict[str, Any] | None) -> float:
	"""`unit_reconstruction_summary.json["grid_sort_metrics"]["template_density"]`.

	Already precomputed by recon. NaN if missing.
	"""
	if not unit_summary_payload:
		return _nan()
	gsm = unit_summary_payload.get("grid_sort_metrics", None)
	if not isinstance(gsm, dict):
		return _nan()
	value = gsm.get("template_density", None)
	if value is None:
		return _nan()
	try:
		return float(value)
	except (TypeError, ValueError):
		return _nan()


def recon_density(
	merged_payload: dict[str, Any] | None,
	branches_payload: dict[str, Any] | None,
) -> float:
	"""`len(electrode_ids) / recon_bbox_area_um2`.

	`recon_bbox_area_um2` = `(max_x - min_x) * (max_y - min_y)` over the union
	of all `polyline_xy` points across every branch. NaN when the bbox area
	is 0 or when no usable polyline points are available.
	"""
	if not merged_payload or not branches_payload:
		return _nan()
	electrode_ids = merged_payload.get("electrode_ids", None)
	if not isinstance(electrode_ids, list) or not electrode_ids:
		return _nan()
	branches = branches_payload.get("branches", None)
	if not isinstance(branches, list) or not branches:
		return _nan()
	xs: list[float] = []
	ys: list[float] = []
	for branch in branches:
		if not isinstance(branch, dict):
			continue
		polyline = branch.get("polyline_xy", None)
		if not isinstance(polyline, list):
			continue
		for node in polyline:
			try:
				xs.append(float(node[0]))
				ys.append(float(node[1]))
			except (TypeError, ValueError, IndexError):
				continue
	if not xs or not ys:
		return _nan()
	width = max(xs) - min(xs)
	height = max(ys) - min(ys)
	area = float(width) * float(height)
	if area <= 0.0:
		return _nan()
	return float(len(electrode_ids)) / area


def passthrough_grid_sort_metric(
	unit_summary_payload: dict[str, Any] | None,
	*,
	key: str,
) -> float:
	"""Read a numeric value from grid_sort_metrics, NaN if missing."""
	if not unit_summary_payload:
		return _nan()
	gsm = unit_summary_payload.get("grid_sort_metrics", None)
	if not isinstance(gsm, dict):
		return _nan()
	value = gsm.get(key, None)
	if value is None:
		return _nan()
	try:
		return float(value)
	except (TypeError, ValueError):
		return _nan()


def compute_unit_metrics(
	*,
	unit_summary_payload: dict[str, Any] | None,
	branches_payload: dict[str, Any] | None,
	merged_payload: dict[str, Any] | None,
	templates_payload: dict[str, Any] | None,
) -> dict[str, float]:
	"""Compute the four starter metrics + numeric passthroughs in one pass.

	Returns a dict with: branch_count, total_branch_length_um, template_density,
	recon_density, max_amplitude_uv, max_ptp_uv, max_delay_ms,
	unit_location_x_um, unit_location_y_um.
	NaN for anything that can't be computed.
	"""
	from .recon_io import get_unit_location_xy  # local import to keep this module dep-free

	x_um, y_um = get_unit_location_xy(templates_payload)
	return {
		"branch_count": branch_count(branches_payload),
		"total_branch_length_um": total_branch_length_um(branches_payload),
		"template_density": template_density(unit_summary_payload),
		"recon_density": recon_density(merged_payload, branches_payload),
		"max_amplitude_uv": passthrough_grid_sort_metric(unit_summary_payload, key="max_amplitude"),
		"max_ptp_uv": passthrough_grid_sort_metric(unit_summary_payload, key="max_ptp"),
		"max_delay_ms": passthrough_grid_sort_metric(unit_summary_payload, key="max_delay"),
		"unit_location_x_um": float("nan") if x_um is None else float(x_um),
		"unit_location_y_um": float("nan") if y_um is None else float(y_um),
	}
