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


def mean_branch_length_um(branches_payload: dict[str, Any] | None) -> float:
	"""Mean of per-branch path lengths within a single unit.

	Same per-branch metric as `total_branch_length_um` divided by the branch
	count. NaN if the payload is missing or the branches list is empty.
	"""
	if not branches_payload:
		return _nan()
	branches = branches_payload.get("branches", None)
	if not isinstance(branches, list) or not branches:
		return _nan()
	per_branch = [
		_branch_path_length_um(branch) for branch in branches if isinstance(branch, dict)
	]
	if not per_branch:
		return _nan()
	return float(sum(per_branch) / len(per_branch))


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
		"mean_branch_length_um": mean_branch_length_um(branches_payload),
		"template_density": template_density(unit_summary_payload),
		"recon_density": recon_density(merged_payload, branches_payload),
		"max_amplitude_uv": passthrough_grid_sort_metric(unit_summary_payload, key="max_amplitude"),
		"max_ptp_uv": passthrough_grid_sort_metric(unit_summary_payload, key="max_ptp"),
		"max_delay_ms": passthrough_grid_sort_metric(unit_summary_payload, key="max_delay"),
		"unit_location_x_um": float("nan") if x_um is None else float(x_um),
		"unit_location_y_um": float("nan") if y_um is None else float(y_um),
	}


WELL_SUMMARY_METRIC_COLUMNS: tuple[str, ...] = (
	"branch_count",
	"total_branch_length_um",
	"mean_branch_length_um",
	"template_density",
	"recon_density",
)


def compute_well_summary(
	units_df: Any,
	identity_cols: dict[str, Any],
) -> dict[str, Any]:
	"""Aggregate a units DataFrame into a one-row well summary dict.

	`units_df` is the per-well `units.parquet` table (or equivalent in-memory
	DataFrame). Returns a flat dict that combines identity columns with:
	  - unit_count_total (all rows)
	  - unit_count_recon_ok (recon_status == "ok")
	  - unit_count_bombcell_good (bombcell_label == "good")
	  - unit_count_bombcell_non_soma_good (bombcell_label == "non_soma_good")
	  - mean_<metric> / median_<metric> for each of the 4 starter metrics,
	    computed over rows with recon_status == "ok" only. NaN when no ok rows.
	"""
	import pandas as pd  # local import keeps tests cheap when only the helper is needed

	row: dict[str, Any] = dict(identity_cols)

	if units_df is None or len(units_df) == 0:
		row["unit_count_total"] = 0
		row["unit_count_recon_ok"] = 0
		row["unit_count_template_ok"] = 0
		row["unit_count_bombcell_good"] = 0
		row["unit_count_bombcell_non_soma_good"] = 0
		for metric in WELL_SUMMARY_METRIC_COLUMNS:
			row[f"mean_{metric}"] = _nan()
			row[f"median_{metric}"] = _nan()
		return row

	row["unit_count_total"] = int(len(units_df))

	recon_status = units_df.get("recon_status", pd.Series(dtype=object))
	ok_mask = recon_status.astype(object) == "ok"
	row["unit_count_recon_ok"] = int(ok_mask.sum())

	template_status = units_df.get("template_status", pd.Series(dtype=object))
	row["unit_count_template_ok"] = int((template_status.astype(object) == "ok").sum())

	bombcell = units_df.get("bombcell_label", pd.Series(dtype=object))
	row["unit_count_bombcell_good"] = int((bombcell.astype(object) == "good").sum())
	row["unit_count_bombcell_non_soma_good"] = int(
		(bombcell.astype(object) == "non_soma_good").sum()
	)

	ok_rows = units_df[ok_mask]
	for metric in WELL_SUMMARY_METRIC_COLUMNS:
		if metric not in ok_rows.columns or len(ok_rows) == 0:
			row[f"mean_{metric}"] = _nan()
			row[f"median_{metric}"] = _nan()
			continue
		series = pd.to_numeric(ok_rows[metric], errors="coerce")
		mean_val = float(series.mean()) if series.notna().any() else _nan()
		median_val = float(series.median()) if series.notna().any() else _nan()
		row[f"mean_{metric}"] = mean_val
		row[f"median_{metric}"] = median_val

	return row
