from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import as_float_list, as_int_list, as_list, jsonable, read_json


def load_templates_for_unit(
	*,
	unit_id: Any,
	merged_units_dir: Path,
	full_channels_templates_dir: Path,
	use_full_channels_templates: bool,
	require_full_channels_templates: bool,
) -> tuple[Any, Any, float, str]:
	import numpy as np  # type: ignore[import-not-found]

	merged_unit_dir = merged_units_dir / f"unit_{unit_id}"
	full_unit_dir = full_channels_templates_dir / f"unit_{unit_id}"

	merged_tmpl_npy = merged_unit_dir / "merged_contributing_template.npy"
	merged_locs_npy = merged_unit_dir / "merged_contributing_channel_locations.npy"
	merged_meta_json = merged_unit_dir / "merged_contributing_template_meta.json"

	full_tmpl_npy = full_unit_dir / "full_template.npy"
	full_locs_npy = full_unit_dir / "full_channel_locations_xy.npy"
	full_meta_json = full_unit_dir / "full_template_meta.json"

	use_full = bool(use_full_channels_templates)
	if use_full and (not full_tmpl_npy.exists() or not full_locs_npy.exists()):
		if bool(require_full_channels_templates):
			raise FileNotFoundError(
				f"Missing full-channel templates for unit {unit_id}: {full_tmpl_npy} and {full_locs_npy}"
			)
		use_full = False

	if use_full:
		tmpl = np.load(full_tmpl_npy)
		locs = np.load(full_locs_npy)
		selected_source = "full_channels_templates"
		meta_path = full_meta_json
	else:
		if not merged_tmpl_npy.exists() or not merged_locs_npy.exists():
			raise FileNotFoundError(f"Missing merged templates for unit {unit_id}")
		tmpl = np.load(merged_tmpl_npy)
		locs = np.load(merged_locs_npy)
		selected_source = "merged_contributing"
		meta_path = merged_meta_json

	if tmpl.ndim != 2:
		raise ValueError(f"Unexpected template shape for unit {unit_id}: {tmpl.shape}")
	if locs.ndim != 2 or locs.shape[1] < 2:
		raise ValueError(f"Unexpected locations shape for unit {unit_id}: {locs.shape}")

	fs_hz = 10_000.0
	if meta_path.exists():
		try:
			meta = read_json(meta_path)
			if isinstance(meta, dict) and meta.get("sampling_frequency_hz") is not None:
				fs_hz = float(meta.get("sampling_frequency_hz"))
		except Exception:
			fs_hz = 10_000.0

	template_ch_by_t = np.asarray(tmpl).T
	locs_xy = np.asarray(locs)[:, :2]
	return template_ch_by_t, locs_xy, float(fs_hz), selected_source


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

