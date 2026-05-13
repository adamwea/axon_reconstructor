"""Read-only JSON readers for recon outputs.

No pickle reading at MVP — every metric is derived from the per-unit JSON
artifacts that recon already writes to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
	if not path.exists():
		return None
	try:
		with path.open("r", encoding="utf-8") as fh:
			payload = json.load(fh)
	except (json.JSONDecodeError, OSError):
		return None
	return payload if isinstance(payload, dict) else None


def read_unit_reconstruction_summary(unit_dir: Path) -> dict[str, Any] | None:
	"""Read `unit_reconstruction_summary.json`. None if missing or malformed."""
	return _read_json(Path(unit_dir) / "unit_reconstruction_summary.json")


def read_branches(unit_dir: Path) -> dict[str, Any] | None:
	"""Read `branches.json`. None if missing or malformed."""
	return _read_json(Path(unit_dir) / "branches.json")


def read_merged_contributing_electrode_ids(unit_dir: Path) -> dict[str, Any] | None:
	"""Read `merged_contributing_electrode_ids.json`. None if missing."""
	return _read_json(Path(unit_dir) / "merged_contributing_electrode_ids.json")


def read_unit_templates_summary(unit_dir: Path) -> dict[str, Any] | None:
	"""Read `unit_templates_summary.json`. None if missing."""
	return _read_json(Path(unit_dir) / "unit_templates_summary.json")


def get_recon_status(unit_summary: dict[str, Any] | None) -> str:
	"""Return the `status` field from `unit_reconstruction_summary.json`.

	Returns "missing" if the summary was absent or malformed; otherwise the
	string status (typically "ok" or "error"). The dashboard treats anything
	other than "ok" as non-passing.
	"""
	if not unit_summary:
		return "missing"
	status = unit_summary.get("status", None)
	return str(status) if status is not None else "missing"


def get_template_status(templates_payload: dict[str, Any] | None) -> str:
	"""Return a coarse "ok" / "missing" template status for a unit.

	The `build_templates` recon phase writes `unit_templates_summary.json`
	only for units whose merged template build succeeded — file presence is
	the success signal. If callers have already loaded the payload via
	`read_unit_templates_summary`, pass it here; otherwise pass `None` and
	the result is "missing".
	"""
	return "ok" if templates_payload else "missing"


def get_unit_id_from_branches(branches_payload: dict[str, Any] | None) -> int | None:
	if not branches_payload:
		return None
	raw = branches_payload.get("unit_id", None)
	if raw is None:
		return None
	try:
		return int(raw)
	except (TypeError, ValueError):
		return None


def get_electrode_ids(merged_payload: dict[str, Any] | None) -> list[Any]:
	if not merged_payload:
		return []
	value = merged_payload.get("electrode_ids", [])
	return list(value) if isinstance(value, list) else []


def get_unit_location_xy(templates_payload: dict[str, Any] | None) -> tuple[float | None, float | None]:
	if not templates_payload:
		return (None, None)
	unit_location = templates_payload.get("unit_location", None)
	if not isinstance(unit_location, dict):
		return (None, None)
	x = unit_location.get("x_um", None)
	y = unit_location.get("y_um", None)
	try:
		x_val = float(x) if x is not None else None
	except (TypeError, ValueError):
		x_val = None
	try:
		y_val = float(y) if y is not None else None
	except (TypeError, ValueError):
		y_val = None
	return (x_val, y_val)


def iter_unit_dirs(recon_outputs_root: Path) -> list[Path]:
	"""Return sorted unit subdirectories under `<well>/recon_outputs/units/`."""
	units_root = Path(recon_outputs_root) / "units"
	if not units_root.is_dir():
		return []
	entries = sorted(units_root.iterdir())
	return [p for p in entries if p.is_dir()]
