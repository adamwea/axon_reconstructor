"""Load every per-well manifest and concat its tables into a single DataFrame per table name.

Read-only; tolerates partially-written wells (a manifest without a `tables`
entry contributes no rows but still survives the load).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


_IDENTITY_COLUMNS_FROM_MANIFEST: tuple[str, ...] = (
	"project",
	"recording_date",
	"chip_id",
	"scan_type",
	"run_id",
	"well_id",
	"dataset_id",
	"DIV",
)
_WELL_ATTRIBUTE_FIELDS: tuple[str, ...] = ("genotype", "media", "plating_density")


def _read_manifest(path: Path) -> dict[str, Any] | None:
	try:
		with Path(path).open("r", encoding="utf-8") as fh:
			payload = json.load(fh)
	except (OSError, json.JSONDecodeError):
		return None
	return payload if isinstance(payload, dict) else None


def _identity_overrides_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
	overrides = {col: manifest.get(col) for col in _IDENTITY_COLUMNS_FROM_MANIFEST}
	well_attrs = manifest.get("well_attributes", {}) or {}
	if isinstance(well_attrs, dict):
		for key in _WELL_ATTRIBUTE_FIELDS:
			overrides[key] = well_attrs.get(key, overrides.get(key, None))
	return overrides


def _stamp_identity(df: pd.DataFrame, overrides: dict[str, Any]) -> pd.DataFrame:
	"""Override or insert identity columns from the manifest.

	Defense in depth: if the parquet was written against an older schema and
	is missing identity columns, we patch them from the manifest before
	concat.
	"""
	for column, value in overrides.items():
		if value is None:
			continue
		df[column] = value
	return df


def load_all(manifest_paths: list[Path] | tuple[Path, ...]) -> dict[str, pd.DataFrame]:
	"""Load every manifest's tables and concat them by table name.

	Returns a dict like `{"units": <units_df>, "well_summary": <ws_df>}`.
	Table names with zero rows across all manifests are still present with an
	empty DataFrame so callers can rely on the keys.
	"""
	per_table: dict[str, list[pd.DataFrame]] = {}

	for manifest_path in manifest_paths:
		manifest = _read_manifest(Path(manifest_path))
		if manifest is None:
			continue
		tables = manifest.get("tables", {})
		if not isinstance(tables, dict) or not tables:
			continue
		manifest_dir = Path(manifest_path).resolve().parent
		identity_overrides = _identity_overrides_from_manifest(manifest)
		for table_name, table_relpath in tables.items():
			if not isinstance(table_relpath, str) or not table_relpath:
				continue
			table_path = (manifest_dir / table_relpath).resolve()
			if not table_path.is_file():
				continue
			try:
				df = pd.read_parquet(table_path)
			except (OSError, ValueError):
				continue
			if df is None or len(df) == 0:
				continue
			df = _stamp_identity(df, identity_overrides)
			per_table.setdefault(str(table_name), []).append(df)

	out: dict[str, pd.DataFrame] = {}
	for name, frames in per_table.items():
		out[name] = pd.concat(frames, ignore_index=True, sort=False)
	# Ensure both expected tables exist (empty if not found anywhere).
	for expected in ("units", "well_summary"):
		out.setdefault(expected, pd.DataFrame())
	return out
