"""Pure pandas-mask filter helpers.

No Dash imports here — every filter is a `df -> pd.Series[bool]` function
that the Dash callback layer composes via `&`. Tests can import this module
without bringing in the full app.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd


def mask_all(df: pd.DataFrame) -> pd.Series:
	"""Return an all-True mask aligned with `df.index`."""
	return pd.Series(True, index=df.index)


def filter_recon_status_ok(df: pd.DataFrame, *, on: bool = True) -> pd.Series:
	"""When `on`, keep rows where recon_status == "ok"; otherwise pass-through."""
	if not on or "recon_status" not in df.columns:
		return mask_all(df)
	return df["recon_status"].astype(object) == "ok"


def filter_bombcell_allowlist(df: pd.DataFrame, allowlist: Iterable[Any] | None) -> pd.Series:
	"""Keep rows whose bombcell_label is in `allowlist`.

	`None` in the allowlist matches rows with a missing bombcell_label.
	Empty allowlist drops everything; `allowlist=None` (no setting) passes all.
	"""
	if "bombcell_label" not in df.columns:
		return mask_all(df)
	if allowlist is None:
		return mask_all(df)
	allow = list(allowlist)
	if not allow:
		return pd.Series(False, index=df.index)
	null_passes = any(item is None for item in allow)
	values = [item for item in allow if item is not None]
	col = df["bombcell_label"]
	mask = col.isin(values) if values else pd.Series(False, index=df.index)
	if null_passes:
		mask = mask | col.isna()
	return mask


def filter_min_numeric(df: pd.DataFrame, column: str, threshold: Any) -> pd.Series:
	"""Keep rows where `column >= threshold`. None/NaN row values pass through.

	`threshold is None` (blank UI input) means no filter — return all-True.
	"""
	if threshold is None or column not in df.columns:
		return mask_all(df)
	try:
		thr = float(threshold)
	except (TypeError, ValueError):
		return mask_all(df)
	col = pd.to_numeric(df[column], errors="coerce")
	# NaN passes through (treated as "include" per plan §4).
	return col.isna() | (col >= thr)


def filter_multiselect(df: pd.DataFrame, column: str, selection: Iterable[Any] | None) -> pd.Series:
	"""Keep rows whose value in `column` is in `selection`.

	`selection is None` or empty → no filter (pass-through, mirrors typical
	multi-select UX where nothing-selected means "show all").
	"""
	if column not in df.columns:
		return mask_all(df)
	if selection is None:
		return mask_all(df)
	values = list(selection)
	if not values:
		return mask_all(df)
	return df[column].isin(values)


def filter_numeric_range(df: pd.DataFrame, column: str, *, lo: Any = None, hi: Any = None) -> pd.Series:
	"""Keep rows where `lo <= column <= hi`. None bound means open on that side."""
	if column not in df.columns:
		return mask_all(df)
	col = pd.to_numeric(df[column], errors="coerce")
	mask = pd.Series(True, index=df.index)
	if lo is not None:
		try:
			mask &= col.isna() | (col >= float(lo))
		except (TypeError, ValueError):
			pass
	if hi is not None:
		try:
			mask &= col.isna() | (col <= float(hi))
		except (TypeError, ValueError):
			pass
	return mask


def apply_filter_spec(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
	"""Apply the slice-4 left-rail filter dict and return the filtered DataFrame.

	`spec` keys (all optional):
	  - require_recon_ok: bool                  (default True)
	  - bombcell_allowlist: list[str | None]    (default None = no filter)
	  - min_num_spikes: int | None              (default 0 means no rows dropped)
	  - min_num_branches: int | None            (default 0)
	  - min_recon_quality_score: float | None   (default None)
	  - project: list[str] | None
	  - chip_id: list[str] | None
	  - well_id: list[str] | None
	  - scan_type: list[str] | None
	  - genotype: list[str] | None
	  - media: list[str] | None
	  - plating_density: list | None
	  - treatment: list[str] | None
	  - run_id: list[str] | None
	  - div_lo / div_hi: numeric DIV range bounds
	"""
	mask = mask_all(df)
	mask &= filter_recon_status_ok(df, on=bool(spec.get("require_recon_ok", True)))
	mask &= filter_bombcell_allowlist(df, spec.get("bombcell_allowlist", None))
	mask &= filter_min_numeric(df, "num_spikes", spec.get("min_num_spikes", None))
	mask &= filter_min_numeric(df, "num_branches", spec.get("min_num_branches", None))
	mask &= filter_min_numeric(df, "recon_quality_score", spec.get("min_recon_quality_score", None))
	for column in ("project", "chip_id", "well_id", "scan_type", "genotype", "media", "plating_density", "treatment", "run_id"):
		mask &= filter_multiselect(df, column, spec.get(column, None))
	mask &= filter_numeric_range(df, "DIV", lo=spec.get("div_lo", None), hi=spec.get("div_hi", None))
	return df.loc[mask].reset_index(drop=True)


# ---------- filter + plot spec JSON round-trip (slice 6) ----------


SPEC_SCHEMA_VERSION = "axon_dashboard_spec_v1"


def filter_spec_to_json(
	filter_spec: dict[str, Any],
	*,
	plot_spec: dict[str, Any] | None = None,
) -> str:
	"""Serialize the current filter + plot UI state to JSON for download/replay.

	The payload is sorted, indented, and includes a schema version + a UTC
	timestamp so consumers can detect drift across dashboard revisions.
	"""
	payload = {
		"schema_version": SPEC_SCHEMA_VERSION,
		"written_at": datetime.now(timezone.utc).isoformat(),
		"filters": dict(filter_spec or {}),
		"plot": dict(plot_spec or {}),
	}
	return json.dumps(payload, indent=2, sort_keys=True, default=str)


def filter_spec_from_json(payload: str | bytes) -> dict[str, Any]:
	"""Parse a previously-serialized filter+plot spec.

	Returns `{"filters": ..., "plot": ...}` (always with both keys). Drops the
	schema_version / written_at metadata so the result plugs straight into
	`apply_filter_spec` + the plot callbacks.
	"""
	if isinstance(payload, bytes):
		payload = payload.decode("utf-8")
	parsed = json.loads(payload)
	if not isinstance(parsed, dict):
		raise ValueError("Filter spec JSON must decode to an object/mapping.")
	filters = parsed.get("filters", {})
	plot = parsed.get("plot", {})
	return {
		"filters": dict(filters) if isinstance(filters, dict) else {},
		"plot": dict(plot) if isinstance(plot, dict) else {},
	}
