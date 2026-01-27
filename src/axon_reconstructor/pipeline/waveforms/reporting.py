"""Waveforms reporting helpers.

This module is a home for JSON/XLSX outputs and other step-level reporting.

Keep waveforms/main.py focused on orchestration; anything that is primarily
"write this artifact" or "load this report" belongs here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _write_wf_rejection_log_xlsx(
	*,
	wf_rejection_log_xlsx: Path,
	rows: list[dict[str, Any]],
	force_restart: bool,
	logger: Any,
) -> None:
	"""Write per-spike waveform rejection log.

	This tracks spikes removed during waveforms extraction filtering (epoch/out-of-epoch
	removals and edge/window violations). The output is designed to be joinable to the
	waveforms analyzers used by footprinting.
	"""

	if wf_rejection_log_xlsx.exists() and (not force_restart):
		logger.info("wf_rejection_log.xlsx exists; not overwriting: %s", wf_rejection_log_xlsx)
		return

	import pandas as pd  # type: ignore[import-not-found]

	if not rows:
		df = pd.DataFrame(
			columns=[
				"scope",
				"source_name",
				"segment_index",
				"rec_name",
				"unit_id",
				"spike_sample_local",
				"spike_sample_concat",
				"spike_time_s",
				"reason",
				"stream_id",
				"sorter",
				"h5_path",
				"fs_hz",
				"ms_before",
				"ms_after",
				"pre_samples",
				"post_samples",
			]
		)
	else:
		df = pd.DataFrame(rows)

	preferred_cols = [
		"scope",
		"source_name",
		"segment_index",
		"rec_name",
		"unit_id",
		"spike_sample_local",
		"spike_sample_concat",
		"spike_time_s",
		"reason",
		"stream_id",
		"sorter",
		"h5_path",
		"fs_hz",
		"ms_before",
		"ms_after",
		"pre_samples",
		"post_samples",
	]
	cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
	df = df.loc[:, cols]

	summary_rows: list[dict[str, Any]] = []
	try:
		summary_rows.append({"metric": "n_rows", "value": int(len(df))})
		for key, label in [
			("scope", "by_scope"),
			("reason", "by_reason"),
			("source_name", "by_source"),
		]:
			if key in df.columns:
				vc = df[key].value_counts(dropna=False)
				for k, v in vc.items():
					summary_rows.append({"metric": f"{label}:{k}", "value": int(v)})
	except Exception:
		pass

	summary_df = pd.DataFrame(summary_rows)

	unit_counts_df = None
	try:
		if not df.empty and {"source_name", "unit_id", "reason"}.issubset(set(df.columns)):
			unit_counts_df = (
				df.groupby(["scope", "source_name", "segment_index", "rec_name", "unit_id", "reason"], dropna=False)
				.size()
				.reset_index(name="n_rejected_spikes")
			)
	except Exception:
		unit_counts_df = None

	max_rows_per_sheet = 1_000_000
	try:
		import xlsxwriter  # type: ignore[import-not-found]

		engine = "xlsxwriter"
	except Exception:
		engine = "openpyxl"

	wf_rejection_log_xlsx.parent.mkdir(parents=True, exist_ok=True)
	with pd.ExcelWriter(wf_rejection_log_xlsx, engine=engine) as writer:
		summary_df.to_excel(writer, sheet_name="summary", index=False)
		if unit_counts_df is not None:
			unit_counts_df.to_excel(writer, sheet_name="unit_counts", index=False)

		if df.empty:
			df.to_excel(writer, sheet_name="rejections_000", index=False)
		else:
			for i0 in range(0, len(df), max_rows_per_sheet):
				chunk = df.iloc[i0 : i0 + max_rows_per_sheet]
				sheet = f"rejections_{i0 // max_rows_per_sheet:03d}"
				chunk.to_excel(writer, sheet_name=sheet, index=False)

	logger.info("Wrote wf_rejection_log.xlsx -> %s (rows=%d)", wf_rejection_log_xlsx, int(len(df)))


def _load_wf_rejection_log_unit_counts(
	*,
	well_out_dir: Path,
	logger: Any,
) -> tuple[Optional[list[dict[str, Any]]], Optional[Path]]:
	"""Best-effort loader for waveforms-stage per-spike rejection counts.

	Reads the lightweight `unit_counts` sheet from:
	  <well>/waveforms_outputs/wf_rejection_log.xlsx
	"""

	wf_rej_xlsx = well_out_dir / "waveforms_outputs" / "wf_rejection_log.xlsx"
	if not wf_rej_xlsx.exists():
		return None, None

	try:
		import pandas as pd  # type: ignore[import-not-found]

		df = pd.read_excel(wf_rej_xlsx, sheet_name="unit_counts")
		if df is None or df.empty:
			return [], wf_rej_xlsx

		rows: list[dict[str, Any]] = []
		for _, r in df.iterrows():
			try:
				rows.append({k: (v.item() if hasattr(v, "item") else v) for k, v in r.to_dict().items()})
			except Exception:
				continue
		return rows, wf_rej_xlsx
	except Exception as e:
		logger.warning("Failed reading wf_rejection_log.xlsx unit_counts: %s", e)
		return None, wf_rej_xlsx


__all__ = [
	"_load_wf_rejection_log_unit_counts",
	"_write_wf_rejection_log_xlsx",
]
