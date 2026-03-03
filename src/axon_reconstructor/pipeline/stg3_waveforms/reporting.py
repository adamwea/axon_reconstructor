"""Waveforms reporting helpers.

This module is a home for JSON/XLSX outputs and other step-level reporting.

Keep waveforms/main.py focused on orchestration; anything that is primarily
"write this artifact" or "load this report" belongs here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _try_channel_location_for_id(*, analyzer: Any, channel_id: Any) -> tuple[Optional[float], Optional[float]]:
	try:
		ch_ids = list(analyzer.get_channel_ids())
	except Exception:
		try:
			ch_ids = list(analyzer.recording.get_channel_ids())
		except Exception:
			return None, None

	try:
		locs = analyzer.get_channel_locations()
	except Exception:
		try:
			locs = analyzer.recording.get_channel_locations()
		except Exception:
			return None, None

	try:
		idx = ch_ids.index(channel_id)
	except Exception:
		return None, None

	try:
		x = float(locs[idx][0])
		y = float(locs[idx][1])
		return x, y
	except Exception:
		return None, None


def _write_best_channel_sources_xlsx(
	*,
	best_channel_sources_xlsx: Path,
	concat_waveforms_dir: Path,
	segment_waveforms_dir: Optional[Path],
	force_restart: bool,
	logger: Any,
) -> None:
	"""Write best-channel provenance across concat vs per-segment analyzers.

	Produces an XLSX that can be consumed by the templates stage to understand:
	- which source has the highest PTP template for each unit
	- which channel id that corresponds to
	- that channel's x/y location (when available)

	Notes:
	- This loads the analyzers from disk, so it can be run standalone as a
	  post-hoc report generator.
	"""

	if best_channel_sources_xlsx.exists() and (not force_restart):
		logger.info("best_channel_sources.xlsx exists; not overwriting: %s", best_channel_sources_xlsx)
		return

	import pandas as pd  # type: ignore[import-not-found]
	import spikeinterface.full as si  # type: ignore[import-not-found]

	from .extraction import _best_ptp_channel_by_unit_from_templates

	rows: list[dict[str, Any]] = []

	# Concat
	concat_analyzer = si.load_sorting_analyzer(concat_waveforms_dir)
	concat_best = _best_ptp_channel_by_unit_from_templates(analyzer=concat_analyzer)
	for unit_id, (ptp_uv, ch_id) in concat_best.items():
		x, y = _try_channel_location_for_id(analyzer=concat_analyzer, channel_id=ch_id)
		rows.append(
			{
				"unit_id": unit_id,
				"source_kind": "concat",
				"source_name": "concat",
				"analyzer_dir": str(concat_waveforms_dir),
				"ptp_uv": float(ptp_uv),
				"channel_id": ch_id,
				"channel_x": x,
				"channel_y": y,
			}
		)

	# Segments
	if segment_waveforms_dir is not None and segment_waveforms_dir.exists():
		for seg_dir in sorted([p for p in segment_waveforms_dir.iterdir() if p.is_dir()]):
			try:
				seg_analyzer = si.load_sorting_analyzer(seg_dir)
			except Exception:
				continue
			try:
				seg_best = _best_ptp_channel_by_unit_from_templates(analyzer=seg_analyzer)
			except Exception:
				continue
			for unit_id, (ptp_uv, ch_id) in seg_best.items():
				x, y = _try_channel_location_for_id(analyzer=seg_analyzer, channel_id=ch_id)
				rows.append(
					{
						"unit_id": unit_id,
						"source_kind": "segment",
						"source_name": str(seg_dir.name),
						"analyzer_dir": str(seg_dir),
						"ptp_uv": float(ptp_uv),
						"channel_id": ch_id,
						"channel_x": x,
						"channel_y": y,
					}
				)

	if not rows:
		df = pd.DataFrame(
			columns=[
				"unit_id",
				"source_kind",
				"source_name",
				"analyzer_dir",
				"ptp_uv",
				"channel_id",
				"channel_x",
				"channel_y",
			]
		)
	else:
		df = pd.DataFrame(rows)

	# Winner per unit (max PTP across all sources)
	winner_df = None
	try:
		if not df.empty and {"unit_id", "ptp_uv"}.issubset(set(df.columns)):
			# idxmax per group can break with all-NaN; guard by filling -inf
			df2 = df.copy()
			df2["ptp_uv"] = pd.to_numeric(df2["ptp_uv"], errors="coerce").fillna(float("-inf"))
			idx = df2.groupby("unit_id")["ptp_uv"].idxmax()
			winner_df = df.loc[idx].copy()
			winner_df = winner_df.rename(
				columns={
					"source_kind": "winner_source_kind",
					"source_name": "winner_source_name",
					"analyzer_dir": "winner_analyzer_dir",
					"ptp_uv": "winner_ptp_uv",
					"channel_id": "winner_channel_id",
					"channel_x": "winner_channel_x",
					"channel_y": "winner_channel_y",
				}
			)
			# Add concat + best-seg columns for convenience
			concat_df = df[df["source_kind"] == "concat"].copy()
			concat_df = concat_df.rename(
				columns={
					"ptp_uv": "concat_ptp_uv",
					"channel_id": "concat_channel_id",
					"channel_x": "concat_channel_x",
					"channel_y": "concat_channel_y",
				}
			)
			concat_df = concat_df[["unit_id", "concat_ptp_uv", "concat_channel_id", "concat_channel_x", "concat_channel_y"]]

			seg_df = df[df["source_kind"] == "segment"].copy()
			if not seg_df.empty:
				seg_df2 = seg_df.copy()
				seg_df2["ptp_uv"] = pd.to_numeric(seg_df2["ptp_uv"], errors="coerce").fillna(float("-inf"))
				idx2 = seg_df2.groupby("unit_id")["ptp_uv"].idxmax()
				best_seg_df = seg_df.loc[idx2].copy()
				best_seg_df = best_seg_df.rename(
					columns={
						"source_name": "best_segment_source_name",
						"analyzer_dir": "best_segment_analyzer_dir",
						"ptp_uv": "best_segment_ptp_uv",
						"channel_id": "best_segment_channel_id",
						"channel_x": "best_segment_channel_x",
						"channel_y": "best_segment_channel_y",
					}
				)
				best_seg_df = best_seg_df[
					[
						"unit_id",
						"best_segment_source_name",
						"best_segment_analyzer_dir",
						"best_segment_ptp_uv",
						"best_segment_channel_id",
						"best_segment_channel_x",
						"best_segment_channel_y",
					]
				]
			else:
				best_seg_df = pd.DataFrame(columns=["unit_id"])

			winner_df = winner_df.merge(concat_df, on="unit_id", how="left")
			winner_df = winner_df.merge(best_seg_df, on="unit_id", how="left")
			try:
				winner_df["segment_beats_concat"] = (
					pd.to_numeric(winner_df.get("best_segment_ptp_uv"), errors="coerce")
					> pd.to_numeric(winner_df.get("concat_ptp_uv"), errors="coerce")
				)
			except Exception:
				pass
	except Exception:
		winner_df = None

	try:
		import xlsxwriter  # type: ignore[import-not-found]

		engine = "xlsxwriter"
	except Exception:
		engine = "openpyxl"

	best_channel_sources_xlsx.parent.mkdir(parents=True, exist_ok=True)
	with pd.ExcelWriter(best_channel_sources_xlsx, engine=engine) as writer:
		if winner_df is not None:
			winner_df.to_excel(writer, sheet_name="best_by_unit", index=False)
		else:
			pd.DataFrame(columns=["unit_id"]).to_excel(writer, sheet_name="best_by_unit", index=False)
		df.to_excel(writer, sheet_name="by_source", index=False)

	logger.info(
		"Wrote best_channel_sources.xlsx -> %s (rows=%d, units=%s)",
		best_channel_sources_xlsx,
		int(len(df)),
		("?" if winner_df is None else int(len(winner_df))),
	)


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
	  <well>/stg3_waveforms_outputs/wf_rejection_log.xlsx
	"""

	wf_rej_xlsx = well_out_dir / "stg3_waveforms_outputs" / "wf_rejection_log.xlsx"
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
	"_write_best_channel_sources_xlsx",
	"_write_wf_rejection_log_xlsx",
]
