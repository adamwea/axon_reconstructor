from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any

import numpy as np

from axon_recon.pipeline.shared.sampling import (
	read_maxwell_sampling_frequency_hz,
	rates_match_hz,
	upsample_channels_by_time,
)

from ..models.inputs import TimeUpsampleConfig


LOGGER = logging.getLogger("axon_recon.templates")


def _unpack_source_payload(
	payload: tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int, float | None]:
	if len(payload) < 5:
		raise ValueError(f"Unexpected source payload length: {len(payload)}")
	template = np.asarray(payload[0], dtype=float)
	locations = np.asarray(payload[1], dtype=float)
	electrode_ids = payload[2]
	channel_ids = payload[3]
	waveform_count = int(payload[4])
	sampling_rate_hz: float | None = None
	if len(payload) >= 6 and payload[5] is not None:
		try:
			rate = float(payload[5])
			if np.isfinite(rate) and rate > 0.0:
				sampling_rate_hz = rate
		except Exception:
			sampling_rate_hz = None
	return template, locations, electrode_ids, channel_ids, waveform_count, sampling_rate_hz


def normalize_merge_method(method: str) -> str:
	m = str(method or "mean_all_waveforms").strip().lower()
	if m in {"average_all_wfs", "mean_waveforms", "mean_all_waveforms", "mean"}:
		return "mean_all_waveforms"
	if m in {"weighted_average", "weighted_by_channel_waveform_count", "weighted"}:
		return "weighted_by_channel_waveform_count"
	return "mean_all_waveforms"


def normalize_overlap_priorities(overlap_match_priority: tuple[str, ...]) -> tuple[str, ...]:
	seen: list[str] = []
	for item in overlap_match_priority:
		t = str(item or "").strip().lower()
		if t not in {"electrode_id", "channel_id", "location"}:
			continue
		if t not in seen:
			seen.append(t)
	if not seen:
		return ("electrode_id", "channel_id", "location")
	return tuple(seen)


def _robust_baseline_pre_negative_peak(wf: np.ndarray) -> np.ndarray:
	w = np.asarray(wf, dtype=float).reshape(-1)
	n = int(w.shape[0])
	if n <= 2:
		return w
	peak_idx = int(np.argmin(w))
	win_size = max(5, int(round(0.10 * n)))
	guard = max(2, int(round(0.05 * n)))
	end = max(0, peak_idx - guard)
	start = max(0, end - win_size)
	pre = w[start:end]
	if int(pre.shape[0]) < 3:
		pre = w[max(0, n - win_size):n]
	if int(pre.shape[0]) == 0:
		return w
	baseline = float(np.median(pre))
	return w - baseline


def _center_waveform(wf: np.ndarray, centering_method: str) -> np.ndarray:
	m = str(centering_method or "pre_peak_robust_baseline").strip().lower()
	if m == "pre_peak_robust_baseline":
		return _robust_baseline_pre_negative_peak(wf)
	return np.asarray(wf, dtype=float)


def _resample_to_len(signal: np.ndarray, target_len: int) -> np.ndarray:
	if int(signal.shape[0]) == int(target_len):
		return signal.astype(float, copy=False)
	if int(signal.shape[0]) <= 1 or int(target_len) <= 1:
		return np.resize(signal.astype(float, copy=False), (int(target_len),))
	x_old = np.linspace(0.0, 1.0, int(signal.shape[0]), dtype=float)
	x_new = np.linspace(0.0, 1.0, int(target_len), dtype=float)
	return np.interp(x_new, x_old, signal.astype(float, copy=False))


def merge_sources_per_channel(
	sources: list[tuple[Any, ...]],
	*,
	enable_merge: bool,
	merge_method: str,
	centering_method: str,
	max_waveforms_per_source_channel: int | None,
	overlap_match_priority: tuple[str, ...],
	location_tolerance_um: float,
) -> tuple[np.ndarray, np.ndarray, list[Any] | None]:
	if not sources:
		raise ValueError("No template sources to merge")

	merge_method_norm = normalize_merge_method(merge_method)
	match_priority = normalize_overlap_priorities(overlap_match_priority)
	loc_tol = max(1e-6, float(location_tolerance_um))
	max_wf: int | None
	if max_waveforms_per_source_channel is None:
		max_wf = None
	else:
		parsed = int(max_waveforms_per_source_channel)
		max_wf = None if parsed <= 0 else parsed
	target_t = int(sources[0][0].shape[1])
	bucket_waveforms: dict[str, np.ndarray] = {}
	bucket_weights: dict[str, float] = {}
	bucket_locations: dict[str, np.ndarray] = {}
	bucket_seen: set[str] = set()

	def _location_key(xy: np.ndarray) -> str:
		qx = int(np.rint(float(xy[0]) / loc_tol))
		qy = int(np.rint(float(xy[1]) / loc_tol))
		return f"loc:{qx}:{qy}"

	def _channel_keys(*, electrode_id: Any, channel_id: Any, loc_xy: np.ndarray) -> list[str]:
		keys: list[str] = []
		for priority in match_priority:
			if priority == "electrode_id" and electrode_id is not None:
				keys.append(f"eid:{electrode_id}")
			elif priority == "channel_id" and channel_id is not None:
				keys.append(f"cid:{channel_id}")
			elif priority == "location":
				keys.append(_location_key(loc_xy))
		return keys

	for payload in sources:
		t_ch_by_t, locs, electrode_ids, channel_ids, source_waveform_count, _ = _unpack_source_payload(payload)
		for ch in range(int(t_ch_by_t.shape[0])):
			eid = (None if electrode_ids is None else electrode_ids[ch])
			cid = (None if channel_ids is None else channel_ids[ch])
			xy = np.asarray(locs[ch, :2], dtype=float)
			keys = _channel_keys(electrode_id=eid, channel_id=cid, loc_xy=xy)

			canonical_key = keys[0] if keys else _location_key(xy)
			for k in keys:
				if k in bucket_seen:
					canonical_key = k
					break

			wave = _resample_to_len(np.asarray(t_ch_by_t[ch, :], dtype=float), target_t)
			wave = _center_waveform(wave, centering_method)
			weight = 1.0
			if merge_method_norm == "weighted_by_channel_waveform_count":
				source_count = max(1, int(source_waveform_count))
				weight = float(source_count if max_wf is None else min(max_wf, source_count))

			if canonical_key in bucket_waveforms:
				if bool(enable_merge):
					bucket_waveforms[canonical_key] = bucket_waveforms[canonical_key] + (wave * weight)
					bucket_weights[canonical_key] = float(bucket_weights[canonical_key]) + float(weight)
				# When merge is disabled, keep first source contribution and ignore overlaps.
			else:
				bucket_waveforms[canonical_key] = wave * float(weight)
				bucket_weights[canonical_key] = float(weight)
				bucket_locations[canonical_key] = xy
				bucket_seen.add(canonical_key)

	keys = sorted(bucket_waveforms.keys())
	merged_waves = []
	merged_locs = []
	merged_electrode_ids: list[Any] = []
	for key in keys:
		den = max(1e-12, float(bucket_weights[key]))
		merged_waves.append(bucket_waveforms[key] / den)
		merged_locs.append(bucket_locations[key])
		if str(key).startswith("eid:"):
			merged_electrode_ids.append(str(key).split(":", 1)[1])
		else:
			merged_electrode_ids.append(None)

	if all(eid is None for eid in merged_electrode_ids):
		return np.vstack(merged_waves), np.asarray(merged_locs, dtype=float), None
	return np.vstack(merged_waves), np.asarray(merged_locs, dtype=float), merged_electrode_ids


def materialize_unit_templates_from_sources(
	*,
	source_payloads: list[tuple[str, tuple[Any, ...]]],
	enable_merge: bool,
	merge_method: str,
	centering_method: str,
	max_waveforms_per_source_channel: int | None,
	overlap_match_priority: tuple[str, ...],
	location_tolerance_um: float,
	execution_upsampling: TimeUpsampleConfig | None = None,
	raw_sampling_rate_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any] | None] | None:
	"""Build merged + full templates for one unit from named source payloads."""
	materialized, _ = materialize_unit_templates_from_sources_with_meta(
		source_payloads=source_payloads,
		enable_merge=enable_merge,
		merge_method=merge_method,
		centering_method=centering_method,
		max_waveforms_per_source_channel=max_waveforms_per_source_channel,
		overlap_match_priority=overlap_match_priority,
		location_tolerance_um=location_tolerance_um,
		execution_upsampling=execution_upsampling,
		raw_sampling_rate_hz=raw_sampling_rate_hz,
	)
	return materialized


def materialize_unit_templates_from_sources_with_meta(
	*,
	source_payloads: list[tuple[str, tuple[Any, ...]]],
	enable_merge: bool,
	merge_method: str,
	centering_method: str,
	max_waveforms_per_source_channel: int | None,
	overlap_match_priority: tuple[str, ...],
	location_tolerance_um: float,
	execution_upsampling: TimeUpsampleConfig | None = None,
	raw_sampling_rate_hz: float | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any] | None] | None, dict[str, Any]]:
	"""Build merged + full templates plus upsampling decision metadata."""
	if not source_payloads:
		return None, {
			"enabled": bool(execution_upsampling.enabled) if execution_upsampling is not None else False,
			"applied": False,
			"skip_reason": "no_sources",
		}
	if execution_upsampling is None:
		execution_upsampling = TimeUpsampleConfig()

	decision: dict[str, Any] = {
		"enabled": bool(execution_upsampling.enabled),
		"method": str(execution_upsampling.method),
		"factor": int(max(1, int(execution_upsampling.factor))),
		"raw_hz": (None if raw_sampling_rate_hz is None else float(raw_sampling_rate_hz)),
		"analyzer_hz": None,
		"target_hz": None,
		"applied": False,
		"skip_reason": None,
	}

	upsample_enabled = bool(execution_upsampling.enabled) and int(execution_upsampling.factor) > 1
	prepared_payloads = list(source_payloads)
	if upsample_enabled:
		raw_hz = None
		if raw_sampling_rate_hz is not None:
			try:
				raw_hz = float(raw_sampling_rate_hz)
			except Exception:
				raw_hz = None

		analyzer_rates: list[float] = []
		for _, payload in prepared_payloads:
			_, _, _, _, _, analyzer_hz = _unpack_source_payload(payload)
			if analyzer_hz is None:
				LOGGER.warning("Skipping templates upsampling: missing analyzer sampling rate in one or more sources")
				decision["skip_reason"] = "missing_analyzer_rate"
				analyzer_rates = []
				break
			analyzer_rates.append(float(analyzer_hz))

		if analyzer_rates:
			base_hz = float(analyzer_rates[0])
			decision["analyzer_hz"] = float(base_hz)
			if any(not rates_match_hz(raw_hz=base_hz, analyzer_hz=hz, atol_hz=float(execution_upsampling.mismatch_tolerance_hz)) for hz in analyzer_rates[1:]):
				LOGGER.warning("Skipping templates upsampling: analyzer sampling rates disagree across sources")
				decision["skip_reason"] = "analyzer_rates_disagree"
			elif raw_hz is None:
				fallback_hz = execution_upsampling.raw_rate_fallback_hz
				if fallback_hz is not None and float(fallback_hz) > 0.0:
					raw_hz = float(fallback_hz)
					decision["raw_hz"] = float(raw_hz)
					decision["raw_rate_source"] = "fallback"
				else:
					LOGGER.warning("Skipping templates upsampling: raw recording sampling rate unavailable")
					decision["skip_reason"] = "raw_rate_unavailable"
			if raw_hz is not None and not rates_match_hz(raw_hz=raw_hz, analyzer_hz=base_hz, atol_hz=float(execution_upsampling.mismatch_tolerance_hz)):
				LOGGER.warning(
					"Skipping templates upsampling: raw sampling rate (%.3f Hz) != analyzer sampling rate (%.3f Hz)",
					float(raw_hz),
					float(base_hz),
				)
				decision["skip_reason"] = "raw_analyzer_mismatch"
			elif raw_hz is not None:
				target_hz = float(base_hz) * float(int(execution_upsampling.factor))
				decision["target_hz"] = float(target_hz)
				if target_hz <= float(base_hz):
					LOGGER.info(
						"Skipping templates upsampling: redundant target_hz=%.3f source_hz=%.3f",
						float(target_hz),
						float(base_hz),
					)
					decision["skip_reason"] = "redundant"
				else:
					resampled_payloads: list[tuple[str, tuple[Any, ...]]] = []
					for src_name, payload in prepared_payloads:
						t_ch_by_t, locs_xy, electrode_ids, channel_ids, waveform_count, analyzer_hz = _unpack_source_payload(payload)
						if analyzer_hz is None:
							continue
						upsampled = upsample_channels_by_time(
							template_c_by_t=t_ch_by_t,
							source_hz=float(analyzer_hz),
							target_hz=float(target_hz),
							method=str(execution_upsampling.method),
						)
						resampled_payloads.append(
							(
								src_name,
								(upsampled, locs_xy, electrode_ids, channel_ids, waveform_count, float(target_hz)),
							)
						)
					if len(resampled_payloads) == len(prepared_payloads):
						prepared_payloads = resampled_payloads
						decision["applied"] = True
						decision["skip_reason"] = None
						LOGGER.info(
							"Applied templates upsampling before merge: source_hz=%.3f target_hz=%.3f factor=%d method=%s",
							float(base_hz),
							float(target_hz),
							int(execution_upsampling.factor),
							str(execution_upsampling.method),
						)
	else:
		decision["skip_reason"] = ("disabled" if not bool(execution_upsampling.enabled) else "factor_le_1")

	payloads = [payload for _, payload in prepared_payloads]
	merged_template, merged_locs, merged_electrode_ids = merge_sources_per_channel(
		payloads,
		enable_merge=bool(enable_merge),
		merge_method=merge_method,
		centering_method=centering_method,
		max_waveforms_per_source_channel=max_waveforms_per_source_channel,
		overlap_match_priority=overlap_match_priority,
		location_tolerance_um=location_tolerance_um,
	)

	full_template = merged_template
	full_locs = merged_locs
	for src_name, payload in prepared_payloads:
		if str(src_name) == "concat":
			full_template = payload[0]
			full_locs = payload[1]
			break

	return (merged_template, merged_locs, full_template, full_locs, merged_electrode_ids), decision


def materialize_unit_templates_by_unit(
	*,
	analyzers: list[tuple[str, Any]],
	unit_ids: list[Any],
	payload_builder: Callable[..., tuple[Any, ...] | None],
	enable_merge: bool,
	merge_method: str,
	centering_method: str,
	max_waveforms_per_source_channel: int | None,
	overlap_match_priority: tuple[str, ...],
	location_tolerance_um: float,
	execution_upsampling: TimeUpsampleConfig | None = None,
	raw_sampling_rate_hz: float | None = None,
) -> dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any] | None]]:
	"""Orchestrate per-unit merge materialization from extracted analyzer payloads."""
	results: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any] | None]] = {}
	for uid in unit_ids:
		source_payloads: list[tuple[str, tuple[Any, ...]]] = []
		for src_name, analyzer in analyzers:
			payload = payload_builder(analyzer=analyzer, unit_id=uid)
			if payload is None:
				continue
			source_payloads.append((str(src_name), payload))

		materialized = materialize_unit_templates_from_sources(
			source_payloads=source_payloads,
			enable_merge=bool(enable_merge),
			merge_method=merge_method,
			centering_method=centering_method,
			max_waveforms_per_source_channel=max_waveforms_per_source_channel,
			overlap_match_priority=overlap_match_priority,
			location_tolerance_um=location_tolerance_um,
			execution_upsampling=execution_upsampling,
			raw_sampling_rate_hz=raw_sampling_rate_hz,
		)
		if materialized is not None:
			results[uid] = materialized

	return results


def materialize_unit_templates_by_unit_with_meta(
	*,
	analyzers: list[tuple[str, Any]],
	unit_ids: list[Any],
	payload_builder: Callable[..., tuple[Any, ...] | None],
	enable_merge: bool,
	merge_method: str,
	centering_method: str,
	max_waveforms_per_source_channel: int | None,
	overlap_match_priority: tuple[str, ...],
	location_tolerance_um: float,
	execution_upsampling: TimeUpsampleConfig | None = None,
	raw_sampling_rate_hz: float | None = None,
) -> tuple[dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any] | None]], dict[Any, dict[str, Any]]]:
	results: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any] | None]] = {}
	decisions: dict[Any, dict[str, Any]] = {}
	total_units = int(len(unit_ids))
	for idx, uid in enumerate(unit_ids, start=1):
		LOGGER.info("Templates materialization unit start: %d/%d unit_id=%s", idx, total_units, uid)
		source_payloads: list[tuple[str, tuple[Any, ...]]] = []
		for src_name, analyzer in analyzers:
			payload = payload_builder(analyzer=analyzer, unit_id=uid)
			if payload is None:
				continue
			source_payloads.append((str(src_name), payload))

		materialized, decision = materialize_unit_templates_from_sources_with_meta(
			source_payloads=source_payloads,
			enable_merge=bool(enable_merge),
			merge_method=merge_method,
			centering_method=centering_method,
			max_waveforms_per_source_channel=max_waveforms_per_source_channel,
			overlap_match_priority=overlap_match_priority,
			location_tolerance_um=location_tolerance_um,
			execution_upsampling=execution_upsampling,
			raw_sampling_rate_hz=raw_sampling_rate_hz,
		)
		decisions[uid] = decision
		if materialized is not None:
			results[uid] = materialized
			LOGGER.info(
				"Templates materialization unit done: %d/%d unit_id=%s sources=%d applied_upsampling=%s",
				idx,
				total_units,
				uid,
				len(source_payloads),
				bool(decision.get("applied", False)),
			)
		else:
			LOGGER.info(
				"Templates materialization unit skipped: %d/%d unit_id=%s sources=%d",
				idx,
				total_units,
				uid,
				len(source_payloads),
			)
	return results, decisions


def materialize_templates_from_spikeinterface(
	*,
	well_out_dir: Path,
	templates_out_dir: Path,
	concat_analyzer_relpath: str | None = None,
	concat_sorting_relpath: str | None = None,
	preprocessed_concat_reldir: str | None = None,
	preprocessed_segments_reldir: str | None = None,
	preproc_seg_sources_reldir: str | None = None,
	analyzer_cache_dir: Path | None = None,
	analyzer_cache_concat_subdir: str = "concat",
	analyzer_cache_segments_subdir: str = "",
	alternate_well_out_dirs: list[Path] | None = None,
	raw_data_h5_path: Path | None = None,
	stream_id: str | None = None,
	unit_ids: list[Any] | None,
	include_concat: bool,
	include_segments: bool,
	require_concat: bool = False,
	require_segments: bool = False,
	waveform_ms_before: float | None = None,
	waveform_ms_after: float | None = None,
	waveform_max_spikes_per_unit: int | None = None,
	execution_upsampling: TimeUpsampleConfig | None = None,
	enable_merge: bool = True,
	merge_method: str = "mean_all_waveforms",
	centering_method: str = "pre_peak_robust_baseline",
	max_waveforms_per_source_channel: int | None = 500,
	overlap_match_priority: tuple[str, ...] = ("electrode_id", "channel_id", "location"),
	location_tolerance_um: float = 1.0,
	debug_overlay: bool = False,
) -> tuple[Path, Path, dict[Any, dict[str, Any]]]:
	"""Build templates artifacts expected by templates v2 from SpikeInterface analyzers."""
	from ..io import (
		write_json,
		resolve_materialized_templates_dirs,
		write_materialized_overlay_waveforms,
		write_materialized_unit_templates,
	)
	from ..integrations.spikeinterface_extract import build_unit_source_payload, load_spikeinterface_analyzers

	analyzers = load_spikeinterface_analyzers(
		well_out_dir=well_out_dir,
		concat_analyzer_relpath=concat_analyzer_relpath,
		concat_sorting_relpath=concat_sorting_relpath,
		preprocessed_concat_reldir=preprocessed_concat_reldir,
		preprocessed_segments_reldir=preprocessed_segments_reldir,
		preproc_seg_sources_reldir=preproc_seg_sources_reldir,
		analyzer_cache_dir=analyzer_cache_dir,
		analyzer_cache_concat_subdir=analyzer_cache_concat_subdir,
		analyzer_cache_segments_subdir=analyzer_cache_segments_subdir,
		alternate_well_out_dirs=alternate_well_out_dirs,
		stream_id=stream_id,
		include_concat=bool(include_concat),
		include_segments=bool(include_segments),
		require_concat=bool(require_concat),
		require_segments=bool(require_segments),
		waveform_ms_before=waveform_ms_before,
		waveform_ms_after=waveform_ms_after,
		waveform_max_spikes_per_unit=waveform_max_spikes_per_unit,
	)
	LOGGER.info(
		"Templates materialization loaded analyzers: count=%d names=%s",
		len(analyzers),
		[str(name) for name, _ in analyzers],
	)

	if unit_ids is None:
		base_unit_ids = list(getattr(analyzers[0][1].sorting, "unit_ids", []))
	else:
		base_unit_ids = list(unit_ids)
	LOGGER.info("Templates materialization target units: count=%d", len(base_unit_ids))

	raw_sampling_rate_hz: float | None = None
	if raw_data_h5_path is not None and stream_id is not None:
		raw_sampling_rate_hz = read_maxwell_sampling_frequency_hz(
			h5_path=Path(raw_data_h5_path),
			stream_id=str(stream_id),
		)

	merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(
		templates_out_dir=templates_out_dir
	)
	concat_analyzer: Any | None = None
	for src_name, analyzer in analyzers:
		if str(src_name) == "concat":
			concat_analyzer = analyzer
			break
	if concat_analyzer is None and analyzers:
		concat_analyzer = analyzers[0][1]

	if concat_analyzer is not None:
		try:
			concat_locs = np.asarray(concat_analyzer.recording.get_channel_locations(), dtype=float)
			if concat_locs.ndim == 2 and int(concat_locs.shape[1]) >= 2 and int(concat_locs.shape[0]) > 0:
				concat_locs = np.asarray(concat_locs[:, :2], dtype=float)
				finite = np.isfinite(concat_locs).all(axis=1)
				if bool(np.any(finite)):
					concat_locs_path = templates_out_dir / "templates" / "concat_channel_locations_xy.npy"
					concat_locs_path.parent.mkdir(parents=True, exist_ok=True)
					np.save(concat_locs_path, np.asarray(concat_locs[finite, :], dtype=float))
		except Exception:
			LOGGER.debug("Failed writing concat channel locations metadata", exc_info=True)

		try:
			if concat_analyzer.has_extension("unit_locations"):
				loc_data_raw = concat_analyzer.get_extension("unit_locations").get_data()
				if hasattr(loc_data_raw, "to_numpy"):
					loc_data_raw = loc_data_raw.to_numpy()
				loc_data = np.asarray(loc_data_raw, dtype=float)
				unit_ids_raw = getattr(getattr(concat_analyzer, "sorting", None), "unit_ids", None)
				if unit_ids_raw is None:
					unit_ids_raw = getattr(concat_analyzer, "unit_ids", [])
				unit_ids_list = list(unit_ids_raw or [])
				rows: list[dict[str, Any]] = []
				if loc_data.ndim == 2 and int(loc_data.shape[1]) >= 2 and int(loc_data.shape[0]) == int(len(unit_ids_list)):
					for uid, xy in zip(unit_ids_list, loc_data[:, :2], strict=False):
						x = float(xy[0])
						y = float(xy[1])
						if (not np.isfinite(x)) or (not np.isfinite(y)):
							continue
						rows.append({"unit_id": uid, "x_um": x, "y_um": y})
				if rows:
					write_json(templates_out_dir / "templates" / "concat_unit_locations.json", rows)
		except Exception:
			LOGGER.debug("Failed writing concat unit locations metadata", exc_info=True)

	materialized_by_unit, upsampling_decisions_by_unit = materialize_unit_templates_by_unit_with_meta(
		analyzers=analyzers,
		unit_ids=base_unit_ids,
		payload_builder=lambda analyzer, unit_id: build_unit_source_payload(
			analyzer=analyzer,
			unit_id=unit_id,
			max_spikes_per_unit=waveform_max_spikes_per_unit,
			waveform_ms_before=waveform_ms_before,
			waveform_ms_after=waveform_ms_after,
		),
		enable_merge=bool(enable_merge),
		merge_method=merge_method,
		centering_method=centering_method,
		max_waveforms_per_source_channel=max_waveforms_per_source_channel,
		overlap_match_priority=overlap_match_priority,
		location_tolerance_um=location_tolerance_um,
		execution_upsampling=execution_upsampling,
		raw_sampling_rate_hz=raw_sampling_rate_hz,
	)
	LOGGER.info(
		"Templates materialization merge complete: materialized_units=%d decisions=%d",
		len(materialized_by_unit),
		len(upsampling_decisions_by_unit),
	)

	for uid, materialized in materialized_by_unit.items():
		merged_template, merged_locs, full_template, full_locs, merged_electrode_ids = materialized
		write_materialized_unit_templates(
			merged_units_dir=merged_units_dir,
			full_channels_templates_dir=full_channels_templates_dir,
			unit_id=uid,
			merged_template=merged_template,
			merged_locations_xy=merged_locs,
			full_template=full_template,
			full_locations_xy=full_locs,
		)
		from ..io import write_materialized_merged_electrode_ids
		write_materialized_merged_electrode_ids(
			merged_units_dir=merged_units_dir,
			unit_id=uid,
			electrode_ids=merged_electrode_ids,
		)

		# Persist top-electrode waveform snippets for waveform-level overlay rendering.
		selected_payload: tuple[Any, ...] | None = None
		for src_name, analyzer in analyzers:
			payload = build_unit_source_payload(
				analyzer=analyzer,
				unit_id=uid,
				max_spikes_per_unit=waveform_max_spikes_per_unit,
				waveform_ms_before=waveform_ms_before,
				waveform_ms_after=waveform_ms_after,
			)
			if payload is None or len(payload) < 9:
				if bool(debug_overlay):
					print(
						"[template_wf_overlay][debug] "
						f"unit={uid} source={src_name} payload_missing_or_short",
						flush=True,
					)
				continue
			if payload[6] is None:
				if bool(debug_overlay):
					print(
						"[template_wf_overlay][debug] "
						f"unit={uid} source={src_name} top_electrode_waveforms=None",
						flush=True,
					)
				continue
			if str(src_name) == "concat":
				selected_payload = payload
				if bool(debug_overlay):
					print(
						"[template_wf_overlay][debug] "
						f"unit={uid} selected source=concat for overlay artifact",
						flush=True,
					)
				break
			if selected_payload is None:
				selected_payload = payload
				if bool(debug_overlay):
					print(
						"[template_wf_overlay][debug] "
						f"unit={uid} selected source={src_name} for overlay artifact",
						flush=True,
					)

		if selected_payload is not None:
			try:
				wf = np.asarray(selected_payload[6], dtype=float)
				top_electrode_id = selected_payload[7]
				total = int(selected_payload[8]) if selected_payload[8] is not None else int(wf.shape[0])
				if wf.ndim == 2 and int(wf.shape[0]) > 0 and int(wf.shape[1]) > 0:
					write_materialized_overlay_waveforms(
						merged_units_dir=merged_units_dir,
						unit_id=uid,
						waveforms_by_t=wf,
						top_electrode_id=top_electrode_id,
						total_waveforms_at_channel=total,
					)
					if bool(debug_overlay):
						print(
							"[template_wf_overlay][debug] "
							f"unit={uid} wrote overlay artifact shape={tuple(wf.shape)} "
							f"top_electrode_id={top_electrode_id} total={total}",
							flush=True,
						)
				elif bool(debug_overlay):
					print(
						"[template_wf_overlay][debug] "
						f"unit={uid} overlay payload had invalid shape={tuple(np.asarray(wf).shape)}",
						flush=True,
					)
			except Exception as exc:
				if bool(debug_overlay):
					print(
						"[template_wf_overlay][debug] "
						f"unit={uid} failed writing overlay artifact (exception={exc!r})",
						flush=True,
					)
				LOGGER.debug("Failed writing overlay waveform artifact for unit %s", uid, exc_info=True)
		elif bool(debug_overlay):
			print(
				"[template_wf_overlay][debug] "
				f"unit={uid} no source produced top-electrode waveforms; artifact not written",
				flush=True,
			)

	return merged_units_dir, full_channels_templates_dir, upsampling_decisions_by_unit
