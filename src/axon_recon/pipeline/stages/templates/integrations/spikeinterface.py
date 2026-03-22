from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]

LOGGER = logging.getLogger("axon_recon.templates.spikeinterface")


def _unit_key(value: Any) -> str:
	try:
		return str(int(value))
	except Exception:
		return str(value)


def _normalize_template_to_channels_by_time(template: Any, n_channels_hint: int) -> np.ndarray:
	t = np.asarray(template, dtype=float)
	if t.ndim != 2:
		raise ValueError(f"Expected 2D template, got shape={getattr(t, 'shape', None)}")
	if int(t.shape[0]) == int(n_channels_hint):
		return t
	if int(t.shape[1]) == int(n_channels_hint):
		return t.T
	# Fallback: choose orientation that is more likely channels x time.
	if int(t.shape[0]) < int(t.shape[1]):
		return t
	return t.T


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


def _normalize_merge_method(method: str) -> str:
	m = str(method or "mean_all_waveforms").strip().lower()
	if m in {"average_all_wfs", "mean_waveforms", "mean_all_waveforms", "mean"}:
		return "mean_all_waveforms"
	if m in {"weighted_average", "weighted_by_channel_waveform_count", "weighted"}:
		return "weighted_by_channel_waveform_count"
	return "mean_all_waveforms"


def _normalize_overlap_priorities(overlap_match_priority: tuple[str, ...]) -> tuple[str, ...]:
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


def _extract_unit_template(analyzer: Any, unit_id: Any) -> np.ndarray | None:
	try:
		has_templates = bool(analyzer.has_extension("templates"))
	except Exception:
		has_templates = False

	if not has_templates:
		try:
			analyzer.compute(["templates"], verbose=False, n_jobs=1)
			has_templates = bool(analyzer.has_extension("templates"))
		except Exception:
			has_templates = False

	if not has_templates:
		return None

	try:
		t_ext = analyzer.get_extension("templates")
	except Exception:
		return None

	if hasattr(t_ext, "get_unit_template"):
		try:
			return np.asarray(t_ext.get_unit_template(unit_id=unit_id), dtype=float)
		except Exception:
			pass

	if hasattr(t_ext, "get_templates"):
		try:
			all_templates = t_ext.get_templates()
			unit_ids = list(getattr(analyzer.sorting, "unit_ids", []))
			idx = unit_ids.index(unit_id)
			return np.asarray(all_templates[idx], dtype=float)
		except Exception:
			return None

	return None


def _extract_sparse_channel_indices(analyzer: Any, unit_id: Any) -> np.ndarray | None:
	sp = getattr(analyzer, "sparsity", None)
	if sp is None:
		try:
			if analyzer.has_extension("waveforms"):
				sp = getattr(analyzer.get_extension("waveforms"), "sparsity", None)
		except Exception:
			sp = None
	if sp is None:
		return None

	mapping = getattr(sp, "unit_id_to_channel_indices", None)
	if callable(mapping):
		try:
			return np.asarray(mapping(unit_id), dtype=int)
		except Exception:
			pass
	if isinstance(mapping, dict):
		if unit_id in mapping:
			try:
				return np.asarray(mapping[unit_id], dtype=int)
			except Exception:
				pass
		for k, v in mapping.items():
			if _unit_key(k) == _unit_key(unit_id):
				try:
					return np.asarray(v, dtype=int)
				except Exception:
					pass

	for name in ("get_channel_indices", "get_channel_indices_for_unit"):
		fn = getattr(sp, name, None)
		if callable(fn):
			try:
				return np.asarray(fn(unit_id), dtype=int)
			except Exception:
				pass

	return None


def _extract_electrode_ids(recording: Any) -> list[Any] | None:
	for key in (
		"electrode_id",
		"electrode",
		"contact_id",
		"contact_ids",
		"contact",
		"site_id",
		"site",
	):
		try:
			if hasattr(recording, "get_property_keys") and key not in set(recording.get_property_keys()):
				continue
			vals = recording.get_property(key)
			if vals is not None:
				return list(vals)
		except Exception:
			continue

	try:
		cv = recording.get_property("contact_vector")
		if isinstance(cv, dict) and "electrode" in cv:
			return list(cv["electrode"])
	except Exception:
		pass
	return None


def _template_payload_for_unit(
	analyzer: Any,
	unit_id: Any,
) -> tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int] | None:
	t = _extract_unit_template(analyzer, unit_id)
	if t is None:
		return None

	locs = np.asarray(analyzer.recording.get_channel_locations(), dtype=float)
	locs = locs[:, :2]
	electrode_ids = _extract_electrode_ids(analyzer.recording)
	try:
		channel_ids = list(analyzer.recording.get_channel_ids())
	except Exception:
		channel_ids = None

	if t.ndim != 2:
		return None

	# Sparse template handling: subset locations by sparsity when possible.
	sparse_inds = _extract_sparse_channel_indices(analyzer, unit_id)
	if sparse_inds is not None and int(sparse_inds.size) > 0:
		if int(t.shape[1]) == int(sparse_inds.size):
			locs = locs[sparse_inds, :]
			if channel_ids is not None:
				try:
					channel_ids = list(np.asarray(channel_ids, dtype=object)[sparse_inds])
				except Exception:
					pass
			if electrode_ids is not None:
				try:
					electrode_ids = list(np.asarray(electrode_ids, dtype=object)[sparse_inds])
				except Exception:
					pass
		elif int(t.shape[0]) == int(sparse_inds.size):
			locs = locs[sparse_inds, :]
			if channel_ids is not None:
				try:
					channel_ids = list(np.asarray(channel_ids, dtype=object)[sparse_inds])
				except Exception:
					pass
			if electrode_ids is not None:
				try:
					electrode_ids = list(np.asarray(electrode_ids, dtype=object)[sparse_inds])
				except Exception:
					pass

	t_ch_by_t = _normalize_template_to_channels_by_time(t, int(locs.shape[0]))
	if int(t_ch_by_t.shape[0]) != int(locs.shape[0]):
		return None

	if electrode_ids is not None and len(electrode_ids) != int(t_ch_by_t.shape[0]):
		electrode_ids = None
	if channel_ids is not None and len(channel_ids) != int(t_ch_by_t.shape[0]):
		channel_ids = None

	# Keep only channels with non-flat waveforms so merged_contributing remains truly contributing.
	ptp = np.ptp(t_ch_by_t, axis=1)
	keep = np.where(ptp > float(np.finfo(float).eps))[0]
	if int(keep.size) == 0:
		# Fallback: preserve at least one channel if all channels are numerically flat.
		keep = np.asarray([int(np.argmax(np.max(np.abs(t_ch_by_t), axis=1)))], dtype=int)
	t_ch_by_t = t_ch_by_t[keep, :]
	locs = locs[keep, :]
	if electrode_ids is not None:
		try:
			electrode_ids = list(np.asarray(electrode_ids, dtype=object)[keep])
		except Exception:
			electrode_ids = None
	if channel_ids is not None:
		try:
			channel_ids = list(np.asarray(channel_ids, dtype=object)[keep])
		except Exception:
			channel_ids = None

	waveform_count = 1
	try:
		sorting = analyzer.sorting
		if hasattr(sorting, "get_num_segments") and hasattr(sorting, "get_unit_spike_train"):
			n_segments = int(sorting.get_num_segments())
			count = 0
			for seg_idx in range(max(1, n_segments)):
				st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=seg_idx)
				count += int(len(st))
			waveform_count = max(1, int(count))
	except Exception:
		waveform_count = 1

	return t_ch_by_t, locs, electrode_ids, channel_ids, int(waveform_count)


def _resample_to_len(signal: np.ndarray, target_len: int) -> np.ndarray:
	if int(signal.shape[0]) == int(target_len):
		return signal.astype(float, copy=False)
	if int(signal.shape[0]) <= 1 or int(target_len) <= 1:
		return np.resize(signal.astype(float, copy=False), (int(target_len),))
	x_old = np.linspace(0.0, 1.0, int(signal.shape[0]), dtype=float)
	x_new = np.linspace(0.0, 1.0, int(target_len), dtype=float)
	return np.interp(x_new, x_old, signal.astype(float, copy=False))


def _merge_sources_per_channel(
	sources: list[tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int]],
	*,
	enable_merge: bool,
	merge_method: str,
	centering_method: str,
	max_waveforms_per_source_channel: int | None,
	overlap_match_priority: tuple[str, ...],
	location_tolerance_um: float,
) -> tuple[np.ndarray, np.ndarray]:
	if not sources:
		raise ValueError("No template sources to merge")

	merge_method_norm = _normalize_merge_method(merge_method)
	match_priority = _normalize_overlap_priorities(overlap_match_priority)
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

	for t_ch_by_t, locs, electrode_ids, channel_ids, source_waveform_count in sources:
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
	for key in keys:
		den = max(1e-12, float(bucket_weights[key]))
		merged_waves.append(bucket_waveforms[key] / den)
		merged_locs.append(bucket_locations[key])

	return np.vstack(merged_waves), np.asarray(merged_locs, dtype=float)


def _load_analyzers(
	*,
	well_out_dir: Path,
	include_concat: bool,
	include_segments: bool,
) -> list[tuple[str, Any]]:
	import spikeinterface.full as si  # type: ignore[import-not-found]

	wf_out = well_out_dir / "stg3_waveforms_outputs"
	concat_dir = wf_out / "concat_waveforms"
	segments_dir = wf_out / "segment_waveforms"

	analyzers: list[tuple[str, Any]] = []
	if include_concat and concat_dir.exists():
		analyzers.append(("concat", si.load_sorting_analyzer(concat_dir)))
	if include_segments and segments_dir.exists():
		for seg_dir in sorted([p for p in segments_dir.iterdir() if p.is_dir()]):
			try:
				analyzers.append((seg_dir.name, si.load_sorting_analyzer(seg_dir)))
			except Exception:
				LOGGER.warning("Skipping unreadable segment analyzer: %s", seg_dir)

	if not analyzers:
		raise FileNotFoundError(
			"No SpikeInterface analyzers found under stg3_waveforms_outputs. "
			"Expected concat_waveforms and/or segment_waveforms."
		)
	return analyzers


def materialize_templates_from_spikeinterface(
	*,
	well_out_dir: Path,
	templates_out_dir: Path,
	unit_ids: list[Any] | None,
	include_concat: bool,
	include_segments: bool,
	enable_merge: bool = True,
	merge_method: str = "mean_all_waveforms",
	centering_method: str = "pre_peak_robust_baseline",
	max_waveforms_per_source_channel: int | None = 500,
	overlap_match_priority: tuple[str, ...] = ("electrode_id", "channel_id", "location"),
	location_tolerance_um: float = 1.0,
) -> tuple[Path, Path]:
	"""Build templates artifacts expected by templates v2 from SpikeInterface analyzers.

	Writes merged/full template npy artifacts under template_outputs/templates.
	Returns (merged_units_dir, full_channels_templates_dir).
	"""

	analyzers = _load_analyzers(
		well_out_dir=well_out_dir,
		include_concat=bool(include_concat),
		include_segments=bool(include_segments),
	)

	if unit_ids is None:
		base_unit_ids = list(getattr(analyzers[0][1].sorting, "unit_ids", []))
	else:
		base_unit_ids = list(unit_ids)

	templates_root = templates_out_dir / "templates"
	merged_units_dir = templates_root / "merged"
	full_channels_templates_dir = templates_root / "full"
	merged_units_dir.mkdir(parents=True, exist_ok=True)
	full_channels_templates_dir.mkdir(parents=True, exist_ok=True)

	for uid in base_unit_ids:
		sources_payloads: list[tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int]] = []
		concat_payload: tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int] | None = None
		for src_name, analyzer in analyzers:
			payload = _template_payload_for_unit(analyzer, uid)
			if payload is None:
				continue
			sources_payloads.append(payload)
			if str(src_name) == "concat":
				concat_payload = payload

		if not sources_payloads:
			continue

		merged_template, merged_locs = _merge_sources_per_channel(
			sources_payloads,
			enable_merge=bool(enable_merge),
			merge_method=merge_method,
			centering_method=centering_method,
			max_waveforms_per_source_channel=max_waveforms_per_source_channel,
			overlap_match_priority=overlap_match_priority,
			location_tolerance_um=location_tolerance_um,
		)
		merged_dir = merged_units_dir / f"unit_{uid}"
		merged_dir.mkdir(parents=True, exist_ok=True)
		np.save(merged_dir / "merged_contributing_template.npy", merged_template)
		np.save(merged_dir / "merged_contributing_channel_locations.npy", merged_locs)

		full_template = merged_template
		full_locs = merged_locs
		if concat_payload is not None:
			full_template = concat_payload[0]
			full_locs = concat_payload[1]
		full_dir = full_channels_templates_dir / f"unit_{uid}"
		full_dir.mkdir(parents=True, exist_ok=True)
		np.save(full_dir / "full_template.npy", full_template)
		np.save(full_dir / "full_channel_locations_xy.npy", full_locs)

	return merged_units_dir, full_channels_templates_dir
