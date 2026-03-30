from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from ..core.source_payloads import normalize_source_payload

LOGGER = logging.getLogger("axon_recon.templates.spikeinterface")


def _read_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _parse_segment_index_from_name(name: str) -> int | None:
	token = str(name or "")
	for prefix in ("seg", "segment"):
		if token.lower().startswith(prefix):
			tail = token[len(prefix) :]
			digits = ""
			for ch in tail:
				if ch.isdigit():
					digits += ch
				else:
					break
			if digits:
				try:
					return int(digits)
				except Exception:
					return None
	if "_" in token:
		left = token.split("_", 1)[0]
		if left.isdigit():
			try:
				return int(left)
			except Exception:
				return None
	return None


def _load_concat_epoch_windows(*, preproc_segments_dir: Path, stream_id: str | None) -> list[dict[str, Any]]:
	parent = Path(preproc_segments_dir).parent
	candidates: list[Path] = []
	if stream_id:
		candidates.append(parent / f"concatenation_stitch_epochs_{stream_id}.json")
	candidates.extend(
		[
			parent / "concatenation_stitch_epochs.json",
			parent / "concat_epochs.json",
		]
	)

	for path in candidates:
		if not path.exists():
			continue
		try:
			payload = _read_json(path)
			if isinstance(payload, list):
				return [item for item in payload if isinstance(item, dict)]
		except Exception:
			continue
	return []


def _load_segment_manifest(*, preproc_segments_dir: Path) -> dict[tuple[int, str], Path]:
	manifest_path = Path(preproc_segments_dir) / "manifest.json"
	if not manifest_path.exists():
		return {}

	try:
		payload = _read_json(manifest_path)
	except Exception:
		return {}

	if not isinstance(payload, dict):
		return {}

	segments = payload.get("segments")
	if not isinstance(segments, list):
		return {}

	mapping: dict[tuple[int, str], Path] = {}
	for item in segments:
		if not isinstance(item, dict):
			continue
		try:
			seg_index = int(item.get("segment_index"))
		except Exception:
			continue
		rec_name = str(item.get("rec_name", ""))
		folder = item.get("folder")
		if not rec_name or folder is None:
			continue
		mapping[(int(seg_index), rec_name)] = Path(str(folder))
	return mapping


def _to_numpy_sorting(*, si_core: Any, unit_trains: dict[Any, list[int]], fs_hz: float) -> Any:
	import numpy as _np  # type: ignore[import-not-found]

	NumpySorting = getattr(si_core, "NumpySorting")
	unit_trains_np: dict[int, "_np.ndarray"] = {
		int(u): _np.asarray(times, dtype=_np.int64) for u, times in unit_trains.items()
	}

	unit_ids = sorted(unit_trains_np.keys())
	all_times: list[int] = []
	all_labels: list[int] = []
	for u in unit_ids:
		times_arr = unit_trains_np[u]
		if times_arr.size:
			all_times.extend(times_arr.tolist())
			all_labels.extend([u] * int(times_arr.size))

	if not all_times:
		return NumpySorting.from_unit_dict(unit_trains_np, sampling_frequency=float(fs_hz))

	times_arr = _np.asarray(all_times, dtype=_np.int64)
	labels_arr = _np.asarray(all_labels, dtype=_np.int64)
	order = _np.argsort(times_arr, kind="mergesort")
	times_arr = times_arr[order]
	labels_arr = labels_arr[order]

	return NumpySorting.from_times_labels(
		times_list=[times_arr],
		labels_list=[labels_arr],
		sampling_frequency=float(fs_hz),
	)


def _build_segment_analyzer_from_preprocessed_recording(
	*,
	si: Any,
	si_core: Any,
	concat_analyzer: Any,
	seg_dir: Path,
	seg_name: str,
	seg_index: int,
	start_sample: int | None,
	end_sample: int | None,
) -> Any | None:
	try:
		seg_rec = si.load_extractor(seg_dir)
	except Exception:
		try:
			seg_rec = si.load(seg_dir)
		except Exception:
			LOGGER.warning("Skipping segment source (unloadable preprocessed recording): %s", seg_dir)
			return None

	sorting = getattr(concat_analyzer, "sorting", None)
	if sorting is None:
		return None

	try:
		unit_ids = list(sorting.get_unit_ids())
	except Exception:
		unit_ids = list(getattr(sorting, "unit_ids", []))
	if not unit_ids:
		return None

	fs_hz: float | None = None
	for getter in (
		lambda: sorting.get_sampling_frequency(),
		lambda: concat_analyzer.recording.get_sampling_frequency(),
		lambda: seg_rec.get_sampling_frequency(),
	):
		try:
			cand = float(getter())
			if cand > 0.0 and np.isfinite(cand):
				fs_hz = cand
				break
		except Exception:
			continue
	if fs_hz is None:
		fs_hz = 10_000.0

	unit_trains_seg: dict[Any, list[int]] = {}

	try:
		n_segments = int(sorting.get_num_segments()) if hasattr(sorting, "get_num_segments") else 1
	except Exception:
		n_segments = 1

	if n_segments > 1 and 0 <= int(seg_index) < int(n_segments):
		for uid in unit_ids:
			try:
				st = sorting.get_unit_spike_train(unit_id=uid, segment_index=int(seg_index))
			except Exception:
				st = []
			unit_trains_seg[uid] = [int(t) for t in st]
	elif start_sample is not None and end_sample is not None:
		start = int(start_sample)
		end = int(end_sample)
		for uid in unit_ids:
			try:
				st = sorting.get_unit_spike_train(unit_id=uid, segment_index=0)
			except TypeError:
				st = sorting.get_unit_spike_train(uid)
			except Exception:
				st = []
			local = [int(t) - int(start) for t in st if int(start) <= int(t) < int(end)]
			unit_trains_seg[uid] = local
	else:
		LOGGER.info(
			"Skipping build for segment %s: missing concat window metadata and concat sorting is single-segment",
			str(seg_name),
		)
		return None

	try:
		seg_sort = _to_numpy_sorting(si_core=si_core, unit_trains=unit_trains_seg, fs_hz=float(fs_hz))
	except Exception:
		LOGGER.warning("Failed creating segment sorting for %s", str(seg_name), exc_info=True)
		return None

	try:
		seg_sort = si.remove_excess_spikes(seg_sort, seg_rec)
		seg_sort = seg_sort.remove_empty_units()
	except Exception:
		pass

	try:
		seg_units = list(getattr(seg_sort, "unit_ids", []))
	except Exception:
		seg_units = []
	if not seg_units:
		LOGGER.info("Skipping segment %s build: no units after cleanup", str(seg_name))
		return None

	try:
		seg_analyzer = si.create_sorting_analyzer(
			seg_sort,
			seg_rec,
			format="memory",
			return_in_uV=True,
		)
		seg_analyzer.compute(["random_spikes", "waveforms", "templates"], verbose=False, n_jobs=1)
		return seg_analyzer
	except Exception:
		LOGGER.warning("Failed building in-memory segment analyzer for %s", str(seg_name), exc_info=True)
		return None


def _unit_key(value: Any) -> str:
	try:
		return str(int(value))
	except Exception:
		return str(value)


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


def _extract_total_waveform_count(analyzer: Any, unit_id: Any) -> int:
	count = 1
	try:
		sorting = analyzer.sorting
		if hasattr(sorting, "get_num_segments") and hasattr(sorting, "get_unit_spike_train"):
			n_segments = int(sorting.get_num_segments())
			count = 0
			for seg_idx in range(max(1, n_segments)):
				st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=seg_idx)
				count += int(len(st))
			count = max(1, int(count))
	except Exception:
		count = 1
	return int(count)


def _parse_waveforms_window_from_extension(wf_ext: Any) -> tuple[float | None, float | None]:
	params_candidates: list[Any] = []
	for attr in ("params", "_params"):
		if hasattr(wf_ext, attr):
			try:
				params_candidates.append(getattr(wf_ext, attr))
			except Exception:
				pass

	for params in params_candidates:
		if not isinstance(params, dict):
			continue
		block = params.get("waveforms", params)
		if not isinstance(block, dict):
			continue
		ms_before = block.get("ms_before", None)
		ms_after = block.get("ms_after", None)
		try:
			ms_before_f = None if ms_before is None else float(ms_before)
		except Exception:
			ms_before_f = None
		try:
			ms_after_f = None if ms_after is None else float(ms_after)
		except Exception:
			ms_after_f = None
		if ms_before_f is not None or ms_after_f is not None:
			return ms_before_f, ms_after_f

	return None, None


def _try_recompute_waveforms_extension(
	*,
	analyzer: Any,
	requested_max_spikes_per_unit: int | None,
	requested_ms_before: float | None,
	requested_ms_after: float | None,
) -> bool:
	"""Best-effort recompute of random_spikes+waveforms+templates with requested semantics.

	Returns True when recompute appears to run successfully.
	"""
	attempted_attr = "_axon_recon_waveforms_recompute_attempted_params"
	attempt_key = (
		None if requested_max_spikes_per_unit is None else int(requested_max_spikes_per_unit),
		None if requested_ms_before is None else float(requested_ms_before),
		None if requested_ms_after is None else float(requested_ms_after),
	)
	seen = getattr(analyzer, attempted_attr, None)
	if isinstance(seen, set) and attempt_key in seen:
		return False
	if not isinstance(seen, set):
		seen = set()
	seen.add(attempt_key)
	try:
		setattr(analyzer, attempted_attr, seen)
	except Exception:
		pass

	try:
		wf_ext = analyzer.get_extension("waveforms")
	except Exception:
		wf_ext = None

	ms_before, ms_after = _parse_waveforms_window_from_extension(wf_ext)
	if requested_ms_before is not None:
		ms_before = float(requested_ms_before)
	if requested_ms_after is not None:
		ms_after = float(requested_ms_after)
	random_spikes_params: dict[str, Any] = {
		"method": "uniform",
		"seed": 0,
	}
	if requested_max_spikes_per_unit is not None and int(requested_max_spikes_per_unit) > 0:
		random_spikes_params["max_spikes_per_unit"] = int(requested_max_spikes_per_unit)

	extension_params: dict[str, Any] = {
		"random_spikes": random_spikes_params,
	}
	if ms_before is not None or ms_after is not None:
		wf_params: dict[str, Any] = {}
		if ms_before is not None:
			wf_params["ms_before"] = float(ms_before)
		if ms_after is not None:
			wf_params["ms_after"] = float(ms_after)
		extension_params["waveforms"] = wf_params

	try:
		analyzer.compute(
			["random_spikes", "waveforms", "templates"],
			extension_params=extension_params,
			verbose=False,
			n_jobs=1,
		)
		return True
	except Exception:
		LOGGER.debug("Failed to recompute waveforms extension with requested cap", exc_info=True)
		return False


def _normalize_requested_max_spikes_per_unit(max_spikes_per_unit: int | None) -> int | None:
	if max_spikes_per_unit is None:
		return None
	try:
		parsed = int(max_spikes_per_unit)
	except Exception:
		return None
	if parsed <= 0:
		return None
	return int(parsed)


def _mark_analyzer_waveforms_prepared(
	*,
	analyzer: Any,
	requested_max_spikes_per_unit: int | None,
	requested_ms_before: float | None,
	requested_ms_after: float | None,
) -> None:
	try:
		setattr(
			analyzer,
			"_axon_recon_prepared_waveforms_signature",
			(
				_normalize_requested_max_spikes_per_unit(requested_max_spikes_per_unit),
				(None if requested_ms_before is None else float(requested_ms_before)),
				(None if requested_ms_after is None else float(requested_ms_after)),
			),
		)
	except Exception:
		pass


def _waveforms_prepared_matches(
	*,
	analyzer: Any,
	requested_max_spikes_per_unit: int | None,
	requested_ms_before: float | None,
	requested_ms_after: float | None,
) -> bool:
	signature = getattr(analyzer, "_axon_recon_prepared_waveforms_signature", None)
	if not isinstance(signature, tuple) or len(signature) != 3:
		return False
	prepared_max, prepared_ms_before, prepared_ms_after = signature
	required_max = _normalize_requested_max_spikes_per_unit(requested_max_spikes_per_unit)
	if prepared_ms_before != (None if requested_ms_before is None else float(requested_ms_before)):
		return False
	if prepared_ms_after != (None if requested_ms_after is None else float(requested_ms_after)):
		return False
	if prepared_max is None:
		return True
	if required_max is None:
		return False
	return int(prepared_max) >= int(required_max)


def _prepare_analyzer_for_payload_extraction(
	*,
	analyzer: Any,
	requested_max_spikes_per_unit: int | None,
	requested_ms_before: float | None,
	requested_ms_after: float | None,
) -> Any:
	if not hasattr(analyzer, "has_extension") or not hasattr(analyzer, "compute"):
		return analyzer
	normalized_max = _normalize_requested_max_spikes_per_unit(requested_max_spikes_per_unit)
	if _waveforms_prepared_matches(
		analyzer=analyzer,
		requested_max_spikes_per_unit=normalized_max,
		requested_ms_before=requested_ms_before,
		requested_ms_after=requested_ms_after,
	):
		return analyzer
	needs_prepare = (
		normalized_max is not None
		or requested_ms_before is not None
		or requested_ms_after is not None
	)
	has_templates = False
	has_waveforms = False
	try:
		has_templates = bool(analyzer.has_extension("templates"))
	except Exception:
		has_templates = False
	try:
		has_waveforms = bool(analyzer.has_extension("waveforms"))
	except Exception:
		has_waveforms = False
	if needs_prepare or (not has_templates):
		_try_recompute_waveforms_extension(
			analyzer=analyzer,
			requested_max_spikes_per_unit=normalized_max,
			requested_ms_before=requested_ms_before,
			requested_ms_after=requested_ms_after,
		)
		_mark_analyzer_waveforms_prepared(
			analyzer=analyzer,
			requested_max_spikes_per_unit=normalized_max,
			requested_ms_before=requested_ms_before,
			requested_ms_after=requested_ms_after,
		)
	return analyzer


def _load_cached_analyzers(*, si: Any, analyzer_cache_dir: Path | None) -> dict[str, Any]:
	if analyzer_cache_dir is None or (not analyzer_cache_dir.exists()):
		return {}
	cached: dict[str, Any] = {}
	for folder in sorted(p for p in analyzer_cache_dir.iterdir() if p.is_dir()):
		try:
			cached[str(folder.name)] = si.load_sorting_analyzer(folder)
		except Exception:
			LOGGER.warning("Failed to load cached analyzer: %s", folder, exc_info=True)
	return cached


def _persist_analyzer_to_cache(*, analyzer: Any, analyzer_cache_dir: Path | None, analyzer_name: str) -> Any:
	if analyzer_cache_dir is None:
		return analyzer
	if not hasattr(analyzer, "save_as"):
		return analyzer
	folder = analyzer_cache_dir / str(analyzer_name)
	folder.parent.mkdir(parents=True, exist_ok=True)
	try:
		if folder.exists():
			shutil.rmtree(folder)
		saved = analyzer.save_as(format="binary_folder", folder=folder)
		return saved
	except Exception:
		LOGGER.warning("Failed to persist analyzer cache for %s at %s", analyzer_name, folder, exc_info=True)
		return analyzer


def build_unit_source_payload(
	*,
	analyzer: Any,
	unit_id: Any,
	max_spikes_per_unit: int | None = None,
	waveform_ms_before: float | None = None,
	waveform_ms_after: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int, float | None, np.ndarray | None, Any, int | None] | None:
	waveform_count = _extract_total_waveform_count(analyzer=analyzer, unit_id=unit_id)
	requested_waveforms = _normalize_requested_max_spikes_per_unit(max_spikes_per_unit)
	if not _waveforms_prepared_matches(
		analyzer=analyzer,
		requested_max_spikes_per_unit=requested_waveforms,
		requested_ms_before=waveform_ms_before,
		requested_ms_after=waveform_ms_after,
	):
		_prepare_analyzer_for_payload_extraction(
			analyzer=analyzer,
			requested_max_spikes_per_unit=requested_waveforms,
			requested_ms_before=waveform_ms_before,
			requested_ms_after=waveform_ms_after,
		)

	t = _extract_unit_template(analyzer, unit_id)
	if t is None:
		return None

	locs = np.asarray(analyzer.recording.get_channel_locations(), dtype=float)
	electrode_ids = _extract_electrode_ids(analyzer.recording)
	try:
		channel_ids = list(analyzer.recording.get_channel_ids())
	except Exception:
		channel_ids = None

	normalized = normalize_source_payload(
		template=t,
		locations_xy=locs,
		electrode_ids=electrode_ids,
		channel_ids=channel_ids,
		sparse_indices=_extract_sparse_channel_indices(analyzer, unit_id),
	)
	if normalized is None:
		return None
	t_ch_by_t, locs_xy, electrode_ids, channel_ids = normalized

	top_electrode_waveforms: np.ndarray | None = None
	top_electrode_id: Any = None
	top_electrode_waveform_count: int | None = None
	try:
		ptp = np.ptp(t_ch_by_t, axis=1)
		top_local_idx = int(np.argmax(ptp)) if int(ptp.size) > 0 else 0
		if electrode_ids is not None and int(top_local_idx) < len(electrode_ids):
			top_electrode_id = electrode_ids[int(top_local_idx)]
		elif channel_ids is not None and int(top_local_idx) < len(channel_ids):
			top_electrode_id = channel_ids[int(top_local_idx)]
		else:
			top_electrode_id = int(top_local_idx)

		if analyzer.has_extension("waveforms"):
			wf_ext = analyzer.get_extension("waveforms")
			wf_all = np.asarray(wf_ext.get_waveforms_one_unit(unit_id=unit_id, force_dense=False), dtype=float)
			need_waveforms = int(waveform_count) if requested_waveforms is None else int(min(max(1, requested_waveforms), int(waveform_count)))
			if requested_waveforms is None and int(wf_all.shape[0]) < int(waveform_count):
				if not _waveforms_prepared_matches(
					analyzer=analyzer,
					requested_max_spikes_per_unit=None,
					requested_ms_before=waveform_ms_before,
					requested_ms_after=waveform_ms_after,
				):
					if _try_recompute_waveforms_extension(
						analyzer=analyzer,
						requested_max_spikes_per_unit=None,
						requested_ms_before=waveform_ms_before,
						requested_ms_after=waveform_ms_after,
					):
						_mark_analyzer_waveforms_prepared(
							analyzer=analyzer,
							requested_max_spikes_per_unit=None,
							requested_ms_before=waveform_ms_before,
							requested_ms_after=waveform_ms_after,
						)
						wf_ext = analyzer.get_extension("waveforms")
						wf_all = np.asarray(wf_ext.get_waveforms_one_unit(unit_id=unit_id, force_dense=False), dtype=float)
			if requested_waveforms is not None and int(wf_all.shape[0]) < int(need_waveforms):
				if _try_recompute_waveforms_extension(
					analyzer=analyzer,
					requested_max_spikes_per_unit=int(need_waveforms),
					requested_ms_before=waveform_ms_before,
					requested_ms_after=waveform_ms_after,
				):
					_mark_analyzer_waveforms_prepared(
						analyzer=analyzer,
						requested_max_spikes_per_unit=int(need_waveforms),
						requested_ms_before=waveform_ms_before,
						requested_ms_after=waveform_ms_after,
					)
					wf_ext = analyzer.get_extension("waveforms")
					wf_all = np.asarray(wf_ext.get_waveforms_one_unit(unit_id=unit_id, force_dense=False), dtype=float)

			if requested_waveforms is None and int(wf_all.shape[0]) > int(waveform_count):
				wf_all = np.asarray(wf_all[: int(waveform_count), :, :], dtype=float)
			if requested_waveforms is not None and int(wf_all.shape[0]) > int(requested_waveforms):
				wf_all = np.asarray(wf_all[: int(requested_waveforms), :, :], dtype=float)

			if wf_all.ndim == 3 and int(wf_all.shape[0]) > 0 and int(wf_all.shape[1]) > 0 and int(wf_all.shape[2]) > 0:
				# Expected sparse-path case: channels match normalized template channels.
				if int(wf_all.shape[2]) == int(t_ch_by_t.shape[0]):
					top_electrode_waveforms = np.asarray(wf_all[:, :, int(top_local_idx)], dtype=float)
				# Dense-path fallback: channel axis may be full recording channels.
				elif int(wf_all.shape[2]) > int(top_local_idx):
					top_electrode_waveforms = np.asarray(wf_all[:, :, int(top_local_idx)], dtype=float)
				if top_electrode_waveforms is not None and top_electrode_waveforms.ndim == 2:
					top_electrode_waveform_count = int(top_electrode_waveforms.shape[0])
	except Exception:
		top_electrode_waveforms = None
		top_electrode_waveform_count = None

	sampling_rate_hz: float | None
	try:
		sampling_rate_hz = float(analyzer.recording.get_sampling_frequency())
		if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
			sampling_rate_hz = None
	except Exception:
		sampling_rate_hz = None

	return (
		t_ch_by_t,
		locs_xy,
		electrode_ids,
		channel_ids,
		int(waveform_count),
		sampling_rate_hz,
		top_electrode_waveforms,
		top_electrode_id,
		top_electrode_waveform_count,
	)


def load_spikeinterface_analyzers(
	*,
	well_out_dir: Path,
	concat_analyzer_relpath: str | None = None,
	preproc_seg_sources_reldir: str | None = None,
	analyzer_cache_dir: Path | None = None,
	stream_id: str | None = None,
	include_concat: bool,
	include_segments: bool,
	waveform_ms_before: float | None = None,
	waveform_ms_after: float | None = None,
	waveform_max_spikes_per_unit: int | None = None,
) -> list[tuple[str, Any]]:
	import spikeinterface.full as si  # type: ignore[import-not-found]

	def _resolve_from_well(raw_path: str | None) -> Path | None:
		if raw_path is None:
			return None
		token = str(raw_path).strip()
		if not token:
			return None
		p = Path(token).expanduser()
		if p.is_absolute() and p.exists():
			return p
		# Treat leading-slash tokens as well-root-relative knobs (config style convention).
		if p.is_absolute():
			token = token.lstrip("/")
			p = Path(token)
		return (well_out_dir / p).resolve()

	wf_out = well_out_dir / "stg3_waveforms_outputs"
	concat_dir = _resolve_from_well(concat_analyzer_relpath) or (wf_out / "concat_waveforms")
	segments_dir = _resolve_from_well(preproc_seg_sources_reldir) or (wf_out / "segment_waveforms")
	cache_root = None if analyzer_cache_dir is None else Path(analyzer_cache_dir).expanduser().resolve()
	cached_analyzers = _load_cached_analyzers(si=si, analyzer_cache_dir=cache_root)

	analyzers: list[tuple[str, Any]] = []
	concat_analyzer_obj: Any | None = None
	if "concat" in cached_analyzers:
		try:
			concat_analyzer_obj = _prepare_analyzer_for_payload_extraction(
				analyzer=cached_analyzers["concat"],
				requested_max_spikes_per_unit=waveform_max_spikes_per_unit,
				requested_ms_before=waveform_ms_before,
				requested_ms_after=waveform_ms_after,
			)
			if include_concat:
				analyzers.append(("concat", concat_analyzer_obj))
		except Exception:
			LOGGER.warning("Failed to prepare cached concat analyzer: %s", cache_root / "concat", exc_info=True)
	elif concat_dir.exists():
		try:
			concat_analyzer_obj = si.load_sorting_analyzer(concat_dir)
			concat_analyzer_obj = _persist_analyzer_to_cache(
				analyzer=concat_analyzer_obj,
				analyzer_cache_dir=cache_root,
				analyzer_name="concat",
			)
			concat_analyzer_obj = _prepare_analyzer_for_payload_extraction(
				analyzer=concat_analyzer_obj,
				requested_max_spikes_per_unit=waveform_max_spikes_per_unit,
				requested_ms_before=waveform_ms_before,
				requested_ms_after=waveform_ms_after,
			)
			if include_concat:
				analyzers.append(("concat", concat_analyzer_obj))
		except Exception:
			LOGGER.warning("Failed to load concat analyzer: %s", concat_dir)
	if include_concat and ("concat" not in cached_analyzers) and (not concat_dir.exists()):
		LOGGER.info("Concat analyzer directory not found: %s", concat_dir)

	if concat_analyzer_obj is None and "concat" in cached_analyzers:
		try:
			concat_analyzer_obj = _prepare_analyzer_for_payload_extraction(
				analyzer=cached_analyzers["concat"],
				requested_max_spikes_per_unit=waveform_max_spikes_per_unit,
				requested_ms_before=waveform_ms_before,
				requested_ms_after=waveform_ms_after,
			)
		except Exception:
			concat_analyzer_obj = None
	if concat_analyzer_obj is None and concat_dir.exists():
		try:
			concat_analyzer_obj = si.load_sorting_analyzer(concat_dir)
			concat_analyzer_obj = _persist_analyzer_to_cache(
				analyzer=concat_analyzer_obj,
				analyzer_cache_dir=cache_root,
				analyzer_name="concat",
			)
			concat_analyzer_obj = _prepare_analyzer_for_payload_extraction(
				analyzer=concat_analyzer_obj,
				requested_max_spikes_per_unit=waveform_max_spikes_per_unit,
				requested_ms_before=waveform_ms_before,
				requested_ms_after=waveform_ms_after,
			)
		except Exception:
			concat_analyzer_obj = None

	if include_segments and segments_dir.exists():
		seg_dirs = sorted([p for p in segments_dir.iterdir() if p.is_dir()])
		seg_dir_by_name = {str(p.name): p for p in seg_dirs}
		segment_names: list[str] = []
		for name in list(seg_dir_by_name.keys()) + sorted(k for k in cached_analyzers.keys() if k != "concat"):
			if name not in segment_names:
				segment_names.append(name)
		unloadable_seg_dirs: list[Path] = []
		for seg_name in segment_names:
			if seg_name in cached_analyzers:
				try:
					seg_analyzer = _prepare_analyzer_for_payload_extraction(
						analyzer=cached_analyzers[seg_name],
						requested_max_spikes_per_unit=waveform_max_spikes_per_unit,
						requested_ms_before=waveform_ms_before,
						requested_ms_after=waveform_ms_after,
					)
					analyzers.append((seg_name, seg_analyzer))
					continue
				except Exception:
					LOGGER.warning("Failed to prepare cached segment analyzer: %s", cache_root / seg_name, exc_info=True)
			seg_dir = seg_dir_by_name.get(seg_name, None)
			if seg_dir is None:
				continue
			try:
				seg_analyzer = si.load_sorting_analyzer(seg_dir)
				seg_analyzer = _persist_analyzer_to_cache(
					analyzer=seg_analyzer,
					analyzer_cache_dir=cache_root,
					analyzer_name=seg_name,
				)
				seg_analyzer = _prepare_analyzer_for_payload_extraction(
					analyzer=seg_analyzer,
					requested_max_spikes_per_unit=waveform_max_spikes_per_unit,
					requested_ms_before=waveform_ms_before,
					requested_ms_after=waveform_ms_after,
				)
				analyzers.append((seg_name, seg_analyzer))
			except Exception:
				unloadable_seg_dirs.append(seg_dir)
				LOGGER.info("Segment directory is not a loadable analyzer (will try build fallback): %s", seg_dir)

		if unloadable_seg_dirs and concat_analyzer_obj is not None:
			try:
				import spikeinterface.core as si_core  # type: ignore[import-not-found]
			except Exception:
				si_core = None
			if si_core is None:
				LOGGER.info(
					"Segment fallback build skipped: spikeinterface.core unavailable for source_dir=%s",
					str(segments_dir),
				)
				unloadable_seg_dirs = []

		if unloadable_seg_dirs and concat_analyzer_obj is not None:
			epochs = _load_concat_epoch_windows(preproc_segments_dir=segments_dir, stream_id=stream_id)
			epoch_by_key: dict[tuple[int, str], tuple[int, int]] = {}
			for ep in epochs:
				try:
					seg_index = int(ep.get("segment_index"))
					rec_name = str(ep.get("rec_name", ""))
					start = int(ep.get("start_sample"))
					end = int(ep.get("end_sample"))
				except Exception:
					continue
				if rec_name and end > start:
					epoch_by_key[(int(seg_index), rec_name)] = (int(start), int(end))

			manifest_by_key = _load_segment_manifest(preproc_segments_dir=segments_dir)
			manifest_inverse = {v.resolve(): k for k, v in manifest_by_key.items()}

			built_count = 0
			for seg_dir in unloadable_seg_dirs:
				seg_name = str(seg_dir.name)
				seg_index = _parse_segment_index_from_name(seg_name)
				rec_name: str | None = None

				resolved_seg = seg_dir.resolve()
				manifest_key = manifest_inverse.get(resolved_seg)
				if manifest_key is not None:
					seg_index = int(manifest_key[0])
					rec_name = str(manifest_key[1])

				start_sample: int | None = None
				end_sample: int | None = None
				if seg_index is not None and rec_name is not None:
					window = epoch_by_key.get((int(seg_index), str(rec_name)))
					if window is not None:
						start_sample, end_sample = int(window[0]), int(window[1])
				elif seg_index is not None:
					for (idx, rn), window in epoch_by_key.items():
						if int(idx) == int(seg_index):
							rec_name = str(rn)
							start_sample, end_sample = int(window[0]), int(window[1])
							break

				if seg_index is None:
					LOGGER.info("Skipping segment fallback build; could not infer segment index: %s", seg_dir)
					continue

				built = _build_segment_analyzer_from_preprocessed_recording(
					si=si,
					si_core=si_core,
					concat_analyzer=concat_analyzer_obj,
					seg_dir=seg_dir,
					seg_name=seg_name,
					seg_index=int(seg_index),
					start_sample=start_sample,
					end_sample=end_sample,
				)
				if built is not None:
					built = _persist_analyzer_to_cache(
						analyzer=built,
						analyzer_cache_dir=cache_root,
						analyzer_name=seg_name,
					)
					built = _prepare_analyzer_for_payload_extraction(
						analyzer=built,
						requested_max_spikes_per_unit=waveform_max_spikes_per_unit,
						requested_ms_before=waveform_ms_before,
						requested_ms_after=waveform_ms_after,
					)
					analyzers.append((seg_name, built))
					built_count += 1

			if built_count > 0:
				LOGGER.info(
					"Built segment analyzers from preprocessed recordings: count=%d source_dir=%s",
					int(built_count),
					str(segments_dir),
				)
		elif unloadable_seg_dirs and concat_analyzer_obj is None:
			LOGGER.info(
				"Segment fallback build skipped: concat analyzer unavailable for source_dir=%s",
				str(segments_dir),
			)
	if include_segments and (not segments_dir.exists()) and (len([k for k in cached_analyzers.keys() if k != "concat"]) == 0):
		LOGGER.info("Segment analyzers directory not found: %s", segments_dir)

	if not analyzers:
		raise FileNotFoundError(
			"No SpikeInterface analyzers found for templates materialization. "
			f"checked concat={concat_dir} segments={segments_dir}."
		)
	LOGGER.info(
		"Loaded SpikeInterface analyzers from concat=%s segments=%s count=%d",
		str(concat_dir),
		str(segments_dir),
		len(analyzers),
	)
	return analyzers
