from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from ..core.source_payloads import normalize_source_payload
from ..models.inputs import AnalyzerPreparationPolicyConfig

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


def _sorting_analyzer_create_kwargs_from_policy(*, policy: AnalyzerPreparationPolicyConfig) -> dict[str, Any]:
	dense_requested = (not bool(policy.compute_sparsity)) or str(policy.sparsity_mode).strip().lower() == "dense"
	if dense_requested:
		return {"format": "memory", "return_in_uV": True, "sparse": False}

	method_raw = str(policy.sparsity_method or "radius").strip().lower()
	method = {"threshold": "snr"}.get(method_raw, method_raw)
	kwargs: dict[str, Any] = {
		"format": "memory",
		"return_in_uV": True,
		"sparse": True,
		"method": method,
	}
	if policy.ms_before is not None:
		kwargs["ms_before"] = float(policy.ms_before)
	if policy.ms_after is not None:
		kwargs["ms_after"] = float(policy.ms_after)
	if policy.sparsity_radius_um is not None:
		kwargs["radius_um"] = float(policy.sparsity_radius_um)
	if policy.sparsity_num_channels is not None:
		kwargs["num_channels"] = int(policy.sparsity_num_channels)
	if policy.sparsity_threshold is not None:
		kwargs["threshold"] = float(policy.sparsity_threshold)
	if str(policy.sparsity_peak_sign or "").strip() != "":
		kwargs["peak_sign"] = str(policy.sparsity_peak_sign)
	if policy.sparsity_num_spikes_for_sparsity is not None:
		kwargs["num_spikes_for_sparsity"] = int(policy.sparsity_num_spikes_for_sparsity)
	if str(policy.sparsity_by_property or "").strip() != "":
		kwargs["by_property"] = str(policy.sparsity_by_property)
	if policy.n_jobs is not None and int(policy.n_jobs) > 0:
		kwargs["n_jobs"] = int(policy.n_jobs)
	if str(policy.chunk_duration or "").strip() != "":
		kwargs["chunk_duration"] = str(policy.chunk_duration)
	return kwargs


def _format_log_value(value: Any) -> str:
	if isinstance(value, float):
		try:
			if np.isfinite(value):
				return f"{float(value):.4g}"
		except Exception:
			pass
	return str(value)


def _format_log_fields(fields: dict[str, Any]) -> str:
	parts: list[str] = []
	for key, value in fields.items():
		if value is None:
			continue
		parts.append(f"{key}={_format_log_value(value)}")
	return " ".join(parts)


def _policy_log_fields(policy: AnalyzerPreparationPolicyConfig) -> dict[str, Any]:
	percentage = None
	if policy.random_spikes_percentage is not None:
		try:
			percentage = f"{float(policy.random_spikes_percentage) * 100.0:.1f}%"
		except Exception:
			percentage = policy.random_spikes_percentage
	return {
		"ms_before": policy.ms_before,
		"ms_after": policy.ms_after,
		"dtype": policy.dtype,
		"max_spikes_per_unit": policy.max_spikes_per_unit,
		"min_spikes_per_unit": policy.min_spikes_per_unit,
		"random_spikes_method": policy.random_spikes_method,
		"random_spikes_percentage": percentage,
		"random_seed": policy.random_seed,
		"log_before_after_spike_counts": bool(policy.log_before_after_spike_counts),
		"margin_size": policy.margin_size,
		"compute_sparsity": bool(policy.compute_sparsity),
		"sparsity_mode": policy.sparsity_mode,
		"sparsity_method": policy.sparsity_method,
		"sparsity_radius_um": policy.sparsity_radius_um,
		"sparsity_num_channels": policy.sparsity_num_channels,
		"sparsity_threshold": policy.sparsity_threshold,
		"sparsity_peak_sign": policy.sparsity_peak_sign,
		"sparsity_num_spikes_for_sparsity": policy.sparsity_num_spikes_for_sparsity,
		"n_jobs": policy.n_jobs,
		"chunk_duration": policy.chunk_duration,
	}


def _format_policy_for_log(policy: AnalyzerPreparationPolicyConfig) -> str:
	return _format_log_fields(_policy_log_fields(policy))


def _count_unit_train_spikes(unit_trains: dict[Any, list[int]]) -> int:
	try:
		return int(sum(len(times) for times in unit_trains.values()))
	except Exception:
		return 0


def _build_segment_analyzer_from_preprocessed_recording(
	*,
	si: Any,
	si_core: Any,
	policy: AnalyzerPreparationPolicyConfig,
	concat_analyzer: Any,
	seg_dir: Path,
	seg_name: str,
	seg_index: int,
	rec_name: str | None,
	start_sample: int | None,
	end_sample: int | None,
) -> Any | None:
	LOGGER.info(
		"Loading preprocessed segment recording for analyzer build: segment=%s path=%s policy=%s",
		str(seg_name),
		str(seg_dir),
		_format_policy_for_log(policy),
	)
	try:
		seg_rec = si.load_extractor(seg_dir)
	except Exception:
		try:
			seg_rec = si.load(seg_dir)
		except Exception:
			LOGGER.warning(
				"Failed to load preprocessed segment recording for analyzer build: segment=%s path=%s",
				str(seg_name),
				str(seg_dir),
			)
			return None
	LOGGER.info(
		"Loaded preprocessed segment recording for analyzer build: segment=%s path=%s",
		str(seg_name),
		str(seg_dir),
	)

	sorting = getattr(concat_analyzer, "sorting", None)
	if sorting is None:
		LOGGER.info("Skipping segment analyzer build: concat sorting unavailable for segment=%s", str(seg_name))
		return None

	try:
		unit_ids = list(sorting.get_unit_ids())
	except Exception:
		unit_ids = list(getattr(sorting, "unit_ids", []))
	if not unit_ids:
		LOGGER.info("Skipping segment analyzer build: concat sorting has no units for segment=%s", str(seg_name))
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
	LOGGER.info(
		"Registered preprocessed segment recording with concat spikes: segment=%s rec_name=%s seg_index=%s start_sample=%s end_sample=%s units=%d total_spikes=%d",
		str(seg_name),
		("<unknown>" if rec_name in {None, ""} else str(rec_name)),
		int(seg_index),
		start_sample,
		end_sample,
		int(len(unit_ids)),
		_count_unit_train_spikes(unit_trains_seg),
	)

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
		create_kwargs = _sorting_analyzer_create_kwargs_from_policy(policy=policy)
		LOGGER.info(
			"Generating segment analyzer: segment=%s units=%d creation=%s",
			str(seg_name),
			int(len(seg_units)),
			_format_log_fields(create_kwargs),
		)
		seg_analyzer = si.create_sorting_analyzer(
			seg_sort,
			seg_rec,
			**create_kwargs,
		)
		compute_kwargs: dict[str, Any] = {
			"verbose": False,
			"n_jobs": (1 if policy.n_jobs is None else int(policy.n_jobs)),
		}
		if str(policy.chunk_duration or "").strip() != "":
			compute_kwargs["chunk_duration"] = str(policy.chunk_duration)
		LOGGER.info(
			"Computing segment analyzer extensions: segment=%s extensions=random_spikes,waveforms,templates compute=%s",
			str(seg_name),
			_format_log_fields(compute_kwargs),
		)
		seg_analyzer.compute(["random_spikes", "waveforms", "templates"], **compute_kwargs)
		LOGGER.info(
			"Generated segment analyzer from preprocessed recording: segment=%s units=%d",
			str(seg_name),
			int(len(seg_units)),
		)
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


def _analyzer_has_sparsity(analyzer: Any) -> bool:
	try:
		return bool(getattr(analyzer, "sparsity", None) is not None)
	except Exception:
		return False


def _normalize_requested_random_spikes_method(method: str | None) -> str:
	value = str(method or "uniform").strip().lower().replace("-", "_").replace(" ", "_")
	if value in {"percentage", "percent", "fraction", "proportion"}:
		return "percentage"
	if value in {"all", "all_spikes", "full"}:
		return "all"
	return "uniform"


def _normalize_requested_random_seed(seed: int | None) -> int | None:
	if seed is None:
		return None
	try:
		return int(seed)
	except Exception:
		return 0


def _normalize_requested_waveform_dtype(dtype: str | None) -> str | None:
	if dtype is None:
		return None
	text = str(dtype).strip()
	return (text or None)


def _normalize_requested_log_counts(value: bool | None) -> bool:
	return bool(value)


def _build_compatibility_extension_params(
	*,
	extension_params: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
	compat_params: dict[str, Any] = {}
	stripped: list[str] = []
	for name, raw_params in extension_params.items():
		if not isinstance(raw_params, dict):
			compat_params[name] = raw_params
			continue
		params = dict(raw_params)
		if name == "random_spikes":
			for key in ("log_before_after_spike_counts", "margin_size"):
				if key in params:
					params.pop(key, None)
					stripped.append(f"{name}.{key}")
		elif name == "waveforms":
			if "dtype" in params:
				params.pop("dtype", None)
				stripped.append(f"{name}.dtype")
		compat_params[name] = params
	return compat_params, tuple(stripped)


def _build_compatibility_compute_kwargs(
	*,
	compute_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
	compat_kwargs = dict(compute_kwargs)
	stripped: list[str] = []
	if "chunk_duration" in compat_kwargs:
		compat_kwargs.pop("chunk_duration", None)
		stripped.append("chunk_duration")
	return compat_kwargs, tuple(stripped)


def _try_recompute_waveforms_extension(
	*,
	analyzer: Any,
	requested_max_spikes_per_unit: int | None,
	requested_min_spikes_per_unit: int | None = None,
	requested_ms_before: float | None,
	requested_ms_after: float | None,
	requested_dtype: str | None = None,
	requested_random_spikes_method: str = "uniform",
	requested_random_spikes_percentage: float | None = None,
	requested_random_seed: int | None = 0,
	requested_log_before_after_spike_counts: bool | None = None,
	requested_margin_size: int | None = None,
	compute_n_jobs: int | None = None,
	compute_chunk_duration: str | None = None,
) -> bool:
	"""Best-effort recompute of random_spikes+waveforms+templates with requested semantics.

	Returns True when recompute appears to run successfully.
	"""
	attempted_attr = "_axon_recon_waveforms_recompute_attempted_params"
	attempt_key = (
		None if requested_max_spikes_per_unit is None else int(requested_max_spikes_per_unit),
		None if requested_min_spikes_per_unit is None else int(requested_min_spikes_per_unit),
		None if requested_ms_before is None else float(requested_ms_before),
		None if requested_ms_after is None else float(requested_ms_after),
		_normalize_requested_waveform_dtype(requested_dtype),
		_normalize_requested_random_spikes_method(requested_random_spikes_method),
		None if requested_random_spikes_percentage is None else float(requested_random_spikes_percentage),
		_normalize_requested_random_seed(requested_random_seed),
		_normalize_requested_log_counts(requested_log_before_after_spike_counts),
		None if requested_margin_size is None else int(requested_margin_size),
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
	method = _normalize_requested_random_spikes_method(requested_random_spikes_method)
	random_spikes_params: dict[str, Any] = {
		"method": method,
	}
	seed = _normalize_requested_random_seed(requested_random_seed)
	if method != "all" and seed is not None:
		random_spikes_params["seed"] = int(seed)
	if requested_margin_size is not None and int(requested_margin_size) > 0:
		random_spikes_params["margin_size"] = int(requested_margin_size)
	if method != "all" and requested_max_spikes_per_unit is not None and int(requested_max_spikes_per_unit) > 0:
		random_spikes_params["max_spikes_per_unit"] = int(requested_max_spikes_per_unit)
	if method == "percentage" and requested_random_spikes_percentage is not None:
		random_spikes_params["percentage"] = float(requested_random_spikes_percentage)
	if method == "percentage" and requested_min_spikes_per_unit is not None and int(requested_min_spikes_per_unit) > 0:
		random_spikes_params["min_spikes_per_unit"] = int(requested_min_spikes_per_unit)
	if _normalize_requested_log_counts(requested_log_before_after_spike_counts):
		random_spikes_params["log_before_after_spike_counts"] = True

	extension_params: dict[str, Any] = {
		"random_spikes": random_spikes_params,
	}
	if ms_before is not None or ms_after is not None:
		wf_params: dict[str, Any] = {}
		if ms_before is not None:
			wf_params["ms_before"] = float(ms_before)
		if ms_after is not None:
			wf_params["ms_after"] = float(ms_after)
		dtype = _normalize_requested_waveform_dtype(requested_dtype)
		if dtype is not None:
			wf_params["dtype"] = dtype
		extension_params["waveforms"] = wf_params
	compute_kwargs: dict[str, Any] = {
		"extension_params": extension_params,
		"verbose": False,
		"n_jobs": (1 if compute_n_jobs is None else int(compute_n_jobs)),
	}
	if compute_chunk_duration not in {None, ""}:
		compute_kwargs["chunk_duration"] = str(compute_chunk_duration)

	compat_extension_params, stripped_extension_keys = _build_compatibility_extension_params(
		extension_params=extension_params,
	)
	compat_compute_kwargs, stripped_compute_keys = _build_compatibility_compute_kwargs(
		compute_kwargs=compute_kwargs,
	)
	compat_stripped_keys = stripped_extension_keys + stripped_compute_keys
	compat_attempt = bool(compat_stripped_keys) and (
		compat_extension_params != extension_params or compat_compute_kwargs != compute_kwargs
	)
	attempts: list[tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]] = [
		(extension_params, compute_kwargs, ()),
	]
	if compat_attempt:
		attempts.append((compat_extension_params, compat_compute_kwargs, compat_stripped_keys))

	for attempt_index, (attempt_extension_params, attempt_compute_kwargs, stripped_keys) in enumerate(attempts):
		try:
			attempt_kwargs = dict(attempt_compute_kwargs)
			attempt_kwargs["extension_params"] = attempt_extension_params
			analyzer.compute(
				["random_spikes", "waveforms", "templates"],
				**attempt_kwargs,
			)
			if attempt_index > 0 and stripped_keys:
				LOGGER.info(
					"Analyzer extension compute succeeded after compatibility fallback: dropped=%s",
					", ".join(str(key) for key in stripped_keys),
				)
			return True
		except Exception:
			if attempt_index + 1 < len(attempts):
				LOGGER.debug(
					"Failed to recompute analyzer extensions with requested options; retrying compatibility fallback",
					exc_info=True,
				)
				continue
			LOGGER.debug("Failed to recompute waveforms extension with requested cap", exc_info=True)
			return False
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
	requested_min_spikes_per_unit: int | None = None,
	requested_ms_before: float | None,
	requested_ms_after: float | None,
	requested_dtype: str | None = None,
	requested_random_spikes_method: str = "uniform",
	requested_random_spikes_percentage: float | None = None,
	requested_random_seed: int | None = 0,
	requested_log_before_after_spike_counts: bool | None = None,
	requested_margin_size: int | None = None,
) -> None:
	try:
		setattr(
			analyzer,
			"_axon_recon_prepared_waveforms_signature",
			(
				_normalize_requested_max_spikes_per_unit(requested_max_spikes_per_unit),
				(None if requested_min_spikes_per_unit is None else int(requested_min_spikes_per_unit)),
				(None if requested_ms_before is None else float(requested_ms_before)),
				(None if requested_ms_after is None else float(requested_ms_after)),
				_normalize_requested_waveform_dtype(requested_dtype),
				_normalize_requested_random_spikes_method(requested_random_spikes_method),
				(None if requested_random_spikes_percentage is None else float(requested_random_spikes_percentage)),
				_normalize_requested_random_seed(requested_random_seed),
				_normalize_requested_log_counts(requested_log_before_after_spike_counts),
				(None if requested_margin_size is None else int(requested_margin_size)),
			),
		)
	except Exception:
		pass


def _waveforms_prepared_matches(
	*,
	analyzer: Any,
	requested_max_spikes_per_unit: int | None,
	requested_min_spikes_per_unit: int | None = None,
	requested_ms_before: float | None,
	requested_ms_after: float | None,
	requested_dtype: str | None = None,
	requested_random_spikes_method: str = "uniform",
	requested_random_spikes_percentage: float | None = None,
	requested_random_seed: int | None = 0,
	requested_log_before_after_spike_counts: bool | None = None,
	requested_margin_size: int | None = None,
) -> bool:
	signature = getattr(analyzer, "_axon_recon_prepared_waveforms_signature", None)
	if not isinstance(signature, tuple):
		return False
	if len(signature) == 10:
		(
			prepared_max,
			prepared_min,
			prepared_ms_before,
			prepared_ms_after,
			prepared_dtype,
			prepared_method,
			prepared_percentage,
			prepared_seed,
			prepared_log_counts,
			prepared_margin_size,
		) = signature
	elif len(signature) == 5:
		prepared_max, prepared_ms_before, prepared_ms_after, prepared_method, prepared_seed = signature
		prepared_min = None
		prepared_dtype = None
		prepared_percentage = None
		prepared_log_counts = False
		prepared_margin_size = None
	elif len(signature) == 3:
		prepared_max, prepared_ms_before, prepared_ms_after = signature
		prepared_min = None
		prepared_dtype = None
		prepared_method = "uniform"
		prepared_seed = 0
		prepared_percentage = None
		prepared_log_counts = False
		prepared_margin_size = None
	else:
		return False
	required_max = _normalize_requested_max_spikes_per_unit(requested_max_spikes_per_unit)
	required_min = (None if requested_min_spikes_per_unit is None else int(requested_min_spikes_per_unit))
	if prepared_ms_before != (None if requested_ms_before is None else float(requested_ms_before)):
		return False
	if prepared_ms_after != (None if requested_ms_after is None else float(requested_ms_after)):
		return False
	if prepared_dtype != _normalize_requested_waveform_dtype(requested_dtype):
		return False
	if prepared_method != _normalize_requested_random_spikes_method(requested_random_spikes_method):
		return False
	if prepared_percentage != (None if requested_random_spikes_percentage is None else float(requested_random_spikes_percentage)):
		return False
	if prepared_seed != _normalize_requested_random_seed(requested_random_seed):
		return False
	if prepared_log_counts != _normalize_requested_log_counts(requested_log_before_after_spike_counts):
		return False
	if prepared_margin_size != (None if requested_margin_size is None else int(requested_margin_size)):
		return False
	if prepared_min != required_min:
		return False
	if prepared_max is None:
		return required_max is None
	if required_max is None:
		return False
	return int(prepared_max) >= int(required_max)


def _prepare_analyzer_for_payload_extraction(
	*,
	analyzer: Any,
	requested_max_spikes_per_unit: int | None,
	requested_min_spikes_per_unit: int | None = None,
	requested_ms_before: float | None,
	requested_ms_after: float | None,
	requested_dtype: str | None = None,
	requested_random_spikes_method: str = "uniform",
	requested_random_spikes_percentage: float | None = None,
	requested_random_seed: int | None = 0,
	requested_log_before_after_spike_counts: bool | None = None,
	requested_margin_size: int | None = None,
	compute_n_jobs: int | None = None,
	compute_chunk_duration: str | None = None,
	log_context: str | None = None,
) -> Any:
	if not hasattr(analyzer, "has_extension") or not hasattr(analyzer, "compute"):
		return analyzer
	normalized_max = _normalize_requested_max_spikes_per_unit(requested_max_spikes_per_unit)
	if _waveforms_prepared_matches(
		analyzer=analyzer,
		requested_max_spikes_per_unit=normalized_max,
		requested_min_spikes_per_unit=requested_min_spikes_per_unit,
		requested_ms_before=requested_ms_before,
		requested_ms_after=requested_ms_after,
		requested_dtype=requested_dtype,
		requested_random_spikes_method=requested_random_spikes_method,
		requested_random_spikes_percentage=requested_random_spikes_percentage,
		requested_random_seed=requested_random_seed,
		requested_log_before_after_spike_counts=requested_log_before_after_spike_counts,
		requested_margin_size=requested_margin_size,
	):
		if log_context is not None:
			LOGGER.info("Analyzer extensions already satisfy requested settings: source=%s", str(log_context))
		return analyzer
	needs_prepare = (
		normalized_max is not None
		or requested_min_spikes_per_unit is not None
		or requested_ms_before is not None
		or requested_ms_after is not None
		or _normalize_requested_waveform_dtype(requested_dtype) is not None
		or _normalize_requested_random_spikes_method(requested_random_spikes_method) != "uniform"
		or requested_random_spikes_percentage is not None
		or _normalize_requested_log_counts(requested_log_before_after_spike_counts)
		or requested_margin_size is not None
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
		if log_context is not None:
			LOGGER.info(
				"Computing analyzer extensions for %s: extensions=random_spikes,waveforms,templates settings=%s",
				str(log_context),
				_format_log_fields(
					{
						"max_spikes_per_unit": normalized_max,
						"min_spikes_per_unit": requested_min_spikes_per_unit,
						"ms_before": requested_ms_before,
						"ms_after": requested_ms_after,
						"dtype": _normalize_requested_waveform_dtype(requested_dtype),
						"random_spikes_method": _normalize_requested_random_spikes_method(requested_random_spikes_method),
						"random_spikes_percentage": requested_random_spikes_percentage,
						"random_seed": _normalize_requested_random_seed(requested_random_seed),
						"log_before_after_spike_counts": _normalize_requested_log_counts(requested_log_before_after_spike_counts),
						"margin_size": requested_margin_size,
						"compute_n_jobs": compute_n_jobs,
						"chunk_duration": compute_chunk_duration,
					},
				),
			)
		recomputed = _try_recompute_waveforms_extension(
			analyzer=analyzer,
			requested_max_spikes_per_unit=normalized_max,
			requested_min_spikes_per_unit=requested_min_spikes_per_unit,
			requested_ms_before=requested_ms_before,
			requested_ms_after=requested_ms_after,
			requested_dtype=requested_dtype,
			requested_random_spikes_method=requested_random_spikes_method,
			requested_random_spikes_percentage=requested_random_spikes_percentage,
			requested_random_seed=requested_random_seed,
			requested_log_before_after_spike_counts=requested_log_before_after_spike_counts,
			requested_margin_size=requested_margin_size,
			compute_n_jobs=compute_n_jobs,
			compute_chunk_duration=compute_chunk_duration,
		)
		if log_context is not None:
			if recomputed:
				LOGGER.info("Analyzer extension compute completed for %s", str(log_context))
			else:
				LOGGER.debug("Analyzer extension compute did not complete for %s", str(log_context))
		_mark_analyzer_waveforms_prepared(
			analyzer=analyzer,
			requested_max_spikes_per_unit=normalized_max,
			requested_min_spikes_per_unit=requested_min_spikes_per_unit,
			requested_ms_before=requested_ms_before,
			requested_ms_after=requested_ms_after,
			requested_dtype=requested_dtype,
			requested_random_spikes_method=requested_random_spikes_method,
			requested_random_spikes_percentage=requested_random_spikes_percentage,
			requested_random_seed=requested_random_seed,
			requested_log_before_after_spike_counts=requested_log_before_after_spike_counts,
			requested_margin_size=requested_margin_size,
		)
	return analyzer


def _load_cached_analyzers(
	*,
	si: Any,
	analyzer_cache_dir: Path | None,
	concat_analyzer_subdir: str = "concat",
	segment_analyzers_subdir: str = "",
) -> dict[str, Any]:
	return _load_cached_analyzers_with_filter(
		si=si,
		analyzer_cache_dir=analyzer_cache_dir,
		concat_analyzer_subdir=concat_analyzer_subdir,
		segment_analyzers_subdir=segment_analyzers_subdir,
		requested_names=None,
	)


def _load_cached_analyzers_with_filter(
	*,
	si: Any,
	analyzer_cache_dir: Path | None,
	concat_analyzer_subdir: str = "concat",
	segment_analyzers_subdir: str = "",
	requested_names: set[str] | None = None,
) -> dict[str, Any]:
	cached_dirs = _discover_cached_analyzer_dirs(
		analyzer_cache_dir=analyzer_cache_dir,
		concat_analyzer_subdir=concat_analyzer_subdir,
		segment_analyzers_subdir=segment_analyzers_subdir,
	)
	if requested_names is not None:
		cached_dirs = {name: path for name, path in cached_dirs.items() if name in requested_names}
	if not cached_dirs:
		return {}
	cached: dict[str, Any] = {}
	for analyzer_name, folder in sorted(cached_dirs.items(), key=lambda item: (0 if item[0] == "concat" else 1, item[0])):
		try:
			cached[str(analyzer_name)] = si.load_sorting_analyzer(folder)
		except Exception:
			if str(analyzer_name) == "concat":
				LOGGER.warning("Failed to load cached concat analyzer: %s", folder, exc_info=True)
			else:
				LOGGER.warning("Failed to load cached segment analyzer: %s", folder, exc_info=True)
	return cached


def discover_cached_spikeinterface_analyzer_source_names(
	*,
	analyzer_cache_dir: Path | None,
	analyzer_cache_concat_subdir: str = "concat",
	analyzer_cache_segments_subdir: str = "",
	include_concat: bool,
	include_segments: bool,
	requested_source_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
	requested_names: set[str] | None = None
	if requested_source_names is not None:
		requested_names = {
			str(name).strip()
			for name in requested_source_names
			if str(name).strip() != ""
		}
		if not requested_names:
			requested_names = None

	cached_dirs = _discover_cached_analyzer_dirs(
		analyzer_cache_dir=analyzer_cache_dir,
		concat_analyzer_subdir=analyzer_cache_concat_subdir,
		segment_analyzers_subdir=analyzer_cache_segments_subdir,
	)
	source_names: list[str] = []
	if bool(include_concat) and "concat" in cached_dirs:
		source_names.append("concat")
	if bool(include_segments):
		for name in sorted(key for key in cached_dirs.keys() if str(key) != "concat"):
			source_names.append(str(name))
	if requested_names is None:
		return source_names
	return [name for name in source_names if name in requested_names]


def load_cached_spikeinterface_analyzers(
	*,
	well_out_dir: Path,
	preprocessed_concat_reldir: str | None = None,
	preprocessed_segments_reldir: str | None = None,
	preproc_seg_sources_reldir: str | None = None,
	analyzer_cache_dir: Path | None,
	analyzer_cache_concat_subdir: str = "concat",
	analyzer_cache_segments_subdir: str = "",
	include_concat: bool,
	include_segments: bool,
	requested_source_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[tuple[str, Any]]:
	import spikeinterface.full as si  # type: ignore[import-not-found]

	requested_names: set[str] | None = None
	if requested_source_names is not None:
		requested_names = {
			str(name).strip()
			for name in requested_source_names
			if str(name).strip() != ""
		}
		if not requested_names:
			requested_names = None
	if requested_names is not None:
		include_concat = bool(include_concat) and ("concat" in requested_names)
		include_segments = bool(include_segments) and any(str(name) != "concat" for name in requested_names)

	def _resolve_from_well(raw_path: str | None) -> Path | None:
		if raw_path is None:
			return None
		token = str(raw_path).strip()
		if not token:
			return None
		path = Path(token).expanduser()
		if path.is_absolute() and path.exists():
			return path
		if path.is_absolute():
			token = token.lstrip("/")
			path = Path(token)
		return (well_out_dir / path).resolve()

	def _load_with_methods(path: Path, method_names: tuple[str, ...]) -> Any | None:
		for name in method_names:
			loader = getattr(si, name, None)
			if not callable(loader):
				continue
			try:
				return loader(path)
			except Exception:
				continue
		return None

	def _attach_temporary_recording_if_missing(*, analyzer: Any, recording: Any | None) -> Any:
		if analyzer is None or recording is None:
			return analyzer
		try:
			if callable(getattr(analyzer, "has_temporary_recording", None)) and bool(analyzer.has_temporary_recording()):
				return analyzer
		except Exception:
			pass
		setter = getattr(analyzer, "set_temporary_recording", None)
		if not callable(setter):
			return analyzer
		try:
			setter(recording)
		except Exception:
			LOGGER.debug("Failed to attach temporary recording to cached templates analyzer", exc_info=True)
		return analyzer

	if analyzer_cache_dir is None or (not analyzer_cache_dir.exists()):
		return []

	segments_sources_reldir = preprocessed_segments_reldir
	if segments_sources_reldir is None:
		segments_sources_reldir = preproc_seg_sources_reldir
	preprocessed_concat_dir = _resolve_from_well(preprocessed_concat_reldir)
	segments_dir = _resolve_from_well(segments_sources_reldir)
	loaded = _load_cached_analyzers_with_filter(
		si=si,
		analyzer_cache_dir=analyzer_cache_dir,
		concat_analyzer_subdir=analyzer_cache_concat_subdir,
		segment_analyzers_subdir=analyzer_cache_segments_subdir,
		requested_names=requested_names,
	)
	analyzers: list[tuple[str, Any]] = []
	if bool(include_concat) and "concat" in loaded:
		recording = None if preprocessed_concat_dir is None else _load_with_methods(preprocessed_concat_dir, ("load_extractor", "load_recording", "load"))
		analyzers.append(("concat", _attach_temporary_recording_if_missing(analyzer=loaded["concat"], recording=recording)))
	if bool(include_segments):
		for source_name in sorted(name for name in loaded.keys() if str(name) != "concat"):
			recording = None
			if segments_dir is not None:
				seg_dir = segments_dir / str(source_name)
				if seg_dir.exists():
					recording = _load_with_methods(seg_dir, ("load_extractor", "load_recording", "load"))
			analyzers.append((str(source_name), _attach_temporary_recording_if_missing(analyzer=loaded[source_name], recording=recording)))
	return analyzers


def discover_spikeinterface_analyzer_source_names(
	*,
	well_out_dir: Path,
	concat_analyzer_relpath: str | None = None,
	concat_sorting_relpath: str | None = None,
	preprocessed_concat_reldir: str | None = None,
	preprocessed_segments_reldir: str | None = None,
	preproc_seg_sources_reldir: str | None = None,
	analyzer_cache_dir: Path | None = None,
	analyzer_cache_concat_subdir: str = "concat",
	analyzer_cache_segments_subdir: str = "",
	include_concat: bool,
	include_segments: bool,
	concat_use_existing_analyzer: bool = True,
	concat_build_if_missing: bool = True,
	segments_use_existing_analyzer: bool = True,
	segments_build_if_missing: bool = True,
	requested_source_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
	requested_names: set[str] | None = None
	if requested_source_names is not None:
		requested_names = {
			str(name).strip()
			for name in requested_source_names
			if str(name).strip() != ""
		}
		if not requested_names:
			requested_names = None

	def _resolve_from_well(raw_path: str | None) -> Path | None:
		if raw_path is None:
			return None
		token = str(raw_path).strip()
		if not token:
			return None
		p = Path(token).expanduser()
		if p.is_absolute() and p.exists():
			return p
		if p.is_absolute():
			token = token.lstrip("/")
			p = Path(token)
		return (well_out_dir / p).resolve()

	wf_out = well_out_dir / "stg3_waveforms_outputs"
	concat_dir = _resolve_from_well(concat_analyzer_relpath) or (wf_out / "concat_waveforms")
	concat_sorting_dir = _resolve_from_well(concat_sorting_relpath)
	preprocessed_concat_dir = _resolve_from_well(preprocessed_concat_reldir)
	segments_sources_reldir = preprocessed_segments_reldir
	if segments_sources_reldir is None:
		segments_sources_reldir = preproc_seg_sources_reldir
	segments_dir = _resolve_from_well(segments_sources_reldir) or (wf_out / "segment_waveforms")
	cache_root = None if analyzer_cache_dir is None else Path(analyzer_cache_dir).expanduser().resolve()
	cached_dirs = _discover_cached_analyzer_dirs(
		analyzer_cache_dir=cache_root,
		concat_analyzer_subdir=str(analyzer_cache_concat_subdir or "concat"),
		segment_analyzers_subdir=str(analyzer_cache_segments_subdir or ""),
	)

	source_names: list[str] = []
	if bool(include_concat):
		has_concat_source = False
		if bool(concat_use_existing_analyzer) and ("concat" in cached_dirs or concat_dir.exists()):
			has_concat_source = True
		elif bool(concat_build_if_missing):
			has_concat_source = (
				concat_sorting_dir is not None
				and preprocessed_concat_dir is not None
				and concat_sorting_dir.exists()
				and preprocessed_concat_dir.exists()
			)
		if has_concat_source:
			source_names.append("concat")

	if bool(include_segments):
		seen_segment_names = set(source_names)
		if segments_dir.exists() and segments_dir.is_dir():
			for seg_dir in sorted(path for path in segments_dir.iterdir() if path.is_dir()):
				seg_name = str(seg_dir.name)
				if seg_name in seen_segment_names:
					continue
				seen_segment_names.add(seg_name)
				source_names.append(seg_name)
		if bool(segments_use_existing_analyzer):
			for cached_name in sorted(name for name in cached_dirs.keys() if str(name) != "concat"):
				if cached_name in seen_segment_names:
					continue
				seen_segment_names.add(cached_name)
				source_names.append(cached_name)
		elif not bool(segments_build_if_missing):
			source_names = [name for name in source_names if str(name) == "concat"]

	if requested_names is None:
		return source_names
	return [name for name in source_names if name in requested_names]


def _persist_analyzer_to_cache(
	*,
	analyzer: Any,
	analyzer_cache_dir: Path | None,
	analyzer_name: str,
	concat_analyzer_subdir: str = "concat",
	segment_analyzers_subdir: str = "",
) -> Any:
	if analyzer_cache_dir is None:
		return analyzer
	if not hasattr(analyzer, "save_as"):
		return analyzer
	concat_subdir = str(concat_analyzer_subdir or "").strip().strip("/")
	segments_subdir = str(segment_analyzers_subdir or "").strip().strip("/")
	if str(analyzer_name) == "concat":
		folder = analyzer_cache_dir / (concat_subdir or "concat")
	else:
		segments_root = analyzer_cache_dir / segments_subdir if segments_subdir else analyzer_cache_dir
		folder = segments_root / str(analyzer_name)
	folder.parent.mkdir(parents=True, exist_ok=True)
	try:
		if folder.exists():
			shutil.rmtree(folder)
		saved = analyzer.save_as(format="binary_folder", folder=folder)
		LOGGER.info("Persisted analyzer cache output: analyzer=%s folder=%s", str(analyzer_name), str(folder))
		return saved
	except Exception:
		LOGGER.warning("Failed to persist analyzer cache for %s at %s", analyzer_name, folder, exc_info=True)
		return analyzer


def _resolve_preparation_policy(
	*,
	policy: AnalyzerPreparationPolicyConfig | None,
	waveform_ms_before: float | None,
	waveform_ms_after: float | None,
	waveform_max_spikes_per_unit: int | None,
) -> AnalyzerPreparationPolicyConfig:
	if policy is not None:
		return policy
	return AnalyzerPreparationPolicyConfig(
		ms_before=waveform_ms_before,
		ms_after=waveform_ms_after,
		max_spikes_per_unit=waveform_max_spikes_per_unit,
		sparsity_mode="inherit",
		compute_sparsity=True,
		sparsity_method="radius",
		sparsity_radius_um=100.0,
		sparsity_num_channels=5,
		sparsity_threshold=5.0,
		sparsity_peak_sign="neg",
		sparsity_num_spikes_for_sparsity=100,
		random_spikes_method="uniform",
		random_seed=0,
	)


def _prepare_loaded_analyzer_with_policy(
	*,
	analyzer: Any,
	policy: AnalyzerPreparationPolicyConfig,
	source_name: str,
) -> Any | None:
	LOGGER.info(
		"Applying analyzer preparation policy: source=%s settings=%s",
		str(source_name),
		_format_policy_for_log(policy),
	)
	prepared = _prepare_analyzer_for_payload_extraction(
		analyzer=analyzer,
		requested_max_spikes_per_unit=policy.max_spikes_per_unit,
		requested_min_spikes_per_unit=policy.min_spikes_per_unit,
		requested_ms_before=policy.ms_before,
		requested_ms_after=policy.ms_after,
		requested_dtype=policy.dtype,
		requested_random_spikes_method=policy.random_spikes_method,
		requested_random_spikes_percentage=policy.random_spikes_percentage,
		requested_random_seed=policy.random_seed,
		requested_log_before_after_spike_counts=policy.log_before_after_spike_counts,
		requested_margin_size=policy.margin_size,
		compute_n_jobs=policy.n_jobs,
		compute_chunk_duration=policy.chunk_duration,
		log_context=str(source_name),
	)
	dense_requested = (not bool(policy.compute_sparsity)) or str(policy.sparsity_mode).strip().lower() == "dense"
	if dense_requested and _analyzer_has_sparsity(prepared):
		LOGGER.info(
			"Analyzer %s is sparse but dense mode was requested; treating it as a cache miss",
			str(source_name),
		)
		return None
	return prepared


def build_unit_source_payload(
	*,
	analyzer: Any,
	unit_id: Any,
	max_spikes_per_unit: int | None = None,
	min_spikes_per_unit: int | None = None,
	waveform_ms_before: float | None = None,
	waveform_ms_after: float | None = None,
	waveform_dtype: str | None = None,
	random_spikes_method: str = "uniform",
	random_spikes_percentage: float | None = None,
	random_seed: int | None = 0,
	log_before_after_spike_counts: bool | None = None,
	margin_size: int | None = None,
	compute_n_jobs: int | None = None,
	compute_chunk_duration: str | None = None,
	include_overlay_waveforms: bool = True,
	allow_prepare: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int, float | None, np.ndarray | None, Any, int | None] | None:
	waveform_count = _extract_total_waveform_count(analyzer=analyzer, unit_id=unit_id)
	requested_waveforms = _normalize_requested_max_spikes_per_unit(max_spikes_per_unit)
	if allow_prepare and not _waveforms_prepared_matches(
		analyzer=analyzer,
		requested_max_spikes_per_unit=requested_waveforms,
		requested_min_spikes_per_unit=min_spikes_per_unit,
		requested_ms_before=waveform_ms_before,
		requested_ms_after=waveform_ms_after,
		requested_dtype=waveform_dtype,
		requested_random_spikes_method=random_spikes_method,
		requested_random_spikes_percentage=random_spikes_percentage,
		requested_random_seed=random_seed,
		requested_log_before_after_spike_counts=log_before_after_spike_counts,
		requested_margin_size=margin_size,
	):
		_prepare_analyzer_for_payload_extraction(
			analyzer=analyzer,
			requested_max_spikes_per_unit=requested_waveforms,
			requested_min_spikes_per_unit=min_spikes_per_unit,
			requested_ms_before=waveform_ms_before,
			requested_ms_after=waveform_ms_after,
			requested_dtype=waveform_dtype,
			requested_random_spikes_method=random_spikes_method,
			requested_random_spikes_percentage=random_spikes_percentage,
			requested_random_seed=random_seed,
			requested_log_before_after_spike_counts=log_before_after_spike_counts,
			requested_margin_size=margin_size,
			compute_n_jobs=compute_n_jobs,
			compute_chunk_duration=compute_chunk_duration,
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
	if include_overlay_waveforms:
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
				if allow_prepare and requested_waveforms is None and int(wf_all.shape[0]) < int(waveform_count):
					if not _waveforms_prepared_matches(
						analyzer=analyzer,
						requested_max_spikes_per_unit=None,
						requested_min_spikes_per_unit=min_spikes_per_unit,
						requested_ms_before=waveform_ms_before,
						requested_ms_after=waveform_ms_after,
						requested_dtype=waveform_dtype,
						requested_random_spikes_method=random_spikes_method,
						requested_random_spikes_percentage=random_spikes_percentage,
						requested_random_seed=random_seed,
						requested_log_before_after_spike_counts=log_before_after_spike_counts,
						requested_margin_size=margin_size,
					):
						if _try_recompute_waveforms_extension(
							analyzer=analyzer,
							requested_max_spikes_per_unit=None,
							requested_min_spikes_per_unit=min_spikes_per_unit,
							requested_ms_before=waveform_ms_before,
							requested_ms_after=waveform_ms_after,
							requested_dtype=waveform_dtype,
							requested_random_spikes_method=random_spikes_method,
							requested_random_spikes_percentage=random_spikes_percentage,
							requested_random_seed=random_seed,
							requested_log_before_after_spike_counts=log_before_after_spike_counts,
							requested_margin_size=margin_size,
							compute_n_jobs=compute_n_jobs,
							compute_chunk_duration=compute_chunk_duration,
						):
							_mark_analyzer_waveforms_prepared(
								analyzer=analyzer,
								requested_max_spikes_per_unit=None,
								requested_min_spikes_per_unit=min_spikes_per_unit,
								requested_ms_before=waveform_ms_before,
								requested_ms_after=waveform_ms_after,
								requested_dtype=waveform_dtype,
								requested_random_spikes_method=random_spikes_method,
								requested_random_spikes_percentage=random_spikes_percentage,
								requested_random_seed=random_seed,
								requested_log_before_after_spike_counts=log_before_after_spike_counts,
								requested_margin_size=margin_size,
							)
							wf_ext = analyzer.get_extension("waveforms")
							wf_all = np.asarray(wf_ext.get_waveforms_one_unit(unit_id=unit_id, force_dense=False), dtype=float)
				if allow_prepare and requested_waveforms is not None and int(wf_all.shape[0]) < int(need_waveforms):
					if _try_recompute_waveforms_extension(
						analyzer=analyzer,
						requested_max_spikes_per_unit=int(need_waveforms),
						requested_min_spikes_per_unit=min_spikes_per_unit,
						requested_ms_before=waveform_ms_before,
						requested_ms_after=waveform_ms_after,
						requested_dtype=waveform_dtype,
						requested_random_spikes_method=random_spikes_method,
						requested_random_spikes_percentage=random_spikes_percentage,
						requested_random_seed=random_seed,
						requested_log_before_after_spike_counts=log_before_after_spike_counts,
						requested_margin_size=margin_size,
						compute_n_jobs=compute_n_jobs,
						compute_chunk_duration=compute_chunk_duration,
					):
						_mark_analyzer_waveforms_prepared(
							analyzer=analyzer,
							requested_max_spikes_per_unit=int(need_waveforms),
							requested_min_spikes_per_unit=min_spikes_per_unit,
							requested_ms_before=waveform_ms_before,
							requested_ms_after=waveform_ms_after,
							requested_dtype=waveform_dtype,
							requested_random_spikes_method=random_spikes_method,
							requested_random_spikes_percentage=random_spikes_percentage,
							requested_random_seed=random_seed,
							requested_log_before_after_spike_counts=log_before_after_spike_counts,
							requested_margin_size=margin_size,
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


def _discover_cached_analyzer_dirs(
	*,
	analyzer_cache_dir: Path | None,
	concat_analyzer_subdir: str = "concat",
	segment_analyzers_subdir: str = "",
) -> dict[str, Path]:
	if analyzer_cache_dir is None or (not analyzer_cache_dir.exists()):
		return {}
	discovered: dict[str, Path] = {}
	concat_subdir = str(concat_analyzer_subdir or "").strip().strip("/")
	segments_subdir = str(segment_analyzers_subdir or "").strip().strip("/")
	concat_folder = analyzer_cache_dir / (concat_subdir or "concat")
	concat_folder_resolved: Path | None = None
	segments_container_resolved: Path | None = None
	if concat_folder.exists() and concat_folder.is_dir():
		discovered["concat"] = concat_folder
		try:
			concat_folder_resolved = concat_folder.resolve()
		except Exception:
			concat_folder_resolved = concat_folder

	segment_roots: list[Path] = []
	if segments_subdir:
		segments_container = analyzer_cache_dir / segments_subdir
		segment_roots.append(segments_container)
		try:
			segments_container_resolved = segments_container.resolve()
		except Exception:
			segments_container_resolved = segments_container
		segment_roots.append(analyzer_cache_dir)
	else:
		segment_roots.append(analyzer_cache_dir)

	seen_segment_roots: set[Path] = set()
	for segments_root in segment_roots:
		try:
			segments_root_resolved = segments_root.resolve()
		except Exception:
			segments_root_resolved = segments_root
		if segments_root_resolved in seen_segment_roots:
			continue
		seen_segment_roots.add(segments_root_resolved)
		if not segments_root.exists() or (not segments_root.is_dir()):
			continue
		for folder in sorted(p for p in segments_root.iterdir() if p.is_dir()):
			try:
				folder_resolved = folder.resolve()
			except Exception:
				folder_resolved = folder
			if concat_folder_resolved is not None and folder_resolved == concat_folder_resolved:
				continue
			if segments_container_resolved is not None and folder_resolved == segments_container_resolved:
				continue
			discovered.setdefault(str(folder.name), folder)
	return discovered


def load_spikeinterface_analyzers(
	*,
	well_out_dir: Path,
	concat_analyzer_relpath: str | None = None,
	concat_sorting_relpath: str | None = None,
	preprocessed_concat_reldir: str | None = None,
	preprocessed_segments_reldir: str | None = None,
	preproc_seg_sources_reldir: str | None = None,
	analyzer_cache_dir: Path | None = None,
	analyzer_cache_concat_subdir: str = "concat",
	analyzer_cache_segments_subdir: str = "",
	alternate_well_out_dirs: list[Path] | tuple[Path, ...] | None = None,
	stream_id: str | None = None,
	include_concat: bool,
	include_segments: bool,
	require_concat: bool = False,
	require_segments: bool = False,
	waveform_ms_before: float | None = None,
	waveform_ms_after: float | None = None,
	waveform_max_spikes_per_unit: int | None = None,
	concat_policy: AnalyzerPreparationPolicyConfig | None = None,
	segments_policy: AnalyzerPreparationPolicyConfig | None = None,
	concat_use_existing_analyzer: bool = True,
	concat_build_if_missing: bool = True,
	segments_use_existing_analyzer: bool = True,
	segments_build_if_missing: bool = True,
	requested_source_names: list[str] | tuple[str, ...] | set[str] | None = None,
	return_stats: bool = False,
) -> Any:
	import spikeinterface.full as si  # type: ignore[import-not-found]
	requested_names: set[str] | None = None
	if requested_source_names is not None:
		requested_names = {
			str(name).strip()
			for name in requested_source_names
			if str(name).strip() != ""
		}
		if not requested_names:
			requested_names = None
	if requested_names is not None:
		include_concat = bool(include_concat) and ("concat" in requested_names)
		include_segments = bool(include_segments) and any(str(name) != "concat" for name in requested_names)
	load_started = perf_counter()
	concat_policy_resolved = _resolve_preparation_policy(
		policy=concat_policy,
		waveform_ms_before=waveform_ms_before,
		waveform_ms_after=waveform_ms_after,
		waveform_max_spikes_per_unit=waveform_max_spikes_per_unit,
	)
	load_stats: dict[str, Any] = {
		"well_out_dir": str(well_out_dir),
		"cache_root": (None if analyzer_cache_dir is None else str(Path(analyzer_cache_dir).expanduser())),
		"concat": {
			"requested": bool(include_concat),
			"source": None,
			"recording_attached": False,
			"sorting_loaded_for_build": False,
			"recording_loaded_for_build": False,
		},
		"segments": {
			"requested": bool(include_segments),
			"recordings_source_dir": None,
			"recordings_discovered": 0,
			"recordings_loaded": 0,
			"cache_hits": 0,
			"disk_hits": 0,
			"queued_for_build": 0,
			"built": 0,
			"recordings_attached": 0,
		},
		"cache": {
			"concat_loaded": False,
			"segments_loaded": 0,
			"persisted": 0,
		},
	}

	def _finalize_result(result: list[tuple[str, Any]]) -> Any:
		load_stats["duration_seconds"] = float(perf_counter() - load_started)
		load_stats["source_count"] = int(len(result))
		load_stats["concat_count"] = int(sum(1 for name, _ in result if str(name) == "concat"))
		load_stats["segment_count"] = int(sum(1 for name, _ in result if str(name) != "concat"))
		if return_stats:
			return result, load_stats
		return result

	segments_policy_resolved = _resolve_preparation_policy(
		policy=segments_policy,
		waveform_ms_before=waveform_ms_before,
		waveform_ms_after=waveform_ms_after,
		waveform_max_spikes_per_unit=waveform_max_spikes_per_unit,
	)

	LOGGER.info(
		"Resolving SpikeInterface analyzers: well_out_dir=%s include_concat=%s include_segments=%s require_concat=%s require_segments=%s cache_root=%s",
		str(well_out_dir),
		bool(include_concat),
		bool(include_segments),
		bool(require_concat),
		bool(require_segments),
		load_stats["cache_root"],
	)
	LOGGER.info("Concat analyzer policy: %s", _format_policy_for_log(concat_policy_resolved))
	LOGGER.info("Segment analyzer policy: %s", _format_policy_for_log(segments_policy_resolved))

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

	def _load_with_methods(path: Path, method_names: tuple[str, ...]) -> Any | None:
		for name in method_names:
			loader = getattr(si, name, None)
			if not callable(loader):
				continue
			try:
				return loader(path)
			except Exception:
				continue
		return None

	wf_out = well_out_dir / "stg3_waveforms_outputs"
	concat_dir = _resolve_from_well(concat_analyzer_relpath) or (wf_out / "concat_waveforms")
	concat_sorting_dir = _resolve_from_well(concat_sorting_relpath)
	preprocessed_concat_dir = _resolve_from_well(preprocessed_concat_reldir)
	segments_sources_reldir = preprocessed_segments_reldir
	if segments_sources_reldir is None:
		segments_sources_reldir = preproc_seg_sources_reldir
	segments_dir = _resolve_from_well(segments_sources_reldir) or (wf_out / "segment_waveforms")
	load_stats["segments"]["recordings_source_dir"] = str(segments_dir)
	cache_root = None if analyzer_cache_dir is None else Path(analyzer_cache_dir).expanduser().resolve()
	recording_cache: dict[Path, Any | None] = {}
	cached_analyzer_dirs = _discover_cached_analyzer_dirs(
		analyzer_cache_dir=cache_root,
		concat_analyzer_subdir=str(analyzer_cache_concat_subdir or "concat"),
		segment_analyzers_subdir=str(analyzer_cache_segments_subdir or ""),
	)
	cached_analyzers = _load_cached_analyzers_with_filter(
		si=si,
		analyzer_cache_dir=cache_root,
		concat_analyzer_subdir=str(analyzer_cache_concat_subdir or "concat"),
		segment_analyzers_subdir=str(analyzer_cache_segments_subdir or ""),
		requested_names=requested_names,
	)
	if cache_root is None:
		LOGGER.info("Templates analyzer cache disabled for this load")
	elif not cache_root.exists():
		LOGGER.info("Templates analyzer cache directory not found: %s", str(cache_root))
	else:
		LOGGER.info(
			"Loaded templates analyzer cache inventory: concat_cached=%s segment_cached_count=%d cache_root=%s",
			("concat" in cached_analyzer_dirs),
			int(len([name for name in cached_analyzer_dirs.keys() if str(name) != "concat"])),
			str(cache_root),
		)

	def _cache_folder_for_name(analyzer_name: str) -> Path | None:
		if cache_root is None:
			return None
		concat_subdir = str(analyzer_cache_concat_subdir or "").strip().strip("/")
		segments_subdir = str(analyzer_cache_segments_subdir or "").strip().strip("/")
		if str(analyzer_name) == "concat":
			return cache_root / (concat_subdir or "concat")
		segments_subdir = str(analyzer_cache_segments_subdir or "").strip().strip("/")
		segments_root = cache_root / segments_subdir if segments_subdir else cache_root
		return segments_root / str(analyzer_name)

	def _persist_to_cache(*, analyzer: Any, analyzer_name: str) -> Any:
		persisted = _persist_analyzer_to_cache(
			analyzer=analyzer,
			analyzer_cache_dir=cache_root,
			analyzer_name=analyzer_name,
			concat_analyzer_subdir=str(analyzer_cache_concat_subdir or "concat"),
			segment_analyzers_subdir=str(analyzer_cache_segments_subdir or ""),
		)
		folder = _cache_folder_for_name(analyzer_name)
		if folder is not None and folder.exists():
			load_stats["cache"]["persisted"] = int(load_stats["cache"]["persisted"]) + 1
		return persisted

	def _load_recording(path: Path | None) -> Any | None:
		if path is None or (not path.exists()):
			return None
		try:
			resolved = path.resolve()
		except Exception:
			resolved = path
		if resolved not in recording_cache:
			recording_cache[resolved] = _load_with_methods(path, ("load_extractor", "load_recording", "load"))
		return recording_cache[resolved]

	def _attach_temporary_recording_if_missing(*, analyzer: Any, recording: Any | None, source_name: str) -> Any:
		if analyzer is None or recording is None:
			return analyzer
		try:
			if callable(getattr(analyzer, "has_temporary_recording", None)) and bool(analyzer.has_temporary_recording()):
				return analyzer
		except Exception:
			pass
		setter = getattr(analyzer, "set_temporary_recording", None)
		if not callable(setter):
			return analyzer
		try:
			setter(recording)
			LOGGER.info("Attached preprocessed recording to analyzer: source=%s", str(source_name))
			if str(source_name) == "concat":
				load_stats["concat"]["recording_attached"] = True
			else:
				load_stats["segments"]["recordings_attached"] = int(load_stats["segments"]["recordings_attached"]) + 1
		except Exception:
			LOGGER.warning(
				"Failed attaching preprocessed recording for analyzer %s",
				str(source_name),
				exc_info=True,
			)
		return analyzer

	def _build_concat_analyzer_from_sorting_and_recording() -> Any | None:
		LOGGER.info(
			"Preparing concat analyzer build: sorting_dir=%s recording_dir=%s use_existing=%s build_if_missing=%s settings=%s",
			str(concat_sorting_dir),
			str(preprocessed_concat_dir),
			bool(concat_use_existing_analyzer),
			bool(concat_build_if_missing),
			_format_policy_for_log(concat_policy_resolved),
		)
		if concat_sorting_dir is None or preprocessed_concat_dir is None:
			LOGGER.info(
				"Concat analyzer build unavailable: concat sorting or preprocessed recording path is not configured"
			)
			return None
		if not concat_sorting_dir.exists():
			LOGGER.info("Concat analyzer build unavailable: concat sorting path missing: %s", str(concat_sorting_dir))
			return None
		if not preprocessed_concat_dir.exists():
			LOGGER.info(
				"Concat analyzer build unavailable: preprocessed concat recording path missing: %s",
				str(preprocessed_concat_dir),
			)
			return None

		LOGGER.info("Loading concat sorting for analyzer build: %s", str(concat_sorting_dir))
		concat_sorting = _load_with_methods(concat_sorting_dir, ("load_sorting", "load_extractor", "load"))
		if concat_sorting is None:
			LOGGER.warning("Failed to load concat sorting for fallback build: %s", str(concat_sorting_dir))
			return None
		LOGGER.info("Loaded concat sorting for analyzer build: %s", str(concat_sorting_dir))
		load_stats["concat"]["sorting_loaded_for_build"] = True

		LOGGER.info("Loading preprocessed concat recording for analyzer build: %s", str(preprocessed_concat_dir))
		concat_recording = _load_with_methods(preprocessed_concat_dir, ("load_extractor", "load_recording", "load"))
		if concat_recording is None:
			LOGGER.warning("Failed to load preprocessed concat recording for fallback build: %s", str(preprocessed_concat_dir))
			return None
		LOGGER.info("Loaded preprocessed concat recording for analyzer build: %s", str(preprocessed_concat_dir))
		load_stats["concat"]["recording_loaded_for_build"] = True

		create_sorting_analyzer = getattr(si, "create_sorting_analyzer", None)
		if not callable(create_sorting_analyzer):
			LOGGER.warning("Concat analyzer build fallback unavailable: spikeinterface.create_sorting_analyzer missing")
			return None

		try:
			create_kwargs = _sorting_analyzer_create_kwargs_from_policy(policy=concat_policy_resolved)
			LOGGER.info("Generating concat analyzer: creation=%s", _format_log_fields(create_kwargs))
			built = create_sorting_analyzer(
				concat_sorting,
				concat_recording,
				**create_kwargs,
			)
			built = _persist_to_cache(analyzer=built, analyzer_name="concat")
			built = _prepare_loaded_analyzer_with_policy(
				analyzer=built,
				policy=concat_policy_resolved,
				source_name="concat",
			)
			if built is None:
				return None
			load_stats["concat"]["source"] = "rebuild"
			LOGGER.info(
				"Built concat analyzer from sorting+recording fallback: sorting=%s recording=%s",
				str(concat_sorting_dir),
				str(preprocessed_concat_dir),
			)
			return built
		except Exception:
			LOGGER.warning(
				"Failed concat analyzer fallback build from sorting=%s recording=%s",
				str(concat_sorting_dir),
				str(preprocessed_concat_dir),
				exc_info=True,
			)
			return None

	analyzers: list[tuple[str, Any]] = []
	concat_analyzer_obj: Any | None = None
	concat_selection_logged = False

	def _ensure_concat_analyzer_loaded(*, register_requested_source: bool) -> Any | None:
		nonlocal concat_analyzer_obj, concat_selection_logged
		if not concat_selection_logged:
			LOGGER.info(
				"Concat analyzer selection: use_existing=%s build_if_missing=%s analyzer_dir=%s sorting_dir=%s recording_dir=%s",
				bool(concat_use_existing_analyzer),
				bool(concat_build_if_missing),
				str(concat_dir),
				str(concat_sorting_dir),
				str(preprocessed_concat_dir),
			)
			if not bool(concat_use_existing_analyzer):
				LOGGER.info("Concat analyzer reuse disabled; skipping existing concat analyzer load")
			concat_selection_logged = True

		if concat_use_existing_analyzer and concat_analyzer_obj is None and "concat" in cached_analyzers:
			try:
				LOGGER.info("Loading concat analyzer from templates cache")
				concat_analyzer_obj = _attach_temporary_recording_if_missing(
					analyzer=cached_analyzers["concat"],
					recording=_load_recording(preprocessed_concat_dir),
					source_name="concat",
				)
				concat_analyzer_obj = _prepare_loaded_analyzer_with_policy(
					analyzer=concat_analyzer_obj,
					policy=concat_policy_resolved,
					source_name="concat",
				)
				if concat_analyzer_obj is not None:
					load_stats["concat"]["source"] = "cache"
					load_stats["cache"]["concat_loaded"] = True
					LOGGER.info("Loaded concat analyzer from templates cache")
			except Exception:
				concat_cache_hint = (
					str(cache_root / (str(analyzer_cache_concat_subdir or "concat").strip().strip("/") or "concat"))
					if cache_root is not None
					else "<cache_root:None>"
				)
				LOGGER.warning("Failed to prepare cached concat analyzer: %s", concat_cache_hint, exc_info=True)
				concat_analyzer_obj = None
		if concat_use_existing_analyzer and concat_analyzer_obj is None and concat_dir.exists():
			try:
				LOGGER.info("Loading existing concat analyzer from disk: %s", str(concat_dir))
				concat_analyzer_obj = si.load_sorting_analyzer(concat_dir)
				concat_analyzer_obj = _attach_temporary_recording_if_missing(
					analyzer=concat_analyzer_obj,
					recording=_load_recording(preprocessed_concat_dir),
					source_name="concat",
				)
				concat_analyzer_obj = _persist_to_cache(analyzer=concat_analyzer_obj, analyzer_name="concat")
				concat_analyzer_obj = _prepare_loaded_analyzer_with_policy(
					analyzer=concat_analyzer_obj,
					policy=concat_policy_resolved,
					source_name="concat",
				)
				if concat_analyzer_obj is not None:
					load_stats["concat"]["source"] = "disk"
					LOGGER.info("Loaded existing concat analyzer from disk: %s", str(concat_dir))
			except Exception:
				LOGGER.warning("Failed to load concat analyzer: %s", concat_dir)
				concat_analyzer_obj = None
		if register_requested_source and concat_use_existing_analyzer and concat_analyzer_obj is None and ("concat" not in cached_analyzers) and (not concat_dir.exists()):
			LOGGER.info("Existing concat analyzer not found on disk: %s", concat_dir)
		if concat_analyzer_obj is None and concat_build_if_missing:
			concat_analyzer_obj = _build_concat_analyzer_from_sorting_and_recording()
		if register_requested_source and concat_analyzer_obj is not None and not any(name == "concat" for name, _ in analyzers):
			analyzers.append(("concat", concat_analyzer_obj))
		return concat_analyzer_obj

	if include_concat:
		_ensure_concat_analyzer_loaded(register_requested_source=True)

	if include_segments and segments_dir.exists():
		seg_dirs = sorted([p for p in segments_dir.iterdir() if p.is_dir()])
		if requested_names is not None:
			seg_dirs = [p for p in seg_dirs if str(p.name) in requested_names]
		load_stats["segments"]["recordings_discovered"] = int(len(seg_dirs))
		LOGGER.info(
			"Discovered preprocessed segment recording sources: count=%d source_dir=%s",
			int(len(seg_dirs)),
			str(segments_dir),
		)
		seg_dir_by_name = {str(p.name): p for p in seg_dirs}
		segment_names: list[str] = []
		for name in list(seg_dir_by_name.keys()) + sorted(k for k in cached_analyzers.keys() if k != "concat"):
			if requested_names is not None and str(name) not in requested_names:
				continue
			if name not in segment_names:
				segment_names.append(name)
		unloadable_seg_dirs: list[Path] = []
		queued_seg_names: set[str] = set()
		for seg_name in segment_names:
			seg_dir = seg_dir_by_name.get(seg_name, None)
			seg_recording = None
			if seg_dir is not None:
				seg_recording = _load_recording(seg_dir)
				if seg_recording is not None:
					load_stats["segments"]["recordings_loaded"] = int(load_stats["segments"]["recordings_loaded"]) + 1
					LOGGER.info(
						"Loaded preprocessed segment recording source: segment=%s path=%s",
						str(seg_name),
						str(seg_dir),
					)
			if segments_use_existing_analyzer and seg_name in cached_analyzers:
				try:
					LOGGER.info("Loading cached segment analyzer: segment=%s", str(seg_name))
					seg_analyzer = _attach_temporary_recording_if_missing(
						analyzer=cached_analyzers[seg_name],
						recording=seg_recording,
						source_name=seg_name,
					)
					seg_analyzer = _prepare_loaded_analyzer_with_policy(
						analyzer=seg_analyzer,
						policy=segments_policy_resolved,
						source_name=seg_name,
					)
					if seg_analyzer is not None:
						analyzers.append((seg_name, seg_analyzer))
						load_stats["segments"]["cache_hits"] = int(load_stats["segments"]["cache_hits"]) + 1
						load_stats["cache"]["segments_loaded"] = int(load_stats["cache"]["segments_loaded"]) + 1
						LOGGER.info("Loaded cached segment analyzer: segment=%s", str(seg_name))
						continue
					if segments_build_if_missing:
						if seg_dir is not None and seg_name not in queued_seg_names:
							unloadable_seg_dirs.append(seg_dir)
							queued_seg_names.add(seg_name)
							load_stats["segments"]["queued_for_build"] = int(load_stats["segments"]["queued_for_build"]) + 1
						continue
				except Exception:
					segment_cache_hint = (
						str(
							(
								(cache_root / str(analyzer_cache_segments_subdir).strip().strip("/"))
								if str(analyzer_cache_segments_subdir or "").strip().strip("/")
								else cache_root
							)
							/ str(seg_name)
						)
						if cache_root is not None
						else "<cache_root:None>"
					)
					LOGGER.warning("Failed to prepare cached segment analyzer: %s", segment_cache_hint, exc_info=True)
			if seg_dir is None:
				continue
			if not segments_use_existing_analyzer:
				if segments_build_if_missing and seg_name not in queued_seg_names:
					unloadable_seg_dirs.append(seg_dir)
					queued_seg_names.add(seg_name)
					load_stats["segments"]["queued_for_build"] = int(load_stats["segments"]["queued_for_build"]) + 1
					LOGGER.info(
						"Segment analyzer reuse disabled; queued local build from preprocessed recording: segment=%s path=%s",
						str(seg_name),
						str(seg_dir),
					)
				continue
			if seg_recording is not None:
				if segments_build_if_missing and seg_name not in queued_seg_names:
					unloadable_seg_dirs.append(seg_dir)
					queued_seg_names.add(seg_name)
					load_stats["segments"]["queued_for_build"] = int(load_stats["segments"]["queued_for_build"]) + 1
					LOGGER.info(
						"Queued local segment analyzer build from preprocessed recording: segment=%s path=%s",
						str(seg_name),
						str(seg_dir),
					)
				continue
			try:
				LOGGER.info("Attempting existing segment analyzer load from disk: segment=%s path=%s", str(seg_name), str(seg_dir))
				seg_analyzer = si.load_sorting_analyzer(seg_dir)
				seg_analyzer = _persist_to_cache(analyzer=seg_analyzer, analyzer_name=seg_name)
				seg_analyzer = _prepare_loaded_analyzer_with_policy(
					analyzer=seg_analyzer,
					policy=segments_policy_resolved,
					source_name=seg_name,
				)
				if seg_analyzer is not None:
					analyzers.append((seg_name, seg_analyzer))
					load_stats["segments"]["disk_hits"] = int(load_stats["segments"]["disk_hits"]) + 1
					LOGGER.info("Loaded existing segment analyzer from disk: segment=%s path=%s", str(seg_name), str(seg_dir))
				elif segments_build_if_missing and seg_name not in queued_seg_names:
					unloadable_seg_dirs.append(seg_dir)
					queued_seg_names.add(seg_name)
					load_stats["segments"]["queued_for_build"] = int(load_stats["segments"]["queued_for_build"]) + 1
			except Exception:
				if segments_build_if_missing and seg_name not in queued_seg_names:
					unloadable_seg_dirs.append(seg_dir)
					queued_seg_names.add(seg_name)
					load_stats["segments"]["queued_for_build"] = int(load_stats["segments"]["queued_for_build"]) + 1
				LOGGER.info(
					"Segment source path is not an existing analyzer; will use local build path if possible: segment=%s path=%s",
					str(seg_name),
					str(seg_dir),
				)

		if segments_build_if_missing and unloadable_seg_dirs and concat_analyzer_obj is None:
			_ensure_concat_analyzer_loaded(register_requested_source=False)

		if segments_build_if_missing and unloadable_seg_dirs and concat_analyzer_obj is not None:
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

		if segments_build_if_missing and unloadable_seg_dirs and concat_analyzer_obj is not None:
			LOGGER.info(
				"Generating segment analyzers from preprocessed recordings: count=%d source_dir=%s settings=%s",
				int(len(unloadable_seg_dirs)),
				str(segments_dir),
				_format_policy_for_log(segments_policy_resolved),
			)
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
					policy=segments_policy_resolved,
					concat_analyzer=concat_analyzer_obj,
					seg_dir=seg_dir,
					seg_name=seg_name,
					seg_index=int(seg_index),
					rec_name=rec_name,
					start_sample=start_sample,
					end_sample=end_sample,
				)
				if built is not None:
					built = _persist_to_cache(analyzer=built, analyzer_name=seg_name)
					built = _prepare_loaded_analyzer_with_policy(
						analyzer=built,
						policy=segments_policy_resolved,
						source_name=seg_name,
					)
					if built is not None:
						analyzers.append((seg_name, built))
						built_count += 1
						load_stats["segments"]["built"] = int(load_stats["segments"]["built"]) + 1

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
		LOGGER.info("Preprocessed segment recordings directory not found: %s", segments_dir)

	has_concat = any(name == "concat" for name, _ in analyzers)
	has_segments = any(name != "concat" for name, _ in analyzers)
	requirements_unmet = (
		(bool(require_concat) and bool(include_concat) and (not has_concat))
		or (bool(require_segments) and bool(include_segments) and (not has_segments))
	)
	if requirements_unmet:
		LOGGER.warning(
			"Required analyzer sources missing after load/build: require_concat=%s require_segments=%s has_concat=%s has_segments=%s",
			bool(require_concat),
			bool(require_segments),
			bool(has_concat),
			bool(has_segments),
		)

	if (not analyzers) or requirements_unmet:
		fallback_well_out_dirs: list[Path] = []
		seen_fallbacks: set[Path] = set()
		for candidate in list(alternate_well_out_dirs or []):
			try:
				candidate_path = Path(candidate).expanduser().resolve()
			except Exception:
				continue
			if candidate_path == well_out_dir.resolve():
				continue
			if candidate_path in seen_fallbacks:
				continue
			seen_fallbacks.add(candidate_path)
			fallback_well_out_dirs.append(candidate_path)

		for fallback_well_out_dir in fallback_well_out_dirs:
			LOGGER.info(
				"No SpikeInterface analyzers found under %s; retrying with fallback well output root %s",
				str(well_out_dir),
				str(fallback_well_out_dir),
			)
			try:
				return load_spikeinterface_analyzers(
					well_out_dir=fallback_well_out_dir,
					concat_analyzer_relpath=concat_analyzer_relpath,
					concat_sorting_relpath=concat_sorting_relpath,
					preprocessed_concat_reldir=preprocessed_concat_reldir,
					preprocessed_segments_reldir=preprocessed_segments_reldir,
					preproc_seg_sources_reldir=preproc_seg_sources_reldir,
					analyzer_cache_dir=analyzer_cache_dir,
					analyzer_cache_concat_subdir=analyzer_cache_concat_subdir,
					analyzer_cache_segments_subdir=analyzer_cache_segments_subdir,
					alternate_well_out_dirs=None,
					stream_id=stream_id,
					include_concat=include_concat,
					include_segments=include_segments,
					require_concat=require_concat,
					require_segments=require_segments,
					waveform_ms_before=waveform_ms_before,
					waveform_ms_after=waveform_ms_after,
					waveform_max_spikes_per_unit=waveform_max_spikes_per_unit,
					concat_policy=concat_policy_resolved,
					segments_policy=segments_policy_resolved,
					concat_use_existing_analyzer=concat_use_existing_analyzer,
					concat_build_if_missing=concat_build_if_missing,
					segments_use_existing_analyzer=segments_use_existing_analyzer,
					segments_build_if_missing=segments_build_if_missing,
					requested_source_names=(None if requested_names is None else list(requested_names)),
					return_stats=return_stats,
				)
			except FileNotFoundError:
				LOGGER.info("Fallback well output root had no analyzers: %s", str(fallback_well_out_dir))
				continue

		fallback_text = ", ".join(str(path) for path in fallback_well_out_dirs)
		raise FileNotFoundError(
			"No SpikeInterface analyzers found for templates materialization. "
			f"checked concat={concat_dir} segments={segments_dir}"
			f" sorting={concat_sorting_dir} preprocessed_concat={preprocessed_concat_dir}"
			f" require_concat={bool(require_concat)} require_segments={bool(require_segments)}"
			f"{'; fallback_well_out_dirs=[' + fallback_text + ']' if fallback_text else ''}."
		)
	LOGGER.info(
		"Loaded SpikeInterface analyzers from concat=%s segments=%s count=%d include_concat=%s include_segments=%s duration_seconds=%.3f",
		str(concat_dir),
		str(segments_dir),
		len(analyzers),
		bool(include_concat),
		bool(include_segments),
		float(perf_counter() - load_started),
	)
	LOGGER.info("SpikeInterface analyzer load stats: %s", load_stats)
	return _finalize_result(analyzers)
