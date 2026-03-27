from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from ..core.source_payloads import normalize_source_payload

LOGGER = logging.getLogger("axon_recon.templates.spikeinterface")


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
	"""Best-effort recompute of random_spikes+waveforms with requested cap semantics.

	Returns True when recompute appears to run successfully.
	"""
	attempted_attr = "_axon_recon_waveforms_recompute_attempted"
	if bool(getattr(analyzer, attempted_attr, False)):
		return False
	setattr(analyzer, attempted_attr, True)

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
			["random_spikes", "waveforms"],
			extension_params=extension_params,
			verbose=False,
			n_jobs=1,
		)
		return True
	except Exception:
		LOGGER.debug("Failed to recompute waveforms extension with requested cap", exc_info=True)
		return False


def build_unit_source_payload(
	*,
	analyzer: Any,
	unit_id: Any,
	max_spikes_per_unit: int | None = None,
	waveform_ms_before: float | None = None,
	waveform_ms_after: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int, float | None, np.ndarray | None, Any, int | None] | None:
	t = _extract_unit_template(analyzer, unit_id)
	if t is None:
		return None

	waveform_count = _extract_total_waveform_count(analyzer=analyzer, unit_id=unit_id)

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
	requested_waveforms: int | None
	if max_spikes_per_unit is None:
		requested_waveforms = None
	else:
		requested_waveforms = int(max_spikes_per_unit)
		if requested_waveforms <= 0:
			requested_waveforms = None
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
			if int(wf_all.shape[0]) < int(need_waveforms):
				if _try_recompute_waveforms_extension(
					analyzer=analyzer,
					requested_max_spikes_per_unit=int(need_waveforms),
					requested_ms_before=waveform_ms_before,
					requested_ms_after=waveform_ms_after,
				):
					wf_ext = analyzer.get_extension("waveforms")
					wf_all = np.asarray(wf_ext.get_waveforms_one_unit(unit_id=unit_id, force_dense=False), dtype=float)

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
