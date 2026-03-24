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


def build_unit_source_payload(
	*,
	analyzer: Any,
	unit_id: Any,
) -> tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None, int, float | None, np.ndarray | None, Any, int | None] | None:
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
