from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float_hz(value: Any) -> float | None:
	try:
		out = float(value)
	except Exception:
		return None
	if not np.isfinite(out) or out <= 0.0:
		return None
	return out


def rates_match_hz(*, raw_hz: float | None, analyzer_hz: float | None, atol_hz: float = 0.5) -> bool:
	raw = _safe_float_hz(raw_hz)
	ana = _safe_float_hz(analyzer_hz)
	if raw is None or ana is None:
		return False
	return abs(raw - ana) <= float(max(0.0, atol_hz))


def _stream_to_well_label(stream_id: str) -> str:
	s = str(stream_id).strip()
	ls = s.lower()
	if ls.startswith("well") and len(s) >= 7 and s[4:].isdigit():
		return f"well{int(s[4:]):03d}"
	if s.isdigit():
		return f"well{int(s):03d}"
	return s


def _well_label_from_well_id(value: Any) -> str | None:
	try:
		return f"well{int(value):03d}"
	except Exception:
		return None


def _read_h5_scalar_dataset(group: Any, path: str) -> Any:
	if path not in group:
		return None
	try:
		value = group[path][()]
	except Exception:
		return None
	try:
		if isinstance(value, np.ndarray):
			if value.shape == ():
				value = value.item()
			elif value.size == 1:
				value = value.ravel()[0].item()
			else:
				return None
	except Exception:
		return None
	if isinstance(value, (bytes, bytearray)):
		try:
			return value.decode(errors="ignore")
		except Exception:
			return str(value)
	return value


def _try_get_attr_hz(obj: Any, names: list[str]) -> float | None:
	try:
		attrs = getattr(obj, "attrs", None)
		if attrs is None:
			return None
		key_map = {str(k).lower(): k for k in attrs.keys()}
	except Exception:
		return None

	for name in names:
		key = key_map.get(str(name).lower())
		if key is None:
			continue
		try:
			value = attrs[key]
		except Exception:
			continue
		hz = _safe_float_hz(value)
		if hz is not None:
			return hz
	return None


def _sorted_data_store_keys(keys: list[str]) -> list[str]:
	def _key_num(k: str) -> int:
		digits = "".join(ch for ch in str(k) if ch.isdigit())
		try:
			return int(digits) if digits else 0
		except Exception:
			return 0

	return sorted(keys, key=_key_num)


def read_maxwell_sampling_frequency_hz(*, h5_path: Path, stream_id: str) -> float | None:
	try:
		import h5py
	except Exception:
		return None

	target_well_label = _stream_to_well_label(str(stream_id))
	attr_names = [
		"sampling_frequency",
		"sampling rate",
		"sampling_rate",
		"samplerate",
		"sample_rate",
		"sampling",
		"fs",
		"frequency",
	]

	try:
		with h5py.File(str(Path(h5_path).expanduser().resolve()), "r") as h5:
			# Preferred source: /data_store/dataXXXX/settings/sampling (copied from stg1 helper approach).
			if "data_store" in h5:
				data_store = h5["data_store"]
				keys = [str(k) for k in data_store.keys() if str(k).startswith("data")]
				rates: list[float] = []
				for key in _sorted_data_store_keys(keys):
					entry = data_store[key]
					well_id = _read_h5_scalar_dataset(entry, "well_id")
					well_label = _well_label_from_well_id(well_id)
					if well_label is not None and str(well_label) != str(target_well_label):
						continue
					hz = _safe_float_hz(_read_h5_scalar_dataset(entry, "settings/sampling"))
					if hz is not None:
						rates.append(float(hz))
				if rates:
					return float(np.median(np.asarray(rates, dtype=float)))

			# Fallback: common sampling-frequency attributes at stream/wells/root scopes.
			candidates: list[Any] = []
			if "wells" in h5:
				wells = h5["wells"]
				if str(target_well_label) in wells:
					candidates.append(wells[str(target_well_label)])
				if str(stream_id) in wells and str(stream_id) != str(target_well_label):
					candidates.append(wells[str(stream_id)])
				candidates.append(wells)
			candidates.extend([h5.get("data_store"), h5])

			for obj in candidates:
				if obj is None:
					continue
				hz = _try_get_attr_hz(obj, attr_names)
				if hz is not None:
					return float(hz)
	except Exception:
		return None

	return None


def upsample_channels_by_time(
	*,
	template_c_by_t: np.ndarray,
	source_hz: float,
	target_hz: float,
	method: str,
) -> np.ndarray:
	t = np.asarray(template_c_by_t, dtype=float)
	if t.ndim != 2:
		raise ValueError(f"Expected 2D channels-by-time template, got shape={t.shape}")

	src = _safe_float_hz(source_hz)
	tgt = _safe_float_hz(target_hz)
	if src is None or tgt is None:
		raise ValueError(f"Invalid sampling rates for upsampling: source_hz={source_hz}, target_hz={target_hz}")
	if tgt <= src:
		return t

	factor = float(tgt / src)
	new_t = max(2, int(round(int(t.shape[1]) * factor)))
	m = str(method or "sinc").strip().lower()

	if m in {"", "sinc", "whittaker-shannon", "whittaker_shannon", "polyphase", "resample_poly"}:
		try:
			from scipy.signal import resample_poly  # type: ignore[import-not-found]

			src_i = int(round(src))
			tgt_i = int(round(tgt))
			g = gcd(max(1, src_i), max(1, tgt_i))
			up = max(1, int(tgt_i // g))
			down = max(1, int(src_i // g))
			return np.asarray(resample_poly(t, up=up, down=down, axis=1), dtype=float)
		except Exception:
			m = "linear"

	x_old = np.arange(int(t.shape[1]), dtype=float)
	x_new = np.linspace(0.0, float(int(t.shape[1]) - 1), new_t, dtype=float)

	if m in {"nearest", "nn"}:
		nearest_idx = np.rint(x_new).astype(int)
		nearest_idx = np.clip(nearest_idx, 0, int(t.shape[1]) - 1)
		return t[:, nearest_idx]

	if m in {"linear", "interp"}:
		out = np.empty((int(t.shape[0]), new_t), dtype=float)
		for ch in range(int(t.shape[0])):
			out[ch, :] = np.interp(x_new, x_old, t[ch, :])
		return out

	raise ValueError(f"Unsupported upsampling method: {method!r}")
