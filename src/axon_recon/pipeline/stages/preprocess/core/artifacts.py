from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_saved_recording(path: Path) -> Any:
	import spikeinterface.full as si  # type: ignore[import-not-found]

	resolved = Path(path).expanduser().resolve()
	for method_name in ("load_extractor", "load_recording", "load"):
		loader = getattr(si, method_name, None)
		if not callable(loader):
			continue
		try:
			return loader(resolved)
		except Exception:
			continue
	raise RuntimeError(f"Could not load saved recording extractor from {resolved}")


def load_segment_manifest(manifest_path: Path) -> list[dict[str, Any]]:
	resolved = Path(manifest_path).expanduser().resolve()
	if not resolved.exists():
		raise FileNotFoundError(f"Preprocess segment manifest not found: {resolved}")
	payload = read_json(resolved)
	if not isinstance(payload, dict):
		raise RuntimeError(f"Invalid preprocess segment manifest payload: {resolved}")
	segments = payload.get("segments", [])
	if not isinstance(segments, list):
		raise RuntimeError(f"Invalid preprocess segment manifest segments payload: {resolved}")
	return [dict(item) for item in segments if isinstance(item, dict)]


def load_concat_manifest(manifest_path: Path) -> dict[str, Any]:
	resolved = Path(manifest_path).expanduser().resolve()
	if not resolved.exists():
		raise FileNotFoundError(f"Concat segment manifest not found: {resolved}")
	payload = read_json(resolved)
	if not isinstance(payload, dict):
		raise RuntimeError(f"Invalid concat segment manifest payload: {resolved}")
	return dict(payload)


def load_recording_metadata(
	*,
	segment_epochs_path: Path,
	contiguous_epochs_path: Path,
	sampling_metadata_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
	segment_payload = read_json(Path(segment_epochs_path).expanduser().resolve())
	contiguous_payload = read_json(Path(contiguous_epochs_path).expanduser().resolve())
	sampling_payload = read_json(Path(sampling_metadata_path).expanduser().resolve())
	if not isinstance(segment_payload, dict):
		raise RuntimeError(f"Invalid segment epochs payload: {segment_epochs_path}")
	if not isinstance(contiguous_payload, dict):
		raise RuntimeError(f"Invalid contiguous epochs payload: {contiguous_epochs_path}")
	if not isinstance(sampling_payload, dict):
		raise RuntimeError(f"Invalid sampling metadata payload: {sampling_metadata_path}")
	return dict(segment_payload), dict(contiguous_payload), dict(sampling_payload)


def load_common_electrodes(path: Path) -> list[int]:
	import numpy as np

	resolved = Path(path).expanduser().resolve()
	if not resolved.exists():
		raise FileNotFoundError(f"Common-electrode artifact not found: {resolved}")
	values = np.asarray(np.load(resolved), dtype=int)
	return [int(value) for value in values.tolist()]


def get_segment_names_from_metadata(segment_epochs_payload: dict[str, Any]) -> list[str]:
	segments = segment_epochs_payload.get("segments", [])
	if not isinstance(segments, list):
		return []
	return [str(item.get("rec_name")) for item in segments if isinstance(item, dict) and str(item.get("rec_name", "")).strip()]


def build_stitch_frames_from_segment_manifest(segment_entries: list[dict[str, Any]]) -> list[int]:
	stitch_frames: list[int] = []
	acc = 0
	for item in segment_entries[:-1]:
		try:
			acc += int(item.get("n_samples", 0) or 0)
		except Exception:
			continue
		stitch_frames.append(int(acc))
	return stitch_frames


def build_segment_time_vector(
	*,
	rec_name: str,
	segment_epochs_payload: dict[str, Any],
	contiguous_epochs_payload: dict[str, Any],
	sampling_metadata_payload: dict[str, Any],
	absolute_origin_s: float | None = None,
) -> Any | None:
	import numpy as np

	segments = segment_epochs_payload.get("segments", [])
	epochs = contiguous_epochs_payload.get("epochs", [])
	sampling_segments = sampling_metadata_payload.get("segments", [])
	if not isinstance(segments, list) or not isinstance(epochs, list) or not isinstance(sampling_segments, list):
		return None

	segment_payload = next(
		(item for item in segments if isinstance(item, dict) and str(item.get("rec_name", "")) == str(rec_name)),
		None,
	)
	if not isinstance(segment_payload, dict):
		return None

	sampling_payload = next(
		(item for item in sampling_segments if isinstance(item, dict) and str(item.get("rec_name", "")) == str(rec_name)),
		None,
	)
	if not isinstance(sampling_payload, dict):
		return None

	try:
		n_samples = int(segment_payload.get("n_samples", sampling_payload.get("n_samples", 0)) or 0)
		fs_hz = float(sampling_payload.get("sampling_frequency_hz", segment_payload.get("sampling_frequency_hz", 0.0)) or 0.0)
	except Exception:
		return None
	if n_samples <= 0 or fs_hz <= 0.0:
		return None

	times = np.full(int(n_samples), np.nan, dtype=float)
	relevant_epochs = [
		dict(item)
		for item in epochs
		if isinstance(item, dict) and str(item.get("rec_name", "")) == str(rec_name)
	]
	for epoch in relevant_epochs:
		try:
			start_sample = int(epoch.get("segment_start_sample", 0) or 0)
			end_sample = int(epoch.get("segment_end_sample", 0) or 0)
		except Exception:
			continue
		if end_sample <= start_sample:
			continue
		start_time = epoch.get("segment_relative_start_s", None)
		if absolute_origin_s is not None:
			start_time = epoch.get("start_time_seconds_since_epoch", None)
			if start_time is None:
				continue
			start_time = float(start_time) - float(absolute_origin_s)
		if start_time is None:
			continue
		length = int(end_sample - start_sample)
		times[start_sample:end_sample] = float(start_time) + (np.arange(length, dtype=float) / float(fs_hz))
	if np.isnan(times).any():
		return None
	return times


def build_concat_time_vector(
	*,
	segment_entries: list[dict[str, Any]],
	segment_epochs_payload: dict[str, Any],
	contiguous_epochs_payload: dict[str, Any],
	sampling_metadata_payload: dict[str, Any],
) -> Any | None:
	import numpy as np

	ordered_names = [str(item.get("rec_name")) for item in segment_entries if str(item.get("rec_name", "")).strip()]
	if not ordered_names:
		ordered_names = get_segment_names_from_metadata(segment_epochs_payload)
	if not ordered_names:
		return None

	segments = segment_epochs_payload.get("segments", [])
	if not isinstance(segments, list):
		return None
	first_segment = next(
		(item for item in segments if isinstance(item, dict) and str(item.get("rec_name", "")) == str(ordered_names[0])),
		None,
	)
	if not isinstance(first_segment, dict):
		return None
	first_start = first_segment.get("start_time_seconds_since_epoch", None)
	if first_start is None:
		return None

	pieces: list[Any] = []
	for rec_name in ordered_names:
		time_vector = build_segment_time_vector(
			rec_name=str(rec_name),
			segment_epochs_payload=segment_epochs_payload,
			contiguous_epochs_payload=contiguous_epochs_payload,
			sampling_metadata_payload=sampling_metadata_payload,
			absolute_origin_s=float(first_start),
		)
		if time_vector is None:
			return None
		pieces.append(time_vector)
	if not pieces:
		return None
	return np.concatenate(pieces).astype(float, copy=False)