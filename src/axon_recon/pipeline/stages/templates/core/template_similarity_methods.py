from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemplateSimilarityFeatures:
	unit_id: Any
	template_c_by_t: np.ndarray
	locations_xy: np.ndarray
	ptp_by_channel: np.ndarray
	normalized_ptp_by_channel: np.ndarray
	dominant_waveform: np.ndarray
	active_amplitude_threshold: float
	amplitude_by_location: dict[tuple[float, float], float]


@dataclass(frozen=True)
class PairwiseTemplateSimilarity:
	method: str
	score: float
	metrics: dict[str, float]
	shared_channel_count: int
	union_channel_count: int


def normalize_template_similarity_method(raw: Any) -> str:
	token = str(raw or "ptp_cosine").strip().lower().replace("-", "_").replace(" ", "_")
	if token in {"ptp_cosine", "cosine", "amplitude_cosine", "footprint_cosine"}:
		return "ptp_cosine"
	if token in {"weighted_jaccard", "amplitude_weighted_jaccard", "footprint_weighted_jaccard"}:
		return "amplitude_weighted_jaccard"
	return "ptp_cosine"


def _coerce_channel_first(template_c_by_t: np.ndarray, locations_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	template = np.asarray(template_c_by_t, dtype=float)
	locations = np.asarray(locations_xy, dtype=float)
	if template.ndim != 2:
		raise ValueError(f"Expected 2D template array, got shape={template.shape}")
	if locations.ndim != 2 or int(locations.shape[1]) < 2:
		raise ValueError(f"Expected Nx2 channel locations, got shape={locations.shape}")
	locations = np.asarray(locations[:, :2], dtype=float)
	if int(template.shape[0]) == int(locations.shape[0]):
		return template, locations
	if int(template.shape[1]) == int(locations.shape[0]):
		return np.asarray(template.T, dtype=float), locations
	raise ValueError(
		"Template/time dimensions do not match channel locations: "
		f"template_shape={template.shape} locations_shape={locations.shape}"
	)


def _normalize_waveform(trace: np.ndarray) -> np.ndarray:
	waveform = np.asarray(trace, dtype=float).reshape(-1)
	if waveform.size == 0:
		return np.zeros((0,), dtype=float)
	max_abs = float(np.nanmax(np.abs(waveform)))
	if (not np.isfinite(max_abs)) or max_abs <= 0.0:
		return np.zeros_like(waveform, dtype=float)
	return np.asarray(waveform / max_abs, dtype=float)


def build_template_similarity_features(
	*,
	unit_id: Any,
	template_c_by_t: np.ndarray,
	locations_xy: np.ndarray,
) -> TemplateSimilarityFeatures:
	template, locations = _coerce_channel_first(template_c_by_t, locations_xy)
	ptp_by_channel = np.ptp(template, axis=1)
	ptp_by_channel = np.asarray(np.nan_to_num(ptp_by_channel, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
	max_ptp = float(np.max(ptp_by_channel)) if int(ptp_by_channel.size) > 0 else 0.0
	if max_ptp > 0.0:
		normalized_ptp_by_channel = np.asarray(ptp_by_channel / max_ptp, dtype=float)
	else:
		normalized_ptp_by_channel = np.zeros_like(ptp_by_channel, dtype=float)
	active_threshold = (max_ptp * 0.05) if max_ptp > 0.0 else 0.0
	dominant_channel = int(np.argmax(ptp_by_channel)) if int(ptp_by_channel.size) > 0 else 0
	dominant_waveform = (
		_normalize_waveform(template[dominant_channel, :])
		if int(template.shape[0]) > 0
		else np.zeros((0,), dtype=float)
	)
	return TemplateSimilarityFeatures(
		unit_id=unit_id,
		template_c_by_t=template,
		locations_xy=locations,
		ptp_by_channel=ptp_by_channel,
		normalized_ptp_by_channel=normalized_ptp_by_channel,
		dominant_waveform=dominant_waveform,
		active_amplitude_threshold=float(active_threshold),
		amplitude_by_location=_location_amplitude_map_from_arrays(locations, ptp_by_channel),
	)


def _location_key(location_xy: np.ndarray) -> tuple[float, float]:
	return (round(float(location_xy[0]), 4), round(float(location_xy[1]), 4))


def _location_amplitude_map_from_arrays(
	locations_xy: np.ndarray,
	ptp_by_channel: np.ndarray,
) -> dict[tuple[float, float], float]:
	amplitude_by_location: dict[tuple[float, float], float] = {}
	for location_xy, ptp_value in zip(locations_xy, ptp_by_channel, strict=False):
		key = _location_key(np.asarray(location_xy, dtype=float))
		amplitude_by_location[key] = max(float(ptp_value), float(amplitude_by_location.get(key, 0.0)))
	return amplitude_by_location


def _location_amplitude_map(features: TemplateSimilarityFeatures) -> dict[tuple[float, float], float]:
	return dict(features.amplitude_by_location)


def build_global_amplitude_matrix(
	features_by_unit: list[TemplateSimilarityFeatures],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
	if not features_by_unit:
		return np.zeros((0, 0), dtype=float), []
	ordered_locations = sorted(
		{
			location_key
			for features in features_by_unit
			for location_key in features.amplitude_by_location.keys()
		}
	)
	location_to_index = {location_key: idx for idx, location_key in enumerate(ordered_locations)}
	amplitude_matrix = np.zeros((len(features_by_unit), len(ordered_locations)), dtype=float)
	for row_idx, features in enumerate(features_by_unit):
		for location_key, amplitude in features.amplitude_by_location.items():
			amplitude_matrix[row_idx, location_to_index[location_key]] = float(amplitude)
	return amplitude_matrix, ordered_locations


def compute_ptp_cosine_similarity_matrix(amplitude_matrix: np.ndarray) -> np.ndarray:
	matrix = np.asarray(amplitude_matrix, dtype=float)
	if matrix.ndim != 2:
		raise ValueError(f"Expected 2D amplitude matrix, got shape={matrix.shape}")
	if int(matrix.shape[0]) == 0:
		return np.zeros((0, 0), dtype=float)
	row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	row_norms = np.asarray(np.where(row_norms > 0.0, row_norms, 1.0), dtype=float)
	normalized = np.asarray(matrix / row_norms, dtype=float)
	cosine = np.asarray(normalized @ normalized.T, dtype=float)
	cosine = np.clip(cosine, 0.0, 1.0)
	if int(cosine.shape[0]) > 0:
		np.fill_diagonal(cosine, 1.0)
	return cosine


def _aligned_amplitude_vectors(
	features_a: TemplateSimilarityFeatures,
	features_b: TemplateSimilarityFeatures,
) -> tuple[np.ndarray, np.ndarray]:
	amp_a = _location_amplitude_map(features_a)
	amp_b = _location_amplitude_map(features_b)
	ordered_keys = sorted(set(amp_a) | set(amp_b))
	vector_a = np.asarray([float(amp_a.get(key, 0.0)) for key in ordered_keys], dtype=float)
	vector_b = np.asarray([float(amp_b.get(key, 0.0)) for key in ordered_keys], dtype=float)
	return vector_a, vector_b


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
	vec_a = np.asarray(vector_a, dtype=float).reshape(-1)
	vec_b = np.asarray(vector_b, dtype=float).reshape(-1)
	if int(vec_a.size) == 0 or int(vec_b.size) == 0:
		return 0.0
	norm_a = float(np.linalg.norm(vec_a))
	norm_b = float(np.linalg.norm(vec_b))
	if norm_a <= 0.0 or norm_b <= 0.0:
		return 0.0
	score = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
	return float(max(0.0, min(1.0, score)))


def _weighted_jaccard(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
	vec_a = np.asarray(vector_a, dtype=float).reshape(-1)
	vec_b = np.asarray(vector_b, dtype=float).reshape(-1)
	if int(vec_a.size) == 0 or int(vec_b.size) == 0:
		return 0.0
	numerator = float(np.sum(np.minimum(vec_a, vec_b)))
	denominator = float(np.sum(np.maximum(vec_a, vec_b)))
	if denominator <= 0.0:
		return 1.0
	return float(max(0.0, min(1.0, numerator / denominator)))


def _occupied_channel_jaccard(
	vector_a: np.ndarray,
	vector_b: np.ndarray,
	*,
	threshold_a: float,
	threshold_b: float,
) -> tuple[float, int, int]:
	vec_a = np.asarray(vector_a, dtype=float).reshape(-1)
	vec_b = np.asarray(vector_b, dtype=float).reshape(-1)
	active_a = vec_a > float(max(0.0, threshold_a))
	active_b = vec_b > float(max(0.0, threshold_b))
	shared = int(np.count_nonzero(active_a & active_b))
	union = int(np.count_nonzero(active_a | active_b))
	if union <= 0:
		return 1.0, shared, union
	return float(shared / union), shared, union


def compute_pairwise_template_similarity(
	*,
	features_a: TemplateSimilarityFeatures,
	features_b: TemplateSimilarityFeatures,
	method: str,
) -> PairwiseTemplateSimilarity:
	resolved_method = normalize_template_similarity_method(method)
	vector_a, vector_b = _aligned_amplitude_vectors(features_a, features_b)
	ptp_cosine = _cosine_similarity(vector_a, vector_b)
	weighted_jaccard = _weighted_jaccard(vector_a, vector_b)
	occupied_jaccard, shared_count, union_count = _occupied_channel_jaccard(
		vector_a,
		vector_b,
		threshold_a=features_a.active_amplitude_threshold,
		threshold_b=features_b.active_amplitude_threshold,
	)
	metrics = {
		"ptp_cosine": float(ptp_cosine),
		"amplitude_weighted_jaccard": float(weighted_jaccard),
		"occupied_channel_jaccard": float(occupied_jaccard),
	}
	return PairwiseTemplateSimilarity(
		method=resolved_method,
		score=float(metrics.get(resolved_method, ptp_cosine)),
		metrics=metrics,
		shared_channel_count=int(shared_count),
		union_channel_count=int(union_count),
	)