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
	waveform_by_location: dict[tuple[float, float], np.ndarray]


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
	if token in {"occupied_channel_jaccard", "occupancy_jaccard", "channel_jaccard"}:
		return "occupied_channel_jaccard"
	if token in {"lagged_cosine", "si_cosine", "spikeinterface_cosine", "cosine_similarity"}:
		return "lagged_cosine"
	if token in {"lagged_l1", "si_l1", "spikeinterface_l1", "template_l1"}:
		return "lagged_l1"
	if token in {"lagged_l2", "si_l2", "spikeinterface_l2", "template_l2"}:
		return "lagged_l2"
	if token in {"slay_mean_similarity", "slay_waveform_similarity", "mean_similarity", "jittered_mean_similarity"}:
		return "slay_mean_similarity"
	if token in {"hybrid_template_similarity", "template_hybrid_similarity", "spikeinterface_slay_hybrid"}:
		return "hybrid_template_similarity"
	return "ptp_cosine"


def normalize_template_similarity_support(raw: Any, default: str = "union") -> str:
	token = str(raw or default).strip().lower().replace("-", "_").replace(" ", "_")
	if token in {"intersection", "shared"}:
		return "intersection"
	if token in {"dense", "all"}:
		return "dense"
	return "union"


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
		waveform_by_location=_location_waveform_map_from_arrays(locations, template, ptp_by_channel),
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


def _location_waveform_map_from_arrays(
	locations_xy: np.ndarray,
	template_c_by_t: np.ndarray,
	ptp_by_channel: np.ndarray,
) -> dict[tuple[float, float], np.ndarray]:
	waveform_by_location: dict[tuple[float, float], np.ndarray] = {}
	best_ptp_by_location: dict[tuple[float, float], float] = {}
	for location_xy, waveform, ptp_value in zip(locations_xy, template_c_by_t, ptp_by_channel, strict=False):
		key = _location_key(np.asarray(location_xy, dtype=float))
		current_ptp = float(best_ptp_by_location.get(key, -np.inf))
		candidate_ptp = float(ptp_value)
		if (key not in waveform_by_location) or (candidate_ptp >= current_ptp):
			waveform_by_location[key] = np.asarray(waveform, dtype=float).reshape(-1).copy()
			best_ptp_by_location[key] = candidate_ptp
	return waveform_by_location


def _location_amplitude_map(features: TemplateSimilarityFeatures) -> dict[tuple[float, float], float]:
	return dict(features.amplitude_by_location)


def _aligned_waveform_templates(
	features_a: TemplateSimilarityFeatures,
	features_b: TemplateSimilarityFeatures,
	*,
	support: str,
) -> tuple[np.ndarray, np.ndarray]:
	support_mode = normalize_template_similarity_support(support, default="union")
	map_a = dict(features_a.waveform_by_location)
	map_b = dict(features_b.waveform_by_location)
	shared_keys = set(map_a) & set(map_b)
	if support_mode == "intersection":
		ordered_keys = sorted(shared_keys)
	elif support_mode == "union":
		if not shared_keys:
			return np.zeros((0, 0), dtype=float), np.zeros((0, 0), dtype=float)
		ordered_keys = sorted(set(map_a) | set(map_b))
	else:
		ordered_keys = sorted(set(map_a) | set(map_b))
	sample_count = int(min(features_a.template_c_by_t.shape[1], features_b.template_c_by_t.shape[1]))
	if sample_count <= 0 or not ordered_keys:
		return np.zeros((0, 0), dtype=float), np.zeros((0, 0), dtype=float)
	template_a = np.zeros((len(ordered_keys), sample_count), dtype=float)
	template_b = np.zeros((len(ordered_keys), sample_count), dtype=float)
	for idx, location_key in enumerate(ordered_keys):
		waveform_a = map_a.get(location_key, None)
		waveform_b = map_b.get(location_key, None)
		if waveform_a is not None:
			template_a[idx, :] = np.asarray(waveform_a, dtype=float).reshape(-1)[:sample_count]
		if waveform_b is not None:
			template_b[idx, :] = np.asarray(waveform_b, dtype=float).reshape(-1)[:sample_count]
	return template_a, template_b


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


def build_dominant_waveform_matrix(features_by_unit: list[TemplateSimilarityFeatures]) -> np.ndarray:
	if not features_by_unit:
		return np.zeros((0, 0), dtype=float)
	max_samples = max(int(np.asarray(features.dominant_waveform, dtype=float).size) for features in features_by_unit)
	if max_samples <= 0:
		return np.zeros((len(features_by_unit), 0), dtype=float)
	waveform_matrix = np.zeros((len(features_by_unit), max_samples), dtype=float)
	for row_idx, features in enumerate(features_by_unit):
		waveform = np.asarray(features.dominant_waveform, dtype=float).reshape(-1)
		if int(waveform.size) <= 0:
			continue
		if int(waveform.size) == max_samples:
			waveform_matrix[row_idx, :] = waveform
			continue
		source_x = np.linspace(0.0, 1.0, int(waveform.size), dtype=float)
		target_x = np.linspace(0.0, 1.0, int(max_samples), dtype=float)
		waveform_matrix[row_idx, :] = np.asarray(np.interp(target_x, source_x, waveform), dtype=float)
	return waveform_matrix


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


def compute_weighted_jaccard_similarity_matrix(amplitude_matrix: np.ndarray) -> np.ndarray:
	matrix = np.asarray(amplitude_matrix, dtype=float)
	if matrix.ndim != 2:
		raise ValueError(f"Expected 2D amplitude matrix, got shape={matrix.shape}")
	row_count = int(matrix.shape[0])
	if row_count <= 0:
		return np.zeros((0, 0), dtype=float)
	matrix = np.asarray(np.clip(matrix, 0.0, None), dtype=float)
	row_sums = np.asarray(np.sum(matrix, axis=1), dtype=float)
	result = np.zeros((row_count, row_count), dtype=float)
	for row_idx in range(row_count):
		numerator = np.asarray(np.minimum(matrix[row_idx : row_idx + 1, :], matrix).sum(axis=1), dtype=float)
		denominator = np.asarray(row_sums[row_idx] + row_sums - numerator, dtype=float)
		result[row_idx, :] = np.asarray(
			np.where(denominator > 0.0, numerator / denominator, 1.0),
			dtype=float,
		)
	result = np.asarray((result + result.T) * 0.5, dtype=float)
	np.fill_diagonal(result, 1.0)
	return np.clip(result, 0.0, 1.0)


def compute_occupied_channel_jaccard_similarity_matrix(
	amplitude_matrix: np.ndarray,
	*,
	active_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	matrix = np.asarray(amplitude_matrix, dtype=float)
	thresholds = np.asarray(active_thresholds, dtype=float).reshape(-1)
	if matrix.ndim != 2:
		raise ValueError(f"Expected 2D amplitude matrix, got shape={matrix.shape}")
	row_count = int(matrix.shape[0])
	if row_count <= 0:
		zeros = np.zeros((0, 0), dtype=float)
		return zeros, zeros.astype(int), zeros.astype(int)
	if int(thresholds.size) != row_count:
		raise ValueError(
			f"active_thresholds length must match amplitude matrix rows: thresholds={thresholds.size} rows={row_count}"
		)
	occupied = np.asarray(matrix > thresholds[:, None], dtype=np.int16)
	shared = np.asarray(occupied @ occupied.T, dtype=int)
	counts = np.asarray(np.sum(occupied, axis=1), dtype=int)
	union = np.asarray(counts[:, None] + counts[None, :] - shared, dtype=int)
	similarity = np.asarray(np.where(union > 0, shared / union, 1.0), dtype=float)
	np.fill_diagonal(similarity, 1.0)
	return np.clip(similarity, 0.0, 1.0), shared, union


def _shifted_waveform_matrix_views(waveform_matrix: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
	matrix = np.asarray(waveform_matrix, dtype=float)
	sample_count = int(matrix.shape[1])
	if sample_count <= 0:
		return np.zeros((matrix.shape[0], 0), dtype=float), np.zeros((matrix.shape[0], 0), dtype=float)
	if shift < 0:
		return matrix[:, : sample_count + shift], matrix[:, (-shift):sample_count]
	if shift > 0:
		return matrix[:, shift:sample_count], matrix[:, : sample_count - shift]
	return matrix[:, :sample_count], matrix[:, :sample_count]


def compute_lagged_waveform_similarity_matrix(
	waveform_matrix: np.ndarray,
	*,
	method: str,
	max_lag_samples: int,
) -> np.ndarray:
	matrix = np.asarray(waveform_matrix, dtype=float)
	if matrix.ndim != 2:
		raise ValueError(f"Expected 2D waveform matrix, got shape={matrix.shape}")
	row_count = int(matrix.shape[0])
	sample_count = int(matrix.shape[1])
	if row_count <= 0:
		return np.zeros((0, 0), dtype=float)
	if sample_count <= 0:
		return np.eye(row_count, dtype=float)
	metric_token = str(method).strip().lower()
	max_lag = int(max(0, min(int(max_lag_samples), sample_count - 1)))
	best_similarity = np.full((row_count, row_count), -np.inf, dtype=float)
	for shift in range(-max_lag, max_lag + 1):
		left, right = _shifted_waveform_matrix_views(matrix, shift)
		if int(left.shape[1]) <= 0 or int(right.shape[1]) <= 0:
			continue
		if metric_token == "cosine":
			left_norms = np.asarray(np.linalg.norm(left, axis=1), dtype=float)
			right_norms = np.asarray(np.linalg.norm(right, axis=1), dtype=float)
			left_norms = np.asarray(np.where(left_norms > 0.0, left_norms, 1.0), dtype=float)
			right_norms = np.asarray(np.where(right_norms > 0.0, right_norms, 1.0), dtype=float)
			dot = np.asarray(left @ right.T, dtype=float)
			similarity = np.asarray(dot / (left_norms[:, None] * right_norms[None, :]), dtype=float)
		elif metric_token == "l2":
			dot = np.asarray(left @ right.T, dtype=float)
			left_sq = np.asarray(np.sum(left * left, axis=1), dtype=float)
			right_sq = np.asarray(np.sum(right * right, axis=1), dtype=float)
			left_norms = np.asarray(np.sqrt(np.maximum(left_sq, 0.0)), dtype=float)
			right_norms = np.asarray(np.sqrt(np.maximum(right_sq, 0.0)), dtype=float)
			denominator = np.asarray(left_norms[:, None] + right_norms[None, :], dtype=float)
			distance_sq = np.asarray(np.maximum(left_sq[:, None] + right_sq[None, :] - (2.0 * dot), 0.0), dtype=float)
			distance = np.asarray(np.sqrt(distance_sq), dtype=float)
			similarity = np.asarray(np.where(denominator > 0.0, 1.0 - (distance / denominator), 0.0), dtype=float)
		else:
			left_abs_sum = np.asarray(np.sum(np.abs(left), axis=1), dtype=float)
			right_abs_sum = np.asarray(np.sum(np.abs(right), axis=1), dtype=float)
			similarity = np.zeros((row_count, row_count), dtype=float)
			for row_idx in range(row_count):
				distance = np.asarray(np.sum(np.abs(left[row_idx : row_idx + 1, :] - right), axis=1), dtype=float)
				denominator = np.asarray(left_abs_sum[row_idx] + right_abs_sum, dtype=float)
				similarity[row_idx, :] = np.asarray(
					np.where(denominator > 0.0, 1.0 - (distance / denominator), 0.0),
					dtype=float,
				)
		best_similarity = np.asarray(np.maximum(best_similarity, similarity), dtype=float)
	best_similarity = np.asarray((best_similarity + best_similarity.T) * 0.5, dtype=float)
	best_similarity = np.clip(best_similarity, 0.0, 1.0)
	np.fill_diagonal(best_similarity, 1.0)
	return best_similarity


def compute_slay_mean_similarity_matrix(
	waveform_matrix: np.ndarray,
	*,
	max_lag_samples: int,
) -> np.ndarray:
	matrix = np.asarray(waveform_matrix, dtype=float)
	if matrix.ndim != 2:
		raise ValueError(f"Expected 2D waveform matrix, got shape={matrix.shape}")
	row_count = int(matrix.shape[0])
	sample_count = int(matrix.shape[1])
	if row_count <= 0:
		return np.zeros((0, 0), dtype=float)
	if sample_count <= 0:
		return np.eye(row_count, dtype=float)
	base_left, base_right = _shifted_waveform_matrix_views(matrix, 0)
	base_dot = np.asarray(base_left @ base_right.T, dtype=float)
	base_norms = np.asarray(np.linalg.norm(base_left, axis=1), dtype=float)
	denominator = np.asarray(np.maximum(base_norms[:, None], base_norms[None, :]) ** 2, dtype=float)
	best_dot = np.asarray(base_dot, dtype=float)
	max_lag = int(max(0, min(int(max_lag_samples), sample_count - 1)))
	for shift in range(-max_lag, max_lag + 1):
		if shift == 0:
			continue
		left, right = _shifted_waveform_matrix_views(matrix, shift)
		if int(left.shape[1]) <= 0 or int(right.shape[1]) <= 0:
			continue
		dot = np.asarray(left @ right.T, dtype=float)
		best_dot = np.asarray(np.maximum(best_dot, dot), dtype=float)
	improvement = np.asarray(best_dot - base_dot, dtype=float)
	best_dot = np.asarray(np.where(improvement >= 0.1, best_dot, base_dot), dtype=float)
	similarity = np.asarray(np.where(denominator > 0.0, best_dot / denominator, 0.0), dtype=float)
	similarity = np.asarray((similarity + similarity.T) * 0.5, dtype=float)
	similarity = np.clip(similarity, 0.0, 1.0)
	np.fill_diagonal(similarity, 1.0)
	return similarity


def compute_hybrid_template_similarity_matrix(
	*,
	amplitude_matrix: np.ndarray,
	waveform_matrix: np.ndarray,
	active_thresholds: np.ndarray,
	max_lag_samples: int,
	waveform_weight: float,
	amplitude_weight: float,
	occupancy_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
	lagged_cosine = compute_lagged_waveform_similarity_matrix(
		waveform_matrix,
		method="cosine",
		max_lag_samples=max_lag_samples,
	)
	weighted_jaccard = compute_weighted_jaccard_similarity_matrix(amplitude_matrix)
	occupied_jaccard, shared_counts, union_counts = compute_occupied_channel_jaccard_similarity_matrix(
		amplitude_matrix,
		active_thresholds=active_thresholds,
	)
	weights = np.asarray(
		[
			max(0.0, float(waveform_weight)),
			max(0.0, float(amplitude_weight)),
			max(0.0, float(occupancy_weight)),
		],
		dtype=float,
	)
	if float(np.sum(weights)) <= 0.0:
		weights = np.asarray([0.5, 0.3, 0.2], dtype=float)
	weights = np.asarray(weights / float(np.sum(weights)), dtype=float)
	hybrid = np.asarray(
		(weights[0] * lagged_cosine) + (weights[1] * weighted_jaccard) + (weights[2] * occupied_jaccard),
		dtype=float,
	)
	hybrid = np.asarray((hybrid + hybrid.T) * 0.5, dtype=float)
	hybrid = np.clip(hybrid, 0.0, 1.0)
	np.fill_diagonal(hybrid, 1.0)
	return hybrid, {
		"lagged_cosine": lagged_cosine,
		"amplitude_weighted_jaccard": weighted_jaccard,
		"occupied_channel_jaccard": occupied_jaccard,
		"shared_channel_count": np.asarray(shared_counts, dtype=float),
		"union_channel_count": np.asarray(union_counts, dtype=float),
		"hybrid_template_similarity": hybrid,
	}


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


def _l1_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
	vec_a = np.asarray(vector_a, dtype=float).reshape(-1)
	vec_b = np.asarray(vector_b, dtype=float).reshape(-1)
	if int(vec_a.size) == 0 or int(vec_b.size) == 0:
		return 0.0
	denominator = float(np.sum(np.abs(vec_a)) + np.sum(np.abs(vec_b)))
	if denominator <= 0.0:
		return 0.0
	distance = float(np.sum(np.abs(vec_a - vec_b)) / denominator)
	return float(max(0.0, min(1.0, 1.0 - distance)))


def _l2_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
	vec_a = np.asarray(vector_a, dtype=float).reshape(-1)
	vec_b = np.asarray(vector_b, dtype=float).reshape(-1)
	if int(vec_a.size) == 0 or int(vec_b.size) == 0:
		return 0.0
	norm_a = float(np.linalg.norm(vec_a))
	norm_b = float(np.linalg.norm(vec_b))
	denominator = norm_a + norm_b
	if denominator <= 0.0:
		return 0.0
	distance = float(np.linalg.norm(vec_a - vec_b) / denominator)
	return float(max(0.0, min(1.0, 1.0 - distance)))


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


def _flatten_template(template_c_by_t: np.ndarray) -> np.ndarray:
	return np.asarray(template_c_by_t, dtype=float).reshape(-1)


def _shifted_template_views(template_a: np.ndarray, template_b: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
	sample_count = int(min(template_a.shape[1], template_b.shape[1]))
	if sample_count <= 0:
		return np.zeros((0, 0), dtype=float), np.zeros((0, 0), dtype=float)
	if shift < 0:
		return template_a[:, : sample_count + shift], template_b[:, (-shift):sample_count]
	if shift > 0:
		return template_a[:, shift:sample_count], template_b[:, : sample_count - shift]
	return template_a[:, :sample_count], template_b[:, :sample_count]


def _lagged_template_similarity(
	template_a: np.ndarray,
	template_b: np.ndarray,
	*,
	metric: str,
	max_lag_samples: int,
) -> tuple[float, int]:
	if int(template_a.size) == 0 or int(template_b.size) == 0:
		return 0.0, 0
	sample_count = int(min(template_a.shape[1], template_b.shape[1]))
	if sample_count <= 0:
		return 0.0, 0
	metric_token = str(metric).strip().lower()
	max_lag = int(max(0, min(int(max_lag_samples), sample_count - 1)))
	base_a, base_b = _shifted_template_views(template_a, template_b, 0)
	best_lag = 0
	if metric_token == "cosine":
		best_score = _cosine_similarity(_flatten_template(base_a), _flatten_template(base_b))
		metric_fn = _cosine_similarity
	elif metric_token == "l1":
		best_score = _l1_similarity(_flatten_template(base_a), _flatten_template(base_b))
		metric_fn = _l1_similarity
	else:
		best_score = _l2_similarity(_flatten_template(base_a), _flatten_template(base_b))
		metric_fn = _l2_similarity
	for shift in range(-max_lag, max_lag + 1):
		if shift == 0:
			continue
		shift_a, shift_b = _shifted_template_views(template_a, template_b, shift)
		if int(shift_a.size) == 0 or int(shift_b.size) == 0:
			continue
		score = metric_fn(_flatten_template(shift_a), _flatten_template(shift_b))
		if score > best_score:
			best_score = float(score)
			best_lag = int(shift)
	return float(best_score), int(best_lag)


def _slay_mean_similarity(
	template_a: np.ndarray,
	template_b: np.ndarray,
	*,
	max_lag_samples: int,
) -> tuple[float, int]:
	if int(template_a.size) == 0 or int(template_b.size) == 0:
		return 0.0, 0
	sample_count = int(min(template_a.shape[1], template_b.shape[1]))
	if sample_count <= 0:
		return 0.0, 0
	base_a, base_b = _shifted_template_views(template_a, template_b, 0)
	flat_base_a = _flatten_template(base_a)
	flat_base_b = _flatten_template(base_b)
	base_dot = float(np.dot(flat_base_a, flat_base_b))
	best_dot = float(base_dot)
	best_lag = 0
	max_lag = int(max(0, min(int(max_lag_samples), sample_count - 1)))
	for shift in range(-max_lag, max_lag + 1):
		if shift == 0:
			continue
		shift_a, shift_b = _shifted_template_views(template_a, template_b, shift)
		if int(shift_a.size) == 0 or int(shift_b.size) == 0:
			continue
		shift_dot = float(np.dot(_flatten_template(shift_a), _flatten_template(shift_b)))
		if shift_dot > best_dot:
			best_dot = shift_dot
			best_lag = int(shift)
	if float(best_dot - base_dot) < 0.1:
		best_dot = base_dot
		best_lag = 0
	norm = float(max(np.linalg.norm(flat_base_a), np.linalg.norm(flat_base_b)))
	if norm <= 0.0:
		return 0.0, 0
	score = float(best_dot / (norm**2))
	return float(max(0.0, min(1.0, score))), int(best_lag)


def _hybrid_template_similarity(
	*,
	lagged_cosine: float,
	weighted_jaccard: float,
	occupied_jaccard: float,
	waveform_weight: float,
	amplitude_weight: float,
	occupancy_weight: float,
) -> float:
	weights = np.asarray(
		[
			max(0.0, float(waveform_weight)),
			max(0.0, float(amplitude_weight)),
			max(0.0, float(occupancy_weight)),
		],
		dtype=float,
	)
	if float(np.sum(weights)) <= 0.0:
		weights = np.asarray([0.5, 0.3, 0.2], dtype=float)
	weights = np.asarray(weights / float(np.sum(weights)), dtype=float)
	score = float(
		(weights[0] * float(lagged_cosine))
		+ (weights[1] * float(weighted_jaccard))
		+ (weights[2] * float(occupied_jaccard))
	)
	return float(max(0.0, min(1.0, score)))


def compute_pairwise_template_similarity(
	*,
	features_a: TemplateSimilarityFeatures,
	features_b: TemplateSimilarityFeatures,
	method: str,
	support: str = "union",
	max_lag_samples: int = 0,
	hybrid_waveform_weight: float = 0.5,
	hybrid_amplitude_weight: float = 0.3,
	hybrid_occupancy_weight: float = 0.2,
) -> PairwiseTemplateSimilarity:
	resolved_method = normalize_template_similarity_method(method)
	support_mode = normalize_template_similarity_support(support, default="union")
	vector_a, vector_b = _aligned_amplitude_vectors(features_a, features_b)
	ptp_cosine = _cosine_similarity(vector_a, vector_b)
	weighted_jaccard = _weighted_jaccard(vector_a, vector_b)
	occupied_jaccard, shared_count, union_count = _occupied_channel_jaccard(
		vector_a,
		vector_b,
		threshold_a=features_a.active_amplitude_threshold,
		threshold_b=features_b.active_amplitude_threshold,
	)
	aligned_template_a, aligned_template_b = _aligned_waveform_templates(
		features_a,
		features_b,
		support=support_mode,
	)
	lagged_cosine, lagged_cosine_lag = _lagged_template_similarity(
		aligned_template_a,
		aligned_template_b,
		metric="cosine",
		max_lag_samples=max_lag_samples,
	)
	lagged_l1, lagged_l1_lag = _lagged_template_similarity(
		aligned_template_a,
		aligned_template_b,
		metric="l1",
		max_lag_samples=max_lag_samples,
	)
	lagged_l2, lagged_l2_lag = _lagged_template_similarity(
		aligned_template_a,
		aligned_template_b,
		metric="l2",
		max_lag_samples=max_lag_samples,
	)
	slay_mean_similarity, slay_mean_similarity_lag = _slay_mean_similarity(
		aligned_template_a,
		aligned_template_b,
		max_lag_samples=max_lag_samples,
	)
	hybrid_template_similarity = _hybrid_template_similarity(
		lagged_cosine=lagged_cosine,
		weighted_jaccard=weighted_jaccard,
		occupied_jaccard=occupied_jaccard,
		waveform_weight=hybrid_waveform_weight,
		amplitude_weight=hybrid_amplitude_weight,
		occupancy_weight=hybrid_occupancy_weight,
	)
	metrics = {
		"ptp_cosine": float(ptp_cosine),
		"amplitude_weighted_jaccard": float(weighted_jaccard),
		"occupied_channel_jaccard": float(occupied_jaccard),
		"lagged_cosine": float(lagged_cosine),
		"lagged_cosine_lag_samples": float(lagged_cosine_lag),
		"lagged_l1": float(lagged_l1),
		"lagged_l1_lag_samples": float(lagged_l1_lag),
		"lagged_l2": float(lagged_l2),
		"lagged_l2_lag_samples": float(lagged_l2_lag),
		"slay_mean_similarity": float(slay_mean_similarity),
		"slay_mean_similarity_lag_samples": float(slay_mean_similarity_lag),
		"hybrid_template_similarity": float(hybrid_template_similarity),
	}
	return PairwiseTemplateSimilarity(
		method=resolved_method,
		score=float(metrics.get(resolved_method, ptp_cosine)),
		metrics=metrics,
		shared_channel_count=int(shared_count),
		union_channel_count=int(union_count),
	)