from __future__ import annotations

from typing import Any

import numpy as np


def _find_local_minima_indices(waveform_t: np.ndarray) -> np.ndarray:
	w = np.asarray(waveform_t, dtype=float).reshape(-1)
	if int(w.shape[0]) < 3:
		return np.asarray([], dtype=int)
	left = w[1:-1] < w[:-2]
	right = w[1:-1] <= w[2:]
	idx = np.nonzero(left & right)[0] + 1
	return idx.astype(int, copy=False)


def _select_peaks_with_min_separation(
	indices: np.ndarray,
	depths: np.ndarray,
	*,
	min_separation_samples: int,
) -> np.ndarray:
	if int(indices.shape[0]) <= 1:
		return indices.astype(int, copy=False)
	order = np.argsort(-np.asarray(depths, dtype=float))
	selected: list[int] = []
	min_sep = max(1, int(min_separation_samples))
	for order_idx in order:
		idx = int(indices[int(order_idx)])
		if all(abs(idx - prev) >= min_sep for prev in selected):
			selected.append(idx)
	if not selected:
		return np.asarray([], dtype=int)
	selected.sort()
	return np.asarray(selected, dtype=int)


def detect_multiple_negative_peaks(
	*,
	template_c_by_t: np.ndarray,
	channel_labels: list[Any] | None,
	prominence_fraction: float,
	min_separation_samples: int,
	max_peaks_per_channel: int = 2,
) -> dict[str, Any]:
	template = np.asarray(template_c_by_t, dtype=float)
	if template.ndim != 2:
		raise ValueError(f"Expected 2D template array (channels x time), got {template.shape}")

	prom_frac = max(0.0, float(prominence_fraction))
	min_sep = max(1, int(min_separation_samples))
	max_keep = max(2, int(max_peaks_per_channel))
	violations: list[dict[str, Any]] = []

	for ch_idx in range(int(template.shape[0])):
		wf = np.asarray(template[ch_idx, :], dtype=float).reshape(-1)
		if int(wf.shape[0]) < 3:
			continue
		minima_idx = _find_local_minima_indices(wf)
		if int(minima_idx.shape[0]) == 0:
			continue
		minima_values = wf[minima_idx]
		depths = -minima_values
		max_depth = float(np.max(depths)) if int(depths.shape[0]) > 0 else 0.0
		if max_depth <= 0.0:
			continue
		depth_threshold = float(max_depth * prom_frac)
		keep_mask = depths >= depth_threshold
		candidate_idx = minima_idx[keep_mask]
		candidate_depths = depths[keep_mask]
		if int(candidate_idx.shape[0]) <= 1:
			continue
		selected_idx = _select_peaks_with_min_separation(
			candidate_idx,
			candidate_depths,
			min_separation_samples=min_sep,
		)
		if int(selected_idx.shape[0]) <= 1:
			continue

		# Keep only the strongest peaks per channel (depth-ranked), then restore temporal order for plotting.
		if int(selected_idx.shape[0]) > max_keep:
			selected_depths = -wf[selected_idx]
			topk_order = np.argsort(-np.asarray(selected_depths, dtype=float))[:max_keep]
			selected_idx = np.sort(selected_idx[topk_order]).astype(int, copy=False)

		label = None
		if channel_labels is not None and int(ch_idx) < int(len(channel_labels)):
			label = channel_labels[ch_idx]
		violations.append(
			{
				"channel_index": int(ch_idx),
				"channel_label": (None if label is None else str(label)),
				"peak_count": int(selected_idx.shape[0]),
				"peak_indices": [int(v) for v in selected_idx.tolist()],
				"peak_values": [float(wf[int(v)]) for v in selected_idx.tolist()],
				"depth_threshold_abs": float(depth_threshold),
			}
		)

	return {
		"detected": bool(len(violations) > 0),
		"violation_count": int(len(violations)),
		"channel_count": int(template.shape[0]),
		"prominence_fraction": float(prom_frac),
		"min_separation_samples": int(min_sep),
			"max_peaks_per_channel": int(max_keep),
		"violations": violations,
	}
