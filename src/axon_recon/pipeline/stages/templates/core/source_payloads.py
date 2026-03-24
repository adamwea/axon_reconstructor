from __future__ import annotations

from typing import Any

import numpy as np


def normalize_template_to_channels_by_time(template: Any, n_channels_hint: int) -> np.ndarray:
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


def normalize_source_payload(
	*,
	template: Any,
	locations_xy: np.ndarray,
	electrode_ids: list[Any] | None,
	channel_ids: list[Any] | None,
	sparse_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[Any] | None, list[Any] | None] | None:
	locs = np.asarray(locations_xy, dtype=float)
	if locs.ndim != 2 or int(locs.shape[1]) < 2:
		return None
	locs = locs[:, :2]

	t = np.asarray(template, dtype=float)
	if t.ndim != 2:
		return None

	# Sparse template handling: subset locations by sparsity when dimensions match.
	if sparse_indices is not None and int(np.asarray(sparse_indices).size) > 0:
		inds = np.asarray(sparse_indices, dtype=int)
		if int(t.shape[1]) == int(inds.size) or int(t.shape[0]) == int(inds.size):
			locs = locs[inds, :]
			if channel_ids is not None:
				try:
					channel_ids = list(np.asarray(channel_ids, dtype=object)[inds])
				except Exception:
					pass
			if electrode_ids is not None:
				try:
					electrode_ids = list(np.asarray(electrode_ids, dtype=object)[inds])
				except Exception:
					pass

	t_ch_by_t = normalize_template_to_channels_by_time(t, int(locs.shape[0]))
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

	return t_ch_by_t, locs, electrode_ids, channel_ids
