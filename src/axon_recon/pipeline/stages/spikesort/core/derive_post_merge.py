"""Derive post-merge templates and unit_locations from the pre-merge cache.

For new unit ``U`` with constituents ``[u_a, u_b, ...]``, the post-merge
template is the spike-count-weighted average of the pre-merge templates of
its constituents. This is mathematically the same template you would get if
SpikeInterface's ``random_spikes`` selection were the union of the
constituents' selections — i.e., the most "comparable" reconstruction
possible without re-traversing the recording.

Singleton groups (a single old constituent) fall out as exact pass-throughs.
Unmerged units (not present in ``new2old``) are the caller's responsibility
to fill from the cache directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .pre_merge_cache import PreMergeCache


@dataclass
class PostMergeDerived:
	templates_by_id: dict[str, np.ndarray]
	unit_locations_by_id: dict[str, np.ndarray]
	spike_counts_by_id: dict[str, int]
	merge_groups: dict[str, list[str]]
	skipped: dict[str, str]


def derive_post_merge(
	*,
	cache: PreMergeCache,
	new2old: Mapping[str, Sequence[object]],
) -> PostMergeDerived:
	index_by_uid = cache.index_by_unit_id()

	templates_by_id: dict[str, np.ndarray] = {}
	unit_locations_by_id: dict[str, np.ndarray] = {}
	spike_counts_by_id: dict[str, int] = {}
	merge_groups: dict[str, list[str]] = {}
	skipped: dict[str, str] = {}

	spike_counts = cache.spike_counts.astype(np.int64, copy=False)

	for new_id_raw, old_ids_raw in new2old.items():
		new_id = str(new_id_raw)
		constituent_ids: list[str] = []
		constituent_rows: list[int] = []
		for old_id_raw in old_ids_raw:
			old_id = str(old_id_raw)
			row = index_by_uid.get(old_id)
			if row is None:
				continue
			constituent_ids.append(old_id)
			constituent_rows.append(int(row))
		if not constituent_rows:
			skipped[new_id] = "no_resolvable_constituents"
			continue
		rows = np.asarray(constituent_rows, dtype=np.int64)
		weights = spike_counts[rows].astype(np.float64)
		total_weight = float(weights.sum())
		if total_weight <= 0.0:
			skipped[new_id] = "zero_total_spike_count"
			continue
		w = (weights / total_weight).astype(np.float32)
		template_new = np.einsum(
			"i,ijk->jk", w, cache.templates[rows].astype(np.float32, copy=False)
		).astype(np.float32)
		loc_new = (w @ cache.unit_locations[rows].astype(np.float32, copy=False)).astype(np.float32)
		templates_by_id[new_id] = template_new
		unit_locations_by_id[new_id] = loc_new
		spike_counts_by_id[new_id] = int(weights.sum())
		merge_groups[new_id] = constituent_ids

	return PostMergeDerived(
		templates_by_id=templates_by_id,
		unit_locations_by_id=unit_locations_by_id,
		spike_counts_by_id=spike_counts_by_id,
		merge_groups=merge_groups,
		skipped=skipped,
	)
