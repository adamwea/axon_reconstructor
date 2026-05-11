"""SortingAnalyzer-shaped read-only view of post-merge state.

Combines the pre-merge cache with SLAy's automerge/new2old.json to expose
exactly the analyzer-API surface the merge_SLAy snapshot capture and
report writers consume - without building a second SortingAnalyzer. The
view assembles each post-merge unit by either passing through the cached
pre-merge value (for unmerged units) or substituting the derived
weighted-average value (for merged units).

The post-merge unit ID set is computed as
``(cache_unit_ids \\ all_constituents_in_new2old) | new2old.keys()``,
which matches what SLAy writes into spike_clusters.npy (unmerged units
keep their original IDs; merged groups get fresh IDs encoded in
new2old.json).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .derive_post_merge import PostMergeDerived, derive_post_merge
from .pre_merge_cache import PreMergeCache


@dataclass
class _PostMergeExtension:
	"""Minimal stand-in for a SortingAnalyzer extension object."""

	_data: np.ndarray
	_templates_view: bool

	def get_data(self) -> np.ndarray:
		return self._data

	def get_templates(self) -> np.ndarray:
		if not self._templates_view:
			raise AttributeError("get_templates is only available on the templates extension")
		return self._data


class PostMergeView:
	"""SortingAnalyzer-like read-only view assembled from cache + derived.

	Implements the subset of the analyzer API exercised by the merge_SLAy
	report path: ``unit_ids``, ``channel_ids``, ``sampling_frequency``,
	``get_channel_locations()``, ``has_extension(name)``,
	``get_extension(name)``, and a ``sorting`` namespace with
	``count_num_spikes_per_unit()`` and ``get_unit_ids()``.
	"""

	def __init__(
		self,
		*,
		cache: PreMergeCache,
		derived: PostMergeDerived,
		post_merge_unit_ids: Sequence[str],
	) -> None:
		self._cache = cache
		self._derived = derived
		self._post_unit_ids: list[str] = [str(u) for u in post_merge_unit_ids]
		self._index_by_cache_uid = cache.index_by_unit_id()

		templates_list: list[np.ndarray] = []
		locations_list: list[np.ndarray] = []
		spike_counts: list[int] = []
		missing_unit_ids: list[str] = []
		for uid in self._post_unit_ids:
			template, location, count, ok = self._resolve_unit(uid)
			if not ok:
				missing_unit_ids.append(uid)
				continue
			templates_list.append(template)
			locations_list.append(location)
			spike_counts.append(int(count))
		# Drop missing rows so the assembled arrays stay aligned with
		# the surviving unit_ids list.
		for uid in missing_unit_ids:
			self._post_unit_ids.remove(uid)
		self._missing_unit_ids = missing_unit_ids
		self._templates_arr: np.ndarray = (
			np.stack(templates_list, axis=0).astype(np.float32, copy=False)
			if templates_list
			else np.zeros((0, cache.n_samples, cache.n_channels), dtype=np.float32)
		)
		self._unit_locations_arr: np.ndarray = (
			np.stack(locations_list, axis=0).astype(np.float32, copy=False)
			if locations_list
			else np.zeros((0, cache.unit_locations.shape[1]), dtype=np.float32)
		)
		self._spike_counts_by_uid: dict[str, int] = {
			uid: int(count) for uid, count in zip(self._post_unit_ids, spike_counts)
		}

	@property
	def unit_ids(self) -> np.ndarray:
		return np.asarray(self._post_unit_ids, dtype=object)

	@property
	def channel_ids(self) -> np.ndarray:
		return np.asarray(self._cache.channel_ids)

	@property
	def sampling_frequency(self) -> float:
		return float(self._cache.sampling_frequency)

	@property
	def missing_unit_ids(self) -> list[str]:
		"""Post-merge unit IDs that could not be resolved (no cache row and
		no derived row). Empty in healthy runs."""
		return list(self._missing_unit_ids)

	def get_channel_locations(self) -> np.ndarray:
		return np.asarray(self._cache.channel_locations)

	def has_extension(self, name: str) -> bool:
		return name in {"templates", "unit_locations"}

	def get_extension(self, name: str) -> _PostMergeExtension | None:
		if name == "templates":
			return _PostMergeExtension(self._templates_arr, _templates_view=True)
		if name == "unit_locations":
			return _PostMergeExtension(self._unit_locations_arr, _templates_view=False)
		return None

	def get_loaded_extension_names(self) -> list[str]:
		return ["templates", "unit_locations"]

	@property
	def sorting(self) -> SimpleNamespace:
		# Return a fresh namespace each call so callers cannot mutate
		# the view's internal state through the sorting handle.
		uids = list(self._post_unit_ids)
		counts = dict(self._spike_counts_by_uid)
		return SimpleNamespace(
			get_unit_ids=lambda: list(uids),
			count_num_spikes_per_unit=lambda: dict(counts),
			unit_ids=np.asarray(uids, dtype=object),
		)

	def _resolve_unit(
		self,
		uid: str,
	) -> tuple[np.ndarray, np.ndarray, int, bool]:
		if uid in self._derived.templates_by_id:
			return (
				self._derived.templates_by_id[uid],
				self._derived.unit_locations_by_id[uid],
				int(self._derived.spike_counts_by_id.get(uid, 0)),
				True,
			)
		row = self._index_by_cache_uid.get(uid)
		if row is None:
			zero_template = np.zeros(
				(self._cache.n_samples, self._cache.n_channels), dtype=np.float32
			)
			zero_loc = np.zeros(self._cache.unit_locations.shape[1], dtype=np.float32)
			return zero_template, zero_loc, 0, False
		return (
			self._cache.templates[row],
			self._cache.unit_locations[row],
			int(self._cache.spike_counts[row]),
			True,
		)


def post_merge_unit_ids_from_new2old(
	*,
	cache: PreMergeCache,
	new2old: Mapping[str, Sequence[object]],
) -> list[str]:
	"""Reconstruct the post-merge unit ID set without reading
	spike_clusters.npy. SLAy keeps unmerged units at their original IDs
	and assigns the merged groups fresh IDs encoded as keys in
	new2old.json; the post-merge set is therefore the union of the new
	keys and the pre-merge IDs not subsumed by any merge group.
	"""
	subsumed: set[str] = set()
	new_ids: list[str] = []
	for new_id_raw, old_ids_raw in new2old.items():
		new_ids.append(str(new_id_raw))
		for old_id_raw in old_ids_raw:
			subsumed.add(str(old_id_raw))
	pre_ids = [str(uid) for uid in cache.unit_ids.tolist()]
	survivors = [uid for uid in pre_ids if uid not in subsumed]
	out: list[str] = []
	seen: set[str] = set()
	for uid in survivors + new_ids:
		if uid in seen:
			continue
		seen.add(uid)
		out.append(uid)
	return out


def build_post_merge_view(
	*,
	cache: PreMergeCache,
	new2old: Mapping[str, Sequence[object]],
) -> PostMergeView:
	"""Convenience constructor: derive + reconstruct unit-id set + view."""
	derived = derive_post_merge(cache=cache, new2old=new2old)
	post_unit_ids = post_merge_unit_ids_from_new2old(cache=cache, new2old=new2old)
	return PostMergeView(cache=cache, derived=derived, post_merge_unit_ids=post_unit_ids)
