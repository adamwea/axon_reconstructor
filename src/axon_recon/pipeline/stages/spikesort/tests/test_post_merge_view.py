from __future__ import annotations

import numpy as np
import pytest

from axon_recon.pipeline.stages.spikesort.core.post_merge_view import (
	PostMergeView,
	build_post_merge_view,
	post_merge_unit_ids_from_new2old,
)
from axon_recon.pipeline.stages.spikesort.core.pre_merge_cache import PreMergeCache


def _cache(
	*,
	unit_ids: np.ndarray,
	templates: np.ndarray,
	unit_locations: np.ndarray,
	spike_counts: np.ndarray,
) -> PreMergeCache:
	n_channels = templates.shape[2]
	return PreMergeCache(
		unit_ids=unit_ids,
		templates=templates.astype(np.float32),
		unit_locations=unit_locations.astype(np.float32),
		spike_counts=spike_counts.astype(np.int64),
		channel_ids=np.arange(n_channels, dtype=np.int64),
		channel_locations=np.zeros((n_channels, 2), dtype=np.float32),
		ms_before=1.0,
		ms_after=2.0,
		sampling_frequency=10_000.0,
	)


def _simple_cache_3units() -> PreMergeCache:
	templates = np.stack(
		[
			np.full((1, 2), 1.0),
			np.full((1, 2), 3.0),
			np.full((1, 2), 7.0),
		]
	).astype(np.float32)
	return _cache(
		unit_ids=np.array([0, 1, 2], dtype=np.int64),
		templates=templates,
		unit_locations=np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]),
		spike_counts=np.array([5, 15, 80]),
	)


def test_post_merge_unit_ids_pure_passthrough() -> None:
	cache = _simple_cache_3units()
	ids = post_merge_unit_ids_from_new2old(cache=cache, new2old={})
	assert ids == ["0", "1", "2"]


def test_post_merge_unit_ids_subsume_merged() -> None:
	cache = _simple_cache_3units()
	# Merge units 0 and 1 into new id 100; unit 2 unchanged.
	ids = post_merge_unit_ids_from_new2old(cache=cache, new2old={"100": [0, 1]})
	assert ids == ["2", "100"]


def test_post_merge_unit_ids_multiple_groups() -> None:
	cache = _simple_cache_3units()
	ids = post_merge_unit_ids_from_new2old(
		cache=cache, new2old={"100": [0, 1], "101": [2]}
	)
	# u0 and u1 are subsumed into 100; u2 is subsumed into 101 (singleton).
	# Survivors before new keys: none.
	assert ids == ["100", "101"]


def test_view_unit_ids_and_arrays_align_with_passthrough_only() -> None:
	cache = _simple_cache_3units()
	view = build_post_merge_view(cache=cache, new2old={})
	assert list(view.unit_ids) == ["0", "1", "2"]
	templates = view.get_extension("templates").get_templates()
	assert templates.shape == (3, 1, 2)
	# Unmerged units should pass through unchanged.
	np.testing.assert_array_equal(templates[0], cache.templates[0])
	np.testing.assert_array_equal(templates[1], cache.templates[1])
	np.testing.assert_array_equal(templates[2], cache.templates[2])
	locations = view.get_extension("unit_locations").get_data()
	assert locations.shape == (3, 2)
	np.testing.assert_array_equal(locations[0], cache.unit_locations[0])


def test_view_merged_units_use_weighted_average() -> None:
	cache = _simple_cache_3units()
	# n0=5, n1=15 -> weights 0.25, 0.75; merged template = 0.25*1 + 0.75*3 = 2.5
	view = build_post_merge_view(cache=cache, new2old={"100": [0, 1]})
	assert list(view.unit_ids) == ["2", "100"]
	templates = view.get_extension("templates").get_templates()
	np.testing.assert_array_equal(templates[0], cache.templates[2])  # passthrough unit 2
	np.testing.assert_allclose(
		templates[1], np.full((1, 2), 2.5, dtype=np.float32), atol=1e-6
	)


def test_view_sorting_count_returns_combined_spike_counts() -> None:
	cache = _simple_cache_3units()
	view = build_post_merge_view(cache=cache, new2old={"100": [0, 1]})
	sorting = view.sorting
	counts = sorting.count_num_spikes_per_unit()
	assert counts == {"2": 80, "100": 20}
	assert sorting.get_unit_ids() == ["2", "100"]
	# Mutating the returned dict must not corrupt the view.
	counts["2"] = -1
	assert view.sorting.count_num_spikes_per_unit() == {"2": 80, "100": 20}


def test_view_channel_metadata_passes_through() -> None:
	cache = _simple_cache_3units()
	view = build_post_merge_view(cache=cache, new2old={})
	np.testing.assert_array_equal(view.channel_ids, cache.channel_ids)
	np.testing.assert_array_equal(view.get_channel_locations(), cache.channel_locations)
	assert view.sampling_frequency == 10_000.0


def test_view_has_extension_and_get_extension_unknown_returns_none() -> None:
	cache = _simple_cache_3units()
	view = build_post_merge_view(cache=cache, new2old={})
	assert view.has_extension("templates") is True
	assert view.has_extension("unit_locations") is True
	assert view.has_extension("waveforms") is False
	assert view.get_extension("templates") is not None
	assert view.get_extension("unit_locations") is not None
	assert view.get_extension("waveforms") is None
	with pytest.raises(AttributeError):
		view.get_extension("unit_locations").get_templates()


def test_view_loaded_extension_names() -> None:
	cache = _simple_cache_3units()
	view = build_post_merge_view(cache=cache, new2old={})
	assert view.get_loaded_extension_names() == ["templates", "unit_locations"]


def test_view_str_int_unit_id_match_across_new2old_and_cache() -> None:
	cache = _simple_cache_3units()
	# new2old keys may arrive as either string ids or numpy ints.
	view = build_post_merge_view(cache=cache, new2old={"100": ["0", "1"]})
	assert list(view.unit_ids) == ["2", "100"]
	templates = view.get_extension("templates").get_templates()
	np.testing.assert_allclose(
		templates[1], np.full((1, 2), 2.5, dtype=np.float32), atol=1e-6
	)


def test_view_with_explicit_unit_id_set_drops_missing() -> None:
	cache = _simple_cache_3units()
	from axon_recon.pipeline.stages.spikesort.core.derive_post_merge import derive_post_merge
	derived = derive_post_merge(cache=cache, new2old={"100": [0, 1]})
	# Caller asks for an extra unit that isn't in cache or derived.
	view = PostMergeView(
		cache=cache,
		derived=derived,
		post_merge_unit_ids=["2", "100", "missing"],
	)
	assert list(view.unit_ids) == ["2", "100"]
	assert view.missing_unit_ids == ["missing"]


def test_view_empty_new2old_and_empty_cache_produces_empty_arrays() -> None:
	cache = _cache(
		unit_ids=np.array([], dtype=np.int64),
		templates=np.zeros((0, 4, 3), dtype=np.float32),
		unit_locations=np.zeros((0, 2), dtype=np.float32),
		spike_counts=np.array([], dtype=np.int64),
	)
	view = build_post_merge_view(cache=cache, new2old={})
	assert list(view.unit_ids) == []
	assert view.get_extension("templates").get_templates().shape == (0, 4, 3)
	assert view.get_extension("unit_locations").get_data().shape == (0, 2)
