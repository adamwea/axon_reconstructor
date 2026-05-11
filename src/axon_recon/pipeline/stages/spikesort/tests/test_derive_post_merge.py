from __future__ import annotations

import numpy as np

from axon_recon.pipeline.stages.spikesort.core.derive_post_merge import derive_post_merge
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


def test_singleton_group_is_exact_passthrough() -> None:
	templates = np.array(
		[
			[[1.0, 2.0], [3.0, 4.0]],
			[[5.0, 6.0], [7.0, 8.0]],
		]
	)
	unit_locations = np.array([[1.0, 2.0], [10.0, 20.0]])
	spike_counts = np.array([100, 200])
	cache = _cache(
		unit_ids=np.array([0, 1]),
		templates=templates,
		unit_locations=unit_locations,
		spike_counts=spike_counts,
	)
	result = derive_post_merge(cache=cache, new2old={"5": [0]})
	np.testing.assert_array_equal(result.templates_by_id["5"], templates[0])
	np.testing.assert_array_equal(result.unit_locations_by_id["5"], unit_locations[0])
	assert result.spike_counts_by_id["5"] == 100
	assert result.merge_groups["5"] == ["0"]
	assert "5" not in result.skipped


def test_two_into_one_weighted_average() -> None:
	# Constituents: u0 (n=2), u1 (n=1). Expected weight = (2/3, 1/3).
	templates = np.array(
		[
			[[3.0, 6.0]],
			[[9.0, 0.0]],
		]
	)  # shape (2, 1, 2)
	unit_locations = np.array([[0.0, 0.0], [3.0, 6.0]])
	spike_counts = np.array([2, 1])
	cache = _cache(
		unit_ids=np.array([0, 1]),
		templates=templates,
		unit_locations=unit_locations,
		spike_counts=spike_counts,
	)
	result = derive_post_merge(cache=cache, new2old={"7": [0, 1]})
	expected_template = (2 * templates[0] + 1 * templates[1]) / 3
	expected_location = (2 * unit_locations[0] + 1 * unit_locations[1]) / 3
	np.testing.assert_allclose(result.templates_by_id["7"], expected_template, rtol=0, atol=1e-6)
	np.testing.assert_allclose(result.unit_locations_by_id["7"], expected_location, rtol=0, atol=1e-6)
	assert result.spike_counts_by_id["7"] == 3
	assert result.merge_groups["7"] == ["0", "1"]


def test_three_into_one_weighted_average() -> None:
	templates = np.stack(
		[
			np.full((1, 2), 1.0),
			np.full((1, 2), 2.0),
			np.full((1, 2), 8.0),
		]
	).astype(np.float32)  # (3, 1, 2)
	unit_locations = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 100.0]])
	spike_counts = np.array([10, 20, 70])  # weights 0.1, 0.2, 0.7
	cache = _cache(
		unit_ids=np.array([10, 20, 30]),
		templates=templates,
		unit_locations=unit_locations,
		spike_counts=spike_counts,
	)
	result = derive_post_merge(cache=cache, new2old={"99": [10, 20, 30]})
	# 0.1*1 + 0.2*2 + 0.7*8 = 0.1 + 0.4 + 5.6 = 6.1
	np.testing.assert_allclose(
		result.templates_by_id["99"], np.full((1, 2), 6.1, dtype=np.float32), rtol=0, atol=1e-6
	)
	expected_loc = 0.1 * unit_locations[0] + 0.2 * unit_locations[1] + 0.7 * unit_locations[2]
	np.testing.assert_allclose(result.unit_locations_by_id["99"], expected_loc, rtol=0, atol=1e-5)
	assert result.spike_counts_by_id["99"] == 100


def test_unresolvable_constituents_go_to_skipped() -> None:
	cache = _cache(
		unit_ids=np.array([0, 1]),
		templates=np.zeros((2, 1, 2), dtype=np.float32),
		unit_locations=np.zeros((2, 2), dtype=np.float32),
		spike_counts=np.array([1, 1]),
	)
	result = derive_post_merge(cache=cache, new2old={"missing": [999, 1000]})
	assert "missing" not in result.templates_by_id
	assert result.skipped["missing"] == "no_resolvable_constituents"


def test_partial_resolution_uses_what_it_finds() -> None:
	templates = np.array([[[2.0, 4.0]], [[0.0, 0.0]]])
	cache = _cache(
		unit_ids=np.array([0, 1]),
		templates=templates,
		unit_locations=np.array([[5.0, 5.0], [0.0, 0.0]]),
		spike_counts=np.array([4, 4]),
	)
	# Constituent 999 does not exist; only 0 resolves -> treated as singleton.
	result = derive_post_merge(cache=cache, new2old={"new": [0, 999]})
	np.testing.assert_array_equal(result.templates_by_id["new"], templates[0])
	assert result.merge_groups["new"] == ["0"]
	assert result.spike_counts_by_id["new"] == 4


def test_zero_total_spike_count_goes_to_skipped() -> None:
	cache = _cache(
		unit_ids=np.array([0, 1]),
		templates=np.zeros((2, 1, 2), dtype=np.float32),
		unit_locations=np.zeros((2, 2), dtype=np.float32),
		spike_counts=np.array([0, 0]),
	)
	result = derive_post_merge(cache=cache, new2old={"empty": [0, 1]})
	assert "empty" not in result.templates_by_id
	assert result.skipped["empty"] == "zero_total_spike_count"


def test_string_unit_ids_in_new2old_match_int_cache_ids() -> None:
	templates = np.array([[[1.0, 1.0]], [[3.0, 3.0]]])
	cache = _cache(
		unit_ids=np.array([0, 1], dtype=np.int64),
		templates=templates,
		unit_locations=np.array([[0.0, 0.0], [10.0, 10.0]]),
		spike_counts=np.array([1, 1]),
	)
	# Same weights -> simple average.
	result = derive_post_merge(cache=cache, new2old={"2": ["0", "1"]})
	expected_template = (templates[0] + templates[1]) / 2
	np.testing.assert_allclose(result.templates_by_id["2"], expected_template, rtol=0, atol=1e-6)
	assert result.merge_groups["2"] == ["0", "1"]


def test_unmerged_units_not_in_new2old_are_not_in_result() -> None:
	cache = _cache(
		unit_ids=np.array([0, 1, 2]),
		templates=np.zeros((3, 1, 2), dtype=np.float32),
		unit_locations=np.zeros((3, 2), dtype=np.float32),
		spike_counts=np.array([1, 1, 1]),
	)
	# Only u1 and u2 merge; u0 is untouched -> caller fills from cache directly.
	result = derive_post_merge(cache=cache, new2old={"m": [1, 2]})
	assert set(result.templates_by_id.keys()) == {"m"}
	assert set(result.merge_groups.keys()) == {"m"}


def test_multiple_groups_in_one_call() -> None:
	templates = np.stack(
		[
			np.full((1, 2), 1.0),
			np.full((1, 2), 3.0),
			np.full((1, 2), 5.0),
			np.full((1, 2), 7.0),
		]
	).astype(np.float32)
	cache = _cache(
		unit_ids=np.array([0, 1, 2, 3]),
		templates=templates,
		unit_locations=np.zeros((4, 2), dtype=np.float32),
		spike_counts=np.array([1, 1, 1, 1]),
	)
	result = derive_post_merge(
		cache=cache,
		new2old={"a": [0, 1], "b": [2, 3], "c": [0]},
	)
	# a = avg(1,3) = 2; b = avg(5,7) = 6; c = passthrough(0) = 1
	np.testing.assert_allclose(result.templates_by_id["a"], np.full((1, 2), 2.0), atol=1e-6)
	np.testing.assert_allclose(result.templates_by_id["b"], np.full((1, 2), 6.0), atol=1e-6)
	np.testing.assert_allclose(result.templates_by_id["c"], np.full((1, 2), 1.0), atol=1e-6)
