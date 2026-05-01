from __future__ import annotations

import numpy as np

from axon_recon.pipeline.stages.reconstruct.templates.core.merge import (
	materialize_unit_templates_by_unit,
	merge_sources_per_channel,
	normalize_merge_method,
	normalize_overlap_priorities,
)


def test_normalize_merge_method_accepts_aliases() -> None:
	assert normalize_merge_method("mean") == "mean_all_waveforms"
	assert normalize_merge_method("weighted_average") == "weighted_by_channel_waveform_count"
	assert normalize_merge_method("weighted") == "weighted_by_channel_waveform_count"
	assert normalize_merge_method("unknown") == "mean_all_waveforms"


def test_normalize_overlap_priorities_filters_invalid_values() -> None:
	assert normalize_overlap_priorities(("electrode_id", "channel_id", "location")) == (
		"electrode_id",
		"channel_id",
		"location",
	)
	assert normalize_overlap_priorities(("bad", "location", "location")) == ("location",)
	assert normalize_overlap_priorities(("bad", "", "none")) == (
		"electrode_id",
		"channel_id",
		"location",
	)


def test_merge_sources_per_channel_means_overlapping_waveforms() -> None:
	# Two sources overlap by electrode id at the first channel and should average to 2.0.
	source1 = (
		np.asarray([[1.0, 1.0], [3.0, 3.0]], dtype=float),
		np.asarray([[0.0, 0.0], [20.0, 0.0]], dtype=float),
		[10, 11],
		[100, 101],
		5,
	)
	source2 = (
		np.asarray([[3.0, 3.0], [7.0, 7.0]], dtype=float),
		np.asarray([[0.0, 0.0], [30.0, 0.0]], dtype=float),
		[10, 12],
		[100, 102],
		20,
	)

	merged_template, merged_locs, merged_eids = merge_sources_per_channel(
		[source1, source2],
		enable_merge=True,
		merge_method="mean_all_waveforms",
		centering_method="none",
		max_waveforms_per_source_channel=None,
		overlap_match_priority=("electrode_id", "channel_id", "location"),
		location_tolerance_um=1.0,
	)

	# Keys are sorted alphabetically; first row is electrode 10 (overlap) and should be averaged.
	assert merged_template.shape == (3, 2)
	np.testing.assert_allclose(merged_template[0, :], np.asarray([2.0, 2.0], dtype=float))
	assert merged_locs.shape == (3, 2)
	assert merged_eids is not None
	assert merged_eids[0] == "10"


def test_merge_sources_per_channel_weighted_with_cap() -> None:
	source1 = (
		np.asarray([[1.0, 1.0]], dtype=float),
		np.asarray([[0.0, 0.0]], dtype=float),
		[10],
		[100],
		2,
	)
	source2 = (
		np.asarray([[5.0, 5.0]], dtype=float),
		np.asarray([[0.0, 0.0]], dtype=float),
		[10],
		[100],
		10,
	)

	merged_template, _, _ = merge_sources_per_channel(
		[source1, source2],
		enable_merge=True,
		merge_method="weighted_average",
		centering_method="none",
		max_waveforms_per_source_channel=4,
		overlap_match_priority=("electrode_id",),
		location_tolerance_um=1.0,
	)

	# Weighted mean with cap: (1*2 + 5*4) / (2+4) = 22/6.
	np.testing.assert_allclose(merged_template[0, :], np.asarray([22.0 / 6.0, 22.0 / 6.0], dtype=float))


def test_materialize_unit_templates_by_unit_orchestrates_payload_collection() -> None:
	analyzers = [("concat", object()), ("segment_0001", object())]

	def _builder(*, analyzer, unit_id):
		_ = analyzer
		if int(unit_id) != 94:
			return None
		if analyzer is analyzers[0][1]:
			return (
				np.asarray([[1.0, 1.0]], dtype=float),
				np.asarray([[0.0, 0.0]], dtype=float),
				[10],
				[100],
				2,
			)
		return (
			np.asarray([[3.0, 3.0]], dtype=float),
			np.asarray([[0.0, 0.0]], dtype=float),
			[10],
			[100],
			4,
		)

	out = materialize_unit_templates_by_unit(
		analyzers=analyzers,
		unit_ids=[94, 95],
		payload_builder=_builder,
		enable_merge=True,
		merge_method="mean_all_waveforms",
		centering_method="none",
		max_waveforms_per_source_channel=None,
		overlap_match_priority=("electrode_id",),
		location_tolerance_um=1.0,
	)

	assert set(out.keys()) == {94}
	merged_template, _, full_template, _, merged_eids = out[94]
	np.testing.assert_allclose(merged_template, np.asarray([[2.0, 2.0]], dtype=float))
	# Concat payload should be selected for full template.
	np.testing.assert_allclose(full_template, np.asarray([[1.0, 1.0]], dtype=float))
	assert merged_eids is not None and merged_eids[0] == "10"
