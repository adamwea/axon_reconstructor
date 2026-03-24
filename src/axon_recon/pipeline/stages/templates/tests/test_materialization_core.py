from __future__ import annotations

import numpy as np

from axon_recon.pipeline.stages.templates.core.materialization import (
	materialize_unit_templates_from_sources,
)
from axon_recon.pipeline.stages.templates.models.inputs import TimeUpsampleConfig


def _payload(
	waves: list[list[float]],
	locs: list[list[float]],
	*,
	electrode_ids: list[int] | None,
	channel_ids: list[int] | None,
	count: int,
):
	return (
		np.asarray(waves, dtype=float),
		np.asarray(locs, dtype=float),
		electrode_ids,
		channel_ids,
		count,
	)


def test_materialize_unit_templates_uses_concat_as_full_template() -> None:
	sources = [
		(
			"concat",
			_payload(
				[[1.0, 1.0], [2.0, 2.0]],
				[[0.0, 0.0], [10.0, 0.0]],
				electrode_ids=[10, 11],
				channel_ids=[100, 101],
				count=3,
			),
		),
		(
			"segment_0001",
			_payload(
				[[3.0, 3.0], [4.0, 4.0]],
				[[0.0, 0.0], [20.0, 0.0]],
				electrode_ids=[10, 12],
				channel_ids=[100, 102],
				count=5,
			),
		),
	]

	out = materialize_unit_templates_from_sources(
		source_payloads=sources,
		enable_merge=True,
		merge_method="mean_all_waveforms",
		centering_method="none",
		max_waveforms_per_source_channel=None,
		overlap_match_priority=("electrode_id", "channel_id", "location"),
		location_tolerance_um=1.0,
	)
	assert out is not None
	merged_template, merged_locs, full_template, full_locs = out

	assert merged_template.shape == (3, 2)
	np.testing.assert_allclose(full_template, sources[0][1][0])
	np.testing.assert_allclose(full_locs, sources[0][1][1])
	assert merged_locs.shape == (3, 2)


def test_materialize_unit_templates_returns_none_for_empty_sources() -> None:
	out = materialize_unit_templates_from_sources(
		source_payloads=[],
		enable_merge=True,
		merge_method="mean_all_waveforms",
		centering_method="none",
		max_waveforms_per_source_channel=None,
		overlap_match_priority=("electrode_id",),
		location_tolerance_um=1.0,
	)
	assert out is None


def test_materialize_unit_templates_applies_execution_upsampling_before_merge() -> None:
	sources = [
		(
			"concat",
			(
				np.asarray([[0.0, 1.0, 0.0, -1.0]], dtype=float),
				np.asarray([[0.0, 0.0]], dtype=float),
				[10],
				[100],
				3,
				10_000.0,
			),
		),
	]

	out = materialize_unit_templates_from_sources(
		source_payloads=sources,
		enable_merge=True,
		merge_method="mean_all_waveforms",
		centering_method="none",
		max_waveforms_per_source_channel=None,
		overlap_match_priority=("electrode_id",),
		location_tolerance_um=1.0,
		execution_upsampling=TimeUpsampleConfig(enabled=True, factor=10, method="linear"),
		raw_sampling_rate_hz=10_000.0,
	)
	assert out is not None
	merged_template, _, _, _ = out
	assert int(merged_template.shape[1]) > 4


def test_materialize_unit_templates_skips_upsampling_on_rate_mismatch() -> None:
	sources = [
		(
			"concat",
			(
				np.asarray([[0.0, 1.0, 0.0, -1.0]], dtype=float),
				np.asarray([[0.0, 0.0]], dtype=float),
				[10],
				[100],
				3,
				100_000.0,
			),
		),
	]

	out = materialize_unit_templates_from_sources(
		source_payloads=sources,
		enable_merge=True,
		merge_method="mean_all_waveforms",
		centering_method="none",
		max_waveforms_per_source_channel=None,
		overlap_match_priority=("electrode_id",),
		location_tolerance_um=1.0,
		execution_upsampling=TimeUpsampleConfig(enabled=True, factor=10, method="linear"),
		raw_sampling_rate_hz=10_000.0,
	)
	assert out is not None
	merged_template, _, _, _ = out
	assert int(merged_template.shape[1]) == 4
