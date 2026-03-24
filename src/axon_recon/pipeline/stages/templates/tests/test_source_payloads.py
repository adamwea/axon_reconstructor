from __future__ import annotations

import numpy as np

from axon_recon.pipeline.stages.templates.core.source_payloads import normalize_source_payload


def test_normalize_source_payload_applies_sparse_subset_and_orientation() -> None:
	template = np.asarray(
		[
			[1.0, 3.0],
			[2.0, 4.0],
			[0.0, 0.0],
			[0.0, 0.0],
		],
		dtype=float,
	)  # time x channels; sparse should select locations [2, 0], then transpose to channels x time.
	locs = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=float)
	electrode_ids = [100, 101, 102]
	channel_ids = [0, 1, 2]
	sparse = np.asarray([2, 0], dtype=int)

	normalized = normalize_source_payload(
		template=template,
		locations_xy=locs,
		electrode_ids=electrode_ids,
		channel_ids=channel_ids,
		sparse_indices=sparse,
	)
	assert normalized is not None
	t_ch_by_t, out_locs, out_eids, out_cids = normalized

	# Should become channels x time and keep sparse ordering [2, 0].
	assert t_ch_by_t.shape == (2, 4)
	np.testing.assert_allclose(out_locs, np.asarray([[20.0, 0.0], [0.0, 0.0]], dtype=float))
	assert out_eids == [102, 100]
	assert out_cids == [2, 0]


def test_normalize_source_payload_keeps_one_channel_when_all_flat() -> None:
	template = np.zeros((2, 4), dtype=float)  # channels x time
	locs = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=float)

	normalized = normalize_source_payload(
		template=template,
		locations_xy=locs,
		electrode_ids=[10, 11],
		channel_ids=[0, 1],
		sparse_indices=None,
	)
	assert normalized is not None
	t_ch_by_t, out_locs, out_eids, out_cids = normalized
	assert t_ch_by_t.shape == (1, 4)
	assert out_locs.shape == (1, 2)
	assert len(out_eids or []) == 1
	assert len(out_cids or []) == 1
