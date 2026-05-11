from __future__ import annotations

import pandas as pd

from ..filters import (
	apply_filter_spec,
	filter_bombcell_allowlist,
	filter_min_numeric,
	filter_multiselect,
	filter_numeric_range,
	filter_recon_status_ok,
	mask_all,
)


def _df():
	return pd.DataFrame(
		{
			"recon_status": ["ok", "ok", "error", "ok"],
			"bombcell_label": ["good", "non_soma_good", "mua", None],
			"num_spikes": [100, 50, 10, 1000],
			"num_branches": [3, 1, 0, 5],
			"recon_quality_score": [None, 0.7, 0.3, None],
			"project": ["A", "A", "B", "B"],
			"DIV": [6, 18, 36, 12],
			"genotype": ["WT", "KO", "WT", "KO"],
		}
	)


def test_mask_all_aligns_with_index() -> None:
	df = _df()
	mask = mask_all(df)
	assert mask.all()
	assert list(mask.index) == list(df.index)


def test_filter_recon_status_ok_default_drops_error_rows() -> None:
	df = _df()
	mask = filter_recon_status_ok(df, on=True)
	assert list(mask) == [True, True, False, True]


def test_filter_recon_status_ok_pass_through_when_off() -> None:
	df = _df()
	mask = filter_recon_status_ok(df, on=False)
	assert mask.all()


def test_filter_bombcell_allowlist_with_none_passes_unlabeled() -> None:
	df = _df()
	mask = filter_bombcell_allowlist(df, ["good", None])
	assert list(mask) == [True, False, False, True]


def test_filter_bombcell_allowlist_empty_drops_everything() -> None:
	df = _df()
	mask = filter_bombcell_allowlist(df, [])
	assert not mask.any()


def test_filter_bombcell_allowlist_none_is_pass_through() -> None:
	df = _df()
	mask = filter_bombcell_allowlist(df, None)
	assert mask.all()


def test_filter_min_numeric_keeps_above_threshold_and_nan_rows() -> None:
	df = _df()
	mask = filter_min_numeric(df, "recon_quality_score", 0.5)
	# Row 0 (None) and Row 1 (0.7) pass; Row 2 (0.3) drops; Row 3 (None) passes.
	assert list(mask) == [True, True, False, True]


def test_filter_min_numeric_threshold_none_passes_all() -> None:
	df = _df()
	mask = filter_min_numeric(df, "num_spikes", None)
	assert mask.all()


def test_filter_min_numeric_missing_column_passes_all() -> None:
	df = _df()
	mask = filter_min_numeric(df, "doesnt_exist", 5)
	assert mask.all()


def test_filter_multiselect_keeps_matching_values() -> None:
	df = _df()
	mask = filter_multiselect(df, "project", ["A"])
	assert list(mask) == [True, True, False, False]


def test_filter_multiselect_empty_or_none_passes_all() -> None:
	df = _df()
	assert filter_multiselect(df, "project", None).all()
	assert filter_multiselect(df, "project", []).all()


def test_filter_numeric_range_inclusive_bounds() -> None:
	df = _df()
	mask = filter_numeric_range(df, "DIV", lo=10, hi=20)
	assert list(mask) == [False, True, False, True]


def test_apply_filter_spec_combines_filters() -> None:
	df = _df()
	filtered = apply_filter_spec(
		df,
		{
			"require_recon_ok": True,
			"bombcell_allowlist": ["good", "non_soma_good"],
			"min_num_spikes": 30,
			"div_lo": 5,
			"div_hi": 30,
		},
	)
	# Expected: index 0 (ok/good/100 spikes/DIV=6) PASS; index 1 (ok/non_soma_good/50/18) PASS;
	# index 2 dropped by recon_status; index 3 dropped by bombcell allowlist (None not allowed).
	assert len(filtered) == 2
	assert list(filtered["bombcell_label"]) == ["good", "non_soma_good"]


def test_apply_filter_spec_default_inputs_keep_all_ok_rows() -> None:
	df = _df()
	filtered = apply_filter_spec(df, {})
	# Default: require_recon_ok=True drops 1 error row; nothing else applied.
	assert len(filtered) == 3
	assert all(s == "ok" for s in filtered["recon_status"])
